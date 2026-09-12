"""Two-way, linearly implicit PBD attachments to articulated rigid links."""

import torch

import quadrants as qd

import genesis as gs
from genesis.utils import array_class
from genesis.utils.geom import qd_inv_transform_by_trans_quat, qd_transform_by_trans_quat
from genesis.utils.misc import qd_to_torch


class PBDRigidAttachment:
    """Eliminate particle velocities from a backward-Euler spring solve before the solver drift steps."""

    def __init__(self, rigid_solver, pbd_solver, attach_info):
        self.rigid_solver = rigid_solver
        self.pbd_solver = pbd_solver
        self.attach_info = attach_info
        self.particles_idx = torch.empty(0, dtype=gs.tc_int, device=gs.device)
        self.jacobian = torch.empty(0, dtype=gs.tc_float, device=gs.device)
        self.bias = torch.empty(0, dtype=gs.tc_float, device=gs.device)
        self.weight = torch.empty(0, dtype=gs.tc_float, device=gs.device)
        self.beta = torch.empty(0, dtype=gs.tc_float, device=gs.device)

    def rebuild(self):
        is_attached = qd_to_torch(self.attach_info.compliance, transpose=True) >= 0
        self.particles_idx = is_attached.any(dim=0).nonzero().flatten().to(gs.tc_int)
        shape = (self.pbd_solver._B, 3 * len(self.particles_idx))
        self.jacobian = torch.zeros((*shape, self.rigid_solver.n_dofs), dtype=gs.tc_float, device=gs.device)
        self.bias = torch.zeros(shape, dtype=gs.tc_float, device=gs.device)
        self.weight = torch.zeros_like(self.bias)
        self.beta = torch.zeros_like(self.bias)

    def solve(self):
        if not len(self.particles_idx):
            return
        rigid = self.rigid_solver
        pbd = self.pbd_solver
        rigid.update_forward_pos()
        self.jacobian.zero_()
        kernel_attachment_system(
            self.particles_idx,
            pbd.substep_dt,
            self.jacobian,
            self.bias,
            self.weight,
            self.beta,
            pbd.particles,
            pbd.particles_ng,
            pbd.particles_info,
            self.attach_info,
            rigid.dyn_state,
            rigid.dyn_info,
            rigid.rigid_config,
        )
        if rigid.n_dofs:
            rigid.update_mass_mat()
            mass = rigid.get_mass_mat().reshape(rigid._B, rigid.n_dofs, rigid.n_dofs)
            velocity = rigid.get_dofs_velocity().reshape(rigid._B, rigid.n_dofs, 1)
            weighted_jacobian = self.weight.unsqueeze(-1) * self.jacobian
            # W = diag(m * h^2 / (m * compliance + h^2)), b = u + C/h.
            # (M + J.T W J) v_new = M v + J.T W b; particle impulse is -W(b - J v_new).
            system = mass + self.jacobian.transpose(-1, -2) @ weighted_jacobian
            rhs = mass @ velocity + weighted_jacobian.transpose(-1, -2) @ self.bias.unsqueeze(-1)
            velocity = torch.linalg.solve(system, rhs)
            correction = self.beta * (self.bias - (self.jacobian @ velocity).squeeze(-1))
            rigid.set_dofs_velocity(velocity.reshape(rigid._B, rigid.n_dofs))
        else:
            correction = self.beta * self.bias
        kernel_apply_attachment(self.particles_idx, correction, pbd.particles)


@qd.kernel
def kernel_set_attachment(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: qd.i32,
    compliance: float,
    particles: qd.template(),
    attach_info: qd.template(),
    links_state: array_class.LinksState,
):
    for i_b_, i_p_ in qd.ndrange(envs_idx.shape[0], particles_idx.shape[1]):
        i_b = envs_idx[i_b_]
        i_p = particles_idx[i_b_, i_p_]
        attach_info[i_p, i_b].link_idx = link_idx
        attach_info[i_p, i_b].local_pos = qd_inv_transform_by_trans_quat(
            particles[i_p, i_b].pos, links_state.pos[link_idx, i_b], links_state.quat[link_idx, i_b]
        )
        attach_info[i_p, i_b].compliance = compliance
        particles[i_p, i_b].free = True


@qd.kernel
def kernel_clear_physical_attachments(envs_idx: qd.types.ndarray(), attach_info: qd.template()):
    for i_p, i_b_ in qd.ndrange(attach_info.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        if attach_info[i_p, i_b].compliance >= 0:
            attach_info[i_p, i_b].link_idx = -1
            attach_info[i_p, i_b].compliance = -1


@qd.kernel
def kernel_attachment_system(
    particles_idx: qd.types.ndarray(),
    dt: float,
    jacobian: qd.types.ndarray(),
    bias: qd.types.ndarray(),
    weight: qd.types.ndarray(),
    beta: qd.types.ndarray(),
    particles: qd.template(),
    particles_ng: qd.template(),
    particles_info: qd.template(),
    attach_info: qd.template(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    for i_b, i_p_ in qd.ndrange(bias.shape[0], particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        for axis in qd.static(range(3)):
            bias[i_b, 3 * i_p_ + axis] = 0
            weight[i_b, 3 * i_p_ + axis] = 0
            beta[i_b, 3 * i_p_ + axis] = 0
        compliance = attach_info[i_p, i_b].compliance
        if compliance >= 0 and particles[i_p, i_b].free and particles_ng[i_p, i_b].active:
            i_link = attach_info[i_p, i_b].link_idx
            anchor = qd_transform_by_trans_quat(
                attach_info[i_p, i_b].local_pos, dyn_state.links.pos[i_link, i_b], dyn_state.links.quat[i_link, i_b]
            )
            mass = particles_info[i_p].mass
            blend = dt * dt / (mass * compliance + dt * dt)
            target = particles[i_p, i_b].vel + (particles[i_p, i_b].pos - anchor) / dt
            for axis in qd.static(range(3)):
                bias[i_b, 3 * i_p_ + axis] = target[axis]
                weight[i_b, 3 * i_p_ + axis] = mass * blend
                beta[i_b, 3 * i_p_ + axis] = blend
            offset = anchor - dyn_state.links.root_COM[i_link, i_b]
            while i_link >= 0:
                I_l = [i_link, i_b] if qd.static(rigid_config.batch_links_info) else i_link
                for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                    motion = dyn_state.dofs.cdof_vel[i_d, i_b] + dyn_state.dofs.cdof_ang[i_d, i_b].cross(offset)
                    for axis in qd.static(range(3)):
                        jacobian[i_b, 3 * i_p_ + axis, i_d] = motion[axis]
                i_link = dyn_info.links.parent_idx[I_l]


@qd.kernel
def kernel_apply_attachment(
    particles_idx: qd.types.ndarray(), correction: qd.types.ndarray(), particles: qd.template()
):
    for i_b, i_p_ in qd.ndrange(correction.shape[0], particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        for axis in qd.static(range(3)):
            particles[i_p, i_b].vel[axis] -= correction[i_b, 3 * i_p_ + axis]
