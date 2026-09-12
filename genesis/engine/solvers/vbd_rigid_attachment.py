"""Forward coupling of tetrahedral tissue to free or articulated rigid links."""

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.vbd_contact import func_contact_link_terms, func_refresh_link_vertices
from genesis.engine.solvers.vbd_rigid import func_attachment_blocks, func_ldlt6_solve
from genesis.engine.solvers.vbd_rigid import func_quaternion_difference, func_quaternion_update
from genesis.utils.array_class import DynState, RigidInfo
from genesis.utils.misc import tensor_to_array


class VBDRigidAttachment:
    def __init__(self, solver, entities):
        self.rigid = solver.sim.rigid_solver
        self.solver = solver
        if self.rigid.substep_dt != solver.substep_dt:
            gs.raise_exception("Rigid and VBD attachment timesteps must match.")
        if any(
            s.is_active
            for s in (
                solver.sim.mpm_solver,
                solver.sim.pbd_solver,
                solver.sim.fem_solver,
                solver.sim.sph_solver,
                solver.sim.sf_solver,
                solver.sim.tool_solver,
                solver.sim.kinematic_solver,
            )
        ):
            gs.raise_exception("VBD rigid attachments support scenes containing rigid and VBD solvers only.")
        self.is_articulated = self.rigid.claim_vbd_links()
        links = [link for entity in entities for link in entity._rigid_links]
        # the one link the free path integrates; fixed links of other entities may share the scene
        self.free_link_idx = -1 if self.is_articulated else links[0].idx
        if not self.is_articulated and any(link.idx != self.free_link_idx for link in links):
            gs.raise_exception("Free-link ownership attaches tissue to that one free link only.")
        links_idx = np.array([link.idx for link in links], dtype=gs.np_int)
        indices = np.concatenate([entity.v_start + entity._rigid_vertices_idx for entity in entities])
        positions = np.concatenate(
            [tensor_to_array(entity.init_positions)[entity._rigid_vertices_idx] for entity in entities]
        )
        self.n_attachments = len(indices)
        self.alpha = 0.95
        self.gamma = 0.99
        link_positions = tensor_to_array(self.rigid.get_links_pos()).reshape(solver._B, self.rigid.n_links, 3)[0]
        link_quaternions = tensor_to_array(self.rigid.get_links_quat()).reshape(solver._B, self.rigid.n_links, 4)[0]
        local = gu.inv_transform_by_quat(positions - link_positions[links_idx], link_quaternions[links_idx])
        info_type = qd.types.struct(vertex=gs.qd_int, link=gs.qd_int, local_pos=gs.qd_vec3)
        state_type = qd.types.struct(multiplier=gs.qd_vec3, stiffness=gs.qd_float)
        link_type = qd.types.struct(
            pos=gs.qd_vec3,
            quat=gs.qd_vec4,
            previous_pos=gs.qd_vec3,
            previous_quat=gs.qd_vec4,
            predicted_pos=gs.qd_vec3,
            predicted_quat=gs.qd_vec4,
            inertia=gs.qd_mat3,
            mass=gs.qd_float,
        )
        self.info = info_type.field(shape=self.n_attachments, layout=qd.Layout.SOA)
        self.state = state_type.field(shape=(self.n_attachments, solver._B), layout=qd.Layout.SOA)
        self.previous_error = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_attachments, solver._B))
        self.link_state = link_type.field(shape=solver._B, layout=qd.Layout.SOA)
        self.vertex_attachment = qd.field(dtype=gs.qd_int, shape=solver.n_vertices)
        lookup = np.full(solver.n_vertices, -1, dtype=gs.np_int)
        lookup[indices] = np.arange(self.n_attachments)
        self.vertex_attachment.from_numpy(lookup)
        self.info.vertex.from_numpy(indices.astype(gs.np_int))
        self.info.link.from_numpy(links_idx)
        self.info.local_pos.from_numpy(local.astype(gs.np_float))
        self.state.multiplier.fill(0.0)
        self.state.stiffness.fill(solver._k_start)
        self.coordinate_info = None
        self.coordinate_state = None
        self.affects = None
        self.articulation_pose = None
        if self.is_articulated:
            pose_type = qd.types.struct(pos=gs.qd_vec3, quat=gs.qd_vec4)
            self.articulation_pose = pose_type.field(shape=(self.rigid.n_links, solver._B))
            coordinate_info_type = qd.types.struct(qpos_idx=gs.qd_int, joint_idx=gs.qd_int)
            coordinate_state_type = qd.types.struct(previous=gs.qd_float, predicted=gs.qd_float, velocity=gs.qd_float)
            self.coordinate_info = coordinate_info_type.field(shape=self.rigid.n_dofs)
            self.coordinate_state = coordinate_state_type.field(shape=(self.rigid.n_dofs, solver._B))
            self.affects = qd.field(dtype=gs.qd_int, shape=(self.rigid.n_dofs, self.n_attachments))
            joints = [joint for joint in self.rigid.joints if joint.type == gs.JOINT_TYPE.REVOLUTE]
            self.coordinate_info.qpos_idx.from_numpy(np.array([joint.q_start for joint in joints], dtype=gs.np_int))
            self.coordinate_info.joint_idx.from_numpy(np.array([joint.idx for joint in joints], dtype=gs.np_int))
            affects = np.zeros((self.rigid.n_dofs, self.n_attachments), dtype=gs.np_int)
            for i_a, link in enumerate(links):
                while True:
                    affects[link.dof_start : link.dof_end, i_a] = 1
                    if link.parent_idx < 0:
                        break
                    link = self.rigid.links[link.parent_idx]
            self.affects.from_numpy(affects)


@qd.func
def func_attachment_pose(i_a, i_b, attachment: qd.template()):
    pos = gs.qd_vec3(0.0, 0.0, 0.0)
    quat = gs.qd_vec4(1.0, 0.0, 0.0, 0.0)
    if qd.static(attachment.is_articulated):
        i_l = attachment.info[i_a].link
        pos = attachment.articulation_pose[i_l, i_b].pos
        quat = attachment.articulation_pose[i_l, i_b].quat
    else:
        pos = attachment.link_state[i_b].pos
        quat = attachment.link_state[i_b].quat
    return pos, quat


@qd.kernel
def kernel_begin_attachment(
    f: int, solver: qd.template(), attachment: qd.template(), dyn_state: DynState, rigid_info: RigidInfo
):
    for i_b in range(solver._B):
        if not solver.env_failed[i_b]:
            pos = gs.qd_vec3(rigid_info.qpos[0, i_b], rigid_info.qpos[1, i_b], rigid_info.qpos[2, i_b])
            quat = gs.qd_vec4(
                rigid_info.qpos[3, i_b], rigid_info.qpos[4, i_b], rigid_info.qpos[5, i_b], rigid_info.qpos[6, i_b]
            )
            velocity = gs.qd_vec3(0.0, 0.0, 0.0)
            angular = gs.qd_vec3(0.0, 0.0, 0.0)
            inertia_local = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i in qd.static(range(3)):
                velocity[i] = dyn_state.dofs.vel[i, i_b] + solver._substep_dt * dyn_state.dofs.acc[i, i_b]
                angular[i] = dyn_state.dofs.vel[i + 3, i_b] + solver._substep_dt * dyn_state.dofs.acc[i + 3, i_b]
                for j in qd.static(range(3)):
                    inertia_local[i, j] = rigid_info.mass_mat[i + 3, j + 3, i_b]
            rotation = gu.qd_quat_to_R(quat, gs.EPS)
            predicted_pos = pos + solver._substep_dt * velocity
            predicted_quat = func_quaternion_update(quat, solver._substep_dt * (rotation @ angular))
            attachment.link_state[i_b].previous_pos = pos
            attachment.link_state[i_b].previous_quat = quat
            attachment.link_state[i_b].predicted_pos = predicted_pos
            attachment.link_state[i_b].predicted_quat = predicted_quat
            attachment.link_state[i_b].pos = predicted_pos
            attachment.link_state[i_b].quat = predicted_quat
            attachment.link_state[i_b].inertia = rotation @ inertia_local @ rotation.transpose()
            attachment.link_state[i_b].mass = rigid_info.mass_mat[0, 0, i_b]
    for i_a, i_b in qd.ndrange(attachment.n_attachments, solver._B):
        if not solver.env_failed[i_b]:
            attachment.previous_error[i_a, i_b] = (
                solver.verts[f, attachment.info[i_a].vertex, i_b].pos
                - attachment.link_state[i_b].previous_pos
                - gu.qd_transform_by_quat_fast(attachment.info[i_a].local_pos, attachment.link_state[i_b].previous_quat)
            )
            attachment.state[i_a, i_b].multiplier *= attachment.alpha * attachment.gamma
            attachment.state[i_a, i_b].stiffness = qd.max(
                solver._k_start, attachment.gamma * attachment.state[i_a, i_b].stiffness
            )


@qd.func
def func_solve_attachment_link(f, i_b, solver: qd.template(), attachment: qd.template()):
    state = attachment.link_state[i_b]
    force = qd.Vector.zero(gs.qd_float, 6)
    hessian = qd.Matrix.zero(gs.qd_float, 6, 6)
    inertia_h = state.inertia / solver._substep_dt**2
    angular_force = -inertia_h @ func_quaternion_difference(state.quat, state.predicted_quat)
    for i in qd.static(range(3)):
        force[i] = -state.mass / solver._substep_dt**2 * (state.pos[i] - state.predicted_pos[i])
        force[i + 3] = angular_force[i]
        hessian[i, i] = state.mass / solver._substep_dt**2
        for j in qd.static(range(3)):
            hessian[i + 3, j + 3] = inertia_h[i, j]
    for i_a in range(attachment.n_attachments):
        soft_force, soft_hessian, rigid_force, rigid_hessian = func_attachment_blocks(
            solver.verts[f + 1, attachment.info[i_a].vertex, i_b].pos,
            state.pos,
            state.quat,
            attachment.info[i_a].local_pos,
            attachment.state[i_a, i_b].multiplier,
            attachment.state[i_a, i_b].stiffness,
            attachment.previous_error[i_a, i_b],
            attachment.alpha,
        )
        force += rigid_force
        hessian += rigid_hessian
    if qd.static(solver.has_contact):
        force_c, hessian_c = func_contact_link_terms(
            f, attachment.free_link_idx, i_b, state.pos, solver, solver.contact
        )
        force += force_c
        hessian += hessian_c
    increment = func_ldlt6_solve(hessian, force)
    attachment.link_state[i_b].pos += increment[:3]
    attachment.link_state[i_b].quat = func_quaternion_update(state.quat, increment[3:6])
    if qd.static(solver.has_contact):
        func_refresh_link_vertices(
            attachment.free_link_idx,
            i_b,
            attachment.link_state[i_b].pos,
            attachment.link_state[i_b].quat,
            solver.contact,
        )


@qd.func
def func_update_attachment_dual(f, i_a, i_b, solver: qd.template(), attachment: qd.template()):
    pos, quat = func_attachment_pose(i_a, i_b, attachment)
    error = (
        solver.verts[f + 1, attachment.info[i_a].vertex, i_b].pos
        - pos
        - gu.qd_transform_by_quat_fast(attachment.info[i_a].local_pos, quat)
        - attachment.alpha * attachment.previous_error[i_a, i_b]
    )
    stiffness = attachment.state[i_a, i_b].stiffness
    attachment.state[i_a, i_b].multiplier += solver._constraint_dual_relaxation * stiffness * error
    attachment.state[i_a, i_b].stiffness = qd.min(
        stiffness + solver._k_start / solver._constraint_tol * error.norm(),
        solver._constraint_k_max_ratio * solver._k_start,
    )


@qd.kernel
def kernel_end_attachment(
    solver: qd.template(), attachment: qd.template(), dyn_state: DynState, rigid_info: RigidInfo, dt: float
):
    for i_b in range(dyn_state.dofs.vel.shape[1]):
        if solver.env_failed[i_b]:
            # a failed environment keeps its pose and velocity
            for j in qd.static(range(7)):
                rigid_info.qpos_next[j, i_b] = rigid_info.qpos[j, i_b]
            for j in qd.static(range(6)):
                dyn_state.dofs.vel_next[j, i_b] = dyn_state.dofs.vel[j, i_b]
        else:
            state = attachment.link_state[i_b]
            velocity = (state.pos - state.previous_pos) / dt
            angular = gu.qd_inv_transform_by_quat(
                func_quaternion_difference(state.quat, state.previous_quat) / dt, state.quat
            )
            for j in qd.static(range(3)):
                rigid_info.qpos_next[j, i_b] = state.pos[j]
                dyn_state.dofs.vel_next[j, i_b] = velocity[j]
                dyn_state.dofs.vel_next[j + 3, i_b] = angular[j]
            for j in qd.static(range(4)):
                rigid_info.qpos_next[j + 3, i_b] = state.quat[j]


@qd.kernel
def kernel_set_attachment_state(
    envs_idx: qd.types.ndarray(), multiplier: qd.types.ndarray(), stiffness: qd.types.ndarray(), state: qd.template()
):
    for i_a, i_b_ in qd.ndrange(state.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            state[i_a, i_b].multiplier[j] = multiplier[i_b, i_a, j]
        state[i_a, i_b].stiffness = stiffness[i_b, i_a]


@qd.kernel
def kernel_set_vertex_state(
    f: int, envs_idx: qd.types.ndarray(), pos: qd.types.ndarray(), vel: qd.types.ndarray(), vertices: qd.template()
):
    for i_v, i_b_ in qd.ndrange(vertices.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            vertices[f, i_v, i_b].pos[j] = pos[i_b, i_v, j]
            vertices[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]
