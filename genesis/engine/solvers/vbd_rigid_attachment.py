"""Forward coupling of tetrahedral tissue to free or articulated rigid links."""

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.vbd_contact import func_contact_link_terms, func_refresh_link_vertices
from genesis.engine.solvers.vbd_mtu import func_mtu_link_terms, func_refresh_mtu_link_anchors
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
        # Free bodies, in a fixed order. Each owns one 6x6 block; `free_slot` sends a link index to its slot, or
        # to -1 for a link that never moves. A fixed link still carries attachments: it just has no block.
        free_joints = [] if self.is_articulated else [j for j in self.rigid.joints if j.type == gs.JOINT_TYPE.FREE]
        self.n_free = len(free_joints)
        # the links whose pose this class owns for the substep, which is what a caller must know to rescale them
        self.free_links = frozenset(joint.link.idx for joint in free_joints)
        free_slot = np.full(self.rigid.n_links, -1, dtype=gs.np_int)
        for slot, joint in enumerate(free_joints):
            free_slot[joint.link.idx] = slot
        # A plain vertex attachment is the one-vertex, one-weight case of a barycentric one: `add_rigid_attachments`
        # stores (v, v, v, v) with weights (1, 0, 0, 0), so both declarations resolve here to the same representation
        # and the same kernel path, exactly as `SurfaceAnchor` and `TissueAnchor` resolve to one form in vbd_mtu.py.
        vertex_links = [link for entity in entities for link in entity._rigid_links]
        bary_links = [link for entity in entities for link in entity._barycentric_links]
        links = vertex_links + bary_links
        if not self.is_articulated:
            for link in links:
                if free_slot[link.idx] < 0 and any(
                    joint.type != gs.JOINT_TYPE.FIXED for joint in self.rigid.links[link.idx].joints
                ):
                    gs.raise_exception(f"Link {link.name} is neither free nor fixed, so VBD cannot own it.")
        links_idx = np.array([link.idx for link in links], dtype=gs.np_int)

        verts_pieces, weights_pieces, positions_pieces = [], [], []
        for entity in entities:
            v = entity._rigid_vertices_idx
            if len(v):
                local = np.stack([v, v, v, v], axis=1)
                w = np.zeros((len(v), 4), dtype=gs.np_float)
                w[:, 0] = 1.0
                verts_pieces.append(entity.v_start + local)
                weights_pieces.append(w)
                positions_pieces.append(tensor_to_array(entity.init_positions)[v])
        for entity in entities:
            v = entity._barycentric_verts
            if len(v):
                w = entity._barycentric_weights
                verts_pieces.append(entity.v_start + v)
                weights_pieces.append(w)
                init = tensor_to_array(entity.init_positions)
                positions_pieces.append(np.einsum("ac,acd->ad", w, init[v]))
        verts = np.concatenate(verts_pieces).astype(gs.np_int)
        weights = np.concatenate(weights_pieces).astype(gs.np_float)
        positions = np.concatenate(positions_pieces).astype(gs.np_float)

        self.n_attachments = len(verts)
        self.alpha = 0.95
        self.gamma = 0.99
        link_positions = tensor_to_array(self.rigid.get_links_pos()).reshape(solver._B, self.rigid.n_links, 3)[0]
        link_quaternions = tensor_to_array(self.rigid.get_links_quat()).reshape(solver._B, self.rigid.n_links, 4)[0]
        local = gu.inv_transform_by_quat(positions - link_positions[links_idx], link_quaternions[links_idx])
        info_type = qd.types.struct(verts=gs.qd_ivec4, weights=gs.qd_vec4, link=gs.qd_int, local_pos=gs.qd_vec3)
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
        self.link_state = link_type.field(shape=(max(self.n_free, 1), solver._B), layout=qd.Layout.SOA)
        # One pose table for every link, free or fixed, so an attachment reads its link's pose the same way in
        # both paths and a fixed link needs no special case.
        pose_type = qd.types.struct(pos=gs.qd_vec3, quat=gs.qd_vec4)
        self.link_pose = pose_type.field(shape=(self.rigid.n_links, solver._B))
        self.free_slot = qd.field(dtype=gs.qd_int, shape=self.rigid.n_links)
        self.free_slot.from_numpy(free_slot)
        free_info_type = qd.types.struct(link=gs.qd_int, q_start=gs.qd_int, dof_start=gs.qd_int)
        self.free_info = free_info_type.field(shape=max(self.n_free, 1))
        for slot, joint in enumerate(free_joints):
            self.free_info[slot].link = joint.link.idx
            self.free_info[slot].q_start = joint.q_start
            self.free_info[slot].dof_start = joint.dof_start
        # A tissue vertex reaches its attachments through this CSR, exactly as vbd_mtu.py's vert_anchor does: the
        # payload is attachment * 4 + corner, so a vertex shared by several anchors (or by none) is not a special
        # case.
        pairs = sorted(
            (int(verts[i_a, corner]), 4 * i_a + corner)
            for i_a in range(self.n_attachments)
            for corner in range(4)
            if weights[i_a, corner] != 0.0
        )
        counts = np.zeros(solver.n_vertices + 1, dtype=gs.np_int)
        for vertex, _ in pairs:
            counts[vertex + 1] += 1
        self.vert_anchor_offset = qd.field(dtype=gs.qd_int, shape=solver.n_vertices + 1)
        self.vert_anchor_offset.from_numpy(np.cumsum(counts).astype(gs.np_int))
        self.vert_anchor = qd.field(dtype=gs.qd_int, shape=max(len(pairs), 1))
        self.vert_anchor.from_numpy(np.array([slot for _, slot in pairs] or [0], dtype=gs.np_int))
        self.info.verts.from_numpy(verts)
        self.info.weights.from_numpy(weights)
        self.info.link.from_numpy(links_idx)
        self.info.local_pos.from_numpy(local.astype(gs.np_float))
        self.state.multiplier.fill(0.0)
        self.state.stiffness.fill(solver._k_start)
        self.coordinate_info = None
        self.coordinate_state = None
        self.affects = None
        if self.is_articulated:
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
    i_l = attachment.info[i_a].link
    pos = attachment.link_pose[i_l, i_b].pos
    quat = attachment.link_pose[i_l, i_b].quat
    return pos, quat


@qd.func
def func_attachment_point(f, i_a, i_b, solver: qd.template(), attachment: qd.template()):
    """World point one attachment binds to the link: the weighted sum of its up to four tissue vertices. A plain
    vertex attachment is the one-weight-of-one case, so this is the only position the block needs."""
    point = gs.qd_vec3(0.0, 0.0, 0.0)
    for corner in qd.static(range(4)):
        point += attachment.info[i_a].weights[corner] * solver.verts[f, attachment.info[i_a].verts[corner], i_b].pos
    return point


@qd.kernel
def kernel_begin_attachment(
    f: int, solver: qd.template(), attachment: qd.template(), dyn_state: DynState, rigid_info: RigidInfo
):
    # every link's pose starts from the rigid solver's own kinematics: a fixed link keeps this pose all substep
    for i_l, i_b in qd.ndrange(attachment.link_pose.shape[0], solver._B):
        if not solver.env_failed[i_b]:
            attachment.link_pose[i_l, i_b].pos = dyn_state.links.pos[i_l, i_b]
            attachment.link_pose[i_l, i_b].quat = dyn_state.links.quat[i_l, i_b]
    for i_f, i_b in qd.ndrange(attachment.n_free, solver._B):
        if not solver.env_failed[i_b]:
            i_q = attachment.free_info[i_f].q_start
            i_d = attachment.free_info[i_f].dof_start
            pos = gs.qd_vec3(rigid_info.qpos[i_q, i_b], rigid_info.qpos[i_q + 1, i_b], rigid_info.qpos[i_q + 2, i_b])
            quat = gs.qd_vec4(
                rigid_info.qpos[i_q + 3, i_b],
                rigid_info.qpos[i_q + 4, i_b],
                rigid_info.qpos[i_q + 5, i_b],
                rigid_info.qpos[i_q + 6, i_b],
            )
            velocity = gs.qd_vec3(0.0, 0.0, 0.0)
            angular = gs.qd_vec3(0.0, 0.0, 0.0)
            inertia_local = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i in qd.static(range(3)):
                velocity[i] = dyn_state.dofs.vel[i_d + i, i_b] + solver._substep_dt * dyn_state.dofs.acc[i_d + i, i_b]
                angular[i] = (
                    dyn_state.dofs.vel[i_d + i + 3, i_b] + solver._substep_dt * dyn_state.dofs.acc[i_d + i + 3, i_b]
                )
                for j in qd.static(range(3)):
                    inertia_local[i, j] = rigid_info.mass_mat[i_d + i + 3, i_d + j + 3, i_b]
            rotation = gu.qd_quat_to_R(quat, gs.EPS)
            predicted_pos = pos + solver._substep_dt * velocity
            predicted_quat = func_quaternion_update(quat, solver._substep_dt * (rotation @ angular))
            attachment.link_state[i_f, i_b].previous_pos = pos
            attachment.link_state[i_f, i_b].previous_quat = quat
            attachment.link_state[i_f, i_b].predicted_pos = predicted_pos
            attachment.link_state[i_f, i_b].predicted_quat = predicted_quat
            attachment.link_state[i_f, i_b].pos = predicted_pos
            attachment.link_state[i_f, i_b].quat = predicted_quat
            attachment.link_state[i_f, i_b].inertia = rotation @ inertia_local @ rotation.transpose()
            attachment.link_state[i_f, i_b].mass = rigid_info.mass_mat[i_d, i_d, i_b]
            attachment.link_pose[attachment.free_info[i_f].link, i_b].pos = predicted_pos
            attachment.link_pose[attachment.free_info[i_f].link, i_b].quat = predicted_quat
    for i_a, i_b in qd.ndrange(attachment.n_attachments, solver._B):
        if not solver.env_failed[i_b]:
            i_l = attachment.info[i_a].link
            i_f = attachment.free_slot[i_l]
            # a fixed link never moved, so its pose at the start of the substep is the pose it still has
            previous_pos = attachment.link_pose[i_l, i_b].pos
            previous_quat = attachment.link_pose[i_l, i_b].quat
            if i_f >= 0:
                previous_pos = attachment.link_state[i_f, i_b].previous_pos
                previous_quat = attachment.link_state[i_f, i_b].previous_quat
            attachment.previous_error[i_a, i_b] = (
                func_attachment_point(f, i_a, i_b, solver, attachment)
                - previous_pos
                - gu.qd_transform_by_quat_fast(attachment.info[i_a].local_pos, previous_quat)
            )
            attachment.state[i_a, i_b].multiplier *= attachment.alpha * attachment.gamma
            attachment.state[i_a, i_b].stiffness = qd.max(
                solver._k_start, attachment.gamma * attachment.state[i_a, i_b].stiffness
            )


@qd.func
def func_solve_attachment_link(f, i_f, i_b, solver: qd.template(), attachment: qd.template()):
    """One free body's 6x6 block. Bodies couple only through the soft elements and contact, never through the
    mass matrix, so solving them one after another is Gauss-Seidel over blocks, which is what AVBD asks for."""
    i_l = attachment.free_info[i_f].link
    state = attachment.link_state[i_f, i_b]
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
        if attachment.info[i_a].link == i_l:
            soft_force, soft_hessian, rigid_force, rigid_hessian = func_attachment_blocks(
                func_attachment_point(f + 1, i_a, i_b, solver, attachment),
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
        force_c, hessian_c = func_contact_link_terms(f, i_l, i_b, state.pos, solver, solver.contact)
        force += force_c
        hessian += hessian_c
    if qd.static(solver.has_mtu):
        force_m, hessian_m = func_mtu_link_terms(f, i_l, i_b, state.pos, solver, solver.mtu)
        force += force_m
        hessian += hessian_m
    increment = func_ldlt6_solve(hessian, force)
    attachment.link_state[i_f, i_b].pos += increment[:3]
    attachment.link_state[i_f, i_b].quat = func_quaternion_update(state.quat, increment[3:6])
    attachment.link_pose[i_l, i_b].pos = attachment.link_state[i_f, i_b].pos
    attachment.link_pose[i_l, i_b].quat = attachment.link_state[i_f, i_b].quat
    if qd.static(solver.has_contact):
        func_refresh_link_vertices(
            i_l, i_b, attachment.link_state[i_f, i_b].pos, attachment.link_state[i_f, i_b].quat, solver.contact
        )
    if qd.static(solver.has_mtu):
        func_refresh_mtu_link_anchors(
            i_l, i_b, attachment.link_state[i_f, i_b].pos, attachment.link_state[i_f, i_b].quat, solver.mtu
        )


@qd.func
def func_update_attachment_dual(f, i_a, i_b, solver: qd.template(), attachment: qd.template()):
    pos, quat = func_attachment_pose(i_a, i_b, attachment)
    error = (
        func_attachment_point(f + 1, i_a, i_b, solver, attachment)
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
        for i_f in range(attachment.n_free):
            i_q = attachment.free_info[i_f].q_start
            i_d = attachment.free_info[i_f].dof_start
            if solver.env_failed[i_b]:
                # a failed environment keeps its pose and velocity
                for j in qd.static(range(7)):
                    rigid_info.qpos_next[i_q + j, i_b] = rigid_info.qpos[i_q + j, i_b]
                for j in qd.static(range(6)):
                    dyn_state.dofs.vel_next[i_d + j, i_b] = dyn_state.dofs.vel[i_d + j, i_b]
            else:
                state = attachment.link_state[i_f, i_b]
                velocity = (state.pos - state.previous_pos) / dt
                angular = gu.qd_inv_transform_by_quat(
                    func_quaternion_difference(state.quat, state.previous_quat) / dt, state.quat
                )
                for j in qd.static(range(3)):
                    rigid_info.qpos_next[i_q + j, i_b] = state.pos[j]
                    dyn_state.dofs.vel_next[i_d + j, i_b] = velocity[j]
                    dyn_state.dofs.vel_next[i_d + j + 3, i_b] = angular[j]
                for j in qd.static(range(4)):
                    rigid_info.qpos_next[i_q + j + 3, i_b] = state.quat[j]


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
