"""Forward coupling of tetrahedral tissue to free or articulated rigid links."""

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.vbd_contact import func_contact_link_terms, func_refresh_link_active_vertices
from genesis.engine.solvers.vbd_joint import func_joint_link_terms
from genesis.engine.solvers.vbd_mtu import func_mtu_link_terms, func_refresh_mtu_link_anchors
from genesis.engine.solvers.vbd_rigid import func_attachment_blocks, func_ldlt6_solve
from genesis.engine.solvers.vbd_rigid import func_quaternion_difference, func_quaternion_update
from genesis.engine.solvers.vbd_rod_native import func_frame, func_mid_scale, func_tangent
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
        glued_links = [link for entity in entities for link in entity._glued_links]
        if not self.is_articulated:
            for link in glued_links:
                if free_slot[link.idx] < 0 and any(
                    joint.type != gs.JOINT_TYPE.FIXED for joint in self.rigid.links[link.idx].joints
                ):
                    gs.raise_exception(f"Link {link.name} is neither free nor fixed, so VBD cannot glue to it.")

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
        verts = np.concatenate(verts_pieces or [np.zeros((0, 4))]).astype(gs.np_int)
        weights = np.concatenate(weights_pieces or [np.zeros((0, 4))]).astype(gs.np_float)
        positions = np.concatenate(positions_pieces or [np.zeros((0, 3))]).astype(gs.np_float)

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
        self.info = info_type.field(shape=max(self.n_attachments, 1), layout=qd.Layout.SOA)
        self.state = state_type.field(shape=(max(self.n_attachments, 1), solver._B), layout=qd.Layout.SOA)
        self.previous_error = qd.Vector.field(3, dtype=gs.qd_float, shape=(max(self.n_attachments, 1), solver._B))
        self.link_state = link_type.field(shape=(max(self.n_free, 1), solver._B), layout=qd.Layout.SOA)
        # VBD's adaptive initialization (Chen et al. 2024 Eq. 17): the fraction of the external acceleration the
        # next substep's start pose takes, from the acceleration this substep actually had along it. A body in
        # free fall takes all of it, one at rest in a contact or on its ligaments none, which keeps its start
        # out of the contact and off its ligaments' slack points. The first substep takes the whole prediction.
        self.gravity_share = qd.field(dtype=gs.qd_float, shape=(max(self.n_free, 1), solver._B))
        self.gravity_share.fill(1.0)
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
        # A link reaches its own attachments the same way. The alternative is the scan this replaces: one thread
        # a body a sweep over every attachment in the scene to find the few that name the link. A stable sort
        # keeps each link's attachments in ascending order, which is the order the scan visited them in, so the
        # wrench and the block are summed as before.
        by_link = np.argsort(links_idx, kind="stable")
        self.link_attachment_offset = qd.field(dtype=gs.qd_int, shape=self.rigid.n_links + 1)
        self.link_attachment_offset.from_numpy(
            np.concatenate(([0], np.cumsum(np.bincount(links_idx, minlength=self.rigid.n_links)))).astype(gs.np_int)
        )
        self.link_attachment = qd.field(dtype=gs.qd_int, shape=max(self.n_attachments, 1))
        self.link_attachment.from_numpy(by_link.astype(gs.np_int) if self.n_attachments else np.zeros(1, gs.np_int))
        if self.n_attachments:
            self.info.verts.from_numpy(verts)
            self.info.weights.from_numpy(weights)
            self.info.link.from_numpy(links_idx)
            self.info.local_pos.from_numpy(local.astype(gs.np_float))
        self._build_glue(solver, entities, link_positions, link_quaternions, free_slot)
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

    def _build_glue(self, solver, entities, link_positions, link_quaternions, free_slot):
        """Glued vertices (`VBDEntity.add_rigid_glue`): which link carries each, its offset in that link's frame, and
        per link the glued vertices and the tetrahedra that touch them, with a mask of the corners it carries."""
        glued, glue_links = [], []
        for entity in entities:
            if len(entity._glued_vertices_idx):
                if entity._n_triangles or entity._n_stencils:
                    gs.raise_exception("Rigid glue supports tetrahedral tissue only, not shells.")
                glued_local = set(int(v) for v in entity._glued_vertices_idx)
                for pair in np.concatenate((entity._distance_constraints, entity._angle_constraints[:, :2]), axis=0):
                    if glued_local.intersection(int(v) for v in pair):
                        gs.raise_exception("A glued vertex cannot also carry a distance or angle constraint.")
                glued.append(entity.v_start + entity._glued_vertices_idx)
                glue_links.extend(link.idx for link in entity._glued_links)
        glued = np.concatenate(glued or [np.zeros(0)]).astype(gs.np_int)
        rod_vertices = {
            int(v) for entity in solver._rod_entities for v in range(entity.v_start, entity.v_start + entity.n_vertices)
        }
        if rod_vertices.intersection(glued.tolist()) and not solver._rod_native:
            gs.raise_exception("Rigid glue on a rod node needs rod_solver='native': the reference sweep would move it.")
        glue_links = np.array(glue_links, dtype=gs.np_int)
        self.n_glued = len(glued)
        self.has_glue = self.n_glued > 0
        self.glue_link = qd.field(dtype=gs.qd_int, shape=solver.n_vertices)
        self.glue_local = qd.Vector.field(3, dtype=gs.qd_float, shape=solver.n_vertices)
        link_of = np.full(solver.n_vertices, -1, dtype=gs.np_int)
        local = np.zeros((solver.n_vertices, 3), dtype=gs.np_float)
        if self.has_glue:
            if self.is_articulated:
                gs.raise_exception("Rigid glue is not supported on articulated links yet.")
            barycentric = np.concatenate(
                [entity.v_start + entity._barycentric_verts for entity in entities if len(entity._barycentric_verts)]
                or [np.zeros((0, 4), dtype=gs.np_int)]
            )
            if np.isin(glued, barycentric).any():
                gs.raise_exception("A glued vertex cannot also carry a barycentric attachment.")
            rest = np.zeros((solver.n_vertices, 3), dtype=gs.np_float)
            for entity in entities:
                rest[entity.v_start : entity.v_start + entity.n_vertices] = tensor_to_array(entity.init_positions)
            link_of[glued] = glue_links
            local[glued] = gu.inv_transform_by_quat(
                rest[glued] - link_positions[glue_links], link_quaternions[glue_links]
            )
        self.glue_link.from_numpy(link_of)
        self.glue_local.from_numpy(local)
        # per link, its glued vertices
        order = np.argsort(glue_links, kind="stable")
        self.link_glue_vert_offset = qd.field(dtype=gs.qd_int, shape=self.rigid.n_links + 1)
        self.link_glue_vert_offset.from_numpy(
            np.concatenate(([0], np.cumsum(np.bincount(glue_links, minlength=self.rigid.n_links)))).astype(gs.np_int)
        )
        self.link_glue_vert = qd.field(dtype=gs.qd_int, shape=max(self.n_glued, 1))
        self.link_glue_vert.from_numpy(glued[order] if self.n_glued else np.zeros(1, dtype=gs.np_int))
        # per link, every tetrahedron with a corner it carries, and which corners
        entries = {}
        if self.has_glue and solver._n_elements:
            elems = solver.elems_info.v.to_numpy()[: solver._n_elements]
            carried = link_of[elems]  # (n_elems, 4)
            for i_e, corners in enumerate(carried):
                for i_l in set(int(c) for c in corners if c >= 0):
                    mask = sum(1 << r for r in range(4) if corners[r] == i_l)
                    entries.setdefault(i_l, []).append((i_e, mask))
        counts = np.zeros(self.rigid.n_links, dtype=gs.np_int)
        flat_elem, flat_mask = [], []
        for i_l in range(self.rigid.n_links):
            for i_e, mask in entries.get(i_l, []):
                flat_elem.append(i_e)
                flat_mask.append(mask)
            counts[i_l] = len(entries.get(i_l, []))
        self.link_glue_elem_offset = qd.field(dtype=gs.qd_int, shape=self.rigid.n_links + 1)
        self.link_glue_elem_offset.from_numpy(np.concatenate(([0], np.cumsum(counts))).astype(gs.np_int))
        self.link_glue_elem = qd.field(dtype=gs.qd_int, shape=max(len(flat_elem), 1))
        self.link_glue_elem.from_numpy(np.array(flat_elem or [0], dtype=gs.np_int))
        self.link_glue_mask = qd.field(dtype=gs.qd_int, shape=max(len(flat_mask), 1))
        self.link_glue_mask.from_numpy(np.array(flat_mask or [0], dtype=gs.np_int))


@qd.func
def func_rigid_jacobian(r):
    """3x6 Jacobian of a point at offset r from a body's origin under (translation, world rotation increment)."""
    jacobian = qd.Matrix.zero(gs.qd_float, 3, 6)
    for row in qd.static(range(3)):
        jacobian[row, row] = 1.0
    jacobian[0, 4] = r[2]
    jacobian[0, 5] = -r[1]
    jacobian[1, 3] = -r[2]
    jacobian[1, 5] = r[0]
    jacobian[2, 3] = r[1]
    jacobian[2, 4] = -r[0]
    return jacobian


@qd.func
def func_glue_point(i_v, i_b, attachment: qd.template()):
    """World position of glued vertex i_v at its link's current pose."""
    i_l = attachment.glue_link[i_v]
    return gu.qd_transform_by_trans_quat(
        attachment.glue_local[i_v], attachment.link_pose[i_l, i_b].pos, attachment.link_pose[i_l, i_b].quat
    )


@qd.func
def func_refresh_glue(f, i_l, i_b, pos, quat, solver: qd.template(), attachment: qd.template()):
    """Place the vertices glued to link i_l at the given pose."""
    for c in range(attachment.link_glue_vert_offset[i_l], attachment.link_glue_vert_offset[i_l + 1]):
        i_v = attachment.link_glue_vert[c]
        solver.verts[f + 1, i_v, i_b].pos = gu.qd_transform_by_trans_quat(attachment.glue_local[i_v], pos, quat)


@qd.func
def func_glue_link_terms(f, i_l, i_b, origin, solver: qd.template(), attachment: qd.template()):
    """Wrench about `origin` and 6x6 block that the vertices glued to link i_l bring to its block: the inertia and
    gravity of their mass, and the stable neo-Hookean and fibre forces of every tetrahedron that touches them.

    Per element, the curvature between two glued corners r and s is the tetrahedron's own Gauss-Newton block
    V mu (w_r . w_s) I + V lam q_r q_s^T (q = cof w), plus V k (w0_r . a)(w0_s . a) u u^T for a fibre, carried
    through both corners' rigid Jacobians: the same blocks the vertex solve uses on its diagonal, so a glued
    corner and a free one see one law. Summing each corner's diagonal alone would drop the cross terms, which is
    the underestimate `func_contact_link_terms` documents."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    inv_h2 = 1.0 / (solver._substep_dt * solver._substep_dt)
    for c in range(attachment.link_glue_vert_offset[i_l], attachment.link_glue_vert_offset[i_l + 1]):
        i_v = attachment.link_glue_vert[c]
        m_h2 = solver.verts_info[i_v].mass * inv_h2
        x = solver.verts[f + 1, i_v, i_b].pos
        J = func_rigid_jacobian(x - origin)
        force6 += J.transpose() @ (-m_h2 * (x - solver._func_inertia_target(f, i_v, i_b)))
        hessian6 += m_h2 * J.transpose() @ J
    identity = qd.Matrix.identity(gs.qd_float, 3)
    for c in range(attachment.link_glue_elem_offset[i_l], attachment.link_glue_elem_offset[i_l + 1]):
        i_e = attachment.link_glue_elem[c]
        mask = attachment.link_glue_mask[c]
        v = solver.elems_info[i_e].v
        F, B = solver._func_deformation(f + 1, i_e, i_b)
        mu = solver.elems_info[i_e].mu
        lam = solver.elems_info[i_e].lam
        cof = solver._func_cofactor(F)
        P = mu * F + lam * (F.determinant() - (1.0 + mu / lam)) * cof
        V = solver.elems_info[i_e].vol_rest
        k_fiber = solver.elems_info[i_e].k_fiber
        B0 = solver.elems_info[i_e].B_rest
        a = solver.elems_info[i_e].fiber
        p0 = solver.verts[f + 1, v[0], i_b].pos
        Ds = qd.Matrix.cols(
            [solver.verts[f + 1, v[1], i_b].pos - p0, solver.verts[f + 1, v[2], i_b].pos - p0,
             solver.verts[f + 1, v[3], i_b].pos - p0]
        )
        u = (Ds @ B0) @ a
        u_hat = u.normalized()
        for r in qd.static(range(4)):
            if (mask >> r) & 1:
                w_r = solver._func_vertex_weight_static(B, r)
                J_r = func_rigid_jacobian(solver.verts[f + 1, v[r], i_b].pos - origin)
                f_r = -V * (P @ w_r)
                if k_fiber > 0.0:
                    f_fib, _, _ = solver._func_fiber_terms(f + 1, i_e, i_b, solver._func_vertex_weight_static(B0, r))
                    f_r += f_fib
                force6 += J_r.transpose() @ f_r
                for s_ in qd.static(range(4)):
                    if (mask >> s_) & 1:
                        w_s = solver._func_vertex_weight_static(B, s_)
                        J_s = func_rigid_jacobian(solver.verts[f + 1, v[s_], i_b].pos - origin)
                        block = V * mu * w_r.dot(w_s) * identity + V * lam * (cof @ w_r).outer_product(cof @ w_s)
                        if k_fiber > 0.0:
                            block += (
                                V * k_fiber * solver._func_vertex_weight_static(B0, r).dot(a)
                                * solver._func_vertex_weight_static(B0, s_).dot(a) * u_hat.outer_product(u_hat)
                            )
                        hessian6 += J_r.transpose() @ block @ J_s
    if qd.static(solver.has_rod_native):
        force_r, hessian_r = func_glue_rod_terms(f, i_l, i_b, origin, solver, attachment, solver.rod_native)
        force6 += force_r
        hessian6 += hessian_r
    return force6, hessian6


@qd.func
def func_glue_rod_terms(f, i_l, i_b, origin, solver: qd.template(), attachment: qd.template(), rod: qd.template()):
    """The stretch and volume rows of every native rod segment with an end glued to link i_l: the rod pulling on
    the bone. Both rows are linear in the segment's two nodes (d/dx of t = +-1/L), so a segment is carried by
    T = sum over its ends on this link of +-(w/L) J_end, with the cross term when both ride on it; taken once,
    through its first end glued here."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    for c in range(attachment.link_glue_vert_offset[i_l], attachment.link_glue_vert_offset[i_l + 1]):
        i_v = attachment.link_glue_vert[c]
        for side in qd.static(range(2)):
            j = rod.seg_prev[i_v]
            if qd.static(side == 1):
                j = rod.seg_next[i_v]
            if j >= 0:
                a = rod.seg[j].node0
                if i_v == a or attachment.glue_link[a] != i_l:
                    L = rod.seg[j].length
                    t = func_tangent(f, j, i_b, solver, rod)
                    d3 = func_frame(rod.quat[j, i_b])[:, 2]
                    mid = func_mid_scale(j, i_b, rod)
                    ws = rod.seg[j].w_str
                    wv = rod.seg[j].w_vol
                    T_s = qd.Matrix.zero(gs.qd_float, 3, 6)
                    T_v = qd.Matrix.zero(gs.qd_float, 3, 6)
                    for end in qd.static(range(2)):
                        node = a + end
                        if attachment.glue_link[node] == i_l:
                            sign = gs.qd_float(1.0)
                            if qd.static(end == 0):
                                sign = -1.0
                            J = func_rigid_jacobian(solver.verts[f + 1, node, i_b].pos - origin)
                            T_s += (sign * ws / L) * J
                            T_v += (sign * wv * mid * mid / L) * J
                    force6 -= T_s.transpose() @ (ws * (t - d3)) + T_v.transpose() @ (wv * (mid * mid * t - d3))
                    hessian6 += T_s.transpose() @ T_s + T_v.transpose() @ T_v
    return force6, hessian6


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
            # the inertia target stays the prediction; only where the sweeps start moves
            start_pos = predicted_pos
            for i in qd.static(range(3)):
                start_pos[i] -= (1.0 - attachment.gravity_share[i_f, i_b]) * (
                    solver._substep_dt * solver._substep_dt * dyn_state.dofs.acc[i_d + i, i_b]
                )
            attachment.link_state[i_f, i_b].previous_pos = pos
            attachment.link_state[i_f, i_b].previous_quat = quat
            attachment.link_state[i_f, i_b].predicted_pos = predicted_pos
            attachment.link_state[i_f, i_b].predicted_quat = predicted_quat
            attachment.link_state[i_f, i_b].pos = start_pos
            attachment.link_state[i_f, i_b].quat = predicted_quat
            attachment.link_state[i_f, i_b].inertia = rotation @ inertia_local @ rotation.transpose()
            attachment.link_state[i_f, i_b].mass = rigid_info.mass_mat[i_d, i_d, i_b]
            attachment.link_pose[attachment.free_info[i_f].link, i_b].pos = start_pos
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
def func_attachment_link_base(f, i_f, i_b, solver: qd.template(), attachment: qd.template()):
    """The first terms of one free body's block: its inertia, its vertex attachments and its glued tissue."""
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
    for c in range(attachment.link_attachment_offset[i_l], attachment.link_attachment_offset[i_l + 1]):
        i_a = attachment.link_attachment[c]
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
    if qd.static(attachment.has_glue):
        force_g, hessian_g = func_glue_link_terms(f, i_l, i_b, state.pos, solver, attachment)
        force += force_g
        hessian += hessian_g
    return force, hessian


@qd.func
def func_attachment_link_tail(i_f, i_b, force, hessian, solver: qd.template(), attachment: qd.template()):
    """The last terms of the block, after contact and muscle-tendon units: joints and rod contact."""
    i_l = attachment.free_info[i_f].link
    if qd.static(solver.has_joint):
        force_j, hessian_j = func_joint_link_terms(i_l, i_b, attachment, solver.joints)
        force += force_j
        hessian += hessian_j
    if qd.static(solver.has_rod_contact):
        force += solver._rod_contacts[0].force[i_l]
        hessian += solver._rod_contacts[0].hessian[i_l]
    return force, hessian


@qd.func
def func_apply_attachment_link(f, i_f, i_b, force, hessian, solver: qd.template(), attachment: qd.template()):
    """Solve the block, move the body and refresh every cache that follows its pose."""
    i_l = attachment.free_info[i_f].link
    state = attachment.link_state[i_f, i_b]
    increment = func_ldlt6_solve(hessian, force)
    attachment.link_state[i_f, i_b].pos += increment[:3]
    attachment.link_state[i_f, i_b].quat = func_quaternion_update(state.quat, increment[3:6])
    attachment.link_pose[i_l, i_b].pos = attachment.link_state[i_f, i_b].pos
    attachment.link_pose[i_l, i_b].quat = attachment.link_state[i_f, i_b].quat
    if qd.static(attachment.has_glue):
        func_refresh_glue(
            f, i_l, i_b, attachment.link_state[i_f, i_b].pos, attachment.link_state[i_f, i_b].quat, solver, attachment
        )
    if qd.static(solver.has_contact):
        func_refresh_link_active_vertices(
            i_l, i_b, attachment.link_state[i_f, i_b].pos, attachment.link_state[i_f, i_b].quat, solver.contact
        )
    if qd.static(solver.has_mtu):
        func_refresh_mtu_link_anchors(
            i_l, i_b, attachment.link_state[i_f, i_b].pos, attachment.link_state[i_f, i_b].quat, solver.mtu
        )


@qd.func
def func_solve_attachment_link(f, i_f, i_b, solver: qd.template(), attachment: qd.template()):
    """One free body's 6x6 block. Bodies couple only through the soft elements and contact, never through the
    mass matrix, so solving them one after another is Gauss-Seidel over blocks, which is what AVBD asks for."""
    force, hessian = func_attachment_link_system(f, i_f, i_b, solver, attachment)
    func_apply_attachment_link(f, i_f, i_b, force, hessian, solver, attachment)


@qd.func
def func_attachment_link_system(f, i_f, i_b, solver: qd.template(), attachment: qd.template()):
    """Negative gradient and 6x6 block of one free body at the current poses, with no state written."""
    i_l = attachment.free_info[i_f].link
    force, hessian = func_attachment_link_base(f, i_f, i_b, solver, attachment)
    if qd.static(solver.has_contact):
        force_c, hessian_c = func_contact_link_terms(
            f, i_l, i_b, attachment.link_state[i_f, i_b].pos, solver, solver.contact
        )
        force += force_c
        hessian += hessian_c
    if qd.static(solver.has_mtu):
        force_m, hessian_m = func_mtu_link_terms(f, i_l, i_b, attachment.link_state[i_f, i_b].pos, solver, solver.mtu)
        force += force_m
        hessian += hessian_m
    return func_attachment_link_tail(i_f, i_b, force, hessian, solver, attachment)


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
                external = gs.qd_vec3(0.0, 0.0, 0.0)
                acceleration = gs.qd_vec3(0.0, 0.0, 0.0)
                for j in qd.static(range(3)):
                    external[j] = dyn_state.dofs.acc[i_d + j, i_b]
                    acceleration[j] = (velocity[j] - dyn_state.dofs.vel[i_d + j, i_b]) / dt
                share = 0.0
                if external.dot(external) > 0.0:
                    share = qd.math.clamp(acceleration.dot(external) / external.dot(external), 0.0, 1.0)
                attachment.gravity_share[i_f, i_b] = share
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
def kernel_set_gravity_share(envs_idx: qd.types.ndarray(), share: qd.types.ndarray(), attachment: qd.template()):
    for i_f, i_b_ in qd.ndrange(attachment.gravity_share.shape[0], envs_idx.shape[0]):
        attachment.gravity_share[i_f, envs_idx[i_b_]] = share[envs_idx[i_b_], i_f]


@qd.kernel
def kernel_set_vertex_state(
    f: int, envs_idx: qd.types.ndarray(), pos: qd.types.ndarray(), vel: qd.types.ndarray(), vertices: qd.template()
):
    for i_v, i_b_ in qd.ndrange(vertices.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            vertices[f, i_v, i_b].pos[j] = pos[i_b, i_v, j]
            vertices[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]
