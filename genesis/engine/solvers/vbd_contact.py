"""Penalty contact between tetrahedral tissue boundaries and rigid link meshes for vertex block descent (VBD).

A contact pair is a point against a triangle or an edge against an edge. Its gap is C = d - thickness with d the
closest-point distance, and while it is active it carries the augmented Lagrangian energy E = lam C + k C^2 / 2 with
a nonpositive multiplier (the pair only pushes apart). With y = min(lam + k C, 0) the force on a participant is
-y dd/dx, where dd/dx = n for the point, -w_i n for the triangle vertices at closest-point weights w, and
(1 - s) n, s n, -(1 - t) n, -t n for the edge endpoints at closest-point parameters s, t. The curvature block of
each participant is the Gauss-Newton term k g g^T; for the point it is at least the exact block because the distance
to a convex primitive is convex (see spikes/verify_avbd_contact_math.py in the application repository).

Friction is the smoothed Coulomb dissipation of the VBD paper (Eq. 15) on the relative tangential slide of the two
participants over the substep, with the normal force, the normal and the closest-point weights frozen at the current
iterate; the block is mu lam_n g P, which is at least the exact Hessian because g' <= 0.

Boundary triangles are wound outward at build (tissue faces away from their tetrahedron, closed rigid meshes to
positive volume), and over the face the point-triangle distance is signed by that normal: a point that got behind
the surface within a substep is pushed back out instead of further in, and a point a whole layer behind fails the
step. Edge-edge pairs are unsigned; their crossing is the sign change of the edges' triple product.

Candidate pairs are collected once per substep from a uniform hash grid of the predicted positions, within one
margin of the thickness, and the multipliers start from zero every substep: the pair set changes with the
geometry, so there is no persistent identity to transport them along. The grid is rebuilt per substep, the pair
list is fixed for its sweeps and symmetric by construction.
"""

from typing import NamedTuple

import numpy as np
import torch

import igl
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_update_cartesian_space_entity
from genesis.utils.array_class import DynInfo, DynState, ErrorCode, RigidInfo
from genesis.utils.misc import qd_to_torch, tensor_to_array

ROLE_POINT = 0
ROLE_TRIANGLE = 1  # roles 1..3: triangle vertex role - 1
ROLE_EDGE_A = 4  # roles 4..5: endpoint of the first edge
ROLE_EDGE_B = 6  # roles 6..7: endpoint of the second edge


class ContactDiagnostics(NamedTuple):
    """Per-environment contact state of the last substep: candidate pair counts, the largest motion of a tissue
    and of a rigid contact vertex over the substep (m), and the raw error word."""

    n_point_pairs: torch.Tensor
    n_edge_pairs: torch.Tensor
    max_tissue_motion: torch.Tensor
    max_rigid_motion: torch.Tensor
    errno: torch.Tensor


class VBDContact:
    def __init__(self, solver, entities, colliders, prescribed, rules):
        self.solver = solver
        self.colliders = [link for link, _ in colliders]
        # a prescribed collider is a fixed-base rigid entity: every link's geoms collide, the base pose is driven
        colliders = list(colliders) + [(link, group) for entity, group, _ in prescribed for link in entity.links]
        n_groups = 1 + max(
            [entity.material.collision_group for entity in entities]
            + [group for _, group in colliders]
            + [max(r[:2]) for r in rules]
        )
        stiffness = np.zeros((n_groups, n_groups))
        friction = np.zeros((n_groups, n_groups))
        thickness = np.zeros((n_groups, n_groups))
        for group_a, group_b, k, mu, h in rules:
            if stiffness[group_a, group_b] > 0.0:
                gs.raise_exception(f"Contact rule for groups ({group_a}, {group_b}) declared twice.")
            stiffness[group_a, group_b] = stiffness[group_b, group_a] = k
            friction[group_a, group_b] = friction[group_b, group_a] = mu
            thickness[group_a, group_b] = thickness[group_b, group_a] = h
        self.n_groups = n_groups

        # Contact vertices: the tissue boundary vertices first, then every vertex of every rigid collider mesh.
        cv_kind, cv_ref, cv_group, cv_owner, triangles, edges = [], [], [], [], [], []
        rv_link, rv_local = [], []
        for entity in entities:
            elems = entity.elems.astype(np.int64)
            faces, tets_idx, _ = igl.boundary_facets(elems)
            faces = self._oriented_outward(faces, elems[tets_idx], tensor_to_array(entity.init_positions))
            boundary, faces_local = np.unique(faces.reshape(-1), return_inverse=True)
            base = len(cv_kind)
            cv_kind.extend([0] * len(boundary))
            cv_ref.extend((entity.v_start + boundary).tolist())
            cv_group.extend([entity.material.collision_group] * len(boundary))
            cv_owner.extend([-1 - entity.idx] * len(boundary))
            triangles.append(base + faces_local.reshape(-1, 3))
            edges.append(base + self._unique_edges(faces_local.reshape(-1, 3)))
        for link, group in colliders:
            if not link.geoms and not any(link is other for entity, _, _ in prescribed for other in entity.links):
                gs.raise_exception(f"Collider link {link.name} has no collision geometry.")
            for geom in link.geoms:
                local = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
                base = len(cv_kind)
                cv_kind.extend([1] * len(local))
                cv_ref.extend(range(len(rv_link), len(rv_link) + len(local)))
                cv_group.extend([group] * len(local))
                cv_owner.extend([link.idx] * len(local))
                rv_link.extend([link.idx] * len(local))
                rv_local.append(local)
                faces = self._faces_with_positive_volume(geom.init_faces.astype(np.int64), local)
                triangles.append(base + faces)
                edges.append(base + self._unique_edges(faces))
        triangles = np.concatenate(triangles)
        edges = np.concatenate(edges)
        self.n_cv = len(cv_kind)
        self.n_rv = len(rv_link)
        self.n_triangles = len(triangles)
        self.n_edges = len(edges)

        cv_type = qd.types.struct(kind=gs.qd_int, ref=gs.qd_int, group=gs.qd_int, owner=gs.qd_int)
        self.cv_info = cv_type.field(shape=self.n_cv, layout=qd.Layout.SOA)
        self.cv_info.kind.from_numpy(np.array(cv_kind, dtype=gs.np_int))
        self.cv_info.ref.from_numpy(np.array(cv_ref, dtype=gs.np_int))
        self.cv_info.group.from_numpy(np.array(cv_group, dtype=gs.np_int))
        self.cv_info.owner.from_numpy(np.array(cv_owner, dtype=gs.np_int))
        self.vertex_cv = qd.field(dtype=gs.qd_int, shape=solver.n_vertices)
        lookup = np.full(solver.n_vertices, -1, dtype=gs.np_int)
        tissue = np.array(cv_kind) == 0
        lookup[np.array(cv_ref)[tissue]] = np.flatnonzero(tissue)
        self.vertex_cv.from_numpy(lookup)
        self.rv_link = qd.field(dtype=gs.qd_int, shape=max(self.n_rv, 1))
        self.rv_local = qd.Vector.field(3, dtype=gs.qd_float, shape=max(self.n_rv, 1))
        self.rv_cv = qd.field(dtype=gs.qd_int, shape=max(self.n_rv, 1))
        if self.n_rv:
            self.rv_link.from_numpy(np.array(rv_link, dtype=gs.np_int))
            self.rv_local.from_numpy(np.concatenate(rv_local).astype(gs.np_float))
            self.rv_cv.from_numpy(np.flatnonzero(np.array(cv_kind) == 1).astype(gs.np_int))
        # rigid contact vertices grouped per link, for the rigid blocks of the coupled solve
        rigid = solver.sim.rigid_solver
        order = (
            np.argsort(np.array(rv_link, dtype=np.int64), kind="stable") if self.n_rv else np.zeros(0, dtype=np.int64)
        )
        self.link_rv_offset = qd.field(dtype=gs.qd_int, shape=rigid.n_links + 1)
        self.link_rv_offset.from_numpy(
            np.searchsorted(np.array(rv_link, dtype=np.int64)[order], np.arange(rigid.n_links + 1)).astype(gs.np_int)
        )
        self.link_rv = qd.field(dtype=gs.qd_int, shape=max(self.n_rv, 1))
        if self.n_rv:
            self.link_rv.from_numpy(order.astype(gs.np_int))
        # dof_moves_link[i_d, i_l]: the hinge coordinate i_d lies between link i_l and the root
        moves = np.zeros((max(rigid.n_dofs, 1), rigid.n_links), dtype=gs.np_int)
        for link in rigid.links:
            i_l = link.idx
            while True:
                moves[link.dof_start : link.dof_end, i_l] = 1
                if link.parent_idx < 0:
                    break
                link = rigid.links[link.parent_idx]
        self.dof_moves_link = qd.field(dtype=gs.qd_int, shape=moves.shape)
        self.dof_moves_link.from_numpy(moves)
        # world positions of the rigid vertices at the current pose and at the start of the substep
        self.rv_pos = qd.Vector.field(3, dtype=gs.qd_float, shape=(max(self.n_rv, 1), solver._B))
        self.rv_pos_prev = qd.Vector.field(3, dtype=gs.qd_float, shape=(max(self.n_rv, 1), solver._B))
        self.tri_cv = qd.field(dtype=gs.qd_ivec3, shape=self.n_triangles)
        self.tri_cv.from_numpy(triangles.astype(gs.np_int))
        self.edge_cv = qd.field(dtype=gs.qd_ivec2, shape=self.n_edges)
        self.edge_cv.from_numpy(edges.astype(gs.np_int))
        # edges incident to a contact vertex, as a CSR, for the edge-edge candidate search
        order = np.argsort(edges.reshape(-1), kind="stable")
        self.cv_edge_offset = qd.field(dtype=gs.qd_int, shape=self.n_cv + 1)
        self.cv_edge_offset.from_numpy(
            np.searchsorted(edges.reshape(-1)[order], np.arange(self.n_cv + 1)).astype(gs.np_int)
        )
        self.cv_edge = qd.field(dtype=gs.qd_int, shape=2 * self.n_edges)
        self.cv_edge.from_numpy((order // 2).astype(gs.np_int))

        self.rule_stiffness = qd.field(dtype=gs.qd_float, shape=(n_groups, n_groups))
        self.rule_friction = qd.field(dtype=gs.qd_float, shape=(n_groups, n_groups))
        self.rule_thickness = qd.field(dtype=gs.qd_float, shape=(n_groups, n_groups))
        self.rule_stiffness.from_numpy(stiffness.astype(gs.np_float))
        self.rule_friction.from_numpy(friction.astype(gs.np_float))
        self.rule_thickness.from_numpy(thickness.astype(gs.np_float))
        self.max_thickness = float(thickness.max())
        # a candidate is any pair within thickness + margin at the predicted positions; the margin is also the
        # largest motion a contact vertex may make in one substep without a candidate being missed
        self.margin = self.max_thickness
        self.cell = 2.0 * (self.max_thickness + self.margin)
        self.hash_buckets = 2 * self.n_cv
        self.hash_cap = solver._contact_cell_cap
        self.cell_n = qd.field(dtype=gs.qd_int, shape=(self.hash_buckets, solver._B))
        self.cell_v = qd.field(dtype=gs.qd_int, shape=(self.hash_buckets, self.hash_cap, solver._B))
        self.cell_of = qd.Vector.field(3, dtype=gs.qd_int, shape=(self.n_cv, solver._B))

        pair_type = qd.types.struct(a=gs.qd_int, b=gs.qd_int, lam=gs.qd_float, k=gs.qd_float)
        self.pair_cap = solver._contact_pair_cap
        self.pt_pairs = pair_type.field(shape=(self.pair_cap, solver._B), layout=qd.Layout.SOA)
        self.ee_pairs = pair_type.field(shape=(self.pair_cap, solver._B), layout=qd.Layout.SOA)
        self.n_pt = qd.field(dtype=gs.qd_int, shape=solver._B)
        self.n_ee = qd.field(dtype=gs.qd_int, shape=solver._B)
        # The pairs each contact vertex takes part in, as a CSR rebuilt every substep: a count per vertex, its
        # exclusive prefix sum, and the flat list of slots (8 * pair + role, pairs of the edge list offset by
        # pair_cap). Every pair registers at most four vertices, which bounds the flat list by the pair caps.
        self.cv_slot_n = qd.field(dtype=gs.qd_int, shape=(self.n_cv, solver._B))
        self.cv_slot_offset = qd.field(dtype=gs.qd_int, shape=(self.n_cv + 1, solver._B))
        self.cv_slot = qd.field(dtype=gs.qd_int, shape=(8 * self.pair_cap, solver._B))
        self.errno = qd.field(dtype=gs.qd_int, shape=solver._B)
        self.max_motion = qd.field(dtype=gs.qd_float, shape=(2, solver._B))
        # Prescribed collider links follow a pose interpolant from the pose at the start of the step to the target
        # set for its end, sampled at every substep; both ends are state.
        # The targets refer to a reference link of the entity; the base link pose that realizes them follows from
        # the constant transform of the reference link in the base frame, read at build.
        self.prescribed_entities = [entity for entity, _, _ in prescribed]
        self.prescribed_links = [link for _, _, link in prescribed]
        self.n_prescribed = len(prescribed)
        pose_type = qd.types.struct(pos=gs.qd_vec3, quat=gs.qd_vec4)
        self.prescribed_link = qd.field(dtype=gs.qd_int, shape=max(self.n_prescribed, 1))
        self.prescribed_base = qd.field(dtype=gs.qd_int, shape=max(self.n_prescribed, 1))
        self.prescribed_entity = qd.field(dtype=gs.qd_int, shape=max(self.n_prescribed, 1))
        self.prescribed_rel = pose_type.field(shape=max(self.n_prescribed, 1), layout=qd.Layout.SOA)
        if self.n_prescribed:
            self.prescribed_link.from_numpy(np.array([link.idx for link in self.prescribed_links], dtype=gs.np_int))
            self.prescribed_base.from_numpy(
                np.array([entity.base_link.idx for entity in self.prescribed_entities], dtype=gs.np_int)
            )
            self.prescribed_entity.from_numpy(
                np.array([entity._idx_in_solver for entity in self.prescribed_entities], dtype=gs.np_int)
            )
            # link poses are not final while the solvers build, so the fixed chain from the base to the reference
            # link is composed from the links' initial parent-relative poses
            rel = [
                self._relative_pose(entity, link)
                for entity, link in zip(self.prescribed_entities, self.prescribed_links)
            ]
            self.prescribed_rel.pos.from_numpy(np.stack([pos for pos, _ in rel]).astype(gs.np_float))
            self.prescribed_rel.quat.from_numpy(np.stack([quat for _, quat in rel]).astype(gs.np_float))
        self.prescribed_start = pose_type.field(shape=(max(self.n_prescribed, 1), solver._B), layout=qd.Layout.SOA)
        self.prescribed_target = pose_type.field(shape=(max(self.n_prescribed, 1), solver._B), layout=qd.Layout.SOA)
        # wrench (world force, world torque about the link origin) the tissue applies to each collider link
        self.link_reaction = qd.Vector.field(6, dtype=qd.f64, shape=(solver.sim.rigid_solver.n_links, solver._B))

    @staticmethod
    def _relative_pose(entity, link):
        """Pose of `link` in the frame of the entity's base link, composed along the fixed parent chain."""
        pos = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        links = link.entity.solver.links
        while link is not entity.base_link:
            pos = np.asarray(link.pos) + gu.transform_by_quat(pos, np.asarray(link.quat))
            quat = gu.transform_quat_by_quat(quat, np.asarray(link.quat))
            link = links[link.parent_idx]
        return pos, quat

    @staticmethod
    def _oriented_outward(faces, tets, positions):
        """Boundary faces wound so their normal points away from the tetrahedron each one belongs to."""
        opposite = np.array([[v for v in tet if v not in face][0] for face, tet in zip(faces, tets)])
        a, b, c = (positions[faces[:, i]] for i in range(3))
        inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), positions[opposite] - a) > 0.0
        faces = faces.copy()
        faces[inward] = faces[inward][:, [0, 2, 1]]
        return faces

    @staticmethod
    def _faces_with_positive_volume(faces, positions):
        """Faces of a closed mesh wound so the enclosed signed volume is positive (normals outward). An open mesh
        keeps its authored winding."""
        a, b, c = (positions[faces[:, i]] for i in range(3))
        volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
        if volume < 0.0:
            return faces[:, [0, 2, 1]]
        return faces

    @staticmethod
    def _unique_edges(faces):
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        return np.unique(edges, axis=0)

    def diagnostics(self):
        motion = qd_to_torch(self.max_motion, transpose=True)
        return ContactDiagnostics(
            qd_to_torch(self.n_pt), qd_to_torch(self.n_ee), motion[:, 0], motion[:, 1], qd_to_torch(self.errno)
        )

    def reactions(self):
        """Wrench on each registered collider, shape (B, n_colliders, 6): world force then world torque about the
        collider's origin (the link origin, or the base link origin of a prescribed entity), from the pair state at
        the end of the last substep. Rigid colliders come first, then prescribed entities, in declaration order."""
        wrenches = qd_to_torch(self.link_reaction, transpose=True).to(gs.tc_float)
        links_pos = qd_to_torch(self.solver.sim.rigid_solver.dyn_state.links.pos, transpose=True).to(gs.tc_float)
        parts = [wrenches[:, [link.idx for link in self.colliders]]]
        for entity, reference in zip(self.prescribed_entities, self.prescribed_links):
            links_idx = [link.idx for link in entity.links]
            force = wrenches[:, links_idx, :3]
            lever = links_pos[:, links_idx] - links_pos[:, [reference.idx]]
            torque = wrenches[:, links_idx, 3:] + torch.linalg.cross(lever, force)
            parts.append(torch.cat((force.sum(dim=1), torque.sum(dim=1)), dim=-1)[:, None])
        return torch.cat(parts, dim=1)


@qd.func
def func_slerp(q0, q1, t):
    """Shortest-arc spherical interpolation between two unit quaternions."""
    dot = q0.dot(q1)
    q1_aligned = q1
    if dot < 0.0:
        q1_aligned = -q1
        dot = -dot
    result = (q0 + t * (q1_aligned - q0)).normalized()
    if dot < 0.9995:
        theta = qd.acos(dot)
        result = (qd.sin((1.0 - t) * theta) * q0 + qd.sin(t * theta) * q1_aligned) / qd.sin(theta)
    return result


@qd.kernel
def kernel_prescribe_links(
    fraction: float,
    is_step_start: qd.template(),
    contact: qd.template(),
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
):
    """Pose of every prescribed entity at the given fraction of the step: the base link pose is written where the
    rigid solver reads a fixed root link's pose (its info) and where the current pose lives (its state), then the
    entity's forward kinematics places its other links and geoms. At the first substep of a step the interpolant
    restarts from the pose the reference link holds, so a step without a new target holds the previous one."""
    if qd.static(is_step_start):
        for i_p, i_b in qd.ndrange(contact.n_prescribed, dyn_state.links.pos.shape[1]):
            i_l = contact.prescribed_link[i_p]
            contact.prescribed_start[i_p, i_b].pos = dyn_state.links.pos[i_l, i_b]
            contact.prescribed_start[i_p, i_b].quat = dyn_state.links.quat[i_l, i_b]
    for i_p, i_b in qd.ndrange(contact.n_prescribed, dyn_state.links.pos.shape[1]):
        i_l = contact.prescribed_base[i_p]
        link_pos = contact.prescribed_start[i_p, i_b].pos + fraction * (
            contact.prescribed_target[i_p, i_b].pos - contact.prescribed_start[i_p, i_b].pos
        )
        link_quat = func_slerp(
            contact.prescribed_start[i_p, i_b].quat, contact.prescribed_target[i_p, i_b].quat, fraction
        )
        quat = gu.qd_transform_quat_by_quat(link_quat, gu.qd_inv_quat(contact.prescribed_rel[i_p].quat))
        pos = link_pos - gu.qd_transform_by_quat(contact.prescribed_rel[i_p].pos, quat)
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        dyn_info.links.pos[I_l] = pos
        dyn_info.links.quat[I_l] = quat
        dyn_state.links.pos[i_l, i_b] = pos
        dyn_state.links.quat[i_l, i_b] = quat
        func_update_cartesian_space_entity(
            contact.prescribed_entity[i_p],
            i_b,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            force_update_fixed_geoms=True,
            is_backward=False,
        )


@qd.kernel
def kernel_set_prescribed_targets(
    pos: qd.types.ndarray(), quat: qd.types.ndarray(), contact: qd.template(), dyn_state: DynState
):
    """Targets for the end of the next step."""
    for i_p, i_b in qd.ndrange(contact.n_prescribed, dyn_state.links.pos.shape[1]):
        for j in qd.static(range(3)):
            contact.prescribed_target[i_p, i_b].pos[j] = pos[i_b, i_p, j]
        for j in qd.static(range(4)):
            contact.prescribed_target[i_p, i_b].quat[j] = quat[i_b, i_p, j]


@qd.func
def func_cv_pos(f, cv, i_b, solver: qd.template(), contact: qd.template()):
    pos = gs.qd_vec3(0.0, 0.0, 0.0)
    if contact.cv_info[cv].kind == 0:
        pos = solver.verts[f + 1, contact.cv_info[cv].ref, i_b].pos
    else:
        pos = contact.rv_pos[contact.cv_info[cv].ref, i_b]
    return pos


@qd.func
def func_cv_pos_prev(f, cv, i_b, solver: qd.template(), contact: qd.template()):
    pos = gs.qd_vec3(0.0, 0.0, 0.0)
    if contact.cv_info[cv].kind == 0:
        pos = solver.verts[f, contact.cv_info[cv].ref, i_b].pos
    else:
        pos = contact.rv_pos_prev[contact.cv_info[cv].ref, i_b]
    return pos


@qd.func
def func_point_triangle_weights(x, a, b, c):
    """Barycentric weights of the closest point of triangle (a, b, c) to x (Ericson, Real-Time Collision
    Detection, section 5.1.5). Every closest-feature region is handled: the face, the three edges and the three
    vertices."""
    ab = b - a
    ac = c - a
    ap = x - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    weights = gs.qd_vec3(1.0, 0.0, 0.0)
    is_done = False
    if d1 <= 0.0 and d2 <= 0.0:
        is_done = True
    bp = x - b
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    if not is_done and d3 >= 0.0 and d4 <= d3:
        weights = gs.qd_vec3(0.0, 1.0, 0.0)
        is_done = True
    vc = d1 * d4 - d3 * d2
    if not is_done and vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        weights = gs.qd_vec3(1.0 - v, v, 0.0)
        is_done = True
    cp = x - c
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)
    if not is_done and d6 >= 0.0 and d5 <= d6:
        weights = gs.qd_vec3(0.0, 0.0, 1.0)
        is_done = True
    vb = d5 * d2 - d1 * d6
    if not is_done and vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        weights = gs.qd_vec3(1.0 - w, 0.0, w)
        is_done = True
    va = d3 * d6 - d5 * d4
    if not is_done and va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        weights = gs.qd_vec3(0.0, 1.0 - w, w)
        is_done = True
    if not is_done:
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        weights = gs.qd_vec3(1.0 - v - w, v, w)
    return weights


@qd.func
def func_segment_parameters(a, b, c, d):
    """Parameters (s, t) of the closest points a + s (b - a) and c + t (d - c) of two segments (Ericson, section
    5.1.9), both clamped to [0, 1]. Parallel segments take s = 0 and the matching t."""
    d1 = b - a
    d2 = d - c
    r = a - c
    aa = d1.dot(d1)
    e = d2.dot(d2)
    f = d2.dot(r)
    cc = d1.dot(r)
    bb = d1.dot(d2)
    denom = aa * e - bb * bb
    s = gs.qd_float(0.0)
    if denom > 0.0:
        s = qd.math.clamp((bb * f - cc * e) / denom, 0.0, 1.0)
    t = (bb * s + f) / e
    if t < 0.0:
        t = gs.qd_float(0.0)
        s = qd.math.clamp(-cc / aa, 0.0, 1.0)
    elif t > 1.0:
        t = gs.qd_float(1.0)
        s = qd.math.clamp((bb - cc) / aa, 0.0, 1.0)
    return s, t


@qd.func
def func_pt_geometry(f, i_p, i_b, solver: qd.template(), contact: qd.template()):
    """Current distance, unit normal and closest-point weights of a point-triangle pair, with the rule thickness.
    Over the face the distance is signed by the outward triangle normal, so a point behind the surface is pushed
    back out. At an edge or a vertex it is the unsigned distance with the normal from the feature to the point: a
    point outside a convex body near one of its edges lies behind the plane of an adjacent face, and pushing it
    towards that plane's front would push it into the body."""
    cv_x = contact.pt_pairs[i_p, i_b].a
    tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
    x = func_cv_pos(f, cv_x, i_b, solver, contact)
    a = func_cv_pos(f, tri[0], i_b, solver, contact)
    b = func_cv_pos(f, tri[1], i_b, solver, contact)
    c = func_cv_pos(f, tri[2], i_b, solver, contact)
    w = func_point_triangle_weights(x, a, b, c)
    rel = x - w[0] * a - w[1] * b - w[2] * c
    d = rel.norm()
    n = rel / d
    if w[0] > 0.0 and w[1] > 0.0 and w[2] > 0.0:
        n = (b - a).cross(c - a).normalized()
        d = rel.dot(n)
    h = contact.rule_thickness[contact.cv_info[cv_x].group, contact.cv_info[tri[0]].group]
    return d, n, w, h


@qd.func
def func_ee_geometry(f, i_p, i_b, solver: qd.template(), contact: qd.template()):
    """Current distance, unit normal (from the second edge towards the first) and closest-point parameters of an
    edge-edge pair, with the rule thickness."""
    ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
    eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
    a = func_cv_pos(f, ea[0], i_b, solver, contact)
    b = func_cv_pos(f, ea[1], i_b, solver, contact)
    c = func_cv_pos(f, eb[0], i_b, solver, contact)
    d = func_cv_pos(f, eb[1], i_b, solver, contact)
    s, t = func_segment_parameters(a, b, c, d)
    rel = a + s * (b - a) - c - t * (d - c)
    dist = rel.norm()
    n = rel / dist
    h = contact.rule_thickness[contact.cv_info[ea[0]].group, contact.cv_info[eb[0]].group]
    return dist, n, s, t, h


@qd.func
def func_multiplier(lam, k, gap):
    return qd.min(lam + k * gap, 0.0)


@qd.func
def func_pt_forces(f, i_p, i_b, solver: qd.template(), contact: qd.template()):
    """Multiplier y, normal n, closest-point weights w, friction scale and tangential slide of a point-triangle
    pair at the current iterate. The force on the point is -y n - scale * slide and the triangle vertices take
    -w_j of it; scale is mu lam_n g, zero when the pair is inactive."""
    d, n, w, h = func_pt_geometry(f, i_p, i_b, solver, contact)
    y = func_multiplier(contact.pt_pairs[i_p, i_b].lam, contact.pt_pairs[i_p, i_b].k, d - h)
    cv_x = contact.pt_pairs[i_p, i_b].a
    tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
    mu = contact.rule_friction[contact.cv_info[cv_x].group, contact.cv_info[tri[0]].group]
    slide = func_cv_pos(f, cv_x, i_b, solver, contact) - func_cv_pos_prev(f, cv_x, i_b, solver, contact)
    for j in qd.static(range(3)):
        slide -= w[j] * (
            func_cv_pos(f, tri[j], i_b, solver, contact) - func_cv_pos_prev(f, tri[j], i_b, solver, contact)
        )
    slide -= slide.dot(n) * n
    scale = gs.qd_float(0.0)
    if y < 0.0 and mu > 0.0:
        scale = -y * mu * func_friction_scale(slide.norm(), solver._friction_eps_v * solver._substep_dt)
    return y, n, w, scale, slide


@qd.func
def func_ee_forces(f, i_p, i_b, solver: qd.template(), contact: qd.template()):
    """Multiplier y, normal n, closest-point parameters s, t, friction scale and tangential slide of an edge-edge
    pair. The force on the first edge's closest point is -y n - scale * slide, split (1 - s, s) over its endpoints;
    the second edge takes the opposite, split (1 - t, t)."""
    d, n, s, t, h = func_ee_geometry(f, i_p, i_b, solver, contact)
    y = func_multiplier(contact.ee_pairs[i_p, i_b].lam, contact.ee_pairs[i_p, i_b].k, d - h)
    ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
    eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
    mu = contact.rule_friction[contact.cv_info[ea[0]].group, contact.cv_info[eb[0]].group]
    slide = (1.0 - s) * (func_cv_pos(f, ea[0], i_b, solver, contact) - func_cv_pos_prev(f, ea[0], i_b, solver, contact))
    slide += s * (func_cv_pos(f, ea[1], i_b, solver, contact) - func_cv_pos_prev(f, ea[1], i_b, solver, contact))
    slide -= (1.0 - t) * (
        func_cv_pos(f, eb[0], i_b, solver, contact) - func_cv_pos_prev(f, eb[0], i_b, solver, contact)
    )
    slide -= t * (func_cv_pos(f, eb[1], i_b, solver, contact) - func_cv_pos_prev(f, eb[1], i_b, solver, contact))
    slide -= slide.dot(n) * n
    scale = gs.qd_float(0.0)
    if y < 0.0 and mu > 0.0:
        scale = -y * mu * func_friction_scale(slide.norm(), solver._friction_eps_v * solver._substep_dt)
    return y, n, s, t, scale, slide


@qd.func
def func_friction_scale(u_norm, eps):
    g = 1.0 / u_norm
    if u_norm < eps:
        g = 2.0 / eps - u_norm / (eps * eps)
    return g


@qd.func
def func_contact_vertex_terms(f, i_v, i_b, solver: qd.template(), contact: qd.template()):
    """Force and Gauss-Newton block that the contact pairs of tissue vertex i_v apply to it at the current iterate,
    normal penalty and friction together."""
    force = gs.qd_vec3(0.0, 0.0, 0.0)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    cv = contact.vertex_cv[i_v]
    if cv >= 0:
        force, hessian = func_contact_cv_terms(f, cv, i_b, solver, contact)
    return force, hessian


@qd.func
def func_contact_cv_terms(f, cv, i_b, solver: qd.template(), contact: qd.template()):
    """Force and Gauss-Newton block of the contact pairs of contact vertex cv at the current iterate."""
    force = gs.qd_vec3(0.0, 0.0, 0.0)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if cv >= 0:
        for slot in range(contact.cv_slot_offset[cv, i_b], contact.cv_slot_offset[cv + 1, i_b]):
            code = contact.cv_slot[slot, i_b]
            role = code % 8
            i_p = code // 8
            weight = gs.qd_float(0.0)  # dd/dx = weight * n for this participant
            y = gs.qd_float(0.0)
            n = gs.qd_vec3(0.0, 0.0, 1.0)
            k = gs.qd_float(0.0)
            scale = gs.qd_float(0.0)
            slide = gs.qd_vec3(0.0, 0.0, 0.0)
            if role < ROLE_EDGE_A:
                y, n, w, scale, slide = func_pt_forces(f, i_p, i_b, solver, contact)
                k = contact.pt_pairs[i_p, i_b].k
                weight = 1.0
                if role >= ROLE_TRIANGLE:
                    weight = -w[role - ROLE_TRIANGLE]
            else:
                i_p = i_p - contact.pair_cap
                y, n, s_, t_, scale, slide = func_ee_forces(f, i_p, i_b, solver, contact)
                k = contact.ee_pairs[i_p, i_b].k
                if role == ROLE_EDGE_A:
                    weight = 1.0 - s_
                elif role == ROLE_EDGE_A + 1:
                    weight = s_
                elif role == ROLE_EDGE_B:
                    weight = -(1.0 - t_)
                else:
                    weight = -t_
            if y < 0.0:
                force -= weight * (y * n + scale * slide)
                hessian += k * weight * weight * n.outer_product(n)
                hessian += scale * weight * weight * (qd.Matrix.identity(gs.qd_float, 3) - n.outer_product(n))
    return force, hessian


@qd.func
def func_contact_link_terms(f, i_l, i_b, origin, solver: qd.template(), contact: qd.template()):
    """Wrench (force, torque about `origin`) and 6x6 Gauss-Newton block of the contact pairs of every rigid contact
    vertex of link i_l, for a free link with the world-frame rotation increment of the attachment block."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    for c in range(contact.link_rv_offset[i_l], contact.link_rv_offset[i_l + 1]):
        i_r = contact.link_rv[c]
        cv = contact.rv_cv[i_r]
        force, hessian = func_contact_cv_terms(f, cv, i_b, solver, contact)
        r = contact.rv_pos[i_r, i_b] - origin
        jacobian = qd.Matrix.zero(gs.qd_float, 3, 6)
        for row in qd.static(range(3)):
            jacobian[row, row] = 1.0
        jacobian[0, 4] = r[2]
        jacobian[0, 5] = -r[1]
        jacobian[1, 3] = -r[2]
        jacobian[1, 5] = r[0]
        jacobian[2, 3] = r[1]
        jacobian[2, 4] = -r[0]
        force6 += jacobian.transpose() @ force
        hessian6 += jacobian.transpose() @ hessian @ jacobian
    return force6, hessian6


@qd.func
def func_contact_dof_terms(f, i_d, i_b, axis, pivot, solver: qd.template(), contact: qd.template()):
    """Generalized force and Gauss-Newton curvature of the contact pairs of every rigid contact vertex that hinge
    coordinate i_d moves: the vertex Jacobian is axis x (p - pivot)."""
    force = gs.qd_float(0.0)
    curvature = gs.qd_float(0.0)
    for i_l in range(contact.dof_moves_link.shape[1]):
        if contact.dof_moves_link[i_d, i_l]:
            for c in range(contact.link_rv_offset[i_l], contact.link_rv_offset[i_l + 1]):
                i_r = contact.link_rv[c]
                force_v, hessian_v = func_contact_cv_terms(f, contact.rv_cv[i_r], i_b, solver, contact)
                jacobian = axis.cross(contact.rv_pos[i_r, i_b] - pivot)
                force += jacobian.dot(force_v)
                curvature += jacobian.dot(hessian_v @ jacobian)
    return force, curvature


@qd.func
def func_refresh_rigid_vertices(i_b, contact: qd.template(), dyn_state: DynState):
    """Rigid contact vertex positions from the current link poses of the rigid state, after a block moved them."""
    for i_r in range(contact.n_rv):
        i_l = contact.rv_link[i_r]
        contact.rv_pos[i_r, i_b] = gu.qd_transform_by_trans_quat(
            contact.rv_local[i_r], dyn_state.links.pos[i_l, i_b], dyn_state.links.quat[i_l, i_b]
        )


@qd.func
def func_refresh_link_vertices(i_l, i_b, pos, quat, contact: qd.template()):
    """Rigid contact vertex positions of one link from a given pose."""
    for c in range(contact.link_rv_offset[i_l], contact.link_rv_offset[i_l + 1]):
        i_r = contact.link_rv[c]
        contact.rv_pos[i_r, i_b] = gu.qd_transform_by_trans_quat(contact.rv_local[i_r], pos, quat)


@qd.func
def func_cell(x, cell_size):
    return qd.Vector([qd.floor(x[0] / cell_size), qd.floor(x[1] / cell_size), qd.floor(x[2] / cell_size)], dt=gs.qd_int)


@qd.func
def func_cell_hash(c, buckets):
    h = ((c[0] * 73856093) ^ (c[1] * 19349663) ^ (c[2] * 83492791)) % buckets
    if h < 0:
        h += buckets
    return h


@qd.func
def func_shares_tetrahedron(cv_a, cv_b, solver: qd.template(), contact: qd.template()):
    """Whether two tissue contact vertices belong to one tetrahedron (or are the same vertex)."""
    shares = cv_a == cv_b
    if contact.cv_info[cv_a].kind == 0 and contact.cv_info[cv_b].kind == 0:
        i_a = contact.cv_info[cv_a].ref
        i_b_ = contact.cv_info[cv_b].ref
        for c in range(solver.vn_offset[i_a], solver.vn_offset[i_a + 1]):
            if solver.vn_vert[c] == i_b_:
                shares = True
    return shares


@qd.func
def func_may_collide(cv_a, cv_b, contact: qd.template()):
    """Whether a rule joins the groups of two contact vertices and they belong to different rigid links."""
    may = contact.rule_stiffness[contact.cv_info[cv_a].group, contact.cv_info[cv_b].group] > 0.0
    if contact.cv_info[cv_a].kind == 1 and contact.cv_info[cv_b].kind == 1:
        if contact.cv_info[cv_a].owner == contact.cv_info[cv_b].owner:
            may = False
    return may


@qd.kernel
def kernel_begin_contact(f: int, solver: qd.template(), contact: qd.template(), dyn_state: DynState):
    """Rigid vertex positions from the link poses, the hash grid of the predicted positions, then the candidate
    pairs of the substep and the per-vertex lists of the pairs they take part in."""
    for i_r, i_b in qd.ndrange(contact.n_rv, solver._B):
        i_l = contact.rv_link[i_r]
        contact.rv_pos_prev[i_r, i_b] = contact.rv_pos[i_r, i_b]
        contact.rv_pos[i_r, i_b] = gu.qd_transform_by_trans_quat(
            contact.rv_local[i_r], dyn_state.links.pos[i_l, i_b], dyn_state.links.quat[i_l, i_b]
        )
    for h, i_b in qd.ndrange(contact.hash_buckets, solver._B):
        contact.cell_n[h, i_b] = 0
    for cv, i_b in qd.ndrange(contact.n_cv, solver._B):
        contact.cv_slot_n[cv, i_b] = 0
        cell = func_cell(func_cv_pos(f, cv, i_b, solver, contact), contact.cell)
        contact.cell_of[cv, i_b] = cell
        h = func_cell_hash(cell, contact.hash_buckets)
        slot = qd.atomic_add(contact.cell_n[h, i_b], 1)
        if slot < contact.hash_cap:
            contact.cell_v[h, slot, i_b] = cv
        else:
            qd.atomic_or(contact.errno[i_b], ErrorCode.OVERFLOW_VBD_CONTACT_CELL)
    for i_b in range(solver._B):
        contact.n_pt[i_b] = 0
        contact.n_ee[i_b] = 0
    reach = contact.max_thickness + contact.margin
    for i_t, i_b in qd.ndrange(contact.n_triangles, solver._B):
        tri = contact.tri_cv[i_t]
        a = func_cv_pos(f, tri[0], i_b, solver, contact)
        b = func_cv_pos(f, tri[1], i_b, solver, contact)
        c = func_cv_pos(f, tri[2], i_b, solver, contact)
        lo = func_cell(qd.min(qd.min(a, b), c) - reach, contact.cell)
        hi = func_cell(qd.max(qd.max(a, b), c) + reach, contact.cell)
        for ci in range(lo[0], hi[0] + 1):
            for cj in range(lo[1], hi[1] + 1):
                for ck in range(lo[2], hi[2] + 1):
                    cell = qd.Vector([ci, cj, ck], dt=gs.qd_int)
                    h = func_cell_hash(cell, contact.hash_buckets)
                    for slot in range(qd.min(contact.cell_n[h, i_b], contact.hash_cap)):
                        cv = contact.cell_v[h, slot, i_b]
                        if (contact.cell_of[cv, i_b] == cell).all() and func_may_collide(cv, tri[0], contact):
                            is_adjacent = False
                            for j in qd.static(range(3)):
                                if func_shares_tetrahedron(cv, tri[j], solver, contact):
                                    is_adjacent = True
                            if not is_adjacent:
                                x = func_cv_pos(f, cv, i_b, solver, contact)
                                w = func_point_triangle_weights(x, a, b, c)
                                d = (x - w[0] * a - w[1] * b - w[2] * c).norm()
                                h_rule = contact.rule_thickness[
                                    contact.cv_info[cv].group, contact.cv_info[tri[0]].group
                                ]
                                if d < h_rule + contact.margin:
                                    i_p = qd.atomic_add(contact.n_pt[i_b], 1)
                                    if i_p < contact.pair_cap:
                                        contact.pt_pairs[i_p, i_b].a = cv
                                        contact.pt_pairs[i_p, i_b].b = i_t
                                        contact.pt_pairs[i_p, i_b].lam = 0.0
                                        contact.pt_pairs[i_p, i_b].k = contact.rule_stiffness[
                                            contact.cv_info[cv].group, contact.cv_info[tri[0]].group
                                        ]
                                    else:
                                        qd.atomic_or(contact.errno[i_b], ErrorCode.OVERFLOW_VBD_CONTACT_PAIRS)
    for i_e, i_b in qd.ndrange(contact.n_edges, solver._B):
        ea = contact.edge_cv[i_e]
        a = func_cv_pos(f, ea[0], i_b, solver, contact)
        b = func_cv_pos(f, ea[1], i_b, solver, contact)
        lo = func_cell(qd.min(a, b) - reach, contact.cell)
        hi = func_cell(qd.max(a, b) + reach, contact.cell)
        for ci in range(lo[0], hi[0] + 1):
            for cj in range(lo[1], hi[1] + 1):
                for ck in range(lo[2], hi[2] + 1):
                    cell = qd.Vector([ci, cj, ck], dt=gs.qd_int)
                    h = func_cell_hash(cell, contact.hash_buckets)
                    for slot in range(qd.min(contact.cell_n[h, i_b], contact.hash_cap)):
                        cv = contact.cell_v[h, slot, i_b]
                        if (contact.cell_of[cv, i_b] == cell).all():
                            for c_e in range(contact.cv_edge_offset[cv], contact.cv_edge_offset[cv + 1]):
                                j_e = contact.cv_edge[c_e]
                                eb = contact.edge_cv[j_e]
                                # an edge is reached through both endpoints: accept it through its first one, or
                                # through the second when the first lies outside the searched cells
                                first = contact.cell_of[eb[0], i_b]
                                is_first_inside = (first >= lo).all() and (first <= hi).all()
                                is_accepted = cv == eb[0] or not is_first_inside
                                if j_e > i_e and is_accepted and func_may_collide(ea[0], eb[0], contact):
                                    is_adjacent = False
                                    for j in qd.static(range(2)):
                                        for l in qd.static(range(2)):
                                            if func_shares_tetrahedron(ea[j], eb[l], solver, contact):
                                                is_adjacent = True
                                    if not is_adjacent:
                                        c = func_cv_pos(f, eb[0], i_b, solver, contact)
                                        d = func_cv_pos(f, eb[1], i_b, solver, contact)
                                        s, t = func_segment_parameters(a, b, c, d)
                                        dist = (a + s * (b - a) - c - t * (d - c)).norm()
                                        h_rule = contact.rule_thickness[
                                            contact.cv_info[ea[0]].group, contact.cv_info[eb[0]].group
                                        ]
                                        if dist < h_rule + contact.margin:
                                            i_p = qd.atomic_add(contact.n_ee[i_b], 1)
                                            if i_p < contact.pair_cap:
                                                contact.ee_pairs[i_p, i_b].a = i_e
                                                contact.ee_pairs[i_p, i_b].b = j_e
                                                contact.ee_pairs[i_p, i_b].lam = 0.0
                                                contact.ee_pairs[i_p, i_b].k = contact.rule_stiffness[
                                                    contact.cv_info[ea[0]].group, contact.cv_info[eb[0]].group
                                                ]
                                            else:
                                                qd.atomic_or(contact.errno[i_b], ErrorCode.OVERFLOW_VBD_CONTACT_PAIRS)
    # count the pairs of every contact vertex, prefix-sum the counts, then fill the flat slot list
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if i_p < qd.min(contact.n_pt[i_b], contact.pair_cap):
            qd.atomic_add(contact.cv_slot_n[contact.pt_pairs[i_p, i_b].a, i_b], 1)
            tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
            for j in qd.static(range(3)):
                qd.atomic_add(contact.cv_slot_n[tri[j], i_b], 1)
        if i_p < qd.min(contact.n_ee[i_b], contact.pair_cap):
            ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
            eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
            for j in qd.static(range(2)):
                qd.atomic_add(contact.cv_slot_n[ea[j], i_b], 1)
                qd.atomic_add(contact.cv_slot_n[eb[j], i_b], 1)
    for i_b in range(solver._B):
        run = 0
        for cv in range(contact.n_cv):
            contact.cv_slot_offset[cv, i_b] = run
            run += contact.cv_slot_n[cv, i_b]
            contact.cv_slot_n[cv, i_b] = 0
        contact.cv_slot_offset[contact.n_cv, i_b] = run
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if i_p < qd.min(contact.n_pt[i_b], contact.pair_cap):
            func_register_slot(contact.pt_pairs[i_p, i_b].a, 8 * i_p + ROLE_POINT, i_b, contact)
            tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
            for j in qd.static(range(3)):
                func_register_slot(tri[j], 8 * i_p + ROLE_TRIANGLE + j, i_b, contact)
        if i_p < qd.min(contact.n_ee[i_b], contact.pair_cap):
            ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
            eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
            for j in qd.static(range(2)):
                func_register_slot(ea[j], 8 * (i_p + contact.pair_cap) + ROLE_EDGE_A + j, i_b, contact)
                func_register_slot(eb[j], 8 * (i_p + contact.pair_cap) + ROLE_EDGE_B + j, i_b, contact)


@qd.func
def func_register_slot(cv, code, i_b, contact: qd.template()):
    slot = contact.cv_slot_offset[cv, i_b] + qd.atomic_add(contact.cv_slot_n[cv, i_b], 1)
    contact.cv_slot[slot, i_b] = code


@qd.func
def func_contact_dual_update(f, w, solver: qd.template(), contact: qd.template()):
    """Relaxed multiplier update lam <- min(lam + w k C, 0) and the stiffness ramp of every candidate pair (Giles et
    al. 2025 Eq. 11 to 13 with the inequality clamp), after a primal sweep."""
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if i_p < qd.min(contact.n_pt[i_b], contact.pair_cap):
            d, n, weights, h = func_pt_geometry(f, i_p, i_b, solver, contact)
            k = contact.pt_pairs[i_p, i_b].k
            contact.pt_pairs[i_p, i_b].lam = qd.min(contact.pt_pairs[i_p, i_b].lam + w * k * (d - h), 0.0)
            k0 = contact.rule_stiffness[
                contact.cv_info[contact.pt_pairs[i_p, i_b].a].group,
                contact.cv_info[contact.tri_cv[contact.pt_pairs[i_p, i_b].b][0]].group,
            ]
            contact.pt_pairs[i_p, i_b].k = qd.min(
                k + k0 / solver._constraint_tol * qd.max(h - d, 0.0), solver._constraint_k_max_ratio * k0
            )
        if i_p < qd.min(contact.n_ee[i_b], contact.pair_cap):
            d, n, s, t, h = func_ee_geometry(f, i_p, i_b, solver, contact)
            k = contact.ee_pairs[i_p, i_b].k
            contact.ee_pairs[i_p, i_b].lam = qd.min(contact.ee_pairs[i_p, i_b].lam + w * k * (d - h), 0.0)
            k0 = contact.rule_stiffness[
                contact.cv_info[contact.edge_cv[contact.ee_pairs[i_p, i_b].a][0]].group,
                contact.cv_info[contact.edge_cv[contact.ee_pairs[i_p, i_b].b][0]].group,
            ]
            contact.ee_pairs[i_p, i_b].k = qd.min(
                k + k0 / solver._constraint_tol * qd.max(h - d, 0.0), solver._constraint_k_max_ratio * k0
            )


@qd.func
def func_accumulate_reaction(f, cv, force, i_b, solver: qd.template(), contact: qd.template(), dyn_state: DynState):
    """Add the force on a rigid contact vertex, and its moment about the link origin, to the link's wrench."""
    if contact.cv_info[cv].kind == 1:
        i_l = contact.cv_info[cv].owner
        lever = func_cv_pos(f, cv, i_b, solver, contact) - dyn_state.links.pos[i_l, i_b]
        torque = lever.cross(force)
        for j in qd.static(range(3)):
            qd.atomic_add(contact.link_reaction[i_l, i_b][j], qd.cast(force[j], qd.f64))
            qd.atomic_add(contact.link_reaction[i_l, i_b][j + 3], qd.cast(torque[j], qd.f64))


@qd.func
def func_edges_crossed(f, i_b, ea, eb, s, t, solver: qd.template(), contact: qd.template()):
    """Whether two edges whose closest points are interior swapped sides over the substep: the sign of the
    triple product (b - a) x (d - c) . (a - c) is compared at the start and at the end of the substep."""
    crossed = False
    if s > 0.0 and s < 1.0 and t > 0.0 and t < 1.0:
        a0 = func_cv_pos_prev(f, ea[0], i_b, solver, contact)
        b0 = func_cv_pos_prev(f, ea[1], i_b, solver, contact)
        c0 = func_cv_pos_prev(f, eb[0], i_b, solver, contact)
        d0 = func_cv_pos_prev(f, eb[1], i_b, solver, contact)
        a1 = func_cv_pos(f, ea[0], i_b, solver, contact)
        b1 = func_cv_pos(f, ea[1], i_b, solver, contact)
        c1 = func_cv_pos(f, eb[0], i_b, solver, contact)
        d1 = func_cv_pos(f, eb[1], i_b, solver, contact)
        side0 = (b0 - a0).cross(d0 - c0).dot(a0 - c0)
        side1 = (b1 - a1).cross(d1 - c1).dot(a1 - c1)
        crossed = side0 * side1 < 0.0
    return crossed


@qd.kernel
def kernel_end_contact(f: int, solver: qd.template(), contact: qd.template(), dyn_state: DynState):
    """Wrenches on the collider links from the final pair state, and the substep's validity checks: finite
    geometry, no contact vertex moved further than the candidate margin, no pair deeper than its thickness."""
    for i_l, i_b in qd.ndrange(contact.link_reaction.shape[0], solver._B):
        contact.link_reaction[i_l, i_b] = qd.Vector.zero(qd.f64, 6)
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if i_p < qd.min(contact.n_pt[i_b], contact.pair_cap):
            d, n, w, h = func_pt_geometry(f, i_p, i_b, solver, contact)
            if not (d == d):
                qd.atomic_or(contact.errno[i_b], ErrorCode.INVALID_VBD_CONTACT_NAN)
            # signed over the face: a point a whole layer behind the surface has passed through it
            if d < -h:
                qd.atomic_or(contact.errno[i_b], ErrorCode.VBD_CONTACT_CROSSING)
            y, n, w, scale, slide = func_pt_forces(f, i_p, i_b, solver, contact)
            if y < 0.0:
                force = -(y * n + scale * slide)
                tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
                func_accumulate_reaction(f, contact.pt_pairs[i_p, i_b].a, force, i_b, solver, contact, dyn_state)
                for j in qd.static(range(3)):
                    func_accumulate_reaction(f, tri[j], -w[j] * force, i_b, solver, contact, dyn_state)
        if i_p < qd.min(contact.n_ee[i_b], contact.pair_cap):
            y, n, s, t, scale, slide = func_ee_forces(f, i_p, i_b, solver, contact)
            if not (n[0] == n[0]):
                qd.atomic_or(contact.errno[i_b], ErrorCode.INVALID_VBD_CONTACT_NAN)
            ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
            eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
            if func_edges_crossed(f, i_b, ea, eb, s, t, solver, contact):
                qd.atomic_or(contact.errno[i_b], ErrorCode.VBD_CONTACT_CROSSING)
            if y < 0.0:
                force = -(y * n + scale * slide)
                func_accumulate_reaction(f, ea[0], (1.0 - s) * force, i_b, solver, contact, dyn_state)
                func_accumulate_reaction(f, ea[1], s * force, i_b, solver, contact, dyn_state)
                func_accumulate_reaction(f, eb[0], -(1.0 - t) * force, i_b, solver, contact, dyn_state)
                func_accumulate_reaction(f, eb[1], -t * force, i_b, solver, contact, dyn_state)
    for kind, i_b in qd.ndrange(2, solver._B):
        contact.max_motion[kind, i_b] = 0.0
    for cv, i_b in qd.ndrange(contact.n_cv, solver._B):
        motion = (func_cv_pos(f, cv, i_b, solver, contact) - func_cv_pos_prev(f, cv, i_b, solver, contact)).norm()
        qd.atomic_max(contact.max_motion[contact.cv_info[cv].kind, i_b], motion)
        if motion > contact.margin:
            qd.atomic_or(contact.errno[i_b], ErrorCode.VBD_CONTACT_MOTION_BOUND)


@qd.kernel
def kernel_reset_contact(envs_idx: qd.types.ndarray(), contact: qd.template(), dyn_state: DynState):
    """Rigid vertex positions from the current link poses for the selected environments, with no motion carried
    over, and their error words cleared."""
    for i_r, i_b_ in qd.ndrange(contact.n_rv, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = contact.rv_link[i_r]
        contact.rv_pos[i_r, i_b] = gu.qd_transform_by_trans_quat(
            contact.rv_local[i_r], dyn_state.links.pos[i_l, i_b], dyn_state.links.quat[i_l, i_b]
        )
        contact.rv_pos_prev[i_r, i_b] = contact.rv_pos[i_r, i_b]
    for i_b_ in range(envs_idx.shape[0]):
        contact.errno[envs_idx[i_b_]] = 0
    for i_p, i_b_ in qd.ndrange(contact.n_prescribed, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = contact.prescribed_link[i_p]
        contact.prescribed_start[i_p, i_b].pos = dyn_state.links.pos[i_l, i_b]
        contact.prescribed_start[i_p, i_b].quat = dyn_state.links.quat[i_l, i_b]
        contact.prescribed_target[i_p, i_b].pos = dyn_state.links.pos[i_l, i_b]
        contact.prescribed_target[i_p, i_b].quat = dyn_state.links.quat[i_l, i_b]
