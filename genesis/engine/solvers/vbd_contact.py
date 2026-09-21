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

# the two kinds of contact pair, as a compile-time switch of the geometry shared by the search and the filter
POINT_TRIANGLE = 0
EDGE_EDGE = 1

# Below this separation (m) the offset between two closest points is taken to have no direction of its own, and
# the geometry's own normal is used instead: the face's for a point-triangle pair, the common perpendicular for
# an edge-edge one. Positions are single precision, so on a body two metres from the origin the last bit is
# already about 0.3 micrometres, and a separation of a nanometre carries no direction worth reading.
CONTACT_DEGENERATE_EPS = 1e-9


# VBD_CONTACT_MOTION_BOUND no longer marks a refused substep (see kernel_end_contact): it marks that this
# substep's safe bound ran out, which the next kernel_begin_contact reads to decide to rebuild. Every other bit
# is a genuine failure and must still latch the environment, which is what this mask keeps in the aggregate check.
# ~x on a positive Python int is -x-1, the correct two's-complement i32 bit pattern for "every bit but this
# one": AND/OR do not care whether the field is read as signed, only bit masking with 0xFFFFFFFF would overflow
# i32's literal range.
_FATAL_ERRNO_MASK = ~int(ErrorCode.VBD_CONTACT_MOTION_BOUND)


class ContactDiagnostics(NamedTuple):
    """Per-environment contact state of the last substep: candidate pair counts, the largest motion of a tissue
    and of a rigid contact vertex over the substep (m), the largest distance either of them ended from the swept
    path the candidate search covered (m), the raw error word, the smallest continuous-collision time of impact
    since the last `clear_toi()` (1 when no substep was rescaled, and always 1 without `VBDOptions.contact_ccd`),
    and how many candidate-set rebuilds this environment has needed since its last reset.

    The motion is reported, not bounded: the search follows it. The deviation is reported too, but neither it
    nor the motion is what triggers a rebuild any more -- `VBDContact.d_budget` is (Wang et al. 2022 Eq. 4),
    and `VBD_CONTACT_MOTION_BOUND` in `errno` marks a substep whose bound ran out and will rebuild next, not a
    refused one.
    """

    n_point_pairs: torch.Tensor
    n_edge_pairs: torch.Tensor
    max_tissue_motion: torch.Tensor
    max_rigid_motion: torch.Tensor
    errno: torch.Tensor
    min_toi: torch.Tensor
    max_tissue_deviation: torch.Tensor
    max_rigid_deviation: torch.Tensor
    rebuild_count: torch.Tensor


class EnvStatus(NamedTuple):
    """Per-environment failure latch of the solver: whether the environment failed, the global substep index of
    the failure (-1 while it runs) and the raw contact error word."""

    is_failed: torch.Tensor
    failed_substep: torch.Tensor
    errno: torch.Tensor


class VBDContact:
    def __init__(self, solver, entities, colliders, prescribed, rules):
        self.solver = solver
        # Only a tissue whose collision group appears in a rule can collide. The others are left out of the
        # contact vertex set entirely: they would find no pair, but would still spend the shared d_budget and
        # trigger rebuilds by their own motion, and a fast connector between two falling bones has no business
        # forcing a contact rebuild it can never take part in.
        ruled = {group for rule in rules for group in rule[:2]}
        entities = [entity for entity in entities if entity.material.collision_group in ruled]
        self.colliders = [link for link, _, _ in colliders]
        # a prescribed collider is a fixed-base rigid entity: every link's geoms collide, the base pose is driven
        colliders = list(colliders) + [(link, group, None) for entity, group, _ in prescribed for link in entity.links]
        n_groups = 1 + max(
            [entity.material.collision_group for entity in entities]
            + [group for _, group, _ in colliders]
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
        # Rest positions of the contact vertices, only ever used to size the grid. A rigid collider's are in its
        # own frame rather than the world, which is enough, because an edge length does not care about the pose.
        cv_rest = []
        for entity in entities:
            if entity.n_triangles:
                # A shell is already a surface: its own triangles are the contact faces, and every vertex of
                # it is on the boundary. There is no tetrahedron to wind the normal against, so the authored
                # winding is the sign convention. The model must declare which side is the lumen; nothing here
                # can infer it, and a wall wound inconsistently will push the wrong way.
                faces = np.asarray(entity.tris, dtype=np.int64)
            else:
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
            cv_rest.append(np.asarray(tensor_to_array(entity.init_positions))[boundary])
        for link, group, regions in colliders:
            if not link.geoms and not any(link is other for entity, _, _ in prescribed for other in entity.links):
                gs.raise_exception(f"Collider link {link.name} has no collision geometry.")
            for geom in link.geoms:
                local = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
                # Orientation first, on the closed mesh, because the volume sign is what says which way is out;
                # only then does a region select part of it.
                faces = self._faces_with_positive_volume(geom.init_faces.astype(np.int64), local)
                if regions is not None:
                    world = gu.transform_by_trans_quat(local, *self._link_rest_pose(link))
                    # A triangle is kept when its bounding sphere reaches into a region: the centroid within the
                    # radius plus the triangle's own longest edge. Testing the corners instead drops a triangle
                    # that is larger than the region and covers it, which the head's rods and the splenial do;
                    # this test keeps a few triangles that only come close, which costs a little and cannot
                    # silently remove a contact.
                    corners = world[faces]
                    centroid = corners.mean(axis=1)
                    span = np.linalg.norm(corners - corners[:, [1, 2, 0]], axis=2).max(axis=1)
                    keep = np.zeros(len(faces), dtype=bool)
                    for x, y, z, radius in regions:
                        keep |= np.linalg.norm(centroid - np.array([x, y, z]), axis=1) <= radius + span
                    faces = faces[keep]
                    if not len(faces):
                        gs.raise_exception(
                            f"No triangle of collider link {link.name} lies inside any of its regions."
                        )
                    used, faces = np.unique(faces.reshape(-1), return_inverse=True)
                    faces = faces.reshape(-1, 3)
                    local = local[used]
                base = len(cv_kind)
                cv_kind.extend([1] * len(local))
                cv_ref.extend(range(len(rv_link), len(rv_link) + len(local)))
                cv_group.extend([group] * len(local))
                cv_owner.extend([link.idx] * len(local))
                rv_link.extend([link.idx] * len(local))
                rv_local.append(local)
                cv_rest.append(np.asarray(local))
                triangles.append(base + faces)
                edges.append(base + self._unique_edges(faces))
        triangles = np.concatenate(triangles)
        edges = np.concatenate(edges)
        rest = np.concatenate(cv_rest)
        self.median_edge = float(np.median(np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)))
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
        # both dimensions are floored at one: a scene whose only contact is tissue against tissue has no
        # rigid link and no rigid dof, and a zero-width field is refused by the backend
        moves = np.zeros((max(rigid.n_dofs, 1), max(rigid.n_links, 1)), dtype=gs.np_int)
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
        # A pair still draws a force once its distance is within thickness + margin (below, `func_pt_forces` /
        # `func_ee_forces` self-gate on it through the inequality-clamped multiplier). A rebuild's search covers
        # further than that, out to thickness + margin_max, precisely so the candidate set stays a safe superset
        # while it is reused across substeps that never rebuild at all -- see margin_max below.
        self.margin = self.max_thickness if solver._contact_margin is None else solver._contact_margin
        if not self.margin > 0.0:
            gs.raise_exception(f"VBDOptions.contact_margin must be above zero, got {self.margin}.")
        # D_max of Wang et al. 2022 ("Fast GPU-Based Two-Way Continuous Collision Handling", Sec. 3.1): a
        # rebuild searches this far so the candidate set stays a safe superset while `d_budget` (D) runs down
        # from it, and `margin` above is D_min, the bound a pair still has to clear to draw a force. Left unset,
        # it defaults to `margin` itself (no extra reach): a build then covers exactly what it always covered,
        # and d_budget starting the substep below margin whenever anything moved means every substep rebuilds,
        # matching the behaviour before this option existed. Multiplying an arbitrary margin up is not a safe
        # general default -- a scene whose margin is already sized against its own geometry (a coarse collider
        # a few tens of millimetres across, say) can have that margin scaled into reach of a face on the far
        # side of the same body, adding candidates that have nothing to do with the approach being tracked and
        # over-constraining the response. Set this explicitly, sized to the scene's own gaps, to amortize
        # rebuilds across substeps.
        self.margin_max = self.margin if solver._contact_margin_max is None else solver._contact_margin_max
        if not self.margin_max >= self.margin:
            gs.raise_exception(
                f"VBDOptions.contact_margin_max ({self.margin_max}) must be at least contact_margin "
                f"({self.margin}): a build that searches less far than a pair still has to clear would let the "
                f"reused set miss pairs margin alone would have caught."
            )
        # Left unset, each pair is judged against its own rule thickness, which is what the check did before the
        # option existed. A single global depth would let the scene's coarsest rule decide for its finest: two
        # rules of 0.05 mm and 5 mm would judge the fine pair at 5 mm, and a vertex twenty layers behind its own
        # face would pass unreported.
        self.crossing_depth_per_pair = solver._contact_crossing_depth is None
        self.crossing_depth = (
            self.max_thickness if solver._contact_crossing_depth is None else solver._contact_crossing_depth
        )
        if not self.crossing_depth > 0.0:
            gs.raise_exception(
                f"VBDOptions.contact_crossing_depth must be above zero, got {self.crossing_depth}."
            )
        # The cell decides nothing about which pairs are found. A primitive's own sweep, grown by the reach, is
        # rasterised into the grid and so is every vertex's swept box, and func_is_canonical_cell accepts each
        # overlap exactly once, so the candidate set is the same at any cell size and only the cost moves. Two
        # costs move against each other: a bigger cell puts each box in fewer cells, and puts more vertices in
        # each cell. The product is least near the mesh's own scale, which is why the default is the median
        # contact edge rather than anything to do with the contact layer.
        #
        # Sizing it off the layer was the mistake this replaces. On the python head a 0.2 mm layer and a 0.2 mm
        # margin gave a 0.8 mm cell for triangles averaging 2.8 mm across and reaching 84 mm, so the mean
        # triangle was rasterised into 350 cells and the worst into 36288: one substep made 20.0 M cell visits
        # and 11.8 M bucket-entry reads to keep 560 pairs, which is 21000 entries read for every pair kept, and
        # about a gigabyte of scattered traffic. At the median edge of 3.1 mm the same substep needs about
        # 2.4 M of both. The floor is twice the reach, so that growing a box by the reach can never add more
        # than one cell a side; below that the grid costs cells without separating anything.
        self.cell = (
            max(2.0 * (self.max_thickness + self.margin_max), self.median_edge)
            if solver._contact_cell_size is None
            else solver._contact_cell_size
        )
        if not self.cell > 0.0:
            gs.raise_exception(f"VBDOptions.contact_cell_size must be above zero, got {self.cell}.")
        self.hash_buckets = 2 * self.n_cv
        self.hash_cap = solver._contact_cell_cap
        self.cell_n = qd.field(dtype=gs.qd_int, shape=(self.hash_buckets, solver._B))
        self.cell_v = qd.field(dtype=gs.qd_int, shape=(self.hash_buckets, self.hash_cap, solver._B))
        # the inclusive cell range a contact vertex sweeps over the substep, and its predicted end position: the
        # grid holds the vertex in every cell of that range, and the pair tests read both ends of the sweep
        self.cv_lo = qd.Vector.field(3, dtype=gs.qd_int, shape=(self.n_cv, solver._B))
        self.cv_hi = qd.Vector.field(3, dtype=gs.qd_int, shape=(self.n_cv, solver._B))
        # the cell each bucket slot was inserted under: two cells of one vertex's own sweep can hash to the
        # same bucket, and this is what a query tells them apart without scanning the bucket (see
        # `func_is_own_cell`)
        self.cell_c = qd.Vector.field(3, dtype=gs.qd_int, shape=(self.hash_buckets, self.hash_cap, solver._B))
        self.cv_pred = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_cv, solver._B))
        # A sweep that spans more cells than this is refused rather than searched: the rasterization is the
        # product of the three spans, so a body that travels hundreds of cells in one substep would cost more to
        # search than to simulate. Shorten the substep or widen the margin, which widens the cell with it.
        self.sweep_cell_cap = solver._contact_sweep_cell_cap

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
        # how far the solved position ends from the swept path the search covered; reported for inspection, no
        # longer what the rebuild decision reads (that is `d_budget` now, driven by the raw motion above)
        self.max_deviation = qd.field(dtype=gs.qd_float, shape=(2, solver._B))
        # the substep's conservative time of impact, and the smallest one since it was last cleared
        self.toi = qd.field(dtype=gs.qd_float, shape=solver._B)
        self.min_toi = qd.field(dtype=gs.qd_float, shape=solver._B)
        self.toi.fill(1.0)
        self.min_toi.fill(1.0)
        # the running safe bound D of Wang et al. 2022 Eq. 4, zero-initialized so the first substep always
        # rebuilds (see kernel_reset_contact), and how many rebuilds this environment has needed in total
        self.d_budget = qd.field(dtype=gs.qd_float, shape=solver._B)
        self.rebuild_count = qd.field(dtype=gs.qd_int, shape=solver._B)
        # this substep's rebuild decision, read by every loop of kernel_begin_contact after the first sets it
        self.rebuilding = qd.field(dtype=gs.qd_int, shape=solver._B)
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
        # wrench (world force, world torque about the link origin) the tissue applies to each collider link, and its
        # time integral since the last clear, accumulated every substep so a caller can balance momentum
        n_links = max(solver.sim.rigid_solver.n_links, 1)  # a tissue-only contact scene has no rigid link
        self.link_reaction = qd.Vector.field(6, dtype=qd.f64, shape=(n_links, solver._B))
        self.link_impulse = qd.Vector.field(6, dtype=qd.f64, shape=(n_links, solver._B))

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
    def _link_rest_pose(link):
        """The link's world pose as built, for measuring a world-space collider region against its geometry."""
        solver = link.entity.solver
        pos = tensor_to_array(solver.get_links_pos()).reshape(-1, solver.n_links, 3)[0][link.idx]
        quat = tensor_to_array(solver.get_links_quat()).reshape(-1, solver.n_links, 4)[0][link.idx]
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
        deviation = qd_to_torch(self.max_deviation, transpose=True)
        return ContactDiagnostics(
            qd_to_torch(self.n_pt),
            qd_to_torch(self.n_ee),
            motion[:, 0],
            motion[:, 1],
            qd_to_torch(self.errno),
            qd_to_torch(self.min_toi),
            deviation[:, 0],
            deviation[:, 1],
            qd_to_torch(self.rebuild_count),
        )

    def clear_toi(self):
        self.min_toi.fill(1.0)

    def reactions(self):
        """Wrench on each registered collider, shape (B, n_colliders, 6): world force then world torque about the
        collider's origin (the link origin, or the reference link origin of a prescribed entity), from the pair state
        at the end of the last substep. Rigid colliders come first, then prescribed entities, in declaration order."""
        return self._per_collider(qd_to_torch(self.link_reaction, transpose=True).to(gs.tc_float))

    def impulses(self):
        """Time integral of `reactions()` over every substep since the last `clear_impulses()`, same layout, in N s
        and N m s. The torque is transported to the collider origin at the time of each substep."""
        return self._per_collider(qd_to_torch(self.link_impulse, transpose=True).to(gs.tc_float))

    def clear_impulses(self):
        self.link_impulse.fill(0.0)

    def _per_collider(self, wrenches):
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
    solver: qd.template(),
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
            if not solver.env_failed[i_b]:
                i_l = contact.prescribed_link[i_p]
                contact.prescribed_start[i_p, i_b].pos = dyn_state.links.pos[i_l, i_b]
                contact.prescribed_start[i_p, i_b].quat = dyn_state.links.quat[i_l, i_b]
    for i_p, i_b in qd.ndrange(contact.n_prescribed, dyn_state.links.pos.shape[1]):
        if not solver.env_failed[i_b]:
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
def func_point_triangle_geometry(x, a, b, c):
    """Distance, unit normal and barycentric weights of a point against a triangle. The distance is signed over
    the face while the projection is interior, and unsigned once it falls on an edge or a corner, where the
    face has no side to be on.

    The normal is the direction of the offset between the two closest points, and the offset stops having one
    when they coincide. A penalty solve drives surfaces together, so a point landing exactly on a face is an
    ordinary state rather than a degenerate input, and dividing the zero offset by its zero length turned it
    into a non-finite normal: a head36 run with the continuous filter on ended at frame 9 that way. The
    triangle still has a normal there, and it is the direction the pair should push along. A triangle with no
    area has none, and that is left to read as non-finite, because a mesh that carries one is broken.
    """
    w = func_point_triangle_weights(x, a, b, c)
    rel = x - w[0] * a - w[1] * b - w[2] * c
    face = (b - a).cross(c - a).normalized()
    d = rel.norm()
    n = face
    if d > CONTACT_DEGENERATE_EPS:
        n = rel / d
    if w[0] > 0.0 and w[1] > 0.0 and w[2] > 0.0:
        n = face
        d = rel.dot(n)
    return d, n, w


@qd.func
def func_edge_edge_geometry(a, b, c, d):
    """Distance, unit normal from the second edge towards the first, and closest-point parameters of two edges.

    As above, the offset has no direction once the closest points coincide. What the pair still has is the
    common perpendicular of the two edges, signed to agree with the offset between their midpoints so that it
    keeps pointing the way the offset did. Two edges that touch *and* are parallel have neither, and that is
    left to read as non-finite.
    """
    s, t = func_segment_parameters(a, b, c, d)
    rel = a + s * (b - a) - c - t * (d - c)
    dist = rel.norm()
    n = rel / dist
    if dist <= CONTACT_DEGENERATE_EPS:
        perp = (b - a).cross(d - c)
        if perp.norm() > CONTACT_DEGENERATE_EPS:
            n = perp.normalized()
            if n.dot((a + b) - (c + d)) < 0.0:
                n = -n
    return dist, n, s, t


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
    d, n, w = func_point_triangle_geometry(x, a, b, c)
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
    dist, n, s, t = func_edge_edge_geometry(a, b, c, d)
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
        # A vertex the search gave no pair contributes an exactly zero force and an exactly zero block, so the
        # two Jacobian products below add nothing to either accumulator: skipping it changes no bit of the
        # result, it only declines to compute one. This loop runs on a single thread, once a free body, once a
        # sweep, and on the python head 286 of the 15224 rigid contact vertices carry a pair, so without the
        # test 98 percent of it builds a 3x6 Jacobian and multiplies through it to add zero.
        if contact.cv_slot_offset[cv + 1, i_b] > contact.cv_slot_offset[cv, i_b]:
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
                cv = contact.rv_cv[i_r]
                # zero contributes nothing here either, for the same reason as in func_contact_link_terms
                if contact.cv_slot_offset[cv + 1, i_b] > contact.cv_slot_offset[cv, i_b]:
                    force_v, hessian_v = func_contact_cv_terms(f, cv, i_b, solver, contact)
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
def func_pair_distance(x0, x1, x2, x3, kind: qd.template()):
    """Clamped closest-point distance of a pair: the point x0 against the triangle (x1, x2, x3), or the edge
    (x0, x1) against the edge (x2, x3). The same quantities the contact rules are written on."""
    distance = gs.qd_float(0.0)
    if qd.static(kind == POINT_TRIANGLE):
        w = func_point_triangle_weights(x0, x1, x2, x3)
        distance = (x0 - w[0] * x1 - w[1] * x2 - w[2] * x3).norm()
    else:
        s, t = func_segment_parameters(x0, x1, x2, x3)
        distance = (x0 + s * (x1 - x0) - x2 - t * (x3 - x2)).norm()
    return distance


@qd.func
def func_sweep_bound(p0, p1, p2, p3, kind: qd.template()):
    """Lipschitz constant of a pair's clamped distance over a linear sweep with these endpoint displacements.

    The distance does not depend on the pair's common translation, so the mean displacement is removed, and what
    remains bounds how fast the distance can change: |d(t2) - d(t1)| <= bound * |t2 - t1|. The additive CCD of
    `vbd_accd.py` advances on the same quantity.
    """
    mean = 0.25 * (p0 + p1 + p2 + p3)
    q0 = p0 - mean
    q1 = p1 - mean
    q2 = p2 - mean
    q3 = p3 - mean
    bound = gs.qd_float(0.0)
    if qd.static(kind == POINT_TRIANGLE):
        bound = q0.norm() + qd.max(q1.norm(), qd.max(q2.norm(), q3.norm()))
    else:
        bound = qd.max(q0.norm(), q1.norm()) + qd.max(q2.norm(), q3.norm())
    return bound


@qd.func
def func_swept_lower_bound(d0, d1, bound):
    """Lower bound of a pair's distance over the whole sweep, from its two endpoint distances and the Lipschitz
    constant between them.

    The distance is above both d0 - bound t and d1 - bound (1 - t), and the smallest value that pair of lines
    allows is their crossing when it falls inside the substep, and the nearer endpoint otherwise. A pair whose
    bound here is inside the layer is collected even though both of its endpoints are outside it, which is the
    pair the endpoint search used to miss.
    """
    lower = qd.min(d0, d1)
    if bound > qd.abs(d0 - d1):
        lower = 0.5 * (d0 + d1 - bound)
    return lower


@qd.func
def func_point_segment_distance(x, a, b):
    """Distance from a point to the clamped segment (a, b)."""
    ab = b - a
    denom = ab.dot(ab)
    s = gs.qd_float(0.0)
    if denom > 0.0:
        s = qd.max(0.0, qd.min(1.0, (x - a).dot(ab) / denom))
    return (x - a - s * ab).norm()


@qd.func
def func_sweep_cells(x_prev, x_pred, cell_size):
    """Inclusive cell range of a vertex's sweep: the grid holds it in every cell of this box."""
    return func_cell(qd.min(x_prev, x_pred), cell_size), func_cell(qd.max(x_prev, x_pred), cell_size)


@qd.func
def func_is_canonical_cell(cv, i_b, cell, lo, contact: qd.template()):
    """Whether `cell` is the one cell of a searched range through which this vertex is accepted.

    A swept vertex sits in every cell of its own range, so a search would otherwise find it once per shared cell
    and collect the same pair that many times. The componentwise largest of the two ranges' lower corners lies in
    both exactly when they overlap, so accepting only there yields each pair once and loses none. It also rejects
    a vertex that only shares the cell's hash bucket, since that vertex is not inside the searched cell.
    """
    canonical = qd.max(contact.cv_lo[cv, i_b], lo)
    return (cell == canonical).all() and (cell <= contact.cv_hi[cv, i_b]).all()


@qd.func
def func_is_own_cell(h, slot, i_b, cell, contact: qd.template()):
    """Whether this bucket slot was inserted under the cell currently being searched.

    Two cells of one swept vertex can hash to the same bucket, which puts the vertex in it twice, and both
    entries then pass the canonical-cell test, since that test reads the searched cell and not the entry. But
    the two entries were inserted under different cells -- that is the only way they could collide rather than
    be the same insert -- so at most one of them can carry the cell being searched. Accepting only that one is
    what used to take `func_is_first_entry`'s O(slot) scan of the bucket; this is O(1).
    """
    return (contact.cell_c[h, slot, i_b] == cell).all()


@qd.func
def func_sweep_overlaps(cv, i_b, lo, hi, contact: qd.template()):
    """Whether a vertex's swept cell range meets a searched cell range."""
    return (contact.cv_hi[cv, i_b] >= lo).all() and (contact.cv_lo[cv, i_b] <= hi).all()


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
    """Rigid vertex positions from the link poses, then, only for an environment whose safe bound is spent, the
    hash grid of the substep's sweeps, the candidate pairs of the substep and the per-vertex lists of the pairs
    they take part in. An environment that still has proximity budget left keeps last build's candidate set and
    pays only for the position refresh (Wang et al. 2022, "Fast GPU-Based Two-Way Continuous Collision
    Handling", Sec. 3.1): the search is not what changes every substep, `contact.d_budget` is."""
    # A failed environment writes nothing here. Its positions, sweeps and pair counts are the evidence of the
    # substep that failed, and the substeps the batch still runs for its other environments would otherwise
    # overwrite them: the vertex a report names is read from these buffers after the raise.
    for i_b in range(solver._B):
        if not solver.env_failed[i_b]:
            contact.rebuilding[i_b] = 0
            if contact.d_budget[i_b] < contact.margin:
                contact.rebuilding[i_b] = 1
                contact.rebuild_count[i_b] += 1
    for i_r, i_b in qd.ndrange(contact.n_rv, solver._B):
        if not solver.env_failed[i_b]:
            i_l = contact.rv_link[i_r]
            pos = dyn_state.links.pos[i_l, i_b]
            quat = dyn_state.links.quat[i_l, i_b]
            if qd.static(solver.has_rigid_attachment):
                # A free body's pose for this substep is the one the attachment predicted from its velocity; the
                # rigid solver's own table still holds the pose of the previous substep, because the solve commits
                # back to it only at the end. Searching from that stale pose gives a free body a sweep of zero
                # length and hands the whole of its travel to the validity guard, which is what refused every
                # large step of the python head. A fixed or prescribed link is not in this table and keeps its
                # own pose.
                if solver.rigid_attachment.free_slot[i_l] >= 0:
                    pos = solver.rigid_attachment.link_pose[i_l, i_b].pos
                    quat = solver.rigid_attachment.link_pose[i_l, i_b].quat
            contact.rv_pos_prev[i_r, i_b] = contact.rv_pos[i_r, i_b]
            contact.rv_pos[i_r, i_b] = gu.qd_transform_by_trans_quat(contact.rv_local[i_r], pos, quat)
    for h, i_b in qd.ndrange(contact.hash_buckets, solver._B):
        if not solver.env_failed[i_b] and contact.rebuilding[i_b]:
            contact.cell_n[h, i_b] = 0
    for cv, i_b in qd.ndrange(contact.n_cv, solver._B):
        if not solver.env_failed[i_b]:
            # the prediction is read every substep (kernel_end_contact's deviation and d_budget update need it
            # regardless of whether the grid is rebuilt this substep); the sweep box and the grid insert are the
            # expensive part, and only run when the safe bound ran out
            pred = func_cv_pos(f, cv, i_b, solver, contact)
            prev = func_cv_pos_prev(f, cv, i_b, solver, contact)
            contact.cv_pred[cv, i_b] = pred
            if contact.rebuilding[i_b]:
                contact.cv_slot_n[cv, i_b] = 0
                lo, hi = func_sweep_cells(prev, pred, contact.cell)
                contact.cv_lo[cv, i_b] = lo
                contact.cv_hi[cv, i_b] = hi
                span = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1)
                if span > contact.sweep_cell_cap:
                    qd.atomic_or(contact.errno[i_b], ErrorCode.OVERFLOW_VBD_CONTACT_SWEEP)
                else:
                    for ci in range(lo[0], hi[0] + 1):
                        for cj in range(lo[1], hi[1] + 1):
                            for ck in range(lo[2], hi[2] + 1):
                                cell = qd.Vector([ci, cj, ck], dt=gs.qd_int)
                                h = func_cell_hash(cell, contact.hash_buckets)
                                slot = qd.atomic_add(contact.cell_n[h, i_b], 1)
                                if slot < contact.hash_cap:
                                    contact.cell_v[h, slot, i_b] = cv
                                    contact.cell_c[h, slot, i_b] = cell
                                else:
                                    qd.atomic_or(contact.errno[i_b], ErrorCode.OVERFLOW_VBD_CONTACT_CELL)
    for i_b in range(solver._B):
        if not solver.env_failed[i_b] and contact.rebuilding[i_b]:
            contact.n_pt[i_b] = 0
            contact.n_ee[i_b] = 0
    # the generous bound (D_max): a rebuild searches this far so the resulting candidate set stays a safe
    # superset for several future substeps, not just the one being built. margin (D_min) is what a candidate
    # still needs to clear to matter for the response (func_pt_forces/func_ee_forces): a pair kept only because
    # it is inside margin_max but outside margin contributes zero force until it actually closes that gap.
    reach = contact.max_thickness + contact.margin_max
    for i_t, i_b in qd.ndrange(contact.n_triangles, solver._B):
        if not solver.env_failed[i_b] and contact.rebuilding[i_b]:
            tri = contact.tri_cv[i_t]
            a = func_cv_pos(f, tri[0], i_b, solver, contact)
            b = func_cv_pos(f, tri[1], i_b, solver, contact)
            c = func_cv_pos(f, tri[2], i_b, solver, contact)
            a0 = func_cv_pos_prev(f, tri[0], i_b, solver, contact)
            b0 = func_cv_pos_prev(f, tri[1], i_b, solver, contact)
            c0 = func_cv_pos_prev(f, tri[2], i_b, solver, contact)
            low = qd.min(qd.min(qd.min(a, b), c), qd.min(qd.min(a0, b0), c0))
            high = qd.max(qd.max(qd.max(a, b), c), qd.max(qd.max(a0, b0), c0))
            lo = func_cell(low - reach, contact.cell)
            hi = func_cell(high + reach, contact.cell)
            for ci in range(lo[0], hi[0] + 1):
                for cj in range(lo[1], hi[1] + 1):
                    for ck in range(lo[2], hi[2] + 1):
                        cell = qd.Vector([ci, cj, ck], dt=gs.qd_int)
                        h = func_cell_hash(cell, contact.hash_buckets)
                        for slot in range(qd.min(contact.cell_n[h, i_b], contact.hash_cap)):
                            cv = contact.cell_v[h, slot, i_b]
                            is_new = func_is_canonical_cell(cv, i_b, cell, lo, contact) and func_is_own_cell(
                                h, slot, i_b, cell, contact
                            )
                            if is_new and func_may_collide(cv, tri[0], contact):
                                is_adjacent = False
                                for j in qd.static(range(3)):
                                    if func_shares_tetrahedron(cv, tri[j], solver, contact):
                                        is_adjacent = True
                                if not is_adjacent:
                                    x = func_cv_pos(f, cv, i_b, solver, contact)
                                    x0 = func_cv_pos_prev(f, cv, i_b, solver, contact)
                                    d = func_swept_lower_bound(
                                        func_pair_distance(x0, a0, b0, c0, POINT_TRIANGLE),
                                        func_pair_distance(x, a, b, c, POINT_TRIANGLE),
                                        func_sweep_bound(x - x0, a - a0, b - b0, c - c0, POINT_TRIANGLE),
                                    )
                                    h_rule = contact.rule_thickness[
                                        contact.cv_info[cv].group, contact.cv_info[tri[0]].group
                                    ]
                                    if d < h_rule + contact.margin_max:
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
        if not solver.env_failed[i_b] and contact.rebuilding[i_b]:
            ea = contact.edge_cv[i_e]
            a = func_cv_pos(f, ea[0], i_b, solver, contact)
            b = func_cv_pos(f, ea[1], i_b, solver, contact)
            a0 = func_cv_pos_prev(f, ea[0], i_b, solver, contact)
            b0 = func_cv_pos_prev(f, ea[1], i_b, solver, contact)
            lo = func_cell(qd.min(qd.min(a, b), qd.min(a0, b0)) - reach, contact.cell)
            hi = func_cell(qd.max(qd.max(a, b), qd.max(a0, b0)) + reach, contact.cell)
            for ci in range(lo[0], hi[0] + 1):
                for cj in range(lo[1], hi[1] + 1):
                    for ck in range(lo[2], hi[2] + 1):
                        cell = qd.Vector([ci, cj, ck], dt=gs.qd_int)
                        h = func_cell_hash(cell, contact.hash_buckets)
                        for slot in range(qd.min(contact.cell_n[h, i_b], contact.hash_cap)):
                            cv = contact.cell_v[h, slot, i_b]
                            is_new = func_is_canonical_cell(cv, i_b, cell, lo, contact) and func_is_own_cell(
                                h, slot, i_b, cell, contact
                            )
                            if is_new:
                                for c_e in range(contact.cv_edge_offset[cv], contact.cv_edge_offset[cv + 1]):
                                    j_e = contact.cv_edge[c_e]
                                    eb = contact.edge_cv[j_e]
                                    # an edge is reached through both endpoints: accept it through its first one, or
                                    # through the second when the first sweeps outside the searched cells
                                    is_accepted = cv == eb[0] or not func_sweep_overlaps(eb[0], i_b, lo, hi, contact)
                                    if j_e > i_e and is_accepted and func_may_collide(ea[0], eb[0], contact):
                                        is_adjacent = False
                                        for j in qd.static(range(2)):
                                            for l in qd.static(range(2)):
                                                if func_shares_tetrahedron(ea[j], eb[l], solver, contact):
                                                    is_adjacent = True
                                        if not is_adjacent:
                                            c = func_cv_pos(f, eb[0], i_b, solver, contact)
                                            d = func_cv_pos(f, eb[1], i_b, solver, contact)
                                            c0 = func_cv_pos_prev(f, eb[0], i_b, solver, contact)
                                            d0 = func_cv_pos_prev(f, eb[1], i_b, solver, contact)
                                            dist = func_swept_lower_bound(
                                                func_pair_distance(a0, b0, c0, d0, EDGE_EDGE),
                                                func_pair_distance(a, b, c, d, EDGE_EDGE),
                                                func_sweep_bound(a - a0, b - b0, c - c0, d - d0, EDGE_EDGE),
                                            )
                                            h_rule = contact.rule_thickness[
                                                contact.cv_info[ea[0]].group, contact.cv_info[eb[0]].group
                                            ]
                                            if dist < h_rule + contact.margin_max:
                                                i_p = qd.atomic_add(contact.n_ee[i_b], 1)
                                                if i_p < contact.pair_cap:
                                                    contact.ee_pairs[i_p, i_b].a = i_e
                                                    contact.ee_pairs[i_p, i_b].b = j_e
                                                    contact.ee_pairs[i_p, i_b].lam = 0.0
                                                    contact.ee_pairs[i_p, i_b].k = contact.rule_stiffness[
                                                        contact.cv_info[ea[0]].group, contact.cv_info[eb[0]].group
                                                    ]
                                                else:
                                                    qd.atomic_or(
                                                        contact.errno[i_b], ErrorCode.OVERFLOW_VBD_CONTACT_PAIRS
                                                    )
    # The dual state resets every substep, rebuild or not: lam and k are an Uzawa iterate that is only meant to
    # converge across one substep's sweeps (kernel_reset_contact does the same for lam at the start of a whole
    # step). A rebuild already zeroes them at insertion, but a reused pair keeps last substep's k, which ramps
    # toward `_contact_k_max_ratio` in `func_contact_dual_update` and would otherwise carry that ramp forward
    # substep after substep for as long as the set is reused, well past what the pair's own current violation
    # asks for.
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if not solver.env_failed[i_b]:
            if i_p < qd.min(contact.n_pt[i_b], contact.pair_cap):
                contact.pt_pairs[i_p, i_b].lam = 0.0
                tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
                contact.pt_pairs[i_p, i_b].k = contact.rule_stiffness[
                    contact.cv_info[contact.pt_pairs[i_p, i_b].a].group, contact.cv_info[tri[0]].group
                ]
            if i_p < qd.min(contact.n_ee[i_b], contact.pair_cap):
                contact.ee_pairs[i_p, i_b].lam = 0.0
                ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
                eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
                contact.ee_pairs[i_p, i_b].k = contact.rule_stiffness[
                    contact.cv_info[ea[0]].group, contact.cv_info[eb[0]].group
                ]
    # count the pairs of every contact vertex, prefix-sum the counts, then fill the flat slot list -- all as
    # stale as the pairs themselves, so none of it is worth redoing on a substep that only reuses them
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if not solver.env_failed[i_b] and contact.rebuilding[i_b]:
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
        if not solver.env_failed[i_b] and contact.rebuilding[i_b]:
            run = 0
            for cv in range(contact.n_cv):
                contact.cv_slot_offset[cv, i_b] = run
                run += contact.cv_slot_n[cv, i_b]
                contact.cv_slot_n[cv, i_b] = 0
            contact.cv_slot_offset[contact.n_cv, i_b] = run
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if not solver.env_failed[i_b] and contact.rebuilding[i_b]:
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
    # A pair takes its index from an atomic, so a vertex's list is filled in whatever order the GPU finished
    # those threads in, and func_contact_cv_terms sums the list in that order. Float addition is not
    # associative, so from the first substep at which one vertex carries two simultaneously active contacts,
    # two runs of one binary round differently: on the python head that is about substep 122 and a couple of
    # last bits, which the scene then multiplies by roughly ten a frame until two identical runs stand 49 mm
    # apart at 100 ms. Sorting each list on the mesh's own indices takes the race out of the sum. The lists are
    # short and mostly empty, and finding the pairs they name was the expensive part.
    for cv, i_b in qd.ndrange(contact.n_cv, solver._B):
        if not solver.env_failed[i_b] and contact.rebuilding[i_b]:
            lo = contact.cv_slot_offset[cv, i_b]
            hi = contact.cv_slot_offset[cv + 1, i_b]
            for i in range(lo + 1, hi):
                code = contact.cv_slot[i, i_b]
                j = i - 1
                while j >= lo and func_slot_precedes(code, contact.cv_slot[j, i_b], i_b, contact):
                    contact.cv_slot[j + 1, i_b] = contact.cv_slot[j, i_b]
                    j -= 1
                contact.cv_slot[j + 1, i_b] = code


@qd.func
def func_register_slot(cv, code, i_b, contact: qd.template()):
    slot = contact.cv_slot_offset[cv, i_b] + qd.atomic_add(contact.cv_slot_n[cv, i_b], 1)
    contact.cv_slot[slot, i_b] = code


@qd.func
def func_slot_identity(code, i_b, contact: qd.template()):
    """The pair a slot belongs to, named by the mesh rather than by the order the search collected it in."""
    role = code % 8
    i_p = code // 8
    kind = 0
    a = 0
    b = 0
    if i_p >= contact.pair_cap:
        kind = 1
        a = contact.ee_pairs[i_p - contact.pair_cap, i_b].a
        b = contact.ee_pairs[i_p - contact.pair_cap, i_b].b
    else:
        a = contact.pt_pairs[i_p, i_b].a
        b = contact.pt_pairs[i_p, i_b].b
    return kind, a, b, role


@qd.func
def func_slot_precedes(code_x, code_y, i_b, contact: qd.template()):
    """Whether one slot sorts before another: pair kind, then the two topology indices, then the role."""
    kind_x, a_x, b_x, role_x = func_slot_identity(code_x, i_b, contact)
    kind_y, a_y, b_y, role_y = func_slot_identity(code_y, i_b, contact)
    precedes = role_x < role_y
    if kind_x != kind_y:
        precedes = kind_x < kind_y
    elif a_x != a_y:
        precedes = a_x < a_y
    elif b_x != b_y:
        precedes = b_x < b_y
    return precedes


@qd.func
def func_contact_dual_update(f, w, solver: qd.template(), contact: qd.template()):
    """Relaxed multiplier update lam <- min(lam + w k C, 0) and the stiffness ramp of every candidate pair (Giles et
    al. 2025 Eq. 11 to 13 with the inequality clamp), after a primal sweep."""
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if i_p < qd.min(contact.n_pt[i_b], contact.pair_cap) and not solver.env_failed[i_b]:
            d, n, weights, h = func_pt_geometry(f, i_p, i_b, solver, contact)
            k = contact.pt_pairs[i_p, i_b].k
            contact.pt_pairs[i_p, i_b].lam = qd.min(contact.pt_pairs[i_p, i_b].lam + w * k * (d - h), 0.0)
            k0 = contact.rule_stiffness[
                contact.cv_info[contact.pt_pairs[i_p, i_b].a].group,
                contact.cv_info[contact.tri_cv[contact.pt_pairs[i_p, i_b].b][0]].group,
            ]
            contact.pt_pairs[i_p, i_b].k = qd.min(
                k + k0 / solver._constraint_tol * qd.max(h - d, 0.0), solver._contact_k_max_ratio * k0
            )
        if i_p < qd.min(contact.n_ee[i_b], contact.pair_cap) and not solver.env_failed[i_b]:
            d, n, s, t, h = func_ee_geometry(f, i_p, i_b, solver, contact)
            k = contact.ee_pairs[i_p, i_b].k
            contact.ee_pairs[i_p, i_b].lam = qd.min(contact.ee_pairs[i_p, i_b].lam + w * k * (d - h), 0.0)
            k0 = contact.rule_stiffness[
                contact.cv_info[contact.edge_cv[contact.ee_pairs[i_p, i_b].a][0]].group,
                contact.cv_info[contact.edge_cv[contact.ee_pairs[i_p, i_b].b][0]].group,
            ]
            contact.ee_pairs[i_p, i_b].k = qd.min(
                k + k0 / solver._constraint_tol * qd.max(h - d, 0.0), solver._contact_k_max_ratio * k0
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


@qd.kernel
def kernel_end_contact(f: int, substep_global: int, solver: qd.template(), contact: qd.template(), dyn_state: DynState):
    """Wrenches on the collider links from the final pair state, the substep's validity checks (finite geometry,
    no pair deeper than its thickness), the failure latch of any environment whose checks failed, and the safe
    bound update of Wang et al. 2022 Eq. 4 that decides whether the *next* substep reuses this one's candidate
    set or rebuilds it. The two crossing tests are skipped under `VBDOptions.contact_ccd`, which prevents a
    crossing instead of reporting one: a sign flip of two edges whose closest parameters changed feature, or a
    point at a face boundary read as behind it, would then be the only thing left for them to find."""
    for i_l, i_b in qd.ndrange(contact.link_reaction.shape[0], solver._B):
        if not solver.env_failed[i_b]:
            contact.link_reaction[i_l, i_b] = qd.Vector.zero(qd.f64, 6)
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if i_p < qd.min(contact.n_pt[i_b], contact.pair_cap) and not solver.env_failed[i_b]:
            d, n, w, h = func_pt_geometry(f, i_p, i_b, solver, contact)
            if not (d == d):
                qd.atomic_or(contact.errno[i_b], ErrorCode.INVALID_VBD_CONTACT_NAN)
            # signed over the face: a point this far behind the surface is past what the penalty can recover.
            # With the continuous filter on there is nothing to detect: no pair ever reaches the gap, so the two
            # crossing tests here would only report their own false positives.
            if qd.static(not solver._contact_ccd):
                depth = contact.crossing_depth
                if qd.static(contact.crossing_depth_per_pair):
                    depth = h
                if d < -depth:
                    qd.atomic_or(contact.errno[i_b], ErrorCode.VBD_CONTACT_CROSSING)
            y, n, w, scale, slide = func_pt_forces(f, i_p, i_b, solver, contact)
            if y < 0.0:
                force = -(y * n + scale * slide)
                tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
                func_accumulate_reaction(f, contact.pt_pairs[i_p, i_b].a, force, i_b, solver, contact, dyn_state)
                for j in qd.static(range(3)):
                    func_accumulate_reaction(f, tri[j], -w[j] * force, i_b, solver, contact, dyn_state)
        if i_p < qd.min(contact.n_ee[i_b], contact.pair_cap) and not solver.env_failed[i_b]:
            y, n, s, t, scale, slide = func_ee_forces(f, i_p, i_b, solver, contact)
            if not (n[0] == n[0]):
                qd.atomic_or(contact.errno[i_b], ErrorCode.INVALID_VBD_CONTACT_NAN)
            ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
            eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
            # There is no edge-edge crossing test. A crossing is measured against a surface and a pair of edges
            # does not have one: the sign of (b - a) x (d - c) . (a - c) says which side of their common
            # perpendicular the edges are on, which is not which side of a body they are on. Two convex
            # surfaces resting against each other slide their boundary edges across each other continuously,
            # and every one of those is a sign change with no penetration anywhere near it. The point-triangle
            # test above does have a surface, so its distance is signed over the face and `d < -depth` reads as
            # "this far behind", which is the question worth asking.
            #
            # Measured on the python head at the substep that used to refuse the run: the edge-edge test
            # reported nine crossings, every one between angular.R and coronoid.R and at 22 to 89 percent of
            # the depth, while the point-triangle test found no point behind a face anywhere in the scene, the
            # most negative signed distance over its 43 interior projections being +0.154 mm. Nothing was
            # penetrating. The test had already been narrowed twice, once to require the pair to be inside the
            # layer and once to require the sign change to clear the noise of nearly parallel edges, and it
            # still could not separate articulation from interpenetration, because the quantity it reads does
            # not carry that difference. `VBDOptions.contact_ccd` prevents a crossing rather than reporting
            # one, and the point-triangle test reports the ones that happen.
            if y < 0.0:
                force = -(y * n + scale * slide)
                func_accumulate_reaction(f, ea[0], (1.0 - s) * force, i_b, solver, contact, dyn_state)
                func_accumulate_reaction(f, ea[1], s * force, i_b, solver, contact, dyn_state)
                func_accumulate_reaction(f, eb[0], -(1.0 - t) * force, i_b, solver, contact, dyn_state)
                func_accumulate_reaction(f, eb[1], -t * force, i_b, solver, contact, dyn_state)
    for kind, i_b in qd.ndrange(2, solver._B):
        if not solver.env_failed[i_b]:
            contact.max_motion[kind, i_b] = 0.0
            contact.max_deviation[kind, i_b] = 0.0
    for cv, i_b in qd.ndrange(contact.n_cv, solver._B):
        if not solver.env_failed[i_b]:
            # The search covered the segment from the start of the substep to the prediction. The solved position
            # is the end of the path actually taken; this is its distance from that segment, reported for
            # inspection (it is not what drives a rebuild -- see the d_budget update below, which reads the raw
            # motion instead). Rescaling by a time of impact lands on the segment itself.
            pos = func_cv_pos(f, cv, i_b, solver, contact)
            prev = func_cv_pos_prev(f, cv, i_b, solver, contact)
            deviation = func_point_segment_distance(pos, prev, contact.cv_pred[cv, i_b])
            qd.atomic_max(contact.max_motion[contact.cv_info[cv].kind, i_b], (pos - prev).norm())
            qd.atomic_max(contact.max_deviation[contact.cv_info[cv].kind, i_b], deviation)
    for i_b in range(solver._B):
        if not solver.env_failed[i_b]:
            # Wang et al. 2022 Eq. 4: the safe bound shrinks by twice the largest displacement since it was
            # last spent, because a pair the last build could have missed needs both of its vertices to have
            # closed half the remaining gap. Once it is gone, next substep's kernel_begin_contact rebuilds with
            # a fresh D_max and this env's bound starts over. The bit is refreshed, not accumulated: it reports
            # this substep's state, not whether one was ever spent before.
            contact.d_budget[i_b] -= 2.0 * qd.max(contact.max_motion[0, i_b], contact.max_motion[1, i_b])
            contact.errno[i_b] &= _FATAL_ERRNO_MASK
            if contact.d_budget[i_b] < contact.margin:
                contact.errno[i_b] |= ErrorCode.VBD_CONTACT_MOTION_BOUND
    for i_b in range(solver._B):
        if (contact.errno[i_b] & _FATAL_ERRNO_MASK) != 0 and not solver.env_failed[i_b]:
            solver.env_failed[i_b] = 1
            solver.failed_substep[i_b] = substep_global
    for i_l, i_b in qd.ndrange(contact.link_impulse.shape[0], solver._B):
        if not solver.env_failed[i_b]:
            contact.link_impulse[i_l, i_b] += contact.link_reaction[i_l, i_b] * solver._substep_dt


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
        contact.toi[envs_idx[i_b_]] = 1.0
        contact.min_toi[envs_idx[i_b_]] = 1.0
        # zero is always below margin (margin > 0 is enforced at construction), so the next kernel_begin_contact
        # rebuilds unconditionally: a reset carries no proximity information forward.
        contact.d_budget[envs_idx[i_b_]] = 0.0
        contact.rebuild_count[envs_idx[i_b_]] = 0
    for i_p, i_b_ in qd.ndrange(contact.n_prescribed, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = contact.prescribed_link[i_p]
        contact.prescribed_start[i_p, i_b].pos = dyn_state.links.pos[i_l, i_b]
        contact.prescribed_start[i_p, i_b].quat = dyn_state.links.quat[i_l, i_b]
        contact.prescribed_target[i_p, i_b].pos = dyn_state.links.pos[i_l, i_b]
        contact.prescribed_target[i_p, i_b].quat = dyn_state.links.quat[i_l, i_b]


@qd.kernel
def kernel_set_prescribed_state(
    envs_idx: qd.types.ndarray(),
    start_pos: qd.types.ndarray(),
    start_quat: qd.types.ndarray(),
    target_pos: qd.types.ndarray(),
    target_quat: qd.types.ndarray(),
    contact: qd.template(),
):
    """Restore the prescribed-motion phase of the selected environments from a snapshot."""
    for i_p, i_b_ in qd.ndrange(contact.n_prescribed, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            contact.prescribed_start[i_p, i_b].pos[j] = start_pos[i_b, i_p, j]
            contact.prescribed_target[i_p, i_b].pos[j] = target_pos[i_b, i_p, j]
        for j in qd.static(range(4)):
            contact.prescribed_start[i_p, i_b].quat[j] = start_quat[i_b, i_p, j]
            contact.prescribed_target[i_p, i_b].quat[j] = target_quat[i_b, i_p, j]
