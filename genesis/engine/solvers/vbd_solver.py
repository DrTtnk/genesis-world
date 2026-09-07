"""Vertex Block Descent solver for stable neo-Hookean tetrahedral solids.

Chen, Liu, Yang, Yuksel, "Vertex Block Descent", SIGGRAPH 2024, Eq. 7-9: implicit Euler solved
by block coordinate descent, one vertex at a time with all others fixed. Vertices are graph-colored
at build (no two vertices of a color share a tetrahedron), so every color is one race-free parallel
sweep and the whole pass is an exact Gauss-Seidel iteration (Fratarcangeli et al. 2016, Vivace).

For this energy the local problem is exactly quadratic in the vertex position: `F` and `J` are
affine in one vertex, so `Psi = mu/2 (I_C - 3) + lam'/2 (J - alpha)^2` is quadratic and the
per-vertex Hessian is constant and positive definite for every `F`, including inverted ones:

    H_i = m_i/h^2 I + sum_tets V (mu |w_i|^2 I + lam' q q^T),   q = cof(F) w_i,
    f_i = -m_i/h^2 (x_i - y_i) - sum_tets V P(F) w_i,           P = mu F + lam' (J - alpha) cof(F),

with `w_i` the row of the inverse rest-shape matrix that belongs to vertex `i` (vertex 0 gets minus
the sum). The step `H_i^-1 f_i` is the exact local minimizer, so the incremental potential never
increases and no line search is needed.

Muscle actuation changes the rest shape, not the energy: a tet with fiber `m` and actuation `a`
uses `B_a = B A`, `A = (1/s) m m^T + sqrt(s) (I - m m^T)`, `s = 1 - a gain`, `det A = 1`.
"""

import numpy as np
import networkx as nx
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.entities.vbd_entity import VBDEntity
from genesis.engine.states.solvers import VBDSolverState

from .base_solver import Solver


@qd.data_oriented
class VBDSolver(Solver):
    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        self._n_iterations = options.n_iterations
        self._acc = qd.f64 if options.accumulate_f64 else gs.qd_float
        self._floor_height = options.floor_height
        self._contact_stiffness = options.contact_stiffness
        self._friction_eps_v = options.friction_eps_v
        self._residual_tol = options.residual_tol
        self._damping = options.damping
        self._constraint_tol = options.constraint_tol
        self._constraint_k_max_ratio = options.constraint_k_max_ratio
        self._constraint_dual_relaxation = options.constraint_dual_relaxation
        self._angle_tol = options.angle_tol
        self._max_sweeps = options.max_sweeps
        self._max_dual_steps = options.max_dual_steps
        self._violation_tol = options.violation_tol
        self._grad_converge = options.grad_converge

    # ------------------------------------------------------------------------------------
    # --------------------------------- initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def init_vertex_fields(self):
        struct_vert_info = qd.types.struct(
            mass=gs.qd_float,
            tangent=gs.qd_vec3,  # friction frame: forward direction on the floor plane
            mu_forward=gs.qd_float,
            mu_backward=gs.qd_float,
            mu_lateral=gs.qd_float,
        )
        struct_vert_state = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3)
        self.verts_info = struct_vert_info.field(shape=(self._n_vertices,), layout=qd.Layout.SOA)
        # Frames: [f] is the state at the start of substep f, [f+1] the state after it. The buffer holds one step's
        # worth of substeps when requires_grad (the adjoint walks them backwards), a sliding pair otherwise.
        self.verts = struct_vert_state.field(
            shape=(self._sim.substeps_local + 1, self._n_vertices, self._B), layout=qd.Layout.SOA
        )
        self.residual = qd.field(dtype=qd.f64, shape=())
        # Adjoint state, one frame per position frame: dL/dx and dL/dv accumulated by the backward pass.
        struct_adj = qd.types.struct(pos=qd.types.vector(3, qd.f64), vel=qd.types.vector(3, qd.f64))
        self.adj = struct_adj.field(shape=(self._sim.substeps_local + 1, self._n_vertices, self._B), layout=qd.Layout.SOA)
        self.z = qd.Vector.field(3, dtype=qd.f64, shape=(self._n_vertices, self._B))  # adjoint of the stationarity condition
        self.xb = qd.Vector.field(3, dtype=qd.f64, shape=(self._n_vertices, self._B))  # running position adjoint of the reverse sweep
        self.yb = qd.Vector.field(3, dtype=qd.f64, shape=(self._n_vertices, self._B))  # adjoint of the predictor y
        self.gbar = qd.Vector.field(3, dtype=qd.f64, shape=(self._n_vertices, self._B))  # its right-hand side
        self.adj_residual = qd.field(dtype=qd.f64, shape=())
        # Replay buffer for the solver-level adjoint: the update applied to each vertex at each sweep of each
        # substep. The reverse pass walks it backwards, subtracting each update to recover the state the forward
        # linearised at, so the block itself is recomputed rather than stored (24 bytes a vertex a sweep, not 72).
        self._record_sweeps = self._sim.requires_grad
        self.sweep_dx = qd.Vector.field(
            3, dtype=qd.f64,
            shape=(self._sim.substeps_local, self._n_iterations, self._n_vertices, self._B) if self._record_sweeps else (1, 1, 1, 1),
        )

    def init_element_fields(self):
        struct_elem_info = qd.types.struct(
            v=gs.qd_ivec4,
            vol_rest=gs.qd_float,
            B_rest=gs.qd_mat3,  # inverse rest-shape matrix Dm^-1
            mu=gs.qd_float,
            lam=gs.qd_float,  # lam' = lam + mu of the stable neo-Hookean model
            gain=gs.qd_float,
            fiber=gs.qd_vec3,
            group=gs.qd_int,  # muscle group, -1 for passive
            k_fiber=gs.qd_float,  # fibre reinforcement stiffness (Pa) along `fiber` on the unactuated F; 0 disables
        )
        self.elems_info = struct_elem_info.field(shape=(self._n_elements,), layout=qd.Layout.SOA)
        self.muscle_actu = qd.field(dtype=gs.qd_float, shape=(max(self._n_muscle_groups, 1), self._B))
        # Analytic bolus: a sphere with prescribed centre, radius and velocity per env, radius <= 0 disables it.
        struct_bolus = qd.types.struct(
            center=gs.qd_vec3,
            radius=gs.qd_float,
            vel=gs.qd_vec3,
            friction=gs.qd_float,
            axis=gs.qd_vec3,  # unit axis of the capsule segment
            half_length=gs.qd_float,  # 0 makes it a sphere
        )
        self.bolus = struct_bolus.field(shape=(self._B,))
        self.muscle_actu_adj = qd.field(dtype=qd.f64, shape=(max(self._n_muscle_groups, 1), self._B))
        self.energy = qd.field(dtype=qd.f64, shape=(self._B,))

    def init_constraint_fields(self):
        # Hard distance constraints |x_a - x_b| = rest between two vertices, augmented Lagrangian (Giles, Diaz,
        # Yuksel 2025): energy k/2 C^2 + lam C, dual update lam += k C after every sweep, stiffness ramp, and a warm
        # start once per step. lam and k live per env; the constraint list per vertex is a CSR like the tets.
        # lo == hi is an equality; otherwise the distance is bounded to [lo, hi] with one clamped multiplier per side
        # (Giles et al. 2025 Eq. 13: lam_hi >= 0 pushes the distance down, lam_lo <= 0 pushes it up)
        # `scale` is the pair's rest length: it turns a violation in metres into a strain, so one tolerance fits a
        # 1 mm rib and a 10 cm spine segment
        struct_cons_info = qd.types.struct(v=gs.qd_ivec2, lo=gs.qd_float, hi=gs.qd_float, scale=gs.qd_float)
        struct_cons_state = qd.types.struct(lam_hi=gs.qd_float, lam_lo=gs.qd_float, k=gs.qd_float)
        n = max(self._n_constraints, 1)
        self.cons_info = struct_cons_info.field(shape=(n,), layout=qd.Layout.SOA)
        self.cons = struct_cons_state.field(shape=(n, self._B), layout=qd.Layout.SOA)
        self.cons_error = qd.field(dtype=qd.f64, shape=())  # absolute: metres, or cosine for an angle
        self.cons_error_rel = qd.field(dtype=qd.f64, shape=())  # relative: strain, or radians for an angle
        # Angle constraints: the cosine between u = x_a - x_b and v = x_c - x_d kept in [lo, hi] (cos of the angle
        # bounds), same augmented Lagrangian with clamped multipliers. Joint limits on rigid vertebra frames.
        # `sin_ref` is the sine of the rest angle: d cos / d theta = -sin, so a cosine violation over it is an angle
        struct_acons_info = qd.types.struct(v=gs.qd_ivec4, lo=gs.qd_float, hi=gs.qd_float, k0=gs.qd_float, sin_ref=gs.qd_float)
        struct_acons_state = qd.types.struct(lam_hi=gs.qd_float, lam_lo=gs.qd_float, k=gs.qd_float)
        na = max(self._n_angle_constraints, 1)
        self.acons_info = struct_acons_info.field(shape=(na,), layout=qd.Layout.SOA)
        self.acons = struct_acons_state.field(shape=(na, self._B), layout=qd.Layout.SOA)
        # Per-substep record of each constraint's total multiplier and stiffness at the end of the solve (frame f+1):
        # the adjoint differentiates the KKT system of substep f and needs the active set and multipliers of that
        # substep, not the live ones. zeta is the dual adjoint (one per constraint), solved with the same augmented
        # Lagrangian machinery as the forward.
        # k_eff = k times the number of unclamped sides (an equality always counts one): zero means inactive, and
        # d mult / dC = k_eff, which is what the adjoint's augmentation and active set need (the multiplier's value
        # alone is not the active set: an unloaded equality has mult == 0 and is still a constraint)
        struct_cons_hist = qd.types.struct(mult=qd.f64, k_eff=qd.f64)
        frames = self._sim.substeps_local + 1
        self.cons_hist = struct_cons_hist.field(shape=(frames, n, self._B), layout=qd.Layout.SOA)
        self.acons_hist = struct_cons_hist.field(shape=(frames, na, self._B), layout=qd.Layout.SOA)
        self.cons_zeta = qd.field(dtype=qd.f64, shape=(n, self._B))
        self.acons_zeta = qd.field(dtype=qd.f64, shape=(na, self._B))

    def init_vvert_fields(self):
        # Same render contract as FEMSolver: several vverts may stand for one simulated vertex.
        struct_vvert_info = qd.types.struct(vert_idx=gs.qd_int)
        self.vverts_info = struct_vvert_info.field(shape=(max(self._n_vverts, 1),), layout=qd.Layout.SOA)
        struct_vvert_state_render = qd.types.struct(pos=gs.qd_vec3)
        self.vverts_render = struct_vvert_state_render.field(
            shape=(max(self._n_vverts, 1), self._B), layout=qd.Layout.SOA
        )
        self.vverts_uvs = qd.field(dtype=gs.qd_vec2, shape=(max(self._n_vverts, 1),))
        self.vfaces_indices = qd.field(dtype=gs.qd_ivec3, shape=(max(self._n_vfaces, 1),))
        self.envs_offset = qd.Vector.field(3, dtype=qd.f32, shape=self._B)
        self.envs_offset.from_numpy(self._scene.envs_offset.astype(np.float32))

    def _compute_vertex_coloring_and_incidence(self, elems, cons, acons):
        """Greedy vertex coloring of the graph of tets and constraints plus the vertex -> incident tet CSR list.

        Returns (perm, color_offsets, n_colors, ve_offset, ve_elem, ve_role, color): vertices sorted by color
        (`perm[color_offsets[c]:color_offsets[c+1]]` is color `c`), and for vertex `i` the incident tets
        `ve_elem[ve_offset[i]:ve_offset[i+1]]` with `ve_role` the local index of `i` in each tet. Two vertices
        that share a tet or a constraint never share a color, so each color is one race-free Gauss-Seidel sweep.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(self._n_vertices))
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            graph.add_edges_from(zip(elems[:, a].tolist(), elems[:, b].tolist()))
        graph.add_edges_from(zip(cons[:, 0].tolist(), cons[:, 1].tolist()))
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            keep = acons[:, a] != acons[:, b]  # a vertex may serve both vectors of an angle constraint
            graph.add_edges_from(zip(acons[keep, a].tolist(), acons[keep, b].tolist()))
        coloring = nx.greedy_color(graph, strategy="smallest_last")
        color = np.array([coloring[i] for i in range(self._n_vertices)], dtype=np.int64)
        assert (color[cons[:, 0]] != color[cons[:, 1]]).all(), "a constraint joins two vertices of the same color"
        assert all(
            (color[acons[:, a]] != color[acons[:, b]])[acons[:, a] != acons[:, b]].all() for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        )
        assert all((color[elems[:, a]] != color[elems[:, b]]).all() for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        n_colors = int(color.max()) + 1
        perm = np.argsort(color, kind="stable")
        color_offsets = np.searchsorted(color[perm], np.arange(n_colors + 1)).tolist()

        inc_vert = elems.reshape(-1)
        inc_elem = np.repeat(np.arange(self._n_elements), 4)
        inc_role = np.tile(np.arange(4), self._n_elements)
        order = np.argsort(inc_vert, kind="stable")
        ve_offset = np.searchsorted(inc_vert[order], np.arange(self._n_vertices + 1))
        return perm, color_offsets, n_colors, ve_offset, inc_elem[order], inc_role[order], color

    def _owner_csr(self, verts_per_constraint, color):
        """Per-vertex CSR of the constraints it owns. The owner is the constraint's vertex of highest color: when its
        color pass runs, every other vertex of the constraint has been updated this sweep and none is being written,
        so the owner can run the constraint's dual update inside its own pass, with no pass and barrier of its own."""
        n = len(verts_per_constraint)
        owner = verts_per_constraint[np.arange(n), np.argmax(color[verts_per_constraint], axis=1)] if n else np.zeros(0, dtype=np.int64)
        order = np.argsort(owner, kind="stable")
        offset = np.searchsorted(owner[order], np.arange(self._n_vertices + 1))
        f_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
        f_offset.from_numpy(offset.astype(gs.np_int))
        f_cons = qd.field(dtype=gs.qd_int, shape=(max(n, 1),))
        if n:
            f_cons.from_numpy(order.astype(gs.np_int))
        return f_offset, f_cons

    def build(self):
        super().build()
        self._n_vertices = self.n_vertices
        self._n_elements = self.n_elements
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces
        self._n_muscle_groups = max((getattr(e.material, "n_groups", 0) for e in self._entities), default=0)

        if self.is_active:
            self.init_vertex_fields()
            self.init_element_fields()
            self.init_vvert_fields()
            self.init_ckpt()
            self.muscle_actu.fill(0.0)
            self.bolus.radius.fill(0.0)

            for entity in self._entities:
                entity._add_to_solver()

            elems = np.concatenate([entity._v_start + entity.elems for entity in self._entities]).astype(np.int64)
            cons = np.concatenate(
                [entity._v_start + entity.distance_constraints for entity in self._entities] + [np.zeros((0, 2), dtype=np.int64)]
            ).astype(np.int64)
            lo = np.concatenate([entity.distance_bounds[:, 0] for entity in self._entities] + [np.zeros(0)])
            hi = np.concatenate([entity.distance_bounds[:, 1] for entity in self._entities] + [np.zeros(0)])
            acons = np.concatenate(
                [entity._v_start + entity.angle_constraints for entity in self._entities] + [np.zeros((0, 4), dtype=np.int64)]
            ).astype(np.int64)
            alo = np.concatenate([entity.angle_bounds[:, 0] for entity in self._entities] + [np.zeros(0)])
            ahi = np.concatenate([entity.angle_bounds[:, 1] for entity in self._entities] + [np.zeros(0)])
            self._n_constraints = len(cons)
            self._n_angle_constraints = len(acons)
            self.init_constraint_fields()
            perm, self._color_offsets, self._n_colors, ve_offset, ve_elem, ve_role, color = (
                self._compute_vertex_coloring_and_incidence(elems, cons, acons)
            )
            self._init_constraints(cons, lo, hi)
            self._init_angle_constraints(acons, alo, ahi)
            if self._damping > 0.0:
                for entity in self._entities:
                    # K0 (the rest Hessian) is positive semidefinite only for lam' >= mu / 3, i.e. nu >= 1/8
                    if entity.material.nu < 0.125:
                        gs.raise_exception(f"Rayleigh damping needs nu >= 0.125 for a positive semidefinite rest Hessian; got nu={entity.material.nu}.")
            self.vo_offset, self.vo_cons = self._owner_csr(cons, color)
            self.vao_offset, self.vao_cons = self._owner_csr(acons, color)
            self.color_perm = qd.field(dtype=gs.qd_int, shape=(self._n_vertices,))
            self.color_perm.from_numpy(perm.astype(gs.np_int))
            self.ve_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
            self.ve_offset.from_numpy(ve_offset.astype(gs.np_int))
            self.ve_elem = qd.field(dtype=gs.qd_int, shape=(len(ve_elem),))
            self.ve_elem.from_numpy(ve_elem.astype(gs.np_int))
            self.ve_role = qd.field(dtype=gs.qd_int, shape=(len(ve_role),))
            self.ve_role.from_numpy(ve_role.astype(gs.np_int))
            # The noise floor of the force assembly: no solve can drive the residual below the rounding error of the
            # terms it sums, so the relative tolerance is floored here. m/h^2 times a tet edge is the force that moves
            # a vertex one edge in one substep, the largest term in the sum; times the relative precision of the
            # accumulator, that is the smallest residual the assembly can resolve.
            edge = float(np.linalg.norm(self.verts.pos.to_numpy()[0, self.elems_info.v.to_numpy()[:, 1], 0] - self.verts.pos.to_numpy()[0, self.elems_info.v.to_numpy()[:, 0], 0], axis=1).mean())
            unit = float(self.verts_info.mass.to_numpy().max()) / self._substep_dt**2 * edge
            self._force_noise = unit * (1e-13 if gs.np_float == np.float64 else 1e-6)
            self.reset_grad()  # after the constraint fields exist: it snapshots the multipliers the first window starts from

    def _init_constraints(self, cons, lo, hi):
        """Bounds (rest length when the entity gave none), per-vertex CSR of incident constraints, and the stiffness
        scale k_start = mean vertex mass / h^2 (the inertia the local solve already carries)."""
        pos = self.verts.pos.to_numpy()[0, :, 0]
        rest = np.linalg.norm(pos[cons[:, 0]] - pos[cons[:, 1]], axis=1) if len(cons) else np.zeros(0)
        lo = np.where(np.isnan(lo), rest, lo)
        hi = np.where(np.isnan(hi), rest, hi)
        assert (lo <= hi).all() and (lo >= 0.0).all(), "distance bounds must satisfy 0 <= lo <= hi"
        inc_vert = cons.reshape(-1)
        inc_cons = np.repeat(np.arange(len(cons)), 2)
        inc_side = np.tile(np.array([1.0, -1.0]), len(cons))  # sign of dC/dx for this vertex
        order = np.argsort(inc_vert, kind="stable")
        vc_offset = np.searchsorted(inc_vert[order], np.arange(self._n_vertices + 1))
        self.vc_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
        self.vc_offset.from_numpy(vc_offset.astype(gs.np_int))
        self.vc_cons = qd.field(dtype=gs.qd_int, shape=(max(len(inc_cons), 1),))
        self.vc_side = qd.field(dtype=gs.qd_float, shape=(max(len(inc_cons), 1),))
        if len(cons):
            self.vc_cons.from_numpy(inc_cons[order].astype(gs.np_int))
            self.vc_side.from_numpy(inc_side[order].astype(gs.np_float))
            self.cons_info.v.from_numpy(cons.astype(gs.np_int))
            self.cons_info.lo.from_numpy(lo.astype(gs.np_float))
            self.cons_info.hi.from_numpy(hi.astype(gs.np_float))
            self.cons_info.scale.from_numpy(np.maximum(rest, gs.EPS).astype(gs.np_float))
        self._k_start = float(self.verts_info.mass.to_numpy().mean() / self._substep_dt**2)
        self.cons.lam_hi.fill(0.0)
        self.cons.lam_lo.fill(0.0)
        self.cons.k.fill(self._k_start)

    def _init_angle_constraints(self, acons, lo, hi):
        """Cosine bounds, per-vertex CSR (constraint, slot 0..3) and the initial stiffness k_start."""
        inc_vert = acons.reshape(-1)
        inc_cons = np.repeat(np.arange(len(acons)), 4)
        inc_slot = np.tile(np.arange(4), len(acons))
        order = np.argsort(inc_vert, kind="stable")
        va_offset = np.searchsorted(inc_vert[order], np.arange(self._n_vertices + 1))
        self.va_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
        self.va_offset.from_numpy(va_offset.astype(gs.np_int))
        self.va_cons = qd.field(dtype=gs.qd_int, shape=(max(len(inc_cons), 1),))
        self.va_slot = qd.field(dtype=gs.qd_int, shape=(max(len(inc_cons), 1),))
        if len(acons):
            assert (lo <= hi).all() and (lo >= -1.0).all() and (hi <= 1.0).all(), "cosine bounds must satisfy -1 <= lo <= hi <= 1"
            self.va_cons.from_numpy(inc_cons[order].astype(gs.np_int))
            self.va_slot.from_numpy(inc_slot[order].astype(gs.np_int))
            self.acons_info.v.from_numpy(acons.astype(gs.np_int))
            self.acons_info.lo.from_numpy(lo.astype(gs.np_float))
            self.acons_info.hi.from_numpy(hi.astype(gs.np_float))
        # base stiffness per constraint so that k |grad C|^2 matches a distance constraint of stiffness k_start:
        # k0 = k_start |u|^2 |v|^2 / (|u|^2 + |v|^2) on the rest vectors (the review's harmonic scale)
        if len(acons):
            pos = self.verts.pos.to_numpy()[0, :, 0]
            lu2 = (np.linalg.norm(pos[acons[:, 0]] - pos[acons[:, 1]], axis=1) ** 2)
            lv2 = (np.linalg.norm(pos[acons[:, 2]] - pos[acons[:, 3]], axis=1) ** 2)
            k0 = self._k_start * lu2 * lv2 / (lu2 + lv2)
            self.acons_info.k0.from_numpy(k0.astype(gs.np_float))
            u = pos[acons[:, 0]] - pos[acons[:, 1]]
            v = pos[acons[:, 2]] - pos[acons[:, 3]]
            cos_rest = (u * v).sum(-1) / np.sqrt(lu2 * lv2)
            # a cosine violation divided by sin(theta) is the angle error; near 0 or 180 degrees the cosine is flat
            # and no cosine tolerance is an angle tolerance, so the reference is floored
            self.acons_info.sin_ref.from_numpy(np.maximum(np.sqrt(np.clip(1.0 - cos_rest**2, 0.0, 1.0)), 0.1).astype(gs.np_float))
            self.acons.k.from_numpy(np.tile(k0.astype(gs.np_float)[:, None], (1, self._B)))
        else:
            self.acons.k.fill(self._k_start)
        self.acons.lam_hi.fill(0.0)
        self.acons.lam_lo.fill(0.0)

    def init_ckpt(self):
        self._ckpt = dict()

    def _snapshot_cons(self):
        return (
            [fld.to_numpy() for fld in (self.cons.lam_hi, self.cons.lam_lo, self.cons.k)],
            [fld.to_numpy() for fld in (self.acons.lam_hi, self.acons.lam_lo, self.acons.k)],
        )

    @property
    def is_active(self):
        return self.n_vertices > 0

    def add_entity(self, idx, material, morph, surface, name: str | None = None) -> "VBDEntity":
        entity = VBDEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            el_start=self.n_elements,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )
        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------ entity data -----------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_add_elements(
        self,
        v_start: qd.i32,
        el_start: qd.i32,
        verts: qd.types.ndarray(),
        elems: qd.types.ndarray(),
        mass: qd.f32,
        mu: qd.f32,
        lam: qd.f32,
        gain: qd.f32,
        mu_forward: qd.f32,
        mu_backward: qd.f32,
        mu_lateral: qd.f32,
    ):
        for i_v_ in range(verts.shape[0]):
            i_v = i_v_ + v_start
            self.verts_info[i_v].mass = mass
            self.verts_info[i_v].tangent = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
            self.verts_info[i_v].mu_forward = mu_forward
            self.verts_info[i_v].mu_backward = mu_backward
            self.verts_info[i_v].mu_lateral = mu_lateral
            for i_b in range(self._B):
                for j in qd.static(range(3)):
                    self.verts[0, i_v, i_b].pos[j] = verts[i_v_, j]
                self.verts[0, i_v, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_e_ in range(elems.shape[0]):
            i_e = i_e_ + el_start
            for j in qd.static(range(4)):
                self.elems_info[i_e].v[j] = elems[i_e_, j] + v_start
            p0 = self.verts[0, self.elems_info[i_e].v[0], 0].pos
            p1 = self.verts[0, self.elems_info[i_e].v[1], 0].pos
            p2 = self.verts[0, self.elems_info[i_e].v[2], 0].pos
            p3 = self.verts[0, self.elems_info[i_e].v[3], 0].pos
            Dm = qd.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
            self.elems_info[i_e].vol_rest = Dm.determinant() / 6.0
            self.elems_info[i_e].B_rest = Dm.inverse()
            self.elems_info[i_e].mu = mu
            self.elems_info[i_e].lam = lam + mu
            self.elems_info[i_e].gain = gain
            self.elems_info[i_e].fiber = qd.Vector.zero(gs.qd_float, 3)
            self.elems_info[i_e].group = -1
            self.elems_info[i_e].k_fiber = 0.0

    @qd.kernel
    def _kernel_add_vverts(
        self,
        vvert_start: qd.i32,
        vface_start: qd.i32,
        v_start: qd.i32,
        verts_idx: qd.types.ndarray(),
        uvs: qd.types.ndarray(element_dim=1),
        vfaces: qd.types.ndarray(element_dim=1),
    ):
        for i_vv_ in range(verts_idx.shape[0]):
            self.vverts_info[i_vv_ + vvert_start].vert_idx = verts_idx[i_vv_] + v_start
        for i_vv_ in range(uvs.shape[0]):
            self.vverts_uvs[i_vv_ + vvert_start] = uvs[i_vv_]
        for i_vf_ in range(vfaces.shape[0]):
            self.vfaces_indices[i_vf_ + vface_start] = vfaces[i_vf_] + vvert_start

    def set_friction_frame(self, v_start, tangent):
        self._kernel_set_friction_frame(v_start, tangent)

    @qd.kernel
    def _kernel_set_friction_frame(self, v_start: qd.i32, tangent: qd.types.ndarray()):
        for i_v_ in range(tangent.shape[0]):
            for j in qd.static(range(3)):
                self.verts_info[i_v_ + v_start].tangent[j] = tangent[i_v_, j]

    def set_bolus(self, center, radius, vel, friction, axis=(0.0, 0.0, 1.0), half_length=0.0):
        """Place the analytic bolus, a capsule: segment of `half_length` along the unit `axis` through `center`
        (B, 3), swept by `radius` (B,); `half_length` 0 is a sphere. `vel` (B, 3) is its prescribed velocity, used
        for the friction slide, `friction` the isotropic coefficient. A radius <= 0 disables it in that env."""
        if self._sim.requires_grad:
            gs.raise_exception("The bolus contact has no adjoint yet; disable requires_grad or the bolus.")
        axis = np.asarray(axis, dtype=gs.np_float)
        self._kernel_set_bolus(center, radius, vel, friction, axis / np.linalg.norm(axis), half_length)

    @qd.kernel
    def _kernel_set_bolus(
        self,
        center: qd.types.ndarray(),
        radius: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        friction: qd.f32,
        axis: qd.types.ndarray(),
        half_length: qd.f32,
    ):
        for i_b in range(self._B):
            for j in qd.static(range(3)):
                self.bolus[i_b].center[j] = center[i_b, j]
                self.bolus[i_b].vel[j] = vel[i_b, j]
                self.bolus[i_b].axis[j] = axis[j]
            self.bolus[i_b].radius = radius[i_b]
            self.bolus[i_b].friction = friction
            self.bolus[i_b].half_length = half_length

    def set_fiber_stiffness(self, el_start, k_fiber):
        self._kernel_set_fiber_stiffness(el_start, k_fiber)

    @qd.kernel
    def _kernel_set_fiber_stiffness(self, el_start: qd.i32, k_fiber: qd.types.ndarray()):
        for i_e_ in range(k_fiber.shape[0]):
            self.elems_info[i_e_ + el_start].k_fiber = k_fiber[i_e_]

    def set_muscle(self, el_start, group, fiber):
        self._kernel_set_muscle(el_start, group, fiber)

    @qd.kernel
    def _kernel_set_muscle(self, el_start: qd.i32, group: qd.types.ndarray(), fiber: qd.types.ndarray()):
        for i_e_ in range(group.shape[0]):
            i_e = i_e_ + el_start
            self.elems_info[i_e].group = group[i_e_]
            for j in qd.static(range(3)):
                self.elems_info[i_e].fiber[j] = fiber[i_e_, j]

    def set_actuation(self, actus):
        self._kernel_set_actuation(actus)

    @qd.kernel
    def _kernel_set_actuation(self, actus: qd.types.ndarray()):
        for i_g, i_b in qd.ndrange(actus.shape[0], actus.shape[1]):
            self.muscle_actu[i_g, i_b] = actus[i_g, i_b]

    # ------------------------------------------------------------------------------------
    # ------------------------------------- physics --------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def _func_rest_inverse(self, i_e, i_b):
        """Inverse rest shape of tet `i_e`, contracted along its fiber when actuated (`B_a = B A`)."""
        B = self.elems_info[i_e].B_rest
        group = self.elems_info[i_e].group
        if group >= 0:
            s = 1.0 - self.muscle_actu[group, i_b] * self.elems_info[i_e].gain
            m = self.elems_info[i_e].fiber
            mmT = m.outer_product(m)
            A = (1.0 / s) * mmT + qd.sqrt(s) * (qd.Matrix.identity(gs.qd_float, 3) - mmT)
            B = B @ A
        return B

    @qd.func
    def _func_deformation(self, fr, i_e, i_b):
        """(F, B_eff) of tet `i_e` in env `i_b` at frame `fr`."""
        v = self.elems_info[i_e].v
        p0 = self.verts[fr, v[0], i_b].pos
        Ds = qd.Matrix.cols(
            [self.verts[fr, v[1], i_b].pos - p0, self.verts[fr, v[2], i_b].pos - p0, self.verts[fr, v[3], i_b].pos - p0]
        )
        B = self._func_rest_inverse(i_e, i_b)
        return Ds @ B, B

    @qd.func
    def _func_cofactor(self, F):
        return qd.Matrix.cols([F[:, 1].cross(F[:, 2]), F[:, 2].cross(F[:, 0]), F[:, 0].cross(F[:, 1])])

    @qd.func
    def _func_vertex_weight(self, B, role):
        """Row of `B` acting on local vertex `role`; vertex 0 carries minus the sum of the other rows."""
        w = qd.Vector.zero(gs.qd_float, 3)
        if role == 0:
            w = -(B[0, :] + B[1, :] + B[2, :])
        else:
            w = B[role - 1, :]
        return w

    @qd.func
    def _func_vertex_weight_static(self, B, role: qd.template()):
        """`_func_vertex_weight` for a compile-time `role`: a runtime `if` would compile `B[-1, :]` for role 0."""
        w = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(role == 0):
            w = -(B[0, :] + B[1, :] + B[2, :])
        else:
            w = B[role - 1, :]
        return w

    @qd.func
    def _func_fiber_terms(self, fr, i_e, i_b, w_i):
        """Fibre reinforcement E = V k/2 (|F0 a| - 1)^2 on the unactuated F0 = Ds B_rest (a spine or tendon: it
        resists length change along `a` whatever the muscle does). Returns (force on the vertex with row
        weight `w_i`, exact 3x3 Hessian block, PSD part of that block). The (l - 1)/l (I - u u^T) part is
        negative in compression, so the forward step uses the PSD part like the contact terms do."""
        v = self.elems_info[i_e].v
        p0 = self.verts[fr, v[0], i_b].pos
        Ds = qd.Matrix.cols([self.verts[fr, v[1], i_b].pos - p0, self.verts[fr, v[2], i_b].pos - p0, self.verts[fr, v[3], i_b].pos - p0])
        F0 = Ds @ self.elems_info[i_e].B_rest
        a = self.elems_info[i_e].fiber
        u = F0 @ a
        l = u.norm()
        u_hat = u / l
        c = self.elems_info[i_e].vol_rest * self.elems_info[i_e].k_fiber * (w_i.dot(a)) ** 2
        force = -self.elems_info[i_e].vol_rest * self.elems_info[i_e].k_fiber * (l - 1.0) * w_i.dot(a) * u_hat
        H_psd = c * u_hat.outer_product(u_hat)
        H_exact = H_psd + c * ((l - 1.0) / l) * (qd.Matrix.identity(gs.qd_float, 3) - u_hat.outer_product(u_hat))
        return force, H_exact, H_psd

    @qd.func
    def _func_inertia_target(self, f, i_v, i_b):
        """y = x^t + h (v^t + h g), the position the vertex would reach with no internal forces."""
        vel = self.verts[f, i_v, i_b].vel + self._gravity[i_b] * self._substep_dt
        return self.verts[f, i_v, i_b].pos + vel * self._substep_dt

    @qd.func
    def _func_vertex_system(self, f, i_v, i_b):
        """Negative gradient `force`, Hessian `H` of the incremental potential of substep `f` with respect to vertex
        `i_v`, evaluated at the current iterate `verts[f+1].pos` with every other vertex fixed, and `K0`, the vertex's
        diagonal block of the rest Hessian that the Rayleigh damping uses."""
        inv_h2 = 1.0 / (self._substep_dt * self._substep_dt)
        m_h2 = qd.cast(self.verts_info[i_v].mass * inv_h2, self._acc)
        x = self.verts[f + 1, i_v, i_b].pos
        force = -m_h2 * qd.cast(x - self._func_inertia_target(f, i_v, i_b), self._acc)
        K = qd.Matrix.zero(self._acc, 3, 3)  # elastic Hessian block (positive semidefinite part)
        # Rayleigh damping C = k_d K0 with K0 the exact Hessian at rest: constant, positive semidefinite, and it
        # annihilates rigid motions, so a coiling body is not dragged. Force -(k_d/h) sum_j K0_ij (x_j - x_j^t) over
        # the vertex and its neighbours; K0 constant makes the damping's Jacobian exact and cheap for the adjoint.
        # (The VBD paper's Eq. 11 keeps only the diagonal block, which drags every vertex against the floor frame.)
        K0 = qd.Matrix.zero(self._acc, 3, 3)
        damp = qd.Vector.zero(self._acc, 3)

        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            role = self.ve_role[c]
            F, B = self._func_deformation(f + 1, i_e, i_b)
            mu = self.elems_info[i_e].mu
            lam = self.elems_info[i_e].lam
            alpha = 1.0 + mu / lam
            cof = self._func_cofactor(F)
            J = F.determinant()
            P = mu * F + lam * (J - alpha) * cof
            w = self._func_vertex_weight(B, role)
            V = self.elems_info[i_e].vol_rest
            q = qd.cast(cof @ w, self._acc)
            force -= qd.cast(V * (P @ w), self._acc)
            K += qd.cast(V * mu * w.norm_sqr(), self._acc) * qd.Matrix.identity(self._acc, 3)
            K += qd.cast(V * lam, self._acc) * q.outer_product(q)
            if self.elems_info[i_e].k_fiber > 0.0:
                w0 = self._func_vertex_weight(self.elems_info[i_e].B_rest, role)
                f_fib, _, H_fib = self._func_fiber_terms(f + 1, i_e, i_b, w0)
                force += qd.cast(f_fib, self._acc)
                K += qd.cast(H_fib, self._acc)
            if qd.static(self._damping > 0.0):
                B0 = self.elems_info[i_e].B_rest
                w0 = self._func_vertex_weight(B0, role)
                K0 += qd.cast(self._func_rest_block(i_e, w0, w0), self._acc)
                for r in qd.static(range(4)):
                    if r != role:
                        j = self.elems_info[i_e].v[r]
                        d_j = self.verts[f + 1, j, i_b].pos - self.verts[f, j, i_b].pos
                        damp += qd.cast(self._func_rest_block(i_e, w0, self._func_vertex_weight_static(B0, r)) @ d_j, self._acc)

        kd_h = qd.cast(self._damping / self._substep_dt, self._acc)
        force -= kd_h * (K0 @ qd.cast(x - self.verts[f, i_v, i_b].pos, self._acc) + damp)
        H = m_h2 * qd.Matrix.identity(self._acc, 3) + K + kd_h * K0

        # Hard distance constraints, augmented Lagrangian: force -(k C + lam) dC/dx, Hessian k n n^T plus the
        # diagonal-norm proxy of the constraint's curvature (|k C + lam| / |e| on the tangent plane), which keeps
        # the block positive definite in compression (Giles et al. 2025, Sec. 3.5).
        for c in range(self.vc_offset[i_v], self.vc_offset[i_v + 1]):
            i_c = self.vc_cons[c]
            side = self.vc_side[c]
            va = self.cons_info[i_c].v[0]
            vb = self.cons_info[i_c].v[1]
            e = self.verts[f + 1, va, i_b].pos - self.verts[f + 1, vb, i_b].pos
            dist = e.norm()
            n = e / dist
            mult, violation = self._func_constraint_mult(i_c, i_b, dist)
            force -= qd.cast(mult * side, self._acc) * qd.cast(n, self._acc)
            if mult != 0.0:  # an active side: its stiffness enters the block (Eq. 14 without rescaling)
                H += qd.cast(self.cons[i_c, i_b].k, self._acc) * qd.cast(n.outer_product(n), self._acc)
            H += qd.cast(qd.abs(mult) / dist, self._acc) * qd.cast(qd.Matrix.identity(gs.qd_float, 3) - n.outer_product(n), self._acc)

        # Angle constraints: C = cos(u, v) bounded; gradient wrt x_a is (v_hat - C u_hat) / |u| (minus for x_b), and
        # wrt x_c is (u_hat - C v_hat) / |v| (minus for x_d). Hessian: k g g^T plus |mult| / |edge|^2 as the PSD proxy
        # of the cosine's curvature.
        for c in range(self.va_offset[i_v], self.va_offset[i_v + 1]):
            i_c = self.va_cons[c]
            slot = self.va_slot[c]
            vq = self.acons_info[i_c].v
            u = self.verts[f + 1, vq[0], i_b].pos - self.verts[f + 1, vq[1], i_b].pos
            vv = self.verts[f + 1, vq[2], i_b].pos - self.verts[f + 1, vq[3], i_b].pos
            lu = u.norm()
            lv = vv.norm()
            u_hat = u / lu
            v_hat = vv / lv
            cosv = u_hat.dot(v_hat)
            g = (v_hat - cosv * u_hat) / lu
            scale = lu
            if slot >= 2:
                g = (u_hat - cosv * v_hat) / lv
                scale = lv
            if slot == 1 or slot == 3:
                g = -g
            mult, _ = self._func_angle_mult(i_c, i_b, cosv)
            force -= qd.cast(mult, self._acc) * qd.cast(g, self._acc)
            if mult != 0.0:
                H += qd.cast(self.acons[i_c, i_b].k, self._acc) * qd.cast(g.outer_product(g), self._acc)
            own = u_hat
            if slot >= 2:
                own = v_hat
            H += qd.cast(qd.abs(mult) / (scale * scale), self._acc) * qd.cast(qd.Matrix.identity(gs.qd_float, 3) - own.outer_product(own), self._acc)

        # Floor contact (VBD paper 3.5): penalty energy k/2 d^2 on the penetration depth d, plus anisotropic
        # Coulomb friction (3.6, Hu et al. 2009 coefficients) on the substep's tangential slide, with the IPC
        # transition f1 blending static and dynamic friction below the speed friction_eps_v. The forward/backward
        # coefficient switch is a tanh blend on the same scale, so the residual stays smooth for the adjoint.
        d = self._floor_height - x[2]
        if d > 0.0:
            k = self._contact_stiffness
            force[2] += qd.cast(k * d, self._acc)
            H[2, 2] += qd.cast(k, self._acc)

            slide = x - self.verts[f, i_v, i_b].pos
            slide[2] = 0.0
            t = self.verts_info[i_v].tangent
            t[2] = 0.0
            t = t.normalized()
            b = qd.Vector([-t[1], t[0], 0.0], dt=gs.qd_float)
            u_t = slide.dot(t)
            u_b = slide.dot(b)
            u_norm = slide.norm()
            eps = self._friction_eps_v * self._substep_dt
            g = 1.0 / u_norm  # f1(|u|) / |u| of Eq. 15, finite at |u| = 0
            if u_norm < eps:
                g = 2.0 / eps - u_norm / (eps * eps)
            mu_f = self.verts_info[i_v].mu_forward
            mu_bw = self.verts_info[i_v].mu_backward
            mu_ax = 0.5 * (mu_f + mu_bw) + 0.5 * (mu_f - mu_bw) * qd.tanh(u_t / eps)
            lam_n = k * d
            force -= qd.cast(lam_n * g * (mu_ax * u_t * t + self.verts_info[i_v].mu_lateral * u_b * b), self._acc)
            H += qd.cast(lam_n * g, self._acc) * (
                qd.cast(mu_ax, self._acc) * qd.cast(t.outer_product(t), self._acc)
                + qd.cast(self.verts_info[i_v].mu_lateral, self._acc) * qd.cast(b.outer_product(b), self._acc)
            )

        # Analytic capsule bolus: the same penalty and IPC-smoothed isotropic Coulomb friction against a moving
        # capsule (sphere when half_length is 0). `rel` is the vector from the closest point of the segment.
        r_b = self.bolus[i_b].radius
        if r_b > 0.0:
            rel = x - self.bolus[i_b].center
            along = qd.min(qd.max(rel.dot(self.bolus[i_b].axis), -self.bolus[i_b].half_length), self.bolus[i_b].half_length)
            rel = rel - along * self.bolus[i_b].axis
            dist = rel.norm()
            pen = r_b - dist  # penetration depth into the sphere
            if pen > 0.0:
                k = self._contact_stiffness
                n_b = rel / dist
                force += qd.cast(k * pen, self._acc) * qd.cast(n_b, self._acc)
                H += qd.cast(k, self._acc) * qd.cast(n_b.outer_product(n_b), self._acc)
                slide = x - self.verts[f, i_v, i_b].pos - self.bolus[i_b].vel * self._substep_dt
                slide -= slide.dot(n_b) * n_b
                u_norm = slide.norm()
                eps = self._friction_eps_v * self._substep_dt
                g = 1.0 / u_norm
                if u_norm < eps:
                    g = 2.0 / eps - u_norm / (eps * eps)
                lam_b = k * pen * self.bolus[i_b].friction
                force -= qd.cast(lam_b * g, self._acc) * qd.cast(slide, self._acc)
                H += qd.cast(lam_b * g, self._acc) * qd.cast(qd.Matrix.identity(gs.qd_float, 3) - n_b.outer_product(n_b), self._acc)
        return force, H, K0

    @qd.func
    def _func_rest_block(self, i_e, w_i, w_j):
        """Block of tet i_e's exact energy Hessian at rest (F = I) between the vertices with unactuated weights w_i and
        w_j: the stable neo-Hookean part V [mu (w_i.w_j) I + lam' w_i w_j^T + mu [w_i x w_j]_x] (the cross term at
        J - alpha = -mu/lam', which vanishes on the diagonal block by itself) plus the fibre part k (w_i.a)(w_j.a) a a^T. Symmetric as a whole,
        positive semidefinite, zero on rigid motions. (Weights come from the callers: the static-index weight function
        must see a compile-time role, the runtime one a runtime role.)"""
        V = self.elems_info[i_e].vol_rest
        mu = self.elems_info[i_e].mu
        lam = self.elems_info[i_e].lam
        blk = V * (mu * w_i.dot(w_j) * qd.Matrix.identity(gs.qd_float, 3) + lam * w_i.outer_product(w_j))
        c = w_i.cross(w_j)  # zero on the diagonal block
        blk += V * mu * qd.Matrix([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
        if self.elems_info[i_e].k_fiber > 0.0:
            a = self.elems_info[i_e].fiber
            blk += V * self.elems_info[i_e].k_fiber * w_i.dot(a) * w_j.dot(a) * a.outer_product(a)
        return blk

    @qd.func
    def _func_solve_vertex(self, f, i_v, i_b, w, ramp, record, sweep):
        """One Newton step of vertex i_v, then the dual updates of the constraints it owns (relaxation w, stiffness
        ramp on or off), recording their multipliers for the adjoint when `record` is set. w = 0 skips the duals.
        `sweep` is the index of this sweep within the substep, for the replay buffer."""
        force, H, K_unused = self._func_vertex_system(f, i_v, i_b)
        dx = H.inverse() @ force
        if qd.static(self._record_sweeps):
            self.sweep_dx[f, sweep, i_v, i_b] = qd.cast(dx, qd.f64)
        self.verts[f + 1, i_v, i_b].pos += qd.cast(dx, gs.qd_float)
        if w > 0.0:
            for c in range(self.vo_offset[i_v], self.vo_offset[i_v + 1]):
                i_c = self.vo_cons[c]
                self._func_dual_update(f, i_c, i_b, w, ramp)
                if record:
                    self._func_record_constraint(f, i_c, i_b)
            for c in range(self.vao_offset[i_v], self.vao_offset[i_v + 1]):
                i_c = self.vao_cons[c]
                self._func_angle_dual_update(f, i_c, i_b, w, ramp)
                if record:
                    self._func_record_angle(f, i_c, i_b)

    @qd.kernel
    def _kernel_predict(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.verts[f + 1, i_v, i_b].pos = self._func_inertia_target(f, i_v, i_b)

    @qd.kernel
    def _kernel_solve_color(self, f: qd.i32, lo: qd.i32, hi: qd.i32):
        """One color of one sweep. Kept for tests that watch the energy sweep by sweep."""
        for k, i_b in qd.ndrange((lo, hi), self._B):
            self._func_solve_vertex(f, self.color_perm[k], i_b, self._constraint_dual_relaxation, 1.0, True, 0)

    @qd.func
    def _func_constraint_mult(self, i_c, i_b, dist):
        """Clamped total multiplier of a bounded distance (Giles et al. 2025 Eq. 13): the upper bound can only pull the
        distance down (>= 0), the lower bound only push it up (<= 0); inside the bounds both vanish as their
        multipliers decay. Returns (mult, violation) with violation the signed distance error used for the ramp."""
        k = self.cons[i_c, i_b].k
        c_hi = dist - self.cons_info[i_c].hi
        c_lo = dist - self.cons_info[i_c].lo
        mult = k * c_hi + self.cons[i_c, i_b].lam_hi  # equality: one unclamped multiplier, kept in lam_hi
        violation = c_hi
        if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
            mult = qd.max(k * c_hi + self.cons[i_c, i_b].lam_hi, 0.0) + qd.min(k * c_lo + self.cons[i_c, i_b].lam_lo, 0.0)
            violation = qd.max(c_hi, 0.0) + qd.min(c_lo, 0.0)
        return mult, violation

    @qd.func
    def _func_angle_mult(self, i_c, i_b, cosv):
        k = self.acons[i_c, i_b].k
        c_hi = cosv - self.acons_info[i_c].hi
        c_lo = cosv - self.acons_info[i_c].lo
        mult = k * c_hi + self.acons[i_c, i_b].lam_hi
        violation = c_hi
        if self.acons_info[i_c].lo < self.acons_info[i_c].hi:
            mult = qd.max(k * c_hi + self.acons[i_c, i_b].lam_hi, 0.0) + qd.min(k * c_lo + self.acons[i_c, i_b].lam_lo, 0.0)
            violation = qd.max(c_hi, 0.0) + qd.min(c_lo, 0.0)
        return mult, violation

    @qd.func
    def _func_angle_dual_update(self, f, i_c, i_b, w, ramp):
        vq = self.acons_info[i_c].v
        u = self.verts[f + 1, vq[0], i_b].pos - self.verts[f + 1, vq[1], i_b].pos
        vv = self.verts[f + 1, vq[2], i_b].pos - self.verts[f + 1, vq[3], i_b].pos
        cosv = u.dot(vv) / (u.norm() * vv.norm())
        k = self.acons[i_c, i_b].k
        if self.acons_info[i_c].lo < self.acons_info[i_c].hi:
            self.acons[i_c, i_b].lam_hi = qd.max(self.acons[i_c, i_b].lam_hi + w * k * (cosv - self.acons_info[i_c].hi), 0.0)
            self.acons[i_c, i_b].lam_lo = qd.min(self.acons[i_c, i_b].lam_lo + w * k * (cosv - self.acons_info[i_c].lo), 0.0)
        else:
            self.acons[i_c, i_b].lam_hi += w * k * (cosv - self.acons_info[i_c].hi)
        _, violation = self._func_angle_mult(i_c, i_b, cosv)
        k0 = self.acons_info[i_c].k0
        self.acons[i_c, i_b].k = qd.min(k + ramp * k0 / self._angle_tol * qd.abs(violation), self._constraint_k_max_ratio * k0)

    @qd.func
    def _func_dual_update(self, f, i_c, i_b, w, ramp):
        """Giles et al. 2025 Eq. 11 to 13: clamped lam += w k C per side, k += ramp beta |C| with beta = k_start /
        constraint_tol. The per-sweep forward uses w = constraint_dual_relaxation and ramp = 1; the exact Uzawa
        iteration under requires_grad uses w = 1 and ramp = 0 (a stiff k only slows the primal Gauss-Seidel)."""
        e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
        dist = e.norm()
        k = self.cons[i_c, i_b].k
        if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
            self.cons[i_c, i_b].lam_hi = qd.max(self.cons[i_c, i_b].lam_hi + w * k * (dist - self.cons_info[i_c].hi), 0.0)
            self.cons[i_c, i_b].lam_lo = qd.min(self.cons[i_c, i_b].lam_lo + w * k * (dist - self.cons_info[i_c].lo), 0.0)
        else:
            self.cons[i_c, i_b].lam_hi += w * k * (dist - self.cons_info[i_c].hi)
        _, violation = self._func_constraint_mult(i_c, i_b, dist)
        self.cons[i_c, i_b].k = qd.min(k + ramp * self._k_start / self._constraint_tol * qd.abs(violation), self._constraint_k_max_ratio * self._k_start)

    @qd.kernel
    def _kernel_sweeps(self, f: qd.i32):
        """`n_iterations` Gauss-Seidel sweeps in one launch. Each top-level loop is a serial task with an implicit
        barrier after it, so the statically unrolled color loops are race-free without a Python round trip. The
        constraints' dual updates ride inside the color pass of their owner vertex (see `_owner_csr`): a pass of
        their own would be a few thousand threads behind a barrier, and cost half the step on the ladder body."""
        for sweep in qd.static(range(self._n_iterations)):
            for c in qd.static(range(self._n_colors)):
                for k, i_b in qd.ndrange((self._color_offsets[c], self._color_offsets[c + 1]), self._B):
                    self._func_solve_vertex(f, self.color_perm[k], i_b, self._constraint_dual_relaxation, 1.0, sweep == self._n_iterations - 1, sweep)

    @qd.func
    def _func_record_constraint(self, f, i_c, i_b):
        e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
        dist = e.norm()
        mult, _ = self._func_constraint_mult(i_c, i_b, dist)
        k = self.cons[i_c, i_b].k
        sides = 1.0
        if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
            sides = 0.0
            if k * (dist - self.cons_info[i_c].hi) + self.cons[i_c, i_b].lam_hi > 0.0:
                sides += 1.0
            if k * (dist - self.cons_info[i_c].lo) + self.cons[i_c, i_b].lam_lo < 0.0:
                sides += 1.0
        self.cons_hist[f + 1, i_c, i_b].mult = mult
        self.cons_hist[f + 1, i_c, i_b].k_eff = k * sides

    @qd.func
    def _func_record_angle(self, f, i_c, i_b):
        vq = self.acons_info[i_c].v
        u = self.verts[f + 1, vq[0], i_b].pos - self.verts[f + 1, vq[1], i_b].pos
        vv = self.verts[f + 1, vq[2], i_b].pos - self.verts[f + 1, vq[3], i_b].pos
        cosv = u.dot(vv) / (u.norm() * vv.norm())
        mult, _ = self._func_angle_mult(i_c, i_b, cosv)
        k = self.acons[i_c, i_b].k
        sides = 1.0
        if self.acons_info[i_c].lo < self.acons_info[i_c].hi:
            sides = 0.0
            if k * (cosv - self.acons_info[i_c].hi) + self.acons[i_c, i_b].lam_hi > 0.0:
                sides += 1.0
            if k * (cosv - self.acons_info[i_c].lo) + self.acons[i_c, i_b].lam_lo < 0.0:
                sides += 1.0
        self.acons_hist[f + 1, i_c, i_b].mult = mult
        self.acons_hist[f + 1, i_c, i_b].k_eff = k * sides

    @qd.kernel
    def _kernel_record(self, f: qd.i32):
        """Record every constraint's multiplier and effective stiffness at the converged state of substep f."""
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            self._func_record_constraint(f, i_c, i_b)
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            self._func_record_angle(f, i_c, i_b)

    @qd.kernel
    def _kernel_primal_sweeps(self, f: qd.i32):
        """`n_iterations` sweeps with the multipliers held fixed: the inner solve of the exact Uzawa iteration used
        under requires_grad. The constraint record is taken by `solve()` when it returns."""
        for sweep in qd.static(range(self._n_iterations)):
            for c in qd.static(range(self._n_colors)):
                for k, i_b in qd.ndrange((self._color_offsets[c], self._color_offsets[c + 1]), self._B):
                    self._func_solve_vertex(f, self.color_perm[k], i_b, 0.0, 0.0, False, sweep)

    @qd.kernel
    def _kernel_dual_update(self, f: qd.i32):
        """One exact Uzawa step: full multiplier update, stiffness held (k_start, or the warm-started value)."""
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            self._func_dual_update(f, i_c, i_b, 1.0, 0.0)
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            self._func_angle_dual_update(f, i_c, i_b, 1.0, 0.0)

    @qd.kernel
    def _kernel_reset_constraints(self, mask: qd.types.ndarray()):
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            if mask[i_b] != 0:
                self.cons[i_c, i_b].lam_hi = 0.0
                self.cons[i_c, i_b].lam_lo = 0.0
                self.cons[i_c, i_b].k = self._k_start
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            if mask[i_b] != 0:
                self.acons[i_c, i_b].lam_hi = 0.0
                self.acons[i_c, i_b].lam_lo = 0.0
                self.acons[i_c, i_b].k = self.acons_info[i_c].k0

    def reset_constraints(self, envs_mask):
        """Clear the constraint multipliers and stiffness ramps of the environments where `envs_mask` (bool tensor of
        shape (n_envs,)) is true: a body put back to its rest state must not keep the tensions of its last episode."""
        self._kernel_reset_constraints(envs_mask.to(torch.int32).contiguous())

    @qd.kernel
    def _kernel_undo_sweep(self, f: qd.i32, sweep: qd.i32, lo: qd.i32, hi: qd.i32):
        """Subtract the updates one colour of one sweep applied, so the positions return to what that colour started
        from. Undoing the colours in reverse recovers each block's linearisation point, and stores no position."""
        for k, i_b in qd.ndrange((lo, hi), self._B):
            i_v = self.color_perm[k]
            self.verts[f + 1, i_v, i_b].pos -= qd.cast(self.sweep_dx[f, sweep, i_v, i_b], gs.qd_float)

    @qd.kernel
    def _kernel_warm_start(self):
        """Multiplier decay per substep: lam <- alpha gamma lam, k <- max(k_start, gamma k) (the AVBD warm start, run
        every substep rather than per frame; see useful_knowledge.md, 2026-09-06)."""
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            self.cons[i_c, i_b].lam_hi *= 0.95 * 0.99
            self.cons[i_c, i_b].lam_lo *= 0.95 * 0.99
            self.cons[i_c, i_b].k = qd.max(self._k_start, 0.99 * self.cons[i_c, i_b].k)
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            self.acons[i_c, i_b].lam_hi *= 0.95 * 0.99
            self.acons[i_c, i_b].lam_lo *= 0.95 * 0.99
            self.acons[i_c, i_b].k = qd.max(self.acons_info[i_c].k0, 0.99 * self.acons[i_c, i_b].k)

    @qd.kernel
    def _kernel_constraint_error(self, f: qd.i32):
        self.cons_error[None] = 0.0
        self.cons_error_rel[None] = 0.0
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
            _, violation = self._func_constraint_mult(i_c, i_b, e.norm())
            qd.atomic_max(self.cons_error[None], qd.cast(qd.abs(violation), qd.f64))
            qd.atomic_max(self.cons_error_rel[None], qd.cast(qd.abs(violation) / self.cons_info[i_c].scale, qd.f64))

    @qd.kernel
    def _kernel_angle_error(self, f: qd.i32):
        self.cons_error[None] = 0.0
        self.cons_error_rel[None] = 0.0
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            vq = self.acons_info[i_c].v
            u = self.verts[f + 1, vq[0], i_b].pos - self.verts[f + 1, vq[1], i_b].pos
            vv = self.verts[f + 1, vq[2], i_b].pos - self.verts[f + 1, vq[3], i_b].pos
            _, violation = self._func_angle_mult(i_c, i_b, u.dot(vv) / (u.norm() * vv.norm()))
            qd.atomic_max(self.cons_error[None], qd.cast(qd.abs(violation), qd.f64))
            qd.atomic_max(self.cons_error_rel[None], qd.cast(qd.abs(violation) / self.acons_info[i_c].sin_ref, qd.f64))

    def angle_constraint_error(self):
        """Largest cosine violation of the angle constraints at the current end-of-substep positions, over all envs."""
        self._kernel_angle_error(self._sim.cur_substep_local - 1 if self._sim.cur_substep_local > 0 else self._sim.substeps_local - 1)
        return float(self.cons_error[None])

    def constraint_error(self):
        """Largest absolute distance-constraint error (m) at the current end-of-substep positions, over all envs."""
        self._kernel_constraint_error(self._sim.cur_substep_local - 1 if self._sim.cur_substep_local > 0 else self._sim.substeps_local - 1)
        return float(self.cons_error[None])

    @qd.kernel
    def _kernel_update_velocity(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.verts[f + 1, i_v, i_b].vel = (self.verts[f + 1, i_v, i_b].pos - self.verts[f, i_v, i_b].pos) / self._substep_dt

    @qd.kernel
    def _kernel_residual(self, f: qd.i32):
        """Largest force component left on any vertex of any env: the stationarity residual of substep `f`."""
        self.residual[None] = 0.0
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            force, H_unused, K_unused = self._func_vertex_system(f, i_v, i_b)
            qd.atomic_max(self.residual[None], qd.cast(qd.abs(force).max(), qd.f64))

    @qd.kernel
    def _kernel_residual_vector(self, f: qd.i32, out: qd.types.ndarray()):
        """r_i = -force_i of substep `f` at the current iterate, shape (B, n_vertices, 3). Test hook."""
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            force, H_unused, K_unused = self._func_vertex_system(f, i_v, i_b)
            for j in qd.static(range(3)):
                out[i_b, i_v, j] = -force[j]

    def _violation(self, f):
        """Largest relative constraint violation at the current iterate of substep f: strain for a distance, radians
        for an angle. Both are dimensionless, so one tolerance covers a 1 mm ligament and a 10 cm segment."""
        worst = 0.0
        if self._n_constraints > 0:
            self._kernel_constraint_error(f)
            worst = float(self.cons_error_rel[None])
        if self._n_angle_constraints > 0:
            self._kernel_angle_error(f)
            worst = max(worst, float(self.cons_error_rel[None]))
        return worst

    def _solve_primal(self, f, force_ref):
        """Sweeps with fixed multipliers until the stationarity residual falls to `residual_tol` of `force_ref`."""
        # stop at the tolerance, or when the residual reaches the noise of the force assembly and no sweep can lower
        # it further (a substep that starts already stationary never reaches a fraction of its own zero)
        target = max(self._residual_tol * force_ref, self._force_noise)
        for _ in range(self._max_sweeps // self._n_iterations):
            self._kernel_residual(f)
            if self.residual[None] < target:
                return
            self._kernel_primal_sweeps(f)
        self._kernel_residual(f)
        gs.raise_exception(
            f"VBD substep did not converge: residual {self.residual[None]:.3e} >= {target:.3e} "
            f"({self._residual_tol:.1e} of the {force_ref:.3e} N the substep started with) after {self._max_sweeps} sweeps."
        )

    def solve(self, f):
        """Fixed sweeps with a dual update per sweep normally. Under requires_grad the adjoint differentiates the
        converged KKT system and inherits any leftover as bias, so the step is an exact Uzawa iteration: primal
        sweeps to a relative force tolerance, one dual update, until the relative constraint violation is below its
        own tolerance too. (A dual update on an unconverged iterate overshoots at the stiffness cap and limit-cycles
        instead of converging.)"""
        if not self._sim.requires_grad or not self._grad_converge:
            self._kernel_sweeps(f)  # the fast path: a fixed number of sweeps, the duals fused into the colour passes
            return
        # The force scale of this substep: the imbalance left at the predicted position, floored by the body's own
        # weight so that a body already at rest still has a finite scale. An absolute newton tolerance is meaningless
        # (a 35 kg body and a 1 g block carry forces four orders apart, and float32's own residual floor is about 1 N).
        self._kernel_residual(f)
        force_ref = float(self.residual[None])
        self._force_ref = force_ref  # kept for inspection: the scale the stationarity tolerance is relative to
        for _ in range(self._max_dual_steps):
            self._solve_primal(f, force_ref)
            if self._violation(f) < self._violation_tol:
                self._kernel_record(f)  # here, not inside the sweeps: an already-stationary iterate sweeps zero times
                return
            self._kernel_dual_update(f)
        gs.raise_exception(
            f"VBD substep did not converge: relative constraint violation {self._violation(f):.3e} >= "
            f"{self._violation_tol:.1e} after {self._max_dual_steps} dual updates."
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------- adjoint --------------------------------------
    # ------------------------------------------------------------------------------------
    # The substep ends at r(x) = 0 with r = -force of `_func_vertex_system`. With J = dr/dx, the adjoint z solves
    # J^T z = gbar, gbar = dL/dx^{t+1} + dL/dv^{t+1} / h, and then
    #   dL/dx^t += (M/h^2) z - dL/dv^{t+1} / h - (dr/dx^t)^T z,   dL/dv^t += (M/h) z,   dL/da -= z . dr/da.
    # The elastic Jacobian is symmetric, so the off-diagonal action of J^T is the off-diagonal action of J;
    # only the per-vertex contact/friction block is nonsymmetric and gets transposed explicitly.

    @qd.func
    def _func_friction_terms(self, f, i_v, i_b):
        """(lam_n, A_f, coupling) of vertex `i_v` at frame f+1: the normal force, the exact tangential Jacobian
        d(lam_n g P u)/du (3x3, nonsymmetric) and the normal coupling -k (g P u) e_z^T. All zero out of contact."""
        x = self.verts[f + 1, i_v, i_b].pos
        d = self._floor_height - x[2]
        lam_n = 0.0
        A_f = qd.Matrix.zero(qd.f64, 3, 3)
        coupling = qd.Matrix.zero(qd.f64, 3, 3)
        if d > 0.0:
            k = self._contact_stiffness
            lam_n = k * d
            slide = x - self.verts[f, i_v, i_b].pos
            slide[2] = 0.0
            t = self.verts_info[i_v].tangent
            t[2] = 0.0
            t = t.normalized()
            b = qd.Vector([-t[1], t[0], 0.0], dt=gs.qd_float)
            u_t = slide.dot(t)
            u_b = slide.dot(b)
            u_norm = slide.norm()
            eps = self._friction_eps_v * self._substep_dt
            g = 1.0 / u_norm
            dg = -1.0 / (u_norm * u_norm)  # g'(|u|)
            if u_norm < eps:
                g = 2.0 / eps - u_norm / (eps * eps)
                dg = -1.0 / (eps * eps)
            mu_f = self.verts_info[i_v].mu_forward
            mu_bw = self.verts_info[i_v].mu_backward
            th = qd.tanh(u_t / eps)
            mu_ax = 0.5 * (mu_f + mu_bw) + 0.5 * (mu_f - mu_bw) * th
            dmu_ax = 0.5 * (mu_f - mu_bw) * (1.0 - th * th) / eps
            mu_l = self.verts_info[i_v].mu_lateral
            Pu = mu_ax * u_t * t + mu_l * u_b * b
            P = mu_ax * t.outer_product(t) + mu_l * b.outer_product(b)
            # the rank-one term is (g'/|u|) (P u) u^T; at |u| = 0 it vanishes with u, so the guarded divisor is exact
            A = g * P + g * dmu_ax * u_t * t.outer_product(t) + (dg / qd.max(u_norm, 1e-300)) * Pu.outer_product(slide)
            A_f = qd.cast(lam_n, qd.f64) * qd.cast(A, qd.f64)
            e_z = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
            coupling = -qd.cast(k, qd.f64) * qd.cast((g * Pu).outer_product(e_z), qd.f64)
        return lam_n, A_f, coupling

    @qd.func
    def _func_distance_block(self, f, i_c, i_b, s, t):
        """Exact d r_s / d x_t of distance constraint i_c at frame f+1 between its slots s and t (f64), from the
        recorded multiplier and stiffness: sigma_s sigma_t (k n n^T + mult / |e| (I - n n^T)), k only when active."""
        e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
        dist = qd.cast(e.norm(), qd.f64)
        n = qd.cast(e, qd.f64) / dist
        nn = n.outer_product(n)
        m = self.cons_hist[f + 1, i_c, i_b].mult
        blk = (m / dist) * (qd.Matrix.identity(qd.f64, 3) - nn) + self.cons_hist[f + 1, i_c, i_b].k_eff * nn
        sign = 1.0
        if s != t:
            sign = -1.0
        return sign * blk

    @qd.func
    def _func_angle_geometry(self, f, i_c, i_b):
        """(u_hat, v_hat, |u|, |v|, cos) of angle constraint i_c at frame f+1, in f64."""
        vq = self.acons_info[i_c].v
        u = qd.cast(self.verts[f + 1, vq[0], i_b].pos - self.verts[f + 1, vq[1], i_b].pos, qd.f64)
        vv = qd.cast(self.verts[f + 1, vq[2], i_b].pos - self.verts[f + 1, vq[3], i_b].pos, qd.f64)
        lu = u.norm()
        lv = vv.norm()
        u_hat = u / lu
        v_hat = vv / lv
        return u_hat, v_hat, lu, lv, u_hat.dot(v_hat)

    @qd.func
    def _func_angle_slot_grad(self, slot, u_hat, v_hat, lu, lv, cosv):
        """Signed gradient of the cosine with respect to the vertex in `slot` (0: +u, 1: -u, 2: +v, 3: -v)."""
        g = (v_hat - cosv * u_hat) / lu
        if slot >= 2:
            g = (u_hat - cosv * v_hat) / lv
        if slot == 1 or slot == 3:
            g = -g
        return g

    @qd.func
    def _func_angle_block(self, f, i_c, i_b, s, t):
        """Exact d r_s / d x_t of angle constraint i_c at frame f+1: k G_s G_t^T (active only) + mult d^2 cos / dx_s dx_t.
        The cosine's Hessian blocks: d g_u/du = -[u n_u^T + n_u u^T + cos (I - u u^T)] / |u|^2 with n_u = v - cos u,
        d g_u/dv = [(I - v v^T) - u n_v^T] / (|u||v|), and the mirror images; slot signs multiply."""
        u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
        I3 = qd.Matrix.identity(qd.f64, 3)
        nu = v_hat - cosv * u_hat
        nv = u_hat - cosv * v_hat
        H = qd.Matrix.zero(qd.f64, 3, 3)
        if s < 2 and t < 2:
            H = -(u_hat.outer_product(nu) + nu.outer_product(u_hat) + cosv * (I3 - u_hat.outer_product(u_hat))) / (lu * lu)
        elif s >= 2 and t >= 2:
            H = -(v_hat.outer_product(nv) + nv.outer_product(v_hat) + cosv * (I3 - v_hat.outer_product(v_hat))) / (lv * lv)
        elif s < 2:
            H = ((I3 - v_hat.outer_product(v_hat)) - u_hat.outer_product(nv)) / (lu * lv)
        else:
            H = ((I3 - u_hat.outer_product(u_hat)) - v_hat.outer_product(nu)) / (lu * lv)
        sign = 1.0
        if (s % 2) != (t % 2):
            sign = -1.0
        m = self.acons_hist[f + 1, i_c, i_b].mult
        blk = (sign * m) * H
        k_eff = self.acons_hist[f + 1, i_c, i_b].k_eff
        if k_eff != 0.0:
            g_s = self._func_angle_slot_grad(s, u_hat, v_hat, lu, lv, cosv)
            g_t = self._func_angle_slot_grad(t, u_hat, v_hat, lu, lv, cosv)
            blk += k_eff * g_s.outer_product(g_t)
        return blk

    @qd.func
    def _func_diag_block(self, f, i_v, i_b):
        """Exact J_ii = dr_i/dx_i: the forward Hessian with the friction block replaced by its exact derivative and
        the constraints' positive semidefinite proxies replaced by their exact curvature at the recorded multipliers."""
        force_unused, H, K_unused = self._func_vertex_system(f, i_v, i_b)
        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            if self.elems_info[i_e].k_fiber > 0.0:
                w0 = self._func_vertex_weight(self.elems_info[i_e].B_rest, self.ve_role[c])
                _, H_exact, H_psd = self._func_fiber_terms(f + 1, i_e, i_b, w0)
                H += qd.cast(H_exact - H_psd, qd.f64)  # the forward kept only the PSD part
        for c in range(self.vc_offset[i_v], self.vc_offset[i_v + 1]):
            i_c = self.vc_cons[c]
            e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
            dist = e.norm()
            n = qd.cast(e / dist, qd.f64)
            nn = n.outer_product(n)
            mult, _ = self._func_constraint_mult(i_c, i_b, dist)
            if mult != 0.0:
                H -= qd.cast(self.cons[i_c, i_b].k, qd.f64) * nn
            H -= qd.cast(qd.abs(mult) / dist, qd.f64) * (qd.Matrix.identity(qd.f64, 3) - nn)
            s = 0
            if self.vc_side[c] < 0.0:
                s = 1
            for t in qd.static(range(2)):
                if self.cons_info[i_c].v[t] == i_v:
                    H += self._func_distance_block(f, i_c, i_b, s, t)
        for c in range(self.va_offset[i_v], self.va_offset[i_v + 1]):
            i_c = self.va_cons[c]
            slot = self.va_slot[c]
            u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
            mult, _ = self._func_angle_mult(i_c, i_b, qd.cast(cosv, gs.qd_float))
            g = self._func_angle_slot_grad(slot, u_hat, v_hat, lu, lv, cosv)
            if mult != 0.0:
                H -= qd.cast(self.acons[i_c, i_b].k, qd.f64) * g.outer_product(g)
            own = u_hat
            scale = lu
            if slot >= 2:
                own = v_hat
                scale = lv
            H -= (qd.abs(qd.cast(mult, qd.f64)) / (scale * scale)) * (qd.Matrix.identity(qd.f64, 3) - own.outer_product(own))
            vq = self.acons_info[i_c].v
            for t in qd.static(range(4)):
                if vq[t] == i_v:
                    H += self._func_angle_block(f, i_c, i_b, slot, t)
        lam_n, A_f, coupling = self._func_friction_terms(f, i_v, i_b)
        if lam_n > 0.0:
            # remove the forward's symmetric friction approximation (lam_n g P) and add the exact terms
            x = self.verts[f + 1, i_v, i_b].pos
            slide = x - self.verts[f, i_v, i_b].pos
            slide[2] = 0.0
            t = self.verts_info[i_v].tangent
            t[2] = 0.0
            t = t.normalized()
            b = qd.Vector([-t[1], t[0], 0.0], dt=gs.qd_float)
            u_norm = slide.norm()
            eps = self._friction_eps_v * self._substep_dt
            g = 1.0 / u_norm
            if u_norm < eps:
                g = 2.0 / eps - u_norm / (eps * eps)
            th = qd.tanh(slide.dot(t) / eps)
            mu_ax = 0.5 * (self.verts_info[i_v].mu_forward + self.verts_info[i_v].mu_backward) + 0.5 * (
                self.verts_info[i_v].mu_forward - self.verts_info[i_v].mu_backward
            ) * th
            P = mu_ax * t.outer_product(t) + self.verts_info[i_v].mu_lateral * b.outer_product(b)
            H -= qd.cast(lam_n * g, qd.f64) * qd.cast(P, qd.f64)
            H += A_f + coupling
        return qd.cast(H, qd.f64)

    @qd.func
    def _func_offdiag_apply(self, f, i_v, i_b, vec):
        """sum over neighbours j of J_ij vec_j for the full stationarity Jacobian: elastic
        J_ij = V [mu (w_i.w_j) I + lam' q_i q_j^T + lam' (J-alpha) K_ij], K_ij = -[F (w_i x w_j)]_x (skew), plus
        (k_d/h) K0_ij from Rayleigh damping and the hard constraints' coupling."""
        sym, cross = self._func_offdiag_parts(f, i_v, i_b, vec)
        out = sym + cross
        if qd.static(self._damping > 0.0):
            out += (self._damping / self._substep_dt) * self._func_offdiag_rest(i_v, i_b, vec)
        for c in range(self.vc_offset[i_v], self.vc_offset[i_v + 1]):
            i_c = self.vc_cons[c]
            s = 0
            if self.vc_side[c] < 0.0:
                s = 1
            for t in qd.static(range(2)):
                j = self.cons_info[i_c].v[t]
                if j != i_v:
                    out += self._func_distance_block(f, i_c, i_b, s, t) @ vec[j, i_b]
        for c in range(self.va_offset[i_v], self.va_offset[i_v + 1]):
            i_c = self.va_cons[c]
            slot = self.va_slot[c]
            vq = self.acons_info[i_c].v
            for t in qd.static(range(4)):
                if vq[t] != i_v:
                    out += self._func_angle_block(f, i_c, i_b, slot, t) @ vec[vq[t], i_b]
        return out

    @qd.func
    def _func_offdiag_rest(self, i_v, i_b, vec):
        """sum over neighbours j of K0_ij vec_j with K0 the rest Hessian: the damping's off-diagonal action."""
        out = qd.Vector.zero(qd.f64, 3)
        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            role = self.ve_role[c]
            B0 = self.elems_info[i_e].B_rest
            w0 = self._func_vertex_weight(B0, role)
            for r in qd.static(range(4)):
                if r != role:
                    out += qd.cast(self._func_rest_block(i_e, w0, self._func_vertex_weight_static(B0, r)), qd.f64) @ vec[self.elems_info[i_e].v[r], i_b]
        return out

    @qd.func
    def _func_offdiag_parts(self, f, i_v, i_b, vec):
        """(sym, cross): the PSD part and the skew cross-term part of sum_j J_ij vec_j over the neighbours."""
        sym = qd.Vector.zero(qd.f64, 3)
        cross = qd.Vector.zero(qd.f64, 3)
        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            role = self.ve_role[c]
            F, B = self._func_deformation(f + 1, i_e, i_b)
            mu = self.elems_info[i_e].mu
            lam = self.elems_info[i_e].lam
            alpha = 1.0 + mu / lam
            cof = self._func_cofactor(F)
            J = F.determinant()
            V = self.elems_info[i_e].vol_rest
            w_i = self._func_vertex_weight(B, role)
            q_i = cof @ w_i
            for r in qd.static(range(4)):
                if r != role:
                    j = self.elems_info[i_e].v[r]
                    w_j = self._func_vertex_weight_static(B, r)
                    q_j = cof @ w_j
                    vj = qd.cast(vec[j, i_b], gs.qd_float)
                    Kv = -(F @ w_i.cross(w_j)).cross(vj)
                    sym += qd.cast(V * (mu * w_i.dot(w_j) * vj + lam * q_j.dot(vj) * q_i), qd.f64)
                    cross += qd.cast(V * lam * (J - alpha) * Kv, qd.f64)
                    if self.elems_info[i_e].k_fiber > 0.0:
                        # fibre coupling: J_ij = V k (w0_i.a)(w0_j.a) [u u^T + (l-1)/l (I - u u^T)], exact and symmetric;
                        # the PSD part joins the damped block, the compression part the undamped one
                        B0 = self.elems_info[i_e].B_rest
                        a = self.elems_info[i_e].fiber
                        wi0 = self._func_vertex_weight(B0, role)
                        wj0 = self._func_vertex_weight_static(B0, r)
                        v0 = self.elems_info[i_e].v
                        p00 = self.verts[f + 1, v0[0], i_b].pos
                        F0 = qd.Matrix.cols([self.verts[f + 1, v0[1], i_b].pos - p00, self.verts[f + 1, v0[2], i_b].pos - p00, self.verts[f + 1, v0[3], i_b].pos - p00]) @ B0
                        u = F0 @ a
                        l = u.norm()
                        u_hat = u / l
                        cf = V * self.elems_info[i_e].k_fiber * wi0.dot(a) * wj0.dot(a)
                        sym += qd.cast(cf * u_hat.dot(vj) * u_hat, qd.f64)
                        cross += qd.cast(cf * ((l - 1.0) / l) * (vj - u_hat.dot(vj) * u_hat), qd.f64)
        return sym, cross

    @qd.kernel
    def _kernel_apply_jacobian(self, f: qd.i32, p: qd.types.ndarray(), out: qd.types.ndarray()):
        """out = J p for the stationarity Jacobian of substep `f`; p and out have shape (B, n_vertices, 3)."""
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for j in qd.static(range(3)):
                self.z[i_v, i_b][j] = p[i_b, i_v, j]
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            r = self._func_diag_block(f, i_v, i_b) @ self.z[i_v, i_b] + self._func_offdiag_apply(f, i_v, i_b, self.z)
            for j in qd.static(range(3)):
                out[i_b, i_v, j] = r[j]

    # With active hard constraints the substep is the KKT point r(x) + G^T mu = 0, C(x) = 0 (G = dC/dx, mu the
    # recorded multipliers). The adjoint is the saddle system J^T z + G^T zeta = gbar, G z = 0, solved like the
    # forward: the k G^T G part of J is the augmentation, zeta += w k (G z) after every sweep drives G z to zero.
    @qd.func
    def _func_constraint_dot_z(self, f, i_c, i_b):
        va = self.cons_info[i_c].v[0]
        vb = self.cons_info[i_c].v[1]
        e = self.verts[f + 1, va, i_b].pos - self.verts[f + 1, vb, i_b].pos
        n = qd.cast(e / e.norm(), qd.f64)
        return n.dot(self.z[va, i_b] - self.z[vb, i_b])

    @qd.func
    def _func_angle_dot_z(self, f, i_c, i_b):
        u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
        vq = self.acons_info[i_c].v
        out = 0.0
        for slot in qd.static(range(4)):
            out += self._func_angle_slot_grad(slot, u_hat, v_hat, lu, lv, cosv).dot(self.z[vq[slot], i_b])
        return out

    @qd.func
    def _func_zeta_force(self, f, i_v, i_b):
        """sum over the active constraints of vertex i_v of zeta_c G_c,i: the dual adjoint's term of the saddle system."""
        out = qd.Vector.zero(qd.f64, 3)
        for c in range(self.vc_offset[i_v], self.vc_offset[i_v + 1]):
            i_c = self.vc_cons[c]
            if self.cons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
                out += (self.cons_zeta[i_c, i_b] * qd.cast(self.vc_side[c], qd.f64)) * qd.cast(e / e.norm(), qd.f64)
        for c in range(self.va_offset[i_v], self.va_offset[i_v + 1]):
            i_c = self.va_cons[c]
            if self.acons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
                out += self.acons_zeta[i_c, i_b] * self._func_angle_slot_grad(self.va_slot[c], u_hat, v_hat, lu, lv, cosv)
        return out

    # ------------------------------------------------------------------------------------
    # ------------------------ solver-level (reverse sweep) adjoint ----------------------
    # ------------------------------------------------------------------------------------
    # The forward is a composition of block updates, so its Jacobian transpose is the same blocks in
    # reverse order (Shu et al. 2026). Each block contributes a local solve H_i^T p = xbar_i, an identity
    # branch, and a scatter of two terms to its neighbours: the gradient tangent -(dg_i/dx_j)^T p, and the
    # Hessian tangent -(dH_i/dx_j) : (p dx^T), which for this material reduces to the same skew matrix the
    # off-diagonal elastic block already uses. Nothing global is assembled and the forward need not converge.

    @qd.func
    def _func_scatter_reverse(self, f, i_v, i_b, p, dx):
        """Scatter one block's two tangents to its neighbours. Elastic terms only for now."""
        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            role = self.ve_role[c]
            F, B = self._func_deformation(f + 1, i_e, i_b)
            mu = self.elems_info[i_e].mu
            lam = self.elems_info[i_e].lam
            alpha = 1.0 + mu / lam
            cof = self._func_cofactor(F)
            J = F.determinant()
            V = self.elems_info[i_e].vol_rest
            w_i = self._func_vertex_weight(B, role)
            q_i = qd.cast(cof @ w_i, qd.f64)
            if self.elems_info[i_e].k_fiber > 0.0:
                # the fibre block is c u u^T with u the normalised fibre direction, so it depends on the state and
                # its tangent does not vanish on the diagonal the way the elastic one does
                B0f = self.elems_info[i_e].B_rest
                af = self.elems_info[i_e].fiber
                w0_i = self._func_vertex_weight(B0f, role)
                vf = self.elems_info[i_e].v
                p0f = self.verts[f + 1, vf[0], i_b].pos
                F0f = qd.Matrix.cols([self.verts[f + 1, vf[1], i_b].pos - p0f, self.verts[f + 1, vf[2], i_b].pos - p0f, self.verts[f + 1, vf[3], i_b].pos - p0f]) @ B0f
                uf = F0f @ af
                lf = uf.norm()
                uh = qd.cast(uf / lf, qd.f64)
                cf_i = qd.cast(self.elems_info[i_e].vol_rest * self.elems_info[i_e].k_fiber * w0_i.dot(af) ** 2, qd.f64)
                perp_p = p - uh.dot(p) * uh
                perp_dx = dx - uh.dot(dx) * uh
                base = cf_i / qd.cast(lf, qd.f64) * (uh.dot(dx) * perp_p + uh.dot(p) * perp_dx)
                for r in qd.static(range(4)):
                    scale = qd.cast(self._func_vertex_weight_static(B0f, r).dot(af), qd.f64)
                    jf = self.elems_info[i_e].v[r]
                    for d in qd.static(range(3)):
                        qd.atomic_add(self.xb[jf, i_b][d], -scale * base[d])
            for r in qd.static(range(4)):
                if r != role:
                    j = self.elems_info[i_e].v[r]
                    w_j = self._func_vertex_weight_static(B, r)
                    q_j = qd.cast(cof @ w_j, qd.f64)
                    a = qd.cast(F @ w_i.cross(w_j), qd.f64)  # K_ij = -[a]_x, so K_ij^T v = a x v
                    Vd = qd.cast(V, qd.f64)
                    lamd = qd.cast(lam, qd.f64)
                    # gradient tangent: (dg_i/dx_j)^T p
                    grad_t = Vd * (qd.cast(mu * w_i.dot(w_j), qd.f64) * p + lamd * q_i.dot(p) * q_j + lamd * qd.cast(J - alpha, qd.f64) * a.cross(p))
                    # Hessian tangent: only q_i depends on the state, and it vanishes on the diagonal
                    hess_t = Vd * lamd * (q_i.dot(dx) * a.cross(p) + q_i.dot(p) * a.cross(dx))
                    if self.elems_info[i_e].k_fiber > 0.0:
                        # the fibre off-diagonal is symmetric, so its transpose is itself
                        B0 = self.elems_info[i_e].B_rest
                        aa = self.elems_info[i_e].fiber
                        wi0 = self._func_vertex_weight(B0, role)
                        wj0 = self._func_vertex_weight_static(B0, r)
                        v0 = self.elems_info[i_e].v
                        p00 = self.verts[f + 1, v0[0], i_b].pos
                        F0 = qd.Matrix.cols([self.verts[f + 1, v0[1], i_b].pos - p00, self.verts[f + 1, v0[2], i_b].pos - p00, self.verts[f + 1, v0[3], i_b].pos - p00]) @ B0
                        u = F0 @ aa
                        l = u.norm()
                        u_hat = qd.cast(u / l, qd.f64)
                        cf = qd.cast(V * self.elems_info[i_e].k_fiber * wi0.dot(aa) * wj0.dot(aa), qd.f64)
                        grad_t += cf * (u_hat.dot(p) * u_hat + qd.cast((l - 1.0) / l, qd.f64) * (p - u_hat.dot(p) * u_hat))
                    if qd.static(self._damping > 0.0):
                        # Rayleigh damping uses the constant rest Hessian, whose transpose is the swap of its arguments
                        B0 = self.elems_info[i_e].B_rest
                        w0_i = self._func_vertex_weight(B0, role)
                        w0_j = self._func_vertex_weight_static(B0, r)
                        kd_h = qd.cast(self._damping / self._substep_dt, qd.f64)
                        damp_t = kd_h * (qd.cast(self._func_rest_block(i_e, w0_j, w0_i), qd.f64) @ p)
                        grad_t += damp_t
                        for d in qd.static(range(3)):  # the same block is how the previous position enters
                            qd.atomic_add(self.adj[f, j, i_b].pos[d], damp_t[d])
                    for d in qd.static(range(3)):
                        qd.atomic_add(self.xb[j, i_b][d], -grad_t[d] - hess_t[d])

    @qd.func
    def _func_friction_tangent(self, f, i_v, i_b, p, dx):
        """The friction block is lam_n g P, and all three factors move with the state: the normal force with the
        penetration depth, g with the sliding speed, and the forward-backward blend with the tanh. The tangent is the
        gradient of the scalar p^T H dx, so it is three terms. Returns its parts for the current and the previous
        position; the previous one carries the opposite sign, because the slide is their difference."""
        tangent = qd.Vector.zero(qd.f64, 3)
        tangent_prev = qd.Vector.zero(qd.f64, 3)
        x = self.verts[f + 1, i_v, i_b].pos
        d = self._floor_height - x[2]
        if d > 0.0:
            k = self._contact_stiffness
            lam_n = k * d
            slide = x - self.verts[f, i_v, i_b].pos
            slide[2] = 0.0
            t = self.verts_info[i_v].tangent
            t[2] = 0.0
            t = t.normalized()
            b = qd.Vector([-t[1], t[0], 0.0], dt=gs.qd_float)
            u_norm = slide.norm()
            eps = self._friction_eps_v * self._substep_dt
            g = 1.0 / u_norm
            dg = -1.0 / (u_norm * u_norm)
            if u_norm < eps:
                g = 2.0 / eps - u_norm / (eps * eps)
                dg = -1.0 / (eps * eps)
            mu_f = self.verts_info[i_v].mu_forward
            mu_bw = self.verts_info[i_v].mu_backward
            th = qd.tanh(slide.dot(t) / eps)
            mu_ax = 0.5 * (mu_f + mu_bw) + 0.5 * (mu_f - mu_bw) * th
            dmu_ax = 0.5 * (mu_f - mu_bw) * (1.0 - th * th) / eps
            td = qd.cast(t, qd.f64)
            bd = qd.cast(b, qd.f64)
            A = qd.cast(mu_ax, qd.f64) * td.dot(p) * td.dot(dx) + qd.cast(self.verts_info[i_v].mu_lateral, qd.f64) * bd.dot(p) * bd.dot(dx)
            e_z = qd.Vector([0.0, 0.0, 1.0], dt=qd.f64)
            by_depth = -qd.cast(k, qd.f64) * qd.cast(g, qd.f64) * A * e_z
            by_slide = qd.cast(lam_n, qd.f64) * A * qd.cast(dg, qd.f64) / qd.max(qd.cast(u_norm, qd.f64), 1e-300) * qd.cast(slide, qd.f64)
            by_blend = qd.cast(lam_n * g * dmu_ax, qd.f64) * td.dot(p) * td.dot(dx) * td
            tangent = by_depth + by_slide + by_blend
            tangent_prev = -(by_slide + by_blend)
        return tangent, tangent_prev

    @qd.kernel
    def _kernel_reverse_init(self, f: qd.i32):
        """The adjoint arriving at the end of the substep, and a clean accumulator for the predictor."""
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.xb[i_v, i_b] = self.adj[f + 1, i_v, i_b].pos + self.adj[f + 1, i_v, i_b].vel / self._substep_dt
            self.yb[i_v, i_b] = qd.Vector.zero(qd.f64, 3)

    @qd.kernel
    def _kernel_reverse_color(self, f: qd.i32, sweep: qd.i32, lo: qd.i32, hi: qd.i32):
        """One colour of one sweep, in reverse. The positions must already be rolled back to this colour's
        linearisation point."""
        for k, i_b in qd.ndrange((lo, hi), self._B):
            i_v = self.color_perm[k]
            xbar = self.xb[i_v, i_b]
            force_unused, H, K_unused = self._func_vertex_system(f, i_v, i_b)
            p = qd.cast(H, qd.f64).transpose().inverse() @ xbar
            dx = self.sweep_dx[f, sweep, i_v, i_b]
            # the block's own branch: identity minus the exact local Jacobian, which cancels exactly when the
            # solver's block is the true local Hessian, as it is for the elastic terms
            self.xb[i_v, i_b] = xbar - self._func_diag_block(f, i_v, i_b).transpose() @ p
            # the predictor enters every block through the inertia term, dg_i/dy = -m/h^2
            self.yb[i_v, i_b] += qd.cast(self.verts_info[i_v].mass / (self._substep_dt * self._substep_dt), qd.f64) * p
            # friction and damping read the previous position, so every block scatters to it as well
            lam_n, A_f, coupling_unused = self._func_friction_terms(f, i_v, i_b)
            if lam_n > 0.0:
                self.adj[f, i_v, i_b].pos += A_f.transpose() @ p
                fr_t, fr_prev = self._func_friction_tangent(f, i_v, i_b, p, dx)
                self.xb[i_v, i_b] -= fr_t
                self.adj[f, i_v, i_b].pos -= fr_prev
            if qd.static(self._damping > 0.0):
                force_u, H_u, K0_ii = self._func_vertex_system(f, i_v, i_b)
                self.adj[f, i_v, i_b].pos += qd.cast(self._damping / self._substep_dt, qd.f64) * (qd.cast(K0_ii, qd.f64) @ p)
            self._func_scatter_reverse(f, i_v, i_b, p, dx)

    @qd.kernel
    def _kernel_reverse_finish(self, f: qd.i32):
        """y = x^t + h (v^t + h g), so the adjoint of the predictor reaches both the position and the velocity."""
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            total = self.xb[i_v, i_b] + self.yb[i_v, i_b]
            self.adj[f, i_v, i_b].pos += total - self.adj[f + 1, i_v, i_b].vel / self._substep_dt
            self.adj[f, i_v, i_b].vel += total * self._substep_dt

    def substep_pre_coupling_grad_sweep(self, f):
        """The solver-level adjoint of substep `f`: the sweeps and colours of the forward, in reverse."""
        self._kernel_reverse_init(f)
        for sweep in reversed(range(self._n_iterations)):
            for c in reversed(range(self._n_colors)):
                lo, hi = self._color_offsets[c], self._color_offsets[c + 1]
                self._kernel_undo_sweep(f, sweep, lo, hi)
                self._kernel_reverse_color(f, sweep, lo, hi)
        self._kernel_reverse_finish(f)

    @qd.kernel
    def _kernel_adjoint_rhs(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.gbar[i_v, i_b] = self.adj[f + 1, i_v, i_b].pos + self.adj[f + 1, i_v, i_b].vel / self._substep_dt
            self.z[i_v, i_b] = qd.Vector.zero(qd.f64, 3)

    @qd.kernel
    def _kernel_adjoint_sweeps(self, f: qd.i32):
        """Colored Gauss-Seidel on J^T z + G^T zeta = gbar: z_i = (J_ii^T)^-1 (gbar_i - sum_j J_ij z_j - (G^T zeta)_i),
        then the dual update zeta += w k (G z) per active constraint."""
        for _ in qd.static(range(self._n_iterations)):
            for c in qd.static(range(self._n_colors)):
                for k, i_b in qd.ndrange((self._color_offsets[c], self._color_offsets[c + 1]), self._B):
                    i_v = self.color_perm[k]
                    rhs = self.gbar[i_v, i_b] - self._func_offdiag_apply(f, i_v, i_b, self.z) - self._func_zeta_force(f, i_v, i_b)
                    self.z[i_v, i_b] = self._func_diag_block(f, i_v, i_b).transpose().inverse() @ rhs
            if qd.static(self._n_constraints > 0):
                for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
                    if self.cons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                        self.cons_zeta[i_c, i_b] += self._constraint_dual_relaxation * self.cons_hist[f + 1, i_c, i_b].k_eff * self._func_constraint_dot_z(f, i_c, i_b)
            if qd.static(self._n_angle_constraints > 0):
                for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
                    if self.acons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                        self.acons_zeta[i_c, i_b] += self._constraint_dual_relaxation * self.acons_hist[f + 1, i_c, i_b].k_eff * self._func_angle_dot_z(f, i_c, i_b)

    @qd.kernel
    def _kernel_adjoint_residual(self, f: qd.i32):
        self.adj_residual[None] = 0.0
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            r = self.gbar[i_v, i_b] - self._func_diag_block(f, i_v, i_b).transpose() @ self.z[i_v, i_b]
            r -= self._func_offdiag_apply(f, i_v, i_b, self.z) + self._func_zeta_force(f, i_v, i_b)
            qd.atomic_max(self.adj_residual[None], qd.abs(r).max())
            qd.atomic_max(self.residual[None], qd.abs(self.gbar[i_v, i_b]).max())
        # A constraint row's error must be measured where it acts: the force `k_eff (G . z) G` that the missing dual
        # adjoint would add to the vertex rows. For a distance |G| is 1 and nothing changes; for an angle |G| is about
        # 1/|u|, and without it a joint limit on a 1.2 cm lever is measured 100 times too small and the adjoint stops
        # while G z is still large on exactly the constraints the skeleton exists for.
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            if self.cons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                qd.atomic_max(self.adj_residual[None], self.cons_hist[f + 1, i_c, i_b].k_eff * qd.abs(self._func_constraint_dot_z(f, i_c, i_b)))
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            if self.acons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
                g_max = 0.0
                for slot in qd.static(range(4)):
                    g_max = qd.max(g_max, self._func_angle_slot_grad(slot, u_hat, v_hat, lu, lv, cosv).norm())
                qd.atomic_max(self.adj_residual[None], self.acons_hist[f + 1, i_c, i_b].k_eff * g_max * qd.abs(self._func_angle_dot_z(f, i_c, i_b)))

    @qd.kernel
    def _kernel_adjoint_accumulate(self, f: qd.i32):
        inv_h = 1.0 / self._substep_dt
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            z = self.z[i_v, i_b]
            m_h = qd.cast(self.verts_info[i_v].mass * inv_h, qd.f64)
            self.adj[f, i_v, i_b].pos += m_h * inv_h * z - self.adj[f + 1, i_v, i_b].vel * inv_h
            self.adj[f, i_v, i_b].vel += m_h * z
            lam_n, A_f, _ = self._func_friction_terms(f, i_v, i_b)
            if lam_n > 0.0:
                self.adj[f, i_v, i_b].pos += A_f.transpose() @ z  # dr/dx^t = -A_f
            if qd.static(self._damping > 0.0):
                # the damping residual +(k_d/h) sum_j K0_ij (x_j - x_j^t) has dr/dx^t = -(k_d/h) K0 (constant, symmetric),
                # so dL/dx^t -= (dr/dx^t)^T z gives +(k_d/h) (K0 z)_i over the diagonal block and the neighbours
                force_unused, H_unused, K0_ii = self._func_vertex_system(f, i_v, i_b)
                kd_h = self._damping * inv_h
                self.adj[f, i_v, i_b].pos += kd_h * (qd.cast(K0_ii, qd.f64) @ z + self._func_offdiag_rest(i_v, i_b, self.z))
        for i_e, i_b in qd.ndrange(self._n_elements, self._B):
            group = self.elems_info[i_e].group
            if group >= 0:
                s_ = 1.0 - self.muscle_actu[group, i_b] * self.elems_info[i_e].gain
                m = self.elems_info[i_e].fiber
                mmT = m.outer_product(m)
                I3 = qd.Matrix.identity(gs.qd_float, 3)
                A_dot = self.elems_info[i_e].gain * ((1.0 / (s_ * s_)) * mmT - (0.5 / qd.sqrt(s_)) * (I3 - mmT))
                F, B = self._func_deformation(f + 1, i_e, i_b)
                B0 = self.elems_info[i_e].B_rest
                A = (1.0 / s_) * mmT + qd.sqrt(s_) * (I3 - mmT)
                F0 = F @ A.inverse()
                F_dot = F0 @ A_dot
                mu = self.elems_info[i_e].mu
                lam = self.elems_info[i_e].lam
                alpha = 1.0 + mu / lam
                cof = self._func_cofactor(F)
                J = F.determinant()
                P = mu * F + lam * (J - alpha) * cof
                dcof = qd.Matrix.cols(
                    [
                        F_dot[:, 1].cross(F[:, 2]) + F[:, 1].cross(F_dot[:, 2]),
                        F_dot[:, 2].cross(F[:, 0]) + F[:, 2].cross(F_dot[:, 0]),
                        F_dot[:, 0].cross(F[:, 1]) + F[:, 0].cross(F_dot[:, 1]),
                    ]
                )
                P_dot = mu * F_dot + lam * (cof * F_dot).sum() * cof + lam * (J - alpha) * dcof
                V = self.elems_info[i_e].vol_rest
                da = 0.0
                for r in qd.static(range(4)):
                    w0 = self._func_vertex_weight_static(B0, r)
                    w = self._func_vertex_weight_static(B, r)
                    dr = V * (P_dot @ w + P @ (A_dot @ w0))
                    da -= qd.cast(self.z[self.elems_info[i_e].v[r], i_b], gs.qd_float).dot(dr)
                self.muscle_actu_adj[group, i_b] += qd.cast(da, qd.f64)

    def substep_pre_coupling_grad(self, f):
        if not self.is_active:
            return
        self.cons_zeta.fill(0.0)
        self.acons_zeta.fill(0.0)
        self._kernel_adjoint_rhs(f)
        self.residual[None] = 0.0
        self._kernel_adjoint_residual(f)
        if self.residual[None] == 0.0:  # nothing flows back into this substep
            return
        for _ in range(self._max_sweeps // self._n_iterations):
            self._kernel_adjoint_sweeps(f)
            self._kernel_adjoint_residual(f)
            if self.adj_residual[None] <= self._residual_tol * self.residual[None]:  # residual holds max |gbar|
                break
        else:
            gs.raise_exception(
                f"VBD adjoint did not converge: residual {self.adj_residual[None]:.3e} after {self._max_sweeps} sweeps."
            )
        self._kernel_adjoint_accumulate(f)

    @qd.kernel
    def _kernel_compute_energy(self, f: qd.i32):
        """Incremental potential of substep `f` per env: inertia term plus the stable neo-Hookean energy of every tet,
        evaluated at frame `f+1`."""
        inv_h2 = 1.0 / (self._substep_dt * self._substep_dt)
        for i_b in range(self._B):
            self.energy[i_b] = 0.0
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            d = self.verts[f + 1, i_v, i_b].pos - self._func_inertia_target(f, i_v, i_b)
            self.energy[i_b] += qd.cast(0.5 * self.verts_info[i_v].mass * inv_h2 * d.norm_sqr(), qd.f64)
        for i_e, i_b in qd.ndrange(self._n_elements, self._B):
            F, _ = self._func_deformation(f + 1, i_e, i_b)
            mu = self.elems_info[i_e].mu
            lam = self.elems_info[i_e].lam
            alpha = 1.0 + mu / lam
            J = F.determinant()
            psi = 0.5 * (mu * (F.norm_sqr() - 3.0) + lam * (J - alpha) ** 2)
            if self.elems_info[i_e].k_fiber > 0.0:
                v = self.elems_info[i_e].v
                p0 = self.verts[f + 1, v[0], i_b].pos
                Ds = qd.Matrix.cols([self.verts[f + 1, v[1], i_b].pos - p0, self.verts[f + 1, v[2], i_b].pos - p0, self.verts[f + 1, v[3], i_b].pos - p0])
                l = (Ds @ self.elems_info[i_e].B_rest @ self.elems_info[i_e].fiber).norm()
                psi += 0.5 * self.elems_info[i_e].k_fiber * (l - 1.0) ** 2
            self.energy[i_b] += qd.cast(self.elems_info[i_e].vol_rest * psi, qd.f64)

    @qd.kernel
    def _kernel_constraint_energy(self, f: qd.i32):
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
            dist = e.norm()
            k = self.cons[i_c, i_b].k
            if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
                c_hi = qd.max(dist - self.cons_info[i_c].hi, 0.0)
                c_lo = qd.min(dist - self.cons_info[i_c].lo, 0.0)
                self.energy[i_b] += qd.cast(0.5 * k * (c_hi * c_hi + c_lo * c_lo) + self.cons[i_c, i_b].lam_hi * c_hi + self.cons[i_c, i_b].lam_lo * c_lo, qd.f64)
            else:
                C = dist - self.cons_info[i_c].hi
                self.energy[i_b] += qd.cast(0.5 * k * C * C + self.cons[i_c, i_b].lam_hi * C, qd.f64)

    def compute_energy(self, f):
        """Incremental potential of substep `f` at the current iterate, shape (B,). Non-increasing across sweeps."""
        self._kernel_compute_energy(f)
        if self._n_constraints > 0:
            self._kernel_constraint_energy(f)
        return self.energy.to_numpy()

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        if self.is_active:
            # every substep, gradients or not: at two sweeps the primal is never converged within a substep, and this
            # decay is the dual damping that stops the multipliers integrating stale violations (once per step, as the
            # AVBD paper does per frame, the ladder python's spine stretch went from 0.2 to 8 percent)
            if self._n_constraints > 0 or self._n_angle_constraints > 0:
                self._kernel_warm_start()
            self._kernel_predict(f)
            self.solve(f)
            self._kernel_update_velocity(f)

    def substep_post_coupling(self, f):
        pass

    def substep_post_coupling_grad(self, f):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def reset_grad(self):
        self.adj.fill(0.0)
        self.muscle_actu_adj.fill(0.0)
        self._cons_window_start = self._snapshot_cons()  # a new rollout's first window starts from the live multipliers
        for entity in self._entities:
            entity.reset_grad()

    def collect_output_grads(self):
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active:
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self._kernel_add_state_pos_grad(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self._kernel_add_state_vel_grad(self._sim.cur_substep_local, state.vel.grad)

    @qd.kernel
    def _kernel_add_state_pos_grad(self, f: qd.i32, pos_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for j in qd.static(range(3)):
                self.adj[f, i_v, i_b].pos[j] += qd.cast(pos_grad[i_b, i_v, j], qd.f64)

    @qd.kernel
    def _kernel_add_state_vel_grad(self, f: qd.i32, vel_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for j in qd.static(range(3)):
                self.adj[f, i_v, i_b].vel[j] += qd.cast(vel_grad[i_b, i_v, j], qd.f64)

    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.verts[target, i_v, i_b] = self.verts[source, i_v, i_b]

    @qd.kernel
    def copy_adj(self, source: qd.i32, target: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.adj[target, i_v, i_b] = self.adj[source, i_v, i_b]

    @qd.kernel
    def reset_adj_till_frame(self, f: qd.i32):
        # Zero out the vertex adjoint in frames [0, f-1], for all vertices, all batch indices.
        for i_f, i_v, i_b in qd.ndrange(f, self._n_vertices, self._B):
            self.adj[i_f, i_v, i_b].pos = qd.Vector.zero(qd.f64, 3)
            self.adj[i_f, i_v, i_b].vel = qd.Vector.zero(qd.f64, 3)

    def save_ckpt(self, ckpt_name):
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros((self._B, self._n_vertices, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["vel"] = torch.zeros((self._B, self._n_vertices, 3), dtype=gs.tc_float)

            self._kernel_get_state(0, self._ckpt[ckpt_name]["pos"], self._ckpt[ckpt_name]["vel"])
            # frame 0 still holds the state this window started from; the multipliers it started from are the ones
            # snapshotted when the previous window ended (the live values are this window's end)
            self._ckpt[ckpt_name]["cons"], self._ckpt[ckpt_name]["acons"] = self._cons_window_start
            self._cons_window_start = self._snapshot_cons()

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

        # The last frame of this window becomes frame 0 of the next.
        self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_adj(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_adj_till_frame(self._sim.substeps_local)

            self._kernel_set_state(0, self._ckpt[ckpt_name]["pos"], self._ckpt[ckpt_name]["vel"])
            for fld, arr in zip((self.cons.lam_hi, self.cons.lam_lo, self.cons.k), self._ckpt[ckpt_name]["cons"]):
                fld.from_numpy(arr)
            for fld, arr in zip((self.acons.lam_hi, self.acons.lam_lo, self.acons.k), self._ckpt[ckpt_name]["acons"]):
                fld.from_numpy(arr)

            for entity in self._entities:
                entity.load_ckpt(ckpt_name)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_set_state(self, f: qd.i32, pos: qd.types.ndarray(), vel: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for j in qd.static(range(3)):
                self.verts[f, i_v, i_b].pos[j] = pos[i_b, i_v, j]
                self.verts[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]

    @qd.kernel
    def _kernel_get_state(self, f: qd.i32, pos: qd.types.ndarray(), vel: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self.verts[f, i_v, i_b].pos[j]
                vel[i_b, i_v, j] = self.verts[f, i_v, i_b].vel[j]

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel)

    def get_state(self, f):
        if not self.is_active:
            return None
        state = VBDSolverState(self._scene)
        self._kernel_get_state(f, state.pos, state.vel)
        return state

    @qd.kernel
    def _kernel_get_state_render(self, f: qd.i32):
        for i_vv, i_b in qd.ndrange(self._n_vverts, self._B):
            i_v = self.vverts_info[i_vv].vert_idx
            for j in qd.static(range(3)):
                pos_j = qd.cast(self.verts[f, i_v, i_b].pos[j], qd.f32)
                self.vverts_render[i_vv, i_b].pos[j] = pos_j + self.envs_offset[i_b][j]

    def get_state_render(self, f):
        """Same contract as `FEMSolver.get_state_render`: (vverts_pos, vverts_uvs, vfaces_indices)."""
        if not self.is_active or self._n_vverts == 0:
            return None, None, None
        self._kernel_get_state_render(f)
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_vertices(self):
        return sum(entity.n_vertices for entity in self._entities)

    @property
    def n_elements(self):
        return sum(entity.n_elements for entity in self._entities)

    @property
    def n_vverts(self):
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def n_colors(self):
        return self._n_colors

    @property
    def n_constraints(self):
        return self._n_constraints

    @property
    def n_angle_constraints(self):
        return self._n_angle_constraints

    @property
    def color_offsets(self):
        return self._color_offsets
