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
        self._max_sweeps = options.max_sweeps

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
        self.gbar = qd.Vector.field(3, dtype=qd.f64, shape=(self._n_vertices, self._B))  # its right-hand side
        self.adj_residual = qd.field(dtype=qd.f64, shape=())

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

    def _compute_vertex_coloring_and_incidence(self, elems):
        """Greedy vertex coloring of the tet adjacency graph plus the vertex -> incident tet CSR list.

        Returns (perm, color_offsets, n_colors, ve_offset, ve_elem, ve_role): vertices sorted by color
        (`perm[color_offsets[c]:color_offsets[c+1]]` is color `c`), and for vertex `i` the incident tets
        `ve_elem[ve_offset[i]:ve_offset[i+1]]` with `ve_role` the local index of `i` in each tet.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(self._n_vertices))
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            graph.add_edges_from(zip(elems[:, a].tolist(), elems[:, b].tolist()))
        coloring = nx.greedy_color(graph, strategy="smallest_last")
        color = np.array([coloring[i] for i in range(self._n_vertices)], dtype=np.int64)
        n_colors = int(color.max()) + 1
        perm = np.argsort(color, kind="stable")
        color_offsets = np.searchsorted(color[perm], np.arange(n_colors + 1)).tolist()

        inc_vert = elems.reshape(-1)
        inc_elem = np.repeat(np.arange(self._n_elements), 4)
        inc_role = np.tile(np.arange(4), self._n_elements)
        order = np.argsort(inc_vert, kind="stable")
        ve_offset = np.searchsorted(inc_vert[order], np.arange(self._n_vertices + 1))
        return perm, color_offsets, n_colors, ve_offset, inc_elem[order], inc_role[order]

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
            self.reset_grad()

            for entity in self._entities:
                entity._add_to_solver()

            elems = np.concatenate([entity._v_start + entity.elems for entity in self._entities]).astype(np.int64)
            perm, self._color_offsets, self._n_colors, ve_offset, ve_elem, ve_role = (
                self._compute_vertex_coloring_and_incidence(elems)
            )
            self.color_perm = qd.field(dtype=gs.qd_int, shape=(self._n_vertices,))
            self.color_perm.from_numpy(perm.astype(gs.np_int))
            self.ve_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
            self.ve_offset.from_numpy(ve_offset.astype(gs.np_int))
            self.ve_elem = qd.field(dtype=gs.qd_int, shape=(len(ve_elem),))
            self.ve_elem.from_numpy(ve_elem.astype(gs.np_int))
            self.ve_role = qd.field(dtype=gs.qd_int, shape=(len(ve_role),))
            self.ve_role.from_numpy(ve_role.astype(gs.np_int))

    def init_ckpt(self):
        self._ckpt = dict()

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
        """Negative gradient `force` and Hessian `H` of the incremental potential of substep `f` with respect to
        vertex `i_v`, evaluated at the current iterate `verts[f+1].pos` with every other vertex fixed."""
        inv_h2 = 1.0 / (self._substep_dt * self._substep_dt)
        m_h2 = qd.cast(self.verts_info[i_v].mass * inv_h2, self._acc)
        x = self.verts[f + 1, i_v, i_b].pos
        force = -m_h2 * qd.cast(x - self._func_inertia_target(f, i_v, i_b), self._acc)
        K = qd.Matrix.zero(self._acc, 3, 3)  # elastic Hessian block, also the Rayleigh damping matrix
        # Rayleigh damping acts on the strain rate: force -(k_d/h) sum_j K_ij (x_j - x_j^t) over the vertex itself and
        # its neighbours, so a rigid motion is not damped. (The VBD paper's Eq. 11 keeps only the diagonal block,
        # which drags every vertex against the floor frame and freezes a body that has to travel.)
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
                for r in qd.static(range(4)):
                    if r != role:
                        j = self.elems_info[i_e].v[r]
                        w_j = self._func_vertex_weight_static(B, r)
                        q_j = cof @ w_j
                        d_j = self.verts[f + 1, j, i_b].pos - self.verts[f, j, i_b].pos
                        # only the positive semidefinite part of the elastic Hessian (the (J - alpha) K_ij cross term is
                        # indefinite under deformation and would let damping inject energy)
                        damp += qd.cast(V * (mu * w.dot(w_j) * d_j + lam * q_j.dot(d_j) * (cof @ w)), self._acc)

        kd_h = qd.cast(self._damping / self._substep_dt, self._acc)
        force -= kd_h * (K @ qd.cast(x - self.verts[f, i_v, i_b].pos, self._acc) + damp)
        H = m_h2 * qd.Matrix.identity(self._acc, 3) + (1.0 + kd_h) * K

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
        return force, H

    @qd.func
    def _func_solve_vertex(self, f, i_v, i_b):
        force, H = self._func_vertex_system(f, i_v, i_b)
        self.verts[f + 1, i_v, i_b].pos += qd.cast(H.inverse() @ force, gs.qd_float)

    @qd.kernel
    def _kernel_predict(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.verts[f + 1, i_v, i_b].pos = self._func_inertia_target(f, i_v, i_b)

    @qd.kernel
    def _kernel_solve_color(self, f: qd.i32, lo: qd.i32, hi: qd.i32):
        """One color of one sweep. Kept for tests that watch the energy sweep by sweep."""
        for k, i_b in qd.ndrange((lo, hi), self._B):
            self._func_solve_vertex(f, self.color_perm[k], i_b)

    @qd.kernel
    def _kernel_sweeps(self, f: qd.i32):
        """`n_iterations` Gauss-Seidel sweeps in one launch. Each top-level loop is a serial task with an implicit
        barrier after it, so the statically unrolled color loops are race-free without a Python round trip."""
        for _ in qd.static(range(self._n_iterations)):
            for c in qd.static(range(self._n_colors)):
                for k, i_b in qd.ndrange((self._color_offsets[c], self._color_offsets[c + 1]), self._B):
                    self._func_solve_vertex(f, self.color_perm[k], i_b)

    @qd.kernel
    def _kernel_update_velocity(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.verts[f + 1, i_v, i_b].vel = (self.verts[f + 1, i_v, i_b].pos - self.verts[f, i_v, i_b].pos) / self._substep_dt

    @qd.kernel
    def _kernel_residual(self, f: qd.i32):
        """Largest force component left on any vertex of any env: the stationarity residual of substep `f`."""
        self.residual[None] = 0.0
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            force, _ = self._func_vertex_system(f, i_v, i_b)
            qd.atomic_max(self.residual[None], qd.cast(qd.abs(force).max(), qd.f64))

    @qd.kernel
    def _kernel_residual_vector(self, f: qd.i32, out: qd.types.ndarray()):
        """r_i = -force_i of substep `f` at the current iterate, shape (B, n_vertices, 3). Test hook."""
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            force, _ = self._func_vertex_system(f, i_v, i_b)
            for j in qd.static(range(3)):
                out[i_b, i_v, j] = -force[j]

    def solve(self, f):
        """Fixed sweeps normally; under requires_grad, sweep until the residual is below tolerance, since the
        adjoint differentiates the converged stationarity condition and inherits any leftover residual as bias."""
        self._kernel_sweeps(f)
        if self._sim.requires_grad:
            for _ in range(self._max_sweeps // self._n_iterations):
                self._kernel_residual(f)
                if self.residual[None] < self._residual_tol:
                    return
                self._kernel_sweeps(f)
            self._kernel_residual(f)
            if self.residual[None] >= self._residual_tol:
                gs.raise_exception(
                    f"VBD substep did not converge: residual {self.residual[None]:.3e} >= {self._residual_tol:.1e} "
                    f"after {self._max_sweeps} sweeps."
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
    def _func_diag_block(self, f, i_v, i_b):
        """Exact J_ii = dr_i/dx_i: the forward Hessian with the friction block replaced by its exact derivative."""
        _, H = self._func_vertex_system(f, i_v, i_b)
        kd_h = 1.0 + self._damping / self._substep_dt
        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            if self.elems_info[i_e].k_fiber > 0.0:
                w0 = self._func_vertex_weight(self.elems_info[i_e].B_rest, self.ve_role[c])
                _, H_exact, H_psd = self._func_fiber_terms(f + 1, i_e, i_b, w0)
                H += qd.cast(kd_h, qd.f64) * qd.cast(H_exact - H_psd, qd.f64)  # the forward kept only the PSD part
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
        (k_d/h) times its positive semidefinite part from Rayleigh damping."""
        sym, cross = self._func_offdiag_parts(f, i_v, i_b, vec)
        return (1.0 + self._damping / self._substep_dt) * sym + cross

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

    @qd.kernel
    def _kernel_adjoint_rhs(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.gbar[i_v, i_b] = self.adj[f + 1, i_v, i_b].pos + self.adj[f + 1, i_v, i_b].vel / self._substep_dt
            self.z[i_v, i_b] = qd.Vector.zero(qd.f64, 3)

    @qd.kernel
    def _kernel_adjoint_sweeps(self, f: qd.i32):
        """Colored Gauss-Seidel on J^T z = gbar: z_i = (J_ii^T)^-1 (gbar_i - sum_j J_ij z_j)."""
        for _ in qd.static(range(self._n_iterations)):
            for c in qd.static(range(self._n_colors)):
                for k, i_b in qd.ndrange((self._color_offsets[c], self._color_offsets[c + 1]), self._B):
                    i_v = self.color_perm[k]
                    rhs = self.gbar[i_v, i_b] - self._func_offdiag_apply(f, i_v, i_b, self.z)
                    self.z[i_v, i_b] = self._func_diag_block(f, i_v, i_b).transpose().inverse() @ rhs

    @qd.kernel
    def _kernel_adjoint_residual(self, f: qd.i32):
        self.adj_residual[None] = 0.0
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            r = self.gbar[i_v, i_b] - self._func_diag_block(f, i_v, i_b).transpose() @ self.z[i_v, i_b]
            r -= self._func_offdiag_apply(f, i_v, i_b, self.z)
            qd.atomic_max(self.adj_residual[None], qd.abs(r).max())
            qd.atomic_max(self.residual[None], qd.abs(self.gbar[i_v, i_b]).max())

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
                # dr/dx^t of the damping force -(k_d/h) sum_j K_ij (x_j - x_j^t) is +(k_d/h) K (symmetric), so the
                # adjoint gets -(k_d/h) (K z)_i over the diagonal block and the neighbours
                _, H = self._func_vertex_system(f, i_v, i_b)
                kd_h = self._damping * inv_h
                K_ii = (qd.cast(H, qd.f64) - qd.cast(self.verts_info[i_v].mass * inv_h * inv_h, qd.f64) * qd.Matrix.identity(qd.f64, 3)) / (1.0 + kd_h)
                sym, _ = self._func_offdiag_parts(f, i_v, i_b, self.z)
                self.adj[f, i_v, i_b].pos -= kd_h * (K_ii @ z + sym)
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
        self._kernel_adjoint_rhs(f)
        self.residual[None] = 0.0
        for _ in range(self._max_sweeps // self._n_iterations):
            self._kernel_adjoint_sweeps(f)
            self._kernel_adjoint_residual(f)
            if self.adj_residual[None] <= self._residual_tol * max(self.residual[None], 1.0):
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

    def compute_energy(self, f):
        """Incremental potential of substep `f` at the current iterate, shape (B,). Non-increasing across sweeps."""
        self._kernel_compute_energy(f)
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
    def color_offsets(self):
        return self._color_offsets
