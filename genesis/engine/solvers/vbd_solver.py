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

    # ------------------------------------------------------------------------------------
    # --------------------------------- initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def init_vertex_fields(self):
        struct_vert_info = qd.types.struct(mass=gs.qd_float)
        struct_vert_state = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            ipos=gs.qd_vec3,  # position at the start of the substep, x^t
            ypos=gs.qd_vec3,  # inertial position, y = x^t + h v^t + h^2 g
        )
        self.verts_info = struct_vert_info.field(shape=(self._n_vertices,), layout=qd.Layout.SOA)
        self.verts = struct_vert_state.field(shape=(self._n_vertices, self._B), layout=qd.Layout.SOA)

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
        )
        self.elems_info = struct_elem_info.field(shape=(self._n_elements,), layout=qd.Layout.SOA)
        self.muscle_actu = qd.field(dtype=gs.qd_float, shape=(max(self._n_muscle_groups, 1), self._B))
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
            self.muscle_actu.fill(0.0)

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
    ):
        for i_v_ in range(verts.shape[0]):
            i_v = i_v_ + v_start
            self.verts_info[i_v].mass = mass
            for i_b in range(self._B):
                for j in qd.static(range(3)):
                    self.verts[i_v, i_b].pos[j] = verts[i_v_, j]
                self.verts[i_v, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_e_ in range(elems.shape[0]):
            i_e = i_e_ + el_start
            for j in qd.static(range(4)):
                self.elems_info[i_e].v[j] = elems[i_e_, j] + v_start
            p0 = self.verts[self.elems_info[i_e].v[0], 0].pos
            p1 = self.verts[self.elems_info[i_e].v[1], 0].pos
            p2 = self.verts[self.elems_info[i_e].v[2], 0].pos
            p3 = self.verts[self.elems_info[i_e].v[3], 0].pos
            Dm = qd.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])
            self.elems_info[i_e].vol_rest = Dm.determinant() / 6.0
            self.elems_info[i_e].B_rest = Dm.inverse()
            self.elems_info[i_e].mu = mu
            self.elems_info[i_e].lam = lam + mu
            self.elems_info[i_e].gain = gain
            self.elems_info[i_e].fiber = qd.Vector.zero(gs.qd_float, 3)
            self.elems_info[i_e].group = -1

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
    def _func_deformation(self, i_e, i_b):
        """(F, B_eff) of tet `i_e` in env `i_b`."""
        v = self.elems_info[i_e].v
        p0 = self.verts[v[0], i_b].pos
        Ds = qd.Matrix.cols([self.verts[v[1], i_b].pos - p0, self.verts[v[2], i_b].pos - p0, self.verts[v[3], i_b].pos - p0])
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
    def _func_solve_vertex(self, i_v, i_b, inv_h2):
        m_h2 = qd.cast(self.verts_info[i_v].mass * inv_h2, self._acc)
        x = self.verts[i_v, i_b].pos
        force = -m_h2 * qd.cast(x - self.verts[i_v, i_b].ypos, self._acc)
        H = m_h2 * qd.Matrix.identity(self._acc, 3)

        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            role = self.ve_role[c]
            F, B = self._func_deformation(i_e, i_b)
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
            H += qd.cast(V * mu * w.norm_sqr(), self._acc) * qd.Matrix.identity(self._acc, 3)
            H += qd.cast(V * lam, self._acc) * q.outer_product(q)

        dx = H.inverse() @ force
        self.verts[i_v, i_b].pos = x + qd.cast(dx, gs.qd_float)

    @qd.kernel
    def _kernel_solve_color(self, f: qd.i32, lo: qd.i32, hi: qd.i32):
        """One color of one sweep. Kept for tests that watch the energy sweep by sweep."""
        inv_h2 = 1.0 / (self._substep_dt * self._substep_dt)
        for k, i_b in qd.ndrange((lo, hi), self._B):
            self._func_solve_vertex(self.color_perm[k], i_b, inv_h2)

    @qd.kernel
    def _kernel_predict(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.verts[i_v, i_b].ipos = self.verts[i_v, i_b].pos
            vel = self.verts[i_v, i_b].vel + self._gravity[i_b] * self._substep_dt
            self.verts[i_v, i_b].ypos = self.verts[i_v, i_b].pos + vel * self._substep_dt
            self.verts[i_v, i_b].pos = self.verts[i_v, i_b].ypos

    @qd.kernel
    def _kernel_substep(self, f: qd.i32):
        """A whole substep in one launch: predict, every sweep over every color, velocity update.

        Each top-level loop is a serial task with an implicit barrier after it, so the statically
        unrolled color loops are race-free Gauss-Seidel sweeps without any Python round trip.
        """
        inv_h2 = 1.0 / (self._substep_dt * self._substep_dt)
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.verts[i_v, i_b].ipos = self.verts[i_v, i_b].pos
            vel = self.verts[i_v, i_b].vel + self._gravity[i_b] * self._substep_dt
            self.verts[i_v, i_b].ypos = self.verts[i_v, i_b].pos + vel * self._substep_dt
            self.verts[i_v, i_b].pos = self.verts[i_v, i_b].ypos

        for _ in qd.static(range(self._n_iterations)):
            for c in qd.static(range(self._n_colors)):
                for k, i_b in qd.ndrange((self._color_offsets[c], self._color_offsets[c + 1]), self._B):
                    self._func_solve_vertex(self.color_perm[k], i_b, inv_h2)

        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.verts[i_v, i_b].vel = (self.verts[i_v, i_b].pos - self.verts[i_v, i_b].ipos) / self._substep_dt

    @qd.kernel
    def _kernel_compute_energy(self):
        """Incremental potential per env: inertia term plus the stable neo-Hookean energy of every tet."""
        inv_h2 = 1.0 / (self._substep_dt * self._substep_dt)
        for i_b in range(self._B):
            self.energy[i_b] = 0.0
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            d = self.verts[i_v, i_b].pos - self.verts[i_v, i_b].ypos
            self.energy[i_b] += qd.cast(0.5 * self.verts_info[i_v].mass * inv_h2 * d.norm_sqr(), qd.f64)
        for i_e, i_b in qd.ndrange(self._n_elements, self._B):
            F, _ = self._func_deformation(i_e, i_b)
            mu = self.elems_info[i_e].mu
            lam = self.elems_info[i_e].lam
            alpha = 1.0 + mu / lam
            J = F.determinant()
            psi = 0.5 * (mu * (F.norm_sqr() - 3.0) + lam * (J - alpha) ** 2)
            self.energy[i_b] += qd.cast(self.elems_info[i_e].vol_rest * psi, qd.f64)

    def compute_energy(self):
        """Incremental potential of the current positions, shape (B,). Non-increasing across sweeps."""
        self._kernel_compute_energy()
        return self.energy.to_numpy()

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        if self.is_active:
            self._kernel_substep(f)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        pass

    def substep_post_coupling_grad(self, f):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def reset_grad(self):
        pass

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_set_state(self, pos: qd.types.ndarray(), vel: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for j in qd.static(range(3)):
                self.verts[i_v, i_b].pos[j] = pos[i_b, i_v, j]
                self.verts[i_v, i_b].vel[j] = vel[i_b, i_v, j]

    @qd.kernel
    def _kernel_get_state(self, pos: qd.types.ndarray(), vel: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self.verts[i_v, i_b].pos[j]
                vel[i_b, i_v, j] = self.verts[i_v, i_b].vel[j]

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(state.pos, state.vel)

    def get_state(self, f):
        if not self.is_active:
            return None
        state = VBDSolverState(self._scene)
        self._kernel_get_state(state.pos, state.vel)
        return state

    @qd.kernel
    def _kernel_get_state_render(self, f: qd.i32):
        for i_vv, i_b in qd.ndrange(self._n_vverts, self._B):
            i_v = self.vverts_info[i_vv].vert_idx
            for j in qd.static(range(3)):
                pos_j = qd.cast(self.verts[i_v, i_b].pos[j], qd.f32)
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
