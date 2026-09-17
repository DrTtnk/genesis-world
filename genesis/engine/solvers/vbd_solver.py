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

from typing import NamedTuple

import networkx as nx
import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.entities.vbd_entity import VBDEntity
from genesis.engine.solvers.vbd_articulation import (
    kernel_begin_articulation,
    kernel_end_articulation,
    kernel_sweeps_articulation,
)
from genesis.engine.solvers.vbd_contact import (
    VBDContact,
    func_contact_dual_update,
    func_contact_vertex_terms,
    kernel_begin_contact,
    kernel_end_contact,
    kernel_prescribe_links,
    kernel_reset_contact,
    kernel_set_prescribed_state,
    kernel_set_prescribed_targets,
)
from genesis.engine.solvers.vbd_rigid import func_attachment_soft_system
from genesis.engine.solvers.vbd_rigid_attachment import (
    VBDRigidAttachment,
    func_attachment_point,
    func_attachment_pose,
    func_solve_attachment_link,
    func_update_attachment_dual,
    kernel_begin_attachment,
    kernel_end_attachment,
    kernel_set_attachment_state,
    kernel_set_vertex_state,
)
from genesis.engine.solvers.vbd_contact import EnvStatus
from genesis.engine.solvers.vbd_tissue_attachment import (
    VBDTissueAttachment,
    func_tissue_attachment_vertex_terms,
    func_update_tissue_attachment_dual,
    kernel_begin_tissue_attachment,
    kernel_set_tissue_attachment_state,
)
from genesis.engine.solvers.vbd_mtu import (
    HillParameters,
    LinkAnchor,
    SurfaceAnchor,
    TissueAnchor,
    UNIT_HILL,
    UNIT_LIGAMENT,
    VBDMTU,
    WorldAnchor,
    func_mtu_vertex_terms,
    kernel_begin_mtu,
    kernel_end_mtu,
    kernel_reset_mtu,
    kernel_set_excitation,
)
from genesis.engine.states.solvers import VBDSolverState
from genesis.utils.array_class import ErrorCode
from genesis.utils.misc import qd_to_torch, sanitize_index

from .base_solver import Solver


class TissueDiagnostics(NamedTuple):
    """Per-environment tissue validity of the last substep: the minimum `J / J0` over all tets (signed volume now
    over signed rest volume, so an uninverted tet is near +1 and an inverted one is negative or zero) and how many
    tets are inverted (`J / J0 <= 0`)."""

    min_j_ratio: torch.Tensor
    n_inverted: torch.Tensor


@qd.kernel
def kernel_set_muscle_state(envs_idx: qd.types.ndarray(), state: qd.types.ndarray(), actuation: qd.template()):
    for i_g, i_b_ in qd.ndrange(actuation.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        actuation[i_g, i_b] = state[i_b, i_g]


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
        self._self_thickness = options.self_collision_thickness
        if self._self_thickness > 0.0 and self._sim.requires_grad:
            gs.raise_exception(
                "Self-collision has no adjoint term yet (story S5), so requires_grad with "
                "self_collision_thickness > 0 would give a wrong gradient."
            )
        self._grad_converge = options.grad_converge
        self.rigid_attachment = None
        self.tissue_attachment = None
        self._tissue_attachment_pairs = []
        self._n_muscle_groups = 0
        self.contact = None
        self._contact_rules = []
        self._rigid_colliders = []
        self._prescribed_colliders = []
        self._contact_pair_cap = options.contact_pair_cap
        self._contact_cell_cap = options.contact_cell_cap
        self._raise_on_env_failure = options.raise_on_env_failure
        self._max_inverted_substeps = options.max_consecutive_inverted_substeps
        self.mtu = None
        self._mtu_units = []
        self._mtu_restraints = []

    @property
    def has_rigid_attachment(self):
        return self.rigid_attachment is not None

    @property
    def has_tissue_attachment(self):
        return self.tissue_attachment is not None

    @property
    def has_contact(self):
        return self.contact is not None

    @property
    def has_mtu(self):
        return self.mtu is not None

    def add_rigid_collider(self, link, collision_group):
        """Let the collision meshes of a rigid link take part in mesh contact with the tissue, in the given
        collision group. Declare before `scene.build()`; the link's pose is read from the rigid solver each
        substep, so a fixed link, a link the tissue drives, or a prescribed link all work."""
        if self._scene.is_built:
            gs.raise_exception("Rigid colliders must be declared before scene.build().")
        if any(link is other for other, _ in self._rigid_colliders) or any(
            link is other for entity, _, _ in self._prescribed_colliders for other in entity.links
        ):
            gs.raise_exception(f"Link {link.name} is already a collider.")
        if not link.geoms:
            gs.raise_exception(f"Collider link {link.name} has no collision geometry.")
        self._rigid_colliders.append((link, int(collision_group)))

    def add_prescribed_collider(self, entity, collision_group, link=None):
        """Let the collision geometry of a fixed rigid entity collide with the tissue while the pose of `link` (its
        base link by default) follows the targets given to `set_prescribed_targets`, sampled at every substep.
        Declare before `scene.build()`. Batched scenes need `RigidOptions.batch_links_info=True` so each
        environment can hold its own pose."""
        if self._scene.is_built:
            gs.raise_exception("Prescribed colliders must be declared before scene.build().")
        if not all(link.is_fixed for link in entity.links) or entity.n_dofs > 0:
            gs.raise_exception("A prescribed collider must be a rigid entity whose links are all fixed.")
        if not any(link.geoms for link in entity.links):
            gs.raise_exception("A prescribed collider needs collision geometry.")
        if any(link is other for link in entity.links for other, _ in self._rigid_colliders):
            gs.raise_exception("A link of the prescribed entity is already a collider.")
        if any(entity is other for other, _, _ in self._prescribed_colliders):
            gs.raise_exception("The entity is already a prescribed collider.")
        if link is None:
            link = entity.base_link
        if not any(link is other for other in entity.links):
            gs.raise_exception("The reference link must belong to the prescribed entity.")
        self._prescribed_colliders.append((entity, int(collision_group), link))

    def set_prescribed_targets(self, pos, quat):
        """Poses the prescribed colliders reach at the end of the next `scene.step()`, in declaration order:
        `pos` of shape (B, C, 3) and unit `quat` of shape (B, C, 4) in (w, x, y, z), on the simulation device. The
        interpolant is linear in position and shortest-arc in orientation from the current pose."""
        n_prescribed = len(self._prescribed_colliders)
        pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
        quat = torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device)
        if pos.shape != (self._B, n_prescribed, 3) or quat.shape != (self._B, n_prescribed, 4):
            gs.raise_exception(
                f"Prescribed targets need shapes ({self._B}, {n_prescribed}, 3) and ({self._B}, {n_prescribed}, 4), "
                f"got {tuple(pos.shape)} and {tuple(quat.shape)}."
            )
        if not bool(torch.isfinite(pos).all()) or not bool(torch.isfinite(quat).all()):
            gs.raise_exception("Prescribed targets must be finite.")
        if bool((torch.linalg.vector_norm(quat, dim=-1) - 1.0).abs().max() > 1e-4):
            gs.raise_exception("Prescribed target quaternions must have unit norm.")
        kernel_set_prescribed_targets(
            pos.contiguous(), quat.contiguous(), self.contact, self._sim.rigid_solver.dyn_state
        )

    def add_contact_rule(self, group_a, group_b, stiffness, friction, thickness):
        """Declare that two collision groups collide, with the penalty stiffness (N/m) of a pair, the isotropic
        Coulomb friction coefficient and the thickness (m) below which a pair is active. Declare before
        `scene.build()`; every pair of groups collides through at most one rule."""
        if self._scene.is_built:
            gs.raise_exception("Contact rules must be declared before scene.build().")
        if not stiffness > 0.0 or not thickness > 0.0 or friction < 0.0:
            gs.raise_exception("A contact rule needs stiffness > 0, thickness > 0 and friction >= 0.")
        self._contact_rules.append((int(group_a), int(group_b), float(stiffness), float(friction), float(thickness)))

    def _resolve_point_anchor(self, anchor):
        """A `TissueAnchor` or `SurfaceAnchor` as (entity, four vertex indices, four weights), the one form
        vbd_mtu.py also reduces both to."""
        if isinstance(anchor, SurfaceAnchor):
            if len(anchor.weights) != 3:
                gs.raise_exception("A surface anchor needs three barycentric weights, one per triangle corner.")
            if abs(sum(anchor.weights) - 1.0) > 1e-6:
                gs.raise_exception(f"The barycentric weights of a surface anchor sum to {sum(anchor.weights)}.")
            if not 0 <= anchor.triangle < len(anchor.entity.tris):
                gs.raise_exception(
                    f"Surface anchor triangle {anchor.triangle} is outside the entity's {len(anchor.entity.tris)} triangles."
                )
            triangle = anchor.entity.tris[anchor.triangle]
            return anchor.entity, tuple(int(v) for v in triangle) + (int(triangle[0]),), tuple(float(w) for w in anchor.weights) + (0.0,)
        if isinstance(anchor, TissueAnchor):
            if len(anchor.vertices) != 4 or len(anchor.weights) != 4:
                gs.raise_exception("A tissue anchor needs four vertices and four barycentric weights.")
            if abs(sum(anchor.weights) - 1.0) > 1e-6:
                gs.raise_exception(f"The barycentric weights of a tissue anchor sum to {sum(anchor.weights)}.")
            if any(not 0 <= v < anchor.entity.n_vertices for v in anchor.vertices):
                gs.raise_exception("A tissue anchor must index its entity's vertices.")
            return anchor.entity, tuple(int(v) for v in anchor.vertices), tuple(float(w) for w in anchor.weights)
        gs.raise_exception(f"Unknown tissue attachment anchor {anchor!r}.")

    def add_tissue_attachment(self, anchor_a, anchor_b):
        """Bind a material point of one tissue to a material point of another (or of the same tissue) with the
        two-way augmented-Lagrangian force of the rigid attachments, and return the attachment's index. Each
        anchor is a `TissueAnchor` or a `SurfaceAnchor`; the points bind with the offset they have now, so the
        rest state is stress-free. Declare before `scene.build()`."""
        if self._scene.is_built:
            gs.raise_exception("Declare tissue attachments before scene.build().")
        a, b = self._resolve_point_anchor(anchor_a), self._resolve_point_anchor(anchor_b)
        if a[0] is b[0] and sorted(zip(a[1], a[2])) == sorted(zip(b[1], b[2])):
            gs.raise_exception("A tissue attachment cannot bind a material point to the same material point.")
        for entity, _, _ in (a, b):
            if entity.scene is not self._scene:
                gs.raise_exception("Both anchors of a tissue attachment must belong to this scene.")
        self._tissue_attachment_pairs.append((a, b))
        return len(self._tissue_attachment_pairs) - 1

    def _add_routed_unit(self, kind, anchors, parameters, activation0=0.0, fibre_length0=None):
        if self._scene.is_built:
            gs.raise_exception("Muscle-tendon units must be declared before scene.build().")
        if len(anchors) < 2:
            gs.raise_exception("A route needs at least two anchors.")
        for anchor in anchors:
            if isinstance(anchor, SurfaceAnchor):
                if len(anchor.weights) != 3:
                    gs.raise_exception("A surface anchor needs three barycentric weights, one per triangle corner.")
                if abs(sum(anchor.weights) - 1.0) > 1e-6:
                    gs.raise_exception(f"The barycentric weights of a surface anchor sum to {sum(anchor.weights)}.")
                if not 0 <= anchor.triangle < len(anchor.entity.tris):
                    gs.raise_exception(
                        f"Surface anchor triangle {anchor.triangle} is outside the entity's "
                        f"{len(anchor.entity.tris)} triangles."
                    )
            elif isinstance(anchor, TissueAnchor):
                if len(anchor.vertices) != 4 or len(anchor.weights) != 4:
                    gs.raise_exception("A tissue anchor needs four vertices and four barycentric weights.")
                if abs(sum(anchor.weights) - 1.0) > 1e-6:
                    gs.raise_exception(f"The barycentric weights of a tissue anchor sum to {sum(anchor.weights)}.")
            elif not isinstance(anchor, (WorldAnchor, LinkAnchor)):
                gs.raise_exception(f"Unknown route anchor {anchor!r}.")
        self._mtu_units.append((kind, list(anchors), parameters, (float(activation0), fibre_length0)))
        return len(self._mtu_units) - 1

    def add_mtu(self, anchors, parameters, activation0=0.0, fibre_length0=None):
        """Declare one routed Hill muscle-tendon unit and return its index. `anchors` is an ordered list of
        `WorldAnchor`, `LinkAnchor` and `TissueAnchor`; `parameters` is a `HillParameters`. Declare before
        `scene.build()`."""
        if not isinstance(parameters, HillParameters):
            gs.raise_exception("A muscle-tendon unit needs HillParameters.")
        if not parameters.f_max > 0.0 or not parameters.l_opt > 0.0 or not parameters.l_slack > 0.0:
            gs.raise_exception("A muscle-tendon unit needs f_max, l_opt and l_slack above zero.")
        if not parameters.v_max > 0.0:
            gs.raise_exception("A muscle-tendon unit needs v_max above zero.")
        if not 0.0 <= activation0 <= 1.0:
            gs.raise_exception(f"An initial activation must be within [0, 1], got {activation0}.")
        if fibre_length0 is not None and not fibre_length0 > 0.0:
            gs.raise_exception(f"An initial fibre length must be above zero, got {fibre_length0}.")
        return self._add_routed_unit(UNIT_HILL, anchors, parameters, activation0, fibre_length0)

    def add_ligament(self, anchors, stiffness, slack_length):
        """Declare one tension-only linear element on the same routing and return its index. It carries
        `stiffness` (N/m) times the extension past `slack_length` (m), and nothing while it is slack."""
        if not stiffness > 0.0 or not slack_length > 0.0:
            gs.raise_exception("A ligament needs stiffness > 0 and slack_length > 0.")
        return self._add_routed_unit(UNIT_LIGAMENT, anchors, (float(stiffness), float(slack_length)))

    def add_rotary_restraint(self, dof, stiffness, rest_angle):
        """Declare a passive torque -stiffness (q - rest_angle) on one joint coordinate. Declare before
        `scene.build()`."""
        if self._scene.is_built:
            gs.raise_exception("Rotary restraints must be declared before scene.build().")
        if not stiffness > 0.0:
            gs.raise_exception("A rotary restraint needs stiffness > 0.")
        self._mtu_restraints.append((int(dof), float(stiffness), float(rest_angle)))
        return len(self._mtu_restraints) - 1

    def set_excitation(self, excitation):
        """Neural excitation in [0, 1] of every muscle-tendon unit, shape (B, M) or (M,). Held for the whole
        step; the activation follows it with the first-order lag of the Hill model."""
        if self.mtu is None:
            gs.raise_exception("No muscle-tendon unit was declared.")
        excitation = torch.as_tensor(excitation, dtype=gs.tc_float, device=gs.device)
        if excitation.ndim == 1:
            excitation = excitation.expand(self._B, -1)
        if excitation.shape != (self._B, self.mtu.n_units):
            gs.raise_exception(
                f"The excitation needs shape ({self._B}, {self.mtu.n_units}), got {tuple(excitation.shape)}."
            )
        if not torch.isfinite(excitation).all() or float(excitation.min()) < 0.0 or float(excitation.max()) > 1.0:
            gs.raise_exception("Every excitation must be finite and within [0, 1].")
        kernel_set_excitation(excitation.contiguous(), self.mtu)

    def mtu_state(self):
        """Activation, fibre length (m), route length (m), fibre velocity (m/s) and tension (N) of every unit,
        each (B, M)."""
        return self.mtu.state_readback()

    def mtu_anchor_forces(self):
        """World pull on every anchor of every unit at the last substep, (B, M, A_max, 3)."""
        return self.mtu.anchor_forces()

    def env_status(self):
        """Per-environment failure latch: `is_failed` (bool, shape (B,)), the global substep index at which the
        environment failed (`failed_substep`, -1 while it runs) and the raw error word (contact and tissue bits
        combined). A failed environment keeps the state of its failed attempt for diagnosis and advances again only
        after a reset."""
        errno = qd_to_torch(self.contact.errno) if self.contact is not None else torch.zeros(self._B, dtype=torch.int32)
        errno = errno | qd_to_torch(self.tissue_errno)
        return EnvStatus(qd_to_torch(self.env_failed) != 0, qd_to_torch(self.failed_substep), errno)

    def contact_diagnostics(self):
        """Candidate pair counts, largest tissue and rigid contact-vertex motion of the last substep, and the raw
        error word, each of shape (B,). See `ContactDiagnostics`."""
        return self.contact.diagnostics()

    def collider_impulses(self):
        """Time integral of `collider_reactions()` over every substep since the last `clear_collider_impulses()`,
        shape (B, n_colliders, 6) in N s and N m s."""
        return self.contact.impulses()

    def clear_collider_impulses(self):
        self.contact.clear_impulses()

    def collider_reactions(self):
        """Wrench the tissue applies to each declared collider link, shape (B, n_colliders, 6): world force then
        world torque about the link origin, from the last substep."""
        return self.contact.reactions()

    def check_errno(self):
        """Raise with every contact and tissue failure recorded since the last check, physical failures first: a
        batch that crossed a surface and overflowed a buffer in the same window reports both."""
        errno = int(qd_to_torch(self.tissue_errno).max())
        if self.contact is not None:
            errno |= int(qd_to_torch(self.contact.errno).max())
        messages = []
        if errno & ErrorCode.VBD_TISSUE_PERSISTENT_INVERSION:
            messages.append(
                "A tet stayed inverted (J/J0 <= 0) for more consecutive substeps than "
                "VBDOptions.max_consecutive_inverted_substeps allows: the inversion did not recover."
            )
        if errno & ErrorCode.VBD_CONTACT_CROSSING:
            messages.append("A contact pair crossed its surface within one substep: the penalty did not hold it.")
        if errno & ErrorCode.VBD_CONTACT_MOTION_BOUND:
            messages.append(
                "A contact vertex moved further than the contact margin in one substep, so a collision may have "
                "been missed. Use more substeps or a larger thickness."
            )
        if errno & ErrorCode.INVALID_VBD_CONTACT_NAN:
            messages.append("A contact pair has a non-finite distance.")
        if errno & ErrorCode.OVERFLOW_VBD_CONTACT_CELL:
            messages.append("More contact vertices in one hash cell than VBDOptions.contact_cell_cap allows.")
        if errno & ErrorCode.OVERFLOW_VBD_CONTACT_PAIRS:
            messages.append("More candidate contact pairs than VBDOptions.contact_pair_cap allows.")
        if messages:
            gs.raise_exception(" ".join(messages))

    # ------------------------------------------------------------------------------------
    # --------------------------------- initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def init_vertex_fields(self):
        struct_vert_info = qd.types.struct(
            mass=gs.qd_float,
            pinned=gs.qd_int,  # 1 when a bone owns this vertex, so the solve treats it as a boundary
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
        # Per-environment failure latch: 1 once a substep of that environment failed. A latched environment skips
        # every update until it is reset, so its state stays at the failed attempt for diagnosis.
        self.env_failed = qd.field(dtype=gs.qd_int, shape=(self._B,))
        self.failed_substep = qd.field(dtype=gs.qd_int, shape=(self._B,))
        self.failed_substep.fill(-1)
        # Where each pinned vertex is told to be. A prescribed boundary is per environment, because
        # every environment poses its skeleton differently, while which vertices are pinned is a
        # property of the rig and so is shared.
        self.pin_target = qd.Vector.field(3, dtype=gs.qd_float, shape=(self._n_vertices, self._B))
        # Adjoint state, one frame per position frame: dL/dx and dL/dv accumulated by the backward pass.
        struct_adj = qd.types.struct(pos=qd.types.vector(3, qd.f64), vel=qd.types.vector(3, qd.f64))
        self.adj = struct_adj.field(
            shape=(self._sim.substeps_local + 1, self._n_vertices, self._B), layout=qd.Layout.SOA
        )
        self.z = qd.Vector.field(
            3, dtype=qd.f64, shape=(self._n_vertices, self._B)
        )  # adjoint of the stationarity condition
        self.xb = qd.Vector.field(
            3, dtype=qd.f64, shape=(self._n_vertices, self._B)
        )  # running position adjoint of the reverse sweep
        self.yb = qd.Vector.field(3, dtype=qd.f64, shape=(self._n_vertices, self._B))  # adjoint of the predictor y
        self.gbar = qd.Vector.field(3, dtype=qd.f64, shape=(self._n_vertices, self._B))  # its right-hand side
        self.adj_residual = qd.field(dtype=qd.f64, shape=())
        # Replay buffer for the solver-level adjoint: the update applied to each vertex at each sweep of each
        # substep. The reverse pass walks it backwards, subtracting each update to recover the state the forward
        # linearised at, so the block itself is recomputed rather than stored (24 bytes a vertex a sweep, not 72).
        self._record_sweeps = self._sim.requires_grad
        # the stiffness ramp is frozen while the sweeps are being differentiated, in the forward and in the reverse
        self._ramp_active = not (self._record_sweeps and not self._grad_converge)
        self.sweep_dx = qd.Vector.field(
            3,
            dtype=qd.f64,
            shape=(self._sim.substeps_local, self._n_iterations, self._n_vertices, self._B)
            if self._record_sweeps
            else (1, 1, 1, 1),
        )
        # Self-collision reads the partner from here, not from the live state. A contact pair is invisible to the
        # static colouring, which is built from the tetrahedron graph and never saw the pair, so both sides can land
        # in the same colour and be solved by the same launch. This buffer is refreshed at the start of every colour
        # pass, so a partner is at most one colour pass stale and no thread reads what another thread is writing.
        self.pos_lag = qd.Vector.field(
            3, dtype=gs.qd_float, shape=(self._n_vertices, self._B) if self._self_thickness > 0.0 else (1, 1)
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
        self.elems_info = struct_elem_info.field(shape=(max(self._n_elements, 1),), layout=qd.Layout.SOA)

        # Shell membrane: a triangle carries the same stable neo-Hookean law as a tet, on the 3x2 in-plane
        # deformation gradient (`genesis/utils/shell.py`, proved in `verify_avbd_shell_math.py`). `B_rest` is
        # the 2x2 rest edge matrix inverse, in an orthonormal frame of the rest triangle's own plane.
        qd_mat2 = qd.types.matrix(2, 2, gs.qd_float)
        struct_tri_info = qd.types.struct(
            v=gs.qd_ivec3,
            area_rest=gs.qd_float,
            B_rest=qd_mat2,
            mu=gs.qd_float,
            lam=gs.qd_float,
            # columns of F at rest, which the Rayleigh damping's rest Hessian needs and nothing else does
            f0_rest=gs.qd_vec3,
            f1_rest=gs.qd_vec3,
        )
        self.tri_info = struct_tri_info.field(shape=(max(self._n_triangles, 1),), layout=qd.Layout.SOA)

        # Shell bending: one quadratic stencil per interior edge, `E = w/2 |K x - K x_rest|^2` with `K x =
        # sum_i c_i x_i` (Bergou et al. 2006's quadratic model, generalised to a curved rest shape). `c` and
        # `w` are fixed at rest and `kx_rest = K x_rest` is precomputed, so the Hessian block of vertex i is
        # the constant scalar `stiffness * w * c_i^2` times the identity: no matrix, ever.
        struct_bend_info = qd.types.struct(
            v=gs.qd_ivec4, c=gs.qd_vec4, w=gs.qd_float, kx_rest=gs.qd_vec3, stiffness=gs.qd_float
        )
        self.bend_info = struct_bend_info.field(shape=(max(self._n_stencils, 1),), layout=qd.Layout.SOA)

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
        # The reaction the body exerts on the bolus, so a caller can let it move under the forces the
        # wall applies instead of scripting its path. A scripted bolus is infinitely strong: it tows
        # the body when it travels, and cannot be transported when it is held.
        self.bolus_force = qd.Vector.field(3, dtype=qd.f64, shape=(self._B,))
        self.muscle_actu_adj = qd.field(dtype=qd.f64, shape=(max(self._n_muscle_groups, 1), self._B))
        self.energy = qd.field(dtype=qd.f64, shape=(self._B,))
        # Tissue validity: the smallest J/J0 (signed tet volume over signed rest volume) and the inverted tet count
        # of the last substep, per env, plus the persistent-inversion latch: a streak of consecutive substeps that
        # each held at least one inverted tet, and the raw error word this latch sets.
        self.min_j_ratio = qd.field(dtype=gs.qd_float, shape=(self._B,))
        self.n_inverted_tets = qd.field(dtype=gs.qd_int, shape=(self._B,))
        self.inverted_streak = qd.field(dtype=gs.qd_int, shape=(self._B,))
        self.tissue_errno = qd.field(dtype=gs.qd_int, shape=(self._B,))
        self.min_j_ratio.fill(1.0)
        self.n_inverted_tets.fill(0)
        self.inverted_streak.fill(0)
        self.tissue_errno.fill(0)

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
        struct_acons_info = qd.types.struct(
            v=gs.qd_ivec4, lo=gs.qd_float, hi=gs.qd_float, k0=gs.qd_float, sin_ref=gs.qd_float
        )
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
        # The multiplier state at the start of every sweep: every position update in that sweep used it, because a
        # constraint's dual update runs in the colour pass of its owner, the last of its vertices to move.
        rec = qd.types.struct(lam_hi=gs.qd_float, lam_lo=gs.qd_float, k=gs.qd_float)
        bar = qd.types.struct(lam_hi=qd.f64, lam_lo=qd.f64, k=qd.f64)
        shape = (self._sim.substeps_local, self._n_iterations, n, self._B) if self._sim.requires_grad else (1, 1, 1, 1)
        self.cons_rec = rec.field(shape=shape, layout=qd.Layout.SOA)
        self.cons_bar = bar.field(shape=(n, self._B), layout=qd.Layout.SOA)
        ashape = (
            (self._sim.substeps_local, self._n_iterations, na, self._B) if self._sim.requires_grad else (1, 1, 1, 1)
        )
        self.acons_rec = rec.field(shape=ashape, layout=qd.Layout.SOA)
        self.acons_bar = bar.field(shape=(na, self._B), layout=qd.Layout.SOA)
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

    def _incidence_csr(self, index_array):
        """Vertex -> incident-element CSR: for vertex `i`, `elem[offset[i]:offset[i+1]]` are the elements that
        touch it and `role[offset[i]:offset[i+1]]` its local index (column of `index_array`) in each."""
        n_local = index_array.shape[1]
        inc_vert = index_array.reshape(-1)
        inc_elem = np.repeat(np.arange(len(index_array)), n_local)
        inc_role = np.tile(np.arange(n_local), len(index_array))
        order = np.argsort(inc_vert, kind="stable")
        offset = np.searchsorted(inc_vert[order], np.arange(self._n_vertices + 1))
        return offset, inc_elem[order], inc_role[order]

    def _compute_vertex_coloring_and_incidence(self, elems, cons, acons, tris, bends):
        """Greedy vertex coloring of the graph of tets, triangles, bending stencils and constraints, plus the
        vertex -> incident-element CSR list of each of the first three.

        Returns (perm, color_offsets, n_colors, ve_offset, ve_elem, ve_role, vt_offset, vt_elem, vt_role,
        vb_offset, vb_elem, vb_role, color): vertices sorted by color (`perm[color_offsets[c]:color_offsets[c+1]]`
        is color `c`), and the CSR triples for tets, triangles and bending stencils. Two vertices that share a
        tet, a triangle, a bending stencil or a constraint never share a color, so each color is one race-free
        Gauss-Seidel sweep: a stencil couples all four of its vertices (`H_ij = w c_i c_j I` for every `i, j`,
        not only `i == j`), so every pair of the four, not only edges of the stencil's two triangles, must differ.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(self._n_vertices))
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            graph.add_edges_from(zip(elems[:, a].tolist(), elems[:, b].tolist()))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            graph.add_edges_from(zip(tris[:, a].tolist(), tris[:, b].tolist()))
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            graph.add_edges_from(zip(bends[:, a].tolist(), bends[:, b].tolist()))
        graph.add_edges_from(zip(cons[:, 0].tolist(), cons[:, 1].tolist()))
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            keep = acons[:, a] != acons[:, b]  # a vertex may serve both vectors of an angle constraint
            graph.add_edges_from(zip(acons[keep, a].tolist(), acons[keep, b].tolist()))
        coloring = nx.greedy_color(graph, strategy="smallest_last")
        color = np.array([coloring[i] for i in range(self._n_vertices)], dtype=np.int64)
        assert (color[cons[:, 0]] != color[cons[:, 1]]).all(), "a constraint joins two vertices of the same color"
        assert all(
            (color[acons[:, a]] != color[acons[:, b]])[acons[:, a] != acons[:, b]].all()
            for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        )
        assert all(
            (color[elems[:, a]] != color[elems[:, b]]).all()
            for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        )
        assert all((color[tris[:, a]] != color[tris[:, b]]).all() for a, b in ((0, 1), (0, 2), (1, 2)))
        assert all(
            (color[bends[:, a]] != color[bends[:, b]]).all()
            for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        )
        n_colors = int(color.max()) + 1
        perm = np.argsort(color, kind="stable")
        color_offsets = np.searchsorted(color[perm], np.arange(n_colors + 1)).tolist()

        ve_offset, ve_elem, ve_role = self._incidence_csr(elems)
        vt_offset, vt_elem, vt_role = self._incidence_csr(tris)
        vb_offset, vb_elem, vb_role = self._incidence_csr(bends)
        return (
            perm,
            color_offsets,
            n_colors,
            ve_offset,
            ve_elem,
            ve_role,
            vt_offset,
            vt_elem,
            vt_role,
            vb_offset,
            vb_elem,
            vb_role,
            color,
        )

    def _owner_csr(self, verts_per_constraint, color):
        """Per-vertex CSR of the constraints it owns. The owner is the constraint's vertex of highest color: when its
        color pass runs, every other vertex of the constraint has been updated this sweep and none is being written,
        so the owner can run the constraint's dual update inside its own pass, with no pass and barrier of its own."""
        n = len(verts_per_constraint)
        owner = (
            verts_per_constraint[np.arange(n), np.argmax(color[verts_per_constraint], axis=1)]
            if n
            else np.zeros(0, dtype=np.int64)
        )
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
        self._n_triangles = self.n_triangles
        self._n_stencils = self.n_stencils
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

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
            tris = np.concatenate(
                [entity._v_start + entity.tris for entity in self._entities if entity.n_triangles]
                + [np.zeros((0, 3), dtype=np.int64)]
            ).astype(np.int64)
            bends = np.concatenate(
                [entity._v_start + entity._bend_v for entity in self._entities if entity.n_stencils]
                + [np.zeros((0, 4), dtype=np.int64)]
            ).astype(np.int64)
            cons = np.concatenate(
                [entity._v_start + entity.distance_constraints for entity in self._entities]
                + [np.zeros((0, 2), dtype=np.int64)]
            ).astype(np.int64)
            lo = np.concatenate([entity.distance_bounds[:, 0] for entity in self._entities] + [np.zeros(0)])
            hi = np.concatenate([entity.distance_bounds[:, 1] for entity in self._entities] + [np.zeros(0)])
            acons = np.concatenate(
                [entity._v_start + entity.angle_constraints for entity in self._entities]
                + [np.zeros((0, 4), dtype=np.int64)]
            ).astype(np.int64)
            alo = np.concatenate([entity.angle_bounds[:, 0] for entity in self._entities] + [np.zeros(0)])
            ahi = np.concatenate([entity.angle_bounds[:, 1] for entity in self._entities] + [np.zeros(0)])
            self._n_constraints = len(cons)
            self._n_angle_constraints = len(acons)
            self.init_constraint_fields()
            (
                perm,
                self._color_offsets,
                self._n_colors,
                ve_offset,
                ve_elem,
                ve_role,
                vt_offset,
                vt_elem,
                vt_role,
                vb_offset,
                vb_elem,
                vb_role,
                color,
            ) = self._compute_vertex_coloring_and_incidence(elems, cons, acons, tris, bends)
            self._init_constraints(cons, lo, hi)
            self._init_angle_constraints(acons, alo, ahi)
            if self._damping > 0.0:
                for entity in self._entities:
                    # The membrane rest block is positive semidefinite at every Poisson ratio (its mu part is a
                    # Kronecker product of the weight Gram matrix with the in-plane projector, its lam part a Gram
                    # matrix), so the tet's nu >= 1/8 condition does not apply to a shell.
                    if isinstance(entity.material, gs.materials.VBD.Shell):
                        continue
                    # K0 (the rest Hessian) is positive semidefinite only for lam' >= mu / 3, i.e. nu >= 1/8
                    if entity.material.nu < 0.125:
                        gs.raise_exception(
                            f"Rayleigh damping needs nu >= 0.125 for a positive semidefinite rest Hessian; got nu={entity.material.nu}."
                        )
            self.vo_offset, self.vo_cons = self._owner_csr(cons, color)
            self.vao_offset, self.vao_cons = self._owner_csr(acons, color)
            self.color_perm = qd.field(dtype=gs.qd_int, shape=(self._n_vertices,))
            self.color_perm.from_numpy(perm.astype(gs.np_int))
            self.ve_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
            self.ve_offset.from_numpy(ve_offset.astype(gs.np_int))
            self.ve_elem = qd.field(dtype=gs.qd_int, shape=(max(len(ve_elem), 1),))
            self.ve_role = qd.field(dtype=gs.qd_int, shape=(max(len(ve_role), 1),))
            if len(ve_elem):
                self.ve_elem.from_numpy(ve_elem.astype(gs.np_int))
                self.ve_role.from_numpy(ve_role.astype(gs.np_int))
            self.vt_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
            self.vt_offset.from_numpy(vt_offset.astype(gs.np_int))
            self.vt_elem = qd.field(dtype=gs.qd_int, shape=(max(len(vt_elem), 1),))
            self.vt_role = qd.field(dtype=gs.qd_int, shape=(max(len(vt_role), 1),))
            if len(vt_elem):
                self.vt_elem.from_numpy(vt_elem.astype(gs.np_int))
                self.vt_role.from_numpy(vt_role.astype(gs.np_int))
            self.vb_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
            self.vb_offset.from_numpy(vb_offset.astype(gs.np_int))
            self.vb_elem = qd.field(dtype=gs.qd_int, shape=(max(len(vb_elem), 1),))
            self.vb_role = qd.field(dtype=gs.qd_int, shape=(max(len(vb_role), 1),))
            if len(vb_elem):
                self.vb_elem.from_numpy(vb_elem.astype(gs.np_int))
                self.vb_role.from_numpy(vb_role.astype(gs.np_int))
            self._init_self_collision(elems, tris, bends)
            # The noise floor of the force assembly: no solve can drive the residual below the rounding error of the
            # terms it sums, so the relative tolerance is floored here. m/h^2 times a tet edge is the force that moves
            # a vertex one edge in one substep, the largest term in the sum; times the relative precision of the
            # accumulator, that is the smallest residual the assembly can resolve.
            # A shell-only scene has no tets, so the edge scale falls back to its triangles.
            edge_v = self.elems_info.v.to_numpy()[:, :2] if self._n_elements else self.tri_info.v.to_numpy()[:, :2]
            pos0 = self.verts.pos.to_numpy()[0]
            edge = float(np.linalg.norm(pos0[edge_v[:, 1], 0] - pos0[edge_v[:, 0], 0], axis=1).mean())
            unit = float(self.verts_info.mass.to_numpy().max()) / self._substep_dt**2 * edge
            self._force_noise = unit * (1e-13 if gs.np_float == np.float64 else 1e-6)
            # The sweep kernel inlines one copy of the whole per-vertex solve for every colour of every sweep, so the
            # compiler's memory grows with their product. At 8 colours and 16 sweeps it reached 160 GB and the machine
            # had to be rescued; fail here instead, with the two numbers that caused it.
            unrolled = self._n_iterations * self._n_colors
            if unrolled > 96:  # 70 has always compiled; 128 on the constrained snake reached 160 GB
                gs.raise_exception(
                    f"VBD would inline {unrolled} copies of the vertex solve ({self._n_iterations} sweeps x "
                    f"{self._n_colors} colours). Compiling that needs tens of gigabytes. Use fewer sweeps, or more "
                    f"substeps instead of more sweeps."
                )
            if (self._n_triangles or self._n_stencils) and self._sim.requires_grad:
                gs.raise_exception("Shell elements have no adjoint yet, so they cannot be used with requires_grad.")
            attached_entities = [
                entity for entity in self._entities if entity._rigid_links or entity._barycentric_links
            ]
            if attached_entities:
                self.rigid_attachment = VBDRigidAttachment(self, attached_entities)
            if self._tissue_attachment_pairs:
                if self._sim.requires_grad:
                    gs.raise_exception("Tissue attachments have no adjoint yet, so they cannot be used with requires_grad.")
                self.tissue_attachment = VBDTissueAttachment(self, self._tissue_attachment_pairs)
            if self._contact_rules:
                if self._sim.requires_grad:
                    gs.raise_exception("Mesh contact has no adjoint, so it cannot be used with requires_grad.")
                rigid = self._sim.rigid_solver
                if self._prescribed_colliders and self._B > 1 and not rigid._options.batch_links_info:
                    gs.raise_exception(
                        "Prescribed colliders in a batched scene need RigidOptions.batch_links_info=True."
                    )
                self.contact = VBDContact(
                    self, self._entities, self._rigid_colliders, self._prescribed_colliders, self._contact_rules
                )
                kernel_reset_contact(torch.arange(self._B, dtype=torch.int32), self.contact, rigid.dyn_state)
            elif self._rigid_colliders or self._prescribed_colliders:
                gs.raise_exception("Colliders were declared without any contact rule.")
            if self._mtu_units:
                if self._sim.requires_grad:
                    gs.raise_exception("Muscle-tendon units have no adjoint, so they cannot be used with requires_grad.")
                self.mtu = VBDMTU(self, self._mtu_units, self._mtu_restraints)
                kernel_reset_mtu(torch.arange(self._B, dtype=torch.int32), self.mtu)
            elif self._mtu_restraints:
                gs.raise_exception("Rotary restraints were declared without any muscle-tendon unit.")
            self.reset_grad()  # after the constraint fields exist: it snapshots the multipliers the first window starts from

    def _init_self_collision(self, elems, tris, bends):
        """Which vertices may not touch each other: those sharing a tetrahedron, a shell triangle or a
        bending stencil are held together by the material and their proximity is the mesh, not a collision.
        Stored as a per-vertex sorted list so the contact loop can skip them with a short scan. Also the
        uniform grid the contact loop searches: cell size from the mesh itself (Teschner et al. 2005),
        buckets addressed by a hash of the cell so that no bound on the body's extent is needed."""
        groups = [
            (elems, {(a, b) for a in range(4) for b in range(4) if a != b}),
            (tris, {(a, b) for a in range(3) for b in range(3) if a != b}),
            (bends, {(a, b) for a in range(4) for b in range(4) if a != b}),
        ]
        pairs = [group[:, [a, b]] for group, pair_set in groups for a, b in sorted(pair_set)]
        pairs.append(np.zeros((0, 2), dtype=np.int64))
        neighbours = np.unique(np.concatenate(pairs), axis=0)
        offset = np.searchsorted(neighbours[:, 0], np.arange(self._n_vertices + 1))
        self.vn_offset = qd.field(dtype=gs.qd_int, shape=(self._n_vertices + 1,))
        self.vn_offset.from_numpy(offset.astype(gs.np_int))
        self.vn_vert = qd.field(dtype=gs.qd_int, shape=(max(len(neighbours), 1),))
        if len(neighbours):
            self.vn_vert.from_numpy(neighbours[:, 1].astype(gs.np_int))
        if self._self_thickness > 0.0:
            pos = self.verts.pos.to_numpy()[0, :, 0]
            edge = float(np.linalg.norm(pos[neighbours[:, 0]] - pos[neighbours[:, 1]], axis=1).mean())
            # A cell must hold every partner a vertex can touch within one cell of its own, so it cannot be
            # smaller than the interaction distance; below the mesh edge it would only add empty cells.
            self._self_cell = max(edge, 2.0 * self._self_thickness)
            self._hash_buckets = 2 * self._n_vertices
            self._hash_cap = 32
        else:
            self._self_cell, self._hash_buckets, self._hash_cap = 1.0, 1, 1
        self.cell_n = qd.field(dtype=gs.qd_int, shape=(self._hash_buckets, self._B))
        self.cell_v = qd.field(dtype=gs.qd_int, shape=(self._hash_buckets, self._hash_cap, self._B))
        self.cell_overflow = qd.field(dtype=gs.qd_int, shape=())
        # The cell each vertex was filed under. Vertices move while the sweeps run, so the cell of a current
        # position is not the cell the grid holds it in. Pairing on the filed cell keeps the candidate set fixed
        # for the whole substep and, above all, symmetric: A finds B exactly when B finds A. An asymmetric pair
        # applies a force to one side only, which injects momentum and throws the bodies apart.
        self.cell_of = qd.Vector.field(
            3, dtype=gs.qd_int, shape=(self._n_vertices, self._B) if self._self_thickness > 0.0 else (1, 1)
        )
        # The low corner of the 2x2x2 block of cells that holds every point within half a cell of the vertex.
        # A cell is at least twice the thickness, so eight cells cover the whole interaction sphere and the
        # twenty seven of a full neighbourhood are not needed.
        self.cell_lo = qd.Vector.field(
            3, dtype=gs.qd_int, shape=(self._n_vertices, self._B) if self._self_thickness > 0.0 else (1, 1)
        )

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
            assert (lo <= hi).all() and (lo >= -1.0).all() and (hi <= 1.0).all(), (
                "cosine bounds must satisfy -1 <= lo <= hi <= 1"
            )
            self.va_cons.from_numpy(inc_cons[order].astype(gs.np_int))
            self.va_slot.from_numpy(inc_slot[order].astype(gs.np_int))
            self.acons_info.v.from_numpy(acons.astype(gs.np_int))
            self.acons_info.lo.from_numpy(lo.astype(gs.np_float))
            self.acons_info.hi.from_numpy(hi.astype(gs.np_float))
        # base stiffness per constraint so that k |grad C|^2 matches a distance constraint of stiffness k_start:
        # k0 = k_start |u|^2 |v|^2 / (|u|^2 + |v|^2) on the rest vectors (the review's harmonic scale)
        if len(acons):
            pos = self.verts.pos.to_numpy()[0, :, 0]
            lu2 = np.linalg.norm(pos[acons[:, 0]] - pos[acons[:, 1]], axis=1) ** 2
            lv2 = np.linalg.norm(pos[acons[:, 2]] - pos[acons[:, 3]], axis=1) ** 2
            k0 = self._k_start * lu2 * lv2 / (lu2 + lv2)
            self.acons_info.k0.from_numpy(k0.astype(gs.np_float))
            u = pos[acons[:, 0]] - pos[acons[:, 1]]
            v = pos[acons[:, 2]] - pos[acons[:, 3]]
            cos_rest = (u * v).sum(-1) / np.sqrt(lu2 * lv2)
            # a cosine violation divided by sin(theta) is the angle error; near 0 or 180 degrees the cosine is flat
            # and no cosine tolerance is an angle tolerance, so the reference is floored
            self.acons_info.sin_ref.from_numpy(
                np.maximum(np.sqrt(np.clip(1.0 - cos_rest**2, 0.0, 1.0)), 0.1).astype(gs.np_float)
            )
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
            tri_start=self.n_triangles,
            bend_start=self.n_stencils,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            muscle_group_start=self._n_muscle_groups,
            name=name,
        )
        self._entities.append(entity)
        if isinstance(material, gs.materials.VBD.Muscle):
            self._n_muscle_groups += material.n_groups
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
            self.verts_info[i_v].pinned = 0
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
    def _kernel_add_shell_elements(
        self,
        v_start: qd.i32,
        tri_start: qd.i32,
        bend_start: qd.i32,
        verts: qd.types.ndarray(),
        mass: qd.types.ndarray(),
        tris: qd.types.ndarray(),
        tri_area_rest: qd.types.ndarray(),
        tri_B_rest: qd.types.ndarray(),
        mu: qd.f32,
        lam: qd.f32,
        mu_forward: qd.f32,
        mu_backward: qd.f32,
        mu_lateral: qd.f32,
        bend_v: qd.types.ndarray(),
        bend_c: qd.types.ndarray(),
        bend_w: qd.types.ndarray(),
        bend_kx_rest: qd.types.ndarray(),
        bending_stiffness: qd.f32,
    ):
        for i_v_ in range(verts.shape[0]):
            i_v = i_v_ + v_start
            self.verts_info[i_v].mass = mass[i_v_]
            self.verts_info[i_v].pinned = 0
            self.verts_info[i_v].tangent = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
            self.verts_info[i_v].mu_forward = mu_forward
            self.verts_info[i_v].mu_backward = mu_backward
            self.verts_info[i_v].mu_lateral = mu_lateral
            for i_b in range(self._B):
                for j in qd.static(range(3)):
                    self.verts[0, i_v, i_b].pos[j] = verts[i_v_, j]
                self.verts[0, i_v, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        # The rest frame (B_rest, area_rest) is computed once in numpy by `genesis/utils/shell.py`, the exact
        # transcription of the spike's `rest_frame`, and only copied in here.
        for i_t_ in range(tris.shape[0]):
            i_t = i_t_ + tri_start
            for j in qd.static(range(3)):
                self.tri_info[i_t].v[j] = tris[i_t_, j] + v_start
            self.tri_info[i_t].area_rest = tri_area_rest[i_t_]
            self.tri_info[i_t].B_rest = qd.Matrix(
                [[tri_B_rest[i_t_, 0], tri_B_rest[i_t_, 1]], [tri_B_rest[i_t_, 2], tri_B_rest[i_t_, 3]]]
            )
            self.tri_info[i_t].mu = mu
            self.tri_info[i_t].lam = lam
            e0 = qd.Vector(
                [verts[tris[i_t_, 1], 0] - verts[tris[i_t_, 0], 0],
                 verts[tris[i_t_, 1], 1] - verts[tris[i_t_, 0], 1],
                 verts[tris[i_t_, 1], 2] - verts[tris[i_t_, 0], 2]],
                dt=gs.qd_float,
            )
            e1 = qd.Vector(
                [verts[tris[i_t_, 2], 0] - verts[tris[i_t_, 0], 0],
                 verts[tris[i_t_, 2], 1] - verts[tris[i_t_, 0], 1],
                 verts[tris[i_t_, 2], 2] - verts[tris[i_t_, 0], 2]],
                dt=gs.qd_float,
            )
            B0 = self.tri_info[i_t].B_rest
            self.tri_info[i_t].f0_rest = e0 * B0[0, 0] + e1 * B0[1, 0]
            self.tri_info[i_t].f1_rest = e0 * B0[0, 1] + e1 * B0[1, 1]

        # The bending coefficients c, weight w and Kx_rest also come from `genesis/utils/shell.py`, an exact
        # transcription of the spike's `bending_coefficients`, `bending_weight` and `Kx_rest`: an SVD null
        # space is not something a kernel should compute, so it never runs on-device.
        for i_s_ in range(bend_v.shape[0]):
            i_s = i_s_ + bend_start
            for j in qd.static(range(4)):
                self.bend_info[i_s].v[j] = bend_v[i_s_, j] + v_start
                self.bend_info[i_s].c[j] = bend_c[i_s_, j]
            self.bend_info[i_s].w = bend_w[i_s_]
            for j in qd.static(range(3)):
                self.bend_info[i_s].kx_rest[j] = bend_kx_rest[i_s_, j]
            self.bend_info[i_s].stiffness = bending_stiffness

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

    @qd.kernel
    def _kernel_set_pinned(self, v_start: qd.i32, pinned: qd.types.ndarray()):
        for i_v_ in range(pinned.shape[0]):
            i_v = i_v_ + v_start
            self.verts_info[i_v].pinned = pinned[i_v_]
            for i_b in range(self._B):
                # A vertex pinned without a target yet holds the pose it is already in
                self.pin_target[i_v, i_b] = self.verts[self._sim.cur_substep_local, i_v, i_b].pos

    @qd.kernel
    def _kernel_set_pin_targets(self, v_start: qd.i32, target: qd.types.ndarray()):
        for i_v_, i_b in qd.ndrange(target.shape[1], self._B):
            for j in qd.static(range(3)):
                self.pin_target[i_v_ + v_start, i_b][j] = target[i_b, i_v_, j]

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
        self._kernel_set_actuation(0, actus)

    @qd.kernel
    def _kernel_set_actuation(self, group_start: int, actus: qd.types.ndarray()):
        for i_g, i_b in qd.ndrange(actus.shape[0], actus.shape[1]):
            self.muscle_actu[group_start + i_g, i_b] = actus[i_g, i_b]

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
    def _func_vertex_weight2_static(self, B, role: qd.template()):
        """`_func_vertex_weight2` for a compile-time `role`, as the damping's static neighbour loop needs."""
        w = qd.Vector.zero(gs.qd_float, 2)
        if qd.static(role == 0):
            w = -(B[0, :] + B[1, :])
        else:
            w = B[role - 1, :]
        return w

    @qd.func
    def _func_vertex_weight2(self, B, role):
        """`_func_vertex_weight` for the 2x2 rest matrix of a shell triangle: row of `B` for local vertex
        `role` in {1, 2}, vertex 0 (role 0) carrying minus the sum of the other rows."""
        w = qd.Vector.zero(gs.qd_float, 2)
        if role == 0:
            w = -(B[0, :] + B[1, :])
        else:
            w = B[role - 1, :]
        return w

    @qd.func
    def _func_cofactor2(self, F):
        """The membrane's 3x2 analogue of `_func_cofactor`: `d J / d F` with `J = |f0 x f1|`, verified in
        `verify_avbd_shell_math.py` against torch autograd on random `F`."""
        n_hat = F[:, 0].cross(F[:, 1]).normalized()
        return qd.Matrix.cols([F[:, 1].cross(n_hat), n_hat.cross(F[:, 0])])

    @qd.func
    def _func_fiber_terms(self, fr, i_e, i_b, w_i):
        """Fibre reinforcement E = V k/2 (|F0 a| - 1)^2 on the unactuated F0 = Ds B_rest (a spine or tendon: it
        resists length change along `a` whatever the muscle does). Returns (force on the vertex with row
        weight `w_i`, exact 3x3 Hessian block, PSD part of that block). The (l - 1)/l (I - u u^T) part is
        negative in compression, so the forward step uses the PSD part like the contact terms do."""
        v = self.elems_info[i_e].v
        p0 = self.verts[fr, v[0], i_b].pos
        Ds = qd.Matrix.cols(
            [self.verts[fr, v[1], i_b].pos - p0, self.verts[fr, v[2], i_b].pos - p0, self.verts[fr, v[3], i_b].pos - p0]
        )
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
                        damp += qd.cast(
                            self._func_rest_block(i_e, w0, self._func_vertex_weight_static(B0, r)) @ d_j, self._acc
                        )

        # Shell membrane: the same stable neo-Hookean law on the 3x2 in-plane deformation gradient
        # `F = Ds B_rest`, `Ds` the deformed edges in world space (`verify_avbd_shell_math.py`, membrane_energy).
        # The block is the same Gauss-Newton form as the tet's: always positive semidefinite, never projected.
        for c in range(self.vt_offset[i_v], self.vt_offset[i_v + 1]):
            i_t = self.vt_elem[c]
            role = self.vt_role[c]
            v = self.tri_info[i_t].v
            p0 = self.verts[f + 1, v[0], i_b].pos
            Ds = qd.Matrix.cols([self.verts[f + 1, v[1], i_b].pos - p0, self.verts[f + 1, v[2], i_b].pos - p0])
            Bm = self.tri_info[i_t].B_rest
            F = Ds @ Bm
            mu = self.tri_info[i_t].mu
            lam = self.tri_info[i_t].lam
            alpha = 1.0 + mu / lam
            n = F[:, 0].cross(F[:, 1])
            J = n.norm()
            cof = self._func_cofactor2(F)
            P = mu * F + lam * (J - alpha) * cof
            w = self._func_vertex_weight2(Bm, role)
            A0 = self.tri_info[i_t].area_rest
            q = qd.cast(cof @ w, self._acc)
            force -= qd.cast(A0 * (P @ w), self._acc)
            K += qd.cast(A0 * mu * w.norm_sqr(), self._acc) * qd.Matrix.identity(self._acc, 3)
            K += qd.cast(A0 * lam, self._acc) * q.outer_product(q)
            if qd.static(self._damping > 0.0):
                w0 = self._func_vertex_weight2(Bm, role)
                K0 += qd.cast(self._func_rest_block2(i_t, w0, w0), self._acc)
                for r in qd.static(range(3)):
                    if r != role:
                        j = v[r]
                        d_j = self.verts[f + 1, j, i_b].pos - self.verts[f, j, i_b].pos
                        damp += qd.cast(
                            self._func_rest_block2(i_t, w0, self._func_vertex_weight2_static(Bm, r)) @ d_j,
                            self._acc,
                        )

        # Shell bending: E = w/2 |K x - K x_rest|^2, K x = sum_i c_i x_i over the stencil's four vertices.
        # grad_i = w c_i (K x - K x_rest), H_ii = w c_i^2 I exactly (verify_avbd_shell_math.py): a scalar
        # times the identity, so it needs no projection.
        for c in range(self.vb_offset[i_v], self.vb_offset[i_v + 1]):
            i_s = self.vb_elem[c]
            role = self.vb_role[c]
            bv = self.bend_info[i_s].v
            coeffs = self.bend_info[i_s].c
            kx = qd.Vector.zero(gs.qd_float, 3)
            for r in qd.static(range(4)):
                kx += coeffs[r] * self.verts[f + 1, bv[r], i_b].pos
            residual = kx - self.bend_info[i_s].kx_rest
            stiffness = self.bend_info[i_s].stiffness
            w_bend = self.bend_info[i_s].w
            c_i = coeffs[role]
            force -= qd.cast(stiffness * w_bend * c_i * residual, self._acc)
            K += qd.cast(stiffness * w_bend * c_i * c_i, self._acc) * qd.Matrix.identity(self._acc, 3)

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
            H += qd.cast(qd.abs(mult) / dist, self._acc) * qd.cast(
                qd.Matrix.identity(gs.qd_float, 3) - n.outer_product(n), self._acc
            )

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
            H += qd.cast(qd.abs(mult) / (scale * scale), self._acc) * qd.cast(
                qd.Matrix.identity(gs.qd_float, 3) - own.outer_product(own), self._acc
            )

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

        # The body against itself: a quadratic penalty on the overlap of two spheres of the given thickness.
        # Vertices that share a tetrahedron are skipped, because their closeness is the mesh rather than a
        # collision. The exact block is indefinite while a pair overlaps, so only its positive semidefinite
        # part is kept, as the floor and the fibre term already do.
        if qd.static(self._self_thickness > 0.0):
            base = self.cell_lo[i_v, i_b]
            for di, dj, dk in qd.ndrange(2, 2, 2):
                cell = base + qd.Vector([di, dj, dk], dt=gs.qd_int)
                h = self._func_cell_hash(cell)
                for slot in range(qd.min(self.cell_n[h, i_b], self._hash_cap)):
                    j = self.cell_v[h, slot, i_b]
                    # Two cells can hash to one bucket, and a vertex must not be visited twice, so a candidate
                    # counts only for the cell it actually sits in.
                    if (self.cell_of[j, i_b] == cell).all():
                        touching_mesh = False
                        for c in range(self.vn_offset[i_v], self.vn_offset[i_v + 1]):
                            if self.vn_vert[c] == j:
                                touching_mesh = True
                        if j != i_v and not touching_mesh:
                            e_s = x - self.pos_lag[j, i_b]
                            d_s = e_s.norm()
                            if d_s < self._self_thickness:
                                n_s = e_s / d_s
                                k_s = self._contact_stiffness
                                force += qd.cast(k_s * (self._self_thickness - d_s), self._acc) * qd.cast(
                                    n_s, self._acc
                                )
                                H += qd.cast(k_s, self._acc) * qd.cast(n_s.outer_product(n_s), self._acc)

        # Analytic capsule bolus: the same penalty and IPC-smoothed isotropic Coulomb friction against a moving
        # capsule (sphere when half_length is 0). `rel` is the vector from the closest point of the segment.
        r_b = self.bolus[i_b].radius
        if r_b > 0.0:
            rel = x - self.bolus[i_b].center
            along = qd.min(
                qd.max(rel.dot(self.bolus[i_b].axis), -self.bolus[i_b].half_length), self.bolus[i_b].half_length
            )
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
                H += qd.cast(lam_b * g, self._acc) * qd.cast(
                    qd.Matrix.identity(gs.qd_float, 3) - n_b.outer_product(n_b), self._acc
                )
        if qd.static(self.has_rigid_attachment):
            # A vertex reaches its attachments through the same CSR vbd_mtu.py uses for a routed anchor: weight w on
            # the force, w^2 on the curvature block, so a plain vertex attachment (weight 1) is unchanged and a
            # barycentric one (weight < 1) carries its fair share.
            for slot in range(
                self.rigid_attachment.vert_anchor_offset[i_v], self.rigid_attachment.vert_anchor_offset[i_v + 1]
            ):
                i_a = self.rigid_attachment.vert_anchor[slot] // 4
                corner = self.rigid_attachment.vert_anchor[slot] % 4
                weight = self.rigid_attachment.info[i_a].weights[corner]
                pos, quat = func_attachment_pose(i_a, i_b, self.rigid_attachment)
                point = func_attachment_point(f + 1, i_a, i_b, self, self.rigid_attachment)
                force_a, hessian_a = func_attachment_soft_system(
                    point,
                    pos,
                    quat,
                    self.rigid_attachment.info[i_a].local_pos,
                    self.rigid_attachment.state[i_a, i_b].multiplier,
                    self.rigid_attachment.state[i_a, i_b].stiffness,
                    self.rigid_attachment.previous_error[i_a, i_b],
                    self.rigid_attachment.alpha,
                )
                force += qd.cast(weight * force_a, self._acc)
                H += qd.cast(weight * weight * hessian_a, self._acc)
        if qd.static(self.has_tissue_attachment):
            force_t, hessian_t = func_tissue_attachment_vertex_terms(f, i_v, i_b, self, self.tissue_attachment)
            force += qd.cast(force_t, self._acc)
            H += qd.cast(hessian_t, self._acc)
        if qd.static(self.has_contact):
            force_c, hessian_c = func_contact_vertex_terms(f, i_v, i_b, self, self.contact)
            force += qd.cast(force_c, self._acc)
            H += qd.cast(hessian_c, self._acc)
        if qd.static(self.has_mtu):
            force_m, hessian_m = func_mtu_vertex_terms(f, i_v, i_b, self, self.mtu)
            force += qd.cast(force_m, self._acc)
            H += qd.cast(hessian_m, self._acc)
        return force, H, K0

    @qd.func
    def _func_rest_block2(self, i_t, w_i, w_j):
        """Block of shell triangle i_t's exact energy Hessian at rest between the vertices with in-plane weights w_i
        and w_j, the matrix the Rayleigh damping uses. With u = F_rest w and n the unit rest normal,

            A0 [ mu (w_i . w_j) (I - n n^T) + lam u_i u_j^T + mu [u_i x u_j]_x ],

        derived and checked against autograd on random triangles, for all nine blocks, in
        `verify_avbd_shell_math.py` (`membrane_rest_block`). At rest F is an isometry, so J = 1 and J - alpha =
        -mu / lam, which is where the in-plane projector and the cross term come from. Symmetric part positive
        semidefinite at every Poisson ratio, and zero on rigid motions, so damping never brakes a coiling gut.
        Bending is deliberately left out: its quadratic model is not rotation invariant about a curved rest, so its
        Hessian does not annihilate a rotation (same spike)."""
        A0 = self.tri_info[i_t].area_rest
        mu = self.tri_info[i_t].mu
        lam = self.tri_info[i_t].lam
        f0 = self.tri_info[i_t].f0_rest
        f1 = self.tri_info[i_t].f1_rest
        n = f0.cross(f1)
        u_i = f0 * w_i[0] + f1 * w_i[1]
        u_j = f0 * w_j[0] + f1 * w_j[1]
        blk = A0 * mu * w_i.dot(w_j) * (qd.Matrix.identity(gs.qd_float, 3) - n.outer_product(n))
        blk += A0 * lam * u_i.outer_product(u_j)
        c = u_i.cross(u_j)  # zero on the diagonal block
        blk += A0 * mu * qd.Matrix([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
        return blk

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
        if not self.verts_info[i_v].pinned and not self.env_failed[i_b]:
            force, H, K_unused = self._func_vertex_system(f, i_v, i_b)
            dx = H.inverse() @ force
            if qd.static(self._record_sweeps):
                self.sweep_dx[f, sweep, i_v, i_b] = qd.cast(dx, qd.f64)
            self.verts[f + 1, i_v, i_b].pos += qd.cast(dx, gs.qd_float)
        if w > 0.0 and not self.env_failed[i_b]:
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

    @qd.func
    def _func_cell(self, x):
        return qd.Vector(
            [qd.floor(x[0] / self._self_cell), qd.floor(x[1] / self._self_cell), qd.floor(x[2] / self._self_cell)],
            dt=gs.qd_int,
        )

    @qd.func
    def _func_cell_hash(self, c):
        """Teschner et al. 2005: three large primes, exclusive or, then the table size. The remainder of a
        negative product is negative in Quadrants as in C, so it is folded back into range."""
        h = ((c[0] * 73856093) ^ (c[1] * 19349663) ^ (c[2] * 83492791)) % self._hash_buckets
        if h < 0:
            h += self._hash_buckets
        return h

    @qd.kernel
    def _kernel_build_hash(self, f: qd.i32):
        """The grid for one substep, from the predicted positions. Contacts change while the body folds, so it
        is rebuilt every substep, but not every sweep: a vertex moves a fraction of a cell in 0.25 ms."""
        for h, i_b in qd.ndrange(self._hash_buckets, self._B):
            self.cell_n[h, i_b] = 0
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            x = self.verts[f + 1, i_v, i_b].pos
            cell = self._func_cell(x)
            self.cell_of[i_v, i_b] = cell
            self.cell_lo[i_v, i_b] = self._func_cell(x - 0.5 * self._self_cell)
            h = self._func_cell_hash(cell)
            slot = qd.atomic_add(self.cell_n[h, i_b], 1)
            if slot < self._hash_cap:
                self.cell_v[h, slot, i_b] = i_v
            else:
                self.cell_overflow[None] = 1

    @qd.kernel
    def _kernel_predict(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            if self.env_failed[i_b]:
                self.verts[f + 1, i_v, i_b].pos = self.verts[f, i_v, i_b].pos
            elif self.verts_info[i_v].pinned:
                self.verts[f + 1, i_v, i_b].pos = self.pin_target[i_v, i_b]
            else:
                self.verts[f + 1, i_v, i_b].pos = self._func_inertia_target(f, i_v, i_b)

    @qd.kernel
    def _kernel_solve_color(self, f: qd.i32, lo: qd.i32, hi: qd.i32):
        """One color of one sweep. Kept for tests that watch the energy sweep by sweep."""
        if qd.static(self._self_thickness > 0.0):
            for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
                self.pos_lag[i_v, i_b] = self.verts[f + 1, i_v, i_b].pos
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
            mult = qd.max(k * c_hi + self.cons[i_c, i_b].lam_hi, 0.0) + qd.min(
                k * c_lo + self.cons[i_c, i_b].lam_lo, 0.0
            )
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
            mult = qd.max(k * c_hi + self.acons[i_c, i_b].lam_hi, 0.0) + qd.min(
                k * c_lo + self.acons[i_c, i_b].lam_lo, 0.0
            )
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
            self.acons[i_c, i_b].lam_hi = qd.max(
                self.acons[i_c, i_b].lam_hi + w * k * (cosv - self.acons_info[i_c].hi), 0.0
            )
            self.acons[i_c, i_b].lam_lo = qd.min(
                self.acons[i_c, i_b].lam_lo + w * k * (cosv - self.acons_info[i_c].lo), 0.0
            )
        else:
            self.acons[i_c, i_b].lam_hi += w * k * (cosv - self.acons_info[i_c].hi)
        _, violation = self._func_angle_mult(i_c, i_b, cosv)
        k0 = self.acons_info[i_c].k0
        self.acons[i_c, i_b].k = qd.min(
            k + ramp * k0 / self._angle_tol * qd.abs(violation), self._constraint_k_max_ratio * k0
        )

    @qd.func
    def _func_dual_update(self, f, i_c, i_b, w, ramp):
        """Giles et al. 2025 Eq. 11 to 13: clamped lam += w k C per side, k += ramp beta |C| with beta = k_start /
        constraint_tol. The per-sweep forward uses w = constraint_dual_relaxation and ramp = 1; the exact Uzawa
        iteration under requires_grad uses w = 1 and ramp = 0 (a stiff k only slows the primal Gauss-Seidel)."""
        e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
        dist = e.norm()
        k = self.cons[i_c, i_b].k
        if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
            self.cons[i_c, i_b].lam_hi = qd.max(
                self.cons[i_c, i_b].lam_hi + w * k * (dist - self.cons_info[i_c].hi), 0.0
            )
            self.cons[i_c, i_b].lam_lo = qd.min(
                self.cons[i_c, i_b].lam_lo + w * k * (dist - self.cons_info[i_c].lo), 0.0
            )
        else:
            self.cons[i_c, i_b].lam_hi += w * k * (dist - self.cons_info[i_c].hi)
        _, violation = self._func_constraint_mult(i_c, i_b, dist)
        self.cons[i_c, i_b].k = qd.min(
            k + ramp * self._k_start / self._constraint_tol * qd.abs(violation),
            self._constraint_k_max_ratio * self._k_start,
        )

    @qd.kernel
    def _kernel_sweeps(self, f: qd.i32, sweep: qd.i32):
        """One sweep. The sweep index is a runtime argument and Python drives the loop, so the body is
        transformed once instead of `n_iterations` times.

        It cannot be a loop inside the kernel: the colour passes below are top-level `ndrange` loops, which is
        what makes them parallel with an implicit barrier between them, and wrapping them in an outer loop would
        serialise the whole solve. So the choice is `n_iterations` inlined copies of this body or `n_iterations`
        kernel launches. The launches cost a few microseconds each against a step of about a millisecond, while
        the inlining costs about 4.6 seconds of Quadrants front-end work per sweep on every process start.
        """
        if True:
            self._func_sweep(f, sweep)
            if qd.static(self.has_rigid_attachment):
                # one block per free body, in order: the bodies couple only through the soft elements, so this
                # is Gauss-Seidel over blocks and a later body already sees the earlier one's new pose
                for i_b in range(self._B):
                    if not self.env_failed[i_b]:
                        for i_f in range(self.rigid_attachment.n_free):
                            func_solve_attachment_link(f, i_f, i_b, self, self.rigid_attachment)
                for i_a, i_b in qd.ndrange(self.rigid_attachment.n_attachments, self._B):
                    if not self.env_failed[i_b]:
                        func_update_attachment_dual(f, i_a, i_b, self, self.rigid_attachment)
            if qd.static(self.has_tissue_attachment):
                for i_a, i_b in qd.ndrange(self.tissue_attachment.n_attachments, self._B):
                    if not self.env_failed[i_b]:
                        func_update_tissue_attachment_dual(f, i_a, i_b, self, self.tissue_attachment)
            # the pairs restart from zero next substep, so the dual update after the last sweep would only skew
            # the reported reactions away from the forces the sweep applied
            if qd.static(self.has_contact):
                if sweep < self._n_iterations - 1:
                    func_contact_dual_update(f, self._constraint_dual_relaxation, self, self.contact)

    @qd.func
    def _func_sweep(self, f, sweep):
        """One Gauss-Seidel sweep. Each top-level loop is a serial task with an implicit
        barrier after it, so the statically unrolled color loops are race-free without a Python round trip. The
        constraints' dual updates ride inside the color pass of their owner vertex (see `_owner_csr`): a pass of
        their own would be a few thousand threads behind a barrier, and cost half the step on the ladder body."""
        if qd.static(self._record_sweeps and self._n_constraints > 0):
            for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
                self.cons_rec[f, sweep, i_c, i_b].lam_hi = self.cons[i_c, i_b].lam_hi
                self.cons_rec[f, sweep, i_c, i_b].lam_lo = self.cons[i_c, i_b].lam_lo
                self.cons_rec[f, sweep, i_c, i_b].k = self.cons[i_c, i_b].k
        if qd.static(self._record_sweeps and self._n_angle_constraints > 0):
            for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
                self.acons_rec[f, sweep, i_c, i_b].lam_hi = self.acons[i_c, i_b].lam_hi
                self.acons_rec[f, sweep, i_c, i_b].lam_lo = self.acons[i_c, i_b].lam_lo
                self.acons_rec[f, sweep, i_c, i_b].k = self.acons[i_c, i_b].k
        for c in qd.static(range(self._n_colors)):
            if qd.static(self._self_thickness > 0.0):
                for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
                    self.pos_lag[i_v, i_b] = self.verts[f + 1, i_v, i_b].pos
            for k, i_b in qd.ndrange((self._color_offsets[c], self._color_offsets[c + 1]), self._B):
                # the stiffness ramp is frozen when the sweeps are being differentiated: its coefficient is
                # k_start / constraint_tol, about 3e9 on the snake, and it multiplies straight into the position
                # adjoint, which makes the executed map wildly expansive (measured: 1e13 over one step against
                # 4.2 with the stiffness held). A fixed stiffness also converges better (useful_knowledge.md).
                self._func_solve_vertex(
                    f,
                    self.color_perm[k],
                    i_b,
                    self._constraint_dual_relaxation,
                    1.0 if qd.static(self._ramp_active) else 0.0,
                    sweep == self._n_iterations - 1,
                    sweep,
                )

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
            if not self.env_failed[i_b]:
                self.cons[i_c, i_b].lam_hi *= 0.95 * 0.99
                self.cons[i_c, i_b].lam_lo *= 0.95 * 0.99
                self.cons[i_c, i_b].k = qd.max(self._k_start, 0.99 * self.cons[i_c, i_b].k)
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            if not self.env_failed[i_b]:
                self.acons[i_c, i_b].lam_hi *= 0.95 * 0.99
                self.acons[i_c, i_b].lam_lo *= 0.95 * 0.99
                self.acons[i_c, i_b].k = qd.max(self.acons_info[i_c].k0, 0.99 * self.acons[i_c, i_b].k)

    @qd.kernel
    def _kernel_constraint_error(self, f: qd.i32):
        self.cons_error[None] = 0.0
        self.cons_error_rel[None] = 0.0
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            e = (
                self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos
                - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
            )
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
        self._kernel_angle_error(
            self._sim.cur_substep_local - 1 if self._sim.cur_substep_local > 0 else self._sim.substeps_local - 1
        )
        return float(self.cons_error[None])

    def constraint_error(self):
        """Largest absolute distance-constraint error (m) at the current end-of-substep positions, over all envs."""
        self._kernel_constraint_error(
            self._sim.cur_substep_local - 1 if self._sim.cur_substep_local > 0 else self._sim.substeps_local - 1
        )
        return float(self.cons_error[None])

    @qd.kernel
    def _kernel_bolus_reaction(self, f: qd.i32):
        """Total force the body applies to the bolus: the negative of the contact and friction forces it
        applies to the body, summed over the vertices touching it."""
        for i_b in range(self._B):
            self.bolus_force[i_b] = qd.Vector.zero(qd.f64, 3)
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            r_b = self.bolus[i_b].radius
            if r_b > 0.0:
                x = self.verts[f + 1, i_v, i_b].pos
                rel = x - self.bolus[i_b].center
                along = qd.min(
                    qd.max(rel.dot(self.bolus[i_b].axis), -self.bolus[i_b].half_length), self.bolus[i_b].half_length
                )
                rel = rel - along * self.bolus[i_b].axis
                dist = rel.norm()
                pen = r_b - dist
                if pen > 0.0:
                    k = self._contact_stiffness
                    n_b = rel / dist
                    on_body = qd.cast(k * pen, qd.f64) * qd.cast(n_b, qd.f64)
                    slide = x - self.verts[f, i_v, i_b].pos - self.bolus[i_b].vel * self._substep_dt
                    slide -= slide.dot(n_b) * n_b
                    u_norm = slide.norm()
                    eps = self._friction_eps_v * self._substep_dt
                    g = 1.0 / u_norm
                    if u_norm < eps:
                        g = 2.0 / eps - u_norm / (eps * eps)
                    on_body -= qd.cast(k * pen * self.bolus[i_b].friction * g, qd.f64) * qd.cast(slide, qd.f64)
                    for d in qd.static(range(3)):
                        qd.atomic_add(self.bolus_force[i_b][d], -on_body[d])

    def bolus_reaction(self):
        """The force the body is applying to the bolus, shape (B, 3), from the state of the last substep."""
        self._kernel_bolus_reaction(
            self._sim.cur_substep_local - 1 if self._sim.cur_substep_local > 0 else self._sim.substeps_local - 1
        )
        return self.bolus_force.to_numpy()

    @qd.kernel
    def _kernel_update_velocity(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            if self.env_failed[i_b]:
                self.verts[f + 1, i_v, i_b].vel = self.verts[f, i_v, i_b].vel
            else:
                self.verts[f + 1, i_v, i_b].vel = (
                    self.verts[f + 1, i_v, i_b].pos - self.verts[f, i_v, i_b].pos
                ) / self._substep_dt

    @qd.kernel
    def _kernel_tissue_diagnostics(self, f: qd.i32, substep_global: qd.i32):
        """Minimum `J / J0` and inverted tet count of substep `f`, per env, at the post-sweep state (frame `f+1`).
        `J / J0` is the current signed tet volume over its signed rest volume, which is exactly `det(F)`: muscle
        actuation leaves it unchanged, because its contraction of the rest shape has `det(A) = 1`. An env whose
        streak of consecutive substeps with any inverted tet passes the configured limit latches, through the
        same failure fields the contact checks use."""
        for i_b in range(self._B):
            if not self.env_failed[i_b]:
                self.min_j_ratio[i_b] = 1e30
                self.n_inverted_tets[i_b] = 0
        for i_e, i_b in qd.ndrange(self._n_elements, self._B):
            if not self.env_failed[i_b]:
                F, _ = self._func_deformation(f + 1, i_e, i_b)
                ratio = F.determinant()
                qd.atomic_min(self.min_j_ratio[i_b], ratio)
                if ratio <= 0.0:
                    qd.atomic_add(self.n_inverted_tets[i_b], 1)
        for i_b in range(self._B):
            if not self.env_failed[i_b]:
                if self.n_inverted_tets[i_b] > 0:
                    self.inverted_streak[i_b] += 1
                else:
                    self.inverted_streak[i_b] = 0
                if self.inverted_streak[i_b] > self._max_inverted_substeps:
                    qd.atomic_or(self.tissue_errno[i_b], ErrorCode.VBD_TISSUE_PERSISTENT_INVERSION)
                    self.env_failed[i_b] = 1
                    self.failed_substep[i_b] = substep_global

    def tissue_diagnostics(self):
        """Minimum `J / J0` and inverted tet count of the last substep, each of shape (B,). See `TissueDiagnostics`."""
        return TissueDiagnostics(qd_to_torch(self.min_j_ratio), qd_to_torch(self.n_inverted_tets))

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
            if self.rigid_attachment is not None and self.rigid_attachment.is_articulated:
                rigid = self.rigid_attachment.rigid
                kernel_sweeps_articulation(
                    f, self, rigid.dyn_state, rigid.dyn_info, rigid.rigid_info, rigid.rigid_config
                )
            else:
                for sweep in range(self._n_iterations):
                    self._kernel_sweeps(f, sweep)
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
            H = -(u_hat.outer_product(nu) + nu.outer_product(u_hat) + cosv * (I3 - u_hat.outer_product(u_hat))) / (
                lu * lu
            )
        elif s >= 2 and t >= 2:
            H = -(v_hat.outer_product(nv) + nv.outer_product(v_hat) + cosv * (I3 - v_hat.outer_product(v_hat))) / (
                lv * lv
            )
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
        the constraints' positive semidefinite proxies replaced by their exact curvature."""
        force_unused, H, K_unused = self._func_vertex_system(f, i_v, i_b)
        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            if self.elems_info[i_e].k_fiber > 0.0:
                w0 = self._func_vertex_weight(self.elems_info[i_e].B_rest, self.ve_role[c])
                _, H_exact, H_psd = self._func_fiber_terms(f + 1, i_e, i_b, w0)
                H += qd.cast(H_exact - H_psd, qd.f64)  # the forward kept only the PSD part
        for c in range(self.vc_offset[i_v], self.vc_offset[i_v + 1]):
            i_c = self.vc_cons[c]
            e = (
                self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos
                - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
            )
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
            H -= (qd.abs(qd.cast(mult, qd.f64)) / (scale * scale)) * (
                qd.Matrix.identity(qd.f64, 3) - own.outer_product(own)
            )
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
            mu_ax = (
                0.5 * (self.verts_info[i_v].mu_forward + self.verts_info[i_v].mu_backward)
                + 0.5 * (self.verts_info[i_v].mu_forward - self.verts_info[i_v].mu_backward) * th
            )
            P = mu_ax * t.outer_product(t) + self.verts_info[i_v].mu_lateral * b.outer_product(b)
            H -= qd.cast(lam_n * g, qd.f64) * qd.cast(P, qd.f64)
            H += A_f + coupling
        return qd.cast(H, qd.f64)

    @qd.func
    def _func_diag_block_live(self, f, i_v, i_b):
        """`_func_diag_block` with the multiplier state taken from the solver rather than from the end-of-substep
        record. The reverse sweep needs this, because every sweep used the state it started from, and the record
        holds only the state the substep ended with."""
        force_unused, H, K_unused = self._func_vertex_system(f, i_v, i_b)
        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            if self.elems_info[i_e].k_fiber > 0.0:
                w0 = self._func_vertex_weight(self.elems_info[i_e].B_rest, self.ve_role[c])
                _, H_exact, H_psd = self._func_fiber_terms(f + 1, i_e, i_b, w0)
                H += qd.cast(H_exact - H_psd, qd.f64)  # the forward kept only the PSD part
        for c in range(self.vc_offset[i_v], self.vc_offset[i_v + 1]):
            i_c = self.vc_cons[c]
            e = (
                self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos
                - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
            )
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
                    H += self._func_distance_block_live(f, i_c, i_b, s, t)
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
            H -= (qd.abs(qd.cast(mult, qd.f64)) / (scale * scale)) * (
                qd.Matrix.identity(qd.f64, 3) - own.outer_product(own)
            )
            vq = self.acons_info[i_c].v
            for t in qd.static(range(4)):
                if vq[t] == i_v:
                    H += self._func_angle_block_live(f, i_c, i_b, slot, t)
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
            mu_ax = (
                0.5 * (self.verts_info[i_v].mu_forward + self.verts_info[i_v].mu_backward)
                + 0.5 * (self.verts_info[i_v].mu_forward - self.verts_info[i_v].mu_backward) * th
            )
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
                    out += (
                        qd.cast(self._func_rest_block(i_e, w0, self._func_vertex_weight_static(B0, r)), qd.f64)
                        @ vec[self.elems_info[i_e].v[r], i_b]
                    )
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
                        F0 = (
                            qd.Matrix.cols(
                                [
                                    self.verts[f + 1, v0[1], i_b].pos - p00,
                                    self.verts[f + 1, v0[2], i_b].pos - p00,
                                    self.verts[f + 1, v0[3], i_b].pos - p00,
                                ]
                            )
                            @ B0
                        )
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
                e = (
                    self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos
                    - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
                )
                out += (self.cons_zeta[i_c, i_b] * qd.cast(self.vc_side[c], qd.f64)) * qd.cast(e / e.norm(), qd.f64)
        for c in range(self.va_offset[i_v], self.va_offset[i_v + 1]):
            i_c = self.va_cons[c]
            if self.acons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
                out += self.acons_zeta[i_c, i_b] * self._func_angle_slot_grad(
                    self.va_slot[c], u_hat, v_hat, lu, lv, cosv
                )
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
    def _func_actuation_reverse(self, f, i_v, i_b, p, dx):
        """The muscle command reaches a block twice, through the rest shape it contracts: once in the local gradient
        and once in the local Hessian, since both are built from the actuated weights. The second is the tangent."""
        for c in range(self.ve_offset[i_v], self.ve_offset[i_v + 1]):
            i_e = self.ve_elem[c]
            group = self.elems_info[i_e].group
            if group >= 0:
                role = self.ve_role[c]
                s_ = 1.0 - self.muscle_actu[group, i_b] * self.elems_info[i_e].gain
                m = self.elems_info[i_e].fiber
                mmT = m.outer_product(m)
                I3 = qd.Matrix.identity(gs.qd_float, 3)
                A_dot = self.elems_info[i_e].gain * ((1.0 / (s_ * s_)) * mmT - (0.5 / qd.sqrt(s_)) * (I3 - mmT))
                F, B = self._func_deformation(f + 1, i_e, i_b)
                A = (1.0 / s_) * mmT + qd.sqrt(s_) * (I3 - mmT)
                F_dot = (F @ A.inverse()) @ A_dot
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
                w0 = self._func_vertex_weight(self.elems_info[i_e].B_rest, role)
                w = self._func_vertex_weight(B, role)
                dg = self.elems_info[i_e].vol_rest * (P_dot @ w + P @ (A_dot @ w0))
                # the block Hessian is built from the same actuated weights, so it moves with the command too
                dw = A_dot @ w0
                q = qd.cast(cof @ w, qd.f64)
                dq = qd.cast(dcof @ w + cof @ dw, qd.f64)
                V = qd.cast(self.elems_info[i_e].vol_rest, qd.f64)
                dH = V * (
                    qd.cast(2.0 * mu * w.dot(dw), qd.f64) * p.dot(dx)
                    + qd.cast(lam, qd.f64) * (dq.dot(p) * q.dot(dx) + q.dot(p) * dq.dot(dx))
                )
                qd.atomic_add(self.muscle_actu_adj[group, i_b], -qd.cast(dg, qd.f64).dot(p) - dH)

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
                F0f = (
                    qd.Matrix.cols(
                        [
                            self.verts[f + 1, vf[1], i_b].pos - p0f,
                            self.verts[f + 1, vf[2], i_b].pos - p0f,
                            self.verts[f + 1, vf[3], i_b].pos - p0f,
                        ]
                    )
                    @ B0f
                )
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
                    grad_t = Vd * (
                        qd.cast(mu * w_i.dot(w_j), qd.f64) * p
                        + lamd * q_i.dot(p) * q_j
                        + lamd * qd.cast(J - alpha, qd.f64) * a.cross(p)
                    )
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
                        F0 = (
                            qd.Matrix.cols(
                                [
                                    self.verts[f + 1, v0[1], i_b].pos - p00,
                                    self.verts[f + 1, v0[2], i_b].pos - p00,
                                    self.verts[f + 1, v0[3], i_b].pos - p00,
                                ]
                            )
                            @ B0
                        )
                        u = F0 @ aa
                        l = u.norm()
                        u_hat = qd.cast(u / l, qd.f64)
                        cf = qd.cast(V * self.elems_info[i_e].k_fiber * wi0.dot(aa) * wj0.dot(aa), qd.f64)
                        grad_t += cf * (
                            u_hat.dot(p) * u_hat + qd.cast((l - 1.0) / l, qd.f64) * (p - u_hat.dot(p) * u_hat)
                        )
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
            A = qd.cast(mu_ax, qd.f64) * td.dot(p) * td.dot(dx) + qd.cast(
                self.verts_info[i_v].mu_lateral, qd.f64
            ) * bd.dot(p) * bd.dot(dx)
            e_z = qd.Vector([0.0, 0.0, 1.0], dt=qd.f64)
            by_depth = -qd.cast(k, qd.f64) * qd.cast(g, qd.f64) * A * e_z
            by_slide = (
                qd.cast(lam_n, qd.f64)
                * A
                * qd.cast(dg, qd.f64)
                / qd.max(qd.cast(u_norm, qd.f64), 1e-300)
                * qd.cast(slide, qd.f64)
            )
            by_blend = qd.cast(lam_n * g * dmu_ax, qd.f64) * td.dot(p) * td.dot(dx) * td
            tangent = by_depth + by_slide + by_blend
            tangent_prev = -(by_slide + by_blend)
        return tangent, tangent_prev

    @qd.func
    def _func_distance_block_live(self, f, i_c, i_b, s, t):
        """d g_s / d x_t of a distance constraint from the live multiplier state, which during the reverse pass is
        the state that sweep started from. Symmetric, so it is its own transpose."""
        e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
        dist = qd.cast(e.norm(), qd.f64)
        n = qd.cast(e, qd.f64) / dist
        nn = n.outer_product(n)
        mult, violation_unused = self._func_constraint_mult(i_c, i_b, qd.cast(dist, gs.qd_float))
        blk = (qd.cast(mult, qd.f64) / dist) * (qd.Matrix.identity(qd.f64, 3) - nn)
        if mult != 0.0:
            blk += qd.cast(self.cons[i_c, i_b].k, qd.f64) * nn
        sign = 1.0
        if s != t:
            sign = -1.0
        return sign * blk

    @qd.func
    def _func_angle_block_live(self, f, i_c, i_b, s, t):
        """d g_s / d x_t of an angle constraint from the live multiplier state."""
        u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
        I3 = qd.Matrix.identity(qd.f64, 3)
        nu = v_hat - cosv * u_hat
        nv = u_hat - cosv * v_hat
        H = qd.Matrix.zero(qd.f64, 3, 3)
        if s < 2 and t < 2:
            H = -(u_hat.outer_product(nu) + nu.outer_product(u_hat) + cosv * (I3 - u_hat.outer_product(u_hat))) / (
                lu * lu
            )
        elif s >= 2 and t >= 2:
            H = -(v_hat.outer_product(nv) + nv.outer_product(v_hat) + cosv * (I3 - v_hat.outer_product(v_hat))) / (
                lv * lv
            )
        elif s < 2:
            H = ((I3 - v_hat.outer_product(v_hat)) - u_hat.outer_product(nv)) / (lu * lv)
        else:
            H = ((I3 - u_hat.outer_product(u_hat)) - v_hat.outer_product(nu)) / (lu * lv)
        sign = 1.0
        if (s % 2) != (t % 2):
            sign = -1.0
        mult, violation_unused = self._func_angle_mult(i_c, i_b, qd.cast(cosv, gs.qd_float))
        blk = (sign * qd.cast(mult, qd.f64)) * H
        if mult != 0.0:
            g_s = self._func_angle_slot_grad(s, u_hat, v_hat, lu, lv, cosv)
            g_t = self._func_angle_slot_grad(t, u_hat, v_hat, lu, lv, cosv)
            blk += qd.cast(self.acons[i_c, i_b].k, qd.f64) * g_s.outer_product(g_t)
        return blk

    @qd.func
    def _func_distance_tangent(self, f, i_c, i_b, t, p, dx):
        """The gradient with respect to vertex `t` of the scalar p^T H dx, where H is the constraint's part of the
        block. Both k n n^T and the |mult| / dist proxy move with the position, so unlike the elastic block this
        does not vanish on the diagonal."""
        e = self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
        dist = qd.cast(e.norm(), qd.f64)
        n = qd.cast(e, qd.f64) / dist
        mult, violation_unused = self._func_constraint_mult(i_c, i_b, qd.cast(dist, gs.qd_float))
        multd = qd.cast(mult, qd.f64)
        kd_raw = qd.cast(self.cons[i_c, i_b].k, qd.f64)
        kd = kd_raw
        if mult == 0.0:
            kd = 0.0
        perp_p = p - n.dot(p) * n
        perp_dx = dx - n.dot(dx) * n
        A = n.dot(p) * n.dot(dx)
        Bq = p.dot(dx)
        sides = 1.0  # d|mult| / d(dist) is k times the number of unclamped sides, from the live state
        if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
            sides = 0.0
            if (
                kd_raw * (dist - qd.cast(self.cons_info[i_c].hi, qd.f64)) + qd.cast(self.cons[i_c, i_b].lam_hi, qd.f64)
                > 0.0
            ):
                sides += 1.0
            if (
                kd_raw * (dist - qd.cast(self.cons_info[i_c].lo, qd.f64)) + qd.cast(self.cons[i_c, i_b].lam_lo, qd.f64)
                < 0.0
            ):
                sides += 1.0
        d_abs = kd_raw * sides
        if multd < 0.0:
            d_abs = -d_abs
        grad = (kd - qd.abs(multd) / dist) * (n.dot(dx) * perp_p + n.dot(p) * perp_dx) / dist
        grad += (Bq - A) * (d_abs - qd.abs(multd) / dist) / dist * n
        sign = 1.0
        if t != 0:
            sign = -1.0
        return sign * grad

    @qd.kernel
    def _kernel_restore_cons(self, f: qd.i32, sweep: qd.i32):
        """Put the multiplier state back to what the given sweep started from."""
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            self.cons[i_c, i_b].lam_hi = self.cons_rec[f, sweep, i_c, i_b].lam_hi
            self.cons[i_c, i_b].lam_lo = self.cons_rec[f, sweep, i_c, i_b].lam_lo
            self.cons[i_c, i_b].k = self.cons_rec[f, sweep, i_c, i_b].k
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            self.acons[i_c, i_b].lam_hi = self.acons_rec[f, sweep, i_c, i_b].lam_hi
            self.acons[i_c, i_b].lam_lo = self.acons_rec[f, sweep, i_c, i_b].lam_lo
            self.acons[i_c, i_b].k = self.acons_rec[f, sweep, i_c, i_b].k

    @qd.kernel
    def _kernel_reverse_dual(self, f: qd.i32, lo: qd.i32, hi: qd.i32):
        """Undo the dual updates owned by one colour, at the positions the step produced, which is what they read.

        The forward updates the multiplier and then the stiffness, so the reverse undoes the stiffness first. Both
        push on the position adjoint through the constraint's own gradient, and a clamped side or a capped stiffness
        contributes nothing, because its derivative there is zero.
        """
        for kk, i_b in qd.ndrange((lo, hi), self._B):
            i_v = self.color_perm[kk]
            for c in range(self.vo_offset[i_v], self.vo_offset[i_v + 1]):
                i_c = self.vo_cons[c]
                va = self.cons_info[i_c].v[0]
                vb = self.cons_info[i_c].v[1]
                e = self.verts[f + 1, va, i_b].pos - self.verts[f + 1, vb, i_b].pos
                dist = e.norm()
                n = qd.cast(e / dist, qd.f64)
                w = qd.cast(self._constraint_dual_relaxation, qd.f64)
                kd = qd.cast(self.cons[i_c, i_b].k, qd.f64)
                distd = qd.cast(dist, qd.f64)
                hi_b = qd.cast(self.cons_info[i_c].hi, qd.f64)
                lo_b = qd.cast(self.cons_info[i_c].lo, qd.f64)
                push = 0.0

                # the stiffness ramp came last in the forward, so it is undone first
                mult_r_unused, violation = self._func_constraint_mult(i_c, i_b, dist)
                beta = qd.cast(self._k_start / self._constraint_tol, qd.f64)
                if qd.static(not self._ramp_active):
                    beta = 0.0
                if kd + beta * qd.abs(qd.cast(violation, qd.f64)) < qd.cast(
                    self._constraint_k_max_ratio * self._k_start, qd.f64
                ):
                    sides_v = 1.0
                    if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
                        sides_v = 0.0
                        if distd > hi_b:
                            sides_v += 1.0
                        if distd < lo_b:
                            sides_v += 1.0
                    s_v = 1.0
                    if violation < 0.0:
                        s_v = -1.0
                    push += beta * s_v * sides_v * self.cons_bar[i_c, i_b].k
                else:
                    self.cons_bar[i_c, i_b].k = 0.0

                # then the multiplier update, which read the stiffness as it was before the ramp
                bar_hi = self.cons_bar[i_c, i_b].lam_hi
                bar_lo = self.cons_bar[i_c, i_b].lam_lo
                active = bar_hi
                if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
                    active = 0.0
                    if qd.cast(self.cons[i_c, i_b].lam_hi, qd.f64) + w * kd * (distd - hi_b) > 0.0:
                        active += bar_hi
                    if qd.cast(self.cons[i_c, i_b].lam_lo, qd.f64) + w * kd * (distd - lo_b) < 0.0:
                        active += bar_lo
                push += w * kd * active
                self.cons_bar[i_c, i_b].k += w * (distd - hi_b) * bar_hi

                for d in qd.static(range(3)):
                    qd.atomic_add(self.xb[va, i_b][d], push * n[d])
                    qd.atomic_add(self.xb[vb, i_b][d], -push * n[d])
            for c in range(self.vao_offset[i_v], self.vao_offset[i_v + 1]):
                i_c = self.vao_cons[c]
                u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
                vq = self.acons_info[i_c].v
                w = qd.cast(self._constraint_dual_relaxation, qd.f64)
                kd = qd.cast(self.acons[i_c, i_b].k, qd.f64)
                hi_b = qd.cast(self.acons_info[i_c].hi, qd.f64)
                lo_b = qd.cast(self.acons_info[i_c].lo, qd.f64)
                push = 0.0
                mult_u, violation = self._func_angle_mult(i_c, i_b, qd.cast(cosv, gs.qd_float))
                beta = qd.cast(self.acons_info[i_c].k0 / self._angle_tol, qd.f64)
                if qd.static(not self._ramp_active):
                    beta = 0.0
                if kd + beta * qd.abs(qd.cast(violation, qd.f64)) < qd.cast(
                    self._constraint_k_max_ratio, qd.f64
                ) * qd.cast(self.acons_info[i_c].k0, qd.f64):
                    sides_v = 1.0
                    if self.acons_info[i_c].lo < self.acons_info[i_c].hi:
                        sides_v = 0.0
                        if cosv > hi_b:
                            sides_v += 1.0
                        if cosv < lo_b:
                            sides_v += 1.0
                    s_v = 1.0
                    if violation < 0.0:
                        s_v = -1.0
                    push += beta * s_v * sides_v * self.acons_bar[i_c, i_b].k
                else:
                    self.acons_bar[i_c, i_b].k = 0.0
                bar_hi = self.acons_bar[i_c, i_b].lam_hi
                bar_lo = self.acons_bar[i_c, i_b].lam_lo
                active = bar_hi
                if self.acons_info[i_c].lo < self.acons_info[i_c].hi:
                    active = 0.0
                    if qd.cast(self.acons[i_c, i_b].lam_hi, qd.f64) + w * kd * (cosv - hi_b) > 0.0:
                        active += bar_hi
                    if qd.cast(self.acons[i_c, i_b].lam_lo, qd.f64) + w * kd * (cosv - lo_b) < 0.0:
                        active += bar_lo
                push += w * kd * active
                self.acons_bar[i_c, i_b].k += w * (cosv - hi_b) * bar_hi
                for slot in qd.static(range(4)):
                    gsl = self._func_angle_slot_grad(slot, u_hat, v_hat, lu, lv, cosv)
                    for d in qd.static(range(3)):
                        qd.atomic_add(self.xb[vq[slot], i_b][d], push * gsl[d])

    @qd.kernel
    def _kernel_reverse_init(self, f: qd.i32):
        """The adjoint arriving at the end of the substep, and a clean accumulator for the predictor."""
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            self.xb[i_v, i_b] = self.adj[f + 1, i_v, i_b].pos + self.adj[f + 1, i_v, i_b].vel / self._substep_dt
            self.yb[i_v, i_b] = qd.Vector.zero(qd.f64, 3)
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            self.cons_bar[i_c, i_b].lam_hi = 0.0
            self.cons_bar[i_c, i_b].lam_lo = 0.0
            self.cons_bar[i_c, i_b].k = 0.0
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            self.acons_bar[i_c, i_b].lam_hi = 0.0
            self.acons_bar[i_c, i_b].lam_lo = 0.0
            self.acons_bar[i_c, i_b].k = 0.0

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
            self.xb[i_v, i_b] = xbar - self._func_diag_block_live(f, i_v, i_b).transpose() @ p
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
                self.adj[f, i_v, i_b].pos += qd.cast(self._damping / self._substep_dt, qd.f64) * (
                    qd.cast(K0_ii, qd.f64) @ p
                )
            self._func_scatter_reverse(f, i_v, i_b, p, dx)
            self._func_actuation_reverse(f, i_v, i_b, p, dx)
            for c in range(self.vc_offset[i_v], self.vc_offset[i_v + 1]):
                i_c = self.vc_cons[c]
                s = 0
                if self.vc_side[c] < 0.0:
                    s = 1
                e = (
                    self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos
                    - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
                )
                nrm = qd.cast(e / e.norm(), qd.f64)
                mult, violation_unused = self._func_constraint_mult(i_c, i_b, e.norm())
                # the multiplier enters the block through the constraint force, d g_s / d lam = sigma_s n
                sigma = qd.cast(self.vc_side[c], qd.f64)
                An_l = nrm.dot(p) * nrm.dot(dx)
                s_m = 1.0
                if mult < 0.0:
                    s_m = -1.0
                # the multiplier reaches the block twice: through the constraint force, and through the |mult| / dist
                # term of the Hessian, which is a tangent and is what the second sweep needs to be exact
                dH_dlam = s_m / qd.cast(e.norm(), qd.f64) * (p.dot(dx) - An_l)
                qd.atomic_add(self.cons_bar[i_c, i_b].lam_hi, -sigma * nrm.dot(p) - dH_dlam)
                # the stiffness enters the same block, through the multiplier and through both Hessian terms
                distd = qd.cast(e.norm(), qd.f64)
                c_hi = distd - qd.cast(self.cons_info[i_c].hi, qd.f64)
                An = nrm.dot(p) * nrm.dot(dx)
                s_mult = 1.0
                if mult < 0.0:
                    s_mult = -1.0
                dH_dk = An + s_mult * c_hi / distd * (p.dot(dx) - An)
                qd.atomic_add(self.cons_bar[i_c, i_b].k, -sigma * c_hi * nrm.dot(p) - dH_dk)
                for t in qd.static(range(2)):
                    j = self.cons_info[i_c].v[t]
                    tangent = self._func_distance_tangent(f, i_c, i_b, t, p, dx)
                    contribution = -tangent
                    if j != i_v:
                        contribution -= self._func_distance_block_live(f, i_c, i_b, s, t).transpose() @ p
                    for d in qd.static(range(3)):
                        qd.atomic_add(self.xb[j, i_b][d], contribution[d])
            for c in range(self.va_offset[i_v], self.va_offset[i_v + 1]):
                i_c = self.va_cons[c]
                slot = self.va_slot[c]
                u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
                vq = self.acons_info[i_c].v
                g_s = self._func_angle_slot_grad(slot, u_hat, v_hat, lu, lv, cosv)
                mult, violation_unused = self._func_angle_mult(i_c, i_b, qd.cast(cosv, gs.qd_float))
                own = u_hat
                scale = lu
                if slot >= 2:
                    own = v_hat
                    scale = lv
                s_m = 1.0
                if mult < 0.0:
                    s_m = -1.0
                proxy = (p.dot(dx) - own.dot(p) * own.dot(dx)) / (scale * scale)
                qd.atomic_add(self.acons_bar[i_c, i_b].lam_hi, -g_s.dot(p) - s_m * proxy)
                dHk = 0.0
                if mult != 0.0:
                    dHk = g_s.dot(p) * g_s.dot(dx)
                qd.atomic_add(
                    self.acons_bar[i_c, i_b].k,
                    -(cosv - qd.cast(self.acons_info[i_c].hi, qd.f64)) * (g_s.dot(p) + s_m * proxy) - dHk,
                )
                for t in qd.static(range(4)):
                    j = vq[t]
                    if j != i_v:
                        blk = self._func_angle_block_live(f, i_c, i_b, slot, t)
                        contribution = -(blk.transpose() @ p)
                        for d in qd.static(range(3)):
                            qd.atomic_add(self.xb[j, i_b][d], contribution[d])

    @qd.kernel
    def _kernel_reverse_finish(self, f: qd.i32):
        """y = x^t + h (v^t + h g), so the adjoint of the predictor reaches both the position and the velocity."""
        for i_v, i_b in qd.ndrange(self._n_vertices, self._B):
            total = self.xb[i_v, i_b] + self.yb[i_v, i_b]
            self.adj[f, i_v, i_b].pos += total - self.adj[f + 1, i_v, i_b].vel / self._substep_dt
            self.adj[f, i_v, i_b].vel += total * self._substep_dt

    def substep_pre_coupling_grad_sweep(self, f):
        """The solver-level adjoint of substep `f`: the sweeps and colours of the forward, in reverse.

        Within a colour the dual updates are undone first, because the forward ran them after the position step and
        they read the position it produced. Only then is the step itself undone, which puts every block at the state
        it linearised at."""
        self._kernel_reverse_init(f)
        for sweep in reversed(range(self._n_iterations)):
            if self._n_constraints > 0:
                self._kernel_restore_cons(f, sweep)
            for c in reversed(range(self._n_colors)):
                lo, hi = self._color_offsets[c], self._color_offsets[c + 1]
                if self._n_constraints > 0:
                    self._kernel_reverse_dual(f, lo, hi)
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
                    rhs = (
                        self.gbar[i_v, i_b]
                        - self._func_offdiag_apply(f, i_v, i_b, self.z)
                        - self._func_zeta_force(f, i_v, i_b)
                    )
                    self.z[i_v, i_b] = self._func_diag_block(f, i_v, i_b).transpose().inverse() @ rhs
            if qd.static(self._n_constraints > 0):
                for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
                    if self.cons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                        self.cons_zeta[i_c, i_b] += (
                            self._constraint_dual_relaxation
                            * self.cons_hist[f + 1, i_c, i_b].k_eff
                            * self._func_constraint_dot_z(f, i_c, i_b)
                        )
            if qd.static(self._n_angle_constraints > 0):
                for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
                    if self.acons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                        self.acons_zeta[i_c, i_b] += (
                            self._constraint_dual_relaxation
                            * self.acons_hist[f + 1, i_c, i_b].k_eff
                            * self._func_angle_dot_z(f, i_c, i_b)
                        )

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
                qd.atomic_max(
                    self.adj_residual[None],
                    self.cons_hist[f + 1, i_c, i_b].k_eff * qd.abs(self._func_constraint_dot_z(f, i_c, i_b)),
                )
        for i_c, i_b in qd.ndrange(self._n_angle_constraints, self._B):
            if self.acons_hist[f + 1, i_c, i_b].k_eff != 0.0:
                u_hat, v_hat, lu, lv, cosv = self._func_angle_geometry(f, i_c, i_b)
                g_max = 0.0
                for slot in qd.static(range(4)):
                    g_max = qd.max(g_max, self._func_angle_slot_grad(slot, u_hat, v_hat, lu, lv, cosv).norm())
                qd.atomic_max(
                    self.adj_residual[None],
                    self.acons_hist[f + 1, i_c, i_b].k_eff * g_max * qd.abs(self._func_angle_dot_z(f, i_c, i_b)),
                )

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
                self.adj[f, i_v, i_b].pos += kd_h * (
                    qd.cast(K0_ii, qd.f64) @ z + self._func_offdiag_rest(i_v, i_b, self.z)
                )
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
        if not self._grad_converge:
            # the forward ran a fixed number of sweeps, so the gradient of that computation is the reverse of it
            self.substep_pre_coupling_grad_sweep(f)
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
                Ds = qd.Matrix.cols(
                    [
                        self.verts[f + 1, v[1], i_b].pos - p0,
                        self.verts[f + 1, v[2], i_b].pos - p0,
                        self.verts[f + 1, v[3], i_b].pos - p0,
                    ]
                )
                l = (Ds @ self.elems_info[i_e].B_rest @ self.elems_info[i_e].fiber).norm()
                psi += 0.5 * self.elems_info[i_e].k_fiber * (l - 1.0) ** 2
            self.energy[i_b] += qd.cast(self.elems_info[i_e].vol_rest * psi, qd.f64)

    @qd.kernel
    def _kernel_constraint_energy(self, f: qd.i32):
        for i_c, i_b in qd.ndrange(self._n_constraints, self._B):
            e = (
                self.verts[f + 1, self.cons_info[i_c].v[0], i_b].pos
                - self.verts[f + 1, self.cons_info[i_c].v[1], i_b].pos
            )
            dist = e.norm()
            k = self.cons[i_c, i_b].k
            if self.cons_info[i_c].lo < self.cons_info[i_c].hi:
                c_hi = qd.max(dist - self.cons_info[i_c].hi, 0.0)
                c_lo = qd.min(dist - self.cons_info[i_c].lo, 0.0)
                self.energy[i_b] += qd.cast(
                    0.5 * k * (c_hi * c_hi + c_lo * c_lo)
                    + self.cons[i_c, i_b].lam_hi * c_hi
                    + self.cons[i_c, i_b].lam_lo * c_lo,
                    qd.f64,
                )
            else:
                C = dist - self.cons_info[i_c].hi
                self.energy[i_b] += qd.cast(0.5 * k * C * C + self.cons[i_c, i_b].lam_hi * C, qd.f64)

    def compute_energy(self, f):
        """Incremental potential of substep `f` at the current iterate, shape (B,). Non-increasing across sweeps.

        The sum covers the inertia and the tetrahedra only. It omits the rigid, attachment, contact and
        muscle-tendon terms, and it has no shell term, so it refuses a scene that holds shell elements rather
        than return a number that is quietly missing most of the energy."""
        if self.n_triangles > 0 or self.n_stencils > 0:
            gs.raise_exception(
                "compute_energy has no shell term, so it would under-report a scene with triangles or bending "
                "stencils. Read the tissue diagnostics instead, or add the shell energy to _kernel_compute_energy."
            )
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
            if self.rigid_attachment is not None:
                rigid = self.rigid_attachment.rigid
                if self.rigid_attachment.is_articulated:
                    kernel_begin_articulation(
                        f,
                        self,
                        self.rigid_attachment,
                        rigid.dyn_state,
                        rigid.dyn_info,
                        rigid.rigid_info,
                        rigid.rigid_config,
                    )
                else:
                    kernel_begin_attachment(f, self, self.rigid_attachment, rigid.dyn_state, rigid.rigid_info)
            if self.tissue_attachment is not None:
                kernel_begin_tissue_attachment(f, self, self.tissue_attachment)
            # every substep, gradients or not: at two sweeps the primal is never converged within a substep, and this
            # decay is the dual damping that stops the multipliers integrating stale violations (once per step, as the
            # AVBD paper does per frame, the ladder python's spine stretch went from 0.2 to 8 percent)
            if self._n_constraints > 0 or self._n_angle_constraints > 0:
                self._kernel_warm_start()
            self._kernel_predict(f)
            if self.contact is not None:
                rigid = self._sim.rigid_solver
                if self.contact.n_prescribed:
                    # f indexes the two-frame buffer without gradients, so the position within the step comes
                    # from the global substep counter
                    substep_in_step = self._sim.cur_substep_global % self._sim.substeps
                    kernel_prescribe_links(
                        (substep_in_step + 1) / self._sim.substeps,
                        substep_in_step == 0,
                        self,
                        self.contact,
                        rigid.dyn_state,
                        rigid.dyn_info,
                        rigid.rigid_info,
                        rigid.rigid_config,
                    )
                kernel_begin_contact(f, self, self.contact, rigid.dyn_state)
            if self.mtu is not None:
                kernel_begin_mtu(f, self, self.mtu, self._sim.rigid_solver.dyn_state)
            if self._self_thickness > 0.0:
                self._kernel_build_hash(f)
                # Reading the flag waits for the device, so it is read once a step, not once a substep. An
                # overflowed cell is therefore reported up to one step late, which is still loud and still stops
                # the run; a wait every substep cost about five times the whole contact solve.
                if f == 0 and self.cell_overflow[None]:
                    gs.raise_exception(
                        f"More than {self._hash_cap} vertices in one self-collision grid cell of "
                        f"{self._self_cell:.4f} m. Raise the cell capacity or the cell size."
                    )
            self.solve(f)
            self._kernel_update_velocity(f)
            self._kernel_tissue_diagnostics(f, self._sim.cur_substep_global)
            if self.contact is not None:
                kernel_end_contact(
                    f, self._sim.cur_substep_global, self, self.contact, self._sim.rigid_solver.dyn_state
                )
            if self.mtu is not None:
                kernel_end_mtu(f, self, self.mtu, self._sim.rigid_solver.dyn_state)
            if self.rigid_attachment is not None:
                if self.rigid_attachment.is_articulated:
                    kernel_end_articulation(
                        self._substep_dt, self, self.rigid_attachment, rigid.dyn_state, rigid.rigid_info
                    )
                else:
                    kernel_end_attachment(
                        self, self.rigid_attachment, rigid.dyn_state, rigid.rigid_info, self._substep_dt
                    )
                rigid.commit_vbd_link()

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
            envs_idx = sanitize_index(envs_idx, -1, self._B, 0, "envs_idx")
            kernel_set_vertex_state(f, envs_idx, state.pos, state.vel, self.verts)
            kernel_set_muscle_state(envs_idx, state.muscle_actuation, self.muscle_actu)
            if self.rigid_attachment is not None:
                kernel_set_attachment_state(
                    envs_idx, state.attachment_multiplier, state.attachment_stiffness, self.rigid_attachment.state
                )
            if self.tissue_attachment is not None:
                kernel_set_tissue_attachment_state(
                    envs_idx,
                    state.tissue_attachment_multiplier,
                    state.tissue_attachment_stiffness,
                    self.tissue_attachment.state,
                )
            kernel_clear_env_failure(
                envs_idx, self.env_failed, self.failed_substep, self.inverted_streak, self.tissue_errno
            )
            if self.contact is not None:
                kernel_reset_contact(envs_idx, self.contact, self._sim.rigid_solver.dyn_state)
            if self.mtu is not None:
                kernel_reset_mtu(envs_idx, self.mtu)
                if state.prescribed_start_pos is not None:
                    kernel_set_prescribed_state(
                        envs_idx,
                        state.prescribed_start_pos,
                        state.prescribed_start_quat,
                        state.prescribed_target_pos,
                        state.prescribed_target_quat,
                        self.contact,
                    )

    def get_state(self, f):
        if not self.is_active:
            return None
        if bool((qd_to_torch(self.env_failed) != 0).any()):
            gs.raise_exception(
                "An environment failed a contact step; its state is diagnostic only. Reset it before taking a snapshot."
            )
        state = VBDSolverState(self._scene)
        self._kernel_get_state(f, state.pos, state.vel)
        state.muscle_actuation = qd_to_torch(self.muscle_actu, transpose=True, copy=True).contiguous()
        if self.rigid_attachment is not None:
            state.attachment_multiplier = qd_to_torch(
                self.rigid_attachment.state.multiplier, transpose=True, copy=True
            ).contiguous()
            state.attachment_stiffness = qd_to_torch(
                self.rigid_attachment.state.stiffness, transpose=True, copy=True
            ).contiguous()
        if self.tissue_attachment is not None:
            state.tissue_attachment_multiplier = qd_to_torch(
                self.tissue_attachment.state.multiplier, transpose=True, copy=True
            ).contiguous()
            state.tissue_attachment_stiffness = qd_to_torch(
                self.tissue_attachment.state.stiffness, transpose=True, copy=True
            ).contiguous()
        if self.contact is not None:
            state.prescribed_start_pos = qd_to_torch(
                self.contact.prescribed_start.pos, transpose=True, copy=True
            ).contiguous()
            state.prescribed_start_quat = qd_to_torch(
                self.contact.prescribed_start.quat, transpose=True, copy=True
            ).contiguous()
            state.prescribed_target_pos = qd_to_torch(
                self.contact.prescribed_target.pos, transpose=True, copy=True
            ).contiguous()
            state.prescribed_target_quat = qd_to_torch(
                self.contact.prescribed_target.quat, transpose=True, copy=True
            ).contiguous()
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
    def n_triangles(self):
        return sum(entity.n_triangles for entity in self._entities)

    @property
    def n_stencils(self):
        return sum(entity.n_stencils for entity in self._entities)

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


@qd.kernel
def kernel_clear_env_failure(
    envs_idx: qd.types.ndarray(),
    env_failed: qd.template(),
    failed_substep: qd.template(),
    inverted_streak: qd.template(),
    tissue_errno: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        env_failed[envs_idx[i_b_]] = 0
        failed_substep[envs_idx[i_b_]] = -1
        inverted_streak[envs_idx[i_b_]] = 0
        tissue_errno[envs_idx[i_b_]] = 0
