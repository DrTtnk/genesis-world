"""Routed Hill muscle-tendon units, ligaments and rotary restraints for vertex block descent (VBD).

A unit follows a polyline route through ordered anchors p_0 .. p_n. Each anchor sits in the world, on a rigid
link at a body offset, or inside the tissue at barycentric weights over four vertices. The route length and its
unit segment directions are

    L = sum_i l_i,   l_i = ||p_(i+1) - p_i||,   u_i = (p_(i+1) - p_i) / l_i,

and a tension T >= 0 pulls every anchor along the route with the generalized force -T dL/dp_i, where

    dL/dp_i = u_(i-1) - u_i        (u_(-1) = u_n = 0 at the ends).

The curvature block each owner receives is dT/dL g g^T + T d2L/dx2 restricted to that owner. The route Hessian
is positive semidefinite for T >= 0, so only the scalar dT/dL needs the positive part. Through the implicitly
solved fibre that scalar is the series combination of two springs,

    dT/dL = k_T A' / (k_T + A'),   k_T = f_max / (eps_ref l_slack),   A' = dA/dl,

which is negative on the descending limb of the force-length bell when the fibre lags its route. Every formula
here, the closed-form fibre slope A' included, is proved against torch autograd on random inputs in
`spikes/verify_avbd_mtu_math.py` of the application repository (34 cases).

The Hill law itself is `spikes/muscle.py` ported unchanged, with one deliberate difference: the reference finds
the tendon-fibre root with a central difference at h = 1e-7 l_opt, which is unusable in float32, so the kernel
uses the closed-form slope. The spike shows both reach the same root within 1e-10 l_opt and 1e-8 N.

Activation and the fibre state advance once per substep, at its start. The sweeps then re-evaluate the route
length and the tendon force at the moving pose against that fixed previous fibre state, so the muscle a command
produces does not depend on the sweep count.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.array_class import DynState
from genesis.utils.misc import qd_to_torch

W = 0.45  # force-length bell width
W_PE = 0.6  # parallel elastic engagement width
K = 0.25  # Hill curvature a / f_max
N = 1.38  # eccentric plateau
K_E = (N - 1.0) / (0.05 * N) - 1.0  # f_v reaches 95 percent of N at v_ce = v_max
EPS_REF = 0.04  # tendon strain at f_max
TAU_ACT, TAU_DEACT = 0.05, 0.15  # activation and deactivation time constants, s

KIND_WORLD, KIND_LINK, KIND_TISSUE = 0, 1, 2
UNIT_HILL, UNIT_LIGAMENT = 0, 1
NEWTON_ITERATIONS = 6


@dataclass(frozen=True)
class HillParameters:
    """Maximum isometric force (N), optimal fibre length (m), tendon slack length (m) and maximum contraction
    speed (m/s) of one muscle-tendon unit."""

    f_max: float
    l_opt: float
    l_slack: float
    v_max: float


@dataclass(frozen=True)
class WorldAnchor:
    pos: tuple


@dataclass(frozen=True)
class LinkAnchor:
    link: object
    local_pos: tuple


@dataclass(frozen=True)
class TissueAnchor:
    """Four vertices of one tetrahedron of a VBD entity and the barycentric weights of the anchor in it."""

    entity: object
    vertices: tuple
    weights: tuple


class MTUState(NamedTuple):
    """Per environment and unit: activation, fibre length (m), route length (m), fibre velocity (m/s) and
    tension (N). A ligament has no fibre, so its fibre length and velocity are zero."""

    activation: torch.Tensor
    fibre_length: torch.Tensor
    route_length: torch.Tensor
    fibre_velocity: torch.Tensor
    tension: torch.Tensor


class VBDMTU:
    def __init__(self, solver, units, restraints):
        self.solver = solver
        self.n_units = len(units)
        self.n_restraints = len(restraints)
        rigid = solver._sim.rigid_solver

        anchor_kind, anchor_link, anchor_local, anchor_verts, anchor_weights, anchor_unit = [], [], [], [], [], []
        offsets = [0]
        for i_m, (_, anchors, _, _) in enumerate(units):
            for anchor in anchors:
                anchor_unit.append(i_m)
                if isinstance(anchor, WorldAnchor):
                    anchor_kind.append(KIND_WORLD)
                    anchor_link.append(-1)
                    anchor_local.append(anchor.pos)
                    anchor_verts.append((0, 0, 0, 0))
                    anchor_weights.append((0.0, 0.0, 0.0, 0.0))
                elif isinstance(anchor, LinkAnchor):
                    anchor_kind.append(KIND_LINK)
                    anchor_link.append(anchor.link.idx)
                    anchor_local.append(anchor.local_pos)
                    anchor_verts.append((0, 0, 0, 0))
                    anchor_weights.append((0.0, 0.0, 0.0, 0.0))
                else:
                    anchor_kind.append(KIND_TISSUE)
                    anchor_link.append(-1)
                    anchor_local.append((0.0, 0.0, 0.0))
                    anchor_verts.append(tuple(anchor.entity.v_start + v for v in anchor.vertices))
                    anchor_weights.append(anchor.weights)
            offsets.append(len(anchor_kind))
        self.n_anchors = len(anchor_kind)

        anchor_type = qd.types.struct(
            kind=gs.qd_int, link=gs.qd_int, local=gs.qd_vec3, verts=gs.qd_ivec4, weights=gs.qd_vec4
        )
        self.anchor = anchor_type.field(shape=self.n_anchors, layout=qd.Layout.SOA)
        self.anchor.kind.from_numpy(np.array(anchor_kind, dtype=gs.np_int))
        self.anchor.link.from_numpy(np.array(anchor_link, dtype=gs.np_int))
        self.anchor.local.from_numpy(np.array(anchor_local, dtype=gs.np_float))
        self.anchor.verts.from_numpy(np.array(anchor_verts, dtype=gs.np_int))
        self.anchor.weights.from_numpy(np.array(anchor_weights, dtype=gs.np_float))
        self.anchor_unit = qd.field(dtype=gs.qd_int, shape=self.n_anchors)
        self.anchor_unit.from_numpy(np.array(anchor_unit, dtype=gs.np_int))

        unit_type = qd.types.struct(
            kind=gs.qd_int,
            first=gs.qd_int,
            last=gs.qd_int,
            f_max=gs.qd_float,
            l_opt=gs.qd_float,
            l_slack=gs.qd_float,
            v_max=gs.qd_float,
            k_tendon=gs.qd_float,
            activation0=gs.qd_float,
            fibre0=gs.qd_float,
        )
        self.unit = unit_type.field(shape=self.n_units, layout=qd.Layout.SOA)
        kinds, firsts, lasts, f_max, l_opt, l_slack, v_max, k_tendon = [], [], [], [], [], [], [], []
        activation0, fibre0 = [], []
        for i_m, (kind, _, parameters, initial) in enumerate(units):
            kinds.append(kind)
            firsts.append(offsets[i_m])
            lasts.append(offsets[i_m + 1])
            activation0.append(initial[0])
            fibre0.append(parameters.l_opt if kind == UNIT_HILL and initial[1] is None else (initial[1] or 0.0))
            if kind == UNIT_HILL:
                f_max.append(parameters.f_max)
                l_opt.append(parameters.l_opt)
                l_slack.append(parameters.l_slack)
                v_max.append(parameters.v_max)
                k_tendon.append(parameters.f_max / (EPS_REF * parameters.l_slack))
            else:
                stiffness, slack = parameters
                f_max.append(0.0)
                l_opt.append(0.0)
                l_slack.append(slack)
                v_max.append(0.0)
                k_tendon.append(stiffness)
        self.unit.kind.from_numpy(np.array(kinds, dtype=gs.np_int))
        self.unit.first.from_numpy(np.array(firsts, dtype=gs.np_int))
        self.unit.last.from_numpy(np.array(lasts, dtype=gs.np_int))
        self.unit.f_max.from_numpy(np.array(f_max, dtype=gs.np_float))
        self.unit.l_opt.from_numpy(np.array(l_opt, dtype=gs.np_float))
        self.unit.l_slack.from_numpy(np.array(l_slack, dtype=gs.np_float))
        self.unit.v_max.from_numpy(np.array(v_max, dtype=gs.np_float))
        self.unit.k_tendon.from_numpy(np.array(k_tendon, dtype=gs.np_float))
        self.unit.activation0.from_numpy(np.array(activation0, dtype=gs.np_float))
        self.unit.fibre0.from_numpy(np.array(fibre0, dtype=gs.np_float))

        state_type = qd.types.struct(
            initialised=gs.qd_int,
            excitation=gs.qd_float,
            activation=gs.qd_float,
            fibre=gs.qd_float,
            fibre_previous=gs.qd_float,
            length=gs.qd_float,
            tension=gs.qd_float,
        )
        self.state = state_type.field(shape=(self.n_units, solver._B), layout=qd.Layout.SOA)

        # A tissue vertex reaches its anchors through this CSR; the payload is anchor * 4 + the corner index.
        pairs = [
            (verts[corner], 4 * i_a + corner)
            for i_a, (kind, verts) in enumerate(zip(anchor_kind, anchor_verts))
            if kind == KIND_TISSUE
            for corner in range(4)
            if anchor_weights[i_a][corner] != 0.0
        ]
        pairs.sort()
        counts = np.zeros(solver._n_vertices + 1, dtype=gs.np_int)
        for vertex, _ in pairs:
            counts[vertex + 1] += 1
        self.vert_anchor_offset = qd.field(dtype=gs.qd_int, shape=solver._n_vertices + 1)
        self.vert_anchor_offset.from_numpy(np.cumsum(counts).astype(gs.np_int))
        self.vert_anchor = qd.field(dtype=gs.qd_int, shape=max(len(pairs), 1))
        self.vert_anchor.from_numpy(np.array([slot for _, slot in pairs] or [0], dtype=gs.np_int))

        n_links = max(rigid.n_links, 1)
        link_pairs = sorted(
            (link, i_a) for i_a, (kind, link) in enumerate(zip(anchor_kind, anchor_link)) if kind == KIND_LINK
        )
        counts = np.zeros(n_links + 1, dtype=gs.np_int)
        for link, _ in link_pairs:
            counts[link + 1] += 1
        self.link_anchor_offset = qd.field(dtype=gs.qd_int, shape=n_links + 1)
        self.link_anchor_offset.from_numpy(np.cumsum(counts).astype(gs.np_int))
        self.link_anchor = qd.field(dtype=gs.qd_int, shape=max(len(link_pairs), 1))
        self.link_anchor.from_numpy(np.array([i_a for _, i_a in link_pairs] or [0], dtype=gs.np_int))

        # dof_moves_link[i_d, i_l]: the coordinate i_d lies between link i_l and the root, so moving it moves
        # every anchor on that link.
        moves = np.zeros((max(rigid.n_dofs, 1), n_links), dtype=gs.np_int)
        for link in rigid.links:
            i_l = link.idx
            while True:
                moves[link.dof_start : link.dof_end, i_l] = 1
                if link.parent_idx < 0:
                    break
                link = rigid.links[link.parent_idx]
        self.dof_moves_link = qd.field(dtype=gs.qd_int, shape=moves.shape)
        self.dof_moves_link.from_numpy(moves)

        self.restraint = qd.types.struct(dof=gs.qd_int, stiffness=gs.qd_float, rest=gs.qd_float).field(
            shape=max(self.n_restraints, 1), layout=qd.Layout.SOA
        )
        if self.n_restraints:
            self.restraint.dof.from_numpy(np.array([r[0] for r in restraints], dtype=gs.np_int))
            self.restraint.stiffness.from_numpy(np.array([r[1] for r in restraints], dtype=gs.np_float))
            self.restraint.rest.from_numpy(np.array([r[2] for r in restraints], dtype=gs.np_float))

        # World position of every link anchor at the current iterate. A rigid block moves its anchors without
        # touching the tissue state, so the sweeps read them from here and every block that moves a link
        # refreshes them, exactly as vbd_contact does with its rigid contact vertices.
        self.anchor_pos = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_anchors, solver._B))
        # world pull on every anchor at the end of the last substep, for the reaction readback
        self.anchor_force = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_anchors, solver._B))

    def state_readback(self):
        length = qd_to_torch(self.state.length, transpose=True, copy=True)
        fibre = qd_to_torch(self.state.fibre, transpose=True, copy=True)
        previous = qd_to_torch(self.state.fibre_previous, transpose=True, copy=True)
        return MTUState(
            activation=qd_to_torch(self.state.activation, transpose=True, copy=True),
            fibre_length=fibre,
            route_length=length,
            fibre_velocity=(fibre - previous) / self.solver._substep_dt,
            tension=qd_to_torch(self.state.tension, transpose=True, copy=True),
        )

    def anchor_forces(self):
        """World pull on every anchor of every unit at the last substep, [B, M, A_max, 3] with unused rows
        zero. The rows of one unit sum to zero: the route is an internal force."""
        forces = qd_to_torch(self.anchor_force, transpose=True, copy=True)
        first = qd_to_torch(self.unit.first).to(torch.int64)
        last = qd_to_torch(self.unit.last).to(torch.int64)
        width = int((last - first).max())
        out = torch.zeros(forces.shape[0], self.n_units, width, 3, dtype=forces.dtype, device=forces.device)
        for i_m in range(self.n_units):
            out[:, i_m, : int(last[i_m] - first[i_m])] = forces[:, int(first[i_m]) : int(last[i_m])]
        return out


@qd.func
def func_refresh_mtu_anchors(i_b, mtu: qd.template(), dyn_state: DynState):
    """Cached world positions of every link anchor from the current link poses."""
    for i_a in range(mtu.n_anchors):
        if mtu.anchor[i_a].kind == KIND_LINK:
            i_l = mtu.anchor[i_a].link
            mtu.anchor_pos[i_a, i_b] = gu.qd_transform_by_trans_quat(
                mtu.anchor[i_a].local, dyn_state.links.pos[i_l, i_b], dyn_state.links.quat[i_l, i_b]
            )


@qd.func
def func_refresh_mtu_link_anchors(i_l, i_b, pos, quat, mtu: qd.template()):
    """Cached world positions of the anchors of one link, from a given pose."""
    for slot in range(mtu.link_anchor_offset[i_l], mtu.link_anchor_offset[i_l + 1]):
        i_a = mtu.link_anchor[slot]
        mtu.anchor_pos[i_a, i_b] = gu.qd_transform_by_trans_quat(mtu.anchor[i_a].local, pos, quat)


@qd.func
def func_anchor_pos(f, i_a, i_b, solver: qd.template(), mtu: qd.template()):
    """World position of one anchor at the current iterate."""
    pos = mtu.anchor[i_a].local
    if mtu.anchor[i_a].kind == KIND_LINK:
        pos = mtu.anchor_pos[i_a, i_b]
    elif mtu.anchor[i_a].kind == KIND_TISSUE:
        pos = gs.qd_vec3(0.0, 0.0, 0.0)
        for corner in qd.static(range(4)):
            pos += mtu.anchor[i_a].weights[corner] * solver.verts[f + 1, mtu.anchor[i_a].verts[corner], i_b].pos
    return pos


@qd.func
def func_route_length(f, i_m, i_b, solver: qd.template(), mtu: qd.template()):
    length = gs.qd_float(0.0)
    for i_a in range(mtu.unit[i_m].first, mtu.unit[i_m].last - 1):
        length += (
            func_anchor_pos(f, i_a + 1, i_b, solver, mtu)
            - func_anchor_pos(f, i_a, i_b, solver, mtu)
        ).norm()
    return length


@qd.func
def func_fibre_slope(fibre, fibre_previous, activation, i_m, dt, mtu: qd.template()):
    """A'(l): the closed-form slope of the fibre's own force, active plus passive, at a fixed route length."""
    l_opt = mtu.unit[i_m].l_opt
    v_max = mtu.unit[i_m].v_max
    f_max = mtu.unit[i_m].f_max
    ratio = fibre / l_opt - 1.0
    bell = qd.exp(-((ratio / W) ** 2))
    bell_slope = -2.0 * ratio * bell / (W * W * l_opt)
    x = (fibre - fibre_previous) / (dt * v_max)
    velocity_slope = (1.0 + 1.0 / K) / (1.0 - x / K) ** 2
    if x > 0.0:
        velocity_slope = (N - 1.0) * K_E / (1.0 + K_E * x) ** 2
    velocity = func_force_velocity(x)
    active = activation * f_max * (bell_slope * velocity + bell * velocity_slope / (dt * v_max))
    return active + 2.0 * f_max * qd.max(ratio / W_PE, 0.0) / (W_PE * l_opt)


@qd.func
def func_force_velocity(x):
    """x = v_ce / v_max, positive when the fibre lengthens."""
    value = (1.0 + x) / (1.0 - x / K)
    if x > 0.0:
        value = N - (N - 1.0) / (1.0 + K_E * x)
    return value


@qd.func
def func_fibre_force(fibre, fibre_previous, activation, i_m, dt, mtu: qd.template()):
    l_opt = mtu.unit[i_m].l_opt
    f_max = mtu.unit[i_m].f_max
    bell = qd.exp(-(((fibre / l_opt - 1.0) / W) ** 2))
    velocity = func_force_velocity((fibre - fibre_previous) / (dt * mtu.unit[i_m].v_max))
    passive = f_max * qd.max((fibre / l_opt - 1.0) / W_PE, 0.0) ** 2
    return activation * f_max * bell * velocity + passive


@qd.func
def func_solve_fibre(length, fibre_previous, activation, i_m, dt, mtu: qd.template()):
    """Backward-Euler root of the tendon-fibre balance at this route length, by Newton with the closed-form
    slope. The root lives where the tendon is taut, so every iterate is clamped there."""
    k_tendon = mtu.unit[i_m].k_tendon
    taut = length - mtu.unit[i_m].l_slack
    fibre = qd.min(fibre_previous, taut)
    for _ in range(NEWTON_ITERATIONS):
        residual = (
            mtu.unit[i_m].f_max * qd.max(length - fibre - mtu.unit[i_m].l_slack, 0.0) / (EPS_REF * mtu.unit[i_m].l_slack)
            - func_fibre_force(fibre, fibre_previous, activation, i_m, dt, mtu)
        )
        slope = -(k_tendon + func_fibre_slope(fibre, fibre_previous, activation, i_m, dt, mtu))
        fibre = qd.min(fibre - residual / slope, taut)
    return fibre


@qd.func
def func_unit_tension(f, i_m, i_b, length, solver: qd.template(), mtu: qd.template()):
    """Tension and total stiffness dT/dL of one unit at the given route length, with its fibre state fixed at
    the value the substep started from. The stiffness is the series combination of the tendon and the fibre."""
    tension = gs.qd_float(0.0)
    stiffness = gs.qd_float(0.0)
    if mtu.unit[i_m].kind == UNIT_LIGAMENT:
        extension = length - mtu.unit[i_m].l_slack
        if extension > 0.0:
            tension = mtu.unit[i_m].k_tendon * extension
            stiffness = mtu.unit[i_m].k_tendon
    else:
        dt = solver._substep_dt
        fibre_previous = mtu.state[i_m, i_b].fibre_previous
        activation = mtu.state[i_m, i_b].activation
        fibre = func_solve_fibre(length, fibre_previous, activation, i_m, dt, mtu)
        extension = length - fibre - mtu.unit[i_m].l_slack
        if extension > 0.0:
            tension = mtu.unit[i_m].f_max * extension / (EPS_REF * mtu.unit[i_m].l_slack)
            k_tendon = mtu.unit[i_m].k_tendon
            slope = func_fibre_slope(fibre, fibre_previous, activation, i_m, dt, mtu)
            # two springs in series; the positive part is the projection the block needs when the fibre lags
            # its route far enough to sit on the descending limb of the bell
            stiffness = qd.max(k_tendon * slope / (k_tendon + slope), 0.0)
    return tension, stiffness


@qd.func
def func_anchor_terms(f, i_a, i_b, solver: qd.template(), mtu: qd.template()):
    """Pull on one anchor and the diagonal curvature block it carries: -T (u_(i-1) - u_i) and
    dT/dL g g^T + T (P_(i-1) / l_(i-1) + P_i / l_i)."""
    i_m = mtu.anchor_unit[i_a]
    length = func_route_length(f, i_m, i_b, solver, mtu)
    tension, stiffness = func_unit_tension(f, i_m, i_b, length, solver, mtu)
    gradient = gs.qd_vec3(0.0, 0.0, 0.0)
    curvature = qd.Matrix.zero(gs.qd_float, 3, 3)
    if tension > 0.0:
        here = func_anchor_pos(f, i_a, i_b, solver, mtu)
        identity = qd.Matrix.identity(gs.qd_float, 3)
        if i_a > mtu.unit[i_m].first:
            segment = here - func_anchor_pos(f, i_a - 1, i_b, solver, mtu)
            l = segment.norm()
            u = segment / l
            gradient += u
            curvature += tension * (identity - u.outer_product(u)) / l
        if i_a < mtu.unit[i_m].last - 1:
            segment = func_anchor_pos(f, i_a + 1, i_b, solver, mtu) - here
            l = segment.norm()
            u = segment / l
            gradient -= u
            curvature += tension * (identity - u.outer_product(u)) / l
        curvature += stiffness * gradient.outer_product(gradient)
    return -tension * gradient, curvature


@qd.func
def func_mtu_vertex_terms(f, i_v, i_b, solver: qd.template(), mtu: qd.template()):
    """Force and curvature block that every route anchored on tissue vertex i_v applies to it."""
    force = gs.qd_vec3(0.0, 0.0, 0.0)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    for slot in range(mtu.vert_anchor_offset[i_v], mtu.vert_anchor_offset[i_v + 1]):
        i_a = mtu.vert_anchor[slot] // 4
        weight = mtu.anchor[i_a].weights[mtu.vert_anchor[slot] % 4]
        anchor_force, anchor_hessian = func_anchor_terms(f, i_a, i_b, solver, mtu)
        force += weight * anchor_force
        hessian += weight * weight * anchor_hessian
    return force, hessian


@qd.func
def func_mtu_link_terms(f, i_l, i_b, origin, solver: qd.template(), mtu: qd.template()):
    """Wrench about `origin` and 6x6 curvature block of every route anchored on link i_l, for a free link with
    the world-frame rotation increment of the attachment block."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    for slot in range(mtu.link_anchor_offset[i_l], mtu.link_anchor_offset[i_l + 1]):
        i_a = mtu.link_anchor[slot]
        force, hessian = func_anchor_terms(f, i_a, i_b, solver, mtu)
        r = func_anchor_pos(f, i_a, i_b, solver, mtu) - origin
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
def func_mtu_dof_terms(f, i_d, i_b, axis, pivot, position, solver: qd.template(), mtu: qd.template()):
    """Generalized force and curvature of every route anchored on a link that hinge coordinate i_d moves, plus
    the rotary restraints on that coordinate. The anchor Jacobian is axis x (p - pivot)."""
    force = gs.qd_float(0.0)
    curvature = gs.qd_float(0.0)
    for i_l in range(mtu.dof_moves_link.shape[1]):
        if mtu.dof_moves_link[i_d, i_l]:
            for slot in range(mtu.link_anchor_offset[i_l], mtu.link_anchor_offset[i_l + 1]):
                i_a = mtu.link_anchor[slot]
                anchor_force, anchor_hessian = func_anchor_terms(f, i_a, i_b, solver, mtu)
                jacobian = axis.cross(func_anchor_pos(f, i_a, i_b, solver, mtu) - pivot)
                force += jacobian.dot(anchor_force)
                curvature += jacobian.dot(anchor_hessian @ jacobian)
    for i_r in range(mtu.n_restraints):
        if mtu.restraint[i_r].dof == i_d:
            force -= mtu.restraint[i_r].stiffness * (position - mtu.restraint[i_r].rest)
            curvature += mtu.restraint[i_r].stiffness
    return force, curvature


@qd.kernel
def kernel_begin_mtu(f: int, solver: qd.template(), mtu: qd.template(), dyn_state: DynState):
    """Advance the muscle state once, at the start of the substep: the activation follows the held excitation,
    and the fibre state of the sweeps is frozen at the value the previous substep left."""
    for i_b in range(solver._B):
        if not solver.env_failed[i_b]:
            func_refresh_mtu_anchors(i_b, mtu, dyn_state)
    for i_m, i_b in qd.ndrange(mtu.n_units, solver._B):
        if not solver.env_failed[i_b] and mtu.unit[i_m].kind == UNIT_HILL:
            if not mtu.state[i_m, i_b].initialised:
                # The fibre starts where the model asked, but a route shorter than that plus l_slack cannot
                # hold it: the tendon carries no compression. Clamp to the taut boundary of the route the unit
                # was built at, or the first substep reads a fibre velocity that is an artifact of the initial
                # condition rather than of the motion.
                taut = func_route_length(f, i_m, i_b, solver, mtu) - mtu.unit[i_m].l_slack
                mtu.state[i_m, i_b].fibre = qd.min(mtu.unit[i_m].fibre0, taut)
                mtu.state[i_m, i_b].initialised = 1
            activation = mtu.state[i_m, i_b].activation
            excitation = mtu.state[i_m, i_b].excitation
            tau = TAU_DEACT
            if excitation >= activation:
                tau = TAU_ACT
            mtu.state[i_m, i_b].activation = activation + (
                1.0 - qd.exp(-solver._substep_dt / tau)
            ) * (excitation - activation)
            mtu.state[i_m, i_b].fibre_previous = mtu.state[i_m, i_b].fibre


@qd.kernel
def kernel_end_mtu(f: int, solver: qd.template(), mtu: qd.template(), dyn_state: DynState):
    """Record the route length, the settled fibre, the tension and the pull on every anchor at the pose the
    substep ended with. These are the reported quantities; the sweeps never read them back."""
    for i_b in range(solver._B):
        if not solver.env_failed[i_b]:
            func_refresh_mtu_anchors(i_b, mtu, dyn_state)
    for i_m, i_b in qd.ndrange(mtu.n_units, solver._B):
        if not solver.env_failed[i_b]:
            length = func_route_length(f, i_m, i_b, solver, mtu)
            mtu.state[i_m, i_b].length = length
            if mtu.unit[i_m].kind == UNIT_HILL:
                mtu.state[i_m, i_b].fibre = func_solve_fibre(
                    length,
                    mtu.state[i_m, i_b].fibre_previous,
                    mtu.state[i_m, i_b].activation,
                    i_m,
                    solver._substep_dt,
                    mtu,
                )
            tension, _ = func_unit_tension(f, i_m, i_b, length, solver, mtu)
            mtu.state[i_m, i_b].tension = tension
    for i_a, i_b in qd.ndrange(mtu.n_anchors, solver._B):
        if not solver.env_failed[i_b]:
            force, _ = func_anchor_terms(f, i_a, i_b, solver, mtu)
            mtu.anchor_force[i_a, i_b] = force


@qd.kernel
def kernel_set_excitation(excitation: qd.types.ndarray(), mtu: qd.template()):
    for i_b, i_m in qd.ndrange(excitation.shape[0], excitation.shape[1]):
        mtu.state[i_m, i_b].excitation = excitation[i_b, i_m]


@qd.kernel
def kernel_reset_mtu(envs_idx: qd.types.ndarray(), mtu: qd.template()):
    for i_m, i in qd.ndrange(mtu.n_units, envs_idx.shape[0]):
        i_b = envs_idx[i]
        mtu.state[i_m, i_b].excitation = 0.0
        mtu.state[i_m, i_b].activation = mtu.unit[i_m].activation0
        mtu.state[i_m, i_b].initialised = 0
        mtu.state[i_m, i_b].fibre = mtu.unit[i_m].fibre0
        mtu.state[i_m, i_b].fibre_previous = mtu.unit[i_m].fibre0
        mtu.state[i_m, i_b].length = 0.0
        mtu.state[i_m, i_b].tension = 0.0
