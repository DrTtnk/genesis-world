"""Native VIPER rods: the reference's block Gauss-Newton, coloured and batched on the device.

The energy is `vbd_rod.RodModel`'s incremental potential, residual row for row. Its blocks are the reference's
too: a node (3 DOF), a segment frame (3 DOF, the world-frame Cayley increment of `vbd_rod.retract`), and the
coupled scales of one rod. Their gradients and Gauss-Newton curvatures are closed forms, checked against Torch AD
of the reference residual on 400 random blocks in `spikes/verify_viper_native_blocks.py` of the application
repository; the sign convention for the curvature is

    p = conj(q_a) q_b, sigma = sign(p_0), kappa = 2 sigma Im(p) / dual,
    d kappa / d delta_b = +(sigma / dual)(p_0 I - [p]x) R_a^T,   d kappa / d delta_a = -(the same),
    d (R e_c) / d delta = -[R e_c]x.

What changes is the order. A rod node is an ordinary VBD vertex: its rows are linear in its position, so the vertex
solve's single Newton step is exact for them, and every other term a vertex can carry (Hill units, attachments,
glue, contact) reaches it the usual way. Frames are solved in two colours, even and odd segments, since only
neighbouring frames share a row; each takes the reference's Armijo backtracking on the rows it touches. The scales
of one rod couple through five-point stencils, so each rod solves its banded block exactly, with the reference's
positivity and descent checks, one thread per rod and environment.
"""

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.array_class import ErrorCode

MAX_ROD_NODES = 64
LINE_SEARCH_STEPS = 24
# the start and every halving of the scale step, evaluated side by side (func_trial_fraction)
N_TRIALS = LINE_SEARCH_STEPS + 1
N_ENERGY_TERMS, N_MAGNITUDE_TERMS = 7, 5


class VBDRodNative:
    def __init__(self, solver, entities, models):
        self.solver = solver
        self.n_rods = len(entities)
        B = solver._B
        counts = [len(model.length) for model in models]
        if max(counts) + 1 > MAX_ROD_NODES:
            gs.raise_exception(f"A native rod has at most {MAX_ROD_NODES} nodes.")
        self.n_segments = int(sum(counts))
        seg_start = np.concatenate(([0], np.cumsum(counts))).astype(gs.np_int)

        def host(value):
            # the reference models' rest data may live on the device
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

        def cat(values):
            return np.concatenate([host(v).astype(np.float64).reshape(len(v), -1) for v in values])

        node0 = np.concatenate([entity.v_start + np.arange(n) for entity, n in zip(entities, counts)])
        first = np.concatenate([np.arange(n) == 0 for n in counts])
        last = np.concatenate([np.arange(n) == n - 1 for n in counts])
        rods = np.concatenate([np.full(n, r) for r, n in enumerate(counts)])
        seg_type = qd.types.struct(
            node0=gs.qd_int, rod=gs.qd_int, first=gs.qd_int, last=gs.qd_int, length=gs.qd_float,
            w_sec=gs.qd_float, w_str=gs.qd_float, w_vol=gs.qd_float, w_rad=gs.qd_float, w_rgr=gs.qd_float,
            # the joint between this segment and the one before it, meaningful when `first` is 0
            dual=gs.qd_float, w_bend=gs.qd_vec3, w_vbend=gs.qd_float, w_surf=gs.qd_float, kappa0=gs.qd_vec3,
        )
        self.seg = seg_type.field(shape=max(self.n_segments, 1), layout=qd.Layout.SOA)
        self.seg.node0.from_numpy(node0.astype(gs.np_int))
        self.seg.rod.from_numpy(rods.astype(gs.np_int))
        self.seg.first.from_numpy(first.astype(gs.np_int))
        self.seg.last.from_numpy(last.astype(gs.np_int))
        self.seg.length.from_numpy(cat([m.length for m in models])[:, 0].astype(gs.np_float))
        self.seg.w_sec.from_numpy(cat([m.section_weight[:, 0, 0] for m in models])[:, 0].astype(gs.np_float))
        self.seg.w_str.from_numpy(cat([m.stretch_weight[:, 0] for m in models])[:, 0].astype(gs.np_float))
        self.seg.w_vol.from_numpy(cat([m.volume_weight[:, 0] for m in models])[:, 0].astype(gs.np_float))
        self.seg.w_rad.from_numpy(cat([m.radius_weight for m in models])[:, 0].astype(gs.np_float))
        self.seg.w_rgr.from_numpy(cat([m.radius_gradient_weight for m in models])[:, 0].astype(gs.np_float))

        def joints(values, width):
            # a rod's joint k (between segments k-1 and k) is stored at its segment k; segment 0 carries zeros
            out = []
            for model, v in zip(models, values):
                v = host(v).astype(np.float64).reshape(len(model.length) - 1, width)
                out.append(np.concatenate((np.zeros((1, width)), v)))
            return np.concatenate(out)

        self.seg.dual.from_numpy(joints([m.dual_length for m in models], 1)[:, 0].astype(gs.np_float))
        self.seg.w_bend.from_numpy(joints([m.bending_weight for m in models], 3).astype(gs.np_float))
        self.seg.w_vbend.from_numpy(joints([m.volume_bending_weight[:, 0] for m in models], 1)[:, 0].astype(gs.np_float))
        self.seg.w_surf.from_numpy(joints([m.surface_weight for m in models], 1)[:, 0].astype(gs.np_float))
        self.seg.kappa0.from_numpy(joints([m.rest_curvature for m in models], 3).astype(gs.np_float))

        rod_type = qd.types.struct(seg_start=gs.qd_int, n_segments=gs.qd_int, node_start=gs.qd_int)
        self.rod = rod_type.field(shape=max(self.n_rods, 1))
        self.rod.seg_start.from_numpy(seg_start[:-1])
        self.rod.n_segments.from_numpy(np.array(counts, dtype=gs.np_int))
        self.rod.node_start.from_numpy(np.array([e.v_start for e in entities], dtype=gs.np_int))
        self._layout = [(int(e.v_start), int(n), int(s0)) for e, n, s0 in zip(entities, counts, seg_start[:-1])]
        # each vertex's two segments, -1 where there is none or the vertex is not a rod node
        seg_prev = np.full(solver.n_vertices, -1, dtype=gs.np_int)
        seg_next = np.full(solver.n_vertices, -1, dtype=gs.np_int)
        for entity, n, s0 in zip(entities, counts, seg_start[:-1]):
            seg_next[entity.v_start : entity.v_start + n] = s0 + np.arange(n)
            seg_prev[entity.v_start + 1 : entity.v_start + n + 1] = s0 + np.arange(n)
        self.seg_prev = qd.field(dtype=gs.qd_int, shape=solver.n_vertices)
        self.seg_next = qd.field(dtype=gs.qd_int, shape=solver.n_vertices)
        self.seg_prev.from_numpy(seg_prev)
        self.seg_next.from_numpy(seg_next)
        # two frame colours: a frame shares rows only with the frames next to it
        parity = np.concatenate([np.arange(n) % 2 for n in counts])
        order = np.argsort(parity, kind="stable")
        self.frame_offsets = [0, int((parity == 0).sum()), self.n_segments]
        self.frame_perm = qd.field(dtype=gs.qd_int, shape=max(self.n_segments, 1))
        self.frame_perm.from_numpy(order.astype(gs.np_int))

        self.quat = qd.Vector.field(4, dtype=gs.qd_float, shape=(max(self.n_segments, 1), B))
        self.scale = qd.field(dtype=gs.qd_float, shape=(solver.n_vertices, B))
        director_type = qd.types.struct(v0=gs.qd_vec3, v1=gs.qd_vec3)
        self.velocity = director_type.field(shape=(max(self.n_segments, 1), B), layout=qd.Layout.SOA)
        self.previous = director_type.field(shape=(max(self.n_segments, 1), B), layout=qd.Layout.SOA)
        self.predicted = director_type.field(shape=(max(self.n_segments, 1), B), layout=qd.Layout.SOA)
        self.band = qd.Vector.field(3, dtype=gs.qd_float, shape=(max(self.n_rods, 1), B, MAX_ROD_NODES))
        self.rhs = qd.field(dtype=gs.qd_float, shape=(max(self.n_rods, 1), B, MAX_ROD_NODES))
        self.grad = qd.field(dtype=gs.qd_float, shape=(max(self.n_rods, 1), B, MAX_ROD_NODES))
        # the scale line search's energy at each trial: per segment term by term, then per rod (energy, magnitude)
        self.trial_terms = qd.Vector.field(
            N_ENERGY_TERMS, dtype=gs.qd_float, shape=(max(self.n_segments, 1), N_TRIALS, B)
        )
        self.trial_magnitude_terms = qd.Vector.field(
            N_MAGNITUDE_TERMS, dtype=gs.qd_float, shape=(max(self.n_segments, 1), N_TRIALS, B)
        )
        self.trial_energy = qd.Vector.field(2, dtype=gs.qd_float, shape=(max(self.n_rods, 1), N_TRIALS, B))
        self.errno = qd.field(dtype=gs.qd_int, shape=B)
        # the block a failure names: segment j for a frame, -1 - r for rod r's scales
        self.failed_block = qd.field(dtype=gs.qd_int, shape=B)
        quat = np.concatenate([m.rest_quat.cpu().numpy() for m in models]).astype(gs.np_float)
        self.quat.from_numpy(np.repeat(quat[:, None], B, axis=1))
        scale = np.zeros((solver.n_vertices, B), dtype=gs.np_float)
        for entity in entities:
            scale[entity.v_start : entity.v_start + entity.n_vertices] = 1.0
        self.scale.from_numpy(scale)
        self.velocity.v0.fill(0.0)
        self.velocity.v1.fill(0.0)
        self.errno.fill(0)

    def get_states(self):
        """One `vbd_rod.RodState` per rod, each field with a leading environment axis: scale (B, n), frames
        (B, n - 1, 4) and director velocity (B, n - 1, 3, 2), the reference's layout per environment."""
        from genesis.engine.solvers.vbd_rod import RodState

        scale, quat = self.scale.to_numpy(), self.quat.to_numpy()
        v0, v1 = self.velocity.v0.to_numpy(), self.velocity.v1.to_numpy()
        states = []
        for node_start, n_segments, s0 in self._layout:
            states.append(RodState(
                torch.tensor(scale[node_start : node_start + n_segments + 1].T, device=gs.device),
                torch.tensor(quat[s0 : s0 + n_segments].transpose(1, 0, 2), device=gs.device),
                torch.tensor(np.stack((v0, v1), axis=-1)[s0 : s0 + n_segments].transpose(1, 0, 2, 3), device=gs.device),
            ))
        return tuple(states)

    def set_states(self, states, envs_idx):
        """Write snapshots taken by `get_states` back, for the environments in `envs_idx`."""
        if len(states) != self.n_rods:
            gs.raise_exception("Rod snapshot entity count differs from the scene.")
        envs = np.asarray(envs_idx.cpu() if hasattr(envs_idx, "cpu") else envs_idx, dtype=np.int64)
        scale, quat = self.scale.to_numpy(), self.quat.to_numpy()
        v0, v1 = self.velocity.v0.to_numpy(), self.velocity.v1.to_numpy()
        for (node_start, n_segments, s0), state in zip(self._layout, states):
            s_new, q_new, v_new = (value.detach().cpu().numpy() for value in state)
            if s_new.shape[1:] != (n_segments + 1,) or q_new.shape[1:] != (n_segments, 4):
                gs.raise_exception("Rod snapshot shape differs from its entity.")
            if not (np.isfinite(s_new).all() and np.isfinite(q_new).all() and np.isfinite(v_new).all()):
                gs.raise_exception("Rod snapshot must be finite.")
            if (s_new <= 0).any() or not np.allclose((q_new**2).sum(-1), 1.0):
                gs.raise_exception("Rod snapshot requires positive scales and unit frames.")
            scale[node_start : node_start + n_segments + 1][:, envs] = s_new[envs].T
            quat[s0 : s0 + n_segments][:, envs] = q_new[envs].transpose(1, 0, 2)
            v0[s0 : s0 + n_segments][:, envs] = v_new[envs][..., 0].transpose(1, 0, 2)
            v1[s0 : s0 + n_segments][:, envs] = v_new[envs][..., 1].transpose(1, 0, 2)
        self.scale.from_numpy(scale)
        self.quat.from_numpy(quat)
        self.velocity.v0.from_numpy(v0)
        self.velocity.v1.from_numpy(v1)
        errno = self.errno.to_numpy()
        errno[envs] = 0
        self.errno.from_numpy(errno)

    def describe_failure(self, i_b):
        """Which rod block the latched failure of environment i_b names."""
        block = int(self.failed_block.to_numpy()[i_b])
        if block < 0:
            return f"the scales of rod {-1 - block}"
        rod = next(r for r, (_, n_segments, s0) in enumerate(self._layout) if s0 <= block < s0 + n_segments)
        return f"frame {block - self._layout[rod][2]} of rod {rod}"


@qd.func
def func_frame(q):
    return gu.qd_quat_to_R(q, gs.EPS)


@qd.func
def func_retract(q, delta):
    """World-frame Cayley step, as `vbd_rod.retract`."""
    dq = qd.Vector([1.0, 0.5 * delta[0], 0.5 * delta[1], 0.5 * delta[2]], dt=gs.qd_float)
    return gu.qd_quat_mul(dq / dq.norm(), q)


@qd.func
def func_relative(qa, qb):
    """Shortest relative frame conj(qa) qb and the sign that made it so."""
    p = gu.qd_quat_mul(gu.qd_inv_quat(qa), qb)
    sigma = gs.qd_float(1.0)
    if p[0] < 0.0:
        sigma = -1.0
    return p, sigma


@qd.func
def func_skew(v):
    return qd.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dt=gs.qd_float)


@qd.func
def func_mid_scale(j, i_b, rod: qd.template()):
    n0 = rod.seg[j].node0
    return 0.5 * (rod.scale[n0, i_b] + rod.scale[n0 + 1, i_b])


@qd.func
def func_tangent(f, j, i_b, solver: qd.template(), rod: qd.template()):
    n0 = rod.seg[j].node0
    return (solver.verts[f + 1, n0 + 1, i_b].pos - solver.verts[f + 1, n0, i_b].pos) / rod.seg[j].length


@qd.func
def func_rod_node_terms(f, i_v, i_b, solver: qd.template(), rod: qd.template()):
    """Force and 3x3 curvature of the stretch and volume rows of a rod node's two segments: linear in the node,
    so the curvature is a multiple of the identity and the vertex solve's Newton step is exact for them."""
    force = gs.qd_vec3(0.0, 0.0, 0.0)
    c = gs.qd_float(0.0)
    for side in qd.static(range(2)):
        j = rod.seg_prev[i_v]
        sign = gs.qd_float(1.0)
        if qd.static(side == 1):
            j = rod.seg_next[i_v]
            sign = -1.0
        if j >= 0:
            L = rod.seg[j].length
            t = func_tangent(f, j, i_b, solver, rod)
            d3 = func_frame(rod.quat[j, i_b])[:, 2]
            mid = func_mid_scale(j, i_b, rod)
            ws = rod.seg[j].w_str
            wv = rod.seg[j].w_vol
            force -= ws * ws * sign / L * (t - d3) + wv * wv * mid * mid * sign / L * (mid * mid * t - d3)
            c += (ws * ws + wv * wv * mid**4) / (L * L)
    return force, c * qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def func_joint_energy(k, s_k, qa, qb, rod: qd.template()):
    """Bending and volume-bending energy of the joint stored at segment k, with the given frames and scale, and
    the energy its terms would have without the subtraction, which bounds the rounding of the first."""
    p, sigma = func_relative(qa, qb)
    kappa = 2.0 * sigma * gs.qd_vec3(p[1], p[2], p[3]) / rod.seg[k].dual
    wb = rod.seg[k].w_bend
    wv = rod.seg[k].w_vbend
    bend = wb * (s_k * kappa - rod.seg[k].kappa0)
    vb = wv * (s_k**3 * kappa - rod.seg[k].kappa0)
    energy = 0.5 * (bend.norm_sqr() + vb[0] * vb[0] + vb[1] * vb[1])
    a = wb * s_k * kappa
    b = wb * rod.seg[k].kappa0
    c = wv * s_k**3 * kappa
    d = wv * rod.seg[k].kappa0
    scale = 0.5 * (a.norm_sqr() + b.norm_sqr() + c[0] * c[0] + c[1] * c[1] + d[0] * d[0] + d[1] * d[1])
    return energy, scale


@qd.func
def func_frame_energy(f, j, i_b, q, solver: qd.template(), rod: qd.template()):
    """Energy of every row that frame j touches, with q in its place, and the energy of its terms before each
    row's subtraction: near convergence a row is a small difference of large terms, so its rounding scales with
    the second, not the first, and that is what a decrease must beat to be resolved."""
    R = func_frame(q)
    mid = func_mid_scale(j, i_b, rod)
    t = func_tangent(f, j, i_b, solver, rod)
    w = rod.seg[j].w_sec / solver._substep_dt
    ws = rod.seg[j].w_str
    wv = rod.seg[j].w_vol
    energy = 0.5 * (w * (mid * R[:, 0] - rod.predicted[j, i_b].v0)).norm_sqr()
    energy += 0.5 * (w * (mid * R[:, 1] - rod.predicted[j, i_b].v1)).norm_sqr()
    energy += 0.5 * (ws * (t - R[:, 2])).norm_sqr()
    energy += 0.5 * (wv * (mid * mid * t - R[:, 2])).norm_sqr()
    scale = 0.5 * w * w * (2.0 * mid * mid + rod.predicted[j, i_b].v0.norm_sqr() + rod.predicted[j, i_b].v1.norm_sqr())
    scale += 0.5 * ws * ws * (t.norm_sqr() + 1.0) + 0.5 * wv * wv * (mid**4 * t.norm_sqr() + 1.0)
    n0 = rod.seg[j].node0
    if not rod.seg[j].first:
        e_b, s_b = func_joint_energy(j, rod.scale[n0, i_b], rod.quat[j - 1, i_b], q, rod)
        energy += e_b
        scale += s_b
    if not rod.seg[j].last:
        e_a, s_a = func_joint_energy(j + 1, rod.scale[n0 + 1, i_b], q, rod.quat[j + 1, i_b], rod)
        energy += e_a
        scale += s_a
    return energy, scale


@qd.func
def func_joint_frame_terms(k, s_k, qa, qb, side_b: qd.template(), rod: qd.template()):
    """Gradient and Gauss-Newton curvature, in the frame on `side_b` (True: segment k) or on the other side
    (segment k - 1), of the bending and volume-bending rows of the joint stored at segment k."""
    p, sigma = func_relative(qa, qb)
    p_vec = gs.qd_vec3(p[1], p[2], p[3])
    kappa = 2.0 * sigma * p_vec / rod.seg[k].dual
    core = p[0] * qd.Matrix.identity(gs.qd_float, 3) - func_skew(p_vec)
    dk = sigma / rod.seg[k].dual * core @ func_frame(qa).transpose()
    if qd.static(not side_b):
        dk = -dk
    g = gs.qd_vec3(0.0, 0.0, 0.0)
    H = qd.Matrix.zero(gs.qd_float, 3, 3)
    wb = rod.seg[k].w_bend
    r_b = wb * (s_k * kappa - rod.seg[k].kappa0)
    J_b = qd.Matrix.zero(gs.qd_float, 3, 3)
    for row in qd.static(range(3)):
        for col in qd.static(range(3)):
            J_b[row, col] = wb[row] * s_k * dk[row, col]
    g += J_b.transpose() @ r_b
    H += J_b.transpose() @ J_b
    wv = rod.seg[k].w_vbend
    r_v = wv * (s_k**3 * kappa - rod.seg[k].kappa0)
    for row in qd.static(range(2)):
        J_row = wv * s_k**3 * gs.qd_vec3(dk[row, 0], dk[row, 1], dk[row, 2])
        g += J_row * r_v[row]
        H += J_row.outer_product(J_row)
    return g, H


@qd.func
def func_frame_system(f, j, i_b, solver: qd.template(), rod: qd.template()):
    """Gradient and Gauss-Newton curvature of frame j in the world-frame rotation increment."""
    q = rod.quat[j, i_b]
    R = func_frame(q)
    mid = func_mid_scale(j, i_b, rod)
    t = func_tangent(f, j, i_b, solver, rod)
    w = rod.seg[j].w_sec / solver._substep_dt
    g = gs.qd_vec3(0.0, 0.0, 0.0)
    H = qd.Matrix.zero(gs.qd_float, 3, 3)
    J0 = -w * mid * func_skew(R[:, 0])
    g += J0.transpose() @ (w * (mid * R[:, 0] - rod.predicted[j, i_b].v0))
    H += J0.transpose() @ J0
    J1 = -w * mid * func_skew(R[:, 1])
    g += J1.transpose() @ (w * (mid * R[:, 1] - rod.predicted[j, i_b].v1))
    H += J1.transpose() @ J1
    d3 = R[:, 2]
    J_s = rod.seg[j].w_str * func_skew(d3)
    g += J_s.transpose() @ (rod.seg[j].w_str * (t - d3))
    H += J_s.transpose() @ J_s
    J_v = rod.seg[j].w_vol * func_skew(d3)
    g += J_v.transpose() @ (rod.seg[j].w_vol * (mid * mid * t - d3))
    H += J_v.transpose() @ J_v
    n0 = rod.seg[j].node0
    if not rod.seg[j].first:
        g_b, H_b = func_joint_frame_terms(j, rod.scale[n0, i_b], rod.quat[j - 1, i_b], q, True, rod)
        g += g_b
        H += H_b
    if not rod.seg[j].last:
        g_a, H_a = func_joint_frame_terms(j + 1, rod.scale[n0 + 1, i_b], q, rod.quat[j + 1, i_b], False, rod)
        g += g_a
        H += H_a
    return g, H


@qd.func
def func_solve_rod_frame(f, j, i_b, solver: qd.template(), rod: qd.template()):
    """One Gauss-Newton step of frame j with the reference's stopping test and Armijo backtracking."""
    g, H = func_frame_system(f, j, i_b, solver, rod)
    delta = -(H.inverse() @ g)
    q = rod.quat[j, i_b]
    # a Cayley step changes a unit quaternion by half the angular step (vbd_rod.RodModel.sweep)
    if 0.5 * delta.norm() > gs.EPS * q.norm():
        energy, magnitude = func_frame_energy(f, j, i_b, q, solver, rod)
        old = 2.0 * energy
        if not (old == old and delta.norm() == delta.norm()):
            # non-finite already: no search can mend it, so it is not reported as one
            qd.atomic_or(rod.errno[i_b], ErrorCode.VBD_ROD_INVALID)
            rod.failed_block[i_b] = j
        floor = 64.0 * gs.EPS * magnitude
        slope = g.dot(delta)
        # Below the energy's rounding the search cannot tell a descent from noise, but the quadratic model is
        # exact there: take the full Gauss-Newton step. Stopping instead would cap the accuracy at sqrt(eps).
        is_accepted = False
        if -slope <= floor:
            rod.quat[j, i_b] = func_retract(q, delta)
            is_accepted = True
        fraction = gs.qd_float(1.0)
        for _ in range(LINE_SEARCH_STEPS):
            if not is_accepted:
                candidate = func_retract(q, fraction * delta)
                e_new, unused_magnitude = func_frame_energy(f, j, i_b, candidate, solver, rod)
                if 2.0 * e_new <= old + 1e-4 * fraction * slope + floor:
                    rod.quat[j, i_b] = candidate
                    is_accepted = True
                fraction *= 0.5
        if not is_accepted:
            qd.atomic_or(rod.errno[i_b], ErrorCode.VBD_ROD_LINE_SEARCH)
            rod.failed_block[i_b] = j


@qd.func
def func_scale_energy(f, r, i_b, fraction, solver: qd.template(), rod: qd.template()):
    """Energy of every scale-dependent row of rod r at the scales plus `fraction` of the stored step."""
    energy = gs.qd_float(0.0)
    scale = gs.qd_float(0.0)
    for c in range(rod.rod[r].n_segments):
        e, m = func_scale_segment_terms(f, rod.rod[r].seg_start + c, i_b, fraction, solver, rod)
        for t in qd.static(range(N_ENERGY_TERMS)):
            energy += e[t]
        for t in qd.static(range(N_MAGNITUDE_TERMS)):
            scale += m[t]
    return energy, scale


@qd.func
def func_scale_segment_terms(f, j, i_b, fraction, solver: qd.template(), rod: qd.template()):
    """Segment j's terms of `func_scale_energy`, one by one in the order it adds them: the section, radius,
    radius-gradient and volume rows, then, past a rod's first segment, the joint's bending and surface rows (zero
    at the first). Adding an exact zero changes no bit of the sum."""
    r = rod.seg[j].rod
    start = rod.rod[r].node_start
    c = j - rod.rod[r].seg_start
    e = qd.Vector.zero(gs.qd_float, N_ENERGY_TERMS)
    m = qd.Vector.zero(gs.qd_float, N_MAGNITUDE_TERMS)
    h = solver._substep_dt
    s0 = rod.scale[start + c, i_b] + fraction * rod.rhs[r, i_b, c]
    s1 = rod.scale[start + c + 1, i_b] + fraction * rod.rhs[r, i_b, c + 1]
    mid = 0.5 * (s0 + s1)
    R = func_frame(rod.quat[j, i_b])
    t = func_tangent(f, j, i_b, solver, rod)
    w = rod.seg[j].w_sec / h
    e[0] = 0.5 * (w * (mid * R[:, 0] - rod.predicted[j, i_b].v0)).norm_sqr()
    e[1] = 0.5 * (w * (mid * R[:, 1] - rod.predicted[j, i_b].v1)).norm_sqr()
    e[2] = 0.5 * (rod.seg[j].w_rad * (mid - 1.0)) ** 2
    w_g = rod.seg[j].w_rgr / rod.seg[j].length
    e[3] = 0.5 * (w_g * (s1 - s0)) ** 2
    wv = rod.seg[j].w_vol
    e[4] = 0.5 * (wv * (mid * mid * t - R[:, 2])).norm_sqr()
    m[0] = 0.5 * w * w * (2.0 * mid * mid + rod.predicted[j, i_b].v0.norm_sqr() + rod.predicted[j, i_b].v1.norm_sqr())
    m[1] = 0.5 * rod.seg[j].w_rad ** 2 * (mid * mid + 1.0) + 0.5 * w_g * w_g * (s0 * s0 + s1 * s1)
    m[2] = 0.5 * wv * wv * (mid**4 * t.norm_sqr() + 1.0)
    if not rod.seg[j].first:
        e_j, s_j = func_joint_energy(j, s0, rod.quat[j - 1, i_b], rod.quat[j, i_b], rod)
        e[5] = e_j
        m[3] = s_j
        sp = rod.scale[start + c - 1, i_b] + fraction * rod.rhs[r, i_b, c - 1]
        La = rod.seg[j - 1].length
        Lb = rod.seg[j].length
        w_s = rod.seg[j].w_surf / rod.seg[j].dual
        e[6] = 0.5 * (w_s * ((s1 - s0) / Lb - (s0 - sp) / La)) ** 2
        m[4] = 0.5 * w_s * w_s * ((s1 * s1 + s0 * s0) / (Lb * Lb) + (s0 * s0 + sp * sp) / (La * La))
    return e, m


@qd.func
def func_sum_scale_trial(r, k, i_b, rod: qd.template()):
    """Rod r's energy and magnitude at trial k from its segments' stored terms, summed in `func_scale_energy`'s
    order."""
    energy = gs.qd_float(0.0)
    scale = gs.qd_float(0.0)
    for c in range(rod.rod[r].n_segments):
        e = rod.trial_terms[rod.rod[r].seg_start + c, k, i_b]
        m = rod.trial_magnitude_terms[rod.rod[r].seg_start + c, k, i_b]
        for t in qd.static(range(N_ENERGY_TERMS)):
            energy += e[t]
        for t in qd.static(range(N_MAGNITUDE_TERMS)):
            scale += m[t]
    rod.trial_energy[r, k, i_b] = gs.qd_vec2(energy, scale)


@qd.func
def func_trial_fraction(k):
    """The step fraction of line-search trial k: 0 for the start, then 1, 1/2, 1/4, ... as the search halves it."""
    fraction = gs.qd_float(0.0)
    if k > 0:
        fraction = 1.0
        for _ in range(k - 1):
            fraction *= 0.5
    return fraction


@qd.func
def func_band_add(r, i_b, i, k, value, rod: qd.template()):
    """H[i, i - k] += value for k in {0, 1, 2} (lower band, symmetric)."""
    rod.band[r, i_b, i][k] += value


@qd.func
def func_solve_rod_scales(f, r, i_b, solver: qd.template(), rod: qd.template()):
    """The coupled scale block of rod r: assemble its banded Gauss-Newton system, solve it exactly, and take the
    reference's positive, descending step. One thread does all of it; the sweep splits the same work over the
    rods and their line-search trials (`VBDSolver._kernel_sweeps`)."""
    func_rod_scales_newton(f, r, i_b, solver, rod)
    for k in range(N_TRIALS):
        energy, magnitude = func_scale_energy(f, r, i_b, func_trial_fraction(k), solver, rod)
        rod.trial_energy[r, k, i_b] = gs.qd_vec2(energy, magnitude)
    func_rod_scales_accept(r, i_b, rod)


@qd.func
def func_rod_scales_newton(f, r, i_b, solver: qd.template(), rod: qd.template()):
    """Assemble rod r's banded scale system and solve it: the step goes to `rhs`, the gradient to `grad`."""
    start = rod.rod[r].node_start
    n = rod.rod[r].n_segments + 1
    h = solver._substep_dt
    for i in range(n):
        rod.band[r, i_b, i] = gs.qd_vec3(0.0, 0.0, 0.0)
        rod.rhs[r, i_b, i] = 0.0
    for c in range(n - 1):
        j = rod.rod[r].seg_start + c
        s0 = rod.scale[start + c, i_b]
        s1 = rod.scale[start + c + 1, i_b]
        mid = 0.5 * (s0 + s1)
        R = func_frame(rod.quat[j, i_b])
        t = func_tangent(f, j, i_b, solver, rod)
        # rows that read both ends of the segment with one partial d each: g += d.r on both, H += d.d on the
        # 2x2 block; the gradient row reads them with opposite signs
        w = rod.seg[j].w_sec / h
        d_sec0 = 0.5 * w * R[:, 0]
        d_sec1 = 0.5 * w * R[:, 1]
        r_sec0 = w * (mid * R[:, 0] - rod.predicted[j, i_b].v0)
        r_sec1 = w * (mid * R[:, 1] - rod.predicted[j, i_b].v1)
        d_vol = rod.seg[j].w_vol * mid * t
        r_vol = rod.seg[j].w_vol * (mid * mid * t - R[:, 2])
        gd = d_sec0.dot(r_sec0) + d_sec1.dot(r_sec1) + d_vol.dot(r_vol)
        hd = d_sec0.norm_sqr() + d_sec1.norm_sqr() + d_vol.norm_sqr()
        w_rad = rod.seg[j].w_rad
        gd += 0.5 * w_rad * w_rad * (mid - 1.0)
        hd += 0.25 * w_rad * w_rad
        w_g = rod.seg[j].w_rgr / rod.seg[j].length
        grad_row = w_g * w_g * (s1 - s0)
        rod.rhs[r, i_b, c] += gd - grad_row
        rod.rhs[r, i_b, c + 1] += gd + grad_row
        func_band_add(r, i_b, c, 0, hd + w_g * w_g, rod)
        func_band_add(r, i_b, c + 1, 0, hd + w_g * w_g, rod)
        func_band_add(r, i_b, c + 1, 1, hd - w_g * w_g, rod)
        if c > 0:
            # joint at node c, stored at segment j
            p, sigma = func_relative(rod.quat[j - 1, i_b], rod.quat[j, i_b])
            kappa = 2.0 * sigma * gs.qd_vec3(p[1], p[2], p[3]) / rod.seg[j].dual
            wb = rod.seg[j].w_bend
            d_b = wb * kappa
            r_b = wb * (s0 * kappa - rod.seg[j].kappa0)
            wv = rod.seg[j].w_vbend
            r_v = wv * (s0**3 * kappa - rod.seg[j].kappa0)
            d_v0 = 3.0 * wv * s0 * s0 * kappa[0]
            d_v1 = 3.0 * wv * s0 * s0 * kappa[1]
            rod.rhs[r, i_b, c] += d_b.dot(r_b) + d_v0 * r_v[0] + d_v1 * r_v[1]
            func_band_add(r, i_b, c, 0, d_b.norm_sqr() + d_v0 * d_v0 + d_v1 * d_v1, rod)
            La = rod.seg[j - 1].length
            Lb = rod.seg[j].length
            ws = rod.seg[j].w_surf / rod.seg[j].dual
            sp = rod.scale[start + c - 1, i_b]
            r_s = ws * ((s1 - s0) / Lb - (s0 - sp) / La)
            d_prev = ws / La
            d_here = -ws * (1.0 / La + 1.0 / Lb)
            d_next = ws / Lb
            rod.rhs[r, i_b, c - 1] += d_prev * r_s
            rod.rhs[r, i_b, c] += d_here * r_s
            rod.rhs[r, i_b, c + 1] += d_next * r_s
            func_band_add(r, i_b, c - 1, 0, d_prev * d_prev, rod)
            func_band_add(r, i_b, c, 0, d_here * d_here, rod)
            func_band_add(r, i_b, c + 1, 0, d_next * d_next, rod)
            func_band_add(r, i_b, c, 1, d_here * d_prev, rod)
            func_band_add(r, i_b, c + 1, 1, d_next * d_here, rod)
            func_band_add(r, i_b, c + 1, 2, d_next * d_prev, rod)
    # banded LDL^T in place, bandwidth 2: band = (d_i, l1_i, l2_i) with L[i, i-1] = l1_i, L[i, i-2] = l2_i
    for i in range(n):
        rod.grad[r, i_b, i] = rod.rhs[r, i_b, i]
        a = rod.band[r, i_b, i][0]
        b = rod.band[r, i_b, i][1]
        c2 = rod.band[r, i_b, i][2]
        l2 = gs.qd_float(0.0)
        l1 = gs.qd_float(0.0)
        if i >= 2:
            l2 = c2 / rod.band[r, i_b, i - 2][0]
        if i >= 1:
            l1 = b
            if i >= 2:
                l1 -= l2 * rod.band[r, i_b, i - 1][1] * rod.band[r, i_b, i - 2][0]
            l1 /= rod.band[r, i_b, i - 1][0]
        d = a
        if i >= 1:
            d -= l1 * l1 * rod.band[r, i_b, i - 1][0]
        if i >= 2:
            d -= l2 * l2 * rod.band[r, i_b, i - 2][0]
        rod.band[r, i_b, i] = gs.qd_vec3(d, l1, l2)
    # H step = -g: forward L y = -g, diagonal, backward L^T x = z; the step replaces the gradient in rhs
    for i in range(n):
        y = -rod.rhs[r, i_b, i]
        if i >= 1:
            y -= rod.band[r, i_b, i][1] * rod.rhs[r, i_b, i - 1]
        if i >= 2:
            y -= rod.band[r, i_b, i][2] * rod.rhs[r, i_b, i - 2]
        rod.rhs[r, i_b, i] = y
    for i in range(n):
        rod.rhs[r, i_b, i] /= rod.band[r, i_b, i][0]
    for i_ in range(n):
        i = n - 1 - i_
        x = rod.rhs[r, i_b, i]
        if i + 1 < n:
            x -= rod.band[r, i_b, i + 1][1] * rod.rhs[r, i_b, i + 1]
        if i + 2 < n:
            x -= rod.band[r, i_b, i + 2][2] * rod.rhs[r, i_b, i + 2]
        rod.rhs[r, i_b, i] = x


@qd.func
def func_rod_scales_accept(r, i_b, rod: qd.template()):
    """Take the reference's positive, descending step along `rhs`, reading trial k's energy and magnitude from
    `trial_energy[r, k]` (k = 0 at the start, k >= 1 at `func_trial_fraction(k)`): the first trial the halving
    search accepts is the same whichever order the energies were computed in."""
    start = rod.rod[r].node_start
    n = rod.rod[r].n_segments + 1
    step_norm = gs.qd_float(0.0)
    scale_norm = gs.qd_float(0.0)
    for i in range(n):
        step_norm += rod.rhs[r, i_b, i] ** 2
        scale_norm += rod.scale[start + i, i_b] ** 2
    if qd.sqrt(step_norm) > gs.EPS * qd.sqrt(scale_norm):
        energy = rod.trial_energy[r, 0, i_b][0]
        magnitude = rod.trial_energy[r, 0, i_b][1]
        old = 2.0 * energy
        if not (old == old and step_norm == step_norm):
            qd.atomic_or(rod.errno[i_b], ErrorCode.VBD_ROD_INVALID)
            rod.failed_block[i_b] = -1 - r
        floor = 64.0 * gs.EPS * magnitude
        slope = gs.qd_float(0.0)
        for i in range(n):
            slope += rod.grad[r, i_b, i] * rod.rhs[r, i_b, i]
        # the full step below the energy's rounding, as for a frame, when it keeps every scale positive
        is_accepted = False
        if -slope <= floor:
            positive = True
            for i in range(n):
                if rod.scale[start + i, i_b] + rod.rhs[r, i_b, i] <= 0.0:
                    positive = False
            if positive:
                for i in range(n):
                    rod.scale[start + i, i_b] += rod.rhs[r, i_b, i]
                is_accepted = True
        fraction = gs.qd_float(1.0)
        for k in range(LINE_SEARCH_STEPS):
            if not is_accepted:
                positive = True
                for i in range(n):
                    if rod.scale[start + i, i_b] + fraction * rod.rhs[r, i_b, i] <= 0.0:
                        positive = False
                if positive:
                    e_new = rod.trial_energy[r, k + 1, i_b][0]
                    if 2.0 * e_new <= old + 1e-4 * fraction * slope + floor:
                        for i in range(n):
                            rod.scale[start + i, i_b] += fraction * rod.rhs[r, i_b, i]
                        is_accepted = True
                fraction *= 0.5
        if not is_accepted:
            qd.atomic_or(rod.errno[i_b], ErrorCode.VBD_ROD_LINE_SEARCH)
            rod.failed_block[i_b] = -1 - r


@qd.kernel
def kernel_rod_begin(solver: qd.template(), rod: qd.template()):
    """Directors of the substep's start and their inertial prediction."""
    for j, i_b in qd.ndrange(rod.n_segments, solver._B):
        if not solver.env_failed[i_b]:
            R = func_frame(rod.quat[j, i_b])
            mid = func_mid_scale(j, i_b, rod)
            rod.previous[j, i_b].v0 = mid * R[:, 0]
            rod.previous[j, i_b].v1 = mid * R[:, 1]
            rod.predicted[j, i_b].v0 = rod.previous[j, i_b].v0 + solver._substep_dt * rod.velocity[j, i_b].v0
            rod.predicted[j, i_b].v1 = rod.previous[j, i_b].v1 + solver._substep_dt * rod.velocity[j, i_b].v1


@qd.kernel
def kernel_rod_end(f: qd.i32, substep_global: qd.i32, solver: qd.template(), rod: qd.template()):
    """Director velocities, and the reference's validity checks, latched as an environment failure."""
    for j, i_b in qd.ndrange(rod.n_segments, solver._B):
        if not solver.env_failed[i_b]:
            q = rod.quat[j, i_b]
            R = func_frame(q)
            mid = func_mid_scale(j, i_b, rod)
            rod.velocity[j, i_b].v0 = (mid * R[:, 0] - rod.previous[j, i_b].v0) / solver._substep_dt
            rod.velocity[j, i_b].v1 = (mid * R[:, 1] - rod.previous[j, i_b].v1) / solver._substep_dt
            n0 = rod.seg[j].node0
            axial = (solver.verts[f + 1, n0 + 1, i_b].pos - solver.verts[f + 1, n0, i_b].pos).dot(R[:, 2])
            s0 = rod.scale[n0, i_b]
            s1 = rod.scale[n0 + 1, i_b]
            if not (axial > 0.0 and s0 > 0.0 and s1 > 0.0 and q.norm() == q.norm()):
                qd.atomic_or(rod.errno[i_b], ErrorCode.VBD_ROD_INVALID)
                rod.failed_block[i_b] = j
    for i_b in range(solver._B):
        if rod.errno[i_b] != 0 and not solver.env_failed[i_b]:
            solver.env_failed[i_b] = 1
            solver.failed_substep[i_b] = substep_global
