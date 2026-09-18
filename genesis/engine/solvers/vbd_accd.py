"""Additive continuous collision detection for the mesh contact of vertex block descent (VBD).

The penalty contact of `vbd_contact.py` holds a pair that stays inside its layer for the substep. It cannot hold
one that passes clean through, and it reports that case as a failed substep. This is the conservative filter that
prevents it: after the substep's primal solve, every candidate pair is swept from the pose at the start of the
substep to the pose the solve reached, additive CCD (Li, Kaufman and Jiang, "Codimensional Incremental Potential
Contact", SIGGRAPH 2021, Algorithm 1) gives a time of impact no later than the true one, and the whole substep is
rescaled by the smallest such time over all pairs. Nothing then reaches the gap, so nothing crosses.

The bound rests on one inequality. The pair's distance does not depend on its common translation, so the mean
displacement is removed, and what remains satisfies |d(t2) - d(t1)| <= l_p |t2 - t1| with l_p the sum of the two
sides' largest vertex displacement. From a state at distance d the pair therefore cannot close the gap in less
than (d - gap) / l_p, and advancing by `scale` of that is safe; the loop repeats until the remaining gap is under
(1 - scale) of the first one, and the sum of the advances taken is the bound. The reference implementation, the
brute-force sweep it was checked against and the measured tightness (the bound is at least 0.88 of the true time
of impact) are in `spikes/verify_accd.py` and its test in the application repository.

Two costs are real and neither is a defect. A pair that comes within the tolerance of the gap without touching is
cut short as well, which slows a grazing substep for nothing (4 of 191 random point-triangle sweeps). And
rescaling a substep after the solve, rather than re-solving it, is an inelastic slowdown: the positions are those
of a shorter step while the clock advances by the whole one, so the pair loses the approach speed it would have
kept. That is what a contact that has to stop a body does anyway, and it is a stop instead of a refused substep.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.vbd_contact import (
    func_cv_pos,
    func_cv_pos_prev,
    func_point_triangle_weights,
    func_refresh_link_vertices,
    func_segment_parameters,
    func_slerp,
)
from genesis.engine.solvers.vbd_mtu import func_refresh_mtu_link_anchors

POINT_TRIANGLE = 0
EDGE_EDGE = 1


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
def func_accd_toi(s0, s1, s2, s3, e0, e1, e2, e3, gap, scale, iterations, kind: qd.template()):
    """Conservative time in [0, 1] at which the pair's distance first reaches `gap` over the linear sweep from
    (s0..s3) to (e0..e3); 1 when it does not reach it, 0 for a pair already at or inside the gap.

    A pair that has not converged within `iterations` returns the advances it has taken, which is a lower bound
    like any other, so the budget trades tightness for time and never safety.
    """
    p0 = e0 - s0
    p1 = e1 - s1
    p2 = e2 - s2
    p3 = e3 - s3
    mean = 0.25 * (p0 + p1 + p2 + p3)
    p0 -= mean
    p1 -= mean
    p2 -= mean
    p3 -= mean
    bound = gs.qd_float(0.0)
    if qd.static(kind == POINT_TRIANGLE):
        bound = p0.norm() + qd.max(p1.norm(), qd.max(p2.norm(), p3.norm()))
    else:
        bound = qd.max(p0.norm(), p1.norm()) + qd.max(p2.norm(), p3.norm())
    distance = func_pair_distance(s0, s1, s2, s3, kind)
    toi = gs.qd_float(1.0)
    if distance <= gap:
        toi = 0.0
    elif bound > 0.0:
        tolerance = (1.0 - scale) * (distance - gap)
        taken = gs.qd_float(0.0)
        x0 = s0
        x1 = s1
        x2 = s2
        x3 = s3
        is_done = False
        for _ in range(iterations):
            if not is_done:
                advance = scale * (distance - gap) / bound
                x0 += advance * p0
                x1 += advance * p1
                x2 += advance * p2
                x3 += advance * p3
                distance = func_pair_distance(x0, x1, x2, x3, kind)
                if taken > 0.0 and distance - gap < tolerance:
                    is_done = True
                else:
                    taken += advance
                    if taken > 1.0:
                        taken = 1.0
                        is_done = True
        toi = taken
    return toi


@qd.kernel
def kernel_accd_toi(f: int, solver: qd.template(), contact: qd.template()):
    """The substep's conservative time of impact per environment: the smallest over every candidate pair."""
    for i_b in range(solver._B):
        contact.toi[i_b] = 1.0
    for i_p, i_b in qd.ndrange(contact.pair_cap, solver._B):
        if i_p < qd.min(contact.n_pt[i_b], contact.pair_cap) and not solver.env_failed[i_b]:
            cv = contact.pt_pairs[i_p, i_b].a
            tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
            toi = func_accd_toi(
                func_cv_pos_prev(f, cv, i_b, solver, contact),
                func_cv_pos_prev(f, tri[0], i_b, solver, contact),
                func_cv_pos_prev(f, tri[1], i_b, solver, contact),
                func_cv_pos_prev(f, tri[2], i_b, solver, contact),
                func_cv_pos(f, cv, i_b, solver, contact),
                func_cv_pos(f, tri[0], i_b, solver, contact),
                func_cv_pos(f, tri[1], i_b, solver, contact),
                func_cv_pos(f, tri[2], i_b, solver, contact),
                solver._contact_ccd_gap,
                solver._contact_ccd_scale,
                solver._contact_ccd_iterations,
                POINT_TRIANGLE,
            )
            qd.atomic_min(contact.toi[i_b], toi)
        if i_p < qd.min(contact.n_ee[i_b], contact.pair_cap) and not solver.env_failed[i_b]:
            ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
            eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
            toi = func_accd_toi(
                func_cv_pos_prev(f, ea[0], i_b, solver, contact),
                func_cv_pos_prev(f, ea[1], i_b, solver, contact),
                func_cv_pos_prev(f, eb[0], i_b, solver, contact),
                func_cv_pos_prev(f, eb[1], i_b, solver, contact),
                func_cv_pos(f, ea[0], i_b, solver, contact),
                func_cv_pos(f, ea[1], i_b, solver, contact),
                func_cv_pos(f, eb[0], i_b, solver, contact),
                func_cv_pos(f, eb[1], i_b, solver, contact),
                solver._contact_ccd_gap,
                solver._contact_ccd_scale,
                solver._contact_ccd_iterations,
                EDGE_EDGE,
            )
            qd.atomic_min(contact.toi[i_b], toi)
    for i_b in range(solver._B):
        if not solver.env_failed[i_b]:
            qd.atomic_min(contact.min_toi[i_b], contact.toi[i_b])


@qd.kernel
def kernel_accd_rescale_verts(f: int, solver: qd.template(), contact: qd.template()):
    """Hold every tissue vertex to the substep's time of impact."""
    for i_v, i_b in qd.ndrange(solver._n_vertices, solver._B):
        if not solver.env_failed[i_b] and contact.toi[i_b] < 1.0:
            start = solver.verts[f, i_v, i_b].pos
            solver.verts[f + 1, i_v, i_b].pos = start + contact.toi[i_b] * (solver.verts[f + 1, i_v, i_b].pos - start)


@qd.kernel
def kernel_accd_rescale_links(solver: qd.template(), contact: qd.template(), attachment: qd.template()):
    """Hold every free rigid body to the substep's time of impact, and refresh what reads its pose.

    The body's velocity is the pose difference over the substep (`kernel_end_attachment`), so rescaling the pose
    rescales the velocity with it: no separate impulse is needed and none is applied.
    """
    for i_f, i_b in qd.ndrange(attachment.n_free, solver._B):
        if not solver.env_failed[i_b] and contact.toi[i_b] < 1.0:
            state = attachment.link_state[i_f, i_b]
            pos = state.previous_pos + contact.toi[i_b] * (state.pos - state.previous_pos)
            quat = func_slerp(state.previous_quat, state.quat, contact.toi[i_b])
            attachment.link_state[i_f, i_b].pos = pos
            attachment.link_state[i_f, i_b].quat = quat
            i_l = attachment.free_info[i_f].link
            attachment.link_pose[i_l, i_b].pos = pos
            attachment.link_pose[i_l, i_b].quat = quat
            func_refresh_link_vertices(i_l, i_b, pos, quat, contact)
            if qd.static(solver.has_mtu):
                func_refresh_mtu_link_anchors(i_l, i_b, pos, quat, solver.mtu)
