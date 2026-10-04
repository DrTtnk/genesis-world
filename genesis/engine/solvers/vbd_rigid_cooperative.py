"""Opt-in CUDA prototype: one cooperating warp assembles contact for each serial free-body block.

The free-body order remains Gauss-Seidel. Every lane reads the current pose of one body and its contact pairs;
lane zero adds the other terms, solves and refreshes its pose before any lane starts the next body. This module is
not wired into normal stepping. Its reduction changes floating-point addition order, so it must pass a numerical
parity gate before use.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.vbd_contact import func_contact_pair_link_terms
from genesis.engine.solvers.vbd_rigid import func_ldlt6_solve
from genesis.engine.solvers.vbd_rigid_attachment import (
    func_apply_attachment_increment,
    func_attachment_link_system,
)


_WARP = 32


@qd.func
def func_contact_link_lane_terms(f, i_l, i_b, origin, lane, solver: qd.template(), contact: qd.template()):
    """Current-pose pair terms of active vertices assigned to one lane, in each vertex's slot order."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    base = contact.link_rv_offset[i_l]
    c = base + lane
    while c < base + contact.link_active_n[i_l, i_b]:
        cv = contact.rv_cv[contact.link_active[c, i_b]]
        for slot in range(contact.cv_slot_offset[cv, i_b], contact.cv_slot_offset[cv + 1, i_b]):
            pair_force, pair_hessian, accepted = func_contact_pair_link_terms(
                f, i_l, i_b, origin, contact.cv_slot[slot, i_b], solver, contact
            )
            if accepted:
                force6 += pair_force
                hessian6 += pair_hessian
        c += _WARP
    return force6, hessian6


@qd.func
def func_reduce_contact_warp(force_lane, hessian_lane):
    """Fixed 32-lane tree reduction of the link's six force and 36 block entries."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    for i in qd.static(range(6)):
        force6[i] = qd.simt.subgroup.reduce_all_add_tiled(force_lane[i], 5)
        for j in qd.static(range(6)):
            hessian6[i, j] = qd.simt.subgroup.reduce_all_add_tiled(hessian_lane[i, j], 5)
    return force6, hessian6


@qd.kernel
def kernel_contact_blocks_cooperative(
    f: int,
    solver: qd.template(),
    attachment: qd.template(),
    force_out: qd.template(),
    hessian_out: qd.template(),
):
    """Diagnostic same-pose contact blocks. Output is indexed (free body, environment)."""
    qd.loop_config(name="vbd_contact_blocks_cooperative", block_dim=_WARP)
    for flat in range(solver._B * _WARP):
        lane = flat % _WARP
        i_b = flat // _WARP
        for i_f in range(attachment.n_free):
            i_l = attachment.free_info[i_f].link
            origin = attachment.link_state[i_f, i_b].pos
            force_lane, hessian_lane = func_contact_link_lane_terms(
                f, i_l, i_b, origin, lane, solver, solver.contact
            )
            force6, hessian6 = func_reduce_contact_warp(force_lane, hessian_lane)
            if lane == 0:
                force_out[i_f, i_b] = force6
                hessian_out[i_f, i_b] = hessian6


@qd.kernel
def kernel_sweep_rigid_contact_cooperative(f: int, solver: qd.template(), attachment: qd.template()):
    """Prototype primal rigid sweep; caller keeps the existing dual and vertex sweep schedule."""
    qd.loop_config(name="vbd_rigid_contact_cooperative", block_dim=_WARP)
    for flat in range(solver._B * _WARP):
        lane = flat % _WARP
        i_b = flat // _WARP
        if not solver.env_failed[i_b]:
            for i_f in range(attachment.n_free):
                i_l = attachment.free_info[i_f].link
                origin = attachment.link_state[i_f, i_b].pos
                force_lane, hessian_lane = func_contact_link_lane_terms(
                    f, i_l, i_b, origin, lane, solver, solver.contact
                )
                contact_force, contact_hessian = func_reduce_contact_warp(force_lane, hessian_lane)
                if lane == 0:
                    force, hessian = func_attachment_link_system(f, i_f, i_b, solver, attachment, False)
                    increment = func_ldlt6_solve(hessian + contact_hessian, force + contact_force)
                    func_apply_attachment_increment(f, i_f, i_b, increment, solver, attachment)
                qd.simt.block.sync()


def contact_blocks_cooperative(f, solver, attachment, force_out, hessian_out):
    """Run the CUDA-only same-pose block probe. CPU use is an error, never a fallback."""
    if gs.backend != gs.cuda:
        gs.raise_exception("Cooperative VBD rigid contact assembly requires CUDA.")
    kernel_contact_blocks_cooperative(f, solver, attachment, force_out, hessian_out)


def sweep_rigid_contact_cooperative(f, solver, attachment):
    """Run the CUDA-only prototype primal rigid sweep without changing normal stepping."""
    if gs.backend != gs.cuda:
        gs.raise_exception("Cooperative VBD rigid contact assembly requires CUDA.")
    kernel_sweep_rigid_contact_cooperative(f, solver, attachment)
