"""Colours of the free bodies for the rigid block Gauss-Seidel of a sweep.

A free body's 6x6 block reads the newest pose of every body it shares an element with: a joint, a muscle-tendon
unit (its tension follows the whole route length), a tetrahedron glued to two links, a native rod segment glued to
two links, and, this substep, a contact pair. Tissue vertices do not count: the rigid loop never moves them. Two
bodies that share none of these can be solved at the same time, so the bodies are coloured greedily, once a
substep after the contact search, and a sweep solves one colour after the other with one thread a body. That is
Gauss-Seidel in the order of the colours, with every body's own terms summed as before.

The static couplings are fixed at build; the contact couplings are read from the substep's candidate slots, which
change only when the contact search runs.

Inside a colour, one body can carry most of the work: a jaw with a few hundred contact slots, or a bone with the
link anchors of long muscle routes. So each colour runs in two passes. The first computes every entry of the
colour's bodies at once, one thread an entry: a contact slot's pair terms or a link anchor's route pull. The second
solves each body's block with one thread a body, summing its entries in the order the serial block visits them, so
the block is the same bit for bit.
"""

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.vbd_contact import ROLE_EDGE_A, func_contact_slot_link_terms
from genesis.engine.solvers.vbd_mtu import KIND_LINK, func_mtu_anchor_link_terms
from genesis.engine.solvers.vbd_rigid_attachment import (
    func_apply_attachment_link,
    func_attachment_link_base,
    func_attachment_link_tail,
)
from genesis.utils.array_class import ErrorCode


class VBDRigidColouring:
    def __init__(self, solver, cap, entry_cap):
        attachment = solver.rigid_attachment
        if cap > 31:
            gs.raise_exception("VBDOptions.rigid_colour_cap is at most 31: the used colours are one 32-bit mask.")
        self.cap = cap
        n_free = attachment.n_free
        free_slot = attachment.free_slot.to_numpy()
        pairs = set()

        def couple(link_a, link_b):
            a, b = int(free_slot[link_a]), int(free_slot[link_b])
            if a >= 0 and b >= 0 and a != b:
                pairs.add((a, b))
                pairs.add((b, a))

        if solver.joints is not None:
            for a, b in zip(solver.joints.info.link_a.to_numpy(), solver.joints.info.link_b.to_numpy()):
                couple(a, b)
        if solver.mtu is not None:
            mtu = solver.mtu
            links = mtu.anchor.link.to_numpy()
            kinds = mtu.anchor.kind.to_numpy()
            for first, last in zip(mtu.unit.first.to_numpy(), mtu.unit.last.to_numpy()):
                touched = sorted({int(links[i]) for i in range(first, last) if kinds[i] == KIND_LINK})
                for i, a in enumerate(touched):
                    for b in touched[i + 1:]:
                        couple(a, b)
        glue_link = attachment.glue_link.to_numpy()
        if attachment.has_glue and solver._n_elements:
            for corners in glue_link[solver.elems_info.v.to_numpy()[: solver._n_elements]]:
                carried = sorted({int(c) for c in corners if c >= 0})
                for i, a in enumerate(carried):
                    for b in carried[i + 1:]:
                        couple(a, b)
        if solver.rod_native is not None:
            node0 = solver.rod_native.seg.node0.to_numpy()
            for a, b in zip(glue_link[node0], glue_link[node0 + 1]):
                if a >= 0 and b >= 0:
                    couple(a, b)
        edges = np.array(sorted(pairs), dtype=gs.np_int).reshape(-1, 2)
        self.static_offset = qd.field(dtype=gs.qd_int, shape=n_free + 1)
        self.static_offset.from_numpy(np.searchsorted(edges[:, 0], np.arange(n_free + 1)).astype(gs.np_int))
        self.static_neighbour = qd.field(dtype=gs.qd_int, shape=max(len(edges), 1))
        self.static_neighbour.from_numpy(edges[:, 1] if len(edges) else np.zeros(1, dtype=gs.np_int))
        self.colour = qd.field(dtype=gs.qd_int, shape=(n_free, solver._B))
        self.n_colours = qd.field(dtype=gs.qd_int, shape=solver._B)
        self.colour_offset = qd.field(dtype=gs.qd_int, shape=(cap + 1, solver._B))
        self.colour_body = qd.field(dtype=gs.qd_int, shape=(n_free, solver._B))
        self.errno = qd.field(dtype=gs.qd_int, shape=solver._B)
        # the entries of a substep, bodies in colour order: each body's contact slots (code = slot), then its
        # link anchors (code = -1 - anchor slot)
        self.entry_cap = entry_cap
        self.entry_code = qd.field(dtype=gs.qd_int, shape=(entry_cap, solver._B))
        self.entry_body = qd.field(dtype=gs.qd_int, shape=(entry_cap, solver._B))
        self.entry_force = qd.Vector.field(6, dtype=gs.qd_float, shape=(entry_cap, solver._B))
        self.entry_hessian = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=(entry_cap, solver._B))
        self.body_entry = qd.Vector.field(3, dtype=gs.qd_int, shape=(n_free, solver._B))  # begin, anchors, end
        self.colour_entry_offset = qd.field(dtype=gs.qd_int, shape=(cap + 1, solver._B))
        self.colour_entry_max = qd.field(dtype=gs.qd_int, shape=cap)  # over the environments


@qd.func
def func_contact_neighbours_mask(i_f, i_b, attachment: qd.template(), contact: qd.template(),
                                 colouring: qd.template()):
    """Colours already given to the free bodies that share one of this body's candidate contact pairs."""
    used = 0
    i_l = attachment.free_info[i_f].link
    base = contact.link_rv_offset[i_l]
    for c in range(base, base + contact.link_active_n[i_l, i_b]):
        cv = contact.rv_cv[contact.link_active[c, i_b]]
        for slot in range(contact.cv_slot_offset[cv, i_b], contact.cv_slot_offset[cv + 1, i_b]):
            code = contact.cv_slot[slot, i_b]
            i_p = code // 8
            cvs = qd.Vector([0, 0, 0, 0], dt=gs.qd_int)
            if code % 8 < ROLE_EDGE_A:
                tri = contact.tri_cv[contact.pt_pairs[i_p, i_b].b]
                cvs = qd.Vector([contact.pt_pairs[i_p, i_b].a, tri[0], tri[1], tri[2]], dt=gs.qd_int)
            else:
                i_p = i_p - contact.pair_cap
                ea = contact.edge_cv[contact.ee_pairs[i_p, i_b].a]
                eb = contact.edge_cv[contact.ee_pairs[i_p, i_b].b]
                cvs = qd.Vector([ea[0], ea[1], eb[0], eb[1]], dt=gs.qd_int)
            for j in qd.static(range(4)):
                if contact.cv_info[cvs[j]].kind == 1:
                    other = attachment.free_slot[contact.cv_info[cvs[j]].owner]
                    if other >= 0 and other != i_f and colouring.colour[other, i_b] >= 0:
                        used |= 1 << colouring.colour[other, i_b]
    return used


@qd.kernel
def kernel_colour_free_bodies(substep_global: int, solver: qd.template(), attachment: qd.template(),
                              colouring: qd.template()):
    """Greedy colouring in slot order, one thread an environment, then the bodies and their entries grouped by
    colour."""
    for k in range(colouring.cap):
        colouring.colour_entry_max[k] = 0
    for i_b in range(solver._B):
        if not solver.env_failed[i_b]:
            for i_f in range(attachment.n_free):
                colouring.colour[i_f, i_b] = -1
            n_colours = 0
            for i_f in range(attachment.n_free):
                used = 0
                for c in range(colouring.static_offset[i_f], colouring.static_offset[i_f + 1]):
                    other = colouring.static_neighbour[c]
                    if colouring.colour[other, i_b] >= 0:
                        used |= 1 << colouring.colour[other, i_b]
                if qd.static(solver.has_contact):
                    used |= func_contact_neighbours_mask(i_f, i_b, attachment, solver.contact, colouring)
                chosen = -1
                for k in range(colouring.cap):
                    if chosen < 0 and (used >> k) & 1 == 0:
                        chosen = k
                if chosen < 0:
                    colouring.errno[i_b] |= ErrorCode.OVERFLOW_VBD_RIGID_COLOURS
                    solver.env_failed[i_b] = 1
                    solver.failed_substep[i_b] = substep_global
                    chosen = 0
                colouring.colour[i_f, i_b] = chosen
                n_colours = qd.max(n_colours, chosen + 1)
            colouring.n_colours[i_b] = n_colours
            # counting sort: the bodies of colour k sit in colour_body[colour_offset[k]:colour_offset[k + 1]], in
            # slot order
            for k in range(colouring.cap + 1):
                colouring.colour_offset[k, i_b] = 0
            for i_f in range(attachment.n_free):
                colouring.colour_offset[colouring.colour[i_f, i_b] + 1, i_b] += 1
            for k in range(colouring.cap):
                colouring.colour_offset[k + 1, i_b] += colouring.colour_offset[k, i_b]
            for k in range(colouring.cap):
                n = colouring.colour_offset[k, i_b]
                for i_f in range(attachment.n_free):
                    if colouring.colour[i_f, i_b] == k:
                        colouring.colour_body[n, i_b] = i_f
                        n += 1
            n_entries = 0
            for k in range(colouring.cap):
                colouring.colour_entry_offset[k, i_b] = n_entries
                for m in range(colouring.colour_offset[k, i_b], colouring.colour_offset[k + 1, i_b]):
                    i_f = colouring.colour_body[m, i_b]
                    i_l = attachment.free_info[i_f].link
                    colouring.body_entry[i_f, i_b][0] = n_entries
                    if qd.static(solver.has_contact):
                        n_entries = func_append_contact_entries(n_entries, i_f, i_l, i_b, solver.contact, colouring)
                    colouring.body_entry[i_f, i_b][1] = n_entries
                    if qd.static(solver.has_mtu):
                        for slot in range(solver.mtu.link_anchor_offset[i_l], solver.mtu.link_anchor_offset[i_l + 1]):
                            n_entries = func_append_entry(n_entries, -1 - slot, i_f, i_b, colouring)
                    colouring.body_entry[i_f, i_b][2] = n_entries
            colouring.colour_entry_offset[colouring.cap, i_b] = n_entries
            if n_entries > colouring.entry_cap:
                colouring.errno[i_b] |= ErrorCode.OVERFLOW_VBD_RIGID_ENTRIES
                solver.env_failed[i_b] = 1
                solver.failed_substep[i_b] = substep_global
    for k, i_b in qd.ndrange(colouring.cap, solver._B):
        if not solver.env_failed[i_b]:
            qd.atomic_max(
                colouring.colour_entry_max[k],
                colouring.colour_entry_offset[k + 1, i_b] - colouring.colour_entry_offset[k, i_b],
            )


@qd.func
def func_append_contact_entries(n_entries, i_f, i_l, i_b, contact: qd.template(), colouring: qd.template()):
    """The body's candidate slots, in the order `func_contact_link_terms` visits them."""
    base = contact.link_rv_offset[i_l]
    for c in range(base, base + contact.link_active_n[i_l, i_b]):
        cv = contact.rv_cv[contact.link_active[c, i_b]]
        for slot in range(contact.cv_slot_offset[cv, i_b], contact.cv_slot_offset[cv + 1, i_b]):
            n_entries = func_append_entry(n_entries, slot, i_f, i_b, colouring)
    return n_entries


@qd.func
def func_append_entry(n_entries, code, i_f, i_b, colouring: qd.template()):
    """Write one entry if it fits; the count runs on either way, so an overflow is measured, then refused."""
    if n_entries < colouring.entry_cap:
        colouring.entry_code[n_entries, i_b] = code
        colouring.entry_body[n_entries, i_b] = i_f
    return n_entries + 1


@qd.func
def func_rigid_entry_terms(f, e, i_b, solver: qd.template(), attachment: qd.template(), colouring: qd.template()):
    """The first pass of a colour: one entry's wrench and block, at the pose its body has before the colour moves."""
    i_f = colouring.entry_body[e, i_b]
    i_l = attachment.free_info[i_f].link
    origin = attachment.link_state[i_f, i_b].pos
    code = colouring.entry_code[e, i_b]
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    if code >= 0:
        if qd.static(solver.has_contact):
            force6, hessian6 = func_contact_slot_link_terms(f, i_l, i_b, origin, code, solver, solver.contact)
    else:
        if qd.static(solver.has_mtu):
            force6, hessian6 = func_mtu_anchor_link_terms(f, -1 - code, i_b, origin, solver, solver.mtu)
    colouring.entry_force[e, i_b] = force6
    colouring.entry_hessian[e, i_b] = hessian6


@qd.func
def func_solve_attachment_link_entries(f, i_f, i_b, solver: qd.template(), attachment: qd.template(),
                                       colouring: qd.template()):
    """The second pass: `func_solve_attachment_link` with the contact and muscle-tendon sums read from the first
    pass's entries."""
    force, hessian = func_attachment_link_system_entries(f, i_f, i_b, solver, attachment, colouring)
    func_apply_attachment_link(f, i_f, i_b, force, hessian, solver, attachment)


@qd.func
def func_attachment_link_system_entries(f, i_f, i_b, solver: qd.template(), attachment: qd.template(),
                                        colouring: qd.template()):
    """`func_attachment_link_system` with the contact and muscle-tendon sums read from the entries, each summed
    alone in slot order and then added, as the serial block adds them: the same block bit for bit."""
    force, hessian = func_attachment_link_base(f, i_f, i_b, solver, attachment)
    span = colouring.body_entry[i_f, i_b]
    if qd.static(solver.has_contact):
        force_c = qd.Vector.zero(gs.qd_float, 6)
        hessian_c = qd.Matrix.zero(gs.qd_float, 6, 6)
        for e in range(span[0], span[1]):
            force_c += colouring.entry_force[e, i_b]
            hessian_c += colouring.entry_hessian[e, i_b]
        force += force_c
        hessian += hessian_c
    if qd.static(solver.has_mtu):
        force_m = qd.Vector.zero(gs.qd_float, 6)
        hessian_m = qd.Matrix.zero(gs.qd_float, 6, 6)
        for e in range(span[1], span[2]):
            force_m += colouring.entry_force[e, i_b]
            hessian_m += colouring.entry_hessian[e, i_b]
        force += force_m
        hessian += hessian_m
    return func_attachment_link_tail(i_f, i_b, force, hessian, solver, attachment)


@qd.kernel
def kernel_clear_rigid_colour_errno(envs_idx: qd.types.ndarray(), colouring: qd.template()):
    for i in range(envs_idx.shape[0]):
        colouring.errno[envs_idx[i]] = 0
