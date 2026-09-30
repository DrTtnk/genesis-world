"""Colours of the free bodies for the rigid block Gauss-Seidel of a sweep.

A free body's 6x6 block reads the newest pose of every body it shares an element with: a joint, a muscle-tendon
unit (its tension follows the whole route length), a tetrahedron glued to two links, a native rod segment glued to
two links, and, this substep, a contact pair. Tissue vertices do not count: the rigid loop never moves them. Two
bodies that share none of these can be solved at the same time, so the bodies are coloured greedily, once a
substep after the contact search, and a sweep solves one colour after the other. That is Gauss-Seidel in the order
of the colours.

The static couplings are fixed at build; the contact couplings are read from the substep's candidate slots, which
change only when the contact search runs.

Inside a colour, one body can carry most of the work: a jaw with a few hundred contact slots, or a bone with many
glued rod ends and long muscle routes. So each colour runs in three passes, each one thread an item:

1. entries, at the poses the colour's bodies have before it moves: a glued vertex's inertia, a glued tetrahedron,
   the rod segments at a glued vertex, a contact slot, a muscle link anchor, a joint;
2. bodies: the inertia and vertex attachments, the entries summed in the order the serial block visits them, the
   6x6 solve and the new pose;
3. refresh, at the new poses: glued vertices, active rigid contact vertices and muscle link anchors.

The serial block and this one sum the same terms in the same order; they agree to rounding (a term summed where it
is computed can have its last multiply fused into the add, one read back from the entry buffer cannot).
"""

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.vbd_contact import (
    func_contact_slot_link_terms,
    func_is_first_link_slot,
    func_pair_participants,
)
from genesis.engine.solvers.vbd_joint import func_joint_terms
from genesis.engine.solvers.vbd_mtu import KIND_LINK, func_mtu_anchor_link_terms
from genesis.engine.solvers.vbd_rigid_attachment import (
    func_attachment_link_own,
    func_glue_elem_terms,
    func_glue_rod_vertex_terms,
    func_glue_vertex_terms,
    func_move_attachment_link,
)
from genesis.utils.array_class import ErrorCode

# entry kinds, in the order a body's entries are laid out and summed
ENTRY_GLUE_VERTEX, ENTRY_GLUE_ELEM, ENTRY_GLUE_ROD, ENTRY_CONTACT, ENTRY_MTU, ENTRY_JOINT = range(6)
N_ENTRY_KINDS = 6
REFRESH_GLUE, REFRESH_CONTACT, REFRESH_MTU = range(3)


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
        n_link_anchors = 0
        if solver.mtu is not None:
            mtu = solver.mtu
            links = mtu.anchor.link.to_numpy()
            kinds = mtu.anchor.kind.to_numpy()
            n_link_anchors = int(mtu.link_anchor_offset.to_numpy()[-1])
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
        # the entries of a substep, bodies in colour order, each body's laid out kind by kind: body_entry[i_f] holds
        # where each kind starts, and where the body's entries end
        self.entry_cap = entry_cap
        self.entry_kind = qd.field(dtype=gs.qd_int, shape=(entry_cap, solver._B))
        self.entry_index = qd.field(dtype=gs.qd_int, shape=(entry_cap, solver._B))
        self.entry_body = qd.field(dtype=gs.qd_int, shape=(entry_cap, solver._B))
        self.entry_force = qd.Vector.field(6, dtype=gs.qd_float, shape=(entry_cap, solver._B))
        self.entry_hessian = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=(entry_cap, solver._B))
        self.body_entry = qd.Vector.field(N_ENTRY_KINDS + 1, dtype=gs.qd_int, shape=(n_free, solver._B))
        self.colour_entry_offset = qd.field(dtype=gs.qd_int, shape=(cap + 1, solver._B))
        self.colour_entry_max = qd.field(dtype=gs.qd_int, shape=cap)  # over the environments
        # the refresh items: every glued vertex, rigid contact vertex and muscle link anchor at most once, so the
        # list never outgrows these
        refresh_cap = max(attachment.n_glued + (solver.contact.n_rv if solver.contact is not None else 0)
                          + n_link_anchors, 1)
        self.refresh_kind = qd.field(dtype=gs.qd_int, shape=(refresh_cap, solver._B))
        self.refresh_index = qd.field(dtype=gs.qd_int, shape=(refresh_cap, solver._B))
        self.refresh_body = qd.field(dtype=gs.qd_int, shape=(refresh_cap, solver._B))
        self.colour_refresh_offset = qd.field(dtype=gs.qd_int, shape=(cap + 1, solver._B))
        self.colour_refresh_max = qd.field(dtype=gs.qd_int, shape=cap)


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
            cvs, _ = func_pair_participants(contact.cv_slot[slot, i_b], i_b, contact)
            for j in qd.static(range(4)):
                if contact.cv_info[cvs[j]].kind == 1:
                    other = attachment.free_slot[contact.cv_info[cvs[j]].owner]
                    if other >= 0 and other != i_f and colouring.colour[other, i_b] >= 0:
                        used |= 1 << colouring.colour[other, i_b]
    return used


@qd.func
def func_append_entry(n, kind, index, i_f, i_b, colouring: qd.template()):
    """Write one entry if it fits; the count runs on either way, so an overflow is measured, then refused."""
    if n < colouring.entry_cap:
        colouring.entry_kind[n, i_b] = kind
        colouring.entry_index[n, i_b] = index
        colouring.entry_body[n, i_b] = i_f
    return n + 1


@qd.func
def func_append_refresh(n, kind, index, i_f, i_b, colouring: qd.template()):
    colouring.refresh_kind[n, i_b] = kind
    colouring.refresh_index[n, i_b] = index
    colouring.refresh_body[n, i_b] = i_f
    return n + 1


@qd.func
def func_append_body_items(i_f, i_b, n, n_refresh, solver: qd.template(), attachment: qd.template(),
                           colouring: qd.template()):
    """One body's entries, kind by kind in the order the serial block sums them, and its refresh items."""
    i_l = attachment.free_info[i_f].link
    colouring.body_entry[i_f, i_b][ENTRY_GLUE_VERTEX] = n
    if qd.static(attachment.has_glue):
        for c in range(attachment.link_glue_vert_offset[i_l], attachment.link_glue_vert_offset[i_l + 1]):
            n = func_append_entry(n, ENTRY_GLUE_VERTEX, c, i_f, i_b, colouring)
            n_refresh = func_append_refresh(n_refresh, REFRESH_GLUE, attachment.link_glue_vert[c], i_f, i_b,
                                            colouring)
    colouring.body_entry[i_f, i_b][ENTRY_GLUE_ELEM] = n
    if qd.static(attachment.has_glue):
        for c in range(attachment.link_glue_elem_offset[i_l], attachment.link_glue_elem_offset[i_l + 1]):
            n = func_append_entry(n, ENTRY_GLUE_ELEM, c, i_f, i_b, colouring)
    colouring.body_entry[i_f, i_b][ENTRY_GLUE_ROD] = n
    if qd.static(attachment.has_glue and solver.has_rod_native):
        for c in range(attachment.link_glue_vert_offset[i_l], attachment.link_glue_vert_offset[i_l + 1]):
            n = func_append_entry(n, ENTRY_GLUE_ROD, c, i_f, i_b, colouring)
    colouring.body_entry[i_f, i_b][ENTRY_CONTACT] = n
    if qd.static(solver.has_contact):
        base = solver.contact.link_rv_offset[i_l]
        for c in range(base, base + solver.contact.link_active_n[i_l, i_b]):
            i_r = solver.contact.link_active[c, i_b]
            n_refresh = func_append_refresh(n_refresh, REFRESH_CONTACT, i_r, i_f, i_b, colouring)
            cv = solver.contact.rv_cv[i_r]
            for slot in range(solver.contact.cv_slot_offset[cv, i_b], solver.contact.cv_slot_offset[cv + 1, i_b]):
                # a slot that is not its pair's first on this link brings an exact zero: leave it out
                if func_is_first_link_slot(solver.contact.cv_slot[slot, i_b], i_l, i_b, solver.contact):
                    n = func_append_entry(n, ENTRY_CONTACT, slot, i_f, i_b, colouring)
    colouring.body_entry[i_f, i_b][ENTRY_MTU] = n
    if qd.static(solver.has_mtu):
        for slot in range(solver.mtu.link_anchor_offset[i_l], solver.mtu.link_anchor_offset[i_l + 1]):
            n = func_append_entry(n, ENTRY_MTU, slot, i_f, i_b, colouring)
            n_refresh = func_append_refresh(n_refresh, REFRESH_MTU, solver.mtu.link_anchor[slot], i_f, i_b,
                                            colouring)
    colouring.body_entry[i_f, i_b][ENTRY_JOINT] = n
    if qd.static(solver.has_joint):
        for c in range(solver.joints.link_joint_offset[i_l], solver.joints.link_joint_offset[i_l + 1]):
            n = func_append_entry(n, ENTRY_JOINT, c, i_f, i_b, colouring)
    colouring.body_entry[i_f, i_b][N_ENTRY_KINDS] = n
    return n, n_refresh


@qd.kernel
def kernel_colour_free_bodies(substep_global: int, solver: qd.template(), attachment: qd.template(),
                              colouring: qd.template()):
    """Greedy colouring in slot order, one thread an environment, then the bodies, their entries and their
    refresh items grouped by colour."""
    for k in range(colouring.cap):
        colouring.colour_entry_max[k] = 0
        colouring.colour_refresh_max[k] = 0
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
            n_refresh = 0
            for k in range(colouring.cap):
                colouring.colour_entry_offset[k, i_b] = n_entries
                colouring.colour_refresh_offset[k, i_b] = n_refresh
                for m in range(colouring.colour_offset[k, i_b], colouring.colour_offset[k + 1, i_b]):
                    n_entries, n_refresh = func_append_body_items(
                        colouring.colour_body[m, i_b], i_b, n_entries, n_refresh, solver, attachment, colouring
                    )
            colouring.colour_entry_offset[colouring.cap, i_b] = n_entries
            colouring.colour_refresh_offset[colouring.cap, i_b] = n_refresh
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
            qd.atomic_max(
                colouring.colour_refresh_max[k],
                colouring.colour_refresh_offset[k + 1, i_b] - colouring.colour_refresh_offset[k, i_b],
            )


@qd.func
def func_rigid_entry_terms(f, e, i_b, solver: qd.template(), attachment: qd.template(), colouring: qd.template()):
    """Pass 1 of a colour: one entry's wrench and block, at the pose its body has before the colour moves."""
    i_f = colouring.entry_body[e, i_b]
    i_l = attachment.free_info[i_f].link
    origin = attachment.link_state[i_f, i_b].pos
    kind = colouring.entry_kind[e, i_b]
    index = colouring.entry_index[e, i_b]
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    if qd.static(attachment.has_glue):
        if kind == ENTRY_GLUE_VERTEX:
            force6, hessian6 = func_glue_vertex_terms(f, index, i_b, origin, solver, attachment)
        if kind == ENTRY_GLUE_ELEM:
            force6, hessian6 = func_glue_elem_terms(f, index, i_b, origin, solver, attachment)
    if qd.static(attachment.has_glue and solver.has_rod_native):
        if kind == ENTRY_GLUE_ROD:
            force6, hessian6 = func_glue_rod_vertex_terms(f, i_l, index, i_b, origin, solver, attachment,
                                                          solver.rod_native)
    if qd.static(solver.has_contact):
        if kind == ENTRY_CONTACT:
            force6, hessian6 = func_contact_slot_link_terms(f, i_l, i_b, origin, index, solver, solver.contact)
    if qd.static(solver.has_mtu):
        if kind == ENTRY_MTU:
            force6, hessian6 = func_mtu_anchor_link_terms(f, index, i_b, origin, solver, solver.mtu)
    if qd.static(solver.has_joint):
        if kind == ENTRY_JOINT:
            force6, hessian6 = func_joint_terms(i_l, index, i_b, attachment, solver.joints)
    colouring.entry_force[e, i_b] = force6
    colouring.entry_hessian[e, i_b] = hessian6


@qd.func
def func_sum_entries(begin, end, i_b, colouring: qd.template()):
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    for e in range(begin, end):
        force6 += colouring.entry_force[e, i_b]
        hessian6 += colouring.entry_hessian[e, i_b]
    return force6, hessian6


@qd.func
def func_attachment_link_system_entries(f, i_f, i_b, solver: qd.template(), attachment: qd.template(),
                                        colouring: qd.template()):
    """`func_attachment_link_system` with every per-item sum read from the entries, each summed alone in the
    serial order and then added as the serial block adds it."""
    force, hessian = func_attachment_link_own(f, i_f, i_b, solver, attachment)
    span = colouring.body_entry[i_f, i_b]
    if qd.static(attachment.has_glue):
        # func_glue_link_terms: vertices and tetrahedra in one sum, then the rod rows
        force_g, hessian_g = func_sum_entries(span[ENTRY_GLUE_VERTEX], span[ENTRY_GLUE_ROD], i_b, colouring)
        if qd.static(solver.has_rod_native):
            force_r, hessian_r = func_sum_entries(span[ENTRY_GLUE_ROD], span[ENTRY_CONTACT], i_b, colouring)
            force_g += force_r
            hessian_g += hessian_r
        force += force_g
        hessian += hessian_g
    if qd.static(solver.has_contact):
        force_c, hessian_c = func_sum_entries(span[ENTRY_CONTACT], span[ENTRY_MTU], i_b, colouring)
        force += force_c
        hessian += hessian_c
    if qd.static(solver.has_mtu):
        force_m, hessian_m = func_sum_entries(span[ENTRY_MTU], span[ENTRY_JOINT], i_b, colouring)
        force += force_m
        hessian += hessian_m
    if qd.static(solver.has_joint):
        force_j, hessian_j = func_sum_entries(span[ENTRY_JOINT], span[N_ENTRY_KINDS], i_b, colouring)
        force += force_j
        hessian += hessian_j
    if qd.static(solver.has_rod_contact):
        i_l = attachment.free_info[i_f].link
        force += solver._rod_contacts[0].force[i_l]
        hessian += solver._rod_contacts[0].hessian[i_l]
    return force, hessian


@qd.func
def func_solve_attachment_link_entries(f, i_f, i_b, solver: qd.template(), attachment: qd.template(),
                                       colouring: qd.template()):
    """Pass 2 of a colour: the body's block from its entries, solved; its caches are pass 3's."""
    force, hessian = func_attachment_link_system_entries(f, i_f, i_b, solver, attachment, colouring)
    func_move_attachment_link(i_f, i_b, force, hessian, attachment)


@qd.func
def func_refresh_item(f, r, i_b, solver: qd.template(), attachment: qd.template(), colouring: qd.template()):
    """Pass 3 of a colour: one glued vertex, rigid contact vertex or muscle link anchor at its body's new pose."""
    i_f = colouring.refresh_body[r, i_b]
    pos = attachment.link_state[i_f, i_b].pos
    quat = attachment.link_state[i_f, i_b].quat
    kind = colouring.refresh_kind[r, i_b]
    index = colouring.refresh_index[r, i_b]
    if qd.static(attachment.has_glue):
        if kind == REFRESH_GLUE:
            solver.verts[f + 1, index, i_b].pos = gu.qd_transform_by_trans_quat(attachment.glue_local[index], pos, quat)
    if qd.static(solver.has_contact):
        if kind == REFRESH_CONTACT:
            solver.contact.rv_pos[index, i_b] = gu.qd_transform_by_trans_quat(solver.contact.rv_local[index], pos, quat)
    if qd.static(solver.has_mtu):
        if kind == REFRESH_MTU:
            solver.mtu.anchor_pos[index, i_b] = gu.qd_transform_by_trans_quat(solver.mtu.anchor[index].local, pos, quat)


@qd.kernel
def kernel_clear_rigid_colour_errno(envs_idx: qd.types.ndarray(), colouring: qd.template()):
    for i in range(envs_idx.shape[0]):
        colouring.errno[envs_idx[i]] = 0
