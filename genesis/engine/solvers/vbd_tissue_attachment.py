"""Point attachments between two tissues for vertex block descent.

An attachment holds a material point of one VBD entity on a material point of another (or of the same one):
C = P_a - P_b - d0 = 0, where P is the weighted sum of up to four vertices, a `TissueAnchor` or a
`SurfaceAnchor` resolved as in vbd_mtu.py, and d0 is the offset the two points had when they were declared, so
the binding is stress-free at rest. It is the augmented Lagrangian of `vbd_rigid_attachment.py` with a second
soft participant instead of a link: energy lam . C + k |C|^2 / 2 with y = lam + k (C - alpha C_prev), force
-w_j y on the side-a vertices and +u_k y on the side-b vertices, curvature w_j^2 k I and u_k^2 k I, the dual
update lam += r k C and the stiffness ladder from the solver's constraint options. There is no adjoint yet, so
the solver refuses it under requires_grad.
"""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.utils.misc import tensor_to_array


class VBDTissueAttachment:
    def __init__(self, solver, pairs):
        self.solver = solver
        self.n_attachments = len(pairs)
        self.alpha = 0.95
        self.gamma = 0.99
        verts_a, weights_a, verts_b, weights_b, offsets = [], [], [], [], []
        for (entity_a, va, wa), (entity_b, vb, wb) in pairs:
            init_a, init_b = tensor_to_array(entity_a.init_positions), tensor_to_array(entity_b.init_positions)
            verts_a.append(np.asarray(va, dtype=gs.np_int) + entity_a.v_start)
            verts_b.append(np.asarray(vb, dtype=gs.np_int) + entity_b.v_start)
            weights_a.append(np.asarray(wa, dtype=gs.np_float))
            weights_b.append(np.asarray(wb, dtype=gs.np_float))
            offsets.append(np.asarray(wa) @ init_a[list(va)] - np.asarray(wb) @ init_b[list(vb)])
        verts_a, verts_b = np.array(verts_a, dtype=gs.np_int), np.array(verts_b, dtype=gs.np_int)
        weights_a, weights_b = np.array(weights_a, dtype=gs.np_float), np.array(weights_b, dtype=gs.np_float)
        info_type = qd.types.struct(
            verts_a=gs.qd_ivec4, weights_a=gs.qd_vec4, verts_b=gs.qd_ivec4, weights_b=gs.qd_vec4, offset=gs.qd_vec3
        )
        state_type = qd.types.struct(multiplier=gs.qd_vec3, stiffness=gs.qd_float)
        self.info = info_type.field(shape=self.n_attachments, layout=qd.Layout.SOA)
        self.state = state_type.field(shape=(self.n_attachments, solver._B), layout=qd.Layout.SOA)
        self.previous_error = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_attachments, solver._B))
        # A vertex reaches its attachments through a CSR, as in vbd_rigid_attachment.py; the payload is
        # attachment * 8 + corner, corners 0..3 on side a and 4..7 on side b, so one vertex can support both
        # sides of one attachment or several attachments without a special case.
        entries = sorted(
            [(int(verts_a[i_a, c]), 8 * i_a + c) for i_a in range(self.n_attachments) for c in range(4) if weights_a[i_a, c] != 0.0]
            + [(int(verts_b[i_a, c]), 8 * i_a + 4 + c) for i_a in range(self.n_attachments) for c in range(4) if weights_b[i_a, c] != 0.0]
        )
        counts = np.zeros(solver.n_vertices + 1, dtype=gs.np_int)
        for vertex, _ in entries:
            counts[vertex + 1] += 1
        self.vert_anchor_offset = qd.field(dtype=gs.qd_int, shape=solver.n_vertices + 1)
        self.vert_anchor_offset.from_numpy(np.cumsum(counts).astype(gs.np_int))
        self.vert_anchor = qd.field(dtype=gs.qd_int, shape=max(len(entries), 1))
        self.vert_anchor.from_numpy(np.array([slot for _, slot in entries] or [0], dtype=gs.np_int))
        self.info.verts_a.from_numpy(verts_a)
        self.info.weights_a.from_numpy(weights_a)
        self.info.verts_b.from_numpy(verts_b)
        self.info.weights_b.from_numpy(weights_b)
        self.info.offset.from_numpy(np.array(offsets, dtype=gs.np_float))
        self.state.multiplier.fill(0.0)
        self.state.stiffness.fill(solver._k_start)


@qd.func
def func_tissue_attachment_gap(f, i_a, i_b, solver: qd.template(), attachment: qd.template()):
    """C = P_a - P_b - d0 at frame f."""
    gap = -attachment.info[i_a].offset
    for corner in qd.static(range(4)):
        gap += attachment.info[i_a].weights_a[corner] * solver.verts[f, attachment.info[i_a].verts_a[corner], i_b].pos
        gap -= attachment.info[i_a].weights_b[corner] * solver.verts[f, attachment.info[i_a].verts_b[corner], i_b].pos
    return gap


@qd.kernel
def kernel_begin_tissue_attachment(f: int, solver: qd.template(), attachment: qd.template()):
    for i_a, i_b in qd.ndrange(attachment.n_attachments, solver._B):
        if not solver.env_failed[i_b]:
            attachment.previous_error[i_a, i_b] = func_tissue_attachment_gap(f, i_a, i_b, solver, attachment)
            attachment.state[i_a, i_b].multiplier *= attachment.alpha * attachment.gamma
            attachment.state[i_a, i_b].stiffness = qd.max(
                solver._k_start, attachment.gamma * attachment.state[i_a, i_b].stiffness
            )


@qd.func
def func_tissue_attachment_vertex_terms(f, i_v, i_b, solver: qd.template(), attachment: qd.template()):
    """Force and curvature block every tissue attachment supported by vertex i_v applies to it."""
    force = gs.qd_vec3(0.0, 0.0, 0.0)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    for slot in range(attachment.vert_anchor_offset[i_v], attachment.vert_anchor_offset[i_v + 1]):
        payload = attachment.vert_anchor[slot]
        i_a = payload // 8
        corner = payload % 8
        weight = 0.0
        sign = 1.0
        if corner < 4:
            weight = attachment.info[i_a].weights_a[corner]
        else:
            weight = attachment.info[i_a].weights_b[corner - 4]
            sign = -1.0
        constraint = (
            func_tissue_attachment_gap(f + 1, i_a, i_b, solver, attachment)
            - attachment.alpha * attachment.previous_error[i_a, i_b]
        )
        stiffness = attachment.state[i_a, i_b].stiffness
        force -= sign * weight * (attachment.state[i_a, i_b].multiplier + stiffness * constraint)
        hessian += weight * weight * stiffness * qd.Matrix.identity(gs.qd_float, 3)
    return force, hessian


@qd.func
def func_update_tissue_attachment_dual(f, i_a, i_b, solver: qd.template(), attachment: qd.template()):
    error = (
        func_tissue_attachment_gap(f + 1, i_a, i_b, solver, attachment)
        - attachment.alpha * attachment.previous_error[i_a, i_b]
    )
    stiffness = attachment.state[i_a, i_b].stiffness
    attachment.state[i_a, i_b].multiplier += solver._constraint_dual_relaxation * stiffness * error
    attachment.state[i_a, i_b].stiffness = qd.min(
        stiffness + solver._k_start / solver._constraint_tol * error.norm(),
        solver._constraint_k_max_ratio * solver._k_start,
    )


@qd.kernel
def kernel_set_tissue_attachment_state(
    envs_idx: qd.types.ndarray(), multiplier: qd.types.ndarray(), stiffness: qd.types.ndarray(), state: qd.template()
):
    for i_a, i_b_ in qd.ndrange(state.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            state[i_a, i_b].multiplier[j] = multiplier[i_b, i_a, j]
        state[i_a, i_b].stiffness = stiffness[i_b, i_a]
