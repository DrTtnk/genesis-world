"""Joints between rigid links: a six-axis spring in a frame fixed to each link at rest.

A joint has a centre and three orthonormal axes, given in the world at the rest pose and stored in each link's
own frame. With x = p + R c for each link's copy of the centre, d = x_b - x_a, e_i = R_a a_i and f_i = R_b b_i
(the axes carried by link a and by link b), its energy is

    E = 1/2 sum_i kt_i (e_i . d)^2 + 1/2 sum_i kr_i |e_i x f_i|^2:

a translational spring along each axis of link a, and an alignment spring between each axis and its twin on link
b. It is zero at rest and smooth everywhere, so a joint has no slack point to cross. Anatomical joints are this
element with some stiffnesses zero: a hinge aligns its axis only and turns freely about it, a slide leaves
translation along its axis free, a ball aligns nothing, a suture holds all six, a syndesmosis holds all six
softly.

A free link's block takes the gradient and the Gauss-Newton curvature of the joint with respect to its increment
(dp, dtheta), dtheta the world-frame rotation of the VBD rigid block:

    du_i/dp_a = -e_i,  du_i/dtheta_a = e_i x (d + r_a),  du_i/dp_b = e_i,  du_i/dtheta_b = r_b x e_i
    dw_i/dtheta_a = e_i f_i^T - (e_i . f_i) I,  dw_i/dtheta_b = (e_i . f_i) I - f_i e_i^T

with u_i = e_i . d, w_i = e_i x f_i and r = R c. `spikes/verify_joint_blocks.py` in the snakeSim repository checks
these against Torch autograd of E on random poses; at rest the Gauss-Newton block is the exact Hessian.
"""

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array


class VBDJoints:
    def __init__(self, solver, joints):
        rigid = solver.sim.rigid_solver
        attachment = solver.rigid_attachment
        B = solver._B
        positions = tensor_to_array(rigid.get_links_pos()).reshape(B, rigid.n_links, 3)[0]
        quaternions = tensor_to_array(rigid.get_links_quat()).reshape(B, rigid.n_links, 4)[0]
        free_slot = attachment.free_slot.to_numpy()
        info_type = qd.types.struct(
            link_a=gs.qd_int, link_b=gs.qd_int, centre_a=gs.qd_vec3, centre_b=gs.qd_vec3,
            axes_a=gs.qd_mat3, axes_b=gs.qd_mat3, kt=gs.qd_vec3, kr=gs.qd_vec3,
        )
        self.n_joints = len(joints)
        self.info = info_type.field(shape=self.n_joints)
        touching = [[] for _ in range(rigid.n_links)]
        for i, (link_a, link_b, centre, axes, kt, kr) in enumerate(joints):
            for link in (link_a, link_b):
                if free_slot[link.idx] < 0 and not link.is_fixed:
                    gs.raise_exception(f"Link {link.name} is neither free nor fixed, so a joint cannot move it.")
            local = []
            for link in (link_a, link_b):
                offset = gu.inv_transform_by_quat(np.asarray(centre) - positions[link.idx], quaternions[link.idx])
                frame = gu.quat_to_R(quaternions[link.idx]).T @ np.asarray(axes)
                local.append((offset, frame))
            self.info[i].link_a = link_a.idx
            self.info[i].link_b = link_b.idx
            self.info[i].centre_a = local[0][0].tolist()
            self.info[i].centre_b = local[1][0].tolist()
            self.info[i].axes_a = local[0][1].tolist()
            self.info[i].axes_b = local[1][1].tolist()
            self.info[i].kt = [float(k) for k in kt]
            self.info[i].kr = [float(k) for k in kr]
            touching[link_a.idx].append(i)
            touching[link_b.idx].append(i)
        offsets = np.concatenate([[0], np.cumsum([len(t) for t in touching])]).astype(gs.np_int)
        self.link_joint_offset = qd.field(dtype=gs.qd_int, shape=rigid.n_links + 1)
        self.link_joint_offset.from_numpy(offsets)
        self.link_joint = qd.field(dtype=gs.qd_int, shape=max(int(offsets[-1]), 1))
        if offsets[-1]:
            self.link_joint.from_numpy(np.concatenate([t for t in touching if t]).astype(gs.np_int))


@qd.func
def func_joint_link_terms(i_l, i_b, attachment: qd.template(), joints: qd.template()):
    """Wrench (force, torque about the link origin) and 6x6 Gauss-Newton block of every joint on link i_l, for a
    free link with the world-frame rotation increment of the attachment block. Both links are read from
    `attachment.link_pose`, which holds each free body's latest solved pose and each fixed link's own."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    for c in range(joints.link_joint_offset[i_l], joints.link_joint_offset[i_l + 1]):
        joint_force, joint_hessian = func_joint_terms(i_l, c, i_b, attachment, joints)
        force6 += joint_force
        hessian6 += joint_hessian
    return force6, hessian6


@qd.func
def func_joint_terms(i_l, c, i_b, attachment: qd.template(), joints: qd.template()):
    """The joint at `link_joint[c]`, seen from link i_l."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    identity = qd.Matrix.identity(gs.qd_float, 3)
    info = joints.info[joints.link_joint[c]]
    pa = attachment.link_pose[info.link_a, i_b].pos
    pb = attachment.link_pose[info.link_b, i_b].pos
    Ra = gu.qd_quat_to_R(attachment.link_pose[info.link_a, i_b].quat, gs.EPS)
    Rb = gu.qd_quat_to_R(attachment.link_pose[info.link_b, i_b].quat, gs.EPS)
    ra = Ra @ info.centre_a
    rb = Rb @ info.centre_b
    d = pb + rb - pa - ra
    for i in qd.static(range(3)):
        e = Ra @ gs.qd_vec3(info.axes_a[0, i], info.axes_a[1, i], info.axes_a[2, i])
        f = Rb @ gs.qd_vec3(info.axes_b[0, i], info.axes_b[1, i], info.axes_b[2, i])
        J = qd.Vector.zero(gs.qd_float, 6)
        Jw = qd.Matrix.zero(gs.qd_float, 3, 3)
        if i_l == info.link_a:
            torque_arm = e.cross(d + ra)
            Jw = e.outer_product(f) - e.dot(f) * identity
            for j in qd.static(range(3)):
                J[j] = -e[j]
                J[j + 3] = torque_arm[j]
        else:
            torque_arm = rb.cross(e)
            Jw = e.dot(f) * identity - f.outer_product(e)
            for j in qd.static(range(3)):
                J[j] = e[j]
                J[j + 3] = torque_arm[j]
        u = e.dot(d)
        force6 -= info.kt[i] * u * J
        hessian6 += info.kt[i] * J.outer_product(J)
        w = e.cross(f)
        torque = Jw.transpose() @ w
        block = Jw.transpose() @ Jw
        for j in qd.static(range(3)):
            force6[j + 3] -= info.kr[i] * torque[j]
            for k in qd.static(range(3)):
                hessian6[j + 3, k + 3] += info.kr[i] * block[j, k]
    return force6, hessian6
