"""Coordinate descent for tetrahedral attachments to fixed-base hinge chains.

The mass matrix is frozen at the start of each substep. Each hinge update
uses its exact first derivative and a nonnegative diagonal curvature bound.
"""

import quadrants as qd

import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_forward_kinematics_batch
from genesis.engine.solvers.vbd_rigid_attachment import func_update_attachment_dual
from genesis.utils.array_class import DynInfo, DynState, RigidInfo


@qd.kernel
def kernel_sweeps_articulation(
    f: int,
    solver: qd.template(),
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
):
    for sweep in qd.static(range(solver._n_iterations)):
        solver._func_sweep(f, sweep)
        for i_b in range(solver._B):
            func_solve_articulation_batch(
                f, i_b, solver, solver.rigid_attachment, dyn_state, dyn_info, rigid_info, rigid_config
            )
            for i_l in range(solver.rigid_attachment.rigid.n_links):
                solver.rigid_attachment.articulation_pose[i_l, i_b].pos = dyn_state.links.pos[i_l, i_b]
                solver.rigid_attachment.articulation_pose[i_l, i_b].quat = dyn_state.links.quat[i_l, i_b]
        for i_a, i_b in qd.ndrange(solver.rigid_attachment.n_attachments, solver._B):
            func_update_attachment_dual(f, i_a, i_b, solver, solver.rigid_attachment)


@qd.kernel
def kernel_begin_articulation(
    f: int,
    solver: qd.template(),
    attachment: qd.template(),
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
):
    for i_b in range(solver._B):
        for i_a in range(attachment.n_attachments):
            i_l = attachment.info[i_a].link
            anchor = gu.qd_transform_by_trans_quat(
                attachment.info[i_a].local_pos, dyn_state.links.pos[i_l, i_b], dyn_state.links.quat[i_l, i_b]
            )
            attachment.previous_error[i_a, i_b] = solver.verts[f, attachment.info[i_a].vertex, i_b].pos - anchor
            attachment.state[i_a, i_b].multiplier *= attachment.alpha * attachment.gamma
            attachment.state[i_a, i_b].stiffness = qd.max(
                solver._k_start, attachment.gamma * attachment.state[i_a, i_b].stiffness
            )
        for i_d in range(attachment.rigid.n_dofs):
            i_q = attachment.coordinate_info[i_d].qpos_idx
            position = rigid_info.qpos[i_q, i_b]
            velocity = dyn_state.dofs.vel[i_d, i_b]
            predicted = position + solver._substep_dt * (velocity + solver._substep_dt * dyn_state.dofs.acc[i_d, i_b])
            attachment.coordinate_state[i_d, i_b].previous = position
            attachment.coordinate_state[i_d, i_b].velocity = velocity
            attachment.coordinate_state[i_d, i_b].predicted = predicted
            if qd.static(rigid_config.enable_joint_limit):
                I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                predicted = qd.math.clamp(predicted, dyn_info.dofs.limit[I_d][0], dyn_info.dofs.limit[I_d][1])
            rigid_info.qpos[i_q, i_b] = predicted
        func_forward_kinematics_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
        for i_l in range(attachment.rigid.n_links):
            attachment.articulation_pose[i_l, i_b].pos = dyn_state.links.pos[i_l, i_b]
            attachment.articulation_pose[i_l, i_b].quat = dyn_state.links.quat[i_l, i_b]


@qd.func
def func_solve_articulation_batch(
    f,
    i_b,
    solver: qd.template(),
    attachment: qd.template(),
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
):
    for i_d in range(attachment.rigid.n_dofs):
        i_q = attachment.coordinate_info[i_d].qpos_idx
        i_j = attachment.coordinate_info[i_d].joint_idx
        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
        damping_h = dyn_info.dofs.damping[I_d] / solver._substep_dt
        position = rigid_info.qpos[i_q, i_b]
        force = -damping_h * (
            position
            - attachment.coordinate_state[i_d, i_b].previous
            - solver._substep_dt * attachment.coordinate_state[i_d, i_b].velocity
        )
        curvature = rigid_info.mass_mat[i_d, i_d, i_b] / solver._substep_dt**2 + damping_h
        for j_d in range(attachment.rigid.n_dofs):
            j_q = attachment.coordinate_info[j_d].qpos_idx
            force -= (
                rigid_info.mass_mat[i_d, j_d, i_b]
                / solver._substep_dt**2
                * (rigid_info.qpos[j_q, i_b] - attachment.coordinate_state[j_d, i_b].predicted)
            )
        axis = dyn_state.joints.xaxis[i_j, i_b]
        pivot = dyn_state.joints.xanchor[i_j, i_b]
        for i_a in range(attachment.n_attachments):
            if attachment.affects[i_d, i_a]:
                i_l = attachment.info[i_a].link
                anchor = gu.qd_transform_by_trans_quat(
                    attachment.info[i_a].local_pos,
                    dyn_state.links.pos[i_l, i_b],
                    dyn_state.links.quat[i_l, i_b],
                )
                error = (
                    solver.verts[f + 1, attachment.info[i_a].vertex, i_b].pos
                    - anchor
                    - attachment.alpha * attachment.previous_error[i_a, i_b]
                )
                stiffness = attachment.state[i_a, i_b].stiffness
                force_scale = attachment.state[i_a, i_b].multiplier + stiffness * error
                jacobian = axis.cross(anchor - pivot)
                force += jacobian.dot(force_scale)
                curvature += stiffness * jacobian.norm_sqr() + qd.abs(force_scale.dot(axis.cross(jacobian)))
        position += force / curvature
        if qd.static(rigid_config.enable_joint_limit):
            limits = dyn_info.dofs.limit[I_d]
            position = qd.math.clamp(position, limits[0], limits[1])
        rigid_info.qpos[i_q, i_b] = position
        func_forward_kinematics_batch(
            i_b,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            is_backward=False,
        )


@qd.kernel
def kernel_end_articulation(dt: float, attachment: qd.template(), dyn_state: DynState, rigid_info: RigidInfo):
    for i_d, i_b in qd.ndrange(attachment.rigid.n_dofs, dyn_state.dofs.vel.shape[1]):
        i_q = attachment.coordinate_info[i_d].qpos_idx
        position = rigid_info.qpos[i_q, i_b]
        rigid_info.qpos_next[i_q, i_b] = position
        dyn_state.dofs.vel_next[i_d, i_b] = (position - attachment.coordinate_state[i_d, i_b].previous) / dt
