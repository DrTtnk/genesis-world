"""Pure rigid-body blocks for two-way vertex block descent attachments.

The helpers return negative energy gradients as forces and positive curvature
blocks. The caller owns state, adds translational mass and world inertia, and
chooses the alternating primal sweep order.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu


@qd.func
def func_skew(vector: qd.types.vector(3)):
    return qd.Matrix(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]],
        dt=gs.qd_float,
    )


@qd.func
def func_quaternion_update(quaternion: qd.types.vector(4), rotation_increment: qd.types.vector(3)):
    """Apply normalize(q + 0.5 * (0, rotation_increment) * q) in the world frame."""
    pure_increment = gs.qd_vec4(0.0, rotation_increment[0], rotation_increment[1], rotation_increment[2])
    return (quaternion + 0.5 * gu.qd_quat_mul(pure_increment, quaternion)).normalized()


@qd.func
def func_quaternion_difference(quaternion: qd.types.vector(4), predicted_quaternion: qd.types.vector(4)):
    """Return the AVBD world-frame local difference 2 * vector(q * inverse(q_pred))."""
    difference = gu.qd_quat_mul(quaternion, gu.qd_inv_quat(predicted_quaternion))
    return gs.qd_vec3(2.0 * difference[1], 2.0 * difference[2], 2.0 * difference[3])


@qd.func
def _func_attachment_values(
    x: qd.types.vector(3),
    p: qd.types.vector(3),
    quaternion: qd.types.vector(4),
    local_anchor: qd.types.vector(3),
    multiplier: qd.types.vector(3),
    stiffness,
    previous_error: qd.types.vector(3),
    alpha,
):
    rotated_anchor = gu.qd_transform_by_quat_fast(local_anchor, quaternion)
    constraint = x - p - rotated_anchor - alpha * previous_error
    force_scale = multiplier + stiffness * constraint
    return rotated_anchor, force_scale


@qd.func
def func_attachment_soft_system(
    x: qd.types.vector(3),
    p: qd.types.vector(3),
    quaternion: qd.types.vector(4),
    local_anchor: qd.types.vector(3),
    multiplier: qd.types.vector(3),
    stiffness,
    previous_error: qd.types.vector(3),
    alpha,
):
    """Return only the soft force and Hessian, without constructing the rigid 6x6 block."""
    _, force_scale = _func_attachment_values(
        x, p, quaternion, local_anchor, multiplier, stiffness, previous_error, alpha
    )
    return -force_scale, stiffness * qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def func_attachment_blocks(
    x: qd.types.vector(3),
    p: qd.types.vector(3),
    quaternion: qd.types.vector(4),
    local_anchor: qd.types.vector(3),
    multiplier: qd.types.vector(3),
    stiffness,
    previous_error: qd.types.vector(3),
    alpha,
):
    """Return attachment forces and local positive-semidefinite curvature blocks.

    C = x - p - R local_anchor - alpha previous_error and y = multiplier + stiffness C.
    The rigid Jacobian is [-I, skew(r)]. Its Gauss-Newton block is augmented by
    diag(norm(K[:, j])), where K is the exact rotational curvature. K itself is
    generally indefinite and is not returned as the solve curvature.
    """
    rotated_anchor, force_scale = _func_attachment_values(
        x, p, quaternion, local_anchor, multiplier, stiffness, previous_error, alpha
    )

    soft_force = -force_scale
    soft_H = stiffness * qd.Matrix.identity(gs.qd_float, 3)

    rigid_force6 = qd.Vector.zero(gs.qd_float, 6)
    rigid_force6[:3] = force_scale
    rigid_force6[3:6] = rotated_anchor.cross(force_scale)

    rigid_jacobian = qd.Matrix.zero(gs.qd_float, 3, 6)
    skew_anchor = func_skew(rotated_anchor)
    for row in qd.static(range(3)):
        rigid_jacobian[row, row] = -1.0
        for column in qd.static(range(3)):
            rigid_jacobian[row, column + 3] = skew_anchor[row, column]
    rigid_H6 = stiffness * rigid_jacobian.transpose() @ rigid_jacobian

    identity = qd.Matrix.identity(gs.qd_float, 3)
    exact_curvature = force_scale.dot(rotated_anchor) * identity - 0.5 * (
        force_scale.outer_product(rotated_anchor) + rotated_anchor.outer_product(force_scale)
    )
    for column in qd.static(range(3)):
        column_norm_squared = gs.qd_float(0.0)
        for row in qd.static(range(3)):
            column_norm_squared += exact_curvature[row, column] * exact_curvature[row, column]
        rigid_H6[column + 3, column + 3] += qd.sqrt(column_norm_squared)

    return soft_force, soft_H, rigid_force6, rigid_H6


@qd.func
def func_ldlt6_solve(matrix: qd.types.matrix(6, 6), right_hand_side: qd.types.vector(6)):
    """Solve a symmetric positive-definite 6x6 system with unpivoted LDL^T."""
    lower = qd.Matrix.identity(gs.qd_float, 6)
    diagonal = qd.Vector.zero(gs.qd_float, 6)

    # Runtime loops keep this fixed-size solve compact in generated code.
    for column in range(6):
        pivot = matrix[column, column]
        for previous in range(column):
            pivot -= lower[column, previous] * lower[column, previous] * diagonal[previous]
        diagonal[column] = pivot

        for row in range(column + 1, 6):
            value = matrix[row, column]
            for previous in range(column):
                value -= lower[row, previous] * lower[column, previous] * diagonal[previous]
            lower[row, column] = value / pivot

    forward = qd.Vector.zero(gs.qd_float, 6)
    for row in range(6):
        value = right_hand_side[row]
        for column in range(row):
            value -= lower[row, column] * forward[column]
        forward[row] = value

    solution = qd.Vector.zero(gs.qd_float, 6)
    for row_offset in range(6):
        row = 5 - row_offset
        value = forward[row] / diagonal[row]
        for column in range(row + 1, 6):
            value -= lower[column, row] * solution[column]
        solution[row] = value

    return solution
