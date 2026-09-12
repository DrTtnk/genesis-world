"""Kernel-level references for the pure VBD rigid attachment helpers."""

import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.vbd_rigid import (
    func_attachment_blocks,
    func_attachment_soft_system,
    func_ldlt6_solve,
    func_quaternion_difference,
    func_quaternion_update,
)
from genesis.utils.misc import qd_to_torch

from ..utils.assertions import assert_allclose


N_CASES = 24


@qd.kernel
def kernel_attachment_reference(
    x: qd.template(),
    p: qd.template(),
    quaternion: qd.template(),
    local_anchor: qd.template(),
    multiplier: qd.template(),
    stiffness: qd.template(),
    previous_error: qd.template(),
    alpha: qd.template(),
    soft_force: qd.template(),
    soft_H: qd.template(),
    soft_only_force: qd.template(),
    soft_only_H: qd.template(),
    rigid_force6: qd.template(),
    rigid_H6: qd.template(),
):
    for i in range(N_CASES):
        soft_force[i], soft_H[i], rigid_force6[i], rigid_H6[i] = func_attachment_blocks(
            x[i], p[i], quaternion[i], local_anchor[i], multiplier[i], stiffness[i], previous_error[i], alpha[i]
        )
        soft_only_force[i], soft_only_H[i] = func_attachment_soft_system(
            x[i], p[i], quaternion[i], local_anchor[i], multiplier[i], stiffness[i], previous_error[i], alpha[i]
        )


@qd.kernel
def kernel_quaternion_reference(
    quaternion: qd.template(),
    predicted_quaternion: qd.template(),
    rotation_increment: qd.template(),
    updated_quaternion: qd.template(),
    quaternion_difference: qd.template(),
):
    for i in range(N_CASES):
        updated_quaternion[i] = func_quaternion_update(quaternion[i], rotation_increment[i])
        quaternion_difference[i] = func_quaternion_difference(quaternion[i], predicted_quaternion[i])


@qd.kernel
def kernel_ldlt6_reference(
    system_matrix: qd.template(),
    right_hand_side: qd.template(),
    solution: qd.template(),
):
    for i in range(N_CASES):
        solution[i] = func_ldlt6_solve(system_matrix[i], right_hand_side[i])


def _vector_field(width, value=None):
    field = qd.Vector.field(width, dtype=gs.qd_float, shape=(N_CASES,))
    if value is not None:
        field.from_torch(value)
    return field


def _matrix_field(rows, columns, value=None):
    field = qd.Matrix.field(rows, columns, dtype=gs.qd_float, shape=(N_CASES,))
    if value is not None:
        field.from_torch(value)
    return field


def _scalar_field(value):
    field = qd.field(dtype=gs.qd_float, shape=(N_CASES,))
    field.from_torch(value)
    return field


def _field_tensor(field, component_shape):
    tensor = qd_to_torch(field, transpose=True)
    assert tensor.shape == (N_CASES, *component_shape)
    assert tensor.device == gs.device
    return tensor


def _quaternion_product(left, right):
    left_scalar, left_vector = left[:, :1], left[:, 1:]
    right_scalar, right_vector = right[:, :1], right[:, 1:]
    return torch.cat(
        (
            left_scalar * right_scalar - (left_vector * right_vector).sum(dim=1, keepdim=True),
            left_scalar * right_vector + right_scalar * left_vector + torch.linalg.cross(left_vector, right_vector),
        ),
        dim=1,
    )


def _rotate_by_quaternion(vector, quaternion):
    quaternion_vector = quaternion[:, 1:]
    return vector + 2.0 * torch.linalg.cross(
        quaternion_vector,
        torch.linalg.cross(quaternion_vector, vector) + quaternion[:, :1] * vector,
    )


def _skew(vector):
    zero = torch.zeros(N_CASES, dtype=vector.dtype, device=vector.device)
    x, y, z = vector.unbind(dim=1)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=1),
            torch.stack((z, zero, -x), dim=1),
            torch.stack((-y, x, zero), dim=1),
        ),
        dim=1,
    )


def test_rigid_attachment_helpers_match_torch_reference():
    generator = torch.Generator(device="cpu").manual_seed(20260912)
    dtype = gs.tc_float
    tolerance = 2e-11 if dtype == torch.float64 else 2e-5

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype, device="cpu").to(gs.device)

    p = random_tensor(N_CASES, 3)
    quaternion = random_tensor(N_CASES, 4)
    quaternion /= torch.linalg.vector_norm(quaternion, dim=1, keepdim=True)
    local_anchor = random_tensor(N_CASES, 3)
    previous_error = random_tensor(N_CASES, 3)
    multiplier = random_tensor(N_CASES, 3)
    residual = random_tensor(N_CASES, 3)
    residual[:12] = 0.0
    multiplier[:6] = 0.0
    multiplier[12:18] = 0.0
    stiffness = torch.exp(random_tensor(N_CASES))
    alpha = torch.rand(N_CASES, generator=generator, dtype=dtype, device="cpu").to(gs.device)
    rotated_anchor = _rotate_by_quaternion(local_anchor, quaternion)
    x = p + rotated_anchor + alpha[:, None] * previous_error + residual

    x_field = _vector_field(3, x)
    p_field = _vector_field(3, p)
    quaternion_field = _vector_field(4, quaternion)
    local_anchor_field = _vector_field(3, local_anchor)
    multiplier_field = _vector_field(3, multiplier)
    stiffness_field = _scalar_field(stiffness)
    previous_error_field = _vector_field(3, previous_error)
    alpha_field = _scalar_field(alpha)
    soft_force_field, soft_only_force_field = _vector_field(3), _vector_field(3)
    soft_H_field, soft_only_H_field = _matrix_field(3, 3), _matrix_field(3, 3)
    rigid_force6_field, rigid_H6_field = _vector_field(6), _matrix_field(6, 6)

    kernel_attachment_reference(
        x_field,
        p_field,
        quaternion_field,
        local_anchor_field,
        multiplier_field,
        stiffness_field,
        previous_error_field,
        alpha_field,
        soft_force_field,
        soft_H_field,
        soft_only_force_field,
        soft_only_H_field,
        rigid_force6_field,
        rigid_H6_field,
    )

    constraint = x - p - rotated_anchor - alpha[:, None] * previous_error
    force_scale = multiplier + stiffness[:, None] * constraint
    identity3 = torch.eye(3, dtype=dtype, device=gs.device).expand(N_CASES, -1, -1)
    expected_soft_force = -force_scale
    expected_soft_H = stiffness[:, None, None] * identity3
    expected_rigid_force6 = torch.cat((force_scale, torch.linalg.cross(rotated_anchor, force_scale)), dim=1)

    rigid_jacobian = torch.cat((-identity3, _skew(rotated_anchor)), dim=2)
    expected_rigid_H6 = stiffness[:, None, None] * rigid_jacobian.transpose(1, 2) @ rigid_jacobian
    exact_curvature = (force_scale * rotated_anchor).sum(dim=1)[:, None, None] * identity3 - 0.5 * (
        force_scale[:, :, None] * rotated_anchor[:, None, :] + rotated_anchor[:, :, None] * force_scale[:, None, :]
    )
    expected_rigid_H6[:, 3:, 3:] += torch.diag_embed(torch.linalg.vector_norm(exact_curvature, dim=1))

    soft_force = _field_tensor(soft_force_field, (3,))
    soft_H = _field_tensor(soft_H_field, (3, 3))
    soft_only_force = _field_tensor(soft_only_force_field, (3,))
    soft_only_H = _field_tensor(soft_only_H_field, (3, 3))
    rigid_force6 = _field_tensor(rigid_force6_field, (6,))
    rigid_H6 = _field_tensor(rigid_H6_field, (6, 6))

    assert_allclose(soft_force, expected_soft_force, tol=tolerance)
    assert_allclose(soft_H, expected_soft_H, tol=tolerance)
    assert_allclose(soft_only_force, expected_soft_force, tol=tolerance)
    assert_allclose(soft_only_H, expected_soft_H, tol=tolerance)
    assert_allclose(rigid_force6, expected_rigid_force6, tol=tolerance)
    assert_allclose(rigid_H6, expected_rigid_H6, tol=tolerance)
    assert_allclose(soft_force, soft_only_force, tol=tolerance)
    assert_allclose(soft_H, soft_only_H, tol=tolerance)
    assert_allclose(rigid_H6, rigid_H6.transpose(1, 2), tol=tolerance)
    assert torch.linalg.eigvalsh(soft_H).min() >= -tolerance
    assert torch.linalg.eigvalsh(rigid_H6).min() >= -tolerance

    predicted_quaternion = random_tensor(N_CASES, 4)
    predicted_quaternion /= torch.linalg.vector_norm(predicted_quaternion, dim=1, keepdim=True)
    rotation_increment = 0.2 * random_tensor(N_CASES, 3)
    updated_quaternion_field, quaternion_difference_field = _vector_field(4), _vector_field(3)
    kernel_quaternion_reference(
        quaternion_field,
        _vector_field(4, predicted_quaternion),
        _vector_field(3, rotation_increment),
        updated_quaternion_field,
        quaternion_difference_field,
    )

    pure_increment = torch.cat((torch.zeros_like(rotation_increment[:, :1]), rotation_increment), dim=1)
    expected_updated_quaternion = quaternion + 0.5 * _quaternion_product(pure_increment, quaternion)
    expected_updated_quaternion /= torch.linalg.vector_norm(expected_updated_quaternion, dim=1, keepdim=True)
    predicted_inverse = predicted_quaternion.clone()
    predicted_inverse[:, 1:] *= -1.0
    expected_quaternion_difference = 2.0 * _quaternion_product(quaternion, predicted_inverse)[:, 1:]
    assert_allclose(_field_tensor(updated_quaternion_field, (4,)), expected_updated_quaternion, tol=tolerance)
    assert_allclose(_field_tensor(quaternion_difference_field, (3,)), expected_quaternion_difference, tol=tolerance)

    mass_factor = random_tensor(N_CASES, 6, 6)
    mass_block = mass_factor @ mass_factor.transpose(1, 2) + 0.5 * torch.eye(6, dtype=dtype, device=gs.device)
    system_matrix = mass_block + expected_rigid_H6
    right_hand_side = random_tensor(N_CASES, 6)
    solution_field = _vector_field(6)
    kernel_ldlt6_reference(
        _matrix_field(6, 6, system_matrix),
        _vector_field(6, right_hand_side),
        solution_field,
    )
    expected_solution = torch.linalg.solve(system_matrix, right_hand_side[:, :, None]).squeeze(2)
    assert_allclose(_field_tensor(solution_field, (6,)), expected_solution, tol=tolerance)
