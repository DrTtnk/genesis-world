"""Physical gates for the passive VIPER rod and its energy derivatives."""

import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from genesis.engine.solvers.vbd_rod import RodAttachment, RodModel, RodParameters, quat_multiply, retract

pytestmark = pytest.mark.parametrize("backend", [None], indirect=True)


def model(bent=False, surface_bend=0.0):
    x = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.04], [0.01 if bent else 0.0, 0.0, 0.08]], dtype=torch.float64)
    q = []
    for d in np.diff(x.numpy(), axis=0):
        w = d / np.linalg.norm(d)
        u = np.cross([0.0, 1.0, 0.0], w)
        u /= np.linalg.norm(u)
        q.append(Rotation.from_matrix(np.column_stack((u, np.cross(w, u), w))).as_quat(scalar_first=True))
    return RodModel(x, torch.tensor(np.array(q)), 0.003, RodParameters(1000.0, 1e3, 1e3, 1e4, 1e6, surface_bend))


@pytest.mark.parametrize("bent", [False, True, "captured", "scale"])
def test_rest_and_rigid_motion(bent):
    if bent == "captured":
        positions = torch.tensor([
            [-2.2705640788190067, 0.056094900239259005, 0.0894109234213829],
            [-2.263560668565333, 0.06784641370177269, 0.062062395620159805],
            [-2.28997153788805, 0.06875496730208397, 0.0425217691808939],
        ], dtype=torch.float64)
        frames = torch.tensor([
            [0.22983634585515067, -0.8360243057751355, 0.4982355008197974, 0.0],
            [0.4502507644304328, -0.030698340814116803, -0.8923742830231446, 0.0],
        ], dtype=positions.dtype)
        rod = RodModel(positions, frames, 0.0005748129842641093,
                       RodParameters(1050.0, 100000.0, 100000.0, 100000.0, 83333.33333333331, 0.0))
    elif bent == "scale":
        positions = torch.tensor(
            [
                [-2.361664533644216, -0.036483779549598694, 0.10579438507556915],
                [-2.362505840952508, -0.04434752278029919, 0.09610581211745739],
                [-2.363347148289904, -0.052211266942322254, 0.08641723915934563],
                [-2.3650297629646957, -0.06793875433504581, 0.06704009184613824],
                [-2.367927646264434, -0.06617214251309633, 0.04785223305225372],
                [-2.3708255290985107, -0.06440553022548556, 0.028664372861385345],
            ],
            dtype=torch.float64,
        )
        frames = torch.tensor(
            [
                [0.33565037298318634, 0.9366415282994301, -0.10020715833290499, 0.0],
                [0.33565040008980346, 0.9366415195841994, -0.10020714899924427, 0.0],
                [0.3356503699475606, 0.9366415296359375, -0.10020715600852548, 0.0],
                [0.08742197119528188, -0.5185307218699681, -0.8505782088846041, 0.0],
                [0.08742196106873935, -0.5185308827263703, -0.8505781118638704, 0.0],
            ],
            dtype=positions.dtype,
        )
        rod = RodModel(positions, frames, 0.000887306395896071,
                       RodParameters(1050.0, 100000.0, 100000.0, 100000.0, 83333.33333333331, 0.0))
    else:
        rod = model(bent)
    rod.begin(rod.rest_pos, torch.zeros_like(rod.rest_pos), 1 / 16000, rod.rest_pos.new_zeros(3))
    solved = rod.sweep(rod.rest_pos, torch.zeros(len(rod.rest_pos), dtype=torch.bool))
    torch.testing.assert_close(solved, rod.rest_pos, atol=1e-15, rtol=0)
    torch.testing.assert_close(rod.quat, rod.rest_quat, atol=1e-15, rtol=0)
    s = torch.ones(len(rod.rest_pos), dtype=torch.float64)
    torch.testing.assert_close(
        rod.elastic_residual(rod.rest_pos, s, rod.rest_quat),
        torch.zeros_like(rod.elastic_residual(rod.rest_pos, s, rod.rest_quat)),
        atol=1e-13,
        rtol=0,
    )
    rotation = torch.tensor(Rotation.from_rotvec([0.8, -0.4, 0.6]).as_matrix())
    quat = torch.tensor(Rotation.from_matrix(rotation.numpy()).as_quat(scalar_first=True))
    q = quat_multiply(quat.expand_as(rod.rest_quat), rod.rest_quat)
    x = rod.rest_pos @ rotation.T + torch.tensor([0.3, -0.7, 0.2])
    assert rod.energy(x, s, q) < 1e-24
    assert abs(rod.mass.sum().item() - rod.parameters.density * np.pi * rod.radius**2 * rod.length.sum().item()) < 1e-16


def test_stretch_volume_response():
    rod = model()
    x = rod.rest_pos * 1.25
    same = rod.elastic_residual(x, torch.ones(3, dtype=torch.float64), rod.rest_quat)
    shrink = rod.elastic_residual(x, torch.ones(3, dtype=torch.float64) / np.sqrt(1.25), rod.rest_quat)
    assert shrink.square().sum() < same.square().sum() / 20
    torch.testing.assert_close(rod.volume(x, torch.ones(3, dtype=torch.float64) / np.sqrt(1.25)), rod.rest_volume)


@pytest.mark.parametrize("seed", [1, 19, 84])
def test_residual_derivatives(seed):
    torch.manual_seed(seed)
    rod = model(bent=True, surface_bend=2.0)
    x = rod.rest_pos + torch.randn_like(rod.rest_pos) * 0.002
    s = 1 + torch.randn(3, dtype=x.dtype) * 0.1
    q = retract(rod.rest_quat, torch.randn((2, 3), dtype=x.dtype) * 0.2)
    if seed % 2:
        q[0] *= -1
    z = torch.cat((x.reshape(-1), s))

    def residual(z):
        return rod.elastic_residual(z[:9].reshape(3, 3), z[9:], q)

    assert torch.autograd.gradcheck(residual, (z.requires_grad_(),), eps=1e-6, atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradgradcheck(lambda z: residual(z).square().sum() / 2, (z,), eps=1e-6, atol=1e-5, rtol=1e-4)
    rotation = torch.zeros((2, 3), dtype=x.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: rod.elastic_residual(x, s, retract(q, t)), (rotation,), eps=1e-6, atol=1e-6, rtol=1e-5
    )


def test_twist_bend_and_sign():
    rod = model()
    s = torch.ones(3, dtype=torch.float64)
    q = retract(rod.rest_quat, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]], dtype=s.dtype))
    assert rod.energy(rod.rest_pos, s, q) > 0
    torch.testing.assert_close(rod.energy(rod.rest_pos, s, q), rod.energy(rod.rest_pos, s, -q))
    original = rod.energy(rod.rest_pos, s, q)
    q[0] *= -1
    torch.testing.assert_close(rod.energy(rod.rest_pos, s, q), original)


def test_one_sweep_decreases_incremental_energy():
    rod = model()
    x = rod.rest_pos.clone()
    x[-1, 0] += 0.005
    rod.begin(x, torch.zeros_like(x), 0.002, torch.zeros(3, dtype=x.dtype))
    before = rod.incremental_energy(x)
    after_x = rod.sweep(x, torch.zeros(3, dtype=torch.bool))
    assert rod.incremental_energy(after_x) < before
    assert (rod.scale > 0).all()
    torch.testing.assert_close(torch.linalg.vector_norm(rod.quat, dim=-1), torch.ones(2, dtype=x.dtype))


def test_loaded_force_torque_balance_and_rotation_invariance():
    rod = model(bent=True)
    x = rod.rest_pos + torch.tensor([[0.002, 0.001, 0.0], [-0.001, 0.0, 0.002], [0.0, 0.001, 0.0]], dtype=torch.float64)
    s = torch.tensor([0.9, 1.1, 1.05], dtype=x.dtype)
    q = retract(rod.rest_quat, torch.tensor([[0.1, 0.0, 0.2], [0.0, 0.2, -0.1]], dtype=x.dtype))
    delta = torch.zeros_like(x, requires_grad=True)
    theta = torch.zeros((2, 3), dtype=x.dtype, requires_grad=True)
    energy = rod.energy(x + delta, s, retract(q, theta))
    gradient, torque = torch.autograd.grad(energy, (delta, theta))
    torch.testing.assert_close(gradient.sum(0), torch.zeros(3, dtype=x.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.linalg.cross(x, gradient).sum(0) + torque.sum(0), torch.zeros(3, dtype=x.dtype), atol=1e-12, rtol=0
    )


def test_uniform_extension_equilibrium_against_scalar_reference():
    rod = model()
    x = rod.rest_pos * 1.25
    pinned = torch.ones(3, dtype=torch.bool)
    for _ in range(5):
        rod.begin(x, torch.zeros_like(x), 0.02, torch.zeros(3, dtype=x.dtype))
        for _ in range(12):
            x = rod.sweep(x, pinned)
        rod.end(x)
    p = rod.parameters
    exact = brentq(
        lambda s: (p.stretch_x + p.stretch_y) * (s - 1) + 2 * p.volume * s * 1.25 * (1.25 * s * s - 1), 0.8, 1.0
    )
    torch.testing.assert_close(rod.scale, torch.full_like(rod.scale, exact), atol=3e-5, rtol=0)
    assert abs(rod.volume(x, rod.scale) / rod.rest_volume - 1) < 2e-4


def test_invalid_geometry_and_snapshot_fail():
    rod = model()
    with pytest.raises(ValueError, match="positive rest length"):
        RodModel(torch.zeros_like(rod.rest_pos), rod.rest_quat, 0.003, rod.parameters)
    state = rod.get_state()
    state.scale[1] = 0
    with pytest.raises(ValueError, match="positive scales"):
        rod.set_state(state)


def _straight_rod(dtype, offset, segment, tilt):
    """Eight segments along an oblique direction at `offset` from the origin (so rounding falls across the
    segments, not only along them), frames built in float64 then cast, each frame's third axis turned by `tilt`
    radians about an axis normal to the rod."""
    w = np.array([1.0, 0.7, -0.4]) / np.linalg.norm([1.0, 0.7, -0.4])
    x = offset + segment * np.arange(9)[:, None] * w
    u = np.cross([0.0, 1.0, 0.0], w)
    u /= np.linalg.norm(u)
    frame = Rotation.from_rotvec(tilt * u) * Rotation.from_matrix(np.column_stack((u, np.cross(w, u), w)))
    q = np.tile(frame.as_quat(scalar_first=True), (8, 1))
    return RodModel(torch.tensor(x, dtype=dtype), torch.tensor(q, dtype=dtype), 1e-4,
                    RodParameters(1000.0, 1e3, 1e3, 1e4, 1e6, 0.0))


def test_the_rest_frame_check_holds_each_precision_to_what_it_can_represent():
    """Half-millimetre segments 0.1 m from the origin, as in the snake head. In float32 their directions carry
    about 1e-5 of rounding, so the check allows what the precision can represent (8 eps |x| / L); in float64
    that bound is far below 1e-6, and 1e-6 stays the tolerance."""
    offset = np.array([0.1, -0.08, 0.12])
    _straight_rod(torch.float32, offset, 5e-4, 0.0)
    with pytest.raises(ValueError, match="align with its centreline"):
        _straight_rod(torch.float32, offset, 5e-4, 1e-3)
    _straight_rod(torch.float64, offset, 5e-4, 0.0)
    with pytest.raises(ValueError, match="align with its centreline"):
        _straight_rod(torch.float64, offset, 5e-4, 1e-5)


def test_random_rotated_rest_states_do_not_request_unresolvable_descent():
    rng = np.random.default_rng(392)
    for _ in range(8):
        base = model()
        R = Rotation.random(random_state=rng)
        q = torch.tensor(R.as_quat(scalar_first=True)).expand_as(base.rest_quat)
        x = base.rest_pos @ torch.tensor(R.as_matrix()).T + torch.tensor(rng.normal(size=3))
        rod = RodModel(x, q, 0.003, base.parameters)
        rod.begin(x, torch.zeros_like(x), 0.01, torch.zeros(3, dtype=x.dtype))
        result = rod.sweep(x, torch.zeros(3, dtype=torch.bool))
        torch.testing.assert_close(result, x, atol=1e-13, rtol=0)


def test_attachment_residual_matches_weighted_native_force():
    torch.manual_seed(319)
    x = torch.randn((5, 3), dtype=torch.float64)
    vertices = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    weights = torch.rand((2, 4), dtype=x.dtype)
    weights /= weights.sum(1, keepdim=True)
    target = torch.randn((2, 3), dtype=x.dtype)
    stiffness = torch.tensor([31.0, 107.0], dtype=x.dtype)
    attachment = RodAttachment(vertices, weights, target, stiffness)
    gradient = torch.autograd.functional.jacobian(lambda x: attachment.residual(x).square().sum() / 2, x)
    force_scale = stiffness[:, None] * ((weights[:, :, None] * x[vertices]).sum(1) - target)
    expected = torch.zeros_like(x)
    expected.index_add_(0, vertices.reshape(-1), (weights[:, :, None] * force_scale[:, None, :]).reshape(-1, 3))
    torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=0)
