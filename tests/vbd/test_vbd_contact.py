import pytest
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.vbd_contact import func_point_triangle_weights, func_segment_parameters
from genesis.utils.misc import qd_to_torch

from ..utils.assertions import assert_allclose


N_CASES = 32


@qd.kernel
def kernel_closest_features_reference(
    x: qd.template(),
    a: qd.template(),
    b: qd.template(),
    c: qd.template(),
    d: qd.template(),
    weights: qd.template(),
    parameters: qd.template(),
):
    for i in range(N_CASES):
        weights[i] = func_point_triangle_weights(x[i], a[i], b[i], c[i])
        s, t = func_segment_parameters(a[i], b[i], c[i], d[i])
        parameters[i] = qd.Vector([s, t], dt=gs.qd_float)


def _closest_point_on_triangle_brute(x, a, b, c):
    grid = torch.linspace(0.0, 1.0, 801, dtype=torch.float64, device="cpu")
    u, v = torch.meshgrid(grid, grid, indexing="ij")
    keep = (u + v) <= 1.0
    u, v = u[keep], v[keep]
    points = (1.0 - u - v)[:, None] * a + u[:, None] * b + v[:, None] * c
    return torch.linalg.vector_norm(points - x, dim=1).min()


def _closest_segments_brute(a, b, c, d):
    grid = torch.linspace(0.0, 1.0, 801, dtype=torch.float64, device="cpu")
    s, t = torch.meshgrid(grid, grid, indexing="ij")
    p = a + s.reshape(-1, 1) * (b - a)
    q = c + t.reshape(-1, 1) * (d - c)
    return torch.linalg.vector_norm(p - q, dim=1).min()


def test_closest_feature_kernels_match_brute_force():
    generator = torch.Generator(device="cpu").manual_seed(20260912)
    dtype = gs.tc_float
    tolerance = 3e-3  # the brute force samples the primitives on an 801-point grid
    points = torch.randn(5, N_CASES, 3, generator=generator, dtype=torch.float64, device="cpu")
    # every closest-feature region of the triangle and every clamp case of the segments
    points[0, :8] = points[1, :8] + 0.05 * torch.randn(8, 3, generator=generator, dtype=torch.float64, device="cpu")
    points[0, 8:16] = 0.5 * (points[1, 8:16] + points[2, 8:16]) + 3.0 * torch.randn(
        8, 3, generator=generator, dtype=torch.float64, device="cpu"
    )
    fields = [qd.Vector.field(3, dtype=gs.qd_float, shape=N_CASES) for _ in range(5)]
    for field, values in zip(fields, points):
        field.from_torch(values.to(dtype).to(gs.device))
    weights = qd.Vector.field(3, dtype=gs.qd_float, shape=N_CASES)
    parameters = qd.Vector.field(2, dtype=gs.qd_float, shape=N_CASES)
    kernel_closest_features_reference(*fields, weights, parameters)
    weights = qd_to_torch(weights).to(torch.float64).cpu()
    parameters = qd_to_torch(parameters).to(torch.float64).cpu()
    x, a, b, c, d = points
    assert_allclose(weights.sum(dim=1), 1.0, tol=1e-6)
    assert (weights >= -1e-7).all()
    assert (parameters >= 0.0).all() and (parameters <= 1.0).all()
    for i in range(N_CASES):
        q = weights[i, 0] * a[i] + weights[i, 1] * b[i] + weights[i, 2] * c[i]
        assert_allclose(
            torch.linalg.vector_norm(x[i] - q), _closest_point_on_triangle_brute(x[i], a[i], b[i], c[i]), tol=tolerance
        )
        p = a[i] + parameters[i, 0] * (b[i] - a[i])
        r = c[i] + parameters[i, 1] * (d[i] - c[i])
        assert_allclose(torch.linalg.vector_norm(p - r), _closest_segments_brute(a[i], b[i], c[i], d[i]), tol=tolerance)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_tissue_rests_on_fixed_rigid_box_within_the_contact_layer(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.0, 0.0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.0, 0.0205),
            nobisect=False,
            maxvolume=1e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.build(n_envs=n_envs)
    thickness = 1e-3
    for _ in range(80):
        scene.step()
    positions = tissue.get_positions()
    lowest = positions[..., 2].min()
    # the tissue sits on the table: inside the layer, never through the surface
    assert lowest > -1e-4
    assert lowest < thickness
    state = tissue.get_state()
    assert torch.linalg.vector_norm(state.vel, dim=-1).max() < 5e-3
    mass = tissue.material.rho * 0.04**3
    reactions = scene.vbd_solver.collider_reactions()
    assert reactions.shape == (max(n_envs, 1), 1, 6)
    assert_allclose(reactions[:, 0, 2], -mass * 9.81, rtol=0.02, atol=0.0)
    assert_allclose(reactions[:, 0, :2], 0.0, atol=0.02 * mass * 9.81)
    assert torch.isfinite(positions).all()
