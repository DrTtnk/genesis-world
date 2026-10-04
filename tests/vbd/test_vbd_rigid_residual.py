"""Final-iterate rigid residuals detect an unresolved coupled load, without advancing it."""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_numpy


@pytest.mark.precision("64")
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("iterations", [1, 40])
def test_rigid_residual_matches_coupled_spring_balance(iterations, n_envs):
    dt = 1e-3
    stiffness = 1e3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=iterations, floor_height=-float("inf"), record_rigid_residual=True,
        ),
        show_viewer=False,
    )
    post = scene.add_entity(
        morph=gs.morphs.Box(size=(0.01,) * 3, pos=(0., 0., 0.3), fixed=True),
        material=gs.materials.Rigid(rho=1500.),
    )
    bodies = [scene.add_entity(
        morph=gs.morphs.Box(size=(0.01,) * 3, pos=(0., 0., z)),
        material=gs.materials.Rigid(rho=1500.),
    ) for z in (0.2, 0.1)]
    solver = scene.vbd_solver
    for body in bodies:
        solver.add_rigid_link(body.links[0])
    for a, b, z in ((post, bodies[0], 0.2), (bodies[0], bodies[1], 0.1)):
        solver.add_rigid_joint(a.links[0], b.links[0], (0., 0., z), np.eye(3),
                               (stiffness,) * 3, (0.,) * 3)
    scene.build(n_envs=n_envs)
    with pytest.raises(gs.GenesisException, match="completed solve"):
        solver.rigid_solve_diagnostics()
    scene.step()
    diagnostic = solver.rigid_solve_diagnostics()
    actual = diagnostic.wrench.cpu().numpy()
    correction = diagnostic.correction.cpu().numpy()
    state = solver.rigid_attachment.link_state
    pos = qd_to_numpy(state.pos)[:2, :, 2].T
    predicted = qd_to_numpy(state.predicted_pos)[:2, :, 2].T
    mass = qd_to_numpy(state.mass)[:2].T
    displacement = pos - (0.2, 0.1)
    expected = -mass / dt**2 * (pos - predicted)
    expected[:, 0] -= stiffness * (2 * displacement[:, 0] - displacement[:, 1])
    expected[:, 1] -= stiffness * (displacement[:, 1] - displacement[:, 0])
    np.testing.assert_allclose(actual[..., 2], expected, atol=1e-10, rtol=1e-7)
    np.testing.assert_allclose(actual[..., [0, 1, 3, 4, 5]], 0., atol=1e-12)
    np.testing.assert_allclose(correction[..., 2], expected / (mass / dt**2 + (2 * stiffness, stiffness)),
                               atol=1e-13, rtol=1e-7)
    if iterations == 1:
        assert np.max(np.abs(actual[..., 2])) > 1e-4
    else:
        assert np.max(np.abs(actual[..., 2])) < 1e-9
    np.testing.assert_array_equal(diagnostic.substep.cpu().numpy(), 0)
    before = {name: qd_to_numpy(getattr(state, name)).copy() for name in ("pos", "quat", "predicted_pos")}
    again = solver.rigid_solve_diagnostics()
    np.testing.assert_array_equal(again.wrench.cpu().numpy(), actual)
    for name, value in before.items():
        np.testing.assert_array_equal(qd_to_numpy(getattr(state, name)), value)
    scene.reset()
    with pytest.raises(gs.GenesisException, match="completed solve"):
        solver.rigid_solve_diagnostics()
    scene.step()
    np.testing.assert_allclose(solver.rigid_solve_diagnostics().wrench.cpu().numpy(), actual, atol=1e-12)
