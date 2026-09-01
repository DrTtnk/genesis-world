import numpy as np
import pytest
import torch

import genesis as gs


def _run(scene, body, gain, n_steps, n_groups):
    """Roll out a constant actuation and return how far the body's centre of mass travels in x."""
    scene.reset()
    pos_init = body.get_state().pos[0].mean(dim=0)[0]
    for _ in range(n_steps):
        body.set_actuation(gain * torch.ones(n_groups, dtype=gs.tc_float, device=gs.device))
        scene.step()
    return body.get_state().pos[0].mean(dim=0)[0] - pos_init


@pytest.mark.precision("64")
@pytest.mark.required
def test_mpm_muscle_actuation_grad(show_viewer):
    """The gradient of motion with respect to muscle actuation must match finite differences.

    The rollout deliberately spans several steps. `substeps_local` defaults to `substeps` under
    `requires_grad`, so a single-step rollout would sit inside one checkpoint segment and would
    not exercise the gradient reset that runs at segment boundaries.
    """
    N_STEPS = 4
    N_GROUPS = 2
    GAIN = 0.4
    EPS = 1e-3

    def build():
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=2e-3, substeps=5, requires_grad=True),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.3, -0.15, -0.05),
                upper_bound=(0.3, 0.15, 0.15),
                grid_density=64,
            ),
            show_viewer=show_viewer,
        )
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=1.0))
        body = scene.add_entity(
            material=gs.materials.MPM.Muscle(E=5e4, nu=0.45, rho=1000.0, n_groups=N_GROUPS),
            morph=gs.morphs.Box(size=(0.2, 0.04, 0.04), pos=(0.0, 0.0, 0.021)),
        )
        scene.build(n_envs=1)

        # Split the body along x, one muscle group per half, fibres along the body axis.
        particles = body._particles
        x = particles[:, 0]
        group = (x > x.mean()).astype(np.int64)
        direction = np.zeros((len(particles), 3))
        direction[:, 0] = 1.0
        body.set_muscle(muscle_group=group, muscle_direction=direction)
        return scene, body

    scene, body = build()

    gain = torch.tensor(GAIN, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    displacement = _run(scene, body, gain, N_STEPS, N_GROUPS)
    scene.backward(displacement)
    analytic = gain.grad.item()

    with torch.no_grad():
        plus = _run(scene, body, torch.tensor(GAIN + EPS, dtype=gs.tc_float, device=gs.device), N_STEPS, N_GROUPS)
        minus = _run(scene, body, torch.tensor(GAIN - EPS, dtype=gs.tc_float, device=gs.device), N_STEPS, N_GROUPS)
    numeric = (plus.item() - minus.item()) / (2 * EPS)

    assert abs(analytic) > 1e-9, "actuation gradient is zero, so it is not reaching the input"
    assert analytic == pytest.approx(numeric, rel=0.05, abs=1e-6), (
        f"analytic gradient {analytic:.6e} disagrees with finite differences {numeric:.6e}"
    )
