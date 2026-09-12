"""Muscle commands and their gradients belong to one entity, not the whole solver."""

import numpy as np
import pytest
import torch

import genesis as gs
from tests.utils.assertions import assert_allclose


def build_muscles(show_viewer, n_envs=0, requires_grad=False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.004, gravity=(0.0, 0.0, 0.0), requires_grad=requires_grad),
        vbd_options=gs.options.VBDOptions(n_iterations=2, grad_converge=False, floor_height=-10.0),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.6, -0.8, 0.5), camera_lookat=(0.0, 0.1, 0.0)),
        show_viewer=show_viewer,
    )
    muscles = [
        scene.add_entity(
            morph=gs.morphs.Box(size=(0.2, 0.1, 0.1), pos=(0.0, y, 0.0), nobisect=False, maxvolume=4e-4),
            material=gs.materials.VBD.Muscle(E=2e4, nu=0.3, n_groups=n_groups),
        )
        for y, n_groups in ((0.0, 2), (0.3, 1))
    ]
    scene.build(n_envs=n_envs)
    for muscle in muscles:
        muscle.set_muscle(
            np.zeros(muscle.n_elements, dtype=np.int32),
            np.tile([1.0, 0.0, 0.0], (muscle.n_elements, 1)),
        )
    return scene, muscles


@pytest.mark.parametrize("n_envs", [0, 2])
def test_independent_muscle_entities_hold_and_reset_commands(show_viewer, n_envs):
    scene, muscles = build_muscles(show_viewer, n_envs)
    for active in (0, 1):
        scene.reset()
        for index, muscle in enumerate(muscles):
            values = np.zeros((muscle.material.n_groups, max(n_envs, 1)))
            if index == active:
                values[0] = [0.4, 0.2] if n_envs else [0.4]
            muscle.set_actuation(values)
        for _ in range(2):
            scene.step()  # Commands must remain independent when held.
        passive = muscles[1 - active]
        assert_allclose(passive.get_positions(), passive.init_positions[None], atol=2e-6)
        positions = muscles[active].get_positions()
        length = positions[..., 0].max(dim=1).values - positions[..., 0].min(dim=1).values
        assert (length < 0.2 - 1e-5).all()
        if n_envs:
            assert length[0] < length[1] - 1e-5
    scene.reset()
    scene.step()
    for muscle in muscles:
        assert_allclose(muscle.get_positions(), muscle.init_positions[None], atol=2e-6)


@pytest.mark.parametrize("precision", ["64"])
def test_each_muscle_entity_receives_its_own_actuation_gradient(show_viewer):
    scene, muscles = build_muscles(show_viewer, requires_grad=True)
    rng = np.random.default_rng(21)
    weights = [gs.tensor(rng.normal(size=(1, muscle.n_vertices, 3))) for muscle in muscles]

    def rollout(values, requires_grad):
        scene.reset()
        controls = [
            gs.tensor(values[:2], requires_grad=requires_grad),
            gs.tensor(values[2:], requires_grad=requires_grad),
        ]
        for muscle, control in zip(muscles, controls):
            muscle.set_actuation(control)
        for _ in range(2):
            scene.step()
        loss = sum((weight * muscle.get_state().pos).sum() for weight, muscle in zip(weights, muscles))
        return loss, controls

    nominal = np.array([0.4, 0.1, 0.2])
    loss, controls = rollout(nominal, True)
    scene.backward(loss)
    analytic = torch.cat([control.grad for control in controls]).cpu().numpy()
    numeric = []
    for coordinate in range(3):
        delta = np.zeros(3)
        delta[coordinate] = 1e-5
        with torch.no_grad():
            plus, _ = rollout(nominal + delta, False)
            minus, _ = rollout(nominal - delta, False)
        numeric.append((plus.item() - minus.item()) / 2e-5)
    assert (np.abs(analytic[[0, 2]]) > 1e-7).all()
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-9)
    assert analytic[1] == 0.0  # No tetrahedron uses the second local group.
