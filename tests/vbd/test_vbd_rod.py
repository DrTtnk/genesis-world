"""Native scene gates for the passive rod primitive."""

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch, tensor_to_array


pytestmark = pytest.mark.precision("64")

def rod_scene(gravity=(0.0, 0.0, -9.81), floor=-float("inf")):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=gravity),
        vbd_options=gs.options.VBDOptions(n_iterations=8, floor_height=floor),
        show_viewer=False,
    )
    rest = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.24], [0.0, 0.0, 0.28]])
    rod = scene.add_entity(
        morph=gs.morphs.Rod(verts=rest, frames=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)), radius=0.003),
        material=gs.materials.VBD.Rod(),
    )
    return scene, rod, rest


def test_rod_free_fall_and_mass():
    scene, rod, rest = rod_scene()
    scene.build()
    scene.step()
    state = scene.sim.vbd_solver.get_state(scene.sim.cur_substep_local)
    np.testing.assert_allclose(tensor_to_array(state.pos)[0], rest + [0.0, 0.0, -0.000981], atol=1e-12, rtol=0)
    torch.testing.assert_close(state.rod_states[0].scale, torch.ones(3, dtype=gs.tc_float), atol=1e-12, rtol=0)
    assert scene.sim.vbd_solver.n_elements == 0
    mass = qd_to_torch(scene.sim.vbd_solver.verts_info.mass)
    torch.testing.assert_close(
        mass,
        torch.tensor(np.pi * 0.003**2 * 1000 * np.array([0.02, 0.04, 0.02]), dtype=gs.tc_float),
        rtol=1e-12,
        atol=1e-15,
    )
    torch.testing.assert_close(scene.sim.vbd_solver.get_gravity(), gs.tensor([0.0, 0.0, -9.81]))


def test_rod_prescribed_extension_and_nonrest_reset():
    scene, rod, rest = rod_scene(gravity=(0.0, 0.0, 0.0))
    scene.build()
    rod.set_pinned(np.ones(3, dtype=bool))
    target = rest.copy()
    target[:, 2] = 0.2 + (target[:, 2] - 0.2) * 1.25
    rod.set_pin_targets(target)
    scene.step()
    snapshot = scene.get_state()
    scene.step()
    expected = scene.sim.vbd_solver.get_state(scene.sim.cur_substep_local)
    scene.reset(snapshot)
    scene.step()
    actual = scene.sim.vbd_solver.get_state(scene.sim.cur_substep_local)
    np.testing.assert_allclose(tensor_to_array(actual.pos)[0], target, atol=1e-14, rtol=0)
    for value, reference in zip(actual.rod_states[0], expected.rod_states[0]):
        torch.testing.assert_close(value, reference, atol=1e-12, rtol=0)
    scales = actual.rod_states[0].scale
    assert bool((scales < 1).all() & (scales > 0).all())
    volume_ratio = 1.25 * (((scales[:-1] + scales[1:]) / 2) ** 2).mean()
    assert abs(volume_ratio - 1) < 2e-4


def test_rod_refuses_unimplemented_floor():
    scene, rod, rest = rod_scene(floor=0.0)
    with pytest.raises(gs.GenesisException, match="floor, self-thickness, damping"):
        scene.build()


def test_rod_refuses_one_way_bone_scene():
    scene, rod, rest = rod_scene()
    scene.add_entity(morph=gs.morphs.Box(size=(0.01, 0.01, 0.01)), material=gs.materials.Rigid())
    with pytest.raises(gs.GenesisException, match="two-way rigid coupling"):
        scene.build()


def test_rod_pose_transforms_centres_and_frames_together():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(floor_height=-float("inf")),
        show_viewer=False,
    )
    rest = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.24], [0.0, 0.0, 0.28]])
    rod = scene.add_entity(
        morph=gs.morphs.Rod(
            verts=rest,
            frames=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
            radius=0.003,
            pos=(0.1, 0.2, 0.3),
            euler=(0.0, 90.0, 0.0),
            offset_pos=(0.02, 0.0, 0.0),
        ),
        material=gs.materials.VBD.Rod(),
    )
    expected = np.array([[0.06, 0.2, 0.52], [0.1, 0.2, 0.52], [0.14, 0.2, 0.52]])
    np.testing.assert_allclose(tensor_to_array(rod.init_positions), expected, atol=1e-12, rtol=0)
    scene.build()
    scene.step()
    state = scene.sim.vbd_solver.get_state(scene.sim.cur_substep_local)
    np.testing.assert_allclose(tensor_to_array(state.pos)[0], expected, atol=1e-12, rtol=0)
