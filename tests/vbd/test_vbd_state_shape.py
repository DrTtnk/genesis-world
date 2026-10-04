"""Malformed snapshots must fail before they can partly restore a live scene."""

import pytest
import torch

import genesis as gs


@pytest.mark.parametrize("n_envs", [0, 2])
def test_malformed_vertex_snapshot_is_refused_without_mutation(n_envs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1),
        vbd_options=gs.options.VBDOptions(n_iterations=1, floor_height=-10.0),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.01,) * 3, pos=(0.0, 0.0, 0.1), maxvolume=4e-7),
        material=gs.materials.VBD.Muscle(E=1e3, nu=0.3),
    )
    scene.build(n_envs=n_envs)
    scene.step()
    solver = scene.vbd_solver
    original = solver.get_state(0)
    batch, vertices, coordinates = original.pos.shape
    malformed_shapes = (
        (batch, vertices - 1, coordinates),
        (batch, vertices + 1, coordinates),
        (batch + 1, vertices, coordinates),
        (batch, vertices, coordinates - 1),
    )
    for field in ("_pos", "_vel"):
        for shape in malformed_shapes:
            state = solver.get_state(0)
            state.pos.add_(0.03)
            state.vel.add_(0.2)
            setattr(state, field, gs.zeros(shape))
            with pytest.raises(gs.GenesisException, match="VBD vertex snapshot shape"):
                solver.set_state(0, state)
            actual = solver.get_state(0)
            torch.testing.assert_close(actual.pos, original.pos, rtol=0, atol=0)
            torch.testing.assert_close(actual.vel, original.vel, rtol=0, atol=0)
