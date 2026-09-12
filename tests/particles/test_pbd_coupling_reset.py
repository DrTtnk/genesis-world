"""A real PBD contact must not leave a pending impulse across scene reset."""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_torch


@pytest.mark.required
def test_reset_clears_pending_pbd_reaction_only_in_reset_environments(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
        pbd_options=gs.options.PBDOptions(particle_size=0.02, lower_bound=(-1, -1, -1)),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Box(size=(0.08, 0.08, 0.04)), material=gs.materials.Rigid())
    scene.add_entity(
        gs.morphs.Mesh(file="meshes/cloth.obj", scale=0.1, pos=(0.015, 0, 0.01)),
        material=gs.materials.PBD.Cloth(air_resistance=0),
    )
    scene.build(n_envs=2)
    links = scene.rigid_solver.dyn_state.links

    def pending():
        return np.concatenate([
            qd_to_torch(field, transpose=True).cpu().numpy()
            for field in (links.cfrc_coupling_vel, links.cfrc_coupling_ang)
        ], axis=-1)

    np.testing.assert_array_equal(pending(), 0)
    scene.step()
    before = pending()
    assert np.linalg.norm(before[0]) > 0.1
    scene.reset(envs_idx=[0])
    after = pending()
    np.testing.assert_array_equal(after[0], 0)
    np.testing.assert_array_equal(after[1], before[1])
