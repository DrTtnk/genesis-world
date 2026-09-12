"""An incrementally positioned fixed collider must move rigid and PBD material."""

import numpy as np
import pytest

import genesis as gs


@pytest.mark.required
def test_fixed_collider_displacement_reaches_both_solvers(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
        pbd_options=gs.options.PBDOptions(particle_size=0.01, lower_bound=(-1, -1, -1)),
        show_viewer=show_viewer,
    )
    driver = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.02), pos=(0, 0, -0.025), fixed=True))
    body = scene.add_entity(gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(-0.05, 0, 0.01)))
    cloth = scene.add_entity(
        gs.morphs.Mesh(file="meshes/cloth.obj", scale=0.04, pos=(0.05, 0, 0)),
        material=gs.materials.PBD.Cloth(air_resistance=0),
    )
    scene.build()
    assert driver.n_dofs == 0 and body.n_dofs == 6
    initial = cloth.get_particles_pos().cpu().numpy().copy()
    for _ in range(20):
        scene.step()
    np.testing.assert_allclose(cloth.get_particles_pos().cpu().numpy(), initial, atol=1e-6)
    np.testing.assert_allclose(body.get_pos().cpu().numpy(), [-0.05, 0, 0.01], atol=1e-6)
    contact = 0.
    for step in range(400):
        position = [0, 0, -0.025 + (step + 1) * 0.0001]
        driver.set_pos(position)
        scene.step()
        np.testing.assert_allclose(driver.get_pos().cpu().numpy(), position, atol=1e-7)
        contact = max(contact, float(body.get_links_net_contact_force().norm()))
    assert contact > 0.001
    # Hold the final boundary to separate response from the soft-contact lag.
    for _ in range(200):
        scene.step()
    assert float(body.get_pos()[2]) > 0.035 - 0.0001
    assert float(cloth.get_particles_pos()[:, 2].mean()) > 0.024
    # The current API is a sequence of stationary boundaries, not a velocity-driven wall.
    np.testing.assert_array_equal(driver.get_vel().cpu().numpy(), 0)
