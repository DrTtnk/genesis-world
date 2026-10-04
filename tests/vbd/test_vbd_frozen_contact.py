"""A real rigid impact and rest gate for the opt-in substep-frozen contact row."""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array


@pytest.mark.parametrize("n_envs", [0, 2])
def test_frozen_contact_stops_a_bone_and_carries_its_weight(n_envs, show_viewer):
    dt = 1e-3
    thickness = 2e-4
    density = 1500.0
    mass = density * 0.02 * 0.01 * 0.003
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=10,
            floor_height=-float("inf"),
            contact_margin=thickness,
            contact_k_max_ratio=4.0,
            contact_linearization="substep",
        ),
        show_viewer=show_viewer,
    )
    plate = scene.add_entity(
        morph=gs.morphs.Box(size=(0.06, 0.06, 0.004), pos=(0.0, 0.0, -0.002), fixed=True),
        material=gs.materials.Rigid(rho=density),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(0.0, 0.0, 0.0018)),
        material=gs.materials.Rigid(rho=density),
    )
    solver = scene.vbd_solver
    solver.add_rigid_link(bone.links[0])
    solver.add_rigid_collider(plate.links[0], collision_group=0)
    solver.add_rigid_collider(bone.links[0], collision_group=1)
    solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.1, thickness=thickness)
    scene.build(n_envs=n_envs)

    initial_bottom = np.asarray(tensor_to_array(bone.get_verts()))[..., 2].min(axis=-1)
    assert np.all(initial_bottom > thickness)
    heights = []
    velocities = []
    for _ in range(120):
        scene.step()
        heights.append(np.asarray(tensor_to_array(bone.get_verts()))[..., 2].min(axis=-1))
        velocities.append(np.asarray(tensor_to_array(bone.get_vel()))[..., 2])
    heights = np.stack(heights)
    velocities = np.stack(velocities)
    # Actual rotated collision vertices must enter the layer and remain above the plate.
    assert np.min(heights) > 0.0
    assert np.max(heights[60:]) < thickness * 1.05
    assert np.max(np.abs(velocities[60:])) < 1e-3
    assert not np.asarray(tensor_to_array(solver.env_status().is_failed)).any()
    pairs = solver.contact.snapshot()
    assert int(np.min(tensor_to_array(pairs.n_pt))) > 0
    assert int(np.min(tensor_to_array(pairs.n_ee))) > 0
    reaction = -np.asarray(tensor_to_array(solver.collider_reactions()))[:, 0, 2]
    np.testing.assert_allclose(reaction, mass * 9.81, rtol=0.02, atol=0.0)

    snapshot = scene.get_state()
    scene.step()
    advanced_pos = np.asarray(tensor_to_array(bone.get_pos())).copy()
    advanced_vel = np.asarray(tensor_to_array(bone.get_vel())).copy()
    scene.reset(snapshot)
    scene.step()
    np.testing.assert_allclose(tensor_to_array(bone.get_pos()), advanced_pos, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(tensor_to_array(bone.get_vel()), advanced_vel, rtol=0.0, atol=1e-10)


