"""Free rigid links can be integrated by VBD without a tissue carrier."""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array


@pytest.mark.parametrize("n_envs", [0, 2])
def test_standalone_free_link_falls_once_and_reset_replays(n_envs, show_viewer):
    dt = 1e-3
    gravity = -9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0.0, 0.0, gravity)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-float("inf")),
        show_viewer=show_viewer,
    )
    skull = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.2), fixed=True),
        material=gs.materials.Rigid(rho=1500.0),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 0.0, 0.1)),
        material=gs.materials.Rigid(rho=1500.0),
    )
    scene.vbd_solver.add_rigid_link(bone.links[0])
    scene.build(n_envs=n_envs)
    initial_bone = tensor_to_array(bone.get_pos()).copy()
    initial_skull = tensor_to_array(skull.get_pos()).copy()
    scene.step()
    first_pos = tensor_to_array(bone.get_pos()).copy()
    first_vel = tensor_to_array(bone.get_vel()).copy()
    np.testing.assert_allclose(first_pos, initial_bone + (0.0, 0.0, gravity * dt * dt), atol=1e-8)
    np.testing.assert_allclose(first_vel, np.broadcast_to((0.0, 0.0, gravity * dt), first_vel.shape), atol=1e-8)
    np.testing.assert_array_equal(tensor_to_array(skull.get_pos()), initial_skull)
    scene.reset()
    scene.step()
    np.testing.assert_allclose(tensor_to_array(bone.get_pos()), first_pos, atol=1e-10)
    np.testing.assert_allclose(tensor_to_array(bone.get_vel()), first_vel, atol=1e-10)


def test_duplicate_rigid_ownership_is_rejected():
    scene = gs.Scene(show_viewer=False)
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 0.0, 0.1)),
        material=gs.materials.Rigid(rho=1500.0),
    )
    scene.vbd_solver.add_rigid_link(bone.links[0])
    with pytest.raises(gs.GenesisException, match="already"):
        scene.vbd_solver.add_rigid_link(bone.links[0])


@pytest.mark.parametrize("n_envs", [0, 2])
def test_standalone_joint_exerts_force_without_tissue(n_envs, show_viewer):
    dt = 1e-3
    gravity = -9.81
    stiffness = 1e3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0.0, 0.0, gravity)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=8, floor_height=-float("inf")),
        show_viewer=show_viewer,
    )
    skull = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.2), fixed=True),
        material=gs.materials.Rigid(rho=1500.0),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 0.0, 0.1)),
        material=gs.materials.Rigid(rho=1500.0),
    )
    solver = scene.vbd_solver
    solver.add_rigid_link(bone.links[0])
    solver.add_rigid_joint(
        skull.links[0], bone.links[0], (0.0, 0.0, 0.1), np.eye(3),
        translational_stiffness=(stiffness,) * 3, rotational_stiffness=(1e-2, 1e-2, 1e-2),
    )
    scene.build(n_envs=n_envs)
    initial = tensor_to_array(bone.get_pos()).copy()
    mass = float(bone.links[0].get_mass())
    for _ in range(100):
        scene.step()
    final_pos = tensor_to_array(bone.get_pos()).copy()
    displacement = final_pos[..., 2] - initial[..., 2]
    np.testing.assert_allclose(displacement, mass * gravity / stiffness, rtol=1e-3, atol=1e-8)

    snapshot = scene.get_state()
    scene.step()
    advanced_pos = tensor_to_array(bone.get_pos()).copy()
    advanced_vel = tensor_to_array(bone.get_vel()).copy()
    scene.reset(snapshot, envs_idx=[0] if n_envs else None)
    if n_envs:
        np.testing.assert_allclose(tensor_to_array(bone.get_pos())[1], advanced_pos[1], atol=1e-12)
        np.testing.assert_allclose(tensor_to_array(bone.get_vel())[1], advanced_vel[1], atol=1e-12)
    scene.step()
    np.testing.assert_allclose(tensor_to_array(bone.get_pos()).reshape(-1, 3)[0], advanced_pos.reshape(-1, 3)[0], atol=1e-10)
    np.testing.assert_allclose(tensor_to_array(bone.get_vel()).reshape(-1, 3)[0], advanced_vel.reshape(-1, 3)[0], atol=1e-10)


def test_declaring_one_of_two_free_links_is_rejected():
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(floor_height=-float("inf")),
        show_viewer=False,
    )
    bones = [
        scene.add_entity(
            morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(x, 0.0, 0.1)),
            material=gs.materials.Rigid(rho=1500.0),
        )
        for x in (0.0, 0.1)
    ]
    scene.vbd_solver.add_rigid_link(bones[0].links[0])
    with pytest.raises(gs.GenesisException, match="every free rigid link"):
        scene.build()
