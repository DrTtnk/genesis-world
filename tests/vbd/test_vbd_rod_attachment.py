"""Off-centre force transfer through native rod and rigid integration."""
import numpy as np
import pytest
import torch

import genesis as gs
from genesis.engine.solvers.vbd_rod import quat_matrix
from genesis.utils.misc import qd_to_torch, tensor_to_array


pytestmark = pytest.mark.precision("64")

def test_rod_offcentre_pull_and_snapshot(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=12,
            floor_height=-float('inf'),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.3, 0.3),
            camera_lookat=(0.0, 0.0, 0.22),
        ),
        show_viewer=show_viewer,
    )
    rest = np.array([[0.01, 0.0, 0.2], [0.01, 0.0, 0.24]])
    rod = scene.add_entity(
        morph=gs.morphs.Rod(
            verts=rest,
            frames=np.array([[1.0, 0.0, 0.0, 0.0]]),
            radius=0.003,
        ),
        material=gs.materials.VBD.Rod(),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.02,
            pos=(0.0, 0.0, 0.2),
        ),
        material=gs.materials.Rigid(),
    )
    rod.add_rigid_attachments([0], bone.links[0])
    scene.build()
    rod.set_pinned([False, True])
    target = rest.copy()
    target[1, 2] += 0.001
    rod.set_pin_targets(target)
    scene.step()
    assert bone.get_pos()[2] > 0.2
    assert bone.get_quat()[2] < 0
    attachment = scene.sim.vbd_solver.rigid_attachment
    local = qd_to_torch(attachment.info.local_pos)
    quat = torch.tensor(tensor_to_array(bone.get_quat()))
    anchor = torch.tensor(tensor_to_array(bone.get_pos())) + local @ quat_matrix(quat).T
    gap = torch.tensor(tensor_to_array(rod.get_positions())[0, :1]) - anchor
    assert gap.norm() < 1e-5
    snapshot = scene.get_state()
    scene.step()
    expected_pos = bone.get_pos().clone()
    expected_quat = bone.get_quat().clone()
    expected_rod = scene.sim.vbd_solver.get_state(scene.sim.cur_substep_local)
    scene.reset(snapshot)
    scene.step()
    torch.testing.assert_close(bone.get_pos(), expected_pos, atol=1e-12, rtol=0)
    torch.testing.assert_close(bone.get_quat(), expected_quat, atol=1e-12, rtol=0)
    actual = scene.sim.vbd_solver.get_state(scene.sim.cur_substep_local)
    torch.testing.assert_close(actual.pos, expected_rod.pos, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual.attachment_multiplier, expected_rod.attachment_multiplier, atol=1e-12, rtol=0)
