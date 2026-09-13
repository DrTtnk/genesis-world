import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_torch, tensor_to_array
from tests.utils.assertions import assert_allclose


def test_vbd_ownership_rejects_unsupported_joint_friction(show_viewer, muscle_hinge_chain_xml):
    root = ET.fromstring(muscle_hinge_chain_xml)
    root.find(".//joint").set("frictionloss", "0.1")
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.6, -0.8, 0.4), camera_lookat=(0.0, 0.0, 0.0)),
        show_viewer=show_viewer,
    )
    skeleton = scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(root, encoding="unicode")))
    tissue = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.02, 0.02)), material=gs.materials.VBD.Base())
    tissue.add_rigid_attachments([0], skeleton.get_link("upper"))
    with pytest.raises(gs.GenesisException, match="does not support joint frictionloss"):
        scene.build()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_free_link_attachment_moves_once_and_transmits_muscle_force(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            substeps=4,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.8, 0.4),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.08, 0.08),
            pos=(0.12, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
        ),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.04, 0.04),
            pos=(0.0, 0.0, 0.03),
            nobisect=False,
            maxvolume=2e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
        ),
    )
    rest = tensor_to_array(tissue.init_positions)
    attached = np.flatnonzero(rest[:, 0] > rest[:, 0].max() - 1e-5)
    tissue.add_rigid_attachments(attached, bone.links[0])
    with pytest.raises(gs.GenesisException, match="only one rigid attachment"):
        tissue.add_rigid_attachments(attached, bone.links[0])
    scene.build(n_envs=n_envs)
    with pytest.raises(gs.GenesisException, match="before scene.build"):
        tissue.add_rigid_attachments(attached, bone.links[0])
    with pytest.raises(gs.GenesisException, match="remain free"):
        tissue.set_pinned(np.ones(tissue.n_vertices, dtype=bool))
    initial = bone.get_pos().clone()
    # Uniform gravity keeps every attachment and tetrahedron at rest.
    scene.step()
    expected = torch.tensor([0.0, 0.0, -9.81 * (0.005 / 4) ** 2 * 4 * 5 / 2], device=gs.device)
    assert_allclose(bone.get_pos() - initial, expected, atol=2e-6)
    assert_allclose(tissue.get_positions() - tissue.init_positions, expected, atol=2e-6)

    scene.reset()
    held = rest[:, 0] < rest[:, 0].min() + 1e-5
    tissue.set_pinned(held)
    tissue.set_muscle(np.zeros(tissue.n_elements), np.tile([1.0, 0.0, 0.0], (tissue.n_elements, 1)))
    tissue.set_actuation([[0.4, 0.2]] if n_envs else [0.4])
    rotations = []
    errors = []
    local = tissue.init_positions[attached] - initial.reshape(-1, 3)[0]
    for _ in range(30):
        scene.step()
        pos = bone.get_pos().reshape(-1, 3)
        quat = bone.get_quat().reshape(-1, 4)
        anchors = pos[:, None] + gu.transform_by_quat(local[None], quat[:, None])
        error = torch.linalg.vector_norm(tissue.get_positions()[:, attached] - anchors, dim=-1).max()
        rotations.append(quat)
        errors.append(error)
    assert (bone.get_pos()[..., 0] < initial[..., 0] - 0.001).all()
    # The initial pull has negative y torque; the undamped link can swing back later.
    assert (rotations[0][..., 2] < -1e-4).all()
    assert torch.stack(errors).max() < 2e-4
    assert torch.isfinite(tissue.get_positions()).all()
    if n_envs:
        assert torch.abs(bone.get_pos()[0, 0] - bone.get_pos()[1, 0]) > 0.001

    snapshot = scene.get_state()
    scene.step()
    expected_pos = bone.get_pos().clone()
    expected_vertices = tissue.get_positions().clone()
    scene.reset(snapshot, envs_idx=[0] if n_envs else None)
    if n_envs:
        assert_allclose(bone.get_pos()[1], expected_pos[1], atol=1e-12)
        assert_allclose(tissue.get_positions()[1], expected_vertices[1], atol=1e-12)
    scene.step()
    assert_allclose(bone.get_pos().reshape(-1, 3)[0], expected_pos.reshape(-1, 3)[0], atol=2e-6)
    assert_allclose(tissue.get_positions()[0], expected_vertices[0], atol=2e-6)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_articulated_muscle_attachments_preserve_joint_limits_and_reset(n_envs, show_viewer, muscle_hinge_chain_xml):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=8,
            floor_height=-10.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.7, 0.35),
            camera_lookat=(0.08, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    skeleton = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=muscle_hinge_chain_xml,
        ),
    )
    tissues = []
    bindings = []
    for centre, length, owners in ((0.0, 0.2, ("base", "upper")), (0.17, 0.06, ("upper", "lower"))):
        tissue = scene.add_entity(
            morph=gs.morphs.Box(
                size=(length, 0.04, 0.04),
                pos=(centre, 0.0, 0.03),
                nobisect=False,
                maxvolume=2e-5,
            ),
            material=gs.materials.VBD.Muscle(
                E=1e5,
                nu=0.3,
            ),
        )
        rest = tensor_to_array(tissue.init_positions)
        for x, owner in zip((rest[:, 0].min(), rest[:, 0].max()), owners):
            vertices = np.flatnonzero(np.abs(rest[:, 0] - x) < 1e-5)
            link = skeleton.get_link(owner)
            tissue.add_rigid_attachments(vertices, link)
            bindings.append((tissue, vertices, link))
        tissues.append(tissue)
    scene.build(n_envs=n_envs)
    local_anchors = [
        gu.inv_transform_by_quat(
            tissue.init_positions[vertices] - link.get_pos().reshape(-1, 3)[0], link.get_quat().reshape(-1, 4)[0]
        )
        for tissue, vertices, link in bindings
    ]
    base = skeleton.get_link("base")
    fixed_position = base.get_pos().clone()
    scene.step()
    assert_allclose(skeleton.get_dofs_position(), 0.0, atol=2e-6)
    for tissue in tissues:
        tissue.set_muscle(np.zeros(tissue.n_elements), np.tile([1.0, 0.0, 0.0], (tissue.n_elements, 1)))
        tissue.set_actuation([[0.4, 0.2]] if n_envs else [0.4])
    errors = []
    for _ in range(80):
        scene.step()
        assert (skeleton.get_dofs_position().abs() <= 0.050001).all()
        for (tissue, vertices, link), anchors in zip(bindings, local_anchors):
            targets = link.get_pos().reshape(-1, 1, 3) + gu.transform_by_quat(
                anchors[None], link.get_quat().reshape(-1, 1, 4)
            )
            errors.append(torch.linalg.vector_norm(tissue.get_positions()[:, vertices] - targets, dim=-1).max())
    assert torch.stack(errors).max() < 2e-4
    assert (skeleton.get_dofs_position().abs().max(dim=-1).values > 0.01).all()
    assert_allclose(skeleton.get_dofs_position().reshape(-1, 2)[0], -0.05, atol=2e-6)
    if n_envs:
        assert (skeleton.get_dofs_position()[1] - skeleton.get_dofs_position()[0] > 0.01).all()
    assert_allclose(base.get_pos(), fixed_position, atol=1e-12)
    snapshot = scene.get_state()
    scene.step()
    expected = skeleton.get_dofs_position().reshape(-1, 2).clone()
    expected_vertices = [tissue.get_positions().clone() for tissue in tissues]
    scene.reset(snapshot, envs_idx=[0] if n_envs else None)
    if n_envs:
        assert_allclose(skeleton.get_dofs_position()[1], expected[1], atol=1e-12)
        for tissue, positions in zip(tissues, expected_vertices):
            assert_allclose(tissue.get_positions()[1], positions[1], atol=1e-12)
    scene.step()
    assert_allclose(skeleton.get_dofs_position().reshape(-1, 2)[0], expected[0], atol=2e-6)
    for tissue, positions in zip(tissues, expected_vertices):
        assert_allclose(tissue.get_positions()[0], positions[0], atol=2e-6)


def _system_momentum(tissue, bone, masses):
    state = tissue.get_state()
    x = state.pos.reshape(-1, 3).double()
    v = state.vel.reshape(-1, 3).double()
    m = masses.double()[:, None]
    x_bone = bone.get_pos().reshape(3).double()
    v_bone = bone.get_vel().reshape(3).double()
    w_bone = bone.get_links_ang().reshape(3).double()
    rotation = torch.as_tensor(
        gu.quat_to_R(tensor_to_array(bone.get_quat().reshape(4))), dtype=torch.float64, device=x.device
    )
    inertia = torch.as_tensor(np.asarray(bone.links[0].inertial_i), dtype=torch.float64, device=x.device)
    mass_bone = float(bone.links[0].inertial_mass)
    linear = (m * v).sum(0) + mass_bone * v_bone
    angular = torch.linalg.cross(x, m * v).sum(0) + torch.linalg.cross(x_bone, mass_bone * v_bone)
    angular = angular + rotation @ inertia @ rotation.T @ w_bone
    return linear, angular, x, m, x_bone


@pytest.mark.parametrize("n_iterations", [4, 8])
def test_closed_actuated_system_keeps_its_momentum_within_the_frozen_budget(n_iterations, show_viewer, momentum_budget):
    budget = momentum_budget["closed_actuated"][str(n_iterations)]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=n_iterations,
            floor_height=-10.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.8, 0.4),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.08, 0.08),
            pos=(0.12, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
        ),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.04, 0.04),
            pos=(0.0, 0.0, 0.03),
            nobisect=False,
            maxvolume=2e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
        ),
    )
    rest = tensor_to_array(tissue.init_positions)
    tissue.add_rigid_attachments(np.flatnonzero(rest[:, 0] > rest[:, 0].max() - 1e-5), bone.links[0])
    scene.build()
    masses = qd_to_torch(scene.vbd_solver.verts_info.mass)
    tissue.set_muscle(np.zeros(tissue.n_elements), np.tile([1.0, 0.0, 0.0], (tissue.n_elements, 1)))
    tissue.set_actuation([0.4])
    linear0, angular0, *_ = _system_momentum(tissue, bone, masses)
    # the system starts at rest, so its momentum must stay exactly zero. There is nothing here for a numerical
    # damping to remove: whatever momentum appears was created by the solver.
    assert float(linear0.norm()) == 0.0
    assert float(angular0.norm()) == 0.0
    worst_linear = worst_angular = 0.0
    history = []
    for _ in range(60):
        scene.step()
        linear, angular, *_ = _system_momentum(tissue, bone, masses)
        worst_linear = max(worst_linear, float((linear - linear0).norm()))
        worst_angular = max(worst_angular, float((angular - angular0).norm()))
        history.append(float((linear - linear0).norm()))
    # internal forces only: the drift is the finite-sweep error of the coupling, bounded by the frozen budget of
    # this sweep count (the 8-sweep budget is the tighter one, so the pair also records the reduction)
    assert (
        worst_linear
        <= budget["linear_momentum"]["atol"] + budget["linear_momentum"]["rtol"] * budget["linear_momentum"]["scale"]
    )
    assert (
        worst_angular
        <= budget["angular_momentum"]["atol"] + budget["angular_momentum"]["rtol"] * budget["angular_momentum"]["scale"]
    )
    # the shape the norms above cannot show. The system starts at exactly zero momentum, so the drift is momentum
    # created, never momentum damped away: that is the injection AVBD section 5.6 describes for position-based
    # error correction. It is a startup transient, though, not a sustained gain: it peaks while the actuation
    # switches on and the constraint error is largest, then decays as the multipliers catch up.
    quarter = len(history) // 4
    peak = max(history[:quarter])
    assert peak > 0.0
    assert max(history[3 * quarter :]) <= budget["momentum_sign"]["late_over_peak"] * peak
    assert torch.isfinite(tissue.get_positions()).all()
