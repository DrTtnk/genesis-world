import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
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
