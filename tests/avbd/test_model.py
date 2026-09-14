"""`gs.avbd.build_model`: turn a validated AVBD packet into a live Genesis scene.

Covers AVBD_API_MAP.md section 2 (build) and section 6 (read: ID maps). The positive-path fixture
(`build_hinge_model_packet`) mirrors `tests/vbd/test_vbd_mtu.py::hinge_scene` closely enough that the
built model must show the same qualitative pull. Every other test here isolates exactly one record
class `build_model` must refuse, built on top of `build_minimal_packet`, so a failure message can be
checked to name that one record.
"""

import dataclasses

import numpy as np
import pytest
import torch

import genesis as gs

from .fixture_packet import _quat, _single_tet_tissue, _tet_verts, build_hinge_model_packet, build_minimal_packet


def _quat_arr(w, x, y, z):
    return _quat(w, x, y, z)


def hinge_model_scene():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2.5e-3, substeps=4, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler, batch_links_info=True),
        vbd_options=gs.options.VBDOptions(n_iterations=4, floor_height=-10.0, damping=2e-3),
        show_viewer=False,
    )
    return scene


########################## the positive path ##########################


def test_build_model_creates_the_scene_and_it_steps():
    scene = hinge_model_scene()
    packet = build_hinge_model_packet()
    model = gs.avbd.build_model(scene, packet)
    scene.build()

    model.set_prescribed_targets(
        torch.tensor([[0.11, 0.0, 0.05]], device=gs.device).unsqueeze(1),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device).unsqueeze(1),
    )
    for _ in range(5):
        scene.step()
    bone = model.ids.links["bone"]
    assert torch.isfinite(bone.get_pos()).all()


def test_build_model_ids_resolve_to_the_right_entities():
    scene = hinge_model_scene()
    packet = build_hinge_model_packet()
    model = gs.avbd.build_model(scene, packet)
    scene.build()

    assert model.ids.links["base"].name == "base"
    assert model.ids.links["bone"].name == "bone"
    assert model.ids.links["meal"].name == "meal"
    assert model.ids.tissues["wall"].n_vertices == 4
    assert set(model.ids.mtus) == {"mtu_flexor"}
    assert set(model.ids.ligaments) == {"lig_main"}
    assert set(model.ids.rotary_restraints) == {"restraint_bone"}
    assert set(model.ids.colliders) == {"pc_meal"}
    assert "j_bone:hinge" in model.ids.coordinates


def test_built_model_pulls_the_hinge_the_way_the_flexor_shortens():
    """Mirrors `test_vbd_mtu.py::test_the_pull_on_a_link_anchor_turns_the_hinge_the_way_the_route_shortens`:
    the flexor MTU's anchors are the same offsets on `base` and `bone`, at the same hinge geometry, so
    exciting it must turn the hinge the same qualitative way (bone qpos goes negative)."""
    scene = hinge_model_scene()
    packet = build_hinge_model_packet()
    model = gs.avbd.build_model(scene, packet)
    scene.build()
    bone_link = model.ids.links["bone"]
    meal_pos = torch.tensor([[0.11, 0.0, 0.05]], device=gs.device).unsqueeze(1)
    meal_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device).unsqueeze(1)
    # The solver's routed-unit table holds MTUs and ligaments together (a ligament has no fibre or
    # activation, but still occupies an excitation slot it never reads), so the excitation tensor spans both.
    n_units = len(model.ids.mtus) + len(model.ids.ligaments)
    excitation = torch.zeros(1, n_units, device=gs.device)
    excitation[0, model.ids.mtus["mtu_flexor"]] = 1.0
    for _ in range(60):
        model.set_prescribed_targets(meal_pos, meal_quat)
        model.set_excitation(excitation)
        scene.step()
    qpos = float(bone_link.entity.get_qpos()[bone_link.dof_start - bone_link.entity.dof_start])
    assert qpos < -1e-2
    tension = float(model.mtu_state().tension[0, model.ids.mtus["mtu_flexor"]])
    assert tension > 0.0
    reactions = model.get_reactions()
    assert reactions.shape == (1, 1, 6)
    assert torch.isfinite(reactions).all()


########################## build-order rejections ##########################


def test_build_model_rejects_a_second_call_after_scene_build():
    scene = hinge_model_scene()
    gs.avbd.build_model(scene, build_minimal_packet())
    scene.build()
    with pytest.raises(gs.GenesisException, match="scene.build"):
        gs.avbd.build_model(scene, build_minimal_packet())


def test_build_model_rejects_a_second_model_in_one_scene():
    scene = hinge_model_scene()
    gs.avbd.build_model(scene, build_minimal_packet())
    with pytest.raises(gs.GenesisException, match="one AVBD model"):
        gs.avbd.build_model(scene, build_minimal_packet())


########################## unsupported-record rejections ##########################


def _add_dynamic_link_and_joint(packet, axes, coordinate_ids, joint_id="j_extra", parent_id="root", child_id="extra"):
    av = gs.avbd
    n = len(axes)
    child = av.Link(
        id=child_id,
        entity_id=f"e_{child_id}",
        motion_mode="dynamic",
        rest_position_m=np.zeros(3),
        rest_quaternion_wxyz=_quat_arr(1.0, 0.0, 0.0, 0.0),
        mass_kg=0.05,
        com_link_m=np.zeros(3),
        inertia_com_kgm2=np.diag([1.0e-5, 1.0e-5, 1.0e-5]),
        collision_geometry_id=None,
        collision_group=None,
    )
    joint = av.Joint(
        id=joint_id,
        parent_link_id=parent_id,
        child_link_id=child_id,
        joint_type="hinge_chain",
        parent_frame_position_m=np.zeros(3),
        parent_frame_quaternion_wxyz=_quat_arr(1.0, 0.0, 0.0, 0.0),
        child_frame_position_m=np.zeros(3),
        child_frame_quaternion_wxyz=_quat_arr(1.0, 0.0, 0.0, 0.0),
        axes=tuple(axes),
        coordinate_ids=tuple(coordinate_ids),
        rest_coordinates=np.zeros(n),
        lower_limits=np.full(n, -0.5),
        upper_limits=np.full(n, 0.5),
        damping=np.full(n, 1.0e-3),
        armature=np.zeros(n),
        parent_world_position_m=None,
        parent_world_quaternion_wxyz=None,
    )
    e_child = av.Entity(id=f"e_{child_id}", link_ids=(child_id,), tissue_ids=(), role="extra", integration_owner=child_id)
    return dataclasses.replace(
        packet, links=packet.links + (child,), joints=packet.joints + (joint,), entities=packet.entities + (e_child,)
    )


def test_build_model_rejects_a_multi_axis_joint():
    packet = _add_dynamic_link_and_joint(build_minimal_packet(), axes=("hinge", "yaw"), coordinate_ids=("j_extra:hinge", "j_extra:yaw"))
    with pytest.raises(gs.GenesisException, match="j_extra.*single-axis"):
        gs.avbd.build_model(hinge_model_scene(), packet)


def test_build_model_rejects_a_free_root_articulated_chain():
    av = gs.avbd
    packet = build_minimal_packet()
    free_root = av.Link(
        id="free_root",
        entity_id="e_free_root",
        motion_mode="dynamic",
        rest_position_m=np.array([0.0, 0.3, 0.0]),
        rest_quaternion_wxyz=_quat_arr(1.0, 0.0, 0.0, 0.0),
        mass_kg=0.2,
        com_link_m=np.zeros(3),
        inertia_com_kgm2=np.diag([1.0e-4, 1.0e-4, 1.0e-4]),
        collision_geometry_id=None,
        collision_group=None,
    )
    e_free_root = av.Entity(id="e_free_root", link_ids=("free_root",), tissue_ids=(), role="free_root", integration_owner="free_root")
    packet = dataclasses.replace(packet, links=packet.links + (free_root,), entities=packet.entities + (e_free_root,))
    packet = _add_dynamic_link_and_joint(packet, axes=("hinge",), coordinate_ids=("j_extra:hinge",), parent_id="free_root")
    with pytest.raises(gs.GenesisException, match="free_root.*free-root articulated chain"):
        gs.avbd.build_model(hinge_model_scene(), packet)


def _packet_with_free_link_and_attachment(law, weights):
    av = gs.avbd
    packet = build_minimal_packet()
    free_link = av.Link(
        id="free_link",
        entity_id="e_free_link",
        motion_mode="dynamic",
        rest_position_m=np.array([0.0, 0.3, 0.0]),
        rest_quaternion_wxyz=_quat_arr(1.0, 0.0, 0.0, 0.0),
        mass_kg=0.05,
        com_link_m=np.zeros(3),
        inertia_com_kgm2=np.diag([1.0e-5, 1.0e-5, 1.0e-5]),
        collision_geometry_id=None,
        collision_group=None,
    )
    e_free_link = av.Entity(id="e_free_link", link_ids=("free_link",), tissue_ids=(), role="free_link", integration_owner="free_link")
    anchor_wall = av.Anchor(
        id="a_wall",
        active=True,
        kind="tissue",
        position_m=None,
        link_id=None,
        position_link_m=None,
        tissue_id="wall",
        tet_index=0,
        barycentric_weights=np.array(weights),
    )
    anchor_link = av.Anchor(
        id="a_free_link",
        active=True,
        kind="link",
        position_m=None,
        link_id="free_link",
        position_link_m=np.zeros(3),
        tissue_id=None,
        tet_index=None,
        barycentric_weights=None,
    )
    attachment = av.Attachment(
        id="att_test",
        tissue_anchor_id="a_wall",
        other_anchor_id="a_free_link",
        law=law,
        stiffness_N_m=800.0 if law == "elastic_point" else None,
        damping_Ns_m=5.0 if law == "elastic_point" else None,
        collision_exclusion=False,
    )
    return dataclasses.replace(
        packet,
        links=packet.links + (free_link,),
        entities=packet.entities + (e_free_link,),
        anchors=packet.anchors + (anchor_wall, anchor_link),
        attachments=packet.attachments + (attachment,),
        required_capabilities=("hard_point_attachment", "free_link"),
    )


def test_build_model_rejects_an_elastic_point_attachment():
    packet = _packet_with_free_link_and_attachment("elastic_point", [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(gs.GenesisException, match="att_test.*elastic_point"):
        gs.avbd.build_model(hinge_model_scene(), packet)


def test_build_model_rejects_a_barycentric_hard_point_attachment():
    packet = _packet_with_free_link_and_attachment("hard_point", [0.4, 0.3, 0.2, 0.1])
    with pytest.raises(gs.GenesisException, match="att_test.*barycentric"):
        gs.avbd.build_model(hinge_model_scene(), packet)


def test_build_model_rejects_a_tissue_to_tissue_attachment():
    av = gs.avbd
    packet = build_minimal_packet()
    other_tissue = _single_tet_tissue("wall_b", "mat_wall", _tet_verts((0.3, 0.0, 0.0)))
    e_wall_b = av.Entity(id="e_wall_b", link_ids=(), tissue_ids=("wall_b",), role="wall_b", integration_owner="wall_b")
    anchor_a = av.Anchor(
        id="a_wall_a", active=True, kind="tissue", position_m=None, link_id=None, position_link_m=None,
        tissue_id="wall", tet_index=0, barycentric_weights=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    anchor_b = av.Anchor(
        id="a_wall_b", active=True, kind="tissue", position_m=None, link_id=None, position_link_m=None,
        tissue_id="wall_b", tet_index=0, barycentric_weights=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    attachment = av.Attachment(
        id="att_seam", tissue_anchor_id="a_wall_a", other_anchor_id="a_wall_b", law="hard_point",
        stiffness_N_m=None, damping_Ns_m=None, collision_exclusion=False,
    )
    packet = dataclasses.replace(
        packet,
        tissues=packet.tissues + (other_tissue,),
        entities=packet.entities + (e_wall_b,),
        anchors=packet.anchors + (anchor_a, anchor_b),
        attachments=packet.attachments + (attachment,),
    )
    with pytest.raises(gs.GenesisException, match="att_seam.*tissue-to-tissue"):
        gs.avbd.build_model(hinge_model_scene(), packet)


def test_build_model_rejects_multiple_materials_on_one_tissue():
    av = gs.avbd
    packet = build_minimal_packet()
    other_material = av.Material(
        id="mat_wall_2", law="neo_hookean_v1", law_params={"E_Pa": 2.0e4, "nu": 0.4}, density_kg_m3=1040.0, thickness_m=0.0015, provenance="synthetic"
    )
    (tissue,) = packet.tissues
    two_material_tissue = dataclasses.replace(tissue, tet_material_ids=("mat_wall", "mat_wall_2"))
    packet = dataclasses.replace(packet, tissues=(two_material_tissue,), materials=packet.materials + (other_material,))
    with pytest.raises(gs.GenesisException, match="wall.*materials"):
        gs.avbd.build_model(hinge_model_scene(), packet)


def test_build_model_rejects_a_region():
    av = gs.avbd
    packet = build_minimal_packet()
    region = av.Region(
        id="stomach", owner_ids=("wall",), geometry={"kind": "convex_hull_of_tissue", "tissue_id": "wall"},
        purpose="stomach_containment", rest_to_current_mapping="wall_local", boundary_semantics="entrance_exit_portals",
    )
    packet = dataclasses.replace(packet, regions=(region,))
    with pytest.raises(gs.GenesisException, match="stomach"):
        gs.avbd.build_model(hinge_model_scene(), packet)


def test_build_model_rejects_an_unadvertised_capability():
    packet = dataclasses.replace(build_minimal_packet(), required_capabilities=("dynamic_tendon_wrapping",))
    with pytest.raises(gs.GenesisException, match="unsupported capabilities"):
        gs.avbd.build_model(hinge_model_scene(), packet)


########################## gs.morphs.TetMesh ##########################


def test_tetmesh_rejects_non_finite_verts():
    verts = _tet_verts((0.0, 0.0, 0.0))
    verts[0, 0] = np.nan
    with pytest.raises(gs.GenesisException, match="non-finite"):
        gs.morphs.TetMesh(verts=verts, elems=np.array([[0, 1, 2, 3]]), faces=np.array([[0, 1, 2]]))


def test_tetmesh_rejects_an_out_of_range_element_index():
    verts = _tet_verts((0.0, 0.0, 0.0))
    with pytest.raises(gs.GenesisException, match="out of range"):
        gs.morphs.TetMesh(verts=verts, elems=np.array([[0, 1, 2, 4]]), faces=np.array([[0, 1, 2]]))


def test_tetmesh_rejects_a_wrong_shaped_elems_array():
    verts = _tet_verts((0.0, 0.0, 0.0))
    with pytest.raises(gs.GenesisException, match="shape"):
        gs.morphs.TetMesh(verts=verts, elems=np.array([[0, 1, 2]]), faces=np.array([[0, 1, 2]]))
