"""Gate A0: load, validate and round-trip the static AVBD model packet.

Covers AVBD_API_MAP.md section 1 and AVBD_MODEL_INTERFACE.md section 6.
"""

import dataclasses
import json

import numpy as np
import pytest

import genesis as gs

from .fixture_packet import build_concave_tissue, build_fixture_packet


def _write(packet, tmp_path, name="model"):
    json_path = tmp_path / f"{name}.json"
    npz_path = tmp_path / f"{name}.npz"
    gs.avbd.write_packet(packet, json_path, npz_path)
    return json_path, npz_path


########################## missing-field coverage ##########################

# One (collection_name, field_name) pair per required field of every record
# type that appears in the fixture, plus every root Packet field. Generated
# from the dataclasses themselves so a newly added field is covered automatically.
_ROOT_FIELD_CASES = [f.name for f in dataclasses.fields(gs.avbd.Packet) if f.name not in gs.avbd.packet._RECORD_CLASSES]

_RECORD_FIELD_CASES = []
for _collection_name, _cls in gs.avbd.packet._RECORD_CLASSES.items():
    for _f in dataclasses.fields(_cls):
        _RECORD_FIELD_CASES.append((_collection_name, _f.name))


@pytest.fixture(scope="module")
def fixture_packet():
    return build_fixture_packet()


@pytest.mark.parametrize("field_name", _ROOT_FIELD_CASES)
def test_load_rejects_each_missing_root_field(tmp_path, fixture_packet, field_name):
    json_path, npz_path = _write(fixture_packet, tmp_path)
    with open(json_path) as fh:
        doc = json.load(fh)
    del doc[field_name]
    with open(json_path, "w") as fh:
        json.dump(doc, fh)

    with pytest.raises(gs.GenesisException, match=field_name):
        gs.avbd.load_packet(json_path, npz_path)


@pytest.mark.parametrize("collection_name, field_name", _RECORD_FIELD_CASES)
def test_load_rejects_each_missing_field(tmp_path, fixture_packet, collection_name, field_name):
    json_path, npz_path = _write(fixture_packet, tmp_path)
    with open(json_path) as fh:
        doc = json.load(fh)
    record = doc[collection_name][0]
    del record[field_name]
    with open(json_path, "w") as fh:
        json.dump(doc, fh)

    with pytest.raises(gs.GenesisException, match=field_name):
        gs.avbd.load_packet(json_path, npz_path)


@pytest.mark.parametrize("collection_name", list(gs.avbd.packet._RECORD_CLASSES))
def test_load_rejects_each_missing_collection(tmp_path, fixture_packet, collection_name):
    json_path, npz_path = _write(fixture_packet, tmp_path)
    with open(json_path) as fh:
        doc = json.load(fh)
    del doc[collection_name]
    with open(json_path, "w") as fh:
        json.dump(doc, fh)

    with pytest.raises(gs.GenesisException, match=collection_name):
        gs.avbd.load_packet(json_path, npz_path)


def test_load_rejects_unknown_interface_version(tmp_path, fixture_packet):
    json_path, npz_path = _write(fixture_packet, tmp_path)
    with open(json_path) as fh:
        doc = json.load(fh)
    doc["interface_version"] = "999"
    with open(json_path, "w") as fh:
        json.dump(doc, fh)

    with pytest.raises(gs.GenesisException, match="interface_version"):
        gs.avbd.load_packet(json_path, npz_path)


def test_load_rejects_wrong_array_shape(tmp_path, fixture_packet):
    json_path, npz_path = _write(fixture_packet, tmp_path)
    npz = dict(np.load(npz_path))
    key = "link.skull.rest_position_m"
    npz[key] = np.array([1.0, 2.0], dtype=np.float64)  # wrong length
    np.savez(npz_path, **npz)

    with pytest.raises(gs.GenesisException, match="shape"):
        gs.avbd.load_packet(json_path, npz_path)


def test_load_rejects_wrong_array_dtype(tmp_path, fixture_packet):
    json_path, npz_path = _write(fixture_packet, tmp_path)
    npz = dict(np.load(npz_path))
    key = "tissue.wall.tets"
    npz[key] = npz[key].astype(np.float64)  # connectivity must stay integer
    np.savez(npz_path, **npz)

    with pytest.raises(gs.GenesisException, match="dtype"):
        gs.avbd.load_packet(json_path, npz_path)


def test_load_rejects_non_finite_array_value(tmp_path, fixture_packet):
    json_path, npz_path = _write(fixture_packet, tmp_path)
    npz = dict(np.load(npz_path))
    key = "link.jaw.com_link_m"
    npz[key] = np.array([np.nan, 0.0, 0.0], dtype=np.float64)
    np.savez(npz_path, **npz)

    with pytest.raises(gs.GenesisException, match="non-finite"):
        gs.avbd.load_packet(json_path, npz_path)


########################## invalid-record coverage ##########################


def _replace_in(collection, record_id, **changes):
    """Return a new tuple of records with the record matching `record_id` replaced."""
    return tuple(dataclasses.replace(r, **changes) if r.id == record_id else r for r in collection)


def _mutate_duplicate_id(packet):
    dup = dataclasses.replace(packet.links[1], id=packet.links[0].id)
    return dataclasses.replace(packet, links=(packet.links[0], dup, packet.links[2]))


def _mutate_unresolved_id(packet):
    joints = _replace_in(packet.joints, packet.joints[0].id, child_link_id="does_not_exist")
    return dataclasses.replace(packet, joints=joints)


def _mutate_non_positive_mass(packet):
    links = _replace_in(packet.links, "jaw", mass_kg=0.0)
    return dataclasses.replace(packet, links=links)


def _mutate_inadmissible_inertia(packet):
    bad = np.diag([1.0e-6, 1.0e-6, 1.0])  # violates the triangle inequality
    links = _replace_in(packet.links, "jaw", inertia_com_kgm2=bad)
    return dataclasses.replace(packet, links=links)


def _mutate_non_positive_tet_volume(packet):
    tissue = packet.tissues[0]
    tets = tissue.tets.copy()
    tets[0] = tets[0][[1, 0, 2, 3]]  # swap two indices: flips the sign of the volume
    tissues = (dataclasses.replace(tissue, tets=tets),)
    return dataclasses.replace(packet, tissues=tissues)


def _mutate_boundary_not_outward(packet):
    tissue = packet.tissues[0]
    faces = tissue.boundary_faces.copy()
    faces[0] = faces[0][[0, 2, 1]]  # reverse one face's winding
    tissues = (dataclasses.replace(tissue, boundary_faces=faces),)
    return dataclasses.replace(packet, tissues=tissues)


def _mutate_lumen_not_closed(packet):
    tissue = packet.tissues[0]
    faces = tissue.boundary_faces[:-1].copy()  # drop one boundary face
    tissues = (dataclasses.replace(tissue, boundary_faces=faces),)
    return dataclasses.replace(packet, tissues=tissues)


def _mutate_contact_pair_without_one_material(packet):
    pairs = _replace_in(packet.contact_pairs, "pair_meal_wall", contact_material_ids=())
    return dataclasses.replace(packet, contact_pairs=pairs)


def _mutate_exclusion_removes_required_pair(packet):
    exclusions = (
        dataclasses.replace(packet.collision_exclusions[0], group_a="cg_meal", group_b="cg_wall"),
    )
    return dataclasses.replace(packet, collision_exclusions=exclusions)


def _mutate_anchor_weights_not_summing_to_one(packet):
    anchors = _replace_in(
        packet.anchors, "a_wall_point", barycentric_weights=np.array([0.4, 0.3, 0.2, 0.2], dtype=np.float64)
    )
    return dataclasses.replace(packet, anchors=anchors)


def _mutate_route_references_inactive_anchor(packet):
    anchors = _replace_in(packet.anchors, "a_jaw_point", active=False)
    return dataclasses.replace(packet, anchors=anchors)


def _mutate_mtu_missing_f_max(packet):
    mtus = _replace_in(packet.mtus, "mtu_main", f_max_N=None)
    return dataclasses.replace(packet, mtus=mtus)


def _mutate_mtu_missing_l_slack(packet):
    mtus = _replace_in(packet.mtus, "mtu_main", l_slack_m=None)
    return dataclasses.replace(packet, mtus=mtus)


def _mutate_ligament_missing_stiffness(packet):
    ligaments = _replace_in(packet.ligaments, "lig_main", stiffness_N_m=None)
    return dataclasses.replace(packet, ligaments=ligaments)


def _mutate_unsupported_capability(packet):
    return dataclasses.replace(packet, required_capabilities=packet.required_capabilities + ("time_travel",))


_INVALID_RECORD_CASES = {
    "duplicate_id": _mutate_duplicate_id,
    "unresolved_id": _mutate_unresolved_id,
    "non_positive_mass": _mutate_non_positive_mass,
    "inadmissible_inertia": _mutate_inadmissible_inertia,
    "non_positive_tet_volume": _mutate_non_positive_tet_volume,
    "boundary_not_outward": _mutate_boundary_not_outward,
    "lumen_not_closed": _mutate_lumen_not_closed,
    "contact_pair_without_one_material": _mutate_contact_pair_without_one_material,
    "exclusion_removes_required_feeding_pair": _mutate_exclusion_removes_required_pair,
    "anchor_weights_not_summing_to_one": _mutate_anchor_weights_not_summing_to_one,
    "route_references_inactive_anchor": _mutate_route_references_inactive_anchor,
    "mtu_missing_f_max_N": _mutate_mtu_missing_f_max,
    "mtu_missing_l_slack_m": _mutate_mtu_missing_l_slack,
    "ligament_missing_stiffness_N_m": _mutate_ligament_missing_stiffness,
    "unsupported_required_capability": _mutate_unsupported_capability,
}


def test_validate_accepts_the_fixture(fixture_packet):
    gs.avbd.validate_packet(fixture_packet)


def test_validate_accepts_a_concave_tissue_with_correctly_wound_faces(fixture_packet):
    """Pin the exact opposite-vertex orientation test against a shape a centroid heuristic gets wrong.

    The tissue is an L-tromino block: manifestly concave (a reflex dihedral at
    the missing corner), with every boundary face wound outward by the
    engine's own `VBDContact._oriented_outward`. A centroid-based "away from
    the mesh centroid" heuristic disagrees with that correct winding at the
    notch, and would reject this fixture; the exact per-tet test must not.
    """
    concave = build_concave_tissue()
    packet = dataclasses.replace(fixture_packet, tissues=fixture_packet.tissues + (concave,))
    gs.avbd.validate_packet(packet)


@pytest.mark.parametrize("case_name", list(_INVALID_RECORD_CASES))
def test_validate_rejects_each_invalid_record(fixture_packet, case_name):
    mutate = _INVALID_RECORD_CASES[case_name]
    bad_packet = mutate(fixture_packet)
    with pytest.raises(gs.GenesisException):
        gs.avbd.validate_packet(bad_packet)


########################## round trip ##########################


def _assert_records_equal(a, b):
    assert type(a) is type(b)
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            assert va.dtype == vb.dtype, f"{type(a).__name__}.{f.name} dtype changed"
            assert np.array_equal(va, vb), f"{type(a).__name__}.{f.name} value changed"
        else:
            assert va == vb, f"{type(a).__name__}.{f.name} value changed: {va!r} != {vb!r}"


def test_asymmetric_round_trip_is_bit_exact(tmp_path, fixture_packet):
    json_path, npz_path = _write(fixture_packet, tmp_path)
    loaded = gs.avbd.load_packet(json_path, npz_path)

    assert loaded.interface_version == fixture_packet.interface_version
    assert loaded.model_id == fixture_packet.model_id
    assert loaded.units == fixture_packet.units
    assert loaded.world_frame == fixture_packet.world_frame
    assert loaded.source_hashes == fixture_packet.source_hashes
    assert loaded.parameter_provenance == fixture_packet.parameter_provenance
    assert loaded.required_capabilities == fixture_packet.required_capabilities

    for collection_name in gs.avbd.packet._RECORD_CLASSES:
        original = getattr(fixture_packet, collection_name)
        round_tripped = getattr(loaded, collection_name)
        assert [r.id for r in round_tripped] == [r.id for r in original], f"{collection_name} order changed"
        for a, b in zip(original, round_tripped):
            _assert_records_equal(a, b)

    # No accidental symmetry: swapping the quaternion's x/y (or the tet's vertex
    # order) would silently pass an isotropic fixture. Assert the asymmetry
    # directly on the values that survived the round trip.
    jaw = next(l for l in loaded.links if l.id == "jaw")
    assert jaw.rest_quaternion_wxyz[1] != jaw.rest_quaternion_wxyz[2] != jaw.rest_quaternion_wxyz[3]
    tissue = loaded.tissues[0]
    assert not np.array_equal(tissue.tets[0], tissue.tets[0][::-1])

    gs.avbd.validate_packet(loaded)


def test_capabilities_is_a_frozenset(fixture_packet):
    assert isinstance(gs.avbd.CAPABILITIES, frozenset)
    assert gs.avbd.CAPABILITIES  # non-empty
    assert set(fixture_packet.required_capabilities) <= gs.avbd.CAPABILITIES
