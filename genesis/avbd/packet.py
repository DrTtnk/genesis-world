"""AVBD static model packet: load, validate and write.

This module reads and writes the static model packet described in
`AVBD_MODEL_INTERFACE.md` section 2. It is pure CPU code: numpy, json, `igl`
(for the exact boundary-orientation check, the same primitive
`genesis/engine/solvers/vbd_contact.py` uses) and the standard library. It
does not import a solver.

The packet is a tree of frozen dataclasses. Numeric arrays load as numpy
`float64` or `int64`, exactly as stored on disk. The module never casts a
dtype and never invents a default value for a missing field.
"""

import dataclasses
import json
import math

import igl

import numpy as np

import genesis as gs


INTERFACE_VERSION = "1"

#: Capability strings the engine can accept in a packet's `required_capabilities`
#: at this delivery stage. See AVBD_API_MAP.md section 8.
CAPABILITIES = frozenset(
    {
        "hard_point_attachment",
        "elastic_point_attachment",
        "fixed_base_chain",
        "free_link",
        "prescribed_link",
        "ellipsoid_collider",
        "contact_penalty_ccd",
        "isotropic_coulomb_friction",
        "per_env_reset",
    }
)


########################## record dataclasses ##########################


@dataclasses.dataclass(frozen=True)
class Entity:
    id: str
    link_ids: tuple[str, ...]
    tissue_ids: tuple[str, ...]
    role: str
    integration_owner: str


@dataclasses.dataclass(frozen=True)
class Link:
    id: str
    entity_id: str
    motion_mode: str  # "dynamic" | "fixed" | "prescribed"
    rest_position_m: np.ndarray  # [3]
    rest_quaternion_wxyz: np.ndarray  # [4]
    mass_kg: float
    com_link_m: np.ndarray  # [3]
    inertia_com_kgm2: np.ndarray  # [3,3], about the COM, in link axes
    collision_geometry_id: str | None
    collision_group: str | None


@dataclasses.dataclass(frozen=True)
class Joint:
    id: str
    parent_link_id: str
    child_link_id: str
    joint_type: str
    parent_frame_position_m: np.ndarray  # [3]
    parent_frame_quaternion_wxyz: np.ndarray  # [4]
    child_frame_position_m: np.ndarray  # [3]
    child_frame_quaternion_wxyz: np.ndarray  # [4]
    axes: tuple[str, ...]  # ordered, length Q
    coordinate_ids: tuple[str, ...]  # ordered, length Q, stable per-coordinate IDs
    rest_coordinates: np.ndarray  # [Q]
    lower_limits: np.ndarray  # [Q]
    upper_limits: np.ndarray  # [Q]
    damping: np.ndarray  # [Q]
    armature: np.ndarray  # [Q]
    parent_world_position_m: np.ndarray | None  # [3], roots only
    parent_world_quaternion_wxyz: np.ndarray | None  # [4], roots only


@dataclasses.dataclass(frozen=True)
class Material:
    id: str
    law: str
    law_params: dict
    density_kg_m3: float
    thickness_m: float | None
    provenance: str


@dataclasses.dataclass(frozen=True)
class Tissue:
    id: str
    rest_positions_m: np.ndarray  # [V,3]
    tets: np.ndarray  # [T,4] int, zero-based, positive-oriented
    boundary_faces: np.ndarray  # [F,3] int, zero-based, outward-oriented
    tet_material_ids: tuple[str, ...]  # length T
    collision_group: str | None
    stress_free_reference: str
    active: bool


@dataclasses.dataclass(frozen=True)
class Anchor:
    """A tagged union. `kind` selects which of the fields below are meaningful."""

    id: str
    active: bool
    kind: str  # "world" | "link" | "tissue"
    position_m: np.ndarray | None  # world
    link_id: str | None  # link
    position_link_m: np.ndarray | None  # link
    tissue_id: str | None  # tissue
    tet_index: int | None  # tissue
    barycentric_weights: np.ndarray | None  # tissue, [4]


@dataclasses.dataclass(frozen=True)
class Route:
    id: str
    anchor_ids: tuple[str, ...]  # ordered, including intermediate guides
    routing_policy: str
    mechanical_owner: str


@dataclasses.dataclass(frozen=True)
class MTU:
    id: str
    route_id: str
    law: str
    f_max_N: float | None
    l_opt_m: float
    l_slack_m: float | None
    v_max_m_s: float
    activation0: float
    fibre_length0_m: float
    law_constants: dict
    passive_mechanics_owner: str


@dataclasses.dataclass(frozen=True)
class Ligament:
    id: str
    route_id: str
    law: str
    slack_length_m: float
    rest_length_m: float
    stiffness_N_m: float | None
    damping_Ns_m: float | None


@dataclasses.dataclass(frozen=True)
class RotaryRestraint:
    id: str
    joint_coordinate_id: str
    rest_angle_rad: float
    law: str
    stiffness_Nm_rad: float
    damping: float | None


@dataclasses.dataclass(frozen=True)
class Attachment:
    id: str
    tissue_anchor_id: str
    other_anchor_id: str
    law: str  # "hard_point" | "elastic_point"
    stiffness_N_m: float | None
    damping_Ns_m: float | None
    collision_exclusion: bool


@dataclasses.dataclass(frozen=True)
class CollisionGroup:
    id: str
    member_ids: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CollisionExclusion:
    id: str
    group_a: str
    group_b: str
    reason: str


@dataclasses.dataclass(frozen=True)
class ContactMaterial:
    id: str
    normal_law: str
    normal_compliance_m_N: float  # zero for hard contact
    friction_law: str
    friction_coefficients: np.ndarray  # [k]
    regularization: dict
    restitution: float  # zero in the initial assisted-feeding model


@dataclasses.dataclass(frozen=True)
class ContactPair:
    id: str
    group_a: str
    group_b: str
    contact_material_ids: tuple[str, ...]  # resolved candidates; valid packets have exactly one
    required: bool  # True: this pair must stay active for feeding


@dataclasses.dataclass(frozen=True)
class PrescribedCollider:
    id: str
    link_id: str
    collision_geometry_id: str
    semiaxes_m: np.ndarray  # [3], radii along local axes
    collision_group: str
    trajectory_source_id: str


@dataclasses.dataclass(frozen=True)
class Region:
    id: str
    owner_ids: tuple[str, ...]  # link and/or tissue IDs
    geometry: dict
    purpose: str
    rest_to_current_mapping: str
    boundary_semantics: str


@dataclasses.dataclass(frozen=True)
class Packet:
    interface_version: str
    model_id: str
    units: dict
    world_frame: dict
    source_hashes: dict
    parameter_provenance: dict
    required_capabilities: tuple[str, ...]
    entities: tuple[Entity, ...]
    links: tuple[Link, ...]
    joints: tuple[Joint, ...]
    materials: tuple[Material, ...]
    tissues: tuple[Tissue, ...]
    anchors: tuple[Anchor, ...]
    routes: tuple[Route, ...]
    mtus: tuple[MTU, ...]
    ligaments: tuple[Ligament, ...]
    rotary_restraints: tuple[RotaryRestraint, ...]
    attachments: tuple[Attachment, ...]
    collision_groups: tuple[CollisionGroup, ...]
    collision_exclusions: tuple[CollisionExclusion, ...]
    contact_materials: tuple[ContactMaterial, ...]
    contact_pairs: tuple[ContactPair, ...]
    prescribed_colliders: tuple[PrescribedCollider, ...]
    regions: tuple[Region, ...]


_RECORD_CLASSES = {
    "entities": Entity,
    "links": Link,
    "joints": Joint,
    "materials": Material,
    "tissues": Tissue,
    "anchors": Anchor,
    "routes": Route,
    "mtus": MTU,
    "ligaments": Ligament,
    "rotary_restraints": RotaryRestraint,
    "attachments": Attachment,
    "collision_groups": CollisionGroup,
    "collision_exclusions": CollisionExclusion,
    "contact_materials": ContactMaterial,
    "contact_pairs": ContactPair,
    "prescribed_colliders": PrescribedCollider,
    "regions": Region,
}

# Array-valued fields: dtype, expected shape (-1 is a wildcard dimension) and whether
# the field is allowed to be absent (JSON null, no array).
_ARRAY_FIELDS = {
    "Link": {
        "rest_position_m": (np.float64, (3,), True),
        "rest_quaternion_wxyz": (np.float64, (4,), True),
        "com_link_m": (np.float64, (3,), True),
        "inertia_com_kgm2": (np.float64, (3, 3), True),
    },
    "Joint": {
        "parent_frame_position_m": (np.float64, (3,), True),
        "parent_frame_quaternion_wxyz": (np.float64, (4,), True),
        "child_frame_position_m": (np.float64, (3,), True),
        "child_frame_quaternion_wxyz": (np.float64, (4,), True),
        "rest_coordinates": (np.float64, (-1,), True),
        "lower_limits": (np.float64, (-1,), True),
        "upper_limits": (np.float64, (-1,), True),
        "damping": (np.float64, (-1,), True),
        "armature": (np.float64, (-1,), True),
        "parent_world_position_m": (np.float64, (3,), False),
        "parent_world_quaternion_wxyz": (np.float64, (4,), False),
    },
    "Tissue": {
        "rest_positions_m": (np.float64, (-1, 3), True),
        "tets": (np.int64, (-1, 4), True),
        "boundary_faces": (np.int64, (-1, 3), True),
    },
    "Anchor": {
        "position_m": (np.float64, (3,), False),
        "position_link_m": (np.float64, (3,), False),
        "barycentric_weights": (np.float64, (4,), False),
    },
    "ContactMaterial": {
        "friction_coefficients": (np.float64, (-1,), True),
    },
    "PrescribedCollider": {
        "semiaxes_m": (np.float64, (3,), True),
    },
}

# List-valued fields that round-trip as a Python tuple.
_TUPLE_FIELDS = {
    "Entity": {"link_ids", "tissue_ids"},
    "Joint": {"axes", "coordinate_ids"},
    "Tissue": {"tet_material_ids"},
    "Anchor": set(),
    "Route": {"anchor_ids"},
    "CollisionGroup": {"member_ids"},
    "ContactPair": {"contact_material_ids"},
    "Region": {"owner_ids"},
}


########################## load ##########################


def _resolve_array(cls_name, rec_id, field_name, npz_key, npz, dtype, shape):
    if npz_key not in npz:
        raise gs.GenesisException(f"{cls_name} '{rec_id}' field '{field_name}' references missing array '{npz_key}'.")
    arr = npz[npz_key]
    if arr.dtype != dtype:
        raise gs.GenesisException(
            f"{cls_name} '{rec_id}' field '{field_name}' has dtype {arr.dtype}, expected {np.dtype(dtype)}."
        )
    if len(shape) != arr.ndim or any(s != -1 and s != arr.shape[i] for i, s in enumerate(shape)):
        raise gs.GenesisException(f"{cls_name} '{rec_id}' field '{field_name}' has shape {arr.shape}, expected {shape}.")
    if not np.isfinite(arr).all():
        raise gs.GenesisException(f"{cls_name} '{rec_id}' field '{field_name}' contains non-finite values.")
    return arr


def _record_from_dict(cls, data, npz):
    cls_name = cls.__name__
    rec_id = data.get("id", "?")
    array_specs = _ARRAY_FIELDS.get(cls_name, {})
    tuple_fields = _TUPLE_FIELDS.get(cls_name, set())
    kwargs = {}
    for f in dataclasses.fields(cls):
        name = f.name
        if name not in data:
            raise gs.GenesisException(f"{cls_name} '{rec_id}' is missing required field '{name}'.")
        raw = data[name]
        if name in array_specs:
            dtype, shape, required = array_specs[name]
            if raw is None:
                if required:
                    raise gs.GenesisException(f"{cls_name} '{rec_id}' is missing required field '{name}'.")
                kwargs[name] = None
            else:
                kwargs[name] = _resolve_array(cls_name, rec_id, name, raw, npz, dtype, shape)
        elif name in tuple_fields:
            kwargs[name] = tuple(raw)
        else:
            if isinstance(raw, float) and not math.isfinite(raw):
                raise gs.GenesisException(f"{cls_name} '{rec_id}' field '{name}' contains a non-finite value.")
            kwargs[name] = raw
    return cls(**kwargs)


def load_packet(json_path, npz_path) -> Packet:
    """Load a static model packet from a JSON metadata file plus an NPZ array file.

    Every field declared on `Packet` and on each record dataclass must be present
    in the JSON document (a nullable field may hold JSON `null`). An array field
    holds the NPZ key as a string. Raise `GenesisException` on an unknown
    `interface_version`, a missing collection, a missing field, a missing array,
    a wrong shape/dtype, or a non-finite value. Every message names the record
    `id` and the field.
    """
    with open(json_path) as fh:
        doc = json.load(fh)
    npz = np.load(npz_path)

    for f in dataclasses.fields(Packet):
        if f.name not in doc:
            raise gs.GenesisException(f"Packet is missing required field '{f.name}'.")

    if doc["interface_version"] != INTERFACE_VERSION:
        raise gs.GenesisException(
            f"Unsupported interface_version '{doc['interface_version']}', expected '{INTERFACE_VERSION}'."
        )

    kwargs = {
        "interface_version": doc["interface_version"],
        "model_id": doc["model_id"],
        "units": doc["units"],
        "world_frame": doc["world_frame"],
        "source_hashes": doc["source_hashes"],
        "parameter_provenance": doc["parameter_provenance"],
        "required_capabilities": tuple(doc["required_capabilities"]),
    }
    for name, cls in _RECORD_CLASSES.items():
        kwargs[name] = tuple(_record_from_dict(cls, rec, npz) for rec in doc[name])

    return Packet(**kwargs)


########################## write ##########################


def _record_to_dict(record, npz_arrays):
    cls_name = type(record).__name__
    array_specs = _ARRAY_FIELDS.get(cls_name, {})
    tuple_fields = _TUPLE_FIELDS.get(cls_name, set())
    rec_id = getattr(record, "id")
    out = {}
    for f in dataclasses.fields(record):
        name = f.name
        value = getattr(record, name)
        if name in array_specs:
            if value is None:
                out[name] = None
            else:
                key = f"{cls_name.lower()}.{rec_id}.{name}"
                npz_arrays[key] = value
                out[name] = key
        elif name in tuple_fields:
            out[name] = list(value)
        else:
            out[name] = value
    return out


def write_packet(packet: Packet, json_path, npz_path) -> None:
    """Write a static model packet back to a JSON metadata file plus an NPZ array file.

    Arrays are written with their stored dtype: no cast. Round-tripping the
    result through `load_packet` reproduces every field exactly.
    """
    npz_arrays = {}
    doc = {}
    for f in dataclasses.fields(Packet):
        name = f.name
        value = getattr(packet, name)
        if name in _RECORD_CLASSES:
            doc[name] = [_record_to_dict(rec, npz_arrays) for rec in value]
        elif name == "required_capabilities":
            doc[name] = list(value)
        else:
            doc[name] = value

    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    np.savez(npz_path, **npz_arrays)


########################## validate ##########################


def _tet_volume(pts, tet):
    a, b, c, d = pts[tet]
    return float(np.dot(np.cross(b - a, c - a), d - a) / 6.0)


def _check_unique_ids(packet):
    seen = {}
    for coll_name in _RECORD_CLASSES:
        for rec in getattr(packet, coll_name):
            if rec.id in seen:
                raise gs.GenesisException(f"Duplicate id '{rec.id}': used by both '{seen[rec.id]}' and '{coll_name}'.")
            seen[rec.id] = coll_name


def _check_capabilities(packet):
    unknown = set(packet.required_capabilities) - CAPABILITIES
    if unknown:
        raise gs.GenesisException(f"Packet requires unsupported capabilities: {sorted(unknown)}.")


def _check_links(packet):
    for link in packet.links:
        if link.motion_mode != "dynamic":
            continue
        if not (link.mass_kg > 0):
            raise gs.GenesisException(f"Link '{link.id}' has non-positive mass {link.mass_kg} kg.")
        inertia = link.inertia_com_kgm2
        if not np.allclose(inertia, inertia.T, atol=1e-9):
            raise gs.GenesisException(f"Link '{link.id}' inertia_com_kgm2 is not symmetric.")
        eigvals = np.linalg.eigvalsh(inertia)
        if eigvals.min() <= 0:
            raise gs.GenesisException(f"Link '{link.id}' inertia_com_kgm2 is not positive definite.")
        ix, iy, iz = eigvals
        if ix + iy < iz - 1e-9 or iy + iz < ix - 1e-9 or iz + ix < iy - 1e-9:
            raise gs.GenesisException(f"Link '{link.id}' inertia_com_kgm2 violates the triangle inequality.")


def _check_boundary_outward(tissue, pts):
    """Check that each boundary face is wound so its normal points away from its own tetrahedron.

    This is the exact test, not a heuristic: it uses the real owning tet of
    each face (via `igl.boundary_facets`), the same primitive
    `VBDContact._oriented_outward` uses in `genesis/engine/solvers/vbd_contact.py`.
    A face is outward-facing when its normal points away from the one tet
    vertex that is not on the face. This holds for a concave solid as well as
    a convex one; a centroid-based check does not.
    """
    facets, tet_of_facet, _ = igl.boundary_facets(tissue.tets)
    owner_by_face = {frozenset(face.tolist()): tet_idx for face, tet_idx in zip(facets, tet_of_facet)}
    for i, face in enumerate(tissue.boundary_faces):
        tet_idx = owner_by_face.get(frozenset(face.tolist()))
        if tet_idx is None:
            raise gs.GenesisException(f"Tissue '{tissue.id}' boundary face {i} is not a boundary face of its tets.")
        tet = tissue.tets[tet_idx]
        opposite = next(v for v in tet if v not in face)
        a, b, c = pts[face[0]], pts[face[1]], pts[face[2]]
        normal = np.cross(b - a, c - a)
        if np.dot(normal, pts[opposite] - a) > 0.0:
            raise gs.GenesisException(f"Tissue '{tissue.id}' boundary face {i} is not outward-oriented.")


def _check_tissues(packet):
    for tissue in packet.tissues:
        pts = tissue.rest_positions_m
        for i, tet in enumerate(tissue.tets):
            volume = _tet_volume(pts, tet)
            if volume <= 0:
                raise gs.GenesisException(f"Tissue '{tissue.id}' tet {i} has non-positive volume {volume}.")

        _check_boundary_outward(tissue, pts)

        edge_count = {}
        for face in tissue.boundary_faces:
            for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = (u, v) if u < v else (v, u)
                edge_count[key] = edge_count.get(key, 0) + 1
        for edge, count in edge_count.items():
            if count != 2:
                raise gs.GenesisException(
                    f"Tissue '{tissue.id}' boundary is not a closed lumen: edge {edge} touches {count} faces."
                )


def _check_anchors(packet):
    for anchor in packet.anchors:
        if anchor.kind == "tissue":
            weights = anchor.barycentric_weights
            total = None if weights is None else float(weights.sum())
            if weights is None or not math.isclose(total, 1.0, abs_tol=1e-9):
                raise gs.GenesisException(f"Anchor '{anchor.id}' barycentric_weights sum to {total}, expected 1.")


def _check_routes(packet):
    anchors_by_id = {a.id: a for a in packet.anchors}
    for route in packet.routes:
        for anchor_id in route.anchor_ids:
            anchor = anchors_by_id.get(anchor_id)
            if anchor is None:
                raise gs.GenesisException(f"Route '{route.id}' references unresolved anchor id '{anchor_id}'.")
            if not anchor.active:
                raise gs.GenesisException(f"Route '{route.id}' references inactive anchor '{anchor_id}'.")


def _check_mtus(packet):
    for mtu in packet.mtus:
        if mtu.f_max_N is None:
            raise gs.GenesisException(f"MTU '{mtu.id}' is missing 'f_max_N'.")
        if mtu.l_slack_m is None:
            raise gs.GenesisException(f"MTU '{mtu.id}' is missing 'l_slack_m'.")


def _check_ligaments(packet):
    for ligament in packet.ligaments:
        if ligament.stiffness_N_m is None:
            raise gs.GenesisException(f"Ligament '{ligament.id}' is missing 'stiffness_N_m'.")


def _check_contact_pairs(packet):
    group_ids = {g.id for g in packet.collision_groups}
    material_ids = {m.id for m in packet.contact_materials}
    for pair in packet.contact_pairs:
        if pair.group_a not in group_ids:
            raise gs.GenesisException(f"Contact pair '{pair.id}' references unresolved collision group '{pair.group_a}'.")
        if pair.group_b not in group_ids:
            raise gs.GenesisException(f"Contact pair '{pair.id}' references unresolved collision group '{pair.group_b}'.")
        if len(pair.contact_material_ids) != 1:
            raise gs.GenesisException(
                f"Contact pair '{pair.id}' resolves to {len(pair.contact_material_ids)} contact materials, expected exactly 1."
            )
        (material_id,) = pair.contact_material_ids
        if material_id not in material_ids:
            raise gs.GenesisException(f"Contact pair '{pair.id}' references unresolved contact material '{material_id}'.")


def _check_collision_exclusions(packet):
    group_ids = {g.id for g in packet.collision_groups}
    required_pairs = {frozenset((p.group_a, p.group_b)) for p in packet.contact_pairs if p.required}
    for exclusion in packet.collision_exclusions:
        if exclusion.group_a not in group_ids or exclusion.group_b not in group_ids:
            raise gs.GenesisException(f"Collision exclusion '{exclusion.id}' references an unresolved collision group.")
        if frozenset((exclusion.group_a, exclusion.group_b)) in required_pairs:
            raise gs.GenesisException(
                f"Collision exclusion '{exclusion.id}' disables the required feeding pair "
                f"({exclusion.group_a}, {exclusion.group_b})."
            )


def _check_referential_integrity(packet):
    link_ids = {l.id for l in packet.links}
    tissue_ids = {t.id for t in packet.tissues}
    anchor_ids = {a.id for a in packet.anchors}
    route_ids = {r.id for r in packet.routes}
    coordinate_ids = {cid for joint in packet.joints for cid in joint.coordinate_ids}

    for entity in packet.entities:
        for link_id in entity.link_ids:
            if link_id not in link_ids:
                raise gs.GenesisException(f"Entity '{entity.id}' references unresolved link id '{link_id}'.")
        for tissue_id in entity.tissue_ids:
            if tissue_id not in tissue_ids:
                raise gs.GenesisException(f"Entity '{entity.id}' references unresolved tissue id '{tissue_id}'.")

    for joint in packet.joints:
        if joint.parent_link_id not in link_ids:
            raise gs.GenesisException(f"Joint '{joint.id}' references unresolved parent link id '{joint.parent_link_id}'.")
        if joint.child_link_id not in link_ids:
            raise gs.GenesisException(f"Joint '{joint.id}' references unresolved child link id '{joint.child_link_id}'.")

    for attachment in packet.attachments:
        if attachment.tissue_anchor_id not in anchor_ids:
            raise gs.GenesisException(
                f"Attachment '{attachment.id}' references unresolved anchor id '{attachment.tissue_anchor_id}'."
            )
        if attachment.other_anchor_id not in anchor_ids:
            raise gs.GenesisException(
                f"Attachment '{attachment.id}' references unresolved anchor id '{attachment.other_anchor_id}'."
            )

    for mtu in packet.mtus:
        if mtu.route_id not in route_ids:
            raise gs.GenesisException(f"MTU '{mtu.id}' references unresolved route id '{mtu.route_id}'.")

    for ligament in packet.ligaments:
        if ligament.route_id not in route_ids:
            raise gs.GenesisException(f"Ligament '{ligament.id}' references unresolved route id '{ligament.route_id}'.")

    for restraint in packet.rotary_restraints:
        if restraint.joint_coordinate_id not in coordinate_ids:
            raise gs.GenesisException(
                f"Rotary restraint '{restraint.id}' references unresolved joint coordinate id "
                f"'{restraint.joint_coordinate_id}'."
            )

    for collider in packet.prescribed_colliders:
        if collider.link_id not in link_ids:
            raise gs.GenesisException(f"Prescribed collider '{collider.id}' references unresolved link id '{collider.link_id}'.")

    for region in packet.regions:
        for owner_id in region.owner_ids:
            if owner_id not in link_ids and owner_id not in tissue_ids:
                raise gs.GenesisException(f"Region '{region.id}' references unresolved owner id '{owner_id}'.")


def _check_finite(packet):
    for coll_name, cls in _RECORD_CLASSES.items():
        specs = _ARRAY_FIELDS.get(cls.__name__, {})
        for rec in getattr(packet, coll_name):
            for field_name in specs:
                arr = getattr(rec, field_name)
                if arr is not None and not np.isfinite(arr).all():
                    raise gs.GenesisException(f"{cls.__name__} '{rec.id}' field '{field_name}' contains non-finite values.")


def validate_packet(packet: Packet) -> None:
    """Validate a loaded packet against the section 6 checklist. Raise on the first violation.

    Checks: unique/resolved IDs, positive dynamic mass and admissible inertia,
    positive tet volume, outward boundary faces, a closed lumen, contact pairs
    with exactly one resolved material, exclusions that would disable a required
    feeding pair, anchor weights summing to one, routes referencing only active
    anchors, MTUs/ligaments carrying their required force/length/stiffness, and
    `required_capabilities` being a subset of `gs.avbd.CAPABILITIES`.
    """
    _check_unique_ids(packet)
    _check_capabilities(packet)
    _check_links(packet)
    _check_tissues(packet)
    _check_anchors(packet)
    _check_routes(packet)
    _check_mtus(packet)
    _check_ligaments(packet)
    _check_contact_pairs(packet)
    _check_collision_exclusions(packet)
    _check_referential_integrity(packet)
    _check_finite(packet)
