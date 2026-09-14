"""AVBD: model packet import and model build for the coupled rigid/tissue/muscle solver.

This package covers gate A0 (reading, validating and writing the static model packet described in
`AVBD_MODEL_INTERFACE.md` section 2) and the model build of `AVBD_API_MAP.md` section 2:
`build_model` turns a validated packet into a live Genesis scene.
"""

from .model import AVBDModel, IdMaps, Pose, build_model
from .packet import (
    CAPABILITIES,
    INTERFACE_VERSION,
    Anchor,
    Attachment,
    CollisionExclusion,
    CollisionGroup,
    ContactMaterial,
    ContactPair,
    Entity,
    Joint,
    Ligament,
    Link,
    Material,
    MTU,
    Packet,
    PrescribedCollider,
    Region,
    RotaryRestraint,
    Route,
    Tissue,
    load_packet,
    validate_packet,
    write_packet,
)

__all__ = [
    "AVBDModel",
    "IdMaps",
    "Pose",
    "build_model",
    "CAPABILITIES",
    "INTERFACE_VERSION",
    "Anchor",
    "Attachment",
    "CollisionExclusion",
    "CollisionGroup",
    "ContactMaterial",
    "ContactPair",
    "Entity",
    "Joint",
    "Ligament",
    "Link",
    "Material",
    "MTU",
    "Packet",
    "PrescribedCollider",
    "Region",
    "RotaryRestraint",
    "Route",
    "Tissue",
    "load_packet",
    "validate_packet",
    "write_packet",
]
