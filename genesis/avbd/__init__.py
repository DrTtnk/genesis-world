"""AVBD: model packet import for the coupled rigid/tissue/muscle solver.

This package currently covers gate A0 only: reading, validating and writing the
static model packet described in `AVBD_MODEL_INTERFACE.md` section 2. It does
not build a runtime model; `build_model` is a later gate.
"""

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
