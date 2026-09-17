"""AVBD model build: turn a validated static packet into a live Genesis scene.

`build_model` is a thin CPU assembler. It creates rigid links and joints as inline MJCF, tissue as a
`TetMesh` entity with a `VBD.Base` material, node attachments, routed Hill MTUs, ligaments, rotary
restraints, and contact rules and prescribed colliders, then returns an `AVBDModel` handle with immutable
ID maps and thin forwarding wrappers.

This module implements the subset of `AVBD_API_MAP.md` section 2 the engine supports today:

- Rigid links and joints, as a fixed base and a single-axis hinge chain, or one free link, or a fixed
  entity registered as a prescribed collider.
- Tissue as an explicit `TetMesh` plus one `VBD.Base` material per tissue (one material per tissue, not
  one per tet).
- `hard_point` attachments from one tissue vertex (a node anchor: one barycentric weight of 1) to a link, and
  `hard_point` attachments between two tissue anchors (barycentric points of the same or different tissues).
- Routes, Hill MTUs, ligaments and rotary restraints, with world, link and tissue anchors.
- Collision groups, contact rules and prescribed ellipsoid colliders.

Every other record class in the packet schema is rejected at build, by name: barycentric attachments to a
link, the `elastic_point` law, more than one material in one tissue, a free-root
articulated chain, regions, and any capability the engine does not advertise.
"""

import dataclasses
import types
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.vbd_mtu import HillParameters, LinkAnchor, TissueAnchor, WorldAnchor

@dataclasses.dataclass(frozen=True)
class Pose:
    """A rigid placement of a whole packet in the scene: world position (m) and orientation (w, x, y, z)."""

    position_m: tuple
    quaternion_wxyz: tuple


@dataclasses.dataclass(frozen=True)
class IdMaps:
    """Stable ID-to-handle maps, fixed once `build_model` returns.

    `links` and `tissues` map a packet ID to the engine handle (a `RigidLink` or a `VBDEntity`) built for
    it. `coordinates`, `mtus`, `ligaments`, `rotary_restraints` and `colliders` map a packet ID to the
    integer index the corresponding solver call uses (a DOF index, an MTU/ligament/restraint index, or a
    prescribed-collider index in `AVBDModel.set_prescribed_targets` / `get_reactions` order).
    """

    links: types.MappingProxyType
    tissues: types.MappingProxyType
    coordinates: types.MappingProxyType
    mtus: types.MappingProxyType
    ligaments: types.MappingProxyType
    rotary_restraints: types.MappingProxyType
    colliders: types.MappingProxyType


class AVBDModel:
    """Handle returned by `build_model`: immutable ID maps plus thin forwards to the solver."""

    def __init__(self, scene, ids: IdMaps):
        self._scene = scene
        self._ids = ids

    @property
    def ids(self) -> IdMaps:
        return self._ids

    def set_excitation(self, u):
        """Forward to `VBDSolver.set_excitation`. `u` is `[B, M]` or `[M]`, in MTU declaration order."""
        self._scene.vbd_solver.set_excitation(u)

    def set_prescribed_targets(self, pos, quat):
        """Forward to `VBDSolver.set_prescribed_targets`, in prescribed-collider declaration order."""
        self._scene.vbd_solver.set_prescribed_targets(pos, quat)

    def mtu_state(self):
        """Forward to `VBDSolver.mtu_state`."""
        return self._scene.vbd_solver.mtu_state()

    def get_reactions(self):
        """Forward to `VBDSolver.collider_reactions`."""
        return self._scene.vbd_solver.collider_reactions()

    def get_link_poses(self):
        """Position and quaternion of every mapped link, as `{id: (pos, quat)}`.

        The API map asks for one stacked `[B, L, 3]` / `[B, L, 4]` pair; a built model may spread its
        links over several `RigidEntity` objects (one per kinematic tree and one per prescribed
        collider), so this wrapper reports per link instead of trying to stack them into one tensor.
        """
        return {link_id: (link.get_pos(), link.get_quat()) for link_id, link in self._ids.links.items()}


def _point_str(v):
    return f"{v[0]} {v[1]} {v[2]}"


def _quat_str(q):
    return f"{q[0]} {q[1]} {q[2]} {q[3]}"


def _apply_placement_pose(pos, quat, placement):
    if placement is None:
        return pos, quat
    return gu.transform_pos_quat_by_trans_quat(
        pos, quat, np.asarray(placement.position_m, dtype=np.float64), np.asarray(placement.quaternion_wxyz, dtype=np.float64)
    )


def _apply_placement_point(pos, placement):
    if placement is None:
        return pos
    return gu.transform_by_trans_quat(
        pos, np.asarray(placement.position_m, dtype=np.float64), np.asarray(placement.quaternion_wxyz, dtype=np.float64)
    )


def _child_body_pose(joint):
    """The child body's pose in its parent body's frame at rest: `parent_frame` composed with the
    inverse of `child_frame`, so that at `q == rest_coordinates` the two joint frames coincide in W."""
    child_pos = np.asarray(joint.child_frame_position_m, dtype=np.float64)
    child_quat = np.asarray(joint.child_frame_quaternion_wxyz, dtype=np.float64)
    inv_quat = gu.inv_quat(child_quat)
    inv_pos = gu.transform_by_quat(-child_pos, inv_quat)
    return gu.transform_pos_quat_by_trans_quat(
        inv_pos,
        inv_quat,
        np.asarray(joint.parent_frame_position_m, dtype=np.float64),
        np.asarray(joint.parent_frame_quaternion_wxyz, dtype=np.float64),
    )


def _fullinertia_str(inertia):
    ixx, iyy, izz = inertia[0, 0], inertia[1, 1], inertia[2, 2]
    ixy, ixz, iyz = inertia[0, 1], inertia[0, 2], inertia[1, 2]
    return f"{ixx} {iyy} {izz} {ixy} {ixz} {iyz}"


def _add_body(parent_elem, link, pos, quat, joint):
    """Append one MJCF `<body>` for `link` at (`pos`, `quat`) relative to `parent_elem`, with its
    `<inertial>` and, if `joint` is given, its single-axis `<joint>`. Returns the new `<body>` element."""
    body = ET.SubElement(parent_elem, "body", name=link.id, pos=_point_str(pos), quat=_quat_str(quat))
    ET.SubElement(
        body,
        "inertial",
        pos=_point_str(link.com_link_m),
        mass=str(float(link.mass_kg)),
        fullinertia=_fullinertia_str(link.inertia_com_kgm2),
    )
    if joint is not None:
        (coordinate_id,) = joint.coordinate_ids
        axis = gu.transform_by_quat(np.array([0.0, 1.0, 0.0]), np.asarray(joint.child_frame_quaternion_wxyz, dtype=np.float64))
        ET.SubElement(
            body,
            "joint",
            name=coordinate_id,
            type="hinge",
            axis=_point_str(axis),
            pos=_point_str(joint.child_frame_position_m),
            range=f"{float(joint.lower_limits[0])} {float(joint.upper_limits[0])}",
            damping=str(float(joint.damping[0])),
            armature=str(float(joint.armature[0])),
            ref=str(float(joint.rest_coordinates[0])),
        )
    return body


def _mjcf_entity(scene, build_body):
    """Build one inline-MJCF entity. `build_body(world)` appends the single root `<body>` (and everything
    nested under it) to the `<worldbody>` element; its own pose and contents are already bound by its caller."""
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    build_body(world)
    return scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(root, encoding="unicode")), material=gs.materials.Rigid())


def _build_rigid_trees(scene, packet, placement):
    """Build one inline-MJCF entity per fixed-base chain or free-standing link, and one per prescribed
    collider. Returns `(links_by_id, coordinates)`: link id to `RigidLink`, joint coordinate id to global
    DOF index."""
    links_by_packet_id = {link.id: link for link in packet.links}
    joint_by_child = {}
    children_by_parent = {link.id: [] for link in packet.links}
    for joint in packet.joints:
        if joint.joint_type != "hinge_chain":
            gs.raise_exception(f"Joint '{joint.id}' has joint_type '{joint.joint_type}'; only 'hinge_chain' is supported.")
        if len(joint.axes) != 1:
            gs.raise_exception(
                f"Joint '{joint.id}' has {len(joint.axes)} axes; only a single-axis hinge joint is supported "
                "(a serial multi-axis hinge chain must be exported as separate single-axis joints)."
            )
        if joint.child_link_id in joint_by_child:
            gs.raise_exception(f"Link '{joint.child_link_id}' is the child of more than one joint.")
        child_link = links_by_packet_id[joint.child_link_id]
        if child_link.motion_mode != "dynamic":
            gs.raise_exception(f"Joint '{joint.id}' targets child link '{joint.child_link_id}', which is not 'dynamic'.")
        joint_by_child[joint.child_link_id] = joint
        children_by_parent[joint.parent_link_id].append(joint.child_link_id)

    prescribed_collider_by_link = {pc.link_id: pc for pc in packet.prescribed_colliders}
    for link in packet.links:
        if link.motion_mode == "prescribed" and link.id not in prescribed_collider_by_link:
            gs.raise_exception(f"Prescribed link '{link.id}' has no matching prescribed_collider record.")
        if link.motion_mode != "prescribed" and link.collision_group is not None:
            # The packet names a collision group for this link, but carries no shape for `add_rigid_collider`
            # to use: only a prescribed collider's `semiaxes_m` gives build_model a concrete geometry today.
            gs.raise_exception(
                f"Link '{link.id}' has collision_group '{link.collision_group}' set, but build_model has no "
                "collision geometry for a non-prescribed link: the packet schema does not carry a shape for it."
            )

    roots = [link for link in packet.links if link.id not in joint_by_child]

    links_by_id = {}
    coordinates = {}
    prescribed_specs = []  # (link, prescribed_collider), built after rigid trees so contact groups exist

    def build_children(parent_elem, parent_link_id, parent_pos, parent_quat):
        for child_id in children_by_parent[parent_link_id]:
            joint = joint_by_child[child_id]
            child_link = links_by_packet_id[child_id]
            child_pos, child_quat = _child_body_pose(joint)
            body = _add_body(parent_elem, child_link, child_pos, child_quat, joint)
            build_children(body, child_id, child_pos, child_quat)

    for root in roots:
        if root.motion_mode == "prescribed":
            if children_by_parent[root.id]:
                gs.raise_exception(f"Prescribed link '{root.id}' has attached joints, which is not supported.")
            prescribed_specs.append((root, prescribed_collider_by_link[root.id]))
            continue

        if root.motion_mode == "dynamic" and children_by_parent[root.id]:
            gs.raise_exception(
                f"Link '{root.id}' is a dynamic root with attached joints: a free-root articulated chain "
                "is not supported."
            )

        pos, quat = _apply_placement_pose(
            np.asarray(root.rest_position_m, dtype=np.float64), np.asarray(root.rest_quaternion_wxyz, dtype=np.float64), placement
        )

        def build(world, root=root, pos=pos, quat=quat):
            body = ET.SubElement(world, "body", name=root.id, pos=_point_str(pos), quat=_quat_str(quat))
            ET.SubElement(
                body,
                "inertial",
                pos=_point_str(root.com_link_m),
                mass=str(float(root.mass_kg)),
                fullinertia=_fullinertia_str(root.inertia_com_kgm2),
            )
            if root.motion_mode == "dynamic":  # a leaf dynamic root: one free link
                ET.SubElement(body, "freejoint")
            build_children(body, root.id, pos, quat)

        entity = _mjcf_entity(scene, build)
        for link in entity.links:
            if link.name in links_by_packet_id:
                links_by_id[link.name] = link
                joint = joint_by_child.get(link.name)
                if joint is not None:
                    (coordinate_id,) = joint.coordinate_ids
                    coordinates[coordinate_id] = link.dof_start

    for root, collider in prescribed_specs:
        pos, quat = _apply_placement_pose(
            np.asarray(root.rest_position_m, dtype=np.float64), np.asarray(root.rest_quaternion_wxyz, dtype=np.float64), placement
        )

        def build(world, root=root, pos=pos, quat=quat, collider=collider):
            body = ET.SubElement(world, "body", name=root.id, pos=_point_str(pos), quat=_quat_str(quat))
            semiaxes = np.asarray(collider.semiaxes_m, dtype=np.float64)
            ET.SubElement(body, "geom", type="ellipsoid", size=_point_str(semiaxes))

        entity = _mjcf_entity(scene, build)
        for link in entity.links:
            if link.name == root.id:
                links_by_id[link.name] = link

    return links_by_id, coordinates, prescribed_specs


def _resolve_anchor(anchor, links_by_id, tissues_by_id, tets_by_tissue_id, placement):
    if anchor.kind == "world":
        return WorldAnchor(tuple(_apply_placement_point(np.asarray(anchor.position_m, dtype=np.float64), placement).tolist()))
    if anchor.kind == "link":
        return LinkAnchor(links_by_id[anchor.link_id], tuple(np.asarray(anchor.position_link_m, dtype=np.float64).tolist()))
    if anchor.kind == "tissue":
        entity = tissues_by_id[anchor.tissue_id]
        vertices = tuple(int(v) for v in tets_by_tissue_id[anchor.tissue_id][anchor.tet_index])
        weights = tuple(float(w) for w in anchor.barycentric_weights)
        return TissueAnchor(entity, vertices, weights)
    gs.raise_exception(f"Anchor '{anchor.id}' has unknown kind '{anchor.kind}'.")


def _build_tissues(scene, packet, group_index, placement):
    materials_by_id = {m.id: m for m in packet.materials}
    tissues_by_id = {}
    tets_by_tissue_id = {}
    for tissue in packet.tissues:
        if not tissue.active:
            continue
        distinct_materials = set(tissue.tet_material_ids)
        if len(distinct_materials) != 1:
            gs.raise_exception(
                f"Tissue '{tissue.id}' references {len(distinct_materials)} distinct materials; only one "
                "material per tissue is supported (per-tet regions are not)."
            )
        (material_id,) = distinct_materials
        material = materials_by_id[material_id]
        if material.law != "neo_hookean_v1":
            gs.raise_exception(f"Tissue '{tissue.id}' material '{material_id}' has law '{material.law}'; only 'neo_hookean_v1' is supported.")
        law_params = material.law_params
        if "E_Pa" not in law_params or "nu" not in law_params:
            gs.raise_exception(f"Material '{material_id}' law_params is missing 'E_Pa' or 'nu'.")

        verts = _apply_placement_point(np.asarray(tissue.rest_positions_m, dtype=np.float64), placement)
        morph = gs.morphs.TetMesh(verts=verts, elems=np.asarray(tissue.tets), faces=np.asarray(tissue.boundary_faces))
        collision_group = group_index[tissue.collision_group] if tissue.collision_group is not None else 0
        vbd_material = gs.materials.VBD.Base(
            E=float(law_params["E_Pa"]), nu=float(law_params["nu"]), rho=float(material.density_kg_m3), collision_group=collision_group
        )
        tissues_by_id[tissue.id] = scene.add_entity(morph=morph, material=vbd_material)
        tets_by_tissue_id[tissue.id] = tissue.tets
    return tissues_by_id, tets_by_tissue_id


def _build_attachments(packet, anchors_by_id, tissues_by_id, tets_by_tissue_id, links_by_id):
    for attachment in packet.attachments:
        tissue_anchor = anchors_by_id[attachment.tissue_anchor_id]
        other_anchor = anchors_by_id[attachment.other_anchor_id]
        if tissue_anchor.kind != "tissue":
            gs.raise_exception(f"Attachment '{attachment.id}' tissue_anchor_id does not reference a tissue anchor.")
        if other_anchor.kind not in ("link", "tissue"):
            gs.raise_exception(f"Attachment '{attachment.id}' targets a '{other_anchor.kind}' anchor; only a link or tissue target is supported.")
        if attachment.law == "elastic_point":
            gs.raise_exception(f"Attachment '{attachment.id}' uses law 'elastic_point', which is not supported; only 'hard_point' is.")
        if attachment.law != "hard_point":
            gs.raise_exception(f"Attachment '{attachment.id}' has unsupported law '{attachment.law}'.")
        if other_anchor.kind == "tissue":
            # a barycentric point of each tissue, bound with the offset they have at rest
            points = []
            for anchor in (tissue_anchor, other_anchor):
                entity = tissues_by_id[anchor.tissue_id]
                tet = tets_by_tissue_id[anchor.tissue_id][anchor.tet_index]
                points.append(TissueAnchor(entity, tuple(int(v) for v in tet), tuple(float(w) for w in anchor.barycentric_weights)))
            entity.scene.vbd_solver.add_tissue_attachment(*points)
            continue

        weights = np.asarray(tissue_anchor.barycentric_weights, dtype=np.float64)
        node = int(np.argmax(weights))
        if not np.isclose(weights[node], 1.0, atol=1e-9):
            gs.raise_exception(
                f"Attachment '{attachment.id}' anchor '{tissue_anchor.id}' is a barycentric (non-node) tissue "
                "anchor; only a node anchor is supported for a hard_point attachment."
            )

        tissue_entity = tissues_by_id[tissue_anchor.tissue_id]
        vertex_idx = int(tets_by_tissue_id[tissue_anchor.tissue_id][tissue_anchor.tet_index][node])
        link = links_by_id[other_anchor.link_id]
        tissue_entity.add_rigid_attachments(np.array([vertex_idx]), link)


def _build_mtus_and_ligaments(scene, packet, anchors_by_id, links_by_id, tissues_by_id, tets_by_tissue_id, coordinates, placement):
    routes_by_id = {r.id: r for r in packet.routes}
    vbd_solver = scene.vbd_solver

    def anchors_of(route_id):
        return [
            _resolve_anchor(anchors_by_id[aid], links_by_id, tissues_by_id, tets_by_tissue_id, placement)
            for aid in routes_by_id[route_id].anchor_ids
        ]

    mtus = {}
    for mtu in packet.mtus:
        if mtu.law != "hill_v1":
            gs.raise_exception(f"MTU '{mtu.id}' has law '{mtu.law}'; only 'hill_v1' is supported.")
        # law_constants is the one the engine cannot honour: the Hill curve constants are shared by every
        # unit. The initial activation and fibre length are carried through to the unit below.
        if mtu.law_constants:
            gs.raise_exception(
                f"MTU '{mtu.id}' carries law_constants {sorted(mtu.law_constants)}; the Hill constants are "
                f"shared by every unit in genesis/engine/solvers/vbd_mtu.py and cannot be set per unit."
            )
        parameters = HillParameters(f_max=mtu.f_max_N, l_opt=mtu.l_opt_m, l_slack=mtu.l_slack_m, v_max=mtu.v_max_m_s)
        mtus[mtu.id] = vbd_solver.add_mtu(
            anchors_of(mtu.route_id),
            parameters,
            activation0=mtu.activation0,
            fibre_length0=mtu.fibre_length0_m,
        )

    ligaments = {}
    for ligament in packet.ligaments:
        if ligament.law != "tension_only_linear":
            gs.raise_exception(f"Ligament '{ligament.id}' has law '{ligament.law}'; only 'tension_only_linear' is supported.")
        if ligament.damping_Ns_m:
            gs.raise_exception(
                f"Ligament '{ligament.id}' asks for damping {ligament.damping_Ns_m} Ns/m; the engine's "
                f"tension-only element has no damping term."
            )
        ligaments[ligament.id] = vbd_solver.add_ligament(
            anchors_of(ligament.route_id), stiffness=ligament.stiffness_N_m, slack_length=ligament.slack_length_m
        )

    rotary_restraints = {}
    for restraint in packet.rotary_restraints:
        if restraint.law != "linear_torque":
            gs.raise_exception(f"Rotary restraint '{restraint.id}' has law '{restraint.law}'; only 'linear_torque' is supported.")
        if restraint.damping:
            gs.raise_exception(
                f"Rotary restraint '{restraint.id}' asks for damping {restraint.damping}; the engine applies "
                f"-k (q - q_rest) with no damping term."
            )
        dof = coordinates[restraint.joint_coordinate_id]
        rotary_restraints[restraint.id] = vbd_solver.add_rotary_restraint(
            dof=dof, stiffness=restraint.stiffness_Nm_rad, rest_angle=restraint.rest_angle_rad
        )

    return mtus, ligaments, rotary_restraints


def _build_contact(scene, packet, group_index, links_by_id, prescribed_specs):
    vbd_solver = scene.vbd_solver
    materials_by_id = {m.id: m for m in packet.contact_materials}
    excluded_pairs = {frozenset((e.group_a, e.group_b)) for e in packet.collision_exclusions}

    colliders = {}
    for root, collider in prescribed_specs:
        entity = links_by_id[root.id].entity
        vbd_solver.add_prescribed_collider(entity, collision_group=group_index[collider.collision_group], link=links_by_id[root.id])
        colliders[collider.id] = len(colliders)

    for pair in packet.contact_pairs:
        if frozenset((pair.group_a, pair.group_b)) in excluded_pairs:
            gs.raise_exception(f"Contact pair '{pair.id}' is also collision-excluded, which is contradictory.")
        (material_id,) = pair.contact_material_ids
        material = materials_by_id[material_id]
        if material.normal_law != "penalty":
            gs.raise_exception(f"Contact material '{material_id}' has normal_law '{material.normal_law}'; only 'penalty' is supported.")
        if material.normal_compliance_m_N != 0.0:
            gs.raise_exception(f"Contact material '{material_id}' has non-zero normal_compliance_m_N; only hard (zero-compliance) contact is supported.")
        if material.friction_law != "coulomb":
            gs.raise_exception(f"Contact material '{material_id}' has friction_law '{material.friction_law}'; only 'coulomb' is supported.")
        if len(material.friction_coefficients) != 1:
            gs.raise_exception(f"Contact material '{material_id}' has {len(material.friction_coefficients)} friction coefficients; only one isotropic coefficient is supported.")
        if material.restitution != 0.0:
            gs.raise_exception(f"Contact material '{material_id}' has non-zero restitution, which is not supported.")
        regularization = material.regularization
        if "stiffness_N_m" not in regularization or "thickness_m" not in regularization:
            gs.raise_exception(
                f"Contact material '{material_id}' regularization is missing 'stiffness_N_m' or 'thickness_m': "
                "the packet schema does not yet carry a dedicated numeric contact-stiffness field, so build_model "
                "reads it from regularization."
            )
        vbd_solver.add_contact_rule(
            group_index[pair.group_a],
            group_index[pair.group_b],
            stiffness=float(regularization["stiffness_N_m"]),
            friction=float(material.friction_coefficients[0]),
            thickness=float(regularization["thickness_m"]),
        )
    return colliders


def build_model(scene, packet, *, placement: Pose | None = None) -> AVBDModel:
    """Build a live scene from a validated AVBD model packet.

    Creates rigid links and joints, tissue, node attachments, muscle-tendon units, ligaments, rotary
    restraints, collision groups, contact rules and prescribed colliders, then returns an `AVBDModel`
    handle. Must be called before `scene.build()`, and at most once per scene.

    See the module docstring for exactly which record types this builds, and the source of each
    "unsupported" error for the exact records it refuses.
    """
    if scene.is_built:
        gs.raise_exception("gs.avbd.build_model must be called before scene.build().")
    if getattr(scene, "_avbd_model_built", False):
        gs.raise_exception("A scene may hold only one AVBD model.")

    gs.avbd.validate_packet(packet)

    if packet.regions:
        names = ", ".join(f"'{r.id}'" for r in packet.regions)
        gs.raise_exception(f"Region records are not supported: {names}.")

    group_index = {cg.id: i for i, cg in enumerate(packet.collision_groups)}
    anchors_by_id = {a.id: a for a in packet.anchors}

    links_by_id, coordinates, prescribed_specs = _build_rigid_trees(scene, packet, placement)
    tissues_by_id, tets_by_tissue_id = _build_tissues(scene, packet, group_index, placement)
    _build_attachments(packet, anchors_by_id, tissues_by_id, tets_by_tissue_id, links_by_id)
    mtus, ligaments, rotary_restraints = _build_mtus_and_ligaments(
        scene, packet, anchors_by_id, links_by_id, tissues_by_id, tets_by_tissue_id, coordinates, placement
    )
    colliders = _build_contact(scene, packet, group_index, links_by_id, prescribed_specs)

    scene._avbd_model_built = True

    ids = IdMaps(
        links=types.MappingProxyType(links_by_id),
        tissues=types.MappingProxyType(tissues_by_id),
        coordinates=types.MappingProxyType(coordinates),
        mtus=types.MappingProxyType(mtus),
        ligaments=types.MappingProxyType(ligaments),
        rotary_restraints=types.MappingProxyType(rotary_restraints),
        colliders=types.MappingProxyType(colliders),
    )
    return AVBDModel(scene, ids)
