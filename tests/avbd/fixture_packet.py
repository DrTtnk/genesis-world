"""Build the synthetic A0 fixture packet used by tests/avbd/test_packet.py.

The fixture is not anatomy: it is a small asymmetric model built to exercise
every record type and every validation branch of `genesis.avbd.packet`. It has
asymmetric geometry, a nonzero COM offset, a two-axis joint, two materials,
two attachment owners (a link and a world point) and two MTUs, per the
importer-fixture checklist in AVBD_MODEL_INTERFACE.md section 6.

The tissue is a solid two-tet bipyramid, not a hollow lumen: A0 tests the
packet schema and its validation branches (closed boundary, positive volume,
outward faces), not anatomical hollowness, which is a later-gate concern.
"""

import numpy as np

import genesis as gs


def _quat(w, x, y, z):
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# A cube split into 6 tets sharing the (0,0,0)-(1,1,1) space diagonal, indexed by
# (dx, dy, dz) role rather than by cube position. Because the role of each corner
# is a fixed function of (dx, dy, dz), adjacent cubes on an integer grid always
# triangulate their shared quad face identically, so the shared face cancels
# exactly and never shows up as spurious boundary.
_CUBE_TETS_BY_ROLE = [
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 1), (0, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (1, 0, 1), (1, 0, 0), (1, 1, 1)),
]


def build_concave_tissue() -> "gs.avbd.Tissue":
    """A 3x3 ring of unit cubes with the centre cube missing: a block with a square hole through it.

    This solid's own centroid sits inside the empty hole, not inside its
    material: a "face normal points away from the mesh centroid" heuristic
    gets every inner-wall face of the hole backwards (confirmed numerically:
    8 of 64 correctly outward-wound faces would be wrongly rejected). The
    exact opposite-vertex test (the one `VBDContact._oriented_outward` uses)
    does not depend on any global reference point, so it gets the hole right.
    """
    vertex_id = {}
    verts = []

    def vertex(point):
        if point not in vertex_id:
            vertex_id[point] = len(verts)
            verts.append(point)
        return vertex_id[point]

    tets = []
    ring_cells = [(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)]  # missing the centre: a hole
    for cell_x, cell_y in ring_cells:
        corner = {
            role: (cell_x + role[0], cell_y + role[1], role[2])
            for role in ((dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1))
        }
        for tet_roles in _CUBE_TETS_BY_ROLE:
            tets.append([vertex(corner[role]) for role in tet_roles])

    rest_positions_m = np.array(verts, dtype=np.float64)
    tets = np.array(tets, dtype=np.int64)

    import igl

    from genesis.engine.solvers.vbd_contact import VBDContact

    boundary_faces, tet_of_facet, _ = igl.boundary_facets(tets)
    # `igl.boundary_facets` does not promise outward winding by itself; reuse the
    # engine's own exact re-orientation (the same one under test in packet.py)
    # to build a boundary that is authored correctly, on purpose.
    boundary_faces = VBDContact._oriented_outward(boundary_faces, tets[tet_of_facet], rest_positions_m).astype(
        np.int64
    )

    av = gs.avbd
    return av.Tissue(
        id="concave_notch",
        rest_positions_m=rest_positions_m,
        tets=tets,
        boundary_faces=boundary_faces,
        tet_material_ids=("mat_wall_muscle",) * len(tets),
        collision_group="cg_concave",
        stress_free_reference="rest",
        active=True,
    )


def build_fixture_packet() -> "gs.avbd.Packet":
    av = gs.avbd

    # -- tissue: a solid, asymmetric two-tet bipyramid --------------------
    # Shared base triangle (v0, v1, v2) is scalene; the two apexes (v3 top,
    # v4 bottom) sit at different heights, so nothing here is symmetric.
    verts = np.array(
        [
            [0.030, 0.000, 0.000],  # v0
            [-0.010, 0.025, 0.000],  # v1
            [-0.015, -0.020, 0.000],  # v2
            [0.000, 0.000, 0.022],  # v3 top apex
            [0.002, -0.003, -0.017],  # v4 bottom apex
        ],
        dtype=np.float64,
    )
    # Orient both tets so the scalar triple product (signed volume) is positive,
    # and so the boundary faces wind outward. Verified numerically below.
    tets = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64)
    # The internal base (v0, v1, v2) is shared by both tets, in opposite winding,
    # and is not part of the boundary. The boundary is the outward-wound three
    # side faces of each tet, verified numerically against the tets above.
    boundary_faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
            [0, 2, 4],
            [0, 4, 1],
            [2, 1, 4],
        ],
        dtype=np.int64,
    )

    tissue = av.Tissue(
        id="wall",
        rest_positions_m=verts,
        tets=tets,
        boundary_faces=boundary_faces,
        tet_material_ids=("mat_wall_muscle", "mat_wall_membrane"),
        collision_group="cg_wall",
        stress_free_reference="rest",
        active=True,
    )

    # -- rigid links --------------------------------------------------------
    skull = av.Link(
        id="skull",
        entity_id="e_skull",
        motion_mode="fixed",
        rest_position_m=np.array([0.0, 0.0, 0.050], dtype=np.float64),
        rest_quaternion_wxyz=_quat(0.98, 0.05, -0.1, 0.15),
        mass_kg=0.30,
        com_link_m=np.array([0.010, 0.0, 0.020], dtype=np.float64),
        inertia_com_kgm2=np.diag([1.0e-4, 1.3e-4, 1.1e-4]),
        collision_geometry_id="skull_mesh",
        collision_group="cg_skull",
    )
    jaw = av.Link(
        id="jaw",
        entity_id="e_jaw",
        motion_mode="dynamic",
        rest_position_m=np.array([0.040, 0.0, 0.010], dtype=np.float64),
        rest_quaternion_wxyz=_quat(0.9, 0.2, -0.05, 0.25),
        mass_kg=0.05,
        com_link_m=np.array([0.008, 0.001, -0.002], dtype=np.float64),
        inertia_com_kgm2=np.diag([2.0e-6, 2.6e-6, 1.4e-6]),
        collision_geometry_id="jaw_mesh",
        collision_group="cg_jaw",
    )
    meal = av.Link(
        id="meal",
        entity_id="e_meal",
        motion_mode="prescribed",
        rest_position_m=np.array([-0.200, 0.0, 0.060], dtype=np.float64),
        rest_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        mass_kg=0.40,
        com_link_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        inertia_com_kgm2=np.diag([1.0e-3, 1.0e-3, 1.0e-3]),
        collision_geometry_id="ellipsoid_meal",
        collision_group="cg_meal",
    )

    # -- joint: two ordered axes, nonzero rest coordinates -------------------
    jaw_joint = av.Joint(
        id="j_jaw",
        parent_link_id="skull",
        child_link_id="jaw",
        joint_type="hinge_chain",
        parent_frame_position_m=np.array([0.030, 0.0, -0.010], dtype=np.float64),
        parent_frame_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        child_frame_position_m=np.array([-0.010, 0.0, 0.005], dtype=np.float64),
        child_frame_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        axes=("hinge", "yaw"),
        coordinate_ids=("j_jaw:hinge", "j_jaw:yaw"),
        rest_coordinates=np.array([0.02, -0.01], dtype=np.float64),
        lower_limits=np.array([-0.1, -0.2], dtype=np.float64),
        upper_limits=np.array([0.6, 0.2], dtype=np.float64),
        damping=np.array([0.002, 0.001], dtype=np.float64),
        armature=np.array([0.0, 0.0], dtype=np.float64),
        parent_world_position_m=None,
        parent_world_quaternion_wxyz=None,
    )

    # -- materials ------------------------------------------------------------
    mat_muscle = av.Material(
        id="mat_wall_muscle",
        law="neo_hookean_v1",
        law_params={"E_Pa": 5.0e4, "nu": 0.45},
        density_kg_m3=1060.0,
        thickness_m=0.002,
        provenance="literature_derived",
    )
    mat_membrane = av.Material(
        id="mat_wall_membrane",
        law="neo_hookean_v1",
        law_params={"E_Pa": 2.0e4, "nu": 0.40},
        density_kg_m3=1040.0,
        thickness_m=0.0015,
        provenance="estimated",
    )

    # -- anchors: one of each kind --------------------------------------------
    anchor_world = av.Anchor(
        id="a_world_hold",
        active=True,
        kind="world",
        position_m=np.array([0.0, 0.0, 0.050], dtype=np.float64),
        link_id=None,
        position_link_m=None,
        tissue_id=None,
        tet_index=None,
        barycentric_weights=None,
    )
    anchor_link = av.Anchor(
        id="a_jaw_point",
        active=True,
        kind="link",
        position_m=None,
        link_id="jaw",
        position_link_m=np.array([0.020, 0.0, 0.0], dtype=np.float64),
        tissue_id=None,
        tet_index=None,
        barycentric_weights=None,
    )
    anchor_tissue = av.Anchor(
        id="a_wall_point",
        active=True,
        kind="tissue",
        position_m=None,
        link_id=None,
        position_link_m=None,
        tissue_id="wall",
        tet_index=0,
        barycentric_weights=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
    )

    # -- routes -----------------------------------------------------------------
    route_main = av.Route(
        id="route_main",
        anchor_ids=("a_world_hold", "a_jaw_point", "a_wall_point"),
        routing_policy="polyline",
        mechanical_owner="mtu_main",
    )
    route_secondary = av.Route(
        id="route_secondary",
        anchor_ids=("a_world_hold", "a_wall_point"),
        routing_policy="polyline",
        mechanical_owner="mtu_secondary",
    )

    # -- MTUs: two distinct units -------------------------------------------
    mtu_main = av.MTU(
        id="mtu_main",
        route_id="route_main",
        law="hill_v1",
        f_max_N=12.0,
        l_opt_m=0.020,
        l_slack_m=0.010,
        v_max_m_s=0.20,
        activation0=0.10,
        fibre_length0_m=0.018,
        law_constants={"a_short": 0.25, "f_len_width": 0.5},
        passive_mechanics_owner="mtu_main",
    )
    mtu_secondary = av.MTU(
        id="mtu_secondary",
        route_id="route_secondary",
        law="hill_v1",
        f_max_N=6.5,
        l_opt_m=0.015,
        l_slack_m=0.007,
        v_max_m_s=0.15,
        activation0=0.0,
        fibre_length0_m=0.014,
        law_constants={"a_short": 0.25, "f_len_width": 0.5},
        passive_mechanics_owner="mtu_secondary",
    )

    # -- ligament ---------------------------------------------------------------
    ligament = av.Ligament(
        id="lig_main",
        route_id="route_secondary",
        law="tension_only_linear",
        slack_length_m=0.015,
        rest_length_m=0.020,
        stiffness_N_m=500.0,
        damping_Ns_m=2.0,
    )

    # -- rotary restraint --------------------------------------------------------
    restraint = av.RotaryRestraint(
        id="restraint_jaw_hinge",
        joint_coordinate_id="j_jaw:hinge",
        rest_angle_rad=0.0,
        law="linear_torque",
        stiffness_Nm_rad=0.05,
        damping=0.001,
    )

    # -- attachments: two different "other" owners (link and world) -------------
    attachment_hard = av.Attachment(
        id="att_hard",
        tissue_anchor_id="a_wall_point",
        other_anchor_id="a_jaw_point",
        law="hard_point",
        stiffness_N_m=None,
        damping_Ns_m=None,
        collision_exclusion=False,
    )
    attachment_elastic = av.Attachment(
        id="att_elastic",
        tissue_anchor_id="a_wall_point",
        other_anchor_id="a_world_hold",
        law="elastic_point",
        stiffness_N_m=800.0,
        damping_Ns_m=5.0,
        collision_exclusion=False,
    )

    # -- collision groups, exclusions, contact materials/pairs -------------------
    cg_wall = av.CollisionGroup(id="cg_wall", member_ids=("wall",))
    cg_jaw = av.CollisionGroup(id="cg_jaw", member_ids=("jaw",))
    cg_skull = av.CollisionGroup(id="cg_skull", member_ids=("skull",))
    cg_meal = av.CollisionGroup(id="cg_meal", member_ids=("meal",))

    exclusion = av.CollisionExclusion(
        id="excl_jaw_wall_seam",
        group_a="cg_jaw",
        group_b="cg_wall",
        reason="jaw is rigidly attached to the wall at this seam, not a contact pair",
    )

    contact_material = av.ContactMaterial(
        id="cm_default",
        normal_law="penalty",
        normal_compliance_m_N=0.0,
        friction_law="coulomb",
        friction_coefficients=np.array([0.40], dtype=np.float64),
        regularization={"eps_v": 1.0e-3},
        restitution=0.0,
    )

    pair_meal_wall = av.ContactPair(
        id="pair_meal_wall",
        group_a="cg_meal",
        group_b="cg_wall",
        contact_material_ids=("cm_default",),
        required=True,
    )
    pair_skull_meal = av.ContactPair(
        id="pair_skull_meal",
        group_a="cg_skull",
        group_b="cg_meal",
        contact_material_ids=("cm_default",),
        required=False,
    )

    # -- prescribed collider ------------------------------------------------------
    prescribed_collider = av.PrescribedCollider(
        id="pc_meal",
        link_id="meal",
        collision_geometry_id="ellipsoid_meal",
        semiaxes_m=np.array([0.022, 0.016, 0.011], dtype=np.float64),
        collision_group="cg_meal",
        trajectory_source_id="feeding_traj_01",
    )

    # -- region -------------------------------------------------------------------
    region = av.Region(
        id="stomach",
        owner_ids=("wall",),
        geometry={"kind": "convex_hull_of_tissue", "tissue_id": "wall"},
        purpose="stomach_containment",
        rest_to_current_mapping="wall_local",
        boundary_semantics="entrance_exit_portals",
    )

    # -- entities -------------------------------------------------------------------
    e_skull = av.Entity(id="e_skull", link_ids=("skull",), tissue_ids=(), role="skull_root", integration_owner="skull")
    e_jaw = av.Entity(id="e_jaw", link_ids=("jaw",), tissue_ids=(), role="jaw", integration_owner="jaw")
    e_wall = av.Entity(id="e_wall", link_ids=(), tissue_ids=("wall",), role="digestive_wall", integration_owner="wall")
    e_meal = av.Entity(id="e_meal", link_ids=("meal",), tissue_ids=(), role="prescribed_meal", integration_owner="meal")

    return av.Packet(
        interface_version=av.INTERFACE_VERSION,
        model_id="avbd_a0_fixture",
        units={"length": "m", "mass": "kg", "time": "s", "angle": "rad"},
        world_frame={"up": "+Z", "longitudinal": "+X_toward_tail"},
        source_hashes={"blender_scene": "deadbeef0001"},
        parameter_provenance={"note": "synthetic A0 fixture, not anatomical data"},
        required_capabilities=("hard_point_attachment", "elastic_point_attachment", "prescribed_link"),
        entities=(e_skull, e_jaw, e_wall, e_meal),
        links=(skull, jaw, meal),
        joints=(jaw_joint,),
        materials=(mat_muscle, mat_membrane),
        tissues=(tissue,),
        anchors=(anchor_world, anchor_link, anchor_tissue),
        routes=(route_main, route_secondary),
        mtus=(mtu_main, mtu_secondary),
        ligaments=(ligament,),
        rotary_restraints=(restraint,),
        attachments=(attachment_hard, attachment_elastic),
        collision_groups=(cg_wall, cg_jaw, cg_skull, cg_meal),
        collision_exclusions=(exclusion,),
        contact_materials=(contact_material,),
        contact_pairs=(pair_meal_wall, pair_skull_meal),
        prescribed_colliders=(prescribed_collider,),
        regions=(region,),
    )


########################## fixtures for gs.avbd.build_model ##########################

# The A0 fixture above exercises the packet schema, not the model builder: several of its records
# (a two-axis joint, an elastic attachment, a barycentric hard-point attachment, two materials on one
# tissue, a region) are exactly the records `build_model` must refuse. The fixtures below build small,
# focused packets instead: one that `build_model` accepts and steps, and one small addition per record
# class `build_model` must reject, isolated from every other rejection so each test sees only the one
# violation it names.


def _single_tet_tissue(tissue_id, material_id, verts, collision_group=None):
    """One tetrahedron as a tissue: the boundary is its own four faces, correctly outward-oriented by
    the same primitive `genesis.avbd.packet._check_boundary_outward` checks against (see
    `build_concave_tissue` above), so its correctness does not rely on a hand-derived winding."""
    import igl

    from genesis.engine.solvers.vbd_contact import VBDContact

    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    boundary_faces, tet_of_facet, _ = igl.boundary_facets(tets)
    boundary_faces = VBDContact._oriented_outward(boundary_faces, tets[tet_of_facet], verts).astype(np.int64)
    return gs.avbd.Tissue(
        id=tissue_id,
        rest_positions_m=verts,
        tets=tets,
        boundary_faces=boundary_faces,
        tet_material_ids=(material_id,),
        collision_group=collision_group,
        stress_free_reference="rest",
        active=True,
    )


def _tet_verts(origin):
    """A well-shaped tetrahedron (positive volume, no degenerate faces) translated to `origin`."""
    origin = np.asarray(origin, dtype=np.float64)
    return origin + np.array(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.005, 0.005, 0.02]], dtype=np.float64
    )


def build_minimal_packet():
    """The smallest packet `build_model` accepts: one fixed link, one one-tet tissue, nothing else.

    Every rejection test below starts from this packet and adds exactly the one record class under
    test, so a failure names that record and not an unrelated one.
    """
    av = gs.avbd
    root = av.Link(
        id="root",
        entity_id="e_root",
        motion_mode="fixed",
        rest_position_m=np.zeros(3),
        rest_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        mass_kg=0.1,
        com_link_m=np.zeros(3),
        inertia_com_kgm2=np.diag([1.0e-4, 1.0e-4, 1.0e-4]),
        collision_geometry_id=None,
        collision_group=None,
    )
    material = av.Material(
        id="mat_wall",
        law="neo_hookean_v1",
        law_params={"E_Pa": 5.0e4, "nu": 0.45},
        density_kg_m3=1060.0,
        thickness_m=0.002,
        provenance="synthetic",
    )
    tissue = _single_tet_tissue("wall", "mat_wall", _tet_verts((0.1, 0.0, 0.0)))
    e_root = av.Entity(id="e_root", link_ids=("root",), tissue_ids=(), role="root", integration_owner="root")
    e_wall = av.Entity(id="e_wall", link_ids=(), tissue_ids=("wall",), role="wall", integration_owner="wall")
    return av.Packet(
        interface_version=av.INTERFACE_VERSION,
        model_id="avbd_model_minimal",
        units={"length": "m", "mass": "kg", "time": "s", "angle": "rad"},
        world_frame={"up": "+Z", "longitudinal": "+X_toward_tail"},
        source_hashes={},
        parameter_provenance={"note": "synthetic build_model fixture"},
        required_capabilities=(),
        entities=(e_root, e_wall),
        links=(root,),
        joints=(),
        materials=(material,),
        tissues=(tissue,),
        anchors=(),
        routes=(),
        mtus=(),
        ligaments=(),
        rotary_restraints=(),
        attachments=(),
        collision_groups=(),
        collision_exclusions=(),
        contact_materials=(),
        contact_pairs=(),
        prescribed_colliders=(),
        regions=(),
    )


def build_hinge_model_packet():
    """A packet exercising every record class `build_model` supports: a fixed base, a single-axis
    hinge bone, a tissue wall hard-attached to the bone, one MTU and one ligament spanning the hinge,
    a rotary restraint on the hinge coordinate, and a prescribed ellipsoid colliding with the wall.

    The hinge and MTU geometry mirror `tests/vbd/test_vbd_mtu.py::hinge_scene` and
    `test_the_pull_on_a_link_anchor_turns_the_hinge_the_way_the_route_shortens`: `base` and `bone`
    share one frame origin at rest (both joint frames are the identity), so the same anchor offsets
    produce the same qualitative pull.
    """
    av = gs.avbd
    f_max, l_opt, l_slack, v_max = 250.0, 0.06, 0.06, 0.3
    rest = l_opt + l_slack

    base = av.Link(
        id="base",
        entity_id="e_base",
        motion_mode="fixed",
        rest_position_m=np.zeros(3),
        rest_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        mass_kg=0.05,
        com_link_m=np.array([-0.06, 0.0, 0.0]),
        inertia_com_kgm2=np.diag([1.0e-4, 1.3e-4, 1.1e-4]),
        collision_geometry_id=None,
        collision_group=None,
    )
    bone = av.Link(
        id="bone",
        entity_id="e_bone",
        motion_mode="dynamic",
        rest_position_m=np.zeros(3),
        rest_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        mass_kg=0.1,
        com_link_m=np.array([0.05, 0.0, 0.0]),
        inertia_com_kgm2=np.diag([2.0e-6, 2.6e-6, 1.4e-6]),
        collision_geometry_id=None,
        collision_group=None,
    )
    meal = av.Link(
        id="meal",
        entity_id="e_meal",
        motion_mode="prescribed",
        rest_position_m=np.array([0.11, 0.0, 0.05]),
        rest_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        mass_kg=0.2,
        com_link_m=np.zeros(3),
        inertia_com_kgm2=np.diag([1.0e-3, 1.0e-3, 1.0e-3]),
        collision_geometry_id="ellipsoid_meal",
        collision_group="cg_meal",
    )

    hinge = av.Joint(
        id="j_bone",
        parent_link_id="base",
        child_link_id="bone",
        joint_type="hinge_chain",
        parent_frame_position_m=np.zeros(3),
        parent_frame_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        child_frame_position_m=np.zeros(3),
        child_frame_quaternion_wxyz=_quat(1.0, 0.0, 0.0, 0.0),
        axes=("hinge",),
        coordinate_ids=("j_bone:hinge",),
        rest_coordinates=np.array([0.0]),
        lower_limits=np.array([-0.6]),
        upper_limits=np.array([0.6]),
        damping=np.array([0.002]),
        armature=np.array([0.0]),
        parent_world_position_m=None,
        parent_world_quaternion_wxyz=None,
    )

    material = av.Material(
        id="mat_wall",
        law="neo_hookean_v1",
        law_params={"E_Pa": 1.0e5, "nu": 0.3},
        density_kg_m3=1000.0,
        thickness_m=0.002,
        provenance="synthetic",
    )
    tissue = _single_tet_tissue("wall", "mat_wall", _tet_verts((0.06, 0.0, 0.006)), collision_group="cg_wall")

    anchor_bone_flexor = av.Anchor(
        id="a_bone_flexor",
        active=True,
        kind="link",
        position_m=None,
        link_id="bone",
        position_link_m=np.array([rest - 0.02, 0.0, 0.012]),
        tissue_id=None,
        tet_index=None,
        barycentric_weights=None,
    )
    anchor_base_flexor = av.Anchor(
        id="a_base_flexor",
        active=True,
        kind="link",
        position_m=None,
        link_id="base",
        position_link_m=np.array([-0.02, 0.0, 0.012]),
        tissue_id=None,
        tet_index=None,
        barycentric_weights=None,
    )
    anchor_bone_ligament = av.Anchor(
        id="a_bone_ligament",
        active=True,
        kind="link",
        position_m=None,
        link_id="bone",
        position_link_m=np.array([rest - 0.02, 0.0, -0.01]),
        tissue_id=None,
        tet_index=None,
        barycentric_weights=None,
    )
    anchor_base_ligament = av.Anchor(
        id="a_base_ligament",
        active=True,
        kind="link",
        position_m=None,
        link_id="base",
        position_link_m=np.array([-0.02, 0.0, -0.01]),
        tissue_id=None,
        tet_index=None,
        barycentric_weights=None,
    )
    anchor_wall_node = av.Anchor(
        id="a_wall_node",
        active=True,
        kind="tissue",
        position_m=None,
        link_id=None,
        position_link_m=None,
        tissue_id="wall",
        tet_index=0,
        barycentric_weights=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    anchor_bone_for_wall = av.Anchor(
        id="a_bone_for_wall",
        active=True,
        kind="link",
        position_m=None,
        link_id="bone",
        position_link_m=np.zeros(3),
        tissue_id=None,
        tet_index=None,
        barycentric_weights=None,
    )

    route_flexor = av.Route(
        id="route_flexor", anchor_ids=("a_base_flexor", "a_bone_flexor"), routing_policy="polyline", mechanical_owner="mtu_flexor"
    )
    route_ligament = av.Route(
        id="route_ligament",
        anchor_ids=("a_base_ligament", "a_bone_ligament"),
        routing_policy="polyline",
        mechanical_owner="lig_main",
    )

    mtu_flexor = av.MTU(
        id="mtu_flexor",
        route_id="route_flexor",
        law="hill_v1",
        f_max_N=f_max,
        l_opt_m=l_opt,
        l_slack_m=l_slack,
        v_max_m_s=v_max,
        activation0=0.0,
        fibre_length0_m=l_opt,
        law_constants={},
        passive_mechanics_owner="mtu_flexor",
    )
    ligament = av.Ligament(
        id="lig_main", route_id="route_ligament", law="tension_only_linear", slack_length_m=0.05, rest_length_m=rest, stiffness_N_m=500.0, damping_Ns_m=2.0
    )
    restraint = av.RotaryRestraint(
        id="restraint_bone", joint_coordinate_id="j_bone:hinge", rest_angle_rad=0.0, law="linear_torque", stiffness_Nm_rad=1.0, damping=0.001
    )
    attachment = av.Attachment(
        id="att_wall_bone",
        tissue_anchor_id="a_wall_node",
        other_anchor_id="a_bone_for_wall",
        law="hard_point",
        stiffness_N_m=None,
        damping_Ns_m=None,
        collision_exclusion=False,
    )

    cg_wall = av.CollisionGroup(id="cg_wall", member_ids=("wall",))
    cg_meal = av.CollisionGroup(id="cg_meal", member_ids=("meal",))
    contact_material = av.ContactMaterial(
        id="cm_wall_meal",
        normal_law="penalty",
        normal_compliance_m_N=0.0,
        friction_law="coulomb",
        friction_coefficients=np.array([0.3]),
        regularization={"stiffness_N_m": 1.0e5, "thickness_m": 1.0e-3},
        restitution=0.0,
    )
    contact_pair = av.ContactPair(
        id="pair_wall_meal", group_a="cg_wall", group_b="cg_meal", contact_material_ids=("cm_wall_meal",), required=False
    )
    prescribed_collider = av.PrescribedCollider(
        id="pc_meal",
        link_id="meal",
        collision_geometry_id="ellipsoid_meal",
        semiaxes_m=np.array([0.03, 0.02, 0.02]),
        collision_group="cg_meal",
        trajectory_source_id="traj_meal",
    )

    e_base = av.Entity(id="e_base", link_ids=("base",), tissue_ids=(), role="base", integration_owner="base")
    e_bone = av.Entity(id="e_bone", link_ids=("bone",), tissue_ids=(), role="bone", integration_owner="bone")
    e_meal = av.Entity(id="e_meal", link_ids=("meal",), tissue_ids=(), role="meal", integration_owner="meal")
    e_wall = av.Entity(id="e_wall", link_ids=(), tissue_ids=("wall",), role="wall", integration_owner="wall")

    return av.Packet(
        interface_version=av.INTERFACE_VERSION,
        model_id="avbd_model_hinge",
        units={"length": "m", "mass": "kg", "time": "s", "angle": "rad"},
        world_frame={"up": "+Z", "longitudinal": "+X_toward_tail"},
        source_hashes={},
        parameter_provenance={"note": "synthetic build_model fixture"},
        required_capabilities=("hard_point_attachment", "fixed_base_chain", "prescribed_link", "ellipsoid_collider", "isotropic_coulomb_friction"),
        entities=(e_base, e_bone, e_meal, e_wall),
        links=(base, bone, meal),
        joints=(hinge,),
        materials=(material,),
        tissues=(tissue,),
        anchors=(
            anchor_bone_flexor,
            anchor_base_flexor,
            anchor_bone_ligament,
            anchor_base_ligament,
            anchor_wall_node,
            anchor_bone_for_wall,
        ),
        routes=(route_flexor, route_ligament),
        mtus=(mtu_flexor,),
        ligaments=(ligament,),
        rotary_restraints=(restraint,),
        attachments=(attachment,),
        collision_groups=(cg_wall, cg_meal),
        collision_exclusions=(),
        contact_materials=(contact_material,),
        contact_pairs=(contact_pair,),
        prescribed_colliders=(prescribed_collider,),
        regions=(),
    )
