import numpy as np
import pytest
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.vbd_contact import func_point_triangle_weights, func_segment_parameters
from genesis.utils.misc import qd_to_torch, tensor_to_array

from ..utils.assertions import assert_allclose


N_CASES = 32


@qd.kernel
def kernel_closest_features_reference(
    x: qd.template(),
    a: qd.template(),
    b: qd.template(),
    c: qd.template(),
    d: qd.template(),
    weights: qd.template(),
    parameters: qd.template(),
):
    for i in range(N_CASES):
        weights[i] = func_point_triangle_weights(x[i], a[i], b[i], c[i])
        s, t = func_segment_parameters(a[i], b[i], c[i], d[i])
        parameters[i] = qd.Vector([s, t], dt=gs.qd_float)


def _closest_point_on_triangle_brute(x, a, b, c):
    grid = torch.linspace(0.0, 1.0, 801, dtype=torch.float64, device="cpu")
    u, v = torch.meshgrid(grid, grid, indexing="ij")
    keep = (u + v) <= 1.0
    u, v = u[keep], v[keep]
    points = (1.0 - u - v)[:, None] * a + u[:, None] * b + v[:, None] * c
    return torch.linalg.vector_norm(points - x, dim=1).min()


def _closest_segments_brute(a, b, c, d):
    grid = torch.linspace(0.0, 1.0, 801, dtype=torch.float64, device="cpu")
    s, t = torch.meshgrid(grid, grid, indexing="ij")
    p = a + s.reshape(-1, 1) * (b - a)
    q = c + t.reshape(-1, 1) * (d - c)
    return torch.linalg.vector_norm(p - q, dim=1).min()


def test_closest_feature_kernels_match_brute_force():
    generator = torch.Generator(device="cpu").manual_seed(20260912)
    dtype = gs.tc_float
    tolerance = 3e-3  # the brute force samples the primitives on an 801-point grid
    points = torch.randn(5, N_CASES, 3, generator=generator, dtype=torch.float64, device="cpu")
    # every closest-feature region of the triangle and every clamp case of the segments
    points[0, :8] = points[1, :8] + 0.05 * torch.randn(8, 3, generator=generator, dtype=torch.float64, device="cpu")
    points[0, 8:16] = 0.5 * (points[1, 8:16] + points[2, 8:16]) + 3.0 * torch.randn(
        8, 3, generator=generator, dtype=torch.float64, device="cpu"
    )
    fields = [qd.Vector.field(3, dtype=gs.qd_float, shape=N_CASES) for _ in range(5)]
    for field, values in zip(fields, points):
        field.from_torch(values.to(dtype).to(gs.device))
    weights = qd.Vector.field(3, dtype=gs.qd_float, shape=N_CASES)
    parameters = qd.Vector.field(2, dtype=gs.qd_float, shape=N_CASES)
    kernel_closest_features_reference(*fields, weights, parameters)
    weights = qd_to_torch(weights).to(torch.float64).cpu()
    parameters = qd_to_torch(parameters).to(torch.float64).cpu()
    x, a, b, c, d = points
    assert_allclose(weights.sum(dim=1), 1.0, tol=1e-6)
    assert (weights >= -1e-7).all()
    assert (parameters >= 0.0).all() and (parameters <= 1.0).all()
    for i in range(N_CASES):
        q = weights[i, 0] * a[i] + weights[i, 1] * b[i] + weights[i, 2] * c[i]
        assert_allclose(
            torch.linalg.vector_norm(x[i] - q), _closest_point_on_triangle_brute(x[i], a[i], b[i], c[i]), tol=tolerance
        )
        p = a[i] + parameters[i, 0] * (b[i] - a[i])
        r = c[i] + parameters[i, 1] * (d[i] - c[i])
        assert_allclose(torch.linalg.vector_norm(p - r), _closest_segments_brute(a[i], b[i], c[i], d[i]), tol=tolerance)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_tissue_rests_on_fixed_rigid_box_within_the_contact_layer(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
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
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.0, 0.0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.0, 0.0205),
            nobisect=False,
            maxvolume=1e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.build(n_envs=n_envs)
    thickness = 1e-3
    for _ in range(80):
        scene.step()
    positions = tissue.get_positions()
    lowest = positions[..., 2].min()
    # the tissue sits on the table: inside the layer, never through the surface
    assert lowest > -1e-4
    assert lowest < thickness
    state = tissue.get_state()
    assert torch.linalg.vector_norm(state.vel, dim=-1).max() < 5e-3
    mass = tissue.material.rho * 0.04**3
    reactions = scene.vbd_solver.collider_reactions()
    assert reactions.shape == (max(n_envs, 1), 1, 6)
    assert_allclose(reactions[:, 0, 2], -mass * 9.81, rtol=0.02, atol=0.0)
    assert_allclose(reactions[:, 0, :2], 0.0, atol=0.02 * mass * 9.81)
    assert torch.isfinite(positions).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_prescribed_ellipsoid_compresses_tissue_and_a_blocked_command_is_rejected(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
            batch_links_info=True,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.0, 0.0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    meal = scene.add_entity(
        morph=gs.morphs.MJCF(
            file='<mujoco><worldbody><body pos="0 0 0.08"><geom type="ellipsoid" size="0.03 0.02 0.02"/></body></worldbody></mujoco>',
            decimate=False,
        ),
        material=gs.materials.Rigid(),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.0, 0.0211),
            nobisect=False,
            maxvolume=1e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_prescribed_collider(meal, collision_group=2, link=meal.links[1])
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(1, 2, stiffness=1e5, friction=0.3, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(0, 2, stiffness=1e5, friction=0.3, thickness=1e-3)
    scene.build(n_envs=n_envs)
    B = max(n_envs, 1)
    start = torch.tensor([0.0, 0.0, 0.08], device=gs.device).expand(B, 1, 3).clone()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device).expand(B, 1, 4).clone()
    # descend 30 mm in 0.15 s: the meal touches the tissue top (z = 0.0411) after 18.9 mm and compresses it 11 mm
    for i in range(60):
        target = start.clone()
        target[..., 2] = 0.08 - 0.03 * (i + 1) / 60
        scene.vbd_solver.set_prescribed_targets(target, quat)
        scene.step()
    # hold the meal so the tissue settles and the force balance is static
    for _ in range(60):
        scene.step()
    positions = tissue.get_positions()
    top = positions[..., 2].max()
    assert top < 0.0411 - 0.004  # the tissue is compressed by the meal
    assert torch.linalg.vector_norm(tissue.get_state().vel, dim=-1).max() < 5e-3
    reactions = scene.vbd_solver.collider_reactions()
    assert reactions.shape == (B, 2, 6)
    # the meal pushes down on the tissue, so the tissue pushes up on the meal, and the table carries the same load
    assert (reactions[:, 1, 2] > 5.0).all()
    assert_allclose(reactions[:, 0, 2], -reactions[:, 1, 2], rtol=0.05, atol=0.0)
    assert (positions[..., 2].min() > -1e-4).all()
    assert torch.isfinite(positions).all()
    # a command through the table: the tissue would have to be crushed to nothing, and the engine refuses
    with pytest.raises(gs.GenesisException, match="crossed|margin|finite"):
        for i in range(40):
            target[..., 2] = 0.05 - 0.06 * (i + 1) / 40
            scene.vbd_solver.set_prescribed_targets(target, quat)
            scene.step()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_prescribed_meal_rotates_hinge_bone_through_wall_and_direct_contact(n_envs, show_viewer, hinge_bone_xml):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
            batch_links_info=True,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.05, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    skeleton = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=hinge_bone_xml,
        ),
        material=gs.materials.Rigid(),
    )
    bone = skeleton.get_link("bone")
    meal = scene.add_entity(
        morph=gs.morphs.MJCF(
            file='<mujoco><worldbody><body pos="0.07 0 0.06"><geom type="ellipsoid" size="0.03 0.02 0.02"/></body></worldbody></mujoco>',
            decimate=False,
        ),
        material=gs.materials.Rigid(),
    )
    # a wall of tissue lying on the far half of the bone, its lower face attached to the bone
    wall = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.02, 0.01),
            pos=(0.08, 0.0, 0.0101),
            nobisect=False,
            maxvolume=2e-6,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    rest = tensor_to_array(wall.init_positions)
    wall.add_rigid_attachments(np.flatnonzero(rest[:, 2] < rest[:, 2].min() + 1e-5), bone)
    scene.vbd_solver.add_rigid_collider(bone, collision_group=0)
    scene.vbd_solver.add_prescribed_collider(meal, collision_group=2, link=meal.links[1])
    scene.vbd_solver.add_contact_rule(1, 2, stiffness=1e5, friction=0.3, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(0, 2, stiffness=1e5, friction=0.3, thickness=1e-3)
    scene.build(n_envs=n_envs)
    B = max(n_envs, 1)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device).expand(B, 1, 4).clone()
    target = torch.tensor([0.07, 0.0, 0.06], device=gs.device).expand(B, 1, 3).clone()
    # the meal descends 40 mm: it meets the wall top (z = 0.0202) after 19.8 mm and presses the bone down
    peak_reaction = torch.zeros(B, device=gs.device)
    for i in range(80):
        target[..., 2] = 0.06 - 0.04 * (i + 1) / 80
        scene.vbd_solver.set_prescribed_targets(target, quat)
        scene.step()
        reactions = scene.vbd_solver.collider_reactions()
        assert reactions.shape == (B, 2, 6)
        peak_reaction = torch.maximum(peak_reaction, reactions[:, 1, 2])
    angle = skeleton.get_dofs_position().reshape(B, 1)
    # pressing the far end down rotates about +y by the right-hand rule: positive angle, inside the limit
    assert (angle[:, 0] > 0.05).all()
    assert (angle[:, 0] < 0.6).all()
    # the wall pushed the meal up while it was loaded; once the bone swings away the meal is unloaded
    assert (peak_reaction > 0.1).all()
    positions = wall.get_positions()
    assert torch.isfinite(positions).all()
    anchors = bone.get_pos().reshape(B, 3)
    assert torch.isfinite(anchors).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_free_bone_with_attached_tissue_rests_on_table_through_rigid_and_tissue_contact(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
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
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.02, 0.0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.3, 0.3, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.04, 0.0, 0.0211),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
        ),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.0, 0.0211),
            nobisect=False,
            maxvolume=1e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    rest = tensor_to_array(tissue.init_positions)
    tissue.add_rigid_attachments(np.flatnonzero(rest[:, 0] > rest[:, 0].max() - 1e-5), bone.links[0])
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_rigid_collider(bone.links[0], collision_group=2)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(0, 2, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.build(n_envs=n_envs)
    B = max(n_envs, 1)
    for _ in range(100):
        scene.step()
    positions = tissue.get_positions()
    bone_pos = bone.get_pos().reshape(B, 3)
    # both rest on the table inside the layer, and the bone kept its height above its own bottom face
    assert (positions[..., 2].min() > -1e-4).all()
    assert (positions[..., 2].min() < 1e-3).all()
    assert (bone_pos[:, 2] > 0.02 - 1e-4).all() and (bone_pos[:, 2] < 0.0211).all()
    assert torch.linalg.vector_norm(tissue.get_state().vel, dim=-1).max() < 5e-3
    assert torch.linalg.vector_norm(bone.get_vel().reshape(B, 3), dim=-1).max() < 5e-3
    weight = (tissue.material.rho + bone.material.rho) * 0.04**3 * 9.81
    reactions = scene.vbd_solver.collider_reactions()
    assert reactions.shape == (B, 2, 6)
    assert_allclose(reactions[:, 0, 2], -weight, rtol=0.03, atol=0.0)
    # the bone's own contact carries the bone's weight; the tissue's carries the tissue's, both act on the table
    assert (reactions[:, 1, 2] > 0.9 * bone.material.rho * 0.04**3 * 9.81).all()


def _prescribed_box_xml(pos, half_size):
    return (
        f'<mujoco><worldbody><body pos="{pos[0]} {pos[1]} {pos[2]}">'
        f'<geom type="box" size="{half_size[0]} {half_size[1]} {half_size[2]}"/></body></worldbody></mujoco>'
    )


def _prescribed_cylinder_xml(pos, radius, half_height):
    return (
        f'<mujoco><worldbody><body pos="{pos[0]} {pos[1]} {pos[2]}">'
        f'<geom type="cylinder" size="{radius} {half_height}"/></body></worldbody></mujoco>'
    )


@pytest.mark.parametrize("n_envs", [0, 2])
def test_rotating_collider_drags_tissue_only_through_friction(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
            batch_links_info=True,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -0.5, 0.3),
            camera_lookat=(0.1, 0.0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    tissues, boxes = [], []
    for x, group in ((0.0, 1), (0.2, 3)):
        tissues.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    size=(0.04, 0.04, 0.02),
                    pos=(x, 0.0, 0.01),
                    nobisect=False,
                    maxvolume=1e-5,
                ),
                material=gs.materials.VBD.Muscle(
                    E=1e5,
                    nu=0.3,
                    collision_group=group,
                ),
            )
        )
        # a cylinder keeps its footprint while it spins, so only friction can move the tissue
        boxes.append(
            scene.add_entity(
                morph=gs.morphs.MJCF(
                    file=_prescribed_cylinder_xml((x, 0.0, 0.031), 0.012, 0.01),
                ),
                material=gs.materials.Rigid(),
            )
        )
    scene.vbd_solver.add_prescribed_collider(boxes[0], collision_group=2, link=boxes[0].links[1])
    scene.vbd_solver.add_prescribed_collider(boxes[1], collision_group=4, link=boxes[1].links[1])
    scene.vbd_solver.add_contact_rule(1, 2, stiffness=1e5, friction=0.6, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(3, 4, stiffness=1e5, friction=0.0, thickness=1e-3)
    scene.build(n_envs=n_envs)
    # the lower faces are held so the blocks cannot spin as a whole
    for tissue in tissues:
        rest = tensor_to_array(tissue.init_positions)
        tissue.set_pinned(rest[:, 2] < rest[:, 2].min() + 1e-5)
    B = max(n_envs, 1)
    pos = torch.tensor([[0.0, 0.0, 0.031], [0.2, 0.0, 0.031]], device=gs.device).expand(B, 2, 3).clone()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device).expand(B, 2, 4).clone()
    # press 0.8 mm into the tissue tops (less than the layer, so the footprint may sweep over unloaded
    # vertices), then rotate a quarter turn about z over 0.2 s
    for i in range(20):
        pos[..., 2] = 0.031 - 0.0008 * (i + 1) / 20
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
    reactions = scene.vbd_solver.collider_reactions()
    assert (reactions[:, :, 2] > 0.5).all()
    before = [tissue.get_positions().clone() for tissue in tissues]
    torque_sign = torch.zeros(B, device=gs.device)
    for i in range(80):
        angle = 0.5 * torch.pi * (i + 1) / 80
        quat[..., 0] = torch.cos(torch.tensor(angle / 2))
        quat[..., 3] = torch.sin(torch.tensor(angle / 2))
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
        reactions = scene.vbd_solver.collider_reactions()
        torque_sign = torque_sign + reactions[:, 0, 5]
    after = [tissue.get_positions() for tissue in tissues]
    swirls = []
    for tissue, x0, x1 in zip(tissues, before, after):
        rest = tensor_to_array(tissue.init_positions)
        top = torch.as_tensor(rest[:, 2] > rest[:, 2].max() - 1e-5, device=gs.device)
        centre = torch.as_tensor(rest.mean(axis=0), device=gs.device)
        radial = (x0[:, top] - centre)[..., :2]
        displacement = (x1[:, top] - x0[:, top])[..., :2]
        # tangential motion in the sense of a positive rotation about z: r x d along +z
        swirls.append((radial[..., 0] * displacement[..., 1] - radial[..., 1] * displacement[..., 0]).mean(dim=-1))
    # the frictional top follows the spin; the frictionless one shows only the faceting of its collider
    assert (swirls[0] > 1e-7).all()
    assert (swirls[1].abs() < 0.01 * swirls[0]).all()
    # the dragged tissue resists the rotation: the torque on the frictional collider about +z is negative, and the
    # frictionless one feels only the faceting of its mesh
    assert (torque_sign < 0.0).all()
    assert (reactions[:, 1, 5].abs() < 0.01 * reactions[:, 0, 5].abs()).all()
    assert all(torch.isfinite(tissue.get_positions()).all() for tissue in tissues)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_sliding_collider_obeys_the_friction_cone_and_an_off_axis_approach_pushes_back(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
            batch_links_info=True,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.0, 0.0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.08, 0.04, 0.02),
            pos=(0.0, 0.0, 0.01),
            nobisect=False,
            maxvolume=1e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=_prescribed_box_xml((-0.03, 0.0, 0.05), (0.01, 0.01, 0.01)),
        ),
        material=gs.materials.Rigid(),
    )
    mu = 0.4
    scene.vbd_solver.add_prescribed_collider(box, collision_group=2, link=box.links[1])
    scene.vbd_solver.add_contact_rule(1, 2, stiffness=1e5, friction=mu, thickness=1e-3)
    scene.build(n_envs=n_envs)
    rest = tensor_to_array(tissue.init_positions)
    tissue.set_pinned(rest[:, 2] < rest[:, 2].min() + 1e-5)
    B = max(n_envs, 1)
    pos = torch.tensor([-0.03, 0.0, 0.05], device=gs.device).expand(B, 1, 3).clone()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device).expand(B, 1, 4).clone()
    # off-axis approach at 45 degrees in the xz plane: 20.8 mm down and along +x over 0.1 s; the box bottom
    # (z = 0.04) meets the tissue top (z = 0.02) after 20 mm, so the last 0.8 mm compresses inside the layer
    for i in range(40):
        pos[..., 0] = -0.03 + 0.0208 * (i + 1) / 40
        pos[..., 2] = 0.05 - 0.0208 * (i + 1) / 40
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
    reactions = scene.vbd_solver.collider_reactions()
    assert (reactions[:, 0, 2] > 0.0).all()
    # the tangential reaction opposes the +x motion and stays inside the Coulomb cone
    assert (reactions[:, 0, 0] < 0.0).all()
    assert (reactions[:, 0, 0].abs() <= mu * reactions[:, 0, 2] * 1.05).all()
    # hold, then slide along +x at 0.1 m/s: the tangential reaction sits on the cone
    for i in range(20):
        scene.step()
    ratios = []
    for i in range(40):
        pos[..., 0] = -0.0092 + 0.1 * 2.5e-3 * (i + 1)
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
        reactions = scene.vbd_solver.collider_reactions()
        if i >= 20:
            ratios.append(-reactions[:, 0, 0] / reactions[:, 0, 2])
    ratio = torch.stack(ratios).mean(dim=0)
    assert_allclose(ratio, mu, rtol=0.1, atol=0.0)
    assert torch.isfinite(tissue.get_positions()).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_thin_wall_holds_a_pressing_collider_and_refuses_to_be_crossed(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
            batch_links_info=True,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    # a 3 mm sheet just above the table's layer, one tetrahedron thick
    wall = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.003),
            pos=(0.0, 0.0, 0.0026),
            nobisect=False,
            maxvolume=2e-8,
        ),
        material=gs.materials.VBD.Muscle(
            E=2e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=_prescribed_box_xml((0.0, 0.0, 0.03), (0.01, 0.01, 0.01)),
        ),
        material=gs.materials.Rigid(),
    )
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_prescribed_collider(box, collision_group=2, link=box.links[1])
    # the sheet's vertices weigh micrograms: a pair stiffness far above their inertia over h^2 (about 10 N/m here)
    # makes the per-sweep dual update overshoot, so the sheet's rules stay within three decades of it
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e4, friction=0.5, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(1, 2, stiffness=1e4, friction=0.5, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(0, 2, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.build(n_envs=n_envs)
    B = max(n_envs, 1)
    pos = torch.tensor([0.0, 0.0, 0.03], device=gs.device).expand(B, 1, 3).clone()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device).expand(B, 1, 4).clone()
    # the box bottom (z = 0.02) reaches the sheet top (z = 0.0041) after 15.9 mm; the sheet then closes its 0.1 mm
    # of clearance to the table layer and takes the remaining travel as compression. Each 1 mm layer holds the
    # sheet's vertices a millimetre off the surface it touches, so the 3 mm sheet ends about 1.6 mm thick
    for i in range(60):
        pos[..., 2] = 0.03 - 0.0164 * (i + 1) / 60
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
    # hold so the sheet settles and the force balance is static
    for _ in range(20):
        scene.step()
    reactions = scene.vbd_solver.collider_reactions()
    assert (reactions[:, 1, 2] > 1.0).all()
    assert_allclose(reactions[:, 0, 2], -reactions[:, 1, 2], rtol=0.1, atol=0.0)
    positions = wall.get_positions()
    assert (positions[..., 2].min() > -1e-4).all()
    assert torch.isfinite(positions).all()
    # driving the box to the table would crush the sheet to nothing: the engine refuses
    with pytest.raises(gs.GenesisException, match="crossed|margin|finite"):
        for i in range(40):
            pos[..., 2] = 0.0136 - 0.02 * (i + 1) / 40
            scene.vbd_solver.set_prescribed_targets(pos, quat)
            scene.step()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_opposed_colliders_squeeze_tissue_symmetrically(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
            batch_links_info=True,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.0, 0.0),
            nobisect=False,
            maxvolume=1e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    left = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=_prescribed_box_xml((-0.04, 0.0, 0.0), (0.01, 0.03, 0.03)),
        ),
        material=gs.materials.Rigid(),
    )
    right = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=_prescribed_box_xml((0.04, 0.0, 0.0), (0.01, 0.03, 0.03)),
        ),
        material=gs.materials.Rigid(),
    )
    scene.vbd_solver.add_prescribed_collider(left, collision_group=2, link=left.links[1])
    scene.vbd_solver.add_prescribed_collider(right, collision_group=2, link=right.links[1])
    scene.vbd_solver.add_contact_rule(1, 2, stiffness=1e5, friction=0.3, thickness=1e-3)
    scene.build(n_envs=n_envs)
    B = max(n_envs, 1)
    pos = torch.tensor([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]], device=gs.device).expand(B, 2, 3).clone()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device).expand(B, 2, 4).clone()
    # each plate travels 14 mm: 9 mm of clearance and 5 mm of squeeze per side
    for i in range(60):
        travel = 0.014 * (i + 1) / 60
        pos[..., 0, 0] = -0.04 + travel
        pos[..., 1, 0] = 0.04 - travel
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
    reactions = scene.vbd_solver.collider_reactions()
    assert (reactions[:, 0, 0] < -2.0).all()
    assert_allclose(reactions[:, 0, 0], -reactions[:, 1, 0], rtol=0.05, atol=0.0)
    positions = tissue.get_positions()
    assert (positions[..., 0].mean(dim=-1).abs() < 5e-4).all()
    assert (positions[..., 0].max(dim=-1).values < 0.02 - 0.004).all()
    assert torch.isfinite(positions).all()


def test_failed_environment_latches_while_its_peer_continues_and_snapshots_replay(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
            batch_links_info=True,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            damping=2e-3,
            raise_on_env_failure=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.0, 0.0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    meal = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=_prescribed_box_xml((0.0, 0.0, 0.06), (0.01, 0.01, 0.01)),
        ),
        material=gs.materials.Rigid(),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.0, 0.0211),
            nobisect=False,
            maxvolume=1e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_prescribed_collider(meal, collision_group=2, link=meal.links[1])
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(1, 2, stiffness=1e5, friction=0.3, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(0, 2, stiffness=1e5, friction=0.3, thickness=1e-3)
    scene.build(n_envs=2)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device).expand(2, 1, 4).clone()
    pos = torch.tensor([0.0, 0.0, 0.06], device=gs.device).expand(2, 1, 3).clone()
    # both environments press 4 mm into the cube (the box bottom meets the top after 8.9 mm)
    for i in range(40):
        pos[..., 2] = 0.06 - 0.0129 * (i + 1) / 40
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
    # a snapshot taken between two commands carries the prescribed phase: the same commands replay bit-exactly
    snapshot = scene.get_state()
    for i in range(5):
        pos[..., 2] = 0.0471 - 0.0005 * (i + 1)
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
    expected_tissue = tissue.get_positions().clone()
    expected_meal = meal.get_links_pos().clone()
    scene.reset(snapshot)
    for i in range(5):
        pos[..., 2] = 0.0471 - 0.0005 * (i + 1)
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
    assert_allclose(tissue.get_positions(), expected_tissue, atol=1e-9)
    assert_allclose(meal.get_links_pos(), expected_meal, atol=1e-9)
    # environment 0 is commanded through the table while environment 1 holds
    for i in range(40):
        pos[0, 0, 2] = 0.0446 - 0.06 * (i + 1) / 40
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
        status = scene.vbd_solver.env_status()
        if bool(status.is_failed[0]):
            break
    status = scene.vbd_solver.env_status()
    assert bool(status.is_failed[0]) and not bool(status.is_failed[1])
    assert int(status.failed_substep[0]) > 0 and int(status.failed_substep[1]) == -1
    assert int(status.errno[0]) != 0
    frozen_tissue = tissue.get_positions()[0].clone()
    frozen_meal = meal.get_links_pos()[0].clone()
    peer_before = tissue.get_positions()[1].clone()
    for i in range(10):
        pos[0, 0, 2] -= 0.0015
        pos[1, 0, 2] -= 0.0002
        scene.vbd_solver.set_prescribed_targets(pos, quat)
        scene.step()
    # the failed environment keeps the state of its failed attempt; its peer keeps advancing
    assert_allclose(tissue.get_positions()[0], frozen_tissue, atol=0.0)
    assert_allclose(meal.get_links_pos()[0], frozen_meal, atol=0.0)
    assert (tissue.get_positions()[1] - peer_before).abs().max() > 1e-4
    assert_allclose(meal.get_links_pos()[1, 1, 2], pos[1, 0, 2], atol=1e-6)
    with pytest.raises(gs.GenesisException, match="diagnostic only"):
        scene.get_state()
    # resetting the failed environment alone clears its latch and leaves the peer untouched
    peer_before = tissue.get_positions()[1].clone()
    scene.reset(snapshot, envs_idx=[0])
    status = scene.vbd_solver.env_status()
    assert not bool(status.is_failed[0]) and int(status.failed_substep[0]) == -1 and int(status.errno[0]) == 0
    assert_allclose(tissue.get_positions()[1], peer_before, atol=0.0)
    assert_allclose(
        tissue.get_positions()[0], snapshot.solvers_state[scene.sim.solvers.index(scene.vbd_solver)].pos[0], atol=0.0
    )
    pos[0, 0, 2] = 0.0471
    scene.vbd_solver.set_prescribed_targets(pos, quat)
    scene.step()
    assert not bool(scene.vbd_solver.env_status().is_failed.any())


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
def test_momentum_balances_gravity_and_the_reported_table_impulse(n_iterations, show_viewer, momentum_budget):
    budget = momentum_budget["reaction_on_table"][str(n_iterations)]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2.5e-3,
            substeps=4,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.Euler,
        ),
        vbd_options=gs.options.VBDOptions(
            n_iterations=n_iterations,
            floor_height=-10.0,
            damping=2e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.02, 0.0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.3, 0.3, 0.02),
            pos=(0.0, 0.0, -0.01),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.04, 0.0, 0.0231),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
        ),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.0, 0.0231),
            nobisect=False,
            maxvolume=1e-5,
        ),
        material=gs.materials.VBD.Muscle(
            E=1e5,
            nu=0.3,
            collision_group=1,
        ),
    )
    rest = tensor_to_array(tissue.init_positions)
    tissue.add_rigid_attachments(np.flatnonzero(rest[:, 0] > rest[:, 0].max() - 1e-5), bone.links[0])
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_rigid_collider(bone.links[0], collision_group=2)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.vbd_solver.add_contact_rule(0, 2, stiffness=1e5, friction=0.5, thickness=1e-3)
    scene.build()
    masses = qd_to_torch(scene.vbd_solver.verts_info.mass)
    mass_bone = float(bone.links[0].inertial_mass)
    total_mass = float(masses.sum()) + mass_bone
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64, device=masses.device)
    table_origin = torch.tensor([0.0, 0.0, -0.01], dtype=torch.float64, device=masses.device)
    linear0, angular0, x_prev, m, x_bone_prev = _system_momentum(tissue, bone, masses)
    gravity_torque = torch.zeros(3, dtype=torch.float64, device=masses.device)
    worst_linear = worst_angular = 0.0
    scene.vbd_solver.clear_collider_impulses()
    for i in range(100):
        scene.step()
        linear, angular, x, m, x_bone = _system_momentum(tissue, bone, masses)
        # gravity torque about the origin, trapezoid over the step
        torque = 0.5 * (torch.linalg.cross(x_prev, m * gravity).sum(0) + torch.linalg.cross(x, m * gravity).sum(0))
        torque = torque + 0.5 * mass_bone * (
            torch.linalg.cross(x_bone_prev, gravity) + torch.linalg.cross(x_bone, gravity)
        )
        gravity_torque = gravity_torque + torque * scene.dt
        x_prev, x_bone_prev = x, x_bone
        # the table's impulse is the force on the table; the system receives its opposite, transported to the origin
        impulse = scene.vbd_solver.collider_impulses()[0, 0].double()
        on_system_force = -impulse[:3]
        on_system_torque = -(impulse[3:] + torch.linalg.cross(table_origin, impulse[:3]))
        expected_linear = linear0 + total_mass * gravity * ((i + 1) * scene.dt) + on_system_force
        expected_angular = angular0 + gravity_torque + on_system_torque
        worst_linear = max(worst_linear, float((linear - expected_linear).norm()))
        worst_angular = max(worst_angular, float((angular - expected_angular).norm()))
    # a duplicated integration step adds a step of gravity that no impulse accounts for, and fails this bound
    assert (
        worst_linear
        <= budget["linear_momentum"]["atol"] + budget["linear_momentum"]["rtol"] * budget["linear_momentum"]["scale"]
    )
    assert (
        worst_angular
        <= budget["angular_momentum"]["atol"] + budget["angular_momentum"]["rtol"] * budget["angular_momentum"]["scale"]
    )
    # at rest on the table the accumulated impulse is the weight over the run
    weight_impulse = total_mass * 9.81 * 100 * scene.dt
    residual = float(impulse[2]) + weight_impulse
    assert (
        abs(residual)
        <= budget["reaction_impulse"]["atol"] + budget["reaction_impulse"]["rtol"] * budget["reaction_impulse"]["scale"]
    )
    # No sign is asserted here. Unlike the closed fixture this one carries real momentum and gravity, and the
    # residual measured -3.91 percent of the weight at 4 sweeps against +0.49 percent at 8: the direction is not
    # stable across sweep counts, so only the magnitude is a property of the solver worth freezing.
    assert torch.isfinite(tissue.get_positions()).all()
