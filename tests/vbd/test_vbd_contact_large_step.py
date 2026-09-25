"""Resting rigid contact at a 1 ms substep: a light bone must settle on a plate, not bounce off it.

At the python head's 62.5 us substep a 0.9 g bone dropped onto a plate settles in the 0.2 mm layer with no
residual motion. At 1 ms, sixteen times the substep, the same bone bounced up to 4 mm and finally crossed the
plate, while its block ended every substep with a 50 N contact force against a 9 mN weight. The rigid block
summed each contact vertex's own curvature k w^2 n n^T, which drops the cross terms between participants of one
pair that ride on the same bone: an edge of the bone against an edge of the plate is stiffer than the sum over
its two ends by (sum w)^2 / sum w^2. Where contact dominates the block, as it does once the pair stiffness is a
hundred times the bone's m/h^2, the Newton step overshot by that factor and the sweeps oscillated. Assembled per
pair, the block is exact and the bone rests. Warm-started multipliers (Giles et al. 2025 Sec. 3.7) were tried
alongside and made no difference here.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

DT = 1e-3
PLATE_TOP = 0.0
THICKNESS = 2e-4


def _bone_on_plate(patch_on_bone, n_iterations=10):
    """A 0.9 g bone whose underside starts 0.1 mm above the layer of a fixed plate. VBD owns the free bodies only
    when a rigid attachment exists, so a small tissue patch is attached either to the plate, which leaves the
    bone free of everything but contact, or on top of the bone, which rides on it."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=n_iterations, floor_height=-1e3, contact_margin=THICKNESS, contact_k_max_ratio=4.0,
            raise_on_env_failure=False,
        ),
        show_viewer=False,
    )
    plate = scene.add_entity(morph=gs.morphs.Box(size=(0.06, 0.06, 0.004), pos=(0.0, 0.0, PLATE_TOP - 0.002),
                                                 fixed=True), material=gs.materials.Rigid(rho=1500.0))
    bone = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(0.0, 0.0, 0.0018)),
                            material=gs.materials.Rigid(rho=1500.0))
    patch = scene.add_entity(
        morph=gs.morphs.Box(size=(0.003, 0.003, 0.003), pos=(0.0, 0.0, 0.0048) if patch_on_bone else (0.025, 0.025, 0.0015),
                            nobisect=False, maxvolume=4e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4, collision_group=2),
    )
    rest = tensor_to_array(patch.init_positions)
    patch.add_rigid_attachments([int(v) for v in np.argsort(rest[:, 2])[:4]], (bone if patch_on_bone else plate).links[0])
    solver = scene.sim.vbd_solver
    solver.add_rigid_collider(plate.links[0], collision_group=0)
    solver.add_rigid_collider(bone.links[0], collision_group=1)
    solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.1, thickness=THICKNESS)
    scene.build()
    return scene, bone, patch


def _settle(scene, bone):
    heights, speeds = [], []
    for _ in range(300):
        scene.step()
        heights.append(float(tensor_to_array(bone.get_pos())[2]) - 0.0015 - PLATE_TOP)
        speeds.append(float(tensor_to_array(bone.get_vel())[2]))
    status = scene.vbd_solver.env_status()
    reaction = -float(tensor_to_array(scene.vbd_solver.collider_reactions())[0, 0, 2])
    return status, np.array(heights[150:]), np.array(speeds[150:]), reaction


def _patch_weight(patch):
    rest = tensor_to_array(patch.init_positions)
    elems = np.asarray(patch.elems)
    return 9.81 * patch.material.rho * np.abs(np.linalg.det(rest[elems[:, 1:]] - rest[elems[:, :1]])).sum() / 6.0


BONE_WEIGHT = 9.81 * 1500.0 * 0.02 * 0.01 * 0.003


@pytest.mark.required
def test_a_light_bone_settles_on_a_plate_at_a_millisecond_substep():
    scene, bone, _ = _bone_on_plate(patch_on_bone=False)
    status, heights, speeds, reaction = _settle(scene, bone)
    print(f"failed={bool(status.is_failed[0])} errno={int(status.errno[0])}; late height "
          f"{1e3 * heights.min():.4f} .. {1e3 * heights.max():.4f} mm, |v_z| max {np.abs(speeds).max():.3g} m/s; "
          f"plate reaction {reaction:.5f} N against weight {BONE_WEIGHT:.5f} N")
    assert not bool(status.is_failed[0]), f"the bone crossed the plate (errno {int(status.errno[0])})"
    assert np.abs(speeds).max() < 1e-3, "a resting bone must not keep bouncing"
    assert heights.min() > 0.0 and heights.max() < THICKNESS * 1.05, "and it must rest within the contact layer"
    assert reaction == pytest.approx(BONE_WEIGHT, rel=0.02), "the layer must carry the weight, no more and no less"


@pytest.mark.xfail(strict=True, reason="at ten sweeps the attached patch's own mesh has not converged at a 1 ms "
                   "substep (its vertices' m/h^2 is ~1/500 of their element stiffness) and pulls on the bone; the "
                   "reaction is exact at 160 sweeps")
def test_a_bone_carrying_attached_tissue_loads_the_plate_with_both_weights():
    """The same bone with a tissue patch attached on top of it. At rest the plate carries both, which is only true
    if the attachment passes the patch's weight to the bone, no more."""
    scene, bone, patch = _bone_on_plate(patch_on_bone=True)
    status, heights, speeds, reaction = _settle(scene, bone)
    weight = BONE_WEIGHT + _patch_weight(patch)
    print(f"failed={bool(status.is_failed[0])}; |v_z| max {np.abs(speeds).max():.3g} m/s; "
          f"plate reaction {reaction:.5f} N against weight {weight:.5f} N")
    assert not bool(status.is_failed[0])
    assert np.abs(speeds).max() < 1e-3
    assert reaction == pytest.approx(weight, rel=0.02)


def _two_bones_falling(contact_ccd):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=10, floor_height=-1e3, contact_margin=THICKNESS, contact_k_max_ratio=4.0,
            contact_ccd=contact_ccd, raise_on_env_failure=False,
        ),
        show_viewer=False,
    )
    plate = scene.add_entity(morph=gs.morphs.Box(size=(0.1, 0.06, 0.004), pos=(0.0, 0.0, PLATE_TOP - 0.002),
                                                 fixed=True), material=gs.materials.Rigid(rho=1500.0))
    # the lander hits the plate at 0.44 m/s, 0.44 mm a substep against the 0.2 mm layer; the other is still in
    # free fall 30 mm up and 40 mm away when that happens
    lander = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(-0.02, 0.0, 0.0115)),
                              material=gs.materials.Rigid(rho=1500.0))
    faller = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(0.02, 0.0, 0.06)),
                              material=gs.materials.Rigid(rho=1500.0))
    patch = scene.add_entity(
        morph=gs.morphs.Box(size=(0.003, 0.003, 0.003), pos=(0.045, 0.025, 0.0015), nobisect=False, maxvolume=4e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4, collision_group=3),
    )
    rest = tensor_to_array(patch.init_positions)
    patch.add_rigid_attachments([int(v) for v in np.argsort(rest[:, 2])[:4]], plate.links[0])
    solver = scene.sim.vbd_solver
    solver.add_rigid_collider(plate.links[0], collision_group=0)
    solver.add_rigid_collider(lander.links[0], collision_group=1)
    solver.add_rigid_collider(faller.links[0], collision_group=2)
    for group in (1, 2):
        solver.add_contact_rule(0, group, stiffness=1e5, friction=0.1, thickness=THICKNESS)
    scene.build()
    return scene, lander, faller


@pytest.mark.required
def test_a_fast_landing_is_stopped_by_the_layer_without_a_continuous_filter():
    """0.44 mm of approach a substep against a 0.2 mm layer. The swept candidate search finds the pair before the
    bone reaches it, and with the block's contact curvature exact the Newton step stops the bone within the
    substep, so no filter is needed; the distant bone, touching nothing, keeps exactly its free fall."""
    scene, lander, faller = _two_bones_falling(contact_ccd=False)
    for step in range(80):
        scene.step()
        t = DT * (step + 1)
        assert float(tensor_to_array(faller.get_vel())[2]) == pytest.approx(-9.81 * t, rel=1e-6)
    status = scene.vbd_solver.env_status()
    height = float(tensor_to_array(lander.get_pos())[2]) - 0.0015 - PLATE_TOP
    print(f"failed={bool(status.is_failed[0])} errno={int(status.errno[0])}; lander {1e3 * height:.4f} mm up")
    assert not bool(status.is_failed[0]), f"the lander crossed the plate (errno {int(status.errno[0])})"
    assert 0.0 < height < THICKNESS * 1.05
