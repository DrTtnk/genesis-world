"""Where a substep's sweeps start, and what the contact pairs carry into it.

Every substep used to start its iterate at the inertial prediction y = x + h v + h^2 g. For two free bones tied
by ligaments and resting on each other at a 1 ms substep, that start puts both bones 9.8 um low, and the
ligaments to the fixed skull must lift the pair back every substep. In the lower bone's block the upper bone is
held at its low pose, so the lower one crosses a ligament's slack point while the ligament's curvature is still
in the block: the coupled 6x6 step then rolls each bone by about 2 mrad for a purely vertical force. Undoing that
roll is the slow mode of the pair; ten sweeps do not, and the leftover roll pumped a pendulum mode until the pair
turned over. A free body now starts where VBD's adaptive initialization puts it (Chen et al. 2024 Eq. 17):
x + h v + h^2 s a, with s the previous substep's acceleration along the external acceleration a, clamped to
[0, 1]. A falling body takes the whole prediction, a resting one none of it. The inertia target stays y.

A start that no longer pre-penetrates a resting contact leaves the contact force to be rebuilt within the
substep, and a pair used to restart at lam = 0 every substep: two bones stacked on a plate then jittered at
1.5 mm/s. The pairs now keep their multiplier and stiffness from one substep to the next, scaled by
alpha gamma and gamma as the augmented Lagrangian attachments already are (Giles et al. 2025 Eq. 19), and a
pair the candidate search finds again after a rebuild takes back what it had.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.vbd_mtu import LinkAnchor
from genesis.utils.misc import tensor_to_array

DT = 1e-3
G = 9.81


def _scene(n_iterations):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -G)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=n_iterations, floor_height=-1e3, contact_margin=2e-4, contact_k_max_ratio=400.0,
            raise_on_env_failure=False,
        ),
        show_viewer=False,
    )


def _glued_patch(scene, pos, link):
    """VBD owns the free bodies only when some tissue is coupled to a rigid link; a patch glued to a fixed body
    hands them over without touching them."""
    patch = scene.add_entity(
        morph=gs.morphs.Box(size=(0.003, 0.003, 0.003), pos=pos, nobisect=False, maxvolume=4e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4, collision_group=9),
    )
    patch.add_rigid_glue(np.arange(patch.n_vertices), link)


def _falling_and_resting_bones():
    """One bone resting on a fixed plate and one falling freely beside it."""
    scene = _scene(10)
    plate = scene.add_entity(morph=gs.morphs.Box(size=(0.06, 0.06, 0.004), pos=(0.0, 0.0, -0.002), fixed=True),
                             material=gs.materials.Rigid(rho=1500.0))
    resting = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(0.0, 0.0, 0.0017)),
                               material=gs.materials.Rigid(rho=1500.0))
    falling = scene.add_entity(morph=gs.morphs.Box(size=(0.01, 0.01, 0.003), pos=(0.1, 0.0, 0.5)),
                               material=gs.materials.Rigid(rho=1500.0))
    _glued_patch(scene, (0.025, 0.025, 0.0015), plate.links[0])
    solver = scene.sim.vbd_solver
    solver.add_rigid_collider(plate.links[0], collision_group=0)
    solver.add_rigid_collider(resting.links[0], collision_group=1)
    solver.add_contact_rule(0, 1, stiffness=1e3, friction=0.1, thickness=2e-4)
    scene.build()
    return scene, resting, falling


def _share(scene, entity):
    attachment = scene.sim.vbd_solver.rigid_attachment
    slot = int(attachment.free_slot.to_numpy()[entity.links[0].idx])
    return float(attachment.gravity_share.to_numpy()[slot, 0])


@pytest.mark.required
def test_a_falling_body_takes_the_whole_prediction_and_a_resting_one_none_of_it():
    scene, resting, falling = _falling_and_resting_bones()
    for _ in range(200):
        scene.step()
    assert _share(scene, falling) == pytest.approx(1.0, abs=1e-9)
    assert _share(scene, resting) < 1e-3
    assert abs(float(tensor_to_array(resting.get_vel())[2])) < 1e-6


SKULL, A, B = (0.0, 0.0, 0.12), (0.0, 0.0, 0.1), (0.002, 0.0, 0.1037)
SIZE_A, SIZE_B = (0.03, 0.01, 0.003), (0.012, 0.008, 0.004)


def _tied_pair():
    """Bone A hangs from a fixed skull by two ligaments; bone B rests on A and is tied to it by two more, all at
    1e4 N/m and at their slack length. A and B collide."""
    scene = _scene(10)
    skull = scene.add_entity(morph=gs.morphs.Box(size=(0.04, 0.03, 0.01), pos=SKULL, fixed=True),
                             material=gs.materials.Rigid(rho=1500.0))
    bone_a = scene.add_entity(morph=gs.morphs.Box(size=SIZE_A, pos=A), material=gs.materials.Rigid(rho=1500.0))
    bone_b = scene.add_entity(morph=gs.morphs.Box(size=SIZE_B, pos=B), material=gs.materials.Rigid(rho=1500.0))
    _glued_patch(scene, (0.015, 0.012, 0.1265), skull.links[0])
    solver = scene.sim.vbd_solver
    bands = []
    for (x, ox, lx), (y, oy, ly) in (
        ((skull, SKULL, (-0.012, 0.0, -0.005)), (bone_a, A, (-0.012, 0.0, 0.0015))),
        ((skull, SKULL, (0.012, 0.0, -0.005)), (bone_a, A, (0.012, 0.0, 0.0015))),
        ((bone_a, A, (-0.006, 0.004, 0.0015)), (bone_b, B, (-0.004, 0.004, -0.001))),
        ((bone_a, A, (0.006, -0.004, 0.0015)), (bone_b, B, (0.004, -0.004, -0.001))),
    ):
        slack = float(np.linalg.norm(np.add(ox, lx) - np.add(oy, ly)))
        solver.add_ligament([LinkAnchor(x.links[0], local_pos=lx), LinkAnchor(y.links[0], local_pos=ly)],
                            stiffness=1e4, slack_length=slack)
        bands.append((x, np.array(lx), y, np.array(ly), slack))
    for group, entity in enumerate((skull, bone_a, bone_b)):
        solver.add_rigid_collider(entity.links[0], collision_group=group)
    solver.add_contact_rule(1, 2, stiffness=1e3, friction=0.1, thickness=2e-4)
    scene.build()
    return scene, bone_a, bone_b, bands


def _pose(entity):
    return tensor_to_array(entity.get_pos()), gs.utils.geom.quat_to_R(tensor_to_array(entity.get_quat()))


def _energy(bones, bands):
    """Kinetic, gravitational and ligament energy; the contact layer's own is negligible at these forces."""
    total = 0.0
    for entity, size in bones:
        mass = 1500.0 * np.prod(size)
        inertia = mass / 12 * np.diag([size[1]**2 + size[2]**2, size[0]**2 + size[2]**2, size[0]**2 + size[1]**2])
        p, R = _pose(entity)
        v, w = tensor_to_array(entity.get_vel()), tensor_to_array(entity.get_ang())
        total += 0.5 * mass * v @ v + 0.5 * w @ (R @ inertia @ R.T) @ w + mass * G * p[2]
    for x, lx, y, ly, slack in bands:
        (px, Rx), (py, Ry) = _pose(x), _pose(y)
        total += 0.5 * 1e4 * max(float(np.linalg.norm(px + Rx @ lx - py - Ry @ ly)) - slack, 0.0) ** 2
    return total


@pytest.mark.required
def test_two_free_bones_tied_and_touching_do_not_gain_energy_at_ten_sweeps():
    scene, bone_a, bone_b, bands = _tied_pair()
    bones = ((bone_a, SIZE_A), (bone_b, SIZE_B))
    drop = 1500.0 * (np.prod(SIZE_A) + np.prod(SIZE_B)) * G * 1e-3  # the energy of a 1 mm drop of both bones
    start = _energy(bones, bands)
    gains = []
    for _ in range(200):
        scene.step()
        gains.append((_energy(bones, bands) - start) / drop)
    assert not bool(scene.sim.vbd_solver.env_status().is_failed[0])
    assert max(gains) < 1e-2, f"the pair gained {max(gains):.3f} of a 1 mm drop"


def _stacked_bones():
    """Two light free bones stacked on a fixed plate: one free-fixed contact and one free-free."""
    scene = _scene(10)
    plate = scene.add_entity(morph=gs.morphs.Box(size=(0.06, 0.06, 0.004), pos=(0.0, 0.0, -0.002), fixed=True),
                             material=gs.materials.Rigid(rho=1500.0))
    lower = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(0.0, 0.0, 0.0017)),
                             material=gs.materials.Rigid(rho=1500.0))
    upper = scene.add_entity(morph=gs.morphs.Box(size=(0.012, 0.008, 0.002), pos=(0.002, 0.0, 0.0044)),
                             material=gs.materials.Rigid(rho=1500.0))
    _glued_patch(scene, (0.025, 0.025, 0.0015), plate.links[0])
    solver = scene.sim.vbd_solver
    for group, entity in enumerate((plate, lower, upper)):
        solver.add_rigid_collider(entity.links[0], collision_group=group)
    solver.add_contact_rule(0, 1, stiffness=1e3, friction=0.1, thickness=2e-4)
    solver.add_contact_rule(1, 2, stiffness=1e3, friction=0.1, thickness=2e-4)
    scene.build()
    return scene, lower, upper


@pytest.mark.required
def test_stacked_bones_rest_when_every_substep_rebuilds_the_candidate_set():
    scene, lower, upper = _stacked_bones()
    speeds = []
    for _ in range(300):
        scene.step()
        speeds.append(max(abs(float(tensor_to_array(e.get_vel())[2])) for e in (lower, upper)))
    contact = scene.sim.vbd_solver.contact
    assert int(contact.rebuild_count.to_numpy()[0]) == 300, "the default search reach rebuilds every substep"
    assert max(speeds[150:]) < 1e-6
    weight = G * 1500.0 * (0.02 * 0.01 * 0.003 + 0.012 * 0.008 * 0.002)
    reaction = -float(tensor_to_array(scene.sim.vbd_solver.collider_reactions())[0, 0, 2])
    assert reaction == pytest.approx(weight, rel=0.01)


@pytest.mark.required
def test_a_snapshot_carries_the_start_share_and_the_contact_multipliers():
    scene, lower, upper = _stacked_bones()
    for _ in range(100):
        scene.step()
    snapshot = scene.get_state()
    for _ in range(5):
        scene.step()
    expected = np.stack([tensor_to_array(e.get_pos()) for e in (lower, upper)])
    scene.reset(snapshot)
    for _ in range(5):
        scene.step()
    replayed = np.stack([tensor_to_array(e.get_pos()) for e in (lower, upper)])
    np.testing.assert_array_equal(replayed, expected)
