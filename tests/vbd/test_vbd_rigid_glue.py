"""Glued vertices: tissue carried rigidly by a bone, with the tissue's forces in the bone's own block.

A rigid attachment is an augmented-Lagrangian constraint between a tissue vertex and a bone, and at the ten
sweeps a 1 ms substep can afford it settles on the wrong force: a 0.03 g patch attached on top of a resting
bone pulled it down with ten times its weight, and only 160 sweeps brought that to the patch's weight. A glued
vertex has no degrees of freedom and no multiplier. The bone carries it, and the elastic force of every
tetrahedron that touches it, with the cross curvature between glued corners of one element, is part of the
bone's 6x6 block, so the coupling is two-way and exact in one sweep. The reference is the attachment itself,
converged at 160 sweeps.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

DT = 1e-3
SIDE = 0.01


def _hanging_bone(glue, n_iterations):
    """A free 3 g bone hung under a 10 mm tissue column whose top face is pinned, joined at the column's bottom
    face by glue or by rigid attachments."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1e3),
        show_viewer=False,
    )
    column = scene.add_entity(
        morph=gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(0.0, 0.0, 0.1), nobisect=False, maxvolume=5e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.005), pos=(0.0, 0.0, 0.1 - 0.5 * SIDE - 0.0025)),
        material=gs.materials.Rigid(rho=1500.0),
    )
    rest = tensor_to_array(column.init_positions)
    bottom = np.flatnonzero(rest[:, 2] < rest[:, 2].min() + 1e-9)
    top = rest[:, 2] > rest[:, 2].max() - 1e-9
    if glue:
        column.add_rigid_glue(bottom, bone.links[0])
    else:
        column.add_rigid_attachments(bottom, bone.links[0])
    scene.build()
    column.set_pinned(top)
    return scene, bone


def _settled_sag(glue, n_iterations, steps=400):
    scene, bone = _hanging_bone(glue, n_iterations)
    z0 = float(tensor_to_array(bone.get_pos())[2])
    heights = []
    for _ in range(steps):
        scene.step()
        heights.append(float(tensor_to_array(bone.get_pos())[2]))
    heights = np.array(heights)
    return z0 - heights[-50:].mean(), np.ptp(heights[-50:]), z0 - heights.min()


@pytest.mark.required
def test_converged_glue_and_converged_attachment_hang_the_bone_at_the_same_height():
    """Both couplings solve the same physics; converged, they must agree to the last digits that matter."""
    glued, _, _ = _settled_sag(glue=True, n_iterations=160)
    attached, _, _ = _settled_sag(glue=False, n_iterations=160)
    print(f"sag at 160 sweeps: glued {1e6 * glued:.4f} um, attached {1e6 * attached:.4f} um")
    assert attached > 1e-5, "fixture sanity check: the column must stretch measurably under the bone"
    assert glued == pytest.approx(attached, rel=1e-3)


@pytest.mark.required
def test_glue_is_near_converged_at_a_sweep_count_where_the_attachment_is_not():
    """Measured: at 40 sweeps the attachment is 2.2 % off the converged sag and glue 0.1 %; at 10 sweeps 166 %
    and 23 %. What glue leaves at 10 sweeps is the column's own Gauss-Seidel convergence, not the coupling."""
    reference, _, _ = _settled_sag(glue=True, n_iterations=160)
    glued, ripple, peak = _settled_sag(glue=True, n_iterations=40)
    attached, _, _ = _settled_sag(glue=False, n_iterations=40)
    print(f"40 sweeps: glued {1e6 * glued:.3f} um, attached {1e6 * attached:.3f} um, converged {1e6 * reference:.3f} um")
    assert glued == pytest.approx(reference, rel=5e-3)
    assert abs(glued - reference) < 0.25 * abs(attached - reference)
    assert ripple < 0.01 * glued, "it must have settled"
    assert peak < 2.5 * glued, "released from rest, the overshoot of an undamped drop onto a spring is at most 2x"


def _bone_carrying_patch(glue_all):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=10, floor_height=-1e3, contact_margin=2e-4,
                                          contact_k_max_ratio=4.0, raise_on_env_failure=False),
        show_viewer=False,
    )
    plate = scene.add_entity(morph=gs.morphs.Box(size=(0.06, 0.06, 0.004), pos=(0.0, 0.0, -0.002), fixed=True),
                             material=gs.materials.Rigid(rho=1500.0))
    bone = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(0.0, 0.0, 0.0018)),
                            material=gs.materials.Rigid(rho=1500.0))
    patch = scene.add_entity(
        morph=gs.morphs.Box(size=(0.003, 0.003, 0.003), pos=(0.0, 0.0, 0.0048), nobisect=False, maxvolume=4e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4, collision_group=2),
    )
    rest = tensor_to_array(patch.init_positions)
    carried = np.arange(patch.n_vertices) if glue_all else np.argsort(rest[:, 2])[:4]
    patch.add_rigid_glue(carried, bone.links[0])
    solver = scene.sim.vbd_solver
    solver.add_rigid_collider(plate.links[0], collision_group=0)
    solver.add_rigid_collider(bone.links[0], collision_group=1)
    solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.1, thickness=2e-4)
    scene.build()
    return scene, bone, patch


@pytest.mark.required
def test_a_bone_carrying_glued_tissue_loads_the_plate_with_both_weights():
    """Every vertex of the patch glued, so no tissue convergence is involved: at ten sweeps the plate must carry
    exactly both weights. With four of twenty glued it is 17.5 % high at ten sweeps and exact at 160, which is
    the patch's own mesh converging, not the coupling."""
    scene, bone, patch = _bone_carrying_patch(glue_all=True)
    speeds = []
    for _ in range(300):
        scene.step()
        speeds.append(float(tensor_to_array(bone.get_vel())[2]))
    rest = tensor_to_array(patch.init_positions)
    elems = np.asarray(patch.elems)
    patch_weight = 9.81 * patch.material.rho * np.abs(np.linalg.det(rest[elems[:, 1:]] - rest[elems[:, :1]])).sum() / 6
    weight = 9.81 * 1500.0 * 0.02 * 0.01 * 0.003 + patch_weight
    reaction = -float(tensor_to_array(scene.vbd_solver.collider_reactions())[0, 0, 2])
    print(f"plate reaction {reaction:.6f} N against weight {weight:.6f} N; |v_z| max {np.abs(speeds[150:]).max():.2e}")
    assert not bool(scene.vbd_solver.env_status().is_failed[0])
    assert np.abs(speeds[150:]).max() < 1e-3
    assert reaction == pytest.approx(weight, rel=1e-3)


def test_glue_is_refused_where_it_has_no_forward_path_yet():
    """Strict over lenient: the continuous filter rescales tissue vertices apart from their bone, and the
    Rayleigh damping couples a vertex to its neighbours' motion, which the bone block does not assemble."""
    for options in (dict(contact_ccd=True), dict(damping=1e-3)):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.81)),
            rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
            vbd_options=gs.options.VBDOptions(n_iterations=4, floor_height=-1e3, **options),
            show_viewer=False,
        )
        column = scene.add_entity(
            morph=gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(0.0, 0.0, 0.1), nobisect=False, maxvolume=5e-9),
            material=gs.materials.VBD.Muscle(E=1e5, nu=0.4),
        )
        bone = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.02, 0.005), pos=(0.0, 0.0, 0.0925)),
                                material=gs.materials.Rigid(rho=1500.0))
        column.add_rigid_glue(np.array([0]), bone.links[0])
        with pytest.raises(gs.GenesisException, match="glue"):
            scene.build()


def test_a_vertex_cannot_be_both_glued_and_attached():
    scene = gs.Scene(show_viewer=False)
    column = scene.add_entity(
        morph=gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(0.0, 0.0, 0.1), nobisect=False, maxvolume=5e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4),
    )
    bone = scene.add_entity(morph=gs.morphs.Box(size=(0.02, 0.02, 0.005), pos=(0.0, 0.0, 0.0925)),
                            material=gs.materials.Rigid(rho=1500.0))
    column.add_rigid_glue(np.array([0, 1]), bone.links[0])
    with pytest.raises(gs.GenesisException, match="glued"):
        column.add_rigid_attachments(np.array([1]), bone.links[0])
