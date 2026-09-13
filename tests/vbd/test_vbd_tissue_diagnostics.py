"""Tissue-validity diagnostic of the VBD solver: the minimum tet Jacobian ratio J/J0 (signed volume now over
signed rest volume, so an uninverted tet is near +1 and an inverted one is negative), the inverted tet count, and
the persistent-inversion latch built on the same per-environment failure fields the contact code uses.

A capsule bolus inflated inside a hollow tube (`trimesh.creation.annulus`) is the driver: growing it well past the
tube's rest lumen radius pushes the inner wall's tets past zero volume. Some configurations recover on their own
within a handful of substeps (the stable neo-Hookean material rides through it); others do not, and that is the
case the persistent-inversion latch exists to catch.
"""

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose

BAR = dict(size=(0.3, 0.04, 0.04), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=1e-6)
R_MIN, R_MAX, HEIGHT, SECTIONS = 0.03, 0.05, 0.15, 24


def _positions(entity, i_b=0):
    return tensor_to_array(entity.get_positions())[i_b]


def _tet_volume_ratio(pos0, pos1, elems):
    """Signed tet volume at `pos1` over signed tet volume at `pos0`, cast to float32 to match the kernel's own
    precision: `det(F)` is exactly this ratio, since `Ds = Ds1` and the rest inverse comes from `pos0`."""
    def shape(q):
        return np.stack([q[elems[:, 1]] - q[elems[:, 0]], q[elems[:, 2]] - q[elems[:, 0]], q[elems[:, 3]] - q[elems[:, 0]]], axis=-1)
    vol0 = np.linalg.det(shape(pos0.astype(np.float32)).astype(np.float64)).astype(np.float32)
    vol1 = np.linalg.det(shape(pos1.astype(np.float32)).astype(np.float64)).astype(np.float32)
    return vol1 / vol0


def _tube_scene(n_iterations, damping, max_consecutive_inverted_substeps=1000, n_envs=None, raise_on_env_failure=True):
    ring_path = "/tmp/vbd_tissue_diagnostics_ring.obj"
    trimesh.creation.annulus(r_min=R_MIN, r_max=R_MAX, height=HEIGHT, sections=SECTIONS).export(ring_path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(
            n_iterations=n_iterations,
            floor_height=-10.0,
            damping=damping,
            max_consecutive_inverted_substeps=max_consecutive_inverted_substeps,
            raise_on_env_failure=raise_on_env_failure,
        ),
        show_viewer=False,
    )
    body = scene.add_entity(
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3),
        morph=gs.morphs.Mesh(file=ring_path, pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=2e-5),
    )
    if n_envs is None:
        scene.build()
    else:
        scene.build(n_envs=n_envs)
    return scene, body


@pytest.mark.required
def test_rest_mesh_reports_unit_ratio_and_no_inverted_tets():
    scene, body = _tube_scene(n_iterations=2, damping=0.005)
    for _ in range(5):
        scene.step()
    diag = scene.vbd_solver.tissue_diagnostics()
    assert float(diag.min_j_ratio[0]) == pytest.approx(1.0, abs=1e-3)
    assert int(diag.n_inverted[0]) == 0


@pytest.mark.required
def test_inverted_tet_count_matches_a_numpy_computation_from_positions_and_elems():
    scene, body = _tube_scene(n_iterations=2, damping=0.005)
    pos0 = _positions(body)
    elems = body.elems

    center = np.zeros((1, 3))
    vel = np.zeros((1, 3))
    scene.vbd_solver.set_bolus(center, np.array([R_MIN * 2.2]), vel, 0.0)
    for _ in range(3):
        scene.step()

    diag = scene.vbd_solver.tissue_diagnostics()
    pos1 = _positions(body)
    ratio = _tet_volume_ratio(pos0, pos1, elems)

    assert int(diag.n_inverted[0]) > 0, "the drive must actually invert tets for this test to be meaningful"
    assert int(diag.n_inverted[0]) == int((ratio <= 0.0).sum())
    assert float(diag.min_j_ratio[0]) == pytest.approx(float(ratio.min()), abs=1e-4)


@pytest.mark.required
def test_transient_inversion_recovers_without_latching():
    """A brief inversion (the wall overshoots, then settles) must not trip the latch: the environment keeps
    stepping and the inverted count returns to zero on its own."""
    scene, body = _tube_scene(n_iterations=8, damping=0.05, max_consecutive_inverted_substeps=10)
    center = np.zeros((1, 3))
    vel = np.zeros((1, 3))
    radii = [R_MIN * 2.2, R_MIN * 2.0, R_MIN * 1.6, R_MIN * 1.2, R_MIN * 0.9]
    saw_inversion = False
    for r in radii:
        scene.vbd_solver.set_bolus(center, np.array([r]), vel, 0.0)
        scene.step()
        saw_inversion = saw_inversion or int(scene.vbd_solver.tissue_diagnostics().n_inverted[0]) > 0
        assert not bool(scene.vbd_solver.env_status().is_failed[0])

    recovered = False
    pos_before = _positions(body).copy()
    for _ in range(20):
        scene.step()
        assert not bool(scene.vbd_solver.env_status().is_failed[0])
        if int(scene.vbd_solver.tissue_diagnostics().n_inverted[0]) == 0:
            recovered = True

    assert saw_inversion, "the drive must actually invert tets for this test to be meaningful"
    assert recovered
    assert float(scene.vbd_solver.tissue_diagnostics().min_j_ratio[0]) > 0.0
    # the environment kept advancing after the episode, it did not freeze
    assert np.abs(_positions(body) - pos_before).max() > 1e-6


@pytest.mark.required
def test_persistent_inversion_latches_while_its_peer_continues(precision):
    """An environment whose bolus keeps it inverted for longer than the configured limit latches, exactly like a
    contact failure; its batch peer, never inverted, is unaffected.

    The peer never had a bolus, so its "stayed exactly at rest" checks hold to machine precision on float64.
    On the GPU float32 backend, measured 2026-09-13, the worst single component drifted by 1.31e-9 (after the
    five extra steps; the earlier check, right at the latch, measured 1.21e-9) - float32 rounding noise on
    an untouched batch peer, not a growing error, since five more steps barely moved it. The float32 budget
    is 2.0 times the larger of the two measured floors.
    """
    peer_at_rest_atol = 1e-9 if precision == "64" else 2.6e-9
    scene, body = _tube_scene(
        n_iterations=2, damping=0.005, max_consecutive_inverted_substeps=5, n_envs=2, raise_on_env_failure=False
    )
    center = np.zeros((2, 3))
    vel = np.zeros((2, 3))
    peer_before = _positions(body, i_b=1).copy()
    status = None
    for _ in range(30):
        # env 0 is driven far past its lumen radius and held there; env 1's bolus stays disabled
        scene.vbd_solver.set_bolus(center, np.array([R_MIN * 6.0, -1.0]), vel, 0.0)
        scene.step()
        status = scene.vbd_solver.env_status()
        if bool(status.is_failed[0]):
            break

    assert bool(status.is_failed[0]) and not bool(status.is_failed[1])
    assert int(status.failed_substep[0]) > 0 and int(status.failed_substep[1]) == -1
    assert int(status.errno[0]) & int(gs.utils.array_class.ErrorCode.VBD_TISSUE_PERSISTENT_INVERSION)
    assert int(status.errno[1]) == 0
    # env 1 never had a bolus: it stayed exactly at rest
    assert_allclose(_positions(body, i_b=1), peer_before, atol=peer_at_rest_atol)

    frozen0 = _positions(body, i_b=0).copy()
    frozen_diag0 = scene.vbd_solver.tissue_diagnostics()
    for _ in range(5):
        scene.vbd_solver.set_bolus(center, np.array([R_MIN * 6.0, -1.0]), vel, 0.0)
        scene.step()
    # the failed environment keeps the state (and the diagnostic) of its failed attempt; its peer is untouched
    assert_allclose(_positions(body, i_b=0), frozen0, atol=0.0)
    assert_allclose(_positions(body, i_b=1), peer_before, atol=peer_at_rest_atol)
    diag = scene.vbd_solver.tissue_diagnostics()
    assert float(diag.min_j_ratio[0]) == float(frozen_diag0.min_j_ratio[0])
    assert int(diag.n_inverted[0]) == int(frozen_diag0.n_inverted[0])


@pytest.mark.required
def test_diagnostic_reports_a_clean_state_when_nothing_inverts():
    """A correctness statement, not a timing one: over many substeps of a body that stays at rest, the diagnostic
    reports the minimum ratio at (numerically) +1 and zero inverted tets throughout, every substep."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=10, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(n_iterations=4, floor_height=-1.0),
        show_viewer=False,
    )
    bar = scene.add_entity(material=gs.materials.VBD.Muscle(E=1e5, nu=0.3), morph=gs.morphs.Box(**BAR))
    scene.build()
    for _ in range(20):
        scene.step()
        diag = scene.vbd_solver.tissue_diagnostics()
        assert int(diag.n_inverted[0]) == 0
        assert float(diag.min_j_ratio[0]) == pytest.approx(1.0, abs=1e-4)
    assert not bool(scene.vbd_solver.env_status().is_failed[0])
