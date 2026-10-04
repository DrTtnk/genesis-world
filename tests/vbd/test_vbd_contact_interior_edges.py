"""Swept edge boxes find contacts whose four endpoints miss the search region."""

import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.vbd_contact import (
    EDGE_EDGE,
    func_cv_pos_prev,
    func_may_collide,
    func_pair_distance,
    func_sweep_bound,
    func_swept_lower_bound,
)
from genesis.utils.array_class import ErrorCode
from genesis.utils.misc import qd_to_torch, tensor_to_array


@qd.kernel
def kernel_exhaustive_edge_candidates(solver: qd.template(), contact: qd.template(), expected: qd.template()):
    """Use the production swept narrowphase on every eligible edge pair, without the spatial hash."""
    for i_e, j_e, i_b in qd.ndrange(contact.n_edges, contact.n_edges, solver._B):
        expected[i_e, j_e, i_b] = 0
        if j_e > i_e:
            ea = contact.edge_cv[i_e]
            eb = contact.edge_cv[j_e]
            if func_may_collide(ea[0], eb[0], contact):
                a0 = func_cv_pos_prev(0, ea[0], i_b, solver, contact)
                b0 = func_cv_pos_prev(0, ea[1], i_b, solver, contact)
                c0 = func_cv_pos_prev(0, eb[0], i_b, solver, contact)
                d0 = func_cv_pos_prev(0, eb[1], i_b, solver, contact)
                a = contact.cv_pred[ea[0], i_b]
                b = contact.cv_pred[ea[1], i_b]
                c = contact.cv_pred[eb[0], i_b]
                d = contact.cv_pred[eb[1], i_b]
                distance = func_swept_lower_bound(
                    func_pair_distance(a0, b0, c0, d0, EDGE_EDGE),
                    func_pair_distance(a, b, c, d, EDGE_EDGE),
                    func_sweep_bound(a - a0, b - b0, c - c0, d - d0, EDGE_EDGE),
                )
                thickness = contact.rule_thickness[contact.cv_info[ea[0]].group, contact.cv_info[eb[0]].group]
                if distance < thickness + contact.margin_max:
                    expected[i_e, j_e, i_b] = 1


def _crossed_bars(n_envs, show_viewer, *, cell_size=None, cell_cap=256, sweep_cap=512,
                  raise_on_env_failure=True):
    thickness = 2e-4
    gap = 1.5e-4
    width = 0.006
    length = 0.06
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=2,
            floor_height=-float("inf"),
            contact_margin=thickness,
            contact_cell_size=cell_size,
            contact_cell_cap=cell_cap,
            contact_sweep_cell_cap=sweep_cap,
            raise_on_env_failure=raise_on_env_failure,
        ),
        show_viewer=show_viewer,
    )
    lower = scene.add_entity(
        morph=gs.morphs.Box(size=(length, width, width), pos=(0.0, 0.0, 0.0), fixed=True),
        material=gs.materials.Rigid(),
    )
    upper = scene.add_entity(
        morph=gs.morphs.Box(size=(width, length, width), pos=(0.0, 0.0, width + gap)),
        material=gs.materials.Rigid(),
    )
    solver = scene.vbd_solver
    solver.add_rigid_link(upper.links[0])
    solver.add_rigid_collider(lower.links[0], collision_group=0)
    solver.add_rigid_collider(upper.links[0], collision_group=1)
    solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.0, thickness=thickness)
    scene.build(n_envs=n_envs)
    return scene, lower, upper


@pytest.mark.parametrize("n_envs", [0, 2])
def test_crossed_long_edges_are_candidates_before_surface_crossing(n_envs, show_viewer):
    scene, lower, upper = _crossed_bars(n_envs, show_viewer)
    upper.set_dofs_velocity((0.0, 0.0, -0.02, 0.0, 0.0, 0.0))
    scene.step()
    # The long horizontal edges meet at their interiors. Endpoints are 30 mm from the center,
    # outside the other edge's 0.4 mm search reach. The surfaces have not crossed.
    lower_top = np.asarray(tensor_to_array(lower.get_verts()))[..., 2].max(axis=-1)
    upper_bottom = np.asarray(tensor_to_array(upper.get_verts()))[..., 2].min(axis=-1)
    separation = upper_bottom - lower_top
    assert np.all((0.0 < separation) & (separation < 2e-4))
    pairs = scene.vbd_solver.contact.snapshot()
    assert (pairs.n_pt == 0).all()
    for i_b in range(max(n_envs, 1)):
        n_ee = int(pairs.n_ee[i_b])
        assert n_ee > 0, "crossed edge interiors in the contact layer need EE candidates"
        edge_pairs = list(zip(pairs.ee_a[i_b, :n_ee].tolist(), pairs.ee_b[i_b, :n_ee].tolist()))
        assert len(set(edge_pairs)) == n_ee, "shared cells and hash aliases must not duplicate EE pairs"


@pytest.mark.parametrize("cell_size,cell_cap,sweep_cap", [(0.012, 256, 512), (0.003, 256, 512), (0.001, 8192, 8192)])
def test_edge_hash_pairset_matches_exhaustive_narrowphase_at_different_cell_sizes(
    cell_size, cell_cap, sweep_cap, show_viewer
):
    scene, _, upper = _crossed_bars(
        2, show_viewer, cell_size=cell_size, cell_cap=cell_cap, sweep_cap=sweep_cap
    )
    upper.set_dofs_velocity((0.0, 0.0, -0.02, 0.0, 0.0, 0.0))
    scene.step()
    solver = scene.vbd_solver
    contact = solver.contact
    expected = qd.field(dtype=gs.qd_int, shape=(contact.n_edges, contact.n_edges, solver._B))
    kernel_exhaustive_edge_candidates(solver, contact, expected)
    reference = tensor_to_array(qd_to_torch(expected))
    snapshot = contact.snapshot()
    for i_b in range(2):
        n_ee = int(snapshot.n_ee[i_b])
        actual = list(zip(snapshot.ee_a[i_b, :n_ee].tolist(), snapshot.ee_b[i_b, :n_ee].tolist()))
        brute = {tuple(pair) for pair in np.argwhere(reference[:, :, i_b] == 1)}
        assert brute
        assert len(actual) == len(set(actual)), "one pair must survive each shared-cell/hash alias only once"
        assert set(actual) == brute
    if cell_size == 0.001:
        # A long 60 mm edge spans more than the entire hash table, forcing repeated hash buckets.
        cells = tensor_to_array(qd_to_torch(contact.edge_hi)) - tensor_to_array(qd_to_torch(contact.edge_lo))
        assert np.max(cells[..., 0:2]) > contact.hash_buckets


def test_static_long_edge_is_not_limited_by_vertex_motion_cap(show_viewer):
    scene, _, _ = _crossed_bars(0, show_viewer, cell_size=0.0008, cell_cap=8192, sweep_cap=1)
    scene.step()
    contact = scene.vbd_solver.contact
    lo = tensor_to_array(qd_to_torch(contact.edge_lo))
    hi = tensor_to_array(qd_to_torch(contact.edge_hi))
    spans = hi - lo + 1
    assert np.max(np.prod(spans, axis=-1)) > 512
    assert not bool(scene.vbd_solver.env_status().is_failed[0])
    assert int(contact.snapshot().n_ee[0]) > 0


def test_moving_vertex_sweep_still_fails_loudly(show_viewer):
    scene, _, upper = _crossed_bars(
        0, show_viewer, cell_size=0.003, sweep_cap=1, raise_on_env_failure=False
    )
    upper.set_dofs_velocity((0.0, 0.0, -1.0, 0.0, 0.0, 0.0))
    scene.step()
    status = scene.vbd_solver.env_status()
    assert bool(status.is_failed[0])
    assert int(status.errno[0]) & int(ErrorCode.OVERFLOW_VBD_CONTACT_SWEEP)
