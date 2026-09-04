"""Vertex Block Descent solver: stable neo-Hookean tets, colored Gauss-Seidel, fiber actuation."""

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

BAR = dict(size=(0.3, 0.04, 0.04), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=1e-6)


def _bar_scene(material, n_iterations=10, substeps=10, gravity=(0.0, 0.0, 0.0), show_viewer=False):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=substeps, gravity=gravity),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations),
        show_viewer=show_viewer,
    )
    bar = scene.add_entity(material=material, morph=gs.morphs.Box(**BAR))
    scene.build()
    return scene, bar


def _positions(bar):
    return tensor_to_array(bar.get_positions())[0]


def _extent(pos, axis):
    return float(pos[:, axis].max() - pos[:, axis].min())


@pytest.mark.required
def test_coloring_gives_every_tet_four_distinct_colors(show_viewer):
    scene, bar = _bar_scene(gs.materials.VBD.Muscle(), show_viewer=show_viewer)
    solver = scene.vbd_solver

    perm = solver.color_perm.to_numpy()
    color = np.empty(solver.n_vertices, dtype=np.int64)
    for c in range(solver.n_colors):
        color[perm[solver.color_offsets[c] : solver.color_offsets[c + 1]]] = c
    tet_colors = color[bar.elems]

    assert bar.n_elements > 300
    assert solver.n_colors >= 4
    assert (np.sort(tet_colors, axis=1)[:, 1:] != np.sort(tet_colors, axis=1)[:, :-1]).all()


@pytest.mark.required
def test_rest_state_stays_put_and_has_the_analytic_rest_energy(show_viewer):
    mat = gs.materials.VBD.Muscle(E=1e5, nu=0.3)
    scene, bar = _bar_scene(mat, show_viewer=show_viewer)
    pos0 = _positions(bar)

    for _ in range(20):
        scene.step()

    pos1 = _positions(bar)
    assert np.abs(pos1 - pos0).max() < 1e-6

    lam_p = mat.lam + mat.mu
    vol_rest = scene.vbd_solver.elems_info.vol_rest.to_numpy().sum()
    assert scene.vbd_solver.compute_energy()[0] == pytest.approx(vol_rest * mat.mu**2 / (2 * lam_p), rel=1e-4)


@pytest.mark.required
def test_incremental_potential_never_increases_across_sweeps(show_viewer):
    scene, bar = _bar_scene(gs.materials.VBD.Muscle(E=1e5, nu=0.3), show_viewer=show_viewer)
    solver = scene.vbd_solver

    rng = np.random.default_rng(0)
    pos, vel = bar.get_state()
    noise = torch.as_tensor(rng.normal(0.0, 2e-3, size=pos.shape), dtype=pos.dtype, device=pos.device)
    solver._kernel_set_state((pos + noise).contiguous(), torch.zeros_like(vel))
    solver._kernel_predict(0)
    e_prev = solver.compute_energy()[0]
    e_start = e_prev

    def sweeps(n):
        nonlocal e_prev
        for _ in range(n):
            for c in range(solver.n_colors):
                solver._kernel_solve_color(0, solver.color_offsets[c], solver.color_offsets[c + 1])
            e = solver.compute_energy()[0]
            assert e <= e_prev * (1 + 1e-6)
            e_prev = e
        return e_prev

    e_100 = sweeps(100)
    e_200 = sweeps(100)
    assert e_100 < 0.9 * e_start
    assert e_100 - e_200 < 1e-6 * e_start


@pytest.mark.required
def test_actuation_contracts_along_the_fiber_at_constant_volume(show_viewer):
    """Fully actuated, every tet must reach the stress-free F = A^-1 = diag(s, 1/sqrt(s), 1/sqrt(s))."""
    gain = 0.3
    scene, bar = _bar_scene(gs.materials.VBD.Muscle(E=1e5, nu=0.3, n_groups=1, gain=gain), n_iterations=60, show_viewer=show_viewer)
    bar.set_muscle(np.zeros(bar.n_elements, dtype=np.int32), np.tile([1.0, 0.0, 0.0], (bar.n_elements, 1)))
    pos0 = _positions(bar)
    for step in range(300):
        bar.set_actuation([min(1.0, step / 100)])
        scene.step()
    pos1 = _positions(bar)
    assert np.isfinite(pos1).all()

    el = bar.elems
    shape = lambda p: np.stack([p[el[:, 1]] - p[el[:, 0]], p[el[:, 2]] - p[el[:, 0]], p[el[:, 3]] - p[el[:, 0]]], axis=-1)
    Ds, Dm = shape(pos1), shape(pos0)
    F = Ds @ np.linalg.inv(Dm)
    s = 1.0 - gain
    np.testing.assert_allclose(F.mean(axis=0), np.diag([s, 1 / np.sqrt(s), 1 / np.sqrt(s)]), atol=1e-2)
    np.testing.assert_allclose(np.linalg.det(Ds) / np.linalg.det(Dm), 1.0, atol=1e-3)


@pytest.mark.required
def test_opposed_fiber_groups_bend_the_bar_both_ways(show_viewer):
    """Contracting the top-side fibers puts the top on the inside of the curve, so the middle drops relative to the
    ends (negative sag); bottom-side fibers do the opposite; null control stays straight."""
    dev = {}
    for actu_top, actu_bottom in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
        scene, bar = _bar_scene(gs.materials.VBD.Muscle(E=1e5, nu=0.3, n_groups=2, gain=0.3), n_iterations=60, show_viewer=show_viewer)
        pos0 = _positions(bar)
        mid_y = pos0[bar.elems].mean(axis=1)[:, 1]
        group = np.full(bar.n_elements, -1, dtype=np.int32)
        group[mid_y > 0.008] = 0
        group[mid_y < -0.008] = 1
        assert (group == 0).sum() > 50 and (group == 1).sum() > 50
        bar.set_muscle(group, np.tile([1.0, 0.0, 0.0], (bar.n_elements, 1)))

        for step in range(300):
            r = min(1.0, step / 100)
            bar.set_actuation([actu_top * r, actu_bottom * r])
            scene.step()
        pos1 = _positions(bar)
        assert np.isfinite(pos1).all()

        ends = np.abs(pos0[:, 0]) > 0.14
        middle = np.abs(pos0[:, 0]) < 0.02
        sag = (pos1[middle, 1].mean() - pos1[ends, 1].mean()) - (pos0[middle, 1].mean() - pos0[ends, 1].mean())
        dev[(actu_top, actu_bottom)] = sag
        print(f"actu=({actu_top}, {actu_bottom}) sag={sag:.5f}", flush=True)

    assert abs(dev[(0.0, 0.0)]) < 1e-4
    assert dev[(1.0, 0.0)] < -0.01
    assert dev[(0.0, 1.0)] > 0.01
    assert abs(dev[(1.0, 0.0)] + dev[(0.0, 1.0)]) < 0.2 * abs(dev[(1.0, 0.0)])
