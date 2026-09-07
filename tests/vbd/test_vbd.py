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
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1.0),  # the bar floats free
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
    assert scene.vbd_solver.compute_energy(0)[0] == pytest.approx(vol_rest * mat.mu**2 / (2 * lam_p), rel=1e-4)


@pytest.mark.required
def test_incremental_potential_never_increases_across_sweeps(show_viewer):
    scene, bar = _bar_scene(gs.materials.VBD.Muscle(E=1e5, nu=0.3), show_viewer=show_viewer)
    solver = scene.vbd_solver

    rng = np.random.default_rng(0)
    state = bar.get_state()
    pos, vel = state.pos, state.vel
    noise = torch.as_tensor(rng.normal(0.0, 2e-3, size=pos.shape), dtype=pos.dtype, device=pos.device)
    solver._kernel_set_state(0, (pos + noise).contiguous(), torch.zeros_like(vel))
    solver._kernel_predict(0)
    e_prev = solver.compute_energy(0)[0]
    e_start = e_prev

    def sweeps(n):
        nonlocal e_prev
        for _ in range(n):
            for c in range(solver.n_colors):
                solver._kernel_solve_color(0, solver.color_offsets[c], solver.color_offsets[c + 1])
            e = solver.compute_energy(0)[0]
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
    scene, bar = _bar_scene(gs.materials.VBD.Muscle(E=1e5, nu=0.3, n_groups=1, gain=gain), n_iterations=2, substeps=40, show_viewer=show_viewer)
    bar.set_muscle(np.zeros(bar.n_elements, dtype=np.int32), np.tile([1.0, 0.0, 0.0], (bar.n_elements, 1)))
    pos0 = _positions(bar)
    el = bar.elems
    shape = lambda p: np.stack([p[el[:, 1]] - p[el[:, 0]], p[el[:, 2]] - p[el[:, 0]], p[el[:, 3]] - p[el[:, 0]]], axis=-1)
    Dm_inv = np.linalg.inv(shape(pos0))
    F_sum, vol_sum, n = 0.0, 0.0, 0
    for step in range(300):
        bar.set_actuation([min(1.0, step / 100)])
        scene.step()
        if step >= 200:  # the bar is undamped, so average over the oscillation around equilibrium
            Ds = shape(_positions(bar))
            F_sum, vol_sum, n = F_sum + (Ds @ Dm_inv).mean(axis=0), vol_sum + np.linalg.det(Ds @ Dm_inv), n + 1
    assert np.isfinite(F_sum).all()

    s = 1.0 - gain
    np.testing.assert_allclose(F_sum / n, np.diag([s, 1 / np.sqrt(s), 1 / np.sqrt(s)]), atol=1e-2)
    np.testing.assert_allclose(vol_sum / n, 1.0, atol=1e-3)


@pytest.mark.required
def test_opposed_fiber_groups_bend_the_bar_both_ways(show_viewer):
    """Contracting the top-side fibers puts the top on the inside of the curve, so the middle drops relative to the
    ends (negative sag); bottom-side fibers do the opposite; null control stays straight."""
    dev = {}
    for actu_top, actu_bottom in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
        scene, bar = _bar_scene(gs.materials.VBD.Muscle(E=1e5, nu=0.3, n_groups=2, gain=0.3), n_iterations=2, substeps=40, show_viewer=show_viewer)
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


def _box_on_floor(material, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=20, gravity=(0.0, 0.0, -9.81)),
        vbd_options=gs.options.VBDOptions(n_iterations=2),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(material=material, morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05), nobisect=False, maxvolume=5e-5))
    scene.build()
    return scene, box


@pytest.mark.required
def test_block_rests_on_the_floor_without_sinking_or_creeping(show_viewer):
    scene, box = _box_on_floor(gs.materials.VBD.Muscle(E=1e6, nu=0.3), show_viewer)
    com0 = _positions(box).mean(axis=0)
    for _ in range(200):
        scene.step()
    pos = _positions(box)
    vel = box.get_state().vel
    assert pos[:, 2].min() > -2e-3
    assert np.abs(pos.mean(axis=0)[:2] - com0[:2]).max() < 1e-4
    assert np.abs(tensor_to_array(vel)).max() < 1e-2


@pytest.mark.required
def test_anisotropic_friction_stops_a_sliding_block_at_the_coulomb_distance(show_viewer):
    """A block kicked at v0 on a floor with coefficient mu stops after v0^2 / (2 mu g). Forward, backward and
    sideways see three different coefficients."""
    mu = dict(forward=0.1, backward=0.4, lateral=0.8)
    v0, g = 1.0, 9.81
    for direction, key in (((1.0, 0.0, 0.0), "forward"), ((-1.0, 0.0, 0.0), "backward"), ((0.0, 1.0, 0.0), "lateral")):
        scene, box = _box_on_floor(
            gs.materials.VBD.Muscle(E=1e6, nu=0.3, mu_forward=mu["forward"], mu_backward=mu["backward"], mu_lateral=mu["lateral"]),
            show_viewer,
        )
        for _ in range(40):  # settle on the floor first
            scene.step()
        state = box.get_state()
        pos, vel = state.pos, state.vel
        vel[:] = torch.as_tensor(np.array(direction) * v0, dtype=vel.dtype, device=vel.device)
        scene.vbd_solver._kernel_set_state(scene.sim.cur_substep_local, pos.contiguous(), vel.contiguous())
        com0 = _positions(box).mean(axis=0)
        for _ in range(400):
            scene.step()
        travelled = float(np.dot(_positions(box).mean(axis=0) - com0, direction))
        expected = v0**2 / (2 * mu[key] * g)
        print(f"{key}: travelled={travelled:.4f} expected={expected:.4f}", flush=True)
        assert travelled == pytest.approx(expected, rel=0.15)


@pytest.mark.required
def test_rayleigh_damping_kills_the_ringing_of_a_dropped_block(show_viewer):
    """Without damping a dropped block keeps bouncing on its own elasticity; with Rayleigh damping the peak
    speed after the first impact decays to a small fraction within a second."""
    peaks = {}
    for damping in (0.0, 0.02):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=5e-3, substeps=20, gravity=(0.0, 0.0, -9.81)),
            vbd_options=gs.options.VBDOptions(n_iterations=2, damping=damping),
            show_viewer=show_viewer,
        )
        box = scene.add_entity(material=gs.materials.VBD.Muscle(E=1e5, nu=0.3), morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.08), nobisect=False, maxvolume=5e-5))
        scene.build()
        speeds = []
        for _ in range(200):
            scene.step()
            speeds.append(float(np.abs(tensor_to_array(box.get_state().vel)).max()))
        speeds = np.array(speeds)
        peaks[damping] = (speeds[20:60].max(), speeds[160:].max())
        print(f"damping={damping}: peak speed early={peaks[damping][0]:.3f} late={peaks[damping][1]:.3f}", flush=True)
    assert peaks[0.02][1] < 0.1 * peaks[0.02][0]
    assert peaks[0.02][1] < 0.3 * peaks[0.0][1]


@pytest.mark.required
def test_sphere_bolus_inflating_inside_a_ring_stretches_it_to_the_sphere(show_viewer, asset_tmp_path):
    """A sphere grown inside a thick tube must carry the inner wall out to its own radius (minus the penalty
    penetration) without inverting a tet, and the ring keeps its volume within the material's compressibility."""
    import trimesh

    ring = str(asset_tmp_path / "vbd_ring.obj")
    trimesh.creation.annulus(r_min=0.05, r_max=0.08, height=0.3, sections=48).export(ring)  # long, so it cannot slide off the sphere
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=20, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(n_iterations=2, damping=0.005, floor_height=-1.0, contact_stiffness=1e5),
        show_viewer=show_viewer,
    )
    body = scene.add_entity(material=gs.materials.VBD.Muscle(E=1e5, nu=0.45), morph=gs.morphs.Mesh(file=ring, pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=1e-5))
    scene.build()
    pos0 = _positions(body)
    el = body.elems
    shape = lambda q: np.stack([q[el[:, 1]] - q[el[:, 0]], q[el[:, 2]] - q[el[:, 0]], q[el[:, 3]] - q[el[:, 0]]], axis=-1)
    vol0 = np.linalg.det(shape(pos0))
    inner = (np.linalg.norm(pos0[:, :2], axis=1) < 0.052) & (np.abs(pos0[:, 2]) < 0.03)

    center = np.zeros((1, 3)); vel = np.zeros((1, 3))
    for step in range(400):
        radius = 0.03 + 0.045 * min(1.0, step / 300)  # ends at 0.075: hoop stretch 1.5
        scene.vbd_solver.set_bolus(center, np.array([radius]), vel, 0.0)
        scene.step()
    pos1 = _positions(body)
    d_inner = np.linalg.norm(pos1[inner], axis=1)  # 3D distance: the sphere surface is not a cylinder
    vol1 = np.linalg.det(shape(pos1))
    print(f"inner-wall distance from centre min={d_inner.min():.4f} mean={d_inner.mean():.4f} (sphere 0.075), volume ratio={vol1.sum() / vol0.sum():.4f}, inverted={(vol1 < 0).sum()}", flush=True)
    assert np.isfinite(pos1).all()
    assert d_inner.min() > 0.075 - 1e-3
    assert abs(d_inner.mean() - 0.075) < 3e-3
    assert (vol1 > 0).all()
    assert abs(vol1.sum() / vol0.sum() - 1.0) < 0.1


@pytest.mark.required
def test_fiber_reinforcement_stops_the_actuated_bar_from_shortening(show_viewer):
    """The same actuated bar as the contraction test, with every tet reinforced along the fiber at a stiffness far
    above the tissue's: a spine. It must keep its length within a few percent where the plain bar shortens 30
    percent, and stay stable."""
    gain = 0.3
    lengths = {}
    for k_fiber in (0.0, 3e6):
        scene, bar = _bar_scene(gs.materials.VBD.Muscle(E=1e5, nu=0.3, n_groups=1, gain=gain), n_iterations=2, substeps=40, show_viewer=show_viewer)
        bar.set_muscle(np.zeros(bar.n_elements, dtype=np.int32), np.tile([1.0, 0.0, 0.0], (bar.n_elements, 1)))
        bar.set_fiber_stiffness(np.full(bar.n_elements, k_fiber))
        pos0 = _positions(bar)
        for step in range(300):
            bar.set_actuation([min(1.0, step / 100)])
            scene.step()
        pos1 = _positions(bar)
        assert np.isfinite(pos1).all()
        lengths[k_fiber] = _extent(pos1, 0) / _extent(pos0, 0)
        print(f"k_fiber={k_fiber:.0e}: length ratio {lengths[k_fiber]:.3f}", flush=True)
    assert lengths[0.0] < 0.75
    assert lengths[3e6] > 0.97


@pytest.mark.required
def test_hard_distance_constraints_make_a_vertex_chain_inextensible(show_viewer):
    """A chain of hard distance constraints along the top edge of the actuated bar must keep every segment within
    constraint_tol where the same bar without them shortens 30 percent; the sum of the segments is the spine
    length, and it must not drift either."""
    gain = 0.3
    results = {}
    for with_spine in (False, True):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=5e-3, substeps=40, gravity=(0.0, 0.0, 0.0)),
            vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1.0, constraint_tol=1e-4),
            show_viewer=show_viewer,
        )
        bar = scene.add_entity(material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, n_groups=1, gain=gain), morph=gs.morphs.Box(**BAR))
        pos0 = tensor_to_array(bar.init_positions)
        top = np.where((pos0[:, 1] > 0.019) & (pos0[:, 2] > 0.019))[0]
        chain = top[np.argsort(pos0[top, 0])]
        pairs = np.stack([chain[:-1], chain[1:]], axis=1)
        if with_spine:
            bar.add_distance_constraints(pairs)
        scene.build()
        bar.set_muscle(np.zeros(bar.n_elements, dtype=np.int32), np.tile([1.0, 0.0, 0.0], (bar.n_elements, 1)))
        p0 = _positions(bar)
        rest = np.linalg.norm(p0[pairs[:, 0]] - p0[pairs[:, 1]], axis=1)
        for step in range(300):
            bar.set_actuation([min(1.0, step / 100)])
            scene.step()
        p1 = _positions(bar)
        assert np.isfinite(p1).all()
        seg = np.linalg.norm(p1[pairs[:, 0]] - p1[pairs[:, 1]], axis=1)
        results[with_spine] = (np.abs(seg - rest).max(), seg.sum() / rest.sum(), _extent(p1, 0) / _extent(p0, 0))
        print(f"spine={with_spine}: max segment error={results[with_spine][0]:.2e} m, chain length ratio={results[with_spine][1]:.4f}, bar length ratio={results[with_spine][2]:.3f}"
              + (f", solver constraint error={scene.vbd_solver.constraint_error():.2e}, n_colors={scene.vbd_solver.n_colors}" if with_spine else ""), flush=True)
    assert results[False][2] < 0.75
    assert results[True][0] < 5e-4
    assert abs(results[True][1] - 1.0) < 2e-3


@pytest.mark.required
def test_bounded_distance_constraints_stop_at_the_bound_and_are_slack_inside(show_viewer):
    """The top-edge chain of the actuated bar with a lower bound at 90 percent of rest: the bar contracts freely
    until the segments reach the bound, then stops there. Segments must end within tolerance below 0.9 rest and the
    bar must be shorter than 0.95 (the bound did not act as an equality)."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=40, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1.0, constraint_tol=1e-4),
        show_viewer=show_viewer,
    )
    bar = scene.add_entity(material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, n_groups=1, gain=0.3), morph=gs.morphs.Box(**BAR))
    pos0 = tensor_to_array(bar.init_positions)
    top = np.where((pos0[:, 1] > 0.019) & (pos0[:, 2] > 0.019))[0]
    chain = top[np.argsort(pos0[top, 0])]
    pairs = np.stack([chain[:-1], chain[1:]], axis=1)
    rest = np.linalg.norm(pos0[pairs[:, 0]] - pos0[pairs[:, 1]], axis=1)
    bar.add_distance_constraints(pairs, lo=0.9 * rest, hi=1.5 * rest)
    scene.build()
    bar.set_muscle(np.zeros(bar.n_elements, dtype=np.int32), np.tile([1.0, 0.0, 0.0], (bar.n_elements, 1)))
    p0 = _positions(bar)
    for step in range(300):
        bar.set_actuation([min(1.0, step / 100)])
        scene.step()
    p1 = _positions(bar)
    seg = np.linalg.norm(p1[pairs[:, 0]] - p1[pairs[:, 1]], axis=1)
    print(f"segment / rest: min={(seg / rest).min():.4f} median={np.median(seg / rest):.4f}; bar length ratio={_extent(p1, 0) / _extent(p0, 0):.3f}; constraint error={scene.vbd_solver.constraint_error():.2e}", flush=True)
    assert np.isfinite(p1).all()
    assert (seg >= 0.9 * rest - 5e-4).all()
    assert np.median(seg / rest) < 0.93
    assert _extent(p1, 0) / _extent(p0, 0) < 0.95


@pytest.mark.required
def test_angle_constraint_caps_the_bend_of_the_actuated_bar(show_viewer):
    """Two muscle groups (top and bottom halves) bend the bar; an angle constraint between the end quarter and the
    middle quarter of its top edge bounded to 10 degrees must hold the measured angle near 10 where the free bar
    bends past 25."""
    angles = {}
    for constrained in (False, True):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=5e-3, substeps=40, gravity=(0.0, 0.0, 0.0)),
            vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1.0, constraint_tol=1e-4),
            show_viewer=show_viewer,
        )
        bar = scene.add_entity(material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, n_groups=2, gain=0.6), morph=gs.morphs.Box(**BAR))
        pos0 = tensor_to_array(bar.init_positions)
        top = np.where((pos0[:, 1] > 0.019) & (pos0[:, 2] > 0.019))[0]
        chain = top[np.argsort(pos0[top, 0])]
        a, b, c, d = chain[-1], chain[3 * len(chain) // 4], chain[len(chain) // 2 + 1], chain[len(chain) // 2 - 1]
        if constrained:
            bar.add_angle_constraints([[a, b, c, d]], 0.0, 10.0)
        scene.build()
        cen = pos0[bar.elems].mean(axis=1)
        bar.set_muscle((cen[:, 2] > 0).astype(np.int32), np.tile([1.0, 0.0, 0.0], (bar.n_elements, 1)))
        for step in range(300):
            bar.set_actuation([min(1.0, step / 100), 0.0])
            scene.step()
        p = _positions(bar)
        u, v = p[a] - p[b], p[c] - p[d]
        angles[constrained] = float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        print(f"constrained={constrained}: angle={angles[constrained]:.1f} deg" + (f", cosine violation={scene.vbd_solver.angle_constraint_error():.2e}" if constrained else ""), flush=True)
        assert np.isfinite(p).all()
    assert angles[False] > 25.0
    assert angles[True] < 12.0


def test_get_state_does_not_retain_states_without_gradients(show_viewer):
    """An RL environment calls get_state every control step for hours; without requires_grad nothing may accumulate."""
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1e-3, substeps=2), vbd_options=gs.options.VBDOptions(n_iterations=2), show_viewer=show_viewer)
    bar = scene.add_entity(material=gs.materials.VBD.Base(E=1e5, nu=0.3), morph=gs.morphs.Box(size=(0.1, 0.05, 0.05), pos=(0.0, 0.0, 0.1), nobisect=False, maxvolume=1e-4))
    scene.build()
    for _ in range(5):
        scene.step()
        bar.get_state()
    assert len(bar._queried_states.states) == 0


def test_multiplier_decay_runs_every_substep(show_viewer):
    """The decay is the dual damping of the two-sweep solve; it runs once per substep, 20 times per step here."""
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=5e-3, substeps=20, gravity=(0.0, 0.0, 0.0)), vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1.0), show_viewer=show_viewer)
    bar = scene.add_entity(material=gs.materials.VBD.Base(E=1e5, nu=0.3), morph=gs.morphs.Box(size=(0.2, 0.05, 0.05), pos=(0.0, 0.0, 0.5), nobisect=False, maxvolume=5e-5))
    p = tensor_to_array(bar.init_positions)
    bar.add_distance_constraints(np.array([[int(np.argmin(p[:, 0])), int(np.argmax(p[:, 0]))]]))
    scene.build()
    solver = scene.vbd_solver
    calls = []
    kernel = solver._kernel_warm_start
    solver._kernel_warm_start = lambda: (calls.append(1), kernel())
    for _ in range(3):
        scene.step()
    assert len(calls) == 60


def test_rayleigh_damping_refuses_a_poisson_ratio_below_one_eighth(show_viewer):
    """The rest Hessian is positive semidefinite only for lam' >= mu / 3; below that damping would inject energy."""
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1e-3, substeps=2), vbd_options=gs.options.VBDOptions(n_iterations=2, damping=0.005), show_viewer=show_viewer)
    scene.add_entity(material=gs.materials.VBD.Base(E=1e5, nu=0.1), morph=gs.morphs.Box(size=(0.1, 0.05, 0.05), pos=(0.0, 0.0, 0.1), nobisect=False, maxvolume=1e-4))
    with pytest.raises(gs.GenesisException):
        scene.build()


def test_the_replay_buffer_reproduces_the_forward_states(show_viewer):
    """The solver-level adjoint walks the sweeps backwards and recovers each linearisation point by subtracting the
    update that sweep applied. That is only sound if the recorded updates reproduce the forward exactly."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=3e-3, substeps=1, gravity=(0.0, 0.0, -9.81), requires_grad=True),
        vbd_options=gs.options.VBDOptions(n_iterations=4, grad_converge=False, contact_stiffness=2e3),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(material=gs.materials.VBD.Base(E=2e4, nu=0.3), morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.048), nobisect=False, maxvolume=3e-4))
    scene.build()
    solver = scene.vbd_solver
    solver._kernel_predict(0)
    predicted = solver.verts.pos.to_numpy()[1].copy()
    solver._kernel_sweeps(0)
    for sweep in reversed(range(4)):
        solver._kernel_undo_sweep(0, sweep)
    np.testing.assert_allclose(solver.verts.pos.to_numpy()[1], predicted, atol=1e-14)
    assert np.abs(solver.sweep_dx.to_numpy()).max() > 1e-6, "the buffer must hold real updates, not zeros"
