"""A5 tissue material fixtures: uniaxial extension/compression, bending, volumetric response,
rest-shape preservation without an initialization kick, and hollow-lumen/mass survival under
refinement. Every comparison in this file is against a reference derived on paper from the
constitutive law or from classic beam theory (`reference_tissue.py`), never against the solver's
own kernels. Budgets live in `manifests/tissue_material.json`, frozen from measured floors after
the assertions below were written; see that file's `description` for the discipline.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array
from tests.vbd.reference_tissue import (
    annulus_volume,
    clamped_guided_beam_shape,
    lame_parameters,
    uniaxial_lateral_stretch,
)

MANIFEST = json.loads((Path(__file__).parent / "manifests" / "tissue_material.json").read_text())


def _positions(entity):
    return tensor_to_array(entity.get_positions())[0]


def _tet_volumes(pos, elems):
    p = pos[elems]
    return np.linalg.det(np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)) / 6.0


@pytest.mark.required
def test_lateral_stretch_closed_form_is_a_root_of_the_kernel_stress():
    """Symbolic proof (sympy, random material/stretch triples) that `uniaxial_lateral_stretch`
    zeros the transverse component of the kernel's own P = mu F + lam' (J - alpha) cof(F) for
    F = diag(lx, ly, ly). This is the derivation `reference_tissue.py` documents; it must hold
    before the closed form is trusted as an independent reference."""
    mu_s, lamp_s, lx_s, ly_s = sp.symbols("mu lamp lx ly", positive=True)
    alpha_s = 1 + mu_s / lamp_s
    F = sp.diag(lx_s, ly_s, ly_s)
    J = F.det()
    cof = J * F.inv().T
    P = mu_s * F + lamp_s * (J - alpha_s) * cof
    Pyy = sp.expand(P[1, 1])

    rng = np.random.default_rng(0)
    worst = 0.0
    checked = 0
    while checked < 200:
        mu_v = float(rng.uniform(1e3, 1e6))
        lamp_v = float(rng.uniform(1e3, 1e6))
        lx_v = float(rng.uniform(0.3, 3.0))
        try:
            ly_v = uniaxial_lateral_stretch(lx_v, mu_v, lamp_v)
        except ValueError:
            continue  # this (mu, lam', lx) triple has no traction-free lateral root; skip it
        residual = float(Pyy.subs({mu_s: mu_v, lamp_s: lamp_v, lx_s: lx_v, ly_s: ly_v}))
        worst = max(worst, abs(residual) / mu_v)  # mu is the natural stress scale here
        checked += 1
    assert worst < 1e-8


def _clamped_guided_beam_symbolic_check():
    """Integrate E I y'''' = 0 symbolically for a beam clamped at x=0 and guided (translated,
    not rotated) at x=L, and confirm the cubic Hermite shape `reference_tissue.py` uses."""
    L, delta, x = sp.symbols("L delta x", positive=True)
    C1, C2, C3, C4 = sp.symbols("C1 C2 C3 C4")
    y_expr = C1 * x**3 + C2 * x**2 + C3 * x + C4
    # y(0) = 0, y'(0) = 0 (clamped root); y(L) = delta, y'(L) = 0 (guided tip, no rotation)
    sol = sp.solve(
        [y_expr.subs(x, 0), y_expr.diff(x).subs(x, 0), y_expr.subs(x, L) - delta, y_expr.diff(x).subs(x, L)],
        [C1, C2, C3, C4],
    )
    shape = sp.simplify(y_expr.subs(sol))
    assert sp.simplify(shape - delta * x**2 * (3 * L - 2 * x) / L**3) == 0


@pytest.mark.required
def test_clamped_guided_beam_formula_matches_the_symbolic_euler_bernoulli_solution():
    _clamped_guided_beam_symbolic_check()


UNIAXIAL_BAR = dict(size=(0.16, 0.04, 0.04), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=1e-6)
UNIAXIAL_BAND = 0.02  # mid-span half-width kept for measurement, clear of the grips' Saint-Venant zone


def _uniaxial_bar_scene(lx, E=1e5, nu=0.3, show_viewer=False):
    """A stocky bar (length/cross-section = 4): slender enough to have a free mid-span, short
    enough that a 15% axial compression does not buckle it (a longer bar was probed and does)."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=20, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1.0, damping=0.02),
        show_viewer=show_viewer,
    )
    bar = scene.add_entity(material=gs.materials.VBD.Base(E=E, nu=nu), morph=gs.morphs.Box(**UNIAXIAL_BAR))
    scene.build()
    pos0 = tensor_to_array(bar.init_positions)
    half = 0.08
    right = pos0[:, 0] > half - 1e-4
    left = pos0[:, 0] < -half + 1e-4
    pinned = right | left
    bar.set_pinned(pinned)
    target = pos0.copy()
    n_steps = 300
    ramp_steps = 150
    for step in range(n_steps):
        r = min(1.0, step / ramp_steps)
        target[right, 0] = pos0[right, 0] * (1.0 + (lx - 1.0) * r)
        target[left, 0] = pos0[left, 0] * (1.0 + (lx - 1.0) * r)
        bar.set_pin_targets(target)
        scene.step()
    return scene, bar, pos0


def _mid_span_lateral_and_volume(bar, pos0, band=UNIAXIAL_BAND):
    pos1 = _positions(bar)
    mid = np.abs(pos0[:, 0]) < band
    extent_y0 = pos0[mid, 1].max() - pos0[mid, 1].min()
    extent_z0 = pos0[mid, 2].max() - pos0[mid, 2].min()
    extent_y1 = pos1[mid, 1].max() - pos1[mid, 1].min()
    extent_z1 = pos1[mid, 2].max() - pos1[mid, 2].min()
    ly_measured = 0.5 * (extent_y1 / extent_y0 + extent_z1 / extent_z0)

    elems = bar.elems
    centroid_x0 = pos0[elems].mean(axis=1)[:, 0]
    mid_elems = np.abs(centroid_x0) < band
    vol0 = _tet_volumes(pos0, elems)
    vol1 = _tet_volumes(pos1, elems)
    assert (vol1 > 0).all(), "a tet inverted under the imposed axial stretch"
    volume_ratio = vol1[mid_elems].sum() / vol0[mid_elems].sum()
    return ly_measured, volume_ratio, pos1


@pytest.mark.required
def test_uniaxial_extension_matches_the_closed_form_lateral_contraction(show_viewer):
    """A bar clamped at both ends and stretched 15% axially must contract laterally, away from
    the grips, by the traction-free root of the kernel's own stress tensor."""
    lx = 1.15
    E, nu = 1e5, 0.3
    mu, lamp = lame_parameters(E, nu)
    ly_expected = uniaxial_lateral_stretch(lx, mu, lamp)

    _, bar, pos0 = _uniaxial_bar_scene(lx, E=E, nu=nu, show_viewer=show_viewer)
    ly_measured, volume_ratio, pos1 = _mid_span_lateral_and_volume(bar, pos0)
    assert np.isfinite(pos1).all()

    budget = MANIFEST["uniaxial_extension"]
    lat = budget["lateral_stretch"]
    err = abs(ly_measured - ly_expected)
    print(f"extension: ly_measured={ly_measured:.5f} ly_expected={ly_expected:.5f} err={err:.2e}", flush=True)
    assert err <= lat["atol"] + lat["rtol"] * lat["scale"]

    vol = budget["volume_ratio"]
    j_expected = lx * ly_expected**2
    err_vol = abs(volume_ratio - j_expected)
    print(f"extension: volume_ratio={volume_ratio:.5f} expected={j_expected:.5f} err={err_vol:.2e}", flush=True)
    assert err_vol <= vol["atol"] + vol["rtol"] * vol["scale"]


@pytest.mark.required
def test_uniaxial_compression_matches_the_closed_form_lateral_expansion(show_viewer):
    """The same bar compressed 15% axially must bulge laterally by the same closed form, on the
    other branch (lx < 1)."""
    lx = 0.85
    E, nu = 1e5, 0.3
    mu, lamp = lame_parameters(E, nu)
    ly_expected = uniaxial_lateral_stretch(lx, mu, lamp)

    _, bar, pos0 = _uniaxial_bar_scene(lx, E=E, nu=nu, show_viewer=show_viewer)
    ly_measured, volume_ratio, pos1 = _mid_span_lateral_and_volume(bar, pos0)
    assert np.isfinite(pos1).all()

    budget = MANIFEST["uniaxial_compression"]
    lat = budget["lateral_stretch"]
    err = abs(ly_measured - ly_expected)
    print(f"compression: ly_measured={ly_measured:.5f} ly_expected={ly_expected:.5f} err={err:.2e}", flush=True)
    assert err <= lat["atol"] + lat["rtol"] * lat["scale"]

    vol = budget["volume_ratio"]
    j_expected = lx * ly_expected**2
    err_vol = abs(volume_ratio - j_expected)
    print(f"compression: volume_ratio={volume_ratio:.5f} expected={j_expected:.5f} err={err_vol:.2e}", flush=True)
    assert err_vol <= vol["atol"] + vol["rtol"] * vol["scale"]


def _grip_axial_reaction_force(scene, bar, pos0, right_mask):
    """The force the tissue exerts on its held right-hand grip, from the solver's own residual
    hook: `r_i = -force_i` of `_func_vertex_system` (`vbd_solver.py::_kernel_residual_vector`,
    labelled a test hook in its docstring) is the net internal force at vertex i, evaluated with
    every vertex fixed at the current iterate. A pinned vertex is never solved, so at a settled
    equilibrium this is exactly the force its grip must supply to hold it there: the reaction.
    Summed over one grip's vertices, its x-component is the axial force the tissue pulls on that
    grip, comparable to a load cell. No engine code was added; this reads an existing hook."""
    solver = scene.vbd_solver
    f = solver._sim.cur_substep_local - 1 if solver._sim.cur_substep_local > 0 else solver._sim.substeps_local - 1
    out = np.zeros((solver._B, solver.n_vertices, 3), dtype=np.float64)
    solver._kernel_residual_vector(f, out)
    reaction = out[0]
    global_idx = np.flatnonzero(right_mask) + bar.v_start
    return reaction[global_idx]


@pytest.mark.required
def test_uniaxial_extension_axial_force_pins_the_absolute_stiffness_scale(show_viewer, precision):
    """The lateral-stretch fixtures above are displacement-controlled and their closed form is a
    function of mu/lam' (Poisson's ratio) alone: multiplying every element's E by a constant
    leaves `uniaxial_lateral_stretch` and the settled shape unchanged, so a bug that scaled all
    tissue stiffness by a constant factor would pass them, the bending fixture (shape-only by
    construction) and the dilation fixture (geometrically trivial) - all eight fixtures already
    in this file. This fixture closes that gap with a force measurement: the axial reaction on
    the held right-hand grip, at two Young's moduli a decade apart, must (1) match the closed-form
    traction P_xx = mu lx + lam' (J - alpha) ly^2, integrated over the rest cross-section, at each
    E individually, and (2) scale between the two E by exactly the modulus ratio - the ratio check
    is what a uniform stiffness-scale bug cannot pass even if every individual fixture above does.
    """
    lx = 1.15
    nu = 0.3
    width = height_dim = 0.04
    area_rest = width * height_dim
    E_values = (1e5, 1e6)

    forces = {}
    for E in E_values:
        scene, bar, pos0 = _uniaxial_bar_scene(lx, E=E, nu=nu, show_viewer=show_viewer)
        right = pos0[:, 0] > 0.08 - 1e-4
        reaction = _grip_axial_reaction_force(scene, bar, pos0, right)
        Fx = float(reaction[:, 0].sum())
        Fy, Fz = float(reaction[:, 1].sum()), float(reaction[:, 2].sum())
        assert abs(Fy) < 0.1 * abs(Fx) and abs(Fz) < 0.1 * abs(Fx), (
            f"grip net lateral force is not small next to the axial one: Fx={Fx} Fy={Fy} Fz={Fz}"
        )

        mu, lamp = lame_parameters(E, nu)
        ly = uniaxial_lateral_stretch(lx, mu, lamp)
        alpha = 1.0 + mu / lamp
        J = lx * ly**2
        Pxx = mu * lx + lamp * (J - alpha) * ly**2
        expected = Pxx * area_rest
        forces[E] = (Fx, expected)
        print(f"E={E:.1e}: Fx_grip={Fx:.6f} N expected={expected:.6f} N", flush=True)

    budget = MANIFEST["axial_force"]
    for E in E_values:
        Fx, expected = forces[E]
        b = budget["absolute"][str(E)]
        err = abs(Fx - expected) / expected  # relative to the analytic reference, not the measurement
        print(f"E={E:.1e}: relative force error={err:.4e}", flush=True)
        assert err <= b["atol"] + b["rtol"] * b["scale"]

    measured_ratio = forces[E_values[1]][0] / forces[E_values[0]][0]
    expected_ratio = E_values[1] / E_values[0]
    ratio_budget = budget["modulus_ratio"][precision]
    ratio_err = abs(measured_ratio - expected_ratio)
    print(f"modulus ratio: measured={measured_ratio:.7f} expected={expected_ratio:.1f} err={ratio_err:.3e}", flush=True)
    assert ratio_err <= ratio_budget["atol"] + ratio_budget["rtol"] * ratio_budget["scale"]


@pytest.mark.required
def test_bending_shape_matches_the_clamped_guided_beam_reference(show_viewer):
    """Clamp one end of a bar and translate the other end, transversely, by a small imposed
    displacement without imposing a rotation there (both are Dirichlet, so neither reveals
    anything by itself); the free interior must settle onto the cubic Hermite shape a linear
    beam predicts for that boundary condition, regardless of the bar's stiffness.

    A first attempt drove the bar with gravity instead, holding one end fixed and reading the
    settled tip sag against the matching Euler-Bernoulli self-weight formula. That measurement
    came back grossly wrong (tens of centimetres of sag for a beam small linear theory put at
    under a centimetre) and, more tellingly, essentially unchanged across two decades of E - a
    beam's elastic sag must scale with 1/E, so a force-driven, magnitude-based bending fixture
    on this solver is not trustworthy yet. This displacement-controlled, shape-only fixture
    sidesteps the question of what force produced a deflection and asks only whether the
    deformation pattern the solver relaxes onto, between two fully prescribed ends, is the
    right one - and it passes cleanly. See the report for both findings.
    """
    length = 0.16
    half = length / 2
    delta_tip = 0.01 * length  # 1% of length: small enough for the linear beam shape to apply
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=20, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1.0, damping=0.02),
        show_viewer=show_viewer,
    )
    bar = scene.add_entity(
        material=gs.materials.VBD.Base(E=1e5, nu=0.3),
        morph=gs.morphs.Box(size=(length, 0.04, 0.04), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=1e-6),
    )
    scene.build()
    pos0 = tensor_to_array(bar.init_positions)
    root = pos0[:, 0] < -half + 1e-4
    tip = pos0[:, 0] > half - 1e-4
    bar.set_pinned(root | tip)
    target = pos0.copy()
    n_steps, ramp_steps = 300, 150
    for step in range(n_steps):
        r = min(1.0, step / ramp_steps)
        target[tip, 2] = pos0[tip, 2] - delta_tip * r
        bar.set_pin_targets(target)
        scene.step()
    pos1 = _positions(bar)
    assert np.isfinite(pos1).all()

    elems = bar.elems
    vol1 = _tet_volumes(pos1, elems)
    assert (vol1 > 0).all()

    budget = MANIFEST["bending"]["shape_ratio"]
    worst = 0.0
    for t in (0.25, 0.5, 0.75):
        xq = -half + t * length
        band = np.abs(pos0[:, 0] - xq) < 0.01
        assert band.sum() > 5, f"too few vertices near x/L = {t} to measure the shape there"
        measured = float((pos0[band, 2] - pos1[band, 2]).mean()) / delta_tip
        expected = clamped_guided_beam_shape(t)
        err = abs(measured - expected)
        worst = max(worst, err)
        print(f"bending t={t}: measured y/delta={measured:.4f} expected={expected:.4f} err={err:.3e}", flush=True)
    assert worst <= budget["atol"] + budget["rtol"] * budget["scale"]


@pytest.mark.required
@pytest.mark.parametrize("s", [1.12, 0.88])
def test_isotropic_dilation_tracks_the_imposed_volumetric_target(show_viewer, s, precision):
    """A cube whose entire boundary is driven to a uniform isotropic scale s must reach that
    exact volumetric ratio s^3 in its interior too: a locking or discretization defect would
    show up as the free interior nodes failing to track the affine target, most visibly at
    near-incompressible Poisson ratios."""
    E, nu = 1e5, 0.4
    half = 0.03
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=20, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1.0, damping=0.02),
        show_viewer=show_viewer,
    )
    cube = scene.add_entity(
        material=gs.materials.VBD.Base(E=E, nu=nu),
        morph=gs.morphs.Box(size=(2 * half, 2 * half, 2 * half), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=2e-6),
    )
    scene.build()
    pos0 = tensor_to_array(cube.init_positions)
    tol = 1e-4
    on_boundary = (
        (np.abs(pos0[:, 0]) > half - tol) | (np.abs(pos0[:, 1]) > half - tol) | (np.abs(pos0[:, 2]) > half - tol)
    )
    assert on_boundary.sum() < pos0.shape[0], "the mesh has no free interior vertex to test"
    cube.set_pinned(on_boundary)

    n_steps = 300
    ramp_steps = 150
    for step in range(n_steps):
        r = min(1.0, step / ramp_steps)
        scale = 1.0 + (s - 1.0) * r
        target = pos0.copy()
        target[on_boundary] = pos0[on_boundary] * scale
        cube.set_pin_targets(target)
        scene.step()

    pos1 = _positions(cube)
    assert np.isfinite(pos1).all()
    elems = cube.elems
    vol0 = _tet_volumes(pos0, elems)
    vol1 = _tet_volumes(pos1, elems)
    assert (vol1 > 0).all()
    volume_ratio = vol1.sum() / vol0.sum()
    expected_ratio = s**3

    interior_err = float(np.abs(pos1[~on_boundary] - pos0[~on_boundary] * s).max())

    budget = MANIFEST["volume_response"]
    vol_budget = budget["volume_ratio"]
    pos_budget = budget["interior_position"][precision]
    err_vol = abs(volume_ratio - expected_ratio)
    print(
        f"dilation s={s}: volume_ratio={volume_ratio:.5f} expected={expected_ratio:.5f} err={err_vol:.2e}, "
        f"interior position error={interior_err:.2e} m",
        flush=True,
    )
    assert err_vol <= vol_budget["atol"] + vol_budget["rtol"] * vol_budget["scale"]
    assert interior_err <= pos_budget["atol"] + pos_budget["rtol"] * pos_budget["scale"]


def _hollow_tube_mesh(path, r_in, r_out, height, sections):
    trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=height, sections=sections).export(str(path))


@pytest.mark.required
def test_hollow_tube_rest_shape_holds_and_mass_survives_refinement(show_viewer, asset_tmp_path, precision):
    """A tube stand-in for the digestive wall: (1) built at rest it must not move under its own
    tetrahedralization, an initialization kick; (2) every tet Jacobian must be positive; (3) the
    tetrahedralized mass must match the analytic hollow-cylinder mass; (4) a finer surface
    refinement must bring the mass closer to that analytic target and clear the lumen better.

    Refinement here means more polygon `sections` around the tube's circular cross-section, not a
    smaller `maxvolume`: a probe showed that shrinking `maxvolume` alone, with the same coarse
    circle, changes the interior tet count but leaves the total mass and lumen radius identical to
    machine precision, because both are set by the *surface* polygon tetgen is handed, not by how
    finely its interior is subdivided. `maxvolume` here is fixed small enough, at both section
    counts, that it is not the bottleneck."""
    r_in, r_out, height, rho = 0.02, 0.03, 0.06, 1000.0
    expected_volume = annulus_volume(r_in, r_out, height)

    results = {}
    for label, sections in (("coarse", 12), ("fine", 40)):
        path = asset_tmp_path / f"tissue_tube_{label}.obj"
        _hollow_tube_mesh(path, r_in, r_out, height, sections)
        maxvolume = 3e-8
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=5e-3, substeps=10, gravity=(0.0, 0.0, 0.0)),
            vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1.0),
            show_viewer=show_viewer,
        )
        tube = scene.add_entity(
            material=gs.materials.VBD.Base(E=1e5, nu=0.3, rho=rho),
            morph=gs.morphs.Mesh(file=str(path), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=maxvolume),
        )
        scene.build()
        pos0 = _positions(tube)
        elems = tube.elems
        vol_rest = _tet_volumes(pos0, elems)
        assert (vol_rest > 0).all(), f"{label}: inverted tet at rest, before any step"

        radial = np.linalg.norm(pos0[:, :2], axis=1)  # trimesh.annulus is centered on the z axis
        min_radial = float(radial.min())

        if label == "coarse":
            for _ in range(20):
                scene.step()
            pos1 = _positions(tube)
            kick = float(np.abs(pos1 - pos0).max())
            results["kick"] = kick

        mass = float(rho * vol_rest.sum())
        results[label] = dict(mass=mass, min_radial=min_radial)
        print(f"{label}: mass={mass:.6f} kg (analytic {rho * expected_volume:.6f}), min radial={min_radial:.4f} m (r_in={r_in})", flush=True)

    budget = MANIFEST["hollow_tube"]
    kick_budget = budget["rest_kick"][precision]
    assert results["kick"] <= kick_budget["atol"] + kick_budget["rtol"] * kick_budget["scale"]

    analytic_mass = rho * expected_volume
    for label in ("coarse", "fine"):
        mass_budget = budget["mass"][label]
        err = abs(results[label]["mass"] - analytic_mass)
        assert err <= mass_budget["atol"] + mass_budget["rtol"] * mass_budget["scale"]
        lumen_budget = budget["lumen_clearance"][label]
        clearance = r_in - results[label]["min_radial"]
        assert clearance <= lumen_budget["atol"] + lumen_budget["rtol"] * lumen_budget["scale"]
