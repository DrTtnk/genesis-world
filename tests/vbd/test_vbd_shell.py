"""VBD shell elements: a triangle membrane and a bending stencil, added alongside the tetrahedral path.

Test 1 (parity) is the gate: the engine's per-vertex force, read through the existing `_kernel_residual_vector`
test hook (`r = -force` at the current iterate), must equal the gradient of the shell math spike's own energies,
imported directly rather than re-derived. Everything after it builds on that gate.
"""

import importlib.util
import sys
from pathlib import Path

import json

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

SPIKE_PATH = Path(__file__).resolve().parents[2].parent / "snakeSimWithAstra" / "spikes" / "verify_avbd_shell_math.py"


def _load_spike():
    spec = importlib.util.spec_from_file_location("verify_avbd_shell_math", SPIKE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = _load_spike()


def _residual_force(solver, f=0):
    """`-force` of `_func_vertex_system` at every vertex, the existing test hook: with `vel = 0`, `gravity = 0`
    and `verts[f + 1] == verts[f]` (a fresh `_kernel_predict`), the inertia term is exactly zero, so this reads
    the elastic force alone."""
    out = np.zeros((solver._B, solver.n_vertices, 3), dtype=gs.np_float)
    solver._kernel_residual_vector(f, out)
    return -out


def _shell_scene(material, verts, faces, dt=1e-3, n_iterations=10, substeps=1, gravity=(0.0, 0.0, 0.0)):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=gravity),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1e3),
        show_viewer=False,
    )
    sheet = scene.add_entity(material=material, morph=gs.morphs.TriMesh(verts=verts, faces=faces))
    scene.build()
    return scene, sheet


def _set_static_state(solver, positions):
    """Put `positions` at both frame 0 and frame 1 with zero velocity, so the elastic force alone is read at
    frame 0 (see `_residual_force`)."""
    pos = np.broadcast_to(positions.astype(gs.np_float), (solver._B, *positions.shape)).copy()
    vel = np.zeros_like(pos)
    solver._kernel_set_state(0, pos, vel)
    solver._kernel_predict(0)


def _triangle_area(pos, tris):
    p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=-1)


# ---------------------------------------------------------------------------------------------------------
# 1. Kernel parity against the spike.
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.required
def test_membrane_force_matches_the_spike(show_viewer):
    """One triangle, random rest and deformed positions: the engine's force must equal the gradient of the
    spike's own `membrane_energy`."""
    tolerance = 1e-6 if gs.np_float == np.float64 else 2e-5
    mu, lam = 3.0e4, 9.0e4
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    material = gs.materials.VBD.Shell(E=E, nu=nu, thickness=1e-3, bending_stiffness=0.0)

    rng = np.random.default_rng(20260914)
    max_err = 0.0
    for _ in range(8):
        rest = rng.normal(size=(3, 3))
        positions = rest + 0.3 * rng.normal(size=(3, 3))
        scene, sheet = _shell_scene(material, rest, np.array([[0, 1, 2]]))
        assert abs(sheet.material.mu - mu) < 1e-6 * mu
        assert abs(sheet.material.lam - lam) < 1e-6 * lam

        _set_static_state(scene.vbd_solver, positions)
        engine_force = _residual_force(scene.vbd_solver)[0]

        x = torch.tensor(positions, dtype=torch.float64, device="cpu", requires_grad=True)
        (grad,) = torch.autograd.grad(spike.membrane_energy(x, torch.tensor(rest, dtype=torch.float64, device="cpu"), mu, lam), x)
        expected_force = -grad.cpu().numpy()

        err = np.abs(engine_force - expected_force).max()
        max_err = max(max_err, err)
        assert err < tolerance * max(1.0, np.abs(expected_force).max())
    print(f"membrane force parity: max abs error {max_err:.3e} N")


@pytest.mark.required
def test_bending_force_matches_the_spike(show_viewer):
    """A folded two-triangle stencil, pure rotation about the shared edge so the membrane contributes exactly
    zero: the engine's force must equal the gradient of the spike's own `bending_energy`."""
    stiffness = 7.0
    # A soft membrane on purpose. The folded stencil is a rigid rotation, so the membrane force is zero in
    # exact arithmetic, but at E = 1e5 the float32 quantization of the rotated positions leaves a spurious
    # membrane force of the order of mu * 1e-7 * area, which is larger than the bending force under test.
    material = gs.materials.VBD.Shell(E=1.0, nu=0.3, thickness=1e-3, bending_stiffness=stiffness)
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0], [0.5, -0.8, 0.0]])
    faces = np.array([[0, 1, 2], [1, 0, 3]])

    tolerance = 1e-6 if gs.np_float == np.float64 else 2e-5
    max_err = 0.0
    for angle in (0.05, 0.2, 0.4, 0.8, 1.2):
        folded = rest.copy()
        folded[3] = [0.5, -0.8 * np.cos(angle), -0.8 * np.sin(angle)]
        scene, sheet = _shell_scene(material, rest, faces)
        assert sheet.n_stencils == 1

        _set_static_state(scene.vbd_solver, folded)
        engine_force = _residual_force(scene.vbd_solver)[0]

        x = torch.tensor(folded, dtype=torch.float64, device="cpu", requires_grad=True)
        (grad,) = torch.autograd.grad(spike.bending_energy(x, torch.tensor(rest, dtype=torch.float64, device="cpu"), stiffness), x)
        expected_force = -grad.cpu().numpy()

        err = np.abs(engine_force - expected_force).max()
        max_err = max(max_err, err)
        assert err < tolerance * max(1.0, np.abs(expected_force).max())
    print(f"bending force parity: max abs error {max_err:.3e} N")


# ---------------------------------------------------------------------------------------------------------
# 2. A flat sheet at rest does not move.
# ---------------------------------------------------------------------------------------------------------


def _grid_sheet(nx, ny, spacing=0.05):
    xs, ys = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing, indexing="ij")
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)], axis=-1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            tris.append([a, b, c])
            tris.append([a, c, d])
    return verts, np.array(tris, dtype=np.int64), idx


@pytest.mark.required
def test_flat_sheet_at_rest_does_not_move(show_viewer):
    verts, tris, _ = _grid_sheet(5, 5)
    material = gs.materials.VBD.Shell(E=1e5, nu=0.3, thickness=1e-3, bending_stiffness=1e-3)
    scene, sheet = _shell_scene(material, verts, tris, substeps=5, n_iterations=8)
    pos0 = tensor_to_array(sheet.get_positions())[0]

    for _ in range(10):
        scene.step()

    pos1 = tensor_to_array(sheet.get_positions())[0]
    drift = np.abs(pos1 - pos0).max()
    assert drift < 1e-6
    print(f"flat sheet at rest: max drift {drift:.3e} m")


# ---------------------------------------------------------------------------------------------------------
# 3. A stretched sheet, released, returns to rest.
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.required
def test_stretched_sheet_returns_to_rest_when_released(show_viewer):
    """The prescribed boundary is walked out to a stretch and back to rest, a few sweeps per step, so the
    interior tracks the elastic equilibrium of its boundary at every step (a quasi-static load/release cycle).
    No damping model is implemented for shells (see the material docstring), so an instantaneously unpinned,
    gravity-free sheet has no way to shed the released elastic energy in finite time; that is a property of
    an undamped conservative system, not a gap in the shell elements, so 'released' is tested this way."""
    nx, ny = 5, 5
    verts, tris, idx = _grid_sheet(nx, ny)
    material = gs.materials.VBD.Shell(E=2e5, nu=0.3, thickness=1e-3, bending_stiffness=1e-3)
    scene, sheet = _shell_scene(material, verts, tris, substeps=4, n_iterations=15)
    rest = tensor_to_array(sheet.get_positions())[0].copy()

    left, right = idx[0, :], idx[-1, :]
    pinned = np.zeros(sheet.n_vertices, dtype=bool)
    pinned[left] = True
    pinned[right] = True
    sheet.set_pinned(pinned)

    stretch = 1.6
    stretched_target = rest.copy()
    stretched_target[right, 0] = rest[left, 0].mean() + stretch * (rest[right, 0] - rest[left, 0].mean())

    n_load, n_unload = 30, 60
    for i in range(n_load):
        alpha = (i + 1) / n_load
        sheet.set_pin_targets(rest * (1.0 - alpha) + stretched_target * alpha)
        scene.step()

    loaded = tensor_to_array(sheet.get_positions())[0]
    rest_extent = rest[:, 0].max() - rest[:, 0].min()
    loaded_extent = loaded[:, 0].max() - loaded[:, 0].min()
    assert loaded_extent > 1.3 * rest_extent  # it really is stretched

    for i in range(n_unload):
        alpha = 1.0 - (i + 1) / n_unload
        sheet.set_pin_targets(rest * (1.0 - alpha) + stretched_target * alpha)
        scene.step()

    released = tensor_to_array(sheet.get_positions())[0]
    err = np.abs(released - rest).max()
    print(
        f"stretched sheet released: extent {rest_extent:.4f} -> {loaded_extent:.4f} -> back; "
        f"max position error vs rest {err:.5f} m (rest spacing 0.05 m)"
    )
    assert err < 0.005


# ---------------------------------------------------------------------------------------------------------
# 4. Bending relaxes toward the rest curvature; in-plane scaling raises no bending force.
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.required
def test_folded_stencil_relaxes_toward_its_rest_curvature(show_viewer):
    rest_angle = 0.6
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0], [0.5, -0.8, 0.0]])
    rest[3] = [0.5, -0.8 * np.cos(rest_angle), -0.8 * np.sin(rest_angle)]
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    material = gs.materials.VBD.Shell(E=1.0, nu=0.3, thickness=1e-3, bending_stiffness=50.0)
    scene, sheet = _shell_scene(material, rest, faces, substeps=20, n_iterations=10)

    start_angle = 1.4  # further folded than rest
    folded = rest.copy()
    folded[3] = [0.5, -0.8 * np.cos(start_angle), -0.8 * np.sin(start_angle)]
    scene.vbd_solver._kernel_set_state(0, folded[None].astype(gs.np_float), np.zeros((1, 4, 3), dtype=gs.np_float))

    def dihedral(pos):
        n0 = np.cross(pos[1] - pos[0], pos[2] - pos[0])
        n1 = np.cross(pos[1] - pos[0], pos[3] - pos[0])
        cos = (n0 @ n1) / (np.linalg.norm(n0) * np.linalg.norm(n1))
        return np.arccos(np.clip(cos, -1.0, 1.0))

    start_dihedral = dihedral(folded)
    rest_dihedral = dihedral(rest)
    for _ in range(120):
        scene.step()
    end_pos = tensor_to_array(sheet.get_positions())[0]
    end_dihedral = dihedral(end_pos)

    print(f"folded stencil: dihedral {start_dihedral:.4f} -> {end_dihedral:.4f} rad (rest {rest_dihedral:.4f})")
    assert abs(end_dihedral - rest_dihedral) < 0.3 * abs(start_dihedral - rest_dihedral)


@pytest.mark.required
def test_in_plane_scaling_raises_no_bending_force(show_viewer):
    """The bending block is blind to any affine deformation of the flattened stencil, so a pure in-plane
    scaling of a flat rest must leave the bending contribution at zero: the total residual force must match
    the membrane-alone force, computed from the already-verified analytic formula."""
    mu, lam = 3.0e4, 9.0e4
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0], [0.5, -0.8, 0.0]])
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    material = gs.materials.VBD.Shell(E=E, nu=nu, thickness=1e-3, bending_stiffness=50.0)
    scene, sheet = _shell_scene(material, rest, faces)

    scale = 2.0
    scaled = rest * np.array([scale, scale, 1.0])
    _set_static_state(scene.vbd_solver, scaled)
    engine_force = _residual_force(scene.vbd_solver)[0]

    # Membrane-only expectation, from the spike's own energy on each triangle separately (no bending term).
    x = torch.tensor(scaled, dtype=torch.float64, device="cpu", requires_grad=True)
    e_membrane = spike.membrane_energy(x[[0, 1, 2]], torch.tensor(rest[[0, 1, 2]], dtype=torch.float64, device="cpu"), mu, lam) + spike.membrane_energy(
        x[[1, 0, 3]], torch.tensor(rest[[1, 0, 3]], dtype=torch.float64, device="cpu"), mu, lam
    )
    (grad,) = torch.autograd.grad(e_membrane, x)
    membrane_only_force = -grad.cpu().numpy()

    err = np.abs(engine_force - membrane_only_force).max()
    print(f"in-plane scaling: max deviation from membrane-only force {err:.3e} N (bending must add nothing)")
    assert err < 1e-6 * max(1.0, np.abs(membrane_only_force).max())


# ---------------------------------------------------------------------------------------------------------
# 5. A corrugated tube, opened from inside, unfolds rather than stretches.
# ---------------------------------------------------------------------------------------------------------


def _corrugated_tube(n_ring=16, n_axial=9, r0=0.02, amplitude=0.008, n_pleats=6, length=0.3):
    theta = np.linspace(0.0, 2 * np.pi, n_ring, endpoint=False)
    z = np.linspace(-length / 2.0, length / 2.0, n_axial)
    r = r0 + amplitude * np.cos(n_pleats * theta)
    verts = np.zeros((n_axial, n_ring, 3))
    for k, zk in enumerate(z):
        verts[k, :, 0] = r * np.cos(theta)
        verts[k, :, 1] = r * np.sin(theta)
        verts[k, :, 2] = zk
    verts = verts.reshape(-1, 3)
    idx = np.arange(n_axial * n_ring).reshape(n_axial, n_ring)
    tris = []
    for k in range(n_axial - 1):
        for i in range(n_ring):
            a, b = idx[k, i], idx[k, (i + 1) % n_ring]
            c, d = idx[k + 1, i], idx[k + 1, (i + 1) % n_ring]
            tris.append([a, b, d])
            tris.append([a, d, c])
    return verts, np.array(tris, dtype=np.int64)


@pytest.mark.required
def test_corrugated_tube_unfolds_rather_than_stretches(show_viewer):
    verts, tris = _corrugated_tube()
    material = gs.materials.VBD.Shell(E=2e4, nu=0.3, thickness=2e-4, bending_stiffness=2e-6)
    scene, tube = _shell_scene(
        material, verts, tris, dt=2e-3, substeps=8, n_iterations=8, gravity=(0.0, 0.0, 0.0)
    )
    solver = scene.vbd_solver

    rest = tensor_to_array(tube.get_positions())[0].copy()
    rest_radius = np.linalg.norm(rest[:, :2], axis=-1)
    lumen_radius_0 = rest_radius.min()
    area_0 = _triangle_area(rest, tube.tris).sum()

    bolus_radius_final = 0.024  # beyond the pleat crest (r0 + amplitude = 0.028) but inside it comfortably
    n_ramp_steps = 200
    for i in range(n_ramp_steps):
        r_bolus = lumen_radius_0 * 0.3 + (bolus_radius_final - lumen_radius_0 * 0.3) * (i + 1) / n_ramp_steps
        solver.set_bolus(
            center=np.zeros((1, 3)), radius=np.array([r_bolus]), vel=np.zeros((1, 3)), friction=0.0,
            axis=(0.0, 0.0, 1.0), half_length=0.2,
        )
        scene.step()
    for _ in range(200):  # let the pleats settle once the bolus stops growing
        scene.step()

    final = tensor_to_array(tube.get_positions())[0]
    final_radius = np.linalg.norm(final[:, :2], axis=-1)
    lumen_radius_1 = final_radius.min()
    area_1 = _triangle_area(final, tube.tris).sum()

    radius_growth = (lumen_radius_1 - lumen_radius_0) / lumen_radius_0
    area_growth = (area_1 - area_0) / area_0
    print(
        f"corrugated tube: lumen radius {lumen_radius_0:.5f} -> {lumen_radius_1:.5f} m "
        f"({radius_growth * 100:.1f}% growth); total area {area_0:.6f} -> {area_1:.6f} m^2 "
        f"({area_growth * 100:.1f}% growth)"
    )
    budget = json.loads((Path(__file__).parent / "manifests" / "shell.json").read_text())["corrugated_tube"]
    assert radius_growth > budget["radius_growth"]["minimum"]
    assert area_growth < budget["area_growth"]["maximum"]


def test_bending_force_stays_accurate_far_from_the_origin(show_viewer):
    """The gut's coordinates reach 2.5 m while its pleats are 8 mm across. This asks whether the bending force
    at 2.5 m still matches the float64 reference to the same accuracy as at the origin, which is a statement
    about absolute accuracy rather than about translation invariance: the two sums that form the residual
    shift together, so invariance holds either way."""
    stiffness = 7.0
    material = gs.materials.VBD.Shell(E=1.0, nu=0.3, thickness=1e-3, bending_stiffness=stiffness)
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0], [0.5, -0.8, 0.0]])
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    folded = rest.copy()
    folded[3] = [0.5, -0.8 * np.cos(0.4), -0.8 * np.sin(0.4)]

    errors = {}
    for offset in (0.0, 2.5):
        shift = np.array([offset, 0.0, 0.0])
        scene, sheet = _shell_scene(material, rest + shift, faces)
        _set_static_state(scene.vbd_solver, folded + shift)
        engine_force = _residual_force(scene.vbd_solver)[0]
        x = torch.tensor(folded + shift, dtype=torch.float64, device="cpu", requires_grad=True)
        (grad,) = torch.autograd.grad(
            spike.bending_energy(x, torch.tensor(rest + shift, dtype=torch.float64, device="cpu"), stiffness), x
        )
        errors[offset] = float(np.abs(engine_force + grad.cpu().numpy()).max())
    print(f"bending parity error at the origin {errors[0.0]:.3e} N, at 2.5 m {errors[2.5]:.3e} N")
    assert errors[2.5] < max(10.0 * errors[0.0], 1e-6)
