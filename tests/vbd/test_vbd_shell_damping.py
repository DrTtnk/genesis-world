"""Rayleigh damping on a shell: it removes energy from deformation and takes none from rigid motion.

The digestive wall rings after every contact without damping, and a ringing wall is not a wall a bolus can be
pushed through. The tet path already damps with C = k_d K0, K0 the exact energy Hessian at rest, chosen because
it is constant, positive semidefinite and zero on rigid motions. The membrane's own rest Hessian is derived and
checked against autograd, block by block, in `spikes/verify_avbd_shell_math.py` (`membrane_rest_block`) in the
snakeSimWithAstra repository; `_func_rest_block2` is its transcription.

Bending is deliberately not damped. Its quadratic model measures |K x - K x_rest|, which is not invariant under
a rotation of a curved rest shape, so its Hessian does not annihilate a rotation and would brake a coiling gut.
The same spike asserts that.

The rigid-motion tests below read the damping force directly instead of watching a trajectory: at rest, with
zero gravity, `_kernel_predict` puts `verts[1] = x + h v` and the inertia and elastic terms are both exactly
zero, so the residual is the damping force alone and the property is read rather than inferred.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array


DAMPING = 0.02


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


def _scene(verts, faces, damping, dt=2e-3, n_iterations=10, gravity=(0.0, 0.0, 0.0)):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=gravity),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1e3, damping=damping),
        show_viewer=False,
    )
    material = gs.materials.VBD.Shell(E=1e5, nu=0.3, thickness=1e-3, bending_stiffness=1e-5)
    sheet = scene.add_entity(material=material, morph=gs.morphs.TriMesh(verts=verts, faces=faces))
    scene.build()
    return scene, sheet


def _damping_force(scene, positions, velocities):
    """Residual at the predicted configuration: with gravity zero, the rest configuration and a fresh predict,
    the inertia and elastic terms vanish and this is the damping force alone."""
    solver = scene.vbd_solver
    pos = np.broadcast_to(positions.astype(gs.np_float), (solver._B, *positions.shape)).copy()
    vel = np.broadcast_to(velocities.astype(gs.np_float), (solver._B, *velocities.shape)).copy()
    solver._kernel_set_state(0, pos, vel)
    solver._kernel_predict(0)
    out = np.zeros((solver._B, solver.n_vertices, 3), dtype=gs.np_float)
    solver._kernel_residual_vector(0, out)
    return -out[0]


@pytest.mark.required
def test_a_shell_with_damping_builds_at_all(show_viewer):
    """Until this change the solver refused a shell whenever damping was on."""
    verts, tris, _ = _grid_sheet(4, 4)
    scene, sheet = _scene(verts, tris, damping=DAMPING)
    scene.step()
    assert np.isfinite(tensor_to_array(sheet.get_positions())).all()


@pytest.mark.required
def test_damping_takes_no_force_from_a_translating_shell(show_viewer):
    verts, tris, _ = _grid_sheet(5, 5)
    scene, _ = _scene(verts, tris, damping=DAMPING)
    velocity = np.broadcast_to(np.array([0.7, -0.4, 0.9]), verts.shape).copy()
    force = _damping_force(scene, verts, velocity)
    stretching = verts * 0.9  # a uniform in-plane contraction, the same size of motion but not rigid
    reference = _damping_force(scene, verts, (stretching - verts) / 2e-3)
    assert np.abs(force).max() < 1e-4 * np.abs(reference).max()


@pytest.mark.required
def test_damping_takes_almost_no_force_from_a_rotating_shell(show_viewer):
    """A rigid rotation is annihilated exactly only in the infinitesimal limit; over one step of 2 ms the
    residual is the second-order part, which must stay far below a deforming motion of the same speed."""
    verts, tris, _ = _grid_sheet(5, 5)
    scene, _ = _scene(verts, tris, damping=DAMPING)
    omega = np.array([0.3, -0.5, 0.8])
    centre = verts.mean(axis=0)
    velocity = np.cross(omega, verts - centre)
    force = _damping_force(scene, verts, velocity)
    speed = np.linalg.norm(velocity, axis=1).max()
    stretching = np.zeros_like(verts)
    stretching[:, 0] = speed * (verts[:, 0] - centre[0]) / np.abs(verts[:, 0] - centre[0]).max()
    reference = _damping_force(scene, verts, stretching)
    assert np.abs(force).max() < 1e-2 * np.abs(reference).max()


@pytest.mark.required
def test_damping_removes_energy_from_a_ringing_sheet(show_viewer):
    """A sheet held at one edge and released from a stretched state rings. With damping the ringing must decay,
    and the undamped run is the control that says the decay is the damping and not the solver."""
    verts, tris, idx = _grid_sheet(6, 6)
    held = np.zeros(len(verts), dtype=bool)
    held[idx[0, :]] = True
    stretched = verts.copy()
    stretched[:, 0] *= 1.25

    energies = {}
    for damping in (0.0, DAMPING):
        scene, sheet = _scene(verts, tris, damping=damping, n_iterations=12)
        sheet.set_pinned(held)
        solver = scene.vbd_solver
        pos = np.broadcast_to(stretched.astype(gs.np_float), (solver._B, *verts.shape)).copy()
        solver._kernel_set_state(0, pos, np.zeros_like(pos))
        tail = []
        for step in range(300):
            scene.step()
            if step >= 200:
                velocity = tensor_to_array(sheet.get_state().vel)[0]
                tail.append(float((velocity**2).sum()))
        energies[damping] = max(tail)

    print(f"peak late kinetic energy: undamped {energies[0.0]:.4e}, damped {energies[DAMPING]:.4e}")
    assert energies[DAMPING] < 0.2 * energies[0.0]
