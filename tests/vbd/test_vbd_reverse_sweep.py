"""The solver-level adjoint: the sweeps of the forward, applied in reverse.

Finite differences of the executed rollout are the truth here. They differentiate exactly the
computation that ran, converged or not, so they judge the reverse sweep at the sweep counts the fast
path really uses. The equation-level adjoint cannot pass this test: it is 38 percent wrong at two
sweeps on a body that does not converge (spikes/executed_adjoint_gap.py in the snakeSim repository).
"""

import numpy as np
import pytest
import torch

import genesis as gs


def _rig(show_viewer, n_iterations, substeps=1):
    """A free elastic block, no floor, no damping, no constraints: only terms proven against autograd."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=substeps, gravity=(0.0, 0.0, -9.81), requires_grad=True),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, grad_converge=False, floor_height=-10.0),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        material=gs.materials.VBD.Base(E=2e4, nu=0.3),
        morph=gs.morphs.Box(size=(0.2, 0.1, 0.1), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=4e-4),
    )
    scene.build()
    return scene, box, scene.vbd_solver


def _state(solver, seed):
    rng = np.random.default_rng(seed)
    n = solver.n_vertices
    x0 = solver.verts.pos.to_numpy()[0, :, 0] + 0.02 * rng.normal(size=(n, 3))  # deformed, so the material is live
    v0 = 0.3 * rng.normal(size=(n, 3))
    return x0, v0, rng.normal(size=(n, 3))


@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_iterations", [1, 2, 4])
def test_the_reverse_sweep_matches_finite_differences_of_the_executed_rollout(show_viewer, n_iterations):
    scene, box, solver = _rig(show_viewer, n_iterations)
    n = solver.n_vertices
    x0, v0, w = _state(solver, 0)

    def forward(x, v):
        solver._kernel_set_state(0, torch.as_tensor(x[None]).contiguous(), torch.as_tensor(v[None]).contiguous())
        solver._kernel_predict(0)
        solver._kernel_sweeps(0)
        solver._kernel_update_velocity(0)
        return float((w * solver.verts.pos.to_numpy()[1, :, 0]).sum())

    forward(x0, v0)
    solver.reset_grad()
    adj = np.zeros((solver._sim.substeps_local + 1, n, 1, 3))
    adj[1, :, 0, :] = w
    solver.adj.pos.from_numpy(adj)
    solver.adj.vel.from_numpy(np.zeros_like(adj))
    solver.substep_pre_coupling_grad_sweep(0)
    gx = solver.adj.pos.to_numpy()[0, :, 0, :]
    gv = solver.adj.vel.to_numpy()[0, :, 0, :]

    eps = 1e-6
    fx, fv = np.zeros((n, 3)), np.zeros((n, 3))
    for i in range(n):
        for c in range(3):
            for arr, base, out in ((x0, x0, fx), (v0, v0, fv)):
                pass
            dx = np.zeros((n, 3))
            dx[i, c] = eps
            fx[i, c] = (forward(x0 + dx, v0) - forward(x0 - dx, v0)) / (2 * eps)
            fv[i, c] = (forward(x0, v0 + dx) - forward(x0, v0 - dx)) / (2 * eps)
    print(f"sweeps={n_iterations}: |dL/dx| max {np.abs(fx).max():.4e}, error {np.abs(gx - fx).max():.3e}; "
          f"|dL/dv| max {np.abs(fv).max():.4e}, error {np.abs(gv - fv).max():.3e}", flush=True)
    np.testing.assert_allclose(gx, fx, atol=1e-7 * np.abs(fx).max(), rtol=0)
    np.testing.assert_allclose(gv, fv, atol=1e-7 * np.abs(fv).max(), rtol=0)


def _rig_loaded(show_viewer, n_iterations):
    """Everything the fast path can carry: floor contact, anisotropic friction, damping and a fibre term."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=1, gravity=(0.0, 0.0, -9.81), requires_grad=True),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, grad_converge=False, contact_stiffness=2e3, damping=0.01),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        material=gs.materials.VBD.Muscle(E=2e4, nu=0.3, gain=0.4, mu_forward=0.2, mu_backward=0.6, mu_lateral=0.9),
        morph=gs.morphs.Box(size=(0.2, 0.1, 0.1), pos=(0.0, 0.0, 0.045), nobisect=False, maxvolume=4e-4),
    )
    scene.build()
    box.set_muscle(np.zeros(box.n_elements, dtype=np.int32), np.tile([0.6, 0.0, 0.8], (box.n_elements, 1)))
    box.set_friction_frame(np.tile([0.8, 0.6, 0.0], (box.n_vertices, 1)))
    box.set_fiber_stiffness(np.full(box.n_elements, 3e5))
    box.set_actuation([0.5])
    return scene, box, scene.vbd_solver


@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_iterations", [1, 2])
def test_the_reverse_sweep_carries_contact_friction_damping_and_fibres(show_viewer, n_iterations):
    scene, box, solver = _rig_loaded(show_viewer, n_iterations)
    n = solver.n_vertices
    rng = np.random.default_rng(1)
    x0 = solver.verts.pos.to_numpy()[0, :, 0].copy()
    x0[:, 2] *= 0.96  # pressed into the floor, so contact and friction are live
    v0 = np.tile([0.6, -0.3, 0.0], (n, 1)) + 0.05 * rng.normal(size=(n, 3))
    w = rng.normal(size=(n, 3))

    def forward(x, v):
        solver._kernel_set_state(0, torch.as_tensor(x[None]).contiguous(), torch.as_tensor(v[None]).contiguous())
        solver._kernel_predict(0)
        solver._kernel_sweeps(0)
        solver._kernel_update_velocity(0)
        return float((w * solver.verts.pos.to_numpy()[1, :, 0]).sum())

    forward(x0, v0)
    solver.reset_grad()
    adj = np.zeros((solver._sim.substeps_local + 1, n, 1, 3))
    adj[1, :, 0, :] = w
    solver.adj.pos.from_numpy(adj)
    solver.adj.vel.from_numpy(np.zeros_like(adj))
    solver.substep_pre_coupling_grad_sweep(0)
    gx = solver.adj.pos.to_numpy()[0, :, 0, :]

    eps = 1e-6
    fx = np.zeros((n, 3))
    for i in range(n):
        for c in range(3):
            d = np.zeros((n, 3))
            d[i, c] = eps
            fx[i, c] = (forward(x0 + d, v0) - forward(x0 - d, v0)) / (2 * eps)
    print(f"loaded, sweeps={n_iterations}: |dL/dx| max {np.abs(fx).max():.4e}, error {np.abs(gx - fx).max():.3e}", flush=True)
    np.testing.assert_allclose(gx, fx, atol=1e-6 * np.abs(fx).max(), rtol=0)


def _rig_constrained(show_viewer, n_iterations):
    """A block with the hard constraints the skeleton uses: an equality distance and a bounded angle."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=1, gravity=(0.0, 0.0, -9.81), requires_grad=True),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, grad_converge=False, floor_height=-10.0),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        material=gs.materials.VBD.Base(E=2e4, nu=0.3),
        morph=gs.morphs.Box(size=(0.2, 0.1, 0.1), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=4e-4),
    )
    p = box.init_positions.cpu().numpy()
    v = [int(np.argmin(p[:, 0])), int(np.argmax(p[:, 0])), int(np.argmin(p[:, 1])), int(np.argmax(p[:, 1]))]
    box.add_distance_constraints(np.array([[v[0], v[1]]]))
    u, vv = p[v[0]] - p[v[1]], p[v[2]] - p[v[3]]
    a0 = np.degrees(np.arccos(u.dot(vv) / np.linalg.norm(u) / np.linalg.norm(vv)))
    box.add_angle_constraints(np.array([[v[0], v[1], v[2], v[3]]]), np.array([a0 - 8.0]), np.array([a0 + 8.0]))
    scene.build()
    return scene, box, scene.vbd_solver


@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_iterations", [1, 2])
def test_the_reverse_sweep_carries_the_augmented_lagrangian(show_viewer, n_iterations):
    """The multipliers and the stiffness are solver state that the forward changes every sweep, so the reverse
    undoes them too. The angle constraint keeps a small residual, from the position tangent of its Hessian proxy."""
    scene, box, solver = _rig_constrained(show_viewer, n_iterations)
    n = solver.n_vertices
    rng = np.random.default_rng(2)
    x0 = solver.verts.pos.to_numpy()[0, :, 0] + 0.01 * rng.normal(size=(n, 3))
    v0 = 0.2 * rng.normal(size=(n, 3))
    w = rng.normal(size=(n, 3))

    def forward(x):
        solver.reset_constraints(torch.ones(1, dtype=torch.bool, device=gs.device))
        solver._kernel_set_state(0, torch.as_tensor(x[None]).contiguous(), torch.as_tensor(v0[None]).contiguous())
        solver._kernel_predict(0)
        solver._kernel_sweeps(0)
        solver._kernel_update_velocity(0)
        return float((w * solver.verts.pos.to_numpy()[1, :, 0]).sum())

    forward(x0)
    solver.reset_grad()
    adj = np.zeros((solver._sim.substeps_local + 1, n, 1, 3))
    adj[1, :, 0, :] = w
    solver.adj.pos.from_numpy(adj)
    solver.adj.vel.from_numpy(np.zeros_like(adj))
    solver.substep_pre_coupling_grad_sweep(0)
    gx = solver.adj.pos.to_numpy()[0, :, 0, :]

    eps, fx = 1e-6, np.zeros((n, 3))
    for i in range(n):
        for c in range(3):
            d = np.zeros((n, 3))
            d[i, c] = eps
            fx[i, c] = (forward(x0 + d) - forward(x0 - d)) / (2 * eps)
    print(f"constrained, sweeps={n_iterations}: |dL/dx| max {np.abs(fx).max():.4e}, error {np.abs(gx - fx).max():.3e}", flush=True)
    np.testing.assert_allclose(gx, fx, atol=1e-3 * np.abs(fx).max(), rtol=0)
