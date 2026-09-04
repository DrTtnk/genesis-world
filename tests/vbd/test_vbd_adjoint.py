"""Hand-written adjoint of the VBD substep, checked against finite differences in float64.

The rigs put every suspect term in play at once: a block in floor contact, sliding obliquely to its friction
tangent with three distinct coefficients, compressed and actuated along a tilted fiber."""

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

FIBER = np.array([0.6, 0.0, 0.8])
W_DIR = np.array([0.8, 0.6, 0.0])  # tangent, oblique to the kick


def _rig(show_viewer, substeps):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=3e-3, substeps=substeps, gravity=(0.0, 0.0, -9.81), requires_grad=True),
        vbd_options=gs.options.VBDOptions(n_iterations=4, residual_tol=1e-9, max_sweeps=4000, contact_stiffness=2e3),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        material=gs.materials.VBD.Muscle(E=2e4, nu=0.3, gain=0.4, mu_forward=0.2, mu_backward=0.6, mu_lateral=0.9),
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.048), nobisect=False, maxvolume=3e-4),
    )
    scene.build()
    box.set_muscle(np.zeros(box.n_elements, dtype=np.int32), np.tile(FIBER / np.linalg.norm(FIBER), (box.n_elements, 1)))
    box.set_friction_frame(np.tile(W_DIR, (box.n_vertices, 1)))
    box.set_actuation([0.5])
    return scene, box


def _kick(scene, box, f):
    """Kick along +x with a -y component so the slide is oblique to the tangent, and squash a little."""
    state = box.get_state()
    pos, vel = state.pos.detach().clone(), state.vel.detach().clone()
    vel[:] = torch.as_tensor([0.6, -0.3, 0.0], dtype=vel.dtype, device=vel.device)
    pos[..., 2] *= 0.97
    scene.vbd_solver._kernel_set_state(f, pos.contiguous(), vel.contiguous())


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_jacobian_matches_finite_differences_of_the_residual(show_viewer):
    scene, box = _rig(show_viewer, substeps=1)
    solver = scene.vbd_solver
    scene.step()  # settle into contact
    _kick(scene, box, 0)
    solver._kernel_predict(0)
    solver._kernel_sweeps(0)  # a generic, unconverged iterate: J is the Jacobian of r at any x

    n = solver.n_vertices
    pos = torch.zeros((1, n, 3), dtype=torch.float64, device=gs.device)
    solver._kernel_get_state(1, pos, torch.zeros_like(pos))

    def residual(p):
        solver._kernel_set_state(1, p.contiguous(), torch.zeros_like(p))
        out = torch.zeros_like(p)
        solver._kernel_residual_vector(0, out)
        return out.cpu().numpy().reshape(-1)

    assert solver.verts[1, 0, 0].pos[2] < solver._floor_height + 1e-9 or True  # contact presence checked below
    contact = int((pos[0, :, 2] < solver._floor_height).sum().item())
    assert contact > 0, "rig must have vertices in floor contact"

    J_fd = np.zeros((3 * n, 3 * n))
    eps = 1e-7
    for k in range(3 * n):
        dp = torch.zeros_like(pos)
        dp.view(-1)[k] = eps
        J_fd[:, k] = (residual(pos + dp) - residual(pos - dp)) / (2 * eps)
    solver._kernel_set_state(1, pos.contiguous(), torch.zeros_like(pos))

    J = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        e = torch.zeros_like(pos)
        e.view(-1)[k] = 1.0
        out = torch.zeros_like(pos)
        solver._kernel_apply_jacobian(0, e.contiguous(), out)
        J[:, k] = out.cpu().numpy().reshape(-1)

    scale = np.abs(J_fd).max()
    print(f"n_vertices={n} contact_vertices={contact} |J|={scale:.3e} max|J - J_fd|={np.abs(J - J_fd).max():.3e} "
          f"asym={np.abs(J - J.T).max():.3e}", flush=True)
    np.testing.assert_allclose(J, J_fd, atol=1e-6 * scale, rtol=0)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_adjoint_gradients_match_finite_differences_over_three_substeps(show_viewer):
    substeps = 3
    scene, box = _rig(show_viewer, substeps=substeps)
    solver = scene.vbd_solver
    scene.step()  # settle into contact; frame 0 is now the start state of the next step
    _kick(scene, box, 0)
    n = solver.n_vertices
    rng = np.random.default_rng(3)
    w = torch.as_tensor(rng.normal(size=(1, n, 3)), dtype=torch.float64, device=gs.device)

    def get(f):
        pos = torch.zeros((1, n, 3), dtype=torch.float64, device=gs.device)
        vel = torch.zeros_like(pos)
        solver._kernel_get_state(f, pos, vel)
        return pos, vel

    x0, v0 = get(0)

    def rollout(x0, v0, actu):
        solver._kernel_set_state(0, x0.contiguous(), v0.contiguous())
        solver.set_actuation(np.array([[actu]], dtype=np.float64))
        for f in range(substeps):
            solver.substep_pre_coupling(f)
        x, v = get(substeps)
        return float((w * x).sum() + 0.5 * (v * v).sum()), x, v

    L0, x_end, v_end = rollout(x0, v0, 0.5)

    solver.reset_grad()
    solver.adj.pos.from_numpy(np.zeros((substeps + 1, n, 1, 3)))
    adj_pos = np.zeros((substeps + 1, n, 1, 3)); adj_pos[substeps, :, 0, :] = w[0].cpu().numpy()
    adj_vel = np.zeros((substeps + 1, n, 1, 3)); adj_vel[substeps, :, 0, :] = v_end[0].cpu().numpy()
    solver.adj.pos.from_numpy(adj_pos)
    solver.adj.vel.from_numpy(adj_vel)
    for f in reversed(range(substeps)):
        solver.substep_pre_coupling_grad(f)
    g_x = solver.adj.pos.to_numpy()[0, :, 0, :]
    g_v = solver.adj.vel.to_numpy()[0, :, 0, :]
    g_a = float(solver.muscle_actu_adj.to_numpy()[0, 0])

    eps = 1e-6
    g_x_fd = np.zeros((n, 3)); g_v_fd = np.zeros((n, 3))
    for i in range(n):
        for j in range(3):
            d = torch.zeros_like(x0); d[0, i, j] = eps
            g_x_fd[i, j] = (rollout(x0 + d, v0, 0.5)[0] - rollout(x0 - d, v0, 0.5)[0]) / (2 * eps)
            g_v_fd[i, j] = (rollout(x0, v0 + d, 0.5)[0] - rollout(x0, v0 - d, 0.5)[0]) / (2 * eps)
    g_a_fd = (rollout(x0, v0, 0.5 + eps)[0] - rollout(x0, v0, 0.5 - eps)[0]) / (2 * eps)

    print(f"|dL/dx|={np.abs(g_x_fd).max():.3e} err={np.abs(g_x - g_x_fd).max():.3e}  "
          f"|dL/dv|={np.abs(g_v_fd).max():.3e} err={np.abs(g_v - g_v_fd).max():.3e}  "
          f"dL/da adjoint={g_a:.6e} fd={g_a_fd:.6e}", flush=True)
    np.testing.assert_allclose(g_x, g_x_fd, atol=1e-6 * np.abs(g_x_fd).max(), rtol=0)
    np.testing.assert_allclose(g_v, g_v_fd, atol=1e-6 * np.abs(g_v_fd).max(), rtol=0)
    assert g_a == pytest.approx(g_a_fd, rel=1e-6)
