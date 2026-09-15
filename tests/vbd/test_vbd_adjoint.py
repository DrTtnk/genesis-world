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


def _constrain(box):
    """Every hard-constraint case at once, all under load: an equality shorter than rest, a bounded distance active
    at its lower bound, a bounded distance inside its bounds (inactive), and an angle active at its lower bound."""
    p = tensor_to_array(box.init_positions)
    q = p - p.mean(axis=0)
    # six distinct corners of the box (each axis with a tie-break that lands on a different corner), then the two
    # vertices farthest from everything already picked
    v = [int(np.argmax(q @ d)) for d in ([1, 0.3, 0.2], [-1, -0.3, -0.2], [-0.3, 1, -0.2], [0.3, -1, 0.2], [-0.3, -0.2, 1], [0.3, 0.2, -1])]
    for _ in range(2):
        v.append(int(np.argmax(np.linalg.norm(p[:, None] - p[v][None], axis=-1).min(axis=1))))
    assert len(set(v)) == 8, v
    d = lambda i, j: float(np.linalg.norm(p[v[i]] - p[v[j]]))
    box.add_distance_constraints(np.array([[v[0], v[1]]]), lo=np.array([0.98 * d(0, 1)]), hi=np.array([0.98 * d(0, 1)]))
    # the active bound is the lower one: the rig's squash shortens every distance and keeps it pressed against it
    box.add_distance_constraints(np.array([[v[2], v[3]], [v[4], v[5]]]), lo=np.array([1.05 * d(2, 3), 0.5 * d(4, 5)]), hi=np.array([1.5 * d(2, 3), 1.5 * d(4, 5)]))
    box.add_distance_constraints(np.array([[v[6], v[7]]]))  # an equality at its rest length: lightly loaded, still a constraint
    u, w = p[v[0]] - p[v[1]], p[v[2]] - p[v[3]]  # two body diagonals, about 109 degrees apart
    a0 = np.degrees(np.arccos(u.dot(w) / np.linalg.norm(u) / np.linalg.norm(w)))
    box.add_angle_constraints(np.array([[v[0], v[1], v[2], v[3]]]), np.array([a0 + 2.0]), np.array([a0 + 40.0]))


def _rig(show_viewer, substeps, k_fiber=0.0, constrained=False):
    # damping: the constrained rig must settle against its bounds instead of ringing off them
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=3e-3, substeps=substeps, gravity=(0.0, 0.0, -9.81), requires_grad=True),
        vbd_options=gs.options.VBDOptions(n_iterations=4, residual_tol=1e-12, violation_tol=1e-12, max_sweeps=4000, contact_stiffness=2e3, damping=0.01 if constrained else 0.0),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        material=gs.materials.VBD.Muscle(E=2e4, nu=0.3, gain=0.4, mu_forward=0.2, mu_backward=0.6, mu_lateral=0.9),
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.048), nobisect=False, maxvolume=3e-4),
    )
    if constrained:
        _constrain(box)
    scene.build()
    box.set_muscle(np.zeros(box.n_elements, dtype=np.int32), np.tile(FIBER / np.linalg.norm(FIBER), (box.n_elements, 1)))
    box.set_friction_frame(np.tile(W_DIR, (box.n_vertices, 1)))
    box.set_fiber_stiffness(np.full(box.n_elements, k_fiber))
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
@pytest.mark.parametrize("k_fiber", [0.0, 3e5])
@pytest.mark.parametrize("constrained", [False, True])
def test_jacobian_matches_finite_differences_of_the_residual(show_viewer, k_fiber, constrained):
    scene, box = _rig(show_viewer, substeps=1, k_fiber=k_fiber, constrained=constrained)
    solver = scene.vbd_solver
    scene.step()  # settle into contact
    _kick(scene, box, 0)
    solver._kernel_predict(0)
    for _sweep in range(solver._n_iterations):
        solver._kernel_sweeps(0, _sweep)  # a generic, unconverged iterate: J is the Jacobian of r at any x
    if constrained:
        mult = solver.cons_hist.mult.to_numpy()[1, :, 0]
        k_eff = solver.cons_hist.k_eff.to_numpy()[1, :, 0]
        assert mult[0] != 0.0 and mult[1] != 0.0 and mult[2] == 0.0, f"active set is not as designed: {mult}"
        assert k_eff[0] > 0.0 and k_eff[1] > 0.0 and k_eff[2] == 0.0 and k_eff[3] > 0.0, f"effective stiffness: {k_eff}"
        assert solver.acons_hist.k_eff.to_numpy()[1, 0, 0] > 0.0

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
@pytest.mark.parametrize("k_fiber", [0.0, 3e5])
@pytest.mark.parametrize("constrained", [False, True])
def test_adjoint_gradients_match_finite_differences_over_three_substeps(show_viewer, k_fiber, constrained):
    substeps = 3
    scene, box = _rig(show_viewer, substeps=substeps, k_fiber=k_fiber, constrained=constrained)
    solver = scene.vbd_solver
    for _ in range(20 if constrained else 1):
        scene.step()  # settle into contact (and against the bounds); frame 0 is now the start state of the next step
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
        # the multipliers are solver state that survives a rollout: without this every finite-difference evaluation
        # starts from the multipliers the previous one left, and the difference measures that drift, not the gradient
        solver.reset_constraints(torch.ones(1, dtype=torch.bool, device=gs.device))
        solver.set_actuation(np.array([[actu]], dtype=np.float64))
        for f in range(substeps):
            solver.substep_pre_coupling(f)
        x, v = get(substeps)
        return float((w * x).sum() + 0.5 * (v * v).sum()), x, v

    L0, x_end, v_end = rollout(x0, v0, 0.5)
    if constrained:
        assert solver.constraint_error() < 1e-6 and solver.angle_constraint_error() < 1e-6  # the tolerance is relative now
        mult = solver.cons_hist.mult.to_numpy()[1:, :, 0]  # the squash presses the bounded pair into its bound in substep 0
        k_eff = solver.cons_hist.k_eff.to_numpy()[1:, :, 0]
        assert (mult[:, 0] != 0.0).all() and mult[0, 1] != 0.0 and (k_eff[:, 2] == 0.0).all(), f"active set is not as designed: {mult}"
        assert (k_eff[:, 3] > 0.0).all(), "the rest-length equality must be recorded active whatever its load"
        assert solver.acons_hist.k_eff.to_numpy()[1, 0, 0] > 0.0

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


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_record_is_taken_at_the_converged_state_even_without_a_sweep(show_viewer):
    """A predicted state that is already stationary sweeps zero times; the record must still be this substep's."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=3e-3, substeps=1, gravity=(0.0, 0.0, 0.0), requires_grad=True),
        vbd_options=gs.options.VBDOptions(n_iterations=4, residual_tol=1e-9, max_sweeps=4000, floor_height=-1.0),
        show_viewer=show_viewer,
    )
    bar = scene.add_entity(material=gs.materials.VBD.Base(E=1e5, nu=0.3), morph=gs.morphs.Box(size=(0.2, 0.05, 0.05), pos=(0.0, 0.0, 0.5), nobisect=False, maxvolume=5e-5))
    p = tensor_to_array(bar.init_positions)
    bar.add_distance_constraints(np.array([[int(np.argmin(p[:, 0])), int(np.argmax(p[:, 0]))]]))  # equality at rest
    scene.build()
    solver = scene.vbd_solver
    solver.cons_hist.mult.fill(123.0)  # stale garbage from "a previous substep"
    solver.cons_hist.k_eff.fill(0.0)
    scene.step()
    assert solver.cons_hist.mult.to_numpy()[1, 0, 0] == 0.0
    assert solver.cons_hist.k_eff.to_numpy()[1, 0, 0] == solver._k_start


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_the_solve_tolerance_is_relative_to_the_body(show_viewer):
    """The same tolerance must mean the same accuracy on bodies of different mass and different constraint length.
    An absolute newton tolerance asks a heavy body for a hundred times more digits than a light one, which is why
    the full snake could not be differentiated at all."""
    errors = {}
    for rho, size in ((1e3, 0.1), (1e6, 0.5)):  # a 1 kg block and a 125 t one, five times longer
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=3e-3, substeps=1, gravity=(0.0, 0.0, -9.81), requires_grad=True),
            vbd_options=gs.options.VBDOptions(n_iterations=4, residual_tol=1e-6, violation_tol=1e-6, max_sweeps=4000, contact_stiffness=2e5),
            show_viewer=show_viewer,
        )
        box = scene.add_entity(
            material=gs.materials.VBD.Base(E=2e4, nu=0.3, rho=rho),
            morph=gs.morphs.Box(size=(size, size, size), pos=(0.0, 0.0, 0.48 * size), nobisect=False, maxvolume=0.3 * size**3),  # pressed into the floor: a real imbalance to solve
        )
        p = tensor_to_array(box.init_positions)
        box.add_distance_constraints(np.array([[int(np.argmin(p[:, 0])), int(np.argmax(p[:, 0]))]]))
        scene.build()
        scene.step()
        solver = scene.vbd_solver
        errors[rho] = (solver.constraint_error() / size, float(solver.residual[None]) / solver._force_ref)
    light, heavy = errors[1e3], errors[1e6]
    print(f"strain error: light={light[0]:.2e} heavy={heavy[0]:.2e}; relative force residual: light={light[1]:.2e} heavy={heavy[1]:.2e}", flush=True)
    # both bodies are solved to the tolerance they were asked for, in their own units: an absolute newton tolerance
    # would put the heavy body 1e5 times worse in strain, or out of reach of float64 entirely
    assert max(light[0], heavy[0]) < 1e-6 and max(light[1], heavy[1]) < 1e-6
