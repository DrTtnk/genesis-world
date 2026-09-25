"""Native VIPER rods against the Torch reference they reimplement.

Both minimise the same incremental potential over the same blocks; they differ in order (the reference sweeps one
rod's nodes, frames and scales in sequence, the native path colours them), so single sweeps differ and converged
substeps must agree. Sweeping enough to converge each substep, a hanging bent rod must follow the reference's
trajectory; the reference's own scene gates must hold natively; and a batch must equal its environments run alone.
"""

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

pytestmark = pytest.mark.precision("64")

MATERIAL = dict(rho=1060.0, stretch_x=2e4, stretch_y=3e4, stretch_z=5e4, volume=1e5, surface_bend=4e3)


def _frames_along(verts, twist):
    """Scalar-first frames whose third axis follows each segment, turned by `twist` about it."""
    frames = []
    for j in range(len(verts) - 1):
        d = verts[j + 1] - verts[j]
        d = d / np.linalg.norm(d)
        axis = np.cross([0.0, 0.0, 1.0], d)
        s, c = np.linalg.norm(axis), d[2]
        if s < 1e-12:
            # parallel or antiparallel to z: no rotation, or half a turn about x
            align = np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
        else:
            half = np.arctan2(s, c) / 2
            align = np.concatenate(([np.cos(half)], np.sin(half) * axis / s))
        spin = np.array([np.cos(twist[j] / 2), 0.0, 0.0, np.sin(twist[j] / 2)])
        w1, v1, w2, v2 = align[0], align[1:], spin[0], spin[1:]
        frames.append(np.concatenate(([w1 * w2 - v1 @ v2], w1 * v2 + w2 * v1 + np.cross(v1, v2))))
    return np.array(frames)


def _bent_rod():
    angles = np.array([0.0, 0.35, -0.2, 0.5, 0.1])
    steps = 0.006 * np.stack((np.sin(np.cumsum(angles)), np.zeros(5), -np.cos(np.cumsum(angles))), axis=1)
    verts = np.concatenate(([[0.0, 0.0, 0.3]], np.array([0.0, 0.0, 0.3]) + np.cumsum(steps, 0)))
    return verts, _frames_along(verts, twist=np.array([0.1, 0.4, -0.3, 0.2, 0.0]))


def _scene(solver, n_iterations, dt=2e-3, gravity=(0.0, 0.0, -9.81), n_envs=None, verts=None, frames=None):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=gravity),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-float("inf"), rod_solver=solver),
        show_viewer=False,
    )
    if verts is None:
        verts, frames = _bent_rod()
    rod = scene.add_entity(morph=gs.morphs.Rod(verts=verts, frames=frames, radius=6e-4),
                           material=gs.materials.VBD.Rod(**MATERIAL))
    if n_envs is None:
        scene.build()
    else:
        scene.build(n_envs=n_envs)
    return scene, rod


def _trajectory(solver, steps=10, n_iterations=400):
    scene, rod = _scene(solver, n_iterations)
    pinned = np.zeros(rod.n_vertices, dtype=bool)
    pinned[0] = True
    rod.set_pinned(pinned)
    out = []
    for _ in range(steps):
        scene.step()
        scene.sim.vbd_solver.check_errno()
        state = rod.get_rod_state()
        scale, quat = (np.asarray(tensor_to_array(v)) for v in (state.scale, state.quat))
        out.append((np.asarray(tensor_to_array(rod.get_positions()))[0], scale.reshape(-1, scale.shape[-1])[0],
                    quat.reshape(-1, *quat.shape[-2:])[0]))
    return out


@pytest.mark.required
def test_a_converged_native_rod_follows_the_reference_trajectory():
    reference, native = _trajectory("reference"), _trajectory("native")
    worst = max(np.abs(a[0] - b[0]).max() for a, b in zip(reference, native))
    moved = np.abs(reference[-1][0] - reference[0][0]).max()
    print(f"after {len(native)} steps: moved {1e3 * moved:.4f} mm, worst position difference {worst:.3e} m")
    assert moved > 1e-4, "fixture sanity check: the rod must swing"
    # each solver converges a substep to about 1e-11 m; over ten steps they stay within a nanometre of each other,
    # a millionth of the swing
    for (x_r, s_r, q_r), (x_n, s_n, q_n) in zip(reference, native):
        np.testing.assert_allclose(x_n, x_r, atol=1e-9, rtol=0)
        np.testing.assert_allclose(s_n, s_r, atol=1e-9, rtol=0)
        # a frame and its negation are the same rotation
        np.testing.assert_allclose(np.abs((q_n * q_r).sum(-1)), 1.0, atol=1e-12, rtol=0)


@pytest.mark.required
def test_a_native_rod_falls_freely_on_its_first_step():
    verts = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.24], [0.0, 0.0, 0.28]])
    scene, rod = _scene("native", 8, dt=0.01, verts=verts, frames=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)))
    scene.step()
    np.testing.assert_allclose(tensor_to_array(rod.get_positions())[0], verts + [0.0, 0.0, -0.000981], atol=1e-12)
    np.testing.assert_allclose(tensor_to_array(rod.get_rod_state().scale)[0], np.ones(3), atol=1e-12)


@pytest.mark.required
def test_a_native_rod_stretched_by_its_pins_thins_as_the_reference_does():
    """The reference's prescribed-extension gate: stretched 25 %, the section thins toward its volume. With this
    material's volume coefficient the volume error is 2.1 % in both."""
    verts = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.24], [0.0, 0.0, 0.28]])
    scales = {}
    for solver in ("reference", "native"):
        scene, rod = _scene(solver, 8, dt=0.01, gravity=(0.0, 0.0, 0.0), verts=verts,
                            frames=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)))
        rod.set_pinned(np.ones(3, dtype=bool))
        target = verts.copy()
        target[:, 2] = 0.2 + (target[:, 2] - 0.2) * 1.25
        rod.set_pin_targets(target)
        for _ in range(3):
            scene.step()
            scene.sim.vbd_solver.check_errno()
        scales[solver] = torch.as_tensor(tensor_to_array(rod.get_rod_state().scale)).reshape(-1, 3)[0]
    assert bool((scales["native"] < 1).all() & (scales["native"] > 0).all())
    # eight sweeps converge neither; the two orders agree to a few nanometres in scale
    torch.testing.assert_close(scales["native"], scales["reference"], atol=1e-7, rtol=0)


@pytest.mark.required
def test_a_batch_of_native_rods_equals_each_environment_alone():
    def run(n_envs, targets):
        scene, rod = _scene("native", 30, n_envs=n_envs)
        pinned = np.zeros(rod.n_vertices, dtype=bool)
        pinned[0] = True
        rod.set_pinned(pinned)
        rod.set_pin_targets(np.asarray(targets))
        for _ in range(10):
            scene.step()
            scene.sim.vbd_solver.check_errno()
        return np.asarray(tensor_to_array(rod.get_positions()))

    verts, _ = _bent_rod()
    a, b = verts.copy(), verts.copy()
    b[0] += [0.004, 0.0, 0.0]
    batch = run(2, np.stack((a, b)))
    alone_a, alone_b = run(None, a), run(None, b)
    np.testing.assert_allclose(batch[0], alone_a[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(batch[1], alone_b[0], atol=1e-12, rtol=0)
    assert np.abs(batch[0] - batch[1]).max() > 1e-4


@pytest.mark.required
def test_native_blocks_are_the_reference_gauss_newton_systems():
    """From one random state, the kernels' node and frame systems and the scale block's full Gauss-Newton step
    must equal those Torch AD gives the reference residual (`vbd_rod.RodModel`), the systems
    `spikes/verify_viper_native_blocks.py` states in closed form."""
    import quadrants as qd

    from genesis.engine.solvers.vbd_rod import RodModel, RodParameters, retract
    from genesis.engine.solvers.vbd_rod_native import func_frame_system, func_rod_node_terms, func_solve_rod_scales, kernel_rod_begin

    dt = 2e-3
    scene, rod = _scene("native", 1, dt=dt)
    solver = scene.sim.vbd_solver
    native = solver.rod_native
    verts, frames = _bent_rod()
    rng = np.random.default_rng(7)
    model = RodModel(torch.tensor(verts), torch.tensor(frames), 6e-4, RodParameters(
        MATERIAL["rho"], MATERIAL["stretch_x"], MATERIAL["stretch_y"], MATERIAL["stretch_z"], MATERIAL["volume"],
        MATERIAL["surface_bend"]))
    x = torch.tensor(verts + rng.normal(0.0, 3e-4, verts.shape))
    velocity = torch.tensor(rng.normal(0.0, 0.05, verts.shape))
    model.quat = retract(model.quat, torch.tensor(rng.normal(0.0, 0.3, (len(frames), 3))))
    model.scale = torch.tensor(1.0 + 0.2 * (rng.random(len(verts)) - 0.5))
    model.director_velocity = torch.tensor(rng.normal(0.0, 0.1, model.director_velocity.shape))
    model.begin(x, velocity, dt, torch.tensor([0.0, 0.0, -9.81]))

    state = solver.get_state(0)
    state.pos[0] = x.to(state.pos)
    state.vel[0] = velocity.to(state.vel)
    state.rod_states = ((model.scale[None], model.quat[None], model.director_velocity[None]),)
    solver.set_state(0, state)
    solver._kernel_predict(0)
    kernel_rod_begin(solver, native)
    pos = torch.zeros((1, solver.n_vertices, 3), dtype=gs.tc_float)
    pos[0] = x.to(pos)
    solver._kernel_set_state(1, pos, velocity[None].to(pos))

    n, m = len(verts), len(frames)
    node_out = qd.field(dtype=gs.qd_float, shape=(n, 12))
    frame_out = qd.field(dtype=gs.qd_float, shape=(m, 12))

    @qd.kernel
    def probe(solver: qd.template(), rod: qd.template()):
        for i in range(n):
            force, H = func_rod_node_terms(0, i, 0, solver, rod)
            for a in qd.static(range(3)):
                node_out[i, a] = force[a]
                for b in qd.static(range(3)):
                    node_out[i, 3 + 3 * a + b] = H[a, b]
        for j in range(m):
            g, H = func_frame_system(0, j, 0, solver, rod)
            for a in qd.static(range(3)):
                frame_out[j, a] = g[a]
                for b in qd.static(range(3)):
                    frame_out[j, 3 + 3 * a + b] = H[a, b]
        for _ in range(1):
            func_solve_rod_scales(0, 0, 0, solver, rod)

    scale_before = model.scale.clone()
    probe(solver, native)
    node_out, frame_out = node_out.to_numpy(), frame_out.to_numpy()
    step = native.rhs.to_numpy()[0, 0, :n]

    def jac(kind, index):
        def residual(d):
            if kind == "node":
                return model.residual(x + torch.nn.functional.one_hot(torch.tensor(index), n).to(x)[:, None] * d,
                                      model.scale, model.quat)
            if kind == "frame":
                mask = torch.nn.functional.one_hot(torch.tensor(index), m).to(x)[:, None]
                return model.residual(x, model.scale, retract(model.quat, mask * d))
            return model.residual(x, model.scale + d, model.quat)
        width = n if kind == "scale" else 3
        J = torch.func.jacrev(residual)(x.new_zeros(width))
        return J, residual(x.new_zeros(width))

    for i in range(n):
        J, r = jac("node", i)
        inertia = model.position_weight[i, 0] ** 2 / dt**2
        g_rod = (J.T @ r).numpy() - float(inertia) * (x[i] - model.predicted_pos[i]).numpy()
        H_rod = (J.T @ J).numpy() - float(inertia) * np.eye(3)
        np.testing.assert_allclose(-node_out[i, :3], g_rod, rtol=1e-9, atol=1e-9 * np.abs(g_rod).max())
        np.testing.assert_allclose(node_out[i, 3:].reshape(3, 3), H_rod, rtol=1e-9, atol=1e-9 * np.abs(H_rod).max())
    for j in range(m):
        J, r = jac("frame", j)
        g, H = (J.T @ r).numpy(), (J.T @ J).numpy()
        np.testing.assert_allclose(frame_out[j, :3], g, rtol=1e-9, atol=1e-9 * np.abs(g).max())
        np.testing.assert_allclose(frame_out[j, 3:].reshape(3, 3), H, rtol=1e-9, atol=1e-9 * np.abs(H).max())
    J, r = jac("scale", 0)
    reference_step = -torch.linalg.solve(J.T @ J, J.T @ r).numpy()
    print(f"scale step native {step}, reference {reference_step}")
    np.testing.assert_allclose(step, reference_step, rtol=1e-8, atol=1e-10)
