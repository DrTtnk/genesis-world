"""End-to-end `scene.backward` gradient of a VBD muscle entity's actuation, checked against finite differences.

Mirrors `tests/grad/test_mpm_muscle_grad.py`: a rollout of several steps, each with its own muscle actuation
value fed through `entity.set_actuation`, differentiated through the full `_tgt`/`_tgt_buffer` input-replay
machinery (not the hand-written adjoint math itself, already covered by `tests/vbd/test_vbd_adjoint.py`).
"""

import numpy as np
import pytest
import torch

import genesis as gs

FIBER = np.array([0.6, 0.0, 0.8])
N_STEPS = 3


def _build(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=3e-3, substeps=2, gravity=(0.0, 0.0, -9.81), requires_grad=True),
        vbd_options=gs.options.VBDOptions(n_iterations=4, residual_tol=1e-9, max_sweeps=4000, contact_stiffness=2e3),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        material=gs.materials.VBD.Muscle(E=2e4, nu=0.3, gain=0.4, mu_forward=0.2, mu_backward=0.6, mu_lateral=0.9),
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.048), nobisect=False, maxvolume=3e-4),
    )
    scene.build()
    box.set_muscle(
        np.zeros(box.n_elements, dtype=np.int32),
        np.tile(FIBER / np.linalg.norm(FIBER), (box.n_elements, 1)),
    )
    return scene, box


def _rollout(scene, box, actu_values, w, requires_grad):
    """Roll `N_STEPS` steps with the given per-step actuation and return `(loss, per-step actuation tensors)`.

    `scene.reset()` restores the exact state captured at `scene.build()` (see `Scene.build`, which calls
    `self._reset()` once before `_is_built` is set), so replaying from there reproduces the same trajectory
    bit for bit given the same actuation.
    """
    scene.reset()
    tensors = []
    for value in actu_values:
        actus = gs.tensor([value], requires_grad=requires_grad)
        tensors.append(actus)
        box.set_actuation(actus)
        scene.step()
    state = box.get_state()
    loss = (w * state.pos).sum() + 0.5 * (state.vel**2).sum()
    return loss, tensors


@pytest.mark.parametrize("precision", ["64"])
def test_vbd_actuation_grad_matches_finite_differences(show_viewer):
    """The gradient of a pos/vel loss w.r.t. each step's muscle actuation must match finite differences."""
    scene, box = _build(show_viewer)

    rng = np.random.default_rng(0)
    w = gs.tensor(rng.normal(size=(1, box.n_vertices, 3)))

    nominal = [0.3 + 0.1 * i for i in range(N_STEPS)]

    loss, tensors = _rollout(scene, box, nominal, w, requires_grad=True)
    scene.backward(loss)
    analytic = [t.grad.item() for t in tensors]

    eps = 1e-5
    numeric = []
    for i in range(N_STEPS):
        plus = list(nominal)
        plus[i] += eps
        minus = list(nominal)
        minus[i] -= eps
        with torch.no_grad():
            loss_plus, _ = _rollout(scene, box, plus, w, requires_grad=False)
            loss_minus, _ = _rollout(scene, box, minus, w, requires_grad=False)
        numeric.append((loss_plus.item() - loss_minus.item()) / (2 * eps))

    for i in range(N_STEPS):
        print(f"step {i}: analytic={analytic[i]:.6e} fd={numeric[i]:.6e}", flush=True)

    for i in range(N_STEPS):
        assert abs(analytic[i]) > 1e-9, f"actuation gradient at step {i} is zero, so it is not reaching the input"
        assert analytic[i] == pytest.approx(numeric[i], rel=1e-5), (
            f"step {i}: analytic gradient {analytic[i]:.6e} disagrees with finite differences {numeric[i]:.6e}"
        )


DECIMATION = 10


def _rollout_decimated(scene, box, actu_values, w, requires_grad):
    """One actuation per control step, held over `DECIMATION` simulation steps: the pattern of the RL environment.
    The gradient of the loss with respect to a control step's actuation must sum over every simulation step that
    used it."""
    scene.reset()
    tensors = []
    for value in actu_values:
        actus = gs.tensor([value], requires_grad=requires_grad)
        tensors.append(actus)
        box.set_actuation(actus)
        for _ in range(DECIMATION):
            scene.step()
    state = box.get_state()
    loss = (w * state.pos).sum() + 0.5 * (state.vel**2).sum()
    return loss, tensors


@pytest.mark.parametrize("precision", ["64"])
def test_vbd_actuation_grad_with_decimation_matches_finite_differences(show_viewer):
    """APG story 1: an actuation set once and held over ten steps gets the gradient of all ten."""
    scene, box = _build(show_viewer)
    rng = np.random.default_rng(1)
    w = gs.tensor(rng.normal(size=(1, box.n_vertices, 3)))
    nominal = [0.3, 0.5]

    loss, tensors = _rollout_decimated(scene, box, nominal, w, requires_grad=True)
    scene.backward(loss)
    analytic = [t.grad.item() for t in tensors]

    eps = 1e-5
    numeric = []
    for i in range(len(nominal)):
        plus, minus = list(nominal), list(nominal)
        plus[i] += eps
        minus[i] -= eps
        with torch.no_grad():
            numeric.append((_rollout_decimated(scene, box, plus, w, False)[0].item() - _rollout_decimated(scene, box, minus, w, False)[0].item()) / (2 * eps))
    for i in range(len(nominal)):
        print(f"control step {i}: analytic={analytic[i]:.6e} fd={numeric[i]:.6e}", flush=True)
        assert abs(analytic[i]) > 1e-9
        assert analytic[i] == pytest.approx(numeric[i], rel=1e-5)


@pytest.mark.parametrize("precision", ["64"])
def test_held_actuation_does_not_leak_into_the_next_rollout(show_viewer):
    """After a reset, steps taken before the first set_actuation carry no actuation and must not back-propagate into the
    previous rollout's tensor."""
    scene, box = _build(show_viewer)
    w = gs.tensor(np.zeros((1, box.n_vertices, 3)))
    loss, tensors = _rollout_decimated(scene, box, [0.4], w, requires_grad=True)
    scene.backward(loss)
    grad_before = tensors[0].grad.clone()
    scene.reset()
    for _ in range(3):
        scene.step()  # no set_actuation: nothing in force
    state = box.get_state()
    scene.backward((state.pos**2).sum())
    assert torch.equal(tensors[0].grad, grad_before)
