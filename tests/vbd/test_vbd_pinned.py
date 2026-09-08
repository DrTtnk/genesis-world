"""Pinned vertices: flesh driven by a prescribed boundary.

A rigged character moves its skeleton and the flesh follows. The bones are prescribed rather than
solved, so the vertices they own are boundary conditions of the solve, not unknowns of it. Pinning
one is therefore exact by construction, and the interesting behaviour is what the free flesh next
to it does.

A prescribed boundary is infinitely strong: nothing the flesh does can slow a pinned vertex down.
That is the deliberate trade and the reason these tests check shape, never force.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

BAR = dict(size=(0.4, 0.06, 0.06), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=2e-5)


def _bar_scene(gravity=(0.0, 0.0, -9.81), n_iterations=8, substeps=10):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=substeps, gravity=gravity),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-10.0),
        show_viewer=False,
    )
    bar = scene.add_entity(material=gs.materials.VBD.Muscle(E=1e5, nu=0.3), morph=gs.morphs.Box(**BAR))
    scene.build()
    return scene, bar


def _rest(bar):
    return tensor_to_array(bar.init_positions)


def _positions(bar):
    return tensor_to_array(bar.get_positions())[0]


def _left_end(bar):
    rest = _rest(bar)
    return rest[:, 0] < rest[:, 0].min() + 0.02


@pytest.mark.required
def test_a_pinned_vertex_stays_exactly_on_its_target_while_the_rest_falls():
    scene, bar = _bar_scene()
    held = _left_end(bar)
    bar.set_pinned(held)
    target = _rest(bar)

    for _ in range(40):
        scene.step()

    pos = _positions(bar)
    assert np.abs(pos[held] - target[held]).max() < 1e-6, "a prescribed vertex is a boundary condition, not a guess"
    assert pos[~held][:, 2].min() < -0.01, "the free flesh must sag under gravity"


@pytest.mark.required
def test_the_same_bar_falls_freely_with_nothing_pinned():
    scene, bar = _bar_scene()
    for _ in range(40):
        scene.step()
    pos = _positions(bar)
    assert pos[:, 2].max() < -0.01, "with no pin the whole bar falls"


@pytest.mark.required
def test_moving_the_targets_drags_the_flesh_along():
    """What the rig will do: the bone moves, the pinned vertices go with it exactly, and the free
    flesh follows behind, stretched."""
    scene, bar = _bar_scene(gravity=(0.0, 0.0, 0.0))
    held = _left_end(bar)
    bar.set_pinned(held)
    rest = _rest(bar)
    lift, ramp = 0.15, 20

    for step in range(40):
        target = rest.copy()
        target[held, 2] += lift * min(1.0, (step + 1) / ramp)
        bar.set_pin_targets(target)
        scene.step()

    pos = _positions(bar)
    assert np.abs(pos[held, 2] - (rest[held, 2] + lift)).max() < 1e-6
    rise = pos[~held, 2] - rest[~held, 2]
    assert rise.max() > 0.01, f"the free flesh must be dragged upward, moved {rise.max():.4f} m"
    # The flesh must deform, not ride along. A rigid translation would lift every free vertex by the
    # same amount; here the near flesh stays with the boundary while the far end swings, so the
    # spread of the rise is a large fraction of the lift itself. Do not test for the far end lagging:
    # the bar is undamped and its period is shorter than the ramp, so the tip has already overshot
    # before the boundary stops.
    assert rise.max() - rise.min() > 0.2 * lift, f"rise spread only {rise.max() - rise.min():.4f} m"


@pytest.mark.required
def test_pinning_every_vertex_makes_the_body_follow_the_targets_exactly():
    """The starting point of the lizard plan: fully skinned first, then release the outer layers."""
    scene, bar = _bar_scene(gravity=(0.0, 0.0, -9.81))
    bar.set_pinned(np.ones(bar.n_vertices, dtype=bool))
    rest = _rest(bar)

    for step in range(10):
        target = rest + np.array([0.0, 0.0, 0.01 * (step + 1)])
        bar.set_pin_targets(target)
        scene.step()

    assert np.abs(_positions(bar) - (rest + np.array([0.0, 0.0, 0.1]))).max() < 1e-6
