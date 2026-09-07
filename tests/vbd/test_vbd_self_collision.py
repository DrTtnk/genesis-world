"""Self-collision, story S2: the contact force alone, with every vertex pair compared.

The broad phase does not exist yet, so these scenes are deliberately tiny. What is under test is
that a vertex pushes a non-neighbour vertex away until they are one thickness apart, and that the
vertices of the same tetrahedron never push each other, however close the mesh puts them.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

CUBE = dict(size=(0.04, 0.04, 0.04), nobisect=False, maxvolume=1e-5)
THICK = 0.02


def _two_cubes(thickness, gap, n_iterations=8, steps=40):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=10, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(
            n_iterations=n_iterations,
            floor_height=-1.0,
            contact_stiffness=1e6,
            self_collision_thickness=thickness,
        ),
        show_viewer=False,
    )
    mat = gs.materials.VBD.Muscle(E=1e5, nu=0.3)
    left = scene.add_entity(material=mat, morph=gs.morphs.Box(pos=(-gap / 2, 0.0, 0.0), **CUBE))
    right = scene.add_entity(material=mat, morph=gs.morphs.Box(pos=(gap / 2, 0.0, 0.0), **CUBE))
    scene.build()
    for _ in range(steps):
        scene.step()
    return scene, left, right


def _closest_cross_distance(left, right):
    a = tensor_to_array(left.get_positions())[0]
    b = tensor_to_array(right.get_positions())[0]
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min())


def _overlap(left, right):
    """How far the left cube reaches past the near face of the right one. The closest vertex pair is
    not the measure here, because a coarse tetrahedral mesh keeps its vertices apart even when the two
    volumes fully overlap."""
    a = tensor_to_array(left.get_positions())[0]
    b = tensor_to_array(right.get_positions())[0]
    return float(a[:, 0].max() - b[:, 0].min())


@pytest.mark.required
def test_two_overlapping_cubes_separate_to_the_thickness():
    """The pair force is lagged (Gauss-Seidel sees a partner from the previous sweep), so the
    penetration is what must converge, not the energy."""
    _, left, right = _two_cubes(THICK, gap=0.03)
    assert _closest_cross_distance(left, right) > 0.9 * THICK
    assert _overlap(left, right) < 0.0  # the volumes no longer share any space


@pytest.mark.required
def test_the_same_cubes_stay_interpenetrated_without_self_collision():
    _, left, right = _two_cubes(0.0, gap=0.03)
    assert _overlap(left, right) > 0.009  # the rest overlap of 0.01, kept


@pytest.mark.required
def test_cubes_further_apart_than_the_thickness_do_not_move():
    scene, left, right = _two_cubes(THICK, gap=0.2, steps=10)
    a = tensor_to_array(left.get_positions())[0]
    assert np.abs(a - tensor_to_array(left.init_positions)).max() < 1e-9
    assert _closest_cross_distance(left, right) > 0.15


@pytest.mark.required
def test_a_single_cube_ignores_its_own_mesh_at_a_thickness_it_cannot_satisfy():
    """Every edge of this cube is shorter than the thickness. Tetrahedron neighbours are exempt, so
    the only pairs left are the diagonal ones, and the mesh must not explode."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=10, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(
            n_iterations=8, floor_height=-1.0, contact_stiffness=1e6, self_collision_thickness=0.005
        ),
        show_viewer=False,
    )
    cube = scene.add_entity(material=gs.materials.VBD.Muscle(E=1e5, nu=0.3), morph=gs.morphs.Box(**CUBE))
    scene.build()
    rest = tensor_to_array(cube.get_positions())[0]
    for _ in range(20):
        scene.step()
    pos = tensor_to_array(cube.get_positions())[0]
    assert np.isfinite(pos).all()
    assert np.abs(pos - rest).max() < 0.02
