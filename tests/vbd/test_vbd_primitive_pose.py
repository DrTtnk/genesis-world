"""Primitive attachment landmarks must not move with tetrahedral refinement."""

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.utils.misc import tensor_to_array


@pytest.mark.parametrize("precision", ["64"])
def test_rotated_box_preserves_its_requested_corner_landmarks(show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.6, -0.8, 0.4), camera_lookat=(0.0, 0.0, 0.0)),
        show_viewer=show_viewer,
    )
    rng = np.random.default_rng(42)
    for _ in range(4):
        size = rng.uniform(0.005, 0.06, 3)
        position = rng.uniform(-0.1, 0.1, 3)
        offset = rng.uniform(-0.02, 0.02, 3)
        rotation = Rotation.random(random_state=rng)
        extra = Rotation.random(random_state=rng)
        tissue = scene.add_entity(
            morph=gs.morphs.Box(
                size=tuple(size),
                pos=tuple(position),
                quat=tuple(rotation.as_quat(scalar_first=True)),
                offset_pos=tuple(offset),
                offset_quat=tuple(extra.as_quat(scalar_first=True)),
            ),
            material=gs.materials.VBD.Base(),
        )
        corners = np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * size
        expected = (rotation * extra).apply(corners) + position + rotation.apply(offset)
        actual = tensor_to_array(tissue.init_positions)
        distance = np.linalg.norm(expected[:, None] - actual[None], axis=-1)
        assert (distance.min(axis=1) < 1e-10).all()
