"""Local primitive geometry test: no external assets required."""
import numpy as np

import genesis as gs


def test_ellipsoid_contact_mesh_accuracy(show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)
    axes = np.array((0.1, 0.085, 0.068))
    entity = scene.add_entity(gs.morphs.MJCF(
        file='<mujoco><worldbody><body><freejoint/><geom type="ellipsoid" size="0.1 0.085 0.068"/></body></worldbody></mujoco>',
        decimate=False,
    ))
    mesh = entity.geoms[0].get_trimesh()
    rng = np.random.default_rng(116)
    triangles = mesh.triangles[rng.integers(len(mesh.faces), size=100)]
    weights = rng.dirichlet(np.ones(3), size=100)
    points = (triangles * weights[:, :, None]).sum(axis=1)
    surface = points / np.linalg.norm(points / axes, axis=1)[:, None]
    # Bound chord error before SDF interpolation; the level-2 mesh exceeds 1 mm.
    assert np.max(np.linalg.norm(surface - points, axis=1)) < 0.00015
