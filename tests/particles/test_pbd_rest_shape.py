"""Curved cloth must be able to start without bending preload."""

import numpy as np
import pytest
import trimesh

import genesis as gs


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("reference", ["flat", "mesh"])
def test_curved_cloth_bending_reference(reference, n_envs, tmp_path, show_viewer):
    grid = np.linspace(-0.04, 0.04, 7)
    vertices = np.array([[x, y, 8 * y**2] for x in grid for y in grid])
    faces = []
    for i in range(6):
        for j in range(6):
            a = i * 7 + j
            faces.extend(((a, a + 7, a + 8), (a, a + 8, a + 1)))
    path = tmp_path / "curved.obj"
    trimesh.Trimesh(vertices, faces, process=False).export(path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
        pbd_options=gs.options.PBDOptions(particle_size=0.01, lower_bound=(-0.1, -0.1, -0.1), upper_bound=(0.1, 0.1, 0.1)),
        show_viewer=show_viewer,
    )
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=str(path), euler=(13, 29, 7)),
        material=gs.materials.PBD.Cloth(bending_reference=reference, air_resistance=0),
    )
    scene.build(n_envs=n_envs)
    rest = cloth.get_particles_pos().clone()
    # Isolate the bending projection from contact and other constraints.
    scene.pbd_solver._kernel_solve_bending(0)
    drift = float((cloth.get_particles_pos() - rest).norm(dim=-1).max())
    if reference == "flat":
        assert drift > 1e-5, drift
    else:
        assert drift < 1e-6, drift
        for _ in range(20):
            scene.step()
        assert float((cloth.get_particles_pos() - rest).norm(dim=-1).max()) < 1e-5
        scene.reset()
        scene.step()
        assert float((cloth.get_particles_pos() - rest).norm(dim=-1).max()) < 1e-6


@pytest.mark.required
def test_bending_reference_default_and_strict_input():
    assert gs.materials.PBD.Cloth().bending_reference == "flat"
    for reference in (True, None, "relaxed", 0):
        with pytest.raises(gs.GenesisException, match="bending_reference"):
            gs.materials.PBD.Cloth(bending_reference=reference)
