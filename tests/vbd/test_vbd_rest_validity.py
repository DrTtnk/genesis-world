"""A rest mesh the solver cannot integrate must be refused at build, not diverge at run time.

A tetrahedron whose rest volume is negative has an inverted rest frame, so `B_rest` is the inverse of a
left-handed matrix and every deformation gradient built from it is reflected. Nothing downstream notices: the
energy is finite, the forces are finite, and the simulation simply blows up somewhere later. That cost me a
confused run on a hand-written connector mesh whose three tetrahedra were all wound the wrong way, and the
symptom was a body accelerating upward, which looks like an engine defect rather than a bad input.

The rule these follow is the project's: strict over lenient, and fail loudly at the point the bad value enters.
"""

import numpy as np
import pytest

import genesis as gs


def _prism(flip):
    """Two triangular faces joined into three tetrahedra: the standard connector shape of a ligament volume."""
    r = 0.0015
    top, bottom = 0.09, 0.086
    verts = np.array([
        [r, 0.0, top], [-0.5 * r, 0.866 * r, top], [-0.5 * r, -0.866 * r, top],
        [r, 0.0, bottom], [-0.5 * r, 0.866 * r, bottom], [-0.5 * r, -0.866 * r, bottom],
    ])
    elems = np.array([[0, 2, 1, 3], [1, 3, 2, 4], [2, 4, 3, 5]])
    if flip:
        elems = elems[:, [1, 0, 2, 3]]
    faces = np.array([[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]])
    return verts, elems, faces


def _scene(verts, elems, faces):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, -9.81)),
        vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1e3),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.TetMesh(verts=verts, elems=elems, faces=faces),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4),
    )
    return scene


@pytest.mark.required
def test_a_correctly_wound_rest_mesh_is_accepted(show_viewer):
    verts, elems, faces = _prism(flip=False)
    _scene(verts, elems, faces).build()


def test_an_inverted_rest_tetrahedron_is_refused():
    """Every tetrahedron wound the wrong way: the whole mesh is inverted at rest."""
    verts, elems, faces = _prism(flip=True)
    with pytest.raises(gs.GenesisException, match="negative rest volume"):
        _scene(verts, elems, faces).build()


def test_one_inverted_tetrahedron_among_good_ones_is_refused_and_named():
    """The likelier authoring mistake is a few bad tetrahedra in an otherwise sound mesh, so the message has to
    say how many and give one index to look at."""
    verts, elems, faces = _prism(flip=False)
    elems = elems.copy()
    elems[1] = elems[1][[1, 0, 2, 3]]
    with pytest.raises(gs.GenesisException, match="1 of 3"):
        _scene(verts, elems, faces).build()


def test_a_degenerate_flat_tetrahedron_is_refused():
    """Zero rest volume inverts nothing but makes `B_rest` singular, so it is just as unusable."""
    verts, elems, faces = _prism(flip=False)
    verts = verts.copy()
    verts[3:] = verts[:3]  # collapse the two footprints onto each other
    with pytest.raises(gs.GenesisException, match="negative rest volume|degenerate"):
        _scene(verts, elems, faces).build()
