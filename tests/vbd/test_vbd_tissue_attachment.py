"""Tissue-to-tissue point attachments: a material point of one VBD entity bound to a material point of another.

`VBDSolver.add_tissue_attachment(anchor_a, anchor_b)` takes two `TissueAnchor` or `SurfaceAnchor` (of the same or
different entities) and holds C = P_a - P_b - d0 = 0, with P the weighted vertex sums and d0 their rest offset,
by the augmented Lagrangian the rigid attachments use (`vbd_rigid_attachment.py`): every supporting vertex is
credited w_j of the force and w_j^2 of the curvature block, side b with the opposite sign. The head model needs
it for its 80 tendon cuffs, whose far end binds a muscle matrix rather than a bone.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.vbd_mtu import SurfaceAnchor, TissueAnchor
from genesis.utils.misc import tensor_to_array

SIDE = 0.02
GAP = 0.01


def _two_boxes(gravity=(0.0, 0.0, -9.81), n_iterations=8, dt=1e-3, requires_grad=False, E=2e4):
    """Box A hangs from pinned top vertices; box B floats `GAP` below it, unsupported until attached."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=gravity, requires_grad=requires_grad),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1e3),
        show_viewer=False,
    )
    a = scene.add_entity(
        morph=gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(0.0, 0.0, 0.5), nobisect=False, maxvolume=5e-8),
        material=gs.materials.VBD.Muscle(E=E, nu=0.3),
    )
    b = scene.add_entity(
        morph=gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(0.0, 0.0, 0.5 - SIDE - GAP), nobisect=False, maxvolume=5e-8),
        material=gs.materials.VBD.Muscle(E=E, nu=0.3),
    )
    return scene, a, b


def _hang(a):
    """Pin box A's top face; pinning is a post-build call."""
    rest_a = tensor_to_array(a.init_positions)
    a.set_pinned(rest_a[:, 2] > rest_a[:, 2].max() - 1e-6)


def _extreme_tet(entity, direction):
    """The tet whose centroid is furthest along `direction`, as (vertex tuple, centroid weights)."""
    rest = tensor_to_array(entity.init_positions)
    tets = np.asarray(entity.elems)
    centroids = rest[tets].mean(1)
    tet = tets[int(np.argmax(centroids @ np.asarray(direction, dtype=float)))]
    return tuple(int(v) for v in tet)


def _centroid_gap(a, b, tet_a, tet_b, offset):
    pa = tensor_to_array(a.get_positions())[0][list(tet_a)].mean(0)
    pb = tensor_to_array(b.get_positions())[0][list(tet_b)].mean(0)
    return np.linalg.norm(pa - pb - offset)


def test_a_box_bound_to_a_hanging_box_is_caught_instead_of_falling():
    scene, a, b = _two_boxes()
    tet_a, tet_b = _extreme_tet(a, (0, 0, -1)), _extreme_tet(b, (0, 0, 1))
    w = (0.25, 0.25, 0.25, 0.25)
    scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), TissueAnchor(b, tet_b, w))
    scene.build()
    _hang(a)
    rest_b = tensor_to_array(b.init_positions)
    offset = tensor_to_array(a.init_positions)[list(tet_a)].mean(0) - rest_b[list(tet_b)].mean(0)
    for _ in range(200):
        scene.step()
    now_b = tensor_to_array(b.get_positions())[0]
    assert np.isfinite(now_b).all()
    drop = float((rest_b - now_b)[:, 2].mean())
    free_fall = 0.5 * 9.81 * (200 * 1e-3) ** 2
    gap = _centroid_gap(a, b, tet_a, tet_b, offset)
    print(f"box B dropped {1000 * drop:.3f} mm against {1000 * free_fall:.1f} mm free fall; anchor gap {1e6 * gap:.2f} um")
    assert drop < 0.1 * free_fall, "the attachment must carry the lower box"
    assert drop > 0.0, "a soft body hanging from a point sags"
    assert gap < 2e-4, "the two anchors must stay together within the constraint tolerance"


@pytest.mark.parametrize("n_iterations", [8, 32])
def test_a_rest_offset_between_the_anchors_is_stress_free(n_iterations):
    """The anchors bind where they are: nothing moves without gravity even though the points do not coincide."""
    scene, a, b = _two_boxes(gravity=(0.0, 0.0, 0.0), n_iterations=n_iterations)
    tet_a, tet_b = _extreme_tet(a, (0, 0, -1)), _extreme_tet(b, (0, 0, 1))
    w = (0.25, 0.25, 0.25, 0.25)
    scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), TissueAnchor(b, tet_b, w))
    scene.build()
    _hang(a)
    rest = np.vstack([tensor_to_array(a.init_positions), tensor_to_array(b.init_positions)])
    for _ in range(20):
        scene.step()
    now = np.vstack([tensor_to_array(a.get_positions())[0], tensor_to_array(b.get_positions())[0]])
    assert np.abs(now - rest).max() < 1e-9


def test_the_pull_is_shared_by_weight_and_is_equal_and_opposite():
    """Read the attachment force with the residual hook at a static configuration, as the barycentric tests do."""
    weights_a, weights_b = (0.5, 0.3, 0.15, 0.05), (0.25, 0.25, 0.25, 0.25)
    scene, a, b = _two_boxes()
    tet_a, tet_b = _extreme_tet(a, (0, 0, -1)), _extreme_tet(b, (0, 0, 1))
    scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, weights_a), TissueAnchor(b, tet_b, weights_b))
    scene.build()
    _hang(a)
    for _ in range(50):
        scene.step()
    solver = scene.vbd_solver
    rest = np.vstack([tensor_to_array(a.init_positions), tensor_to_array(b.init_positions)])
    pos = np.broadcast_to(rest.astype(gs.np_float), (solver._B, *rest.shape)).copy()
    solver._kernel_set_state(0, pos, np.zeros_like(pos))
    solver._kernel_predict(0)
    out = np.zeros((solver._B, solver.n_vertices, 3), dtype=gs.np_float)
    solver._kernel_residual_vector(0, out)
    force = -out[0]
    fa = force[[a.v_start + v for v in tet_a]]
    fb = force[[b.v_start + v for v in tet_b]]
    total = np.linalg.norm(fa.sum(0))
    assert total > 0.0
    np.testing.assert_allclose(fa.sum(0) + fb.sum(0), 0.0, atol=1e-6 * total)
    for got, want in zip(np.linalg.norm(fa, axis=1) / total, weights_a):
        assert abs(got - want) < 0.01
    for got, want in zip(np.linalg.norm(fb, axis=1) / total, weights_b):
        assert abs(got - want) < 0.01


def test_a_surface_anchor_binds_a_tet_box_to_a_shell_point():
    """A shell sheet hung from a point of the lower box: the surface anchor resolves like a tet anchor with a
    fourth weight of zero, so the sheet is carried and nothing is NaN."""
    scene, a, b = _two_boxes()
    rest = np.array([[0.0, 0.0, 0.4], [0.06, 0.0, 0.4], [0.0, 0.06, 0.4], [0.06, 0.06, 0.4]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    sheet = scene.add_entity(
        material=gs.materials.VBD.Shell(E=1e3, nu=0.3, thickness=1e-3, bending_stiffness=0.0),
        morph=gs.morphs.TriMesh(verts=rest, faces=faces),
    )
    tet_a, tet_b = _extreme_tet(a, (0, 0, -1)), _extreme_tet(b, (0, 0, 1))
    w = (0.25, 0.25, 0.25, 0.25)
    scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), TissueAnchor(b, tet_b, w))
    scene.vbd_solver.add_tissue_attachment(TissueAnchor(b, _extreme_tet(b, (0, 0, -1)), w), SurfaceAnchor(sheet, 0, (0.6, 0.3, 0.1)))
    scene.build()
    _hang(a)
    for _ in range(100):
        scene.step()
    now = tensor_to_array(sheet.get_positions())[0]
    assert np.isfinite(now).all()
    # a sheet hung from one corner pivots, so its mean falls; what must hold is the corner on the box's point
    tet_low = _extreme_tet(b, (0, 0, -1))
    box_point = tensor_to_array(b.get_positions())[0][list(tet_low)].mean(0)
    sheet_point = np.array([0.6, 0.3, 0.1]) @ now[faces[0]]
    offset = tensor_to_array(b.init_positions)[list(tet_low)].mean(0) - np.array([0.6, 0.3, 0.1]) @ rest[faces[0]]
    assert np.linalg.norm(box_point - sheet_point - offset) < 2e-4
    assert (rest - now)[:, 2].mean() < 0.5 * 9.81 * 0.1**2, "the sheet must not be in free fall"


def test_declarations_that_cannot_mean_anything_are_refused():
    scene, a, b = _two_boxes()
    tet_a, tet_b = _extreme_tet(a, (0, 0, -1)), _extreme_tet(b, (0, 0, 1))
    w = (0.25, 0.25, 0.25, 0.25)
    with pytest.raises(gs.GenesisException, match="sum to"):
        scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, (0.5, 0.5, 0.5, 0.5)), TissueAnchor(b, tet_b, w))
    with pytest.raises(gs.GenesisException, match="same material point"):
        scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), TissueAnchor(a, tet_a, w))
    with pytest.raises(gs.GenesisException, match="Unknown"):
        scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), "not an anchor")
    scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), TissueAnchor(b, tet_b, w))
    scene.build()
    with pytest.raises(gs.GenesisException, match="before scene.build"):
        scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), TissueAnchor(b, tet_b, w))


def test_pinning_an_attached_vertex_afterwards_is_refused():
    scene, a, b = _two_boxes()
    tet_a, tet_b = _extreme_tet(a, (0, 0, -1)), _extreme_tet(b, (0, 0, 1))
    w = (0.25, 0.25, 0.25, 0.25)
    scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), TissueAnchor(b, tet_b, w))
    pinned = np.zeros(b.n_vertices, dtype=bool)
    pinned[list(tet_b)] = True
    with pytest.raises(gs.GenesisException, match="pinned"):
        b.set_pinned(pinned)


def test_no_adjoint_yet_so_requires_grad_is_refused_at_build():
    scene, a, b = _two_boxes(requires_grad=True)
    tet_a, tet_b = _extreme_tet(a, (0, 0, -1)), _extreme_tet(b, (0, 0, 1))
    w = (0.25, 0.25, 0.25, 0.25)
    scene.vbd_solver.add_tissue_attachment(TissueAnchor(a, tet_a, w), TissueAnchor(b, tet_b, w))
    with pytest.raises(gs.GenesisException, match="requires_grad"):
        scene.build()
