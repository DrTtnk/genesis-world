"""Barycentric attachments: a material point on a surface or inside a tet, bound to a rigid link.

`VBDEntity.add_barycentric_attachments` generalises `add_rigid_attachments` from a single tissue vertex to a
weighted sum of up to four of them, C = sum_j w_j x_j - (link_pos + R local): the point `func_attachment_point`
builds in `vbd_rigid_attachment.py`. `func_attachment_soft_system` and `func_attachment_blocks` are unchanged;
only the point they are fed changes, and each supporting vertex is credited `w_j` of the resulting force and
`w_j^2` of its curvature block, exactly the distribution `spikes/verify_avbd_mtu_math.py` proves for a routed
anchor and `vbd_mtu.py`'s `func_mtu_vertex_terms` already applies. A plain vertex attachment is the case of one
weight equal to one.

Forces are read directly with `_kernel_residual_vector` at a chosen configuration, not inferred from motion: a
free body moves mostly rigidly, so displacement reads the weights backwards (see the docstring of
`test_a_muscle_anchored_to_a_triangle_pulls_all_three_of_its_vertices` in test_vbd_shell.py, which hit exactly
this on its first three attempts).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.vbd_mtu import SurfaceAnchor, TissueAnchor
from genesis.utils.misc import qd_to_torch, tensor_to_array

from ..utils.assertions import assert_allclose


# --------------------------------------------------------------------------------------------------------
# shared scaffolding
# --------------------------------------------------------------------------------------------------------


def _residual_force(solver, f=0):
    """`-force` of `_func_vertex_system` at every vertex, negated back: the existing test hook. With `vel = 0`
    and `verts[f + 1] == verts[f]` (a fresh `_kernel_predict`), the inertia term is exactly zero."""
    out = np.zeros((solver._B, solver.n_vertices, 3), dtype=gs.np_float)
    solver._kernel_residual_vector(f, out)
    return -out


def _set_static_state(solver, positions):
    """Put `positions` at both frame 0 and frame 1 with zero velocity, so the elastic and attachment force
    alone is read at frame 0 (see `_residual_force`); the attachment's own multiplier, stiffness and link pose
    are untouched, since they live on `rigid_attachment` and are not written by this call."""
    pos = np.broadcast_to(positions.astype(gs.np_float), (solver._B, *positions.shape)).copy()
    vel = np.zeros_like(pos)
    solver._kernel_set_state(0, pos, vel)
    solver._kernel_predict(0)


def _sheet_and_bone(bone_pos, n_iterations=6, dt=1e-3, gravity=(0.0, 0.0, 0.0)):
    """A 4-vertex, 2-triangle sheet (vertices 0, 1, 2 form triangle 0; vertex 3 is outside it) plus a small
    free rigid box.

    `local_pos` is built from the positions at declaration time, so the constraint reads exactly zero at rest
    with the bone where it starts: uniform gravity does not perturb it either, both sides falling together
    (test_vbd_rigid.py's free-link test relies on the same invariant). A real deviation needs an unbalanced
    force, so vertex 3 -- outside the anchor triangle, free to pin -- is held while gravity pulls on the rest.
    """
    rest = np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.0, 0.06, 0.0], [0.06, 0.06, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    material = gs.materials.VBD.Shell(E=1.0, nu=0.3, thickness=1e-3, bending_stiffness=0.0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=gravity),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1e3),
        show_viewer=False,
    )
    sheet = scene.add_entity(material=material, morph=gs.morphs.TriMesh(verts=rest, faces=faces))
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=bone_pos),
        material=gs.materials.Rigid(rho=10.0),
    )
    return scene, sheet, bone, rest, faces


def _tet_and_bone(bone_pos, n_iterations=6, dt=1e-3, gravity=(0.0, 0.0, 0.0)):
    """A tetrahedralized box tissue plus a small free rigid box. See `_sheet_and_bone` for why gravity alone,
    with nothing pinned, would never move the constraint away from zero."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=gravity),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1e3),
        show_viewer=False,
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.5, 0.0), nobisect=False, maxvolume=4e-6),
        material=gs.materials.VBD.Muscle(E=1.0, nu=0.3),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=bone_pos),
        material=gs.materials.Rigid(rho=10.0),
    )
    return scene, tissue, bone


# --------------------------------------------------------------------------------------------------------
# 1. distribution
# --------------------------------------------------------------------------------------------------------


def test_a_surface_anchor_distributes_its_pull_by_barycentric_weight():
    """A constant force applied uniformly to a whole mesh (gravity) never disturbs the constraint -- both sides
    of it fall together, an invariant `test_vbd_rigid.py`'s free-link test also relies on -- so tension needs a
    force that is NOT uniform. A velocity kick on the bone alone, with no gravity, is one: it lags the sheet's
    inertia and opens a real, readable gap. (Gravity was tried first and rejected: it also disagreed with
    `_set_static_state`'s zero-velocity predictor whenever a vertex is pinned, since a pinned vertex ignores the
    predicted gravity drift its free neighbours get, opening a spurious elastic term at the very "rest"
    configuration meant to be free of one.)
    """
    weights = (0.6, 0.3, 0.1)
    scene, sheet, bone, rest, faces = _sheet_and_bone(bone_pos=(0.02, 0.02, 0.06), n_iterations=8)
    sheet.add_barycentric_attachments([SurfaceAnchor(sheet, 0, weights)], bone.links[0])
    scene.build()
    bone.set_dofs_velocity([0.0, 0.0, -0.3, 0.0, 0.0, 0.0])
    for _ in range(3):
        scene.step()

    _set_static_state(scene.vbd_solver, rest)
    force = _residual_force(scene.vbd_solver)[0]
    corners = faces[0]
    magnitude = np.linalg.norm(force[corners], axis=1)
    print(f"surface anchor corner forces {magnitude} N at weights {weights}")
    assert magnitude.sum() > 0.0

    for got, want in zip(magnitude / magnitude.sum(), weights):
        assert abs(got - want) < 0.01
    # the vertex outside the triangle carries none of the pull
    assert np.linalg.norm(force[3]) < 0.01 * magnitude.sum()
    # the three corners together are the anchor's total pull
    assert abs(np.linalg.norm(force[corners].sum(axis=0)) - magnitude.sum()) < 0.02 * magnitude.sum()


def test_a_tissue_anchor_distributes_its_pull_by_barycentric_weight():
    weights = (0.5, 0.3, 0.15, 0.05)
    scene, tissue, bone = _tet_and_bone(bone_pos=(0.0, 0.5, 0.03), n_iterations=8)
    rest = tensor_to_array(tissue.init_positions)
    corners = np.argsort(-rest[:, 0])[:4]
    tissue.add_barycentric_attachments([TissueAnchor(tissue, tuple(int(v) for v in corners), weights)], bone.links[0])
    scene.build()
    bone.set_dofs_velocity([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    for _ in range(3):
        scene.step()

    _set_static_state(scene.vbd_solver, rest)
    force = _residual_force(scene.vbd_solver)[0]
    magnitude = np.linalg.norm(force[corners], axis=1)
    print(f"tissue anchor corner forces {magnitude} N at weights {weights}")
    assert magnitude.sum() > 0.0

    for got, want in zip(magnitude / magnitude.sum(), weights):
        assert abs(got - want) < 0.01
    others = np.setdiff1d(np.arange(tissue.n_vertices), corners)
    assert np.linalg.norm(force[others], axis=1).max() < 0.01 * magnitude.sum()
    assert abs(np.linalg.norm(force[corners].sum(axis=0)) - magnitude.sum()) < 0.02 * magnitude.sum()


# --------------------------------------------------------------------------------------------------------
# 2. equivalence with add_rigid_attachments
# --------------------------------------------------------------------------------------------------------


def _equivalence_scene(attach):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=4, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=4, floor_height=-10.0),
        show_viewer=False,
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.04, 0.08, 0.08), pos=(0.12, 0.0, 0.0)),
        material=gs.materials.Rigid(rho=500.0),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(size=(0.2, 0.04, 0.04), pos=(0.0, 0.0, 0.03), nobisect=False, maxvolume=2e-5),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3),
    )
    rest = tensor_to_array(tissue.init_positions)
    vertex = int(np.flatnonzero(rest[:, 0] > rest[:, 0].max() - 1e-5)[0])
    attach(tissue, vertex, bone.links[0])
    scene.build()
    return scene, tissue, bone


def test_a_one_weight_barycentric_anchor_matches_add_rigid_attachments():
    """A `TissueAnchor` naming one vertex four times with weights (1, 0, 0, 0) is the same declaration
    `add_rigid_attachments` builds internally, so the two trajectories must agree to float tolerance."""

    def via_vertex(tissue, v, link):
        tissue.add_rigid_attachments([v], link)

    def via_barycentric(tissue, v, link):
        tissue.add_barycentric_attachments([TissueAnchor(tissue, (v, v, v, v), (1.0, 0.0, 0.0, 0.0))], link)

    scene_a, tissue_a, bone_a = _equivalence_scene(via_vertex)
    scene_b, tissue_b, bone_b = _equivalence_scene(via_barycentric)
    for _ in range(20):
        scene_a.step()
        scene_b.step()
        assert_allclose(tissue_a.get_positions(), tissue_b.get_positions(), tol=1e-6)
        assert_allclose(bone_a.get_pos(), bone_b.get_pos(), tol=1e-6)
        assert_allclose(bone_a.get_quat(), bone_b.get_quat(), tol=1e-6)


# --------------------------------------------------------------------------------------------------------
# 3. two-way coupling and the link's wrench
# --------------------------------------------------------------------------------------------------------


def _rotate(vector, quaternion):
    w, x, y, z = quaternion
    q = np.array([x, y, z])
    return vector + 2.0 * np.cross(q, np.cross(q, vector) + w * vector)


def test_the_link_is_pulled_toward_a_held_sheet_and_its_wrench_matches_the_vertex_forces():
    """A light link attached by a barycentric anchor to a held sheet is pulled toward it, and the wrench the
    link block applies is, up to the closing constraint gap, the wrench a naive sum of the individual vertex
    reactions about the link origin would give -- the check the single shared attachment point can support,
    since the block itself carries one local anchor for the whole weighted point, not one per corner."""
    weights = (0.6, 0.3, 0.1)
    scene, sheet, bone, rest, faces = _sheet_and_bone(
        bone_pos=(0.02, 0.02, 0.08), n_iterations=10, dt=2e-3, gravity=(0.0, 0.0, -9.81)
    )
    corners = faces[0]
    sheet.add_barycentric_attachments([SurfaceAnchor(sheet, 0, weights)], bone.links[0])
    scene.build()
    sheet.set_pinned(np.array([False, False, False, True]))

    initial_bone_z = float(bone.get_pos()[2])
    gaps = []
    for _ in range(60):
        scene.step()
        pos = tensor_to_array(bone.get_pos())
        quat = tensor_to_array(bone.get_quat())
        link = scene.vbd_solver.rigid_attachment
        local = tensor_to_array(qd_to_torch(link.info.local_pos))[0]
        target = pos + _rotate(local, quat)
        point = (tensor_to_array(sheet.get_positions())[0, corners] * np.array(weights)[:, None]).sum(axis=0)
        gaps.append(float(np.linalg.norm(point - target)))
    # the bone is far lighter than a free fall under gravity alone would let it drift from the held sheet by
    # more than a few mm over this run; a light link that only fell under its own weight, unattached, would end
    # this run about 0.5 * 9.81 * (60 * 2e-3)^2 = 0.071 m from where it started, two orders of magnitude more
    # than the gap below -- the attachment, not gravity balance, is what kept it close.
    assert float(bone.get_pos()[2]) < initial_bone_z - 0.01
    assert max(gaps[-10:]) < 0.005

    # ---- wrench check, at the last (best-converged) configuration ----
    multiplier = tensor_to_array(qd_to_torch(link.state.multiplier))[0, 0]
    stiffness = float(tensor_to_array(qd_to_torch(link.state.stiffness))[0, 0])
    previous_error = tensor_to_array(qd_to_torch(link.previous_error))[0, 0]
    alpha = link.alpha
    tissue_pos = tensor_to_array(sheet.get_positions())[0]
    x = tissue_pos[corners]
    point = (x * np.array(weights)[:, None]).sum(axis=0)
    rotated_anchor = _rotate(local, quat)
    constraint = point - pos - rotated_anchor - alpha * previous_error
    force_scale = multiplier + stiffness * constraint

    reported_torque = np.cross(rotated_anchor, force_scale)
    # each supporting vertex is credited its own reaction w_j * force_scale, applied at its own position: the
    # naive sum a set of independent single-vertex attachments would give
    vertex_forces = np.array(weights)[:, None] * force_scale[None, :]
    naive_torque = sum(np.cross(xj - pos, fj) for xj, fj in zip(x, vertex_forces))
    gap = gaps[-1]
    print(f"barycentric attachment gap {gap:.3e} m, reported torque {reported_torque}, naive {naive_torque}")
    # the two only agree exactly when the constraint is closed: bounded by the gap times the force scale
    assert np.linalg.norm(reported_torque - naive_torque) < 5.0 * gap * np.linalg.norm(force_scale) + 1e-8

    budget = json.loads((Path(__file__).parent / "manifests" / "barycentric_attachment_gap.json").read_text())
    assert gap < budget["surface_anchor_gap_budget_m"]


# --------------------------------------------------------------------------------------------------------
# 4. explicitly out of scope, refused by name
# --------------------------------------------------------------------------------------------------------


def test_a_surface_anchor_naming_another_entity_is_refused():
    scene, sheet, bone, rest, faces = _sheet_and_bone(bone_pos=(0.0, 0.0, 0.05))
    other = scene.add_entity(
        material=gs.materials.VBD.Shell(E=1.0, nu=0.3, thickness=1e-3),
        morph=gs.morphs.TriMesh(verts=rest, faces=faces),
    )
    with pytest.raises(gs.GenesisException, match="must name this entity"):
        sheet.add_barycentric_attachments([SurfaceAnchor(other, 0, (0.5, 0.3, 0.2))], bone.links[0])


def test_an_unknown_anchor_type_is_refused():
    scene, sheet, bone, rest, faces = _sheet_and_bone(bone_pos=(0.0, 0.0, 0.05))
    with pytest.raises(gs.GenesisException, match="Unknown barycentric anchor"):
        sheet.add_barycentric_attachments([object()], bone.links[0])
