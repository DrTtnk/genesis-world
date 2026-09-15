"""The shell's material domain, and the separation of mechanical thickness from contact distance.

Astra's DECISIONS30 asks for two things the engine must not get wrong silently.

`mechanical_thickness_m = 0.0015` and `contact_pair_distance_m = 0.002` are independent quantities: the first
is what the sheet is made of and sets its mass, the second is the distance between the two surfaces at which a
pair activates, measured once for the pair and not per participant. Changing the pair distance must change no
mass, and changing the sheet thickness must change no contact distance. That is asserted here rather than
documented, because the previous docstring claimed the one field did both and the code never did.

The domain rejection list (nonfinite, E <= 0, rho <= 0, nu outside (-1, 0.5), thickness <= 0, contact < 0,
bending < 0) is a mathematical-input gate, not a material-validation gate: A5 stays open, and passing these
says nothing about whether a material is physiological.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array


BASELINE = dict(E=30000.0, nu=0.4, rho=1000.0, thickness=0.0015, bending_stiffness=2e-6)


def _sheet(spacing=0.05, n=4):
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing, indexing="ij")
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=-1)
    idx = np.arange(n * n).reshape(n, n)
    tris = [t for i in range(n - 1) for j in range(n - 1)
            for t in ([idx[i, j], idx[i + 1, j], idx[i + 1, j + 1]], [idx[i, j], idx[i + 1, j + 1], idx[i, j + 1]])]
    return verts, np.array(tris, dtype=np.int64)


def _built(material, contact_distance=None):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        vbd_options=gs.options.VBDOptions(n_iterations=4, floor_height=-1e3),
        show_viewer=False,
    )
    verts, tris = _sheet()
    sheet = scene.add_entity(material=material, morph=gs.morphs.TriMesh(verts=verts, faces=tris))
    if contact_distance is not None:
        # a rule needs both of its groups populated: an empty group leaves the contact system with a
        # zero-length vertex field, which the backend refuses at build
        partner = gs.materials.VBD.Shell(**dict(BASELINE, collision_group=1))
        scene.add_entity(material=partner, morph=gs.morphs.TriMesh(verts=verts + np.array([0.0, 0.0, 0.1]), faces=tris))
        scene.sim.vbd_solver.add_contact_rule(0, 1, stiffness=1e4, friction=0.0, thickness=contact_distance)
    scene.build()
    return scene, sheet


# ---------------------------------------------------------------------------------------------------------
# 1. Mechanical thickness and contact distance are independent
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.required
def test_vertex_mass_follows_mechanical_thickness_only(show_viewer):
    """Astra's numbers: doubling the sheet thickness doubles every vertex mass, and declaring a different
    contact pair distance changes none of them."""
    _, thin = _built(gs.materials.VBD.Shell(**BASELINE))
    thick = dict(BASELINE, thickness=2.0 * BASELINE["thickness"])
    _, doubled = _built(gs.materials.VBD.Shell(**thick))
    _, wide_contact = _built(gs.materials.VBD.Shell(**BASELINE), contact_distance=0.002)
    _, narrow_contact = _built(gs.materials.VBD.Shell(**BASELINE), contact_distance=0.02)

    def masses(entity):
        solver = entity.solver
        return tensor_to_array(solver.verts_info.mass.to_torch(gs.device))[entity.v_start : entity.v_start + entity.n_vertices]

    np.testing.assert_allclose(masses(doubled), 2.0 * masses(thin), rtol=1e-6)
    np.testing.assert_allclose(masses(wide_contact), masses(thin), rtol=1e-6)
    np.testing.assert_allclose(masses(narrow_contact), masses(thin), rtol=1e-6)


@pytest.mark.required
def test_the_contact_pair_distance_is_the_rule_not_the_sheet_thickness(show_viewer):
    """The distance stored for the pair is exactly what `add_contact_rule` was given, whatever the sheet is
    made of: it is the total separation between the two surfaces, not a per-participant margin."""
    for thickness in (0.0015, 0.05):
        scene, _ = _built(gs.materials.VBD.Shell(**dict(BASELINE, thickness=thickness)), contact_distance=0.002)
        stored = tensor_to_array(scene.vbd_solver.contact.rule_thickness.to_torch(gs.device))
        assert stored[0, 1] == pytest.approx(0.002)
        assert scene.vbd_solver.contact.max_thickness == pytest.approx(0.002)


# ---------------------------------------------------------------------------------------------------------
# 2. Mathematical domain, rejected at declaration
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("E", 0.0), ("E", -1.0), ("E", float("nan")), ("E", float("inf")),
        ("rho", 0.0), ("rho", -1.0), ("rho", float("nan")),
        ("nu", 0.5), ("nu", 0.7), ("nu", -1.0), ("nu", float("nan")),
        ("thickness", 0.0), ("thickness", -1e-3), ("thickness", float("inf")),
        ("bending_stiffness", -1e-9), ("bending_stiffness", float("nan")),
    ],
)
def test_a_material_outside_the_mathematical_domain_is_refused(field, value):
    with pytest.raises(Exception):
        gs.materials.VBD.Shell(**dict(BASELINE, **{field: value}))


def test_the_baseline_material_and_a_disabled_bending_are_both_accepted():
    """0 bending is off, not invalid: Astra keeps it tunable and 2e-6 is the trial baseline."""
    gs.materials.VBD.Shell(**BASELINE)
    gs.materials.VBD.Shell(**dict(BASELINE, bending_stiffness=0.0))


@pytest.mark.required
def test_a_contact_rule_outside_its_domain_is_refused(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1),
        vbd_options=gs.options.VBDOptions(n_iterations=4, floor_height=-1e3),
        show_viewer=False,
    )
    verts, tris = _sheet()
    scene.add_entity(material=gs.materials.VBD.Shell(**BASELINE), morph=gs.morphs.TriMesh(verts=verts, faces=tris))
    solver = scene.sim.vbd_solver
    for stiffness, friction, thickness in [(0.0, 0.0, 0.002), (1e4, -0.1, 0.002), (1e4, 0.0, 0.0), (1e4, 0.0, -0.002)]:
        with pytest.raises(gs.GenesisException):
            solver.add_contact_rule(0, 1, stiffness=stiffness, friction=friction, thickness=thickness)
