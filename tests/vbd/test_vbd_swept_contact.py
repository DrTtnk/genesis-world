"""Swept candidate search: the contact set covers the path a vertex takes, not the point it ends at.

`kernel_begin_contact` used to build the hash grid and the candidate pairs from the predicted end-of-substep
positions alone. Two consequences followed, and both are the subject of this file. A pair that passed through
another surface between the two endpoints was never a candidate, so the penalty could not hold it and continuous
collision detection had nothing to sweep. And the validity guard had to hold *every* contact vertex to one global
margin per substep, because any travel beyond the searched neighbourhood could hide such a pair -- which refused
the substep of a body that was nowhere near anything it could touch, purely for moving.

With the search swept from the position at the start of the substep to the prediction, the covered region grows
with each vertex's own travel, and the only motion the search does not account for is the deviation of the solve
from that swept path. That deviation is what the guard now measures, so the margin is sized by the contact
response rather than by the speed of the scene.
"""

import types

import numpy as np
import pytest
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.vbd_contact import (
    EDGE_EDGE,
    POINT_TRIANGLE,
    func_cell_hash,
    func_is_canonical_cell,
    func_is_own_cell,
    func_pair_distance,
    func_sweep_bound,
    func_swept_lower_bound,
)
from genesis.utils.misc import qd_to_torch, tensor_to_array

N_CASES = 64


@qd.kernel
def kernel_swept_lower_bound_reference(starts: qd.template(), ends: qd.template(), lower: qd.template()):
    """The bound the candidate search accepts on, for both pair kinds, over a linear sweep."""
    for i in range(N_CASES):
        for kind in qd.static((POINT_TRIANGLE, EDGE_EDGE)):
            lower[i, kind] = func_swept_lower_bound(
                func_pair_distance(starts[i, 0], starts[i, 1], starts[i, 2], starts[i, 3], kind),
                func_pair_distance(ends[i, 0], ends[i, 1], ends[i, 2], ends[i, 3], kind),
                func_sweep_bound(
                    ends[i, 0] - starts[i, 0],
                    ends[i, 1] - starts[i, 1],
                    ends[i, 2] - starts[i, 2],
                    ends[i, 3] - starts[i, 3],
                    kind,
                ),
            )


def test_the_swept_bound_never_exceeds_the_distance_anywhere_on_the_sweep():
    """The search accepts a pair when this bound is inside the layer, so the bound must never sit above the true
    distance at any time of the sweep, or a pair that really does touch would be dropped. It must also reject
    enough to be worth searching on, which these sweeps measure: they displace each of the four points by about
    as much as the points are apart, which is far harsher than a substep of a simulation, and even there more
    than half of the pairs that stay 5 mm apart are still rejected at a 1 mm layer.
    """
    rng = np.random.default_rng(7)
    starts = np.zeros((N_CASES, 4, 3))
    ends = np.zeros((N_CASES, 4, 3))
    for i in range(N_CASES):
        starts[i] = rng.normal(size=(4, 3)) * 0.01
        ends[i] = starts[i] + rng.normal(size=(4, 3)) * 0.01
    start_field = qd.Vector.field(3, dtype=gs.qd_float, shape=(N_CASES, 4))
    end_field = qd.Vector.field(3, dtype=gs.qd_float, shape=(N_CASES, 4))
    lower_field = qd.field(dtype=gs.qd_float, shape=(N_CASES, 2))
    start_field.from_numpy(starts.astype(gs.np_float))
    end_field.from_numpy(ends.astype(gs.np_float))
    kernel_swept_lower_bound_reference(start_field, end_field, lower_field)
    lower = qd_to_torch(lower_field).cpu().numpy()

    @qd.kernel
    def kernel_sampled_distance(
        starts: qd.template(), ends: qd.template(), out: qd.template(), samples: int
    ):
        for i in range(N_CASES):
            for kind in qd.static((POINT_TRIANGLE, EDGE_EDGE)):
                out[i, kind] = 1e9
                for j in range(samples):
                    t = j / (samples - 1.0)
                    out[i, kind] = qd.min(
                        out[i, kind],
                        func_pair_distance(
                            starts[i, 0] + t * (ends[i, 0] - starts[i, 0]),
                            starts[i, 1] + t * (ends[i, 1] - starts[i, 1]),
                            starts[i, 2] + t * (ends[i, 2] - starts[i, 2]),
                            starts[i, 3] + t * (ends[i, 3] - starts[i, 3]),
                            kind,
                        ),
                    )

    sampled_field = qd.field(dtype=gs.qd_float, shape=(N_CASES, 2))
    kernel_sampled_distance(start_field, end_field, sampled_field, 801)
    sampled = qd_to_torch(sampled_field).cpu().numpy()
    assert (lower <= sampled + 1e-12).all(), "the bound must be below the distance everywhere on the sweep"
    far = sampled > 5e-3
    assert far.sum() > 16, "the sweeps must include pairs that stay apart, or the next line proves nothing"
    assert (lower[far] >= 1e-3).mean() > 0.4, "and the bound must reject most of them at a 1 mm layer"

    # a pair carried along without moving relative to itself has no slack at all: the mean displacement is the
    # whole displacement, so the Lipschitz constant is zero and the bound is the distance
    carried = starts + np.array([0.0, 0.0, -0.016])
    carried_field = qd.Vector.field(3, dtype=gs.qd_float, shape=(N_CASES, 4))
    carried_field.from_numpy(carried.astype(gs.np_float))
    kernel_swept_lower_bound_reference(start_field, carried_field, lower_field)
    kernel_sampled_distance(start_field, carried_field, sampled_field, 65)
    assert np.allclose(qd_to_torch(lower_field).cpu().numpy(), qd_to_torch(sampled_field).cpu().numpy())


def _falling_tissue_scene(margin=None, contact_ccd=False):
    """A 40 mm tissue cube dropped 0.18 m onto a fixed plate, with a 0.2 mm contact layer. It reaches 1.9 m/s,
    which is 3.7 mm a substep, nineteen times the layer, while touching nothing at all on the way down."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-10.0,
            contact_margin=margin,
            contact_ccd=contact_ccd,
            raise_on_env_failure=False,
        ),
        show_viewer=False,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(size=(0.4, 0.4, 0.02), pos=(0.0, 0.0, -0.01), fixed=True),
        material=gs.materials.Rigid(),
    )
    falling = scene.add_entity(
        morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.2), nobisect=False, maxvolume=1e-5),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, collision_group=1),
    )
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.5, thickness=2e-4)
    scene.build()
    return scene, falling


def test_travel_alone_no_longer_refuses_a_substep():
    """The head36 failure in miniature: 0.2 mm of margin, 3.7 mm of free fall a substep, and no pair anywhere
    near the mover. The travel is reported and the deviation from the swept path stays at zero, because a free
    fall is exactly what the prediction expected."""
    scene, falling = _falling_tissue_scene()
    assert scene.vbd_solver.contact.margin == pytest.approx(2e-4)
    worst_motion, worst_deviation = 0.0, 0.0
    for _ in range(80):
        scene.step()
        diagnostics = scene.vbd_solver.contact_diagnostics()
        worst_motion = max(worst_motion, float(diagnostics.max_tissue_motion[0]))
        worst_deviation = max(worst_deviation, float(diagnostics.max_tissue_deviation[0]))
    status = scene.vbd_solver.env_status()
    lowest = float(tensor_to_array(falling.get_positions())[0][:, 2].min())
    print(f"free fall: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, "
          f"lowest {1000 * lowest:.3f} mm, motion {1e6 * worst_motion:.1f} um, "
          f"deviation {1e6 * worst_deviation:.1f} um")
    assert not bool(status.is_failed[0]), f"the travel alone must not refuse a substep (errno {int(status.errno[0])})"
    assert lowest > 0.02, "the cube must still be in the air, so this is about travel and not about impact"
    assert worst_motion > 10.0 * scene.vbd_solver.contact.margin, "and it must really have outrun the old bound"
    assert worst_deviation == pytest.approx(0.0, abs=1e-9), "a free fall departs from its prediction by nothing"


def test_a_fast_cube_lands_and_rests_in_its_contact_layer():
    """The same fall carried through the impact. The margin no longer has to cover the 3.7 mm of approach, only
    the response to it, which measures 0.33 mm here."""
    scene, falling = _falling_tissue_scene(margin=1e-3, contact_ccd=True)
    for _ in range(140):
        scene.step()
    status = scene.vbd_solver.env_status()
    diagnostics = scene.vbd_solver.contact_diagnostics()
    lowest = float(tensor_to_array(falling.get_positions())[0][:, 2].min())
    print(f"landing: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, "
          f"lowest {1000 * lowest:.3f} mm, min_toi {float(diagnostics.min_toi[0]):.4f}")
    assert not bool(status.is_failed[0]), f"the landing must be taken (errno {int(status.errno[0])})"
    assert lowest > 1e-4, "the cube must rest on its contact layer, not inside the plate"
    assert lowest < 1e-3, "and it must have landed rather than been held in the air"


def _fast_block_scene(contact_ccd, margin, velocity=-8.0):
    """A tissue block shot at a fixed 2 mm plate at 16 mm a substep, with a margin far smaller than that travel:
    the predicted end of the substep is already past the plate, so nothing but a swept search finds the pair."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-1e3,
            raise_on_env_failure=False,
            contact_margin=margin,
            contact_ccd=contact_ccd,
        ),
        show_viewer=False,
    )
    plate = scene.add_entity(
        morph=gs.morphs.Box(size=(0.2, 0.2, 0.002), pos=(0.0, 0.0, 0.0), fixed=True),
        material=gs.materials.Rigid(),
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.022), nobisect=False, maxvolume=1e-6),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, collision_group=1),
    )
    scene.vbd_solver.add_rigid_collider(plate.links[0], collision_group=0)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.0, thickness=2e-4)
    scene.build()
    state = scene.vbd_solver.get_state(0)
    state._vel[:, block.v_start : block.v_start + block.n_vertices] = torch.tensor(
        (0.0, 0.0, velocity), dtype=state._vel.dtype, device=state._vel.device
    )
    scene.vbd_solver.set_state(0, state)
    return scene, block


def test_the_swept_search_finds_a_pair_the_endpoint_search_cannot_see():
    """`test_vbd_accd.py` stops the same block with a 25 mm margin, which exceeds the travel and pays for it in
    grid cells: the cell is twice thickness plus margin, so 25 mm of margin gives a 50 mm cell. Three millimetres
    is enough once the search follows the path, which is an eighth of the cell and a fifth of the travel."""
    scene, block = _fast_block_scene(contact_ccd=True, margin=3e-3)
    for _ in range(6):
        scene.step()
    status = scene.vbd_solver.env_status()
    diagnostics = scene.vbd_solver.contact_diagnostics()
    lowest = float(tensor_to_array(block.get_positions())[0][:, 2].min())
    print(f"swept: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, "
          f"lowest {1000 * lowest:.3f} mm, min_toi {float(diagnostics.min_toi[0]):.4f}, "
          f"pairs {int(diagnostics.n_point_pairs[0])}")
    assert not bool(status.is_failed[0]), f"the substep must be taken, not refused (errno {int(status.errno[0])})"
    assert float(diagnostics.min_toi[0]) < 1.0, "the filter must have had a pair to rescale"
    assert int(diagnostics.n_point_pairs[0]) > 0, "and pairs must still be live once it rests"
    assert lowest > 0.001, "the block must stay above the plate's top face"
    assert lowest < 0.0015, "and it must have travelled to the plate rather than stopped in mid air"


def test_a_response_that_outruns_the_margin_triggers_a_rebuild_not_a_refusal():
    """This used to be the guard refusing a substep: the same impact at a 1 mm margin throws the block's
    vertices 2.98 mm off the path the search covered. Wang et al. 2022's bound (Sec. 3.1, Eq. 4) turns that
    overrun into a trigger instead of a failure -- `VBDContact.d_budget` for this environment drops below
    `margin`, so the *next* `kernel_begin_contact` rebuilds the candidate set at the wider `margin_max` reach.
    The substep that spent the budget is kept, not discarded, and the run keeps going."""
    scene, block = _fast_block_scene(contact_ccd=True, margin=1e-3)
    scene.step()
    status = scene.vbd_solver.env_status()
    diagnostics = scene.vbd_solver.contact_diagnostics()
    d_budget = float(qd_to_torch(scene.vbd_solver.contact.d_budget)[0])
    print(f"outrun: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, "
          f"deviation {1e6 * float(diagnostics.max_tissue_deviation[0]):.1f} um, "
          f"d_budget {1e6 * d_budget:.1f} um, rebuilds {int(diagnostics.rebuild_count[0])}")
    assert not bool(status.is_failed[0]), "the overrun must trigger a rebuild, not refuse the substep"
    assert float(diagnostics.max_tissue_deviation[0]) > scene.vbd_solver.contact.margin, "fixture sanity check"
    assert d_budget < scene.vbd_solver.contact.margin, "the overrun must have spent the safe bound"
    assert int(diagnostics.rebuild_count[0]) == 1, "the cold start is the first rebuild"
    scene.step()
    diagnostics = scene.vbd_solver.contact_diagnostics()
    assert not bool(scene.vbd_solver.env_status().is_failed[0])
    assert int(diagnostics.rebuild_count[0]) == 2, "the spent budget must force a second rebuild next substep"


def test_a_tunnelling_block_without_the_filter_is_still_refused():
    """With the pairs found but no filter to rescale the substep, one penalty sweep cannot stop 16 mm of travel
    inside a 0.2 mm layer, and the run must fail loudly rather than pass the block through the plate."""
    scene, block = _fast_block_scene(contact_ccd=False, margin=3e-3)
    for _ in range(4):
        scene.step()
    status = scene.vbd_solver.env_status()
    lowest = float(tensor_to_array(block.get_positions())[0][:, 2].min())
    print(f"no filter: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, "
          f"lowest {1000 * lowest:.3f} mm")
    assert bool(status.is_failed[0]), "a block the penalty cannot hold must not be reported as a taken substep"
    assert int(status.errno[0]) != 0


def _resting_pair_scene(gap, slide, steps=3):
    """Two crossed edges held `gap` apart, one sliding `slide` along the other each step. They are at a real
    angle, so the side test's magnitude floor does not exempt them: this is articulation, not noise."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=2, floor_height=-1e3, raise_on_env_failure=False),
        show_viewer=False,
    )
    lower = scene.add_entity(
        morph=gs.morphs.Box(size=(0.06, 0.006, 0.006), pos=(0.0, 0.0, 0.0), fixed=True),
        material=gs.materials.Rigid(),
    )
    upper = scene.add_entity(
        morph=gs.morphs.Box(size=(0.006, 0.06, 0.006), pos=(0.0, 0.0, 0.006 + gap), nobisect=False,
                            maxvolume=1e-7),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, collision_group=1),
    )
    scene.vbd_solver.add_rigid_collider(lower.links[0], collision_group=0)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.0, thickness=2e-4)
    scene.build()
    state = scene.vbd_solver.get_state(0)
    state._vel[:] = torch.tensor((slide, 0.0, 0.0), dtype=state._vel.dtype, device=state._vel.device)
    scene.vbd_solver.set_state(0, state)
    for _ in range(steps):
        scene.step()
    return scene


def test_edges_that_slide_across_each_other_in_contact_are_not_a_crossing():
    """The edge-edge crossing gate compared an UNSIGNED distance against the depth, so "closer than the
    thickness" -- which is exactly where a load-bearing contact sits -- satisfied half the test, and any real
    sign change satisfied the rest. Two crossed bars in contact, sliding, are articulation and must pass. The
    point-triangle gate next to it has always been signed and asks `d < -depth`; this makes its sibling ask the
    same question."""
    scene = _resting_pair_scene(gap=1.5e-4, slide=0.02)
    status = scene.vbd_solver.env_status()
    print(f"sliding in contact: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}")
    assert not bool(status.is_failed[0]), (
        f"two bars sliding across each other in contact are not a crossing (errno {int(status.errno[0])})")


def test_a_latched_failure_keeps_the_buffers_of_the_substep_that_failed():
    """A batch does not stop when one environment latches: its other environments keep stepping, and the kernel
    that rebuilds the contact buffers used to rebuild them for the failed environment too. The vertex a failure
    report names is read out of those buffers afterwards, so every multi-substep failure lost its attribution.
    The head36 probe at four substeps reported zero candidate pairs and a deviation from a later substep."""
    scene, _ = _fast_block_scene(contact_ccd=False, margin=3e-3)
    for _ in range(8):
        scene.step()
        if bool(scene.vbd_solver.env_status().is_failed[0]):
            break
    assert bool(scene.vbd_solver.env_status().is_failed[0]), "the fixture must latch a failure to preserve"
    latched = scene.vbd_solver.contact_diagnostics()
    frozen = {name: value.clone() for name, value in latched._asdict().items()}
    positions = qd_to_torch(scene.vbd_solver.contact.cv_pred).clone()
    failed_substep = int(scene.vbd_solver.env_status().failed_substep[0])

    for _ in range(8):
        scene.step()

    after = scene.vbd_solver.contact_diagnostics()
    for name, value in after._asdict().items():
        assert torch.equal(value, frozen[name]), f"{name} was overwritten after the failure latch"
    assert torch.equal(qd_to_torch(scene.vbd_solver.contact.cv_pred), positions), "predictions were overwritten"
    assert int(scene.vbd_solver.env_status().failed_substep[0]) == failed_substep
    assert int(frozen["n_point_pairs"][0]) + int(frozen["n_edge_pairs"][0]) > 0, "the evidence must not be empty"


def _two_block_scene(velocity, cell=None):
    """Two tissue blocks a hair apart, both carried at the same velocity: the pair geometry is identical at every
    speed, only the swept cell ranges grow."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=1, floor_height=-1e3, contact_margin=1e-3, contact_cell_size=cell,
            raise_on_env_failure=False
        ),
        show_viewer=False,
    )
    lower = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=1e-6),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, collision_group=1),
    )
    upper = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.0205), nobisect=False, maxvolume=1e-6),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, collision_group=2),
    )
    scene.vbd_solver.add_contact_rule(1, 2, stiffness=1e5, friction=0.0, thickness=2e-4)
    scene.build()
    state = scene.vbd_solver.get_state(0)
    state._vel[:] = torch.tensor((0.0, 0.0, velocity), dtype=state._vel.dtype, device=state._vel.device)
    scene.vbd_solver.set_state(0, state)
    scene.step()
    return scene


def test_a_vertex_swept_across_many_cells_is_collected_once():
    """A vertex now sits in every cell of its swept range, so a search that shares several of those cells would
    find it several times and collect the same pair twice, which would double the stiffness holding it. The two
    scenes here have the same geometry and no relative motion, so their candidate sets must match exactly; only
    the number of cells the vertices occupy differs, by a factor of eight."""
    still = _two_block_scene(0.0).vbd_solver.contact_diagnostics()
    carried = _two_block_scene(-8.0).vbd_solver.contact_diagnostics()
    print(f"still: {int(still.n_point_pairs[0])} pt, {int(still.n_edge_pairs[0])} ee; "
          f"carried: {int(carried.n_point_pairs[0])} pt, {int(carried.n_edge_pairs[0])} ee")
    assert int(still.n_point_pairs[0]) > 0 and int(still.n_edge_pairs[0]) > 0, "the fixture must find pairs"
    assert int(carried.n_point_pairs[0]) == int(still.n_point_pairs[0])
    assert int(carried.n_edge_pairs[0]) == int(still.n_edge_pairs[0])


def test_a_vertex_whose_swept_box_hashes_two_cells_into_one_bucket_is_still_collected_once():
    """`func_cell_hash` is a generic spatial hash, not injective per vertex: two cells of one vertex's own
    swept range can land in the same bucket, and both entries pass `func_is_canonical_cell` identically,
    since that test reads the searched cell and the vertex's own bounds, never which slot the entry sits
    in. A one-bucket table (`hash_buckets=1`) forces every cell to collide, so a vertex that spans two
    cells of its own sweep is guaranteed to appear twice in the bucket. `contact.cell_c` -- the cell an
    entry was inserted under -- and `func_is_own_cell`, which checks it against the cell currently being
    searched, are what keep that vertex collected exactly once, in O(1) rather than the O(slot) scan
    `func_is_first_entry` used to do."""
    contact = types.SimpleNamespace(
        cell_c=qd.Vector.field(3, dtype=gs.qd_int, shape=(1, 2, 1)),
        cv_lo=qd.Vector.field(3, dtype=gs.qd_int, shape=(1, 1)),
        cv_hi=qd.Vector.field(3, dtype=gs.qd_int, shape=(1, 1)),
    )
    cell_v = qd.field(dtype=gs.qd_int, shape=(1, 2, 1))
    hits = qd.field(dtype=gs.qd_int, shape=(2,))

    @qd.kernel
    def kernel():
        # one vertex (cv 0) whose own sweep spans exactly two cells, K and C
        cell_k = qd.Vector([0, 0, 0], dt=gs.qd_int)
        cell_c = qd.Vector([1, 0, 0], dt=gs.qd_int)
        contact.cv_lo[0, 0] = cell_k
        contact.cv_hi[0, 0] = cell_c
        # insert cv 0 at both of its cells, exactly as kernel_begin_contact's cv loop does
        h_k = func_cell_hash(cell_k, 1)
        cell_v[h_k, 0, 0] = 0
        contact.cell_c[h_k, 0, 0] = cell_k
        h_c = func_cell_hash(cell_c, 1)
        cell_v[h_c, 1, 0] = 0
        contact.cell_c[h_c, 1, 0] = cell_c
        # search cell K (the canonical cell of the pair, since the search box's own lower corner is K
        # too): the bucket holds both of cv 0's entries, and only the one stored under K may match
        hits[0] = 0
        for slot in range(2):
            cv = cell_v[h_k, slot, 0]
            is_new = func_is_canonical_cell(cv, 0, cell_k, cell_k, contact) and func_is_own_cell(
                h_k, slot, 0, cell_k, contact
            )
            if is_new:
                hits[0] += 1
        # search cell C: never canonical against a search box whose lower corner is K, so it must
        # contribute nothing, collision or not
        hits[1] = 0
        for slot in range(2):
            cv = cell_v[h_c, slot, 0]
            is_new = func_is_canonical_cell(cv, 0, cell_c, cell_k, contact) and func_is_own_cell(
                h_c, slot, 0, cell_c, contact
            )
            if is_new:
                hits[1] += 1

    kernel()
    assert int(hits[0]) == 1, "the canonical cell must accept the vertex exactly once despite the collision"
    assert int(hits[1]) == 0, "the non-canonical cell must never accept it, collision or not"


def _candidate_set(scene):
    """The pairs the search kept, as comparable sets of participant indices."""
    contact = scene.vbd_solver.contact
    n_pt = min(int(qd_to_torch(contact.n_pt)[0]), contact.pair_cap)
    n_ee = min(int(qd_to_torch(contact.n_ee)[0]), contact.pair_cap)
    pt = {(int(a), int(b)) for a, b in zip(
        qd_to_torch(contact.pt_pairs.a)[:n_pt, 0].tolist(), qd_to_torch(contact.pt_pairs.b)[:n_pt, 0].tolist())}
    ee = {(int(a), int(b)) for a, b in zip(
        qd_to_torch(contact.ee_pairs.a)[:n_ee, 0].tolist(), qd_to_torch(contact.ee_pairs.b)[:n_ee, 0].tolist())}
    return pt, ee


def test_the_candidate_set_does_not_depend_on_the_grid_cell_size():
    """The cell is a performance parameter and nothing else. A pair is collected because the searched box, which
    is the primitive's own sweep grown by the reach, overlaps the vertex's swept box; both are rasterised into
    whatever grid is in use, and `func_is_canonical_cell` accepts each overlap exactly once. So the set is an
    invariant of the cell size, and that is what makes the size free to choose. The head36 contact grid was
    sized off the contact layer at 0.8 mm while its triangles average 2.8 mm across, which put a single
    triangle in 350 cells and one of them in 36288."""
    sets = {}
    for cell in (None, 5e-4, 4e-3, 2e-2):
        pt, ee = _candidate_set(_two_block_scene(0.0, cell=cell))
        sets[cell] = (pt, ee)
    reference = sets[None]
    assert reference[0] and reference[1], "the fixture must find pairs of both kinds"
    for cell, found in sets.items():
        assert found[0] == reference[0], f"point-triangle set changed at cell {cell}"
        assert found[1] == reference[1], f"edge-edge set changed at cell {cell}"


def test_the_grid_cell_follows_the_mesh_and_not_the_contact_layer():
    """Left to itself the cell is the median contact edge, floored at twice the reach so that growing the box by
    the reach can never add more than a cell on a side. The blocks here are meshed far coarser than their
    0.2 mm layer, so the mesh is what decides."""
    contact = _two_block_scene(0.0).vbd_solver.contact
    floor = 2.0 * (contact.max_thickness + contact.margin_max)
    assert contact.cell > floor, "this fixture is meant to be mesh-limited, not reach-limited"
    assert contact.cell == pytest.approx(contact.median_edge, rel=1e-6)


def test_a_free_rigid_body_departs_from_its_prediction_by_nothing_while_it_falls():
    """The tissue half of this was never the hard half. A free rigid body's pose for the substep lives in the
    attachment's own table until the solve commits it back to the rigid solver, so a search that read the rigid
    solver's table saw the body standing still, gave it a sweep of zero length, and handed its whole travel to
    the guard. That is what refused every large step of the python head, whose movers are all bones."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=4, floor_height=-1e3, raise_on_env_failure=False),
        show_viewer=False,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(size=(0.4, 0.4, 0.02), pos=(0.0, 0.0, -0.01), fixed=True),
        material=gs.materials.Rigid(),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.3)),
        material=gs.materials.Rigid(rho=500.0),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.33), nobisect=False, maxvolume=1e-6),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, collision_group=1),
    )
    rest = tensor_to_array(tissue.init_positions)
    tissue.add_rigid_attachments(np.flatnonzero(rest[:, 2] < rest[:, 2].min() + 1e-5), bone.links[0])
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_rigid_collider(bone.links[0], collision_group=2)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.0, thickness=2e-4)
    scene.vbd_solver.add_contact_rule(0, 2, stiffness=1e5, friction=0.0, thickness=2e-4)
    scene.build()
    assert bone.links[0].idx in scene.vbd_solver.rigid_attachment.free_links, "the body must be VBD's to predict"
    worst_motion, worst_deviation = 0.0, 0.0
    for _ in range(60):
        scene.step()
        diagnostics = scene.vbd_solver.contact_diagnostics()
        worst_motion = max(worst_motion, float(diagnostics.max_rigid_motion[0]))
        worst_deviation = max(worst_deviation, float(diagnostics.max_rigid_deviation[0]))
    status = scene.vbd_solver.env_status()
    print(f"free bone: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, "
          f"motion {1e6 * worst_motion:.1f} um, deviation {1e6 * worst_deviation:.1f} um, "
          f"margin {1e6 * scene.vbd_solver.contact.margin:.1f} um")
    assert not bool(status.is_failed[0]), f"the fall must not be refused (errno {int(status.errno[0])})"
    assert worst_motion > 5.0 * scene.vbd_solver.contact.margin, "the bone must really have outrun the margin"
    assert worst_deviation < 0.1 * worst_motion, "and its travel must be the prediction's, not the solve's"
