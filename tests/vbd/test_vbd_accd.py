"""Additive continuous collision detection: the bound itself, and the substep rescaling built on it."""

import numpy as np
import pytest
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.vbd_accd import EDGE_EDGE, POINT_TRIANGLE, func_accd_toi
from genesis.utils.misc import qd_to_torch, tensor_to_array

N_CASES = 64
SCALE = 0.9
GAP = 1e-4
ITERATIONS = 64


@qd.kernel
def kernel_accd_reference(
    starts: qd.template(), ends: qd.template(), toi: qd.template(), gap: float, scale: float, iterations: int
):
    for i in range(N_CASES):
        toi[i, 0] = func_accd_toi(
            starts[i, 0],
            starts[i, 1],
            starts[i, 2],
            starts[i, 3],
            ends[i, 0],
            ends[i, 1],
            ends[i, 2],
            ends[i, 3],
            gap,
            scale,
            iterations,
            POINT_TRIANGLE,
        )
        toi[i, 1] = func_accd_toi(
            starts[i, 0],
            starts[i, 1],
            starts[i, 2],
            starts[i, 3],
            ends[i, 0],
            ends[i, 1],
            ends[i, 2],
            ends[i, 3],
            gap,
            scale,
            iterations,
            EDGE_EDGE,
        )


def _point_triangle_distance(x, a, b, c):
    """Clamped closest-point distance in float64, over every closest-feature region: the face, the three edges
    and the three vertices. `test_vbd_contact.py` already checks this construction against a dense sampling of
    the triangle, so here it is the exact reference the bound is measured against."""
    normal = np.cross(b - a, c - a)
    area = np.linalg.norm(normal)
    candidates = [np.linalg.norm(x - v) for v in (a, b, c)]
    for u, v in ((a, b), (b, c), (c, a)):
        edge = v - u
        s = np.clip((x - u) @ edge / (edge @ edge), 0.0, 1.0)
        candidates.append(np.linalg.norm(x - (u + s * edge)))
    if area > 0.0:
        unit = normal / area
        projected = x - (x - a) @ unit * unit
        weights = [
            np.cross(c - b, projected - b) @ unit / area,
            np.cross(a - c, projected - c) @ unit / area,
            np.cross(b - a, projected - a) @ unit / area,
        ]
        if min(weights) >= 0.0:
            candidates.append(abs((x - a) @ unit))
    return float(min(candidates))


def _segment_segment_distance(a, b, c, d):
    """Clamped closest distance of two segments in float64, by the interior solution when it is interior and the
    four point-segment cases otherwise."""
    u, v, w = b - a, d - c, a - c
    uu, vv, uv = u @ u, v @ v, u @ v
    denom = uu * vv - uv * uv
    candidates = []
    if denom > 0.0:
        s = (uv * (v @ w) - vv * (u @ w)) / denom
        t = (uu * (v @ w) - uv * (u @ w)) / denom
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            candidates.append(np.linalg.norm(a + s * u - c - t * v))
    for point, start, edge in ((a, c, v), (b, c, v), (c, a, u), (d, a, u)):
        r = np.clip((point - start) @ edge / (edge @ edge), 0.0, 1.0)
        candidates.append(np.linalg.norm(point - (start + r * edge)))
    return float(min(candidates))


def _pair_distance(x, kind):
    if kind == POINT_TRIANGLE:
        return _point_triangle_distance(x[0], x[1], x[2], x[3])
    return _segment_segment_distance(x[0], x[1], x[2], x[3])


def _sweep_toi(start, end, kind, samples=801):
    """First time in a dense sweep at which the pair is at or inside the gap, else 1."""
    for t in np.linspace(0.0, 1.0, samples):
        if _pair_distance(start + t * (end - start), kind) <= GAP:
            return float(t)
    return 1.0


def _crossing_pair(rng, kind):
    """A sweep that really does pass through: the point crosses the triangle inside its face, or each edge
    crosses to the far side of the other. Random displacements alone almost never hit (5 of 128 tried)."""
    if kind == POINT_TRIANGLE:
        triangle = rng.normal(size=(3, 3)) * 0.01
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        inside = triangle.mean(axis=0)
        start = np.vstack([inside + normal * rng.uniform(0.002, 0.02), triangle])
        end = np.vstack([inside - normal * rng.uniform(0.002, 0.02), triangle])
        return start, end
    a = rng.normal(size=3) * 0.01
    b = a + rng.normal(size=3) * 0.02
    mid = 0.5 * (a + b)
    across = np.cross(b - a, rng.normal(size=3))
    across /= np.linalg.norm(across)
    along = np.cross(b - a, across)
    along /= np.linalg.norm(along)
    offset = along * rng.uniform(0.002, 0.02)
    start = np.vstack([a, b, mid + offset - across * 0.01, mid + offset + across * 0.01])
    end = np.vstack([a, b, mid - offset - across * 0.01, mid - offset + across * 0.01])
    return start, end


def test_the_bound_allows_no_contact_and_never_exceeds_the_swept_time_of_impact():
    """The two properties the rescaling rests on, checked against a dense sweep of the same geometry: nothing
    touches before the time the bound returns, and the bound is never later than the true impact."""
    generator = np.random.default_rng(20260918)
    # half the rows are free sweeps in a 20 mm box, which mostly miss or graze; the other half are built to
    # pass through, a quarter for each kind, so both the safe side and the impact side are exercised
    starts = [generator.normal(size=(4, 3)) * 0.01 for _ in range(N_CASES // 2)]
    ends = [start + generator.normal(size=(4, 3)) * 0.01 for start in starts]
    for kind in (POINT_TRIANGLE, EDGE_EDGE):
        for _ in range(N_CASES // 4):
            start, end = _crossing_pair(generator, kind)
            starts.append(start)
            ends.append(end)
    starts, ends = np.stack(starts), np.stack(ends)
    start_field = qd.Vector.field(3, dtype=gs.qd_float, shape=(N_CASES, 4))
    end_field = qd.Vector.field(3, dtype=gs.qd_float, shape=(N_CASES, 4))
    start_field.from_torch(torch.as_tensor(starts, dtype=gs.tc_float).to(gs.device))
    end_field.from_torch(torch.as_tensor(ends, dtype=gs.tc_float).to(gs.device))
    toi_field = qd.field(dtype=gs.qd_float, shape=(N_CASES, 2))
    kernel_accd_reference(start_field, end_field, toi_field, GAP, SCALE, ITERATIONS)
    toi = qd_to_torch(toi_field).to(torch.float64).cpu().numpy()

    assert ((toi >= 0.0) & (toi <= 1.0)).all()
    crossing = 0
    for i in range(N_CASES):
        for kind in (POINT_TRIANGLE, EDGE_EDGE):
            start, end = starts[i], ends[i]
            bound = toi[i, kind]
            if _pair_distance(start, kind) <= GAP:
                assert bound == 0.0
                continue
            # nothing may touch before the bound: sample the interval it declares safe
            for t in np.linspace(0.0, bound, 65):
                assert _pair_distance(start + t * (end - start), kind) > GAP
            swept = _sweep_toi(start, end, kind)
            if swept < 1.0:
                crossing += 1
                assert bound <= swept + 2e-3
    assert crossing > 24, "the sweeps must include real impacts, or this proves nothing"


def test_a_scale_outside_the_open_unit_interval_is_refused():
    for scale in (0.0, 1.0, 1.5):
        with pytest.raises(Exception):
            gs.Scene(vbd_options=gs.options.VBDOptions(contact_ccd=True, contact_ccd_scale=scale))


def _fast_block_scene(contact_ccd, velocity=-8.0):
    """A tissue block shot at a fixed 2 mm plate at 16 mm a substep, twenty times the contact layer and eight
    times the plate's thickness: without the filter the substep has to be refused."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-1e3,
            raise_on_env_failure=False,
            contact_margin=2.5e-2,
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


def test_without_the_filter_a_tunnelling_block_is_refused():
    """The state of affairs the filter exists to change, so that the next test proves something."""
    scene, block = _fast_block_scene(contact_ccd=False)
    for _ in range(4):
        scene.step()
    status = scene.vbd_solver.env_status()
    lowest = float(tensor_to_array(block.get_positions())[0][:, 2].min())
    print(f"no filter: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, lowest {1000*lowest:.2f} mm")
    assert bool(status.is_failed[0])
    assert int(status.errno[0]) != 0


def test_the_filter_stops_a_tunnelling_block_at_the_plate():
    scene, block = _fast_block_scene(contact_ccd=True)
    for _ in range(4):
        scene.step()
    status = scene.vbd_solver.env_status()
    diagnostics = scene.vbd_solver.contact_diagnostics()
    positions = tensor_to_array(block.get_positions())[0]
    lowest = float(positions[:, 2].min())
    print(
        f"filter: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, "
        f"lowest {1000*lowest:.3f} mm, min_toi {float(diagnostics.min_toi[0]):.4f}"
    )
    assert not bool(status.is_failed[0]), f"the substep must be taken, not refused (errno {int(status.errno[0])})"
    assert lowest > 0.001, "the block must stay above the plate's top face"
    assert float(diagnostics.min_toi[0]) < 1.0, "the substep must actually have been rescaled"
    assert lowest < 0.004, "and must have travelled to the plate instead of being frozen in place"


def test_the_filter_holds_a_free_body_and_its_tissue_together():
    """A free rigid bone is rescaled with the tissue it carries: one time of impact for the whole substep, so
    the attachment it is bound by is not stretched by the rescaling."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=4,
            floor_height=-1e3,
            raise_on_env_failure=False,
            contact_margin=2.5e-2,
            contact_ccd=True,
        ),
        show_viewer=False,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(size=(0.2, 0.2, 0.02), pos=(0.0, 0.0, -0.01), fixed=True),
        material=gs.materials.Rigid(),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.03, 0.0, 0.022)),
        material=gs.materials.Rigid(rho=500.0),
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.022), nobisect=False, maxvolume=1e-6),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, collision_group=1),
    )
    rest = tensor_to_array(tissue.init_positions)
    tissue.add_rigid_attachments(np.flatnonzero(rest[:, 0] > rest[:, 0].max() - 1e-5), bone.links[0])
    scene.vbd_solver.add_rigid_collider(table.links[0], collision_group=0)
    scene.vbd_solver.add_rigid_collider(bone.links[0], collision_group=2)
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.0, thickness=2e-4)
    scene.vbd_solver.add_contact_rule(0, 2, stiffness=1e5, friction=0.0, thickness=2e-4)
    scene.build()
    bone.set_dofs_velocity(torch.tensor([0.0, 0.0, -6.0, 0.0, 0.0, 0.0], dtype=gs.tc_float))
    state = scene.vbd_solver.get_state(0)
    state._vel[:, tissue.v_start : tissue.v_start + tissue.n_vertices] = torch.tensor(
        (0.0, 0.0, -6.0), dtype=state._vel.dtype, device=state._vel.device
    )
    scene.vbd_solver.set_state(0, state)
    for _ in range(4):
        scene.step()
    status = scene.vbd_solver.env_status()
    diagnostics = scene.vbd_solver.contact_diagnostics()
    bone_z = float(bone.get_pos().reshape(1, 3)[0, 2])
    tissue_low = float(tensor_to_array(tissue.get_positions())[0][:, 2].min())
    print(
        f"free body: failed={bool(status.is_failed[0])}, errno={int(status.errno[0])}, bone z {1000*bone_z:.3f} mm, "
        f"tissue lowest {1000*tissue_low:.3f} mm, min_toi {float(diagnostics.min_toi[0]):.4f}"
    )
    assert not bool(status.is_failed[0]), f"the substep must be taken, not refused (errno {int(status.errno[0])})"
    assert float(diagnostics.min_toi[0]) < 1.0
    assert bone_z > 0.01, "the bone's centre must stay a half-width above the table"
    assert tissue_low > -1e-4, "and the tissue must stay out of the table"


def test_a_prescribed_collider_with_the_filter_is_refused():
    """A driven pose is not the solver's to rescale, so the combination fails loudly at build."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=4, floor_height=-1e3, contact_ccd=True),
        show_viewer=False,
    )
    meal = scene.add_entity(
        morph=gs.morphs.MJCF(
            file='<mujoco><worldbody><body pos="0 0 0.08"><geom type="ellipsoid" size="0.03 0.02 0.02"/></body></worldbody></mujoco>',
            decimate=False,
        ),
        material=gs.materials.Rigid(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.0), nobisect=False, maxvolume=1e-6),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3, collision_group=1),
    )
    scene.vbd_solver.add_prescribed_collider(meal, collision_group=0, link=meal.links[1])
    scene.vbd_solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.0, thickness=2e-4)
    with pytest.raises(Exception, match="prescribed"):
        scene.build()
