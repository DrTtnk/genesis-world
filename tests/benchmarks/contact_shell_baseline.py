"""What the contact system costs and where it breaks on a curved single-sided shell.

A trough -- an arc of a cylinder, open at both ends and along both long sides, so every boundary is a free edge
-- is launched along its own axis and pulled down onto a fixed ellipsoid. It is codimensional and curved, its
leading free edge meets the collider before any face does, and once it lands it slides the length of the
collider in sustained tangential contact. That is the configuration a digestive wall is in, at a size small
enough to sweep parameters over.

Two measurements, selected by `--mode`:

`sweep` runs the scene over substep counts with the continuous-collision filter on and off, and reports the
candidate counts, the grid memory, the smallest time of impact, how often the candidate set was rebuilt, how
far a shell vertex reached inside the collider, and how far the shell actually travelled. A configuration that
neither fails nor moves is as much a failure as one that crosses, and only the last column tells them apart.

`capacity` runs one configuration over a range of `contact_pair_cap` with the scene held fixed. The cost of a
substep must not depend on a capacity that nothing occupies; this is what says whether it does.

Not a pytest: it reports rather than asserts, and the `benchmarks` marker is reserved for the tracked
regression suite.
"""

import argparse
import json
import time
from typing import NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array


SHELL_GROUP = 1
COLLIDER_GROUP = 0
SEMI_AXES = (0.025, 0.012, 0.012)
LAUNCH_SPEED = 1.5
START_X = -0.090


class Row(NamedTuple):
    """One configuration and what it did. `travel_mm` against the free-flight distance is the whole point: a
    filter that clamps every substep leaves a run that reports success and has not moved."""

    contact_ccd: bool
    substeps: int
    pair_cap: int
    frames_run: int
    failed_frame: int
    errno: int
    peak_point_pairs: int
    peak_edge_pairs: int
    min_toi: float
    rebuilds: int
    depth_mm: float
    travel_mm: float
    ms_per_substep: float
    grid_mb: float
    cell_mm: float


def trough(radius, half_angle, length, n_along, n_around):
    x = np.linspace(-0.5 * length, 0.5 * length, n_along)
    angle = np.linspace(-half_angle, half_angle, n_around)
    grid_x, grid_a = np.meshgrid(x, angle, indexing="ij")
    verts = np.stack([grid_x, radius * np.sin(grid_a), -radius * np.cos(grid_a)], axis=-1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_along - 1), np.arange(n_around - 1), indexing="ij")
    a = (i * n_around + j).reshape(-1)
    b, c, d = a + n_around, a + 1, a + n_around + 1
    # wound so the normal points away from the axis: the concave side is the lumen, and the collider is on it
    faces = np.concatenate([np.stack([a, c, b], axis=-1), np.stack([b, c, d], axis=-1)])
    return verts, faces


def ellipsoid_xml(semi_axes):
    return (
        '<mujoco><worldbody><body pos="0 0 0">'
        f'<geom type="ellipsoid" size="{semi_axes[0]} {semi_axes[1]} {semi_axes[2]}"/>'
        "</body></worldbody></mujoco>"
    )


def inside_depth(points, semi_axes):
    """How far each point is inside the ellipsoid, along the ray from the centre. Exact on the axes and an
    underestimate elsewhere, which is the safe direction for a penetration report."""
    scaled = np.linalg.norm(points / semi_axes, axis=1)
    inside = scaled < 1.0
    depth = np.zeros(len(points))
    depth[inside] = np.linalg.norm(points[inside], axis=1) * (1.0 / scaled[inside] - 1.0)
    return depth


def build(substeps, contact_ccd, pair_cap, n_along, n_around, margin, stiffness, friction, thickness):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2.5e-3, substeps=substeps, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=12,
            floor_height=-1e3,
            raise_on_env_failure=False,
            contact_margin=margin,
            contact_ccd=contact_ccd,
            contact_pair_cap=pair_cap,
        ),
        show_viewer=False,
    )
    verts, faces = trough(0.030, np.deg2rad(100.0), 0.120, n_along, n_around)
    shell = scene.add_entity(
        material=gs.materials.VBD.Shell(
            E=3e5, nu=0.3, thickness=1.5e-3, bending_stiffness=1e-6, collision_group=SHELL_GROUP
        ),
        morph=gs.morphs.TriMesh(verts=verts + np.array([START_X, 0.0, 0.028]), faces=faces),
    )
    collider = scene.add_entity(morph=gs.morphs.MJCF(file=ellipsoid_xml(SEMI_AXES)), material=gs.materials.Rigid())
    scene.vbd_solver.add_rigid_collider(collider.links[-1], collision_group=COLLIDER_GROUP)
    scene.vbd_solver.add_contact_rule(
        COLLIDER_GROUP, SHELL_GROUP, stiffness=stiffness, friction=friction, thickness=thickness
    )
    scene.build()
    # `vel` is a read-only property, so the backing tensor is written in place
    state = scene.vbd_solver.get_state(0)
    state._vel[:, shell.v_start : shell.v_start + shell.n_vertices] = torch.tensor(
        [LAUNCH_SPEED, 0.0, -0.4], dtype=state._vel.dtype, device=state._vel.device
    )
    scene.vbd_solver.set_state(0, state)
    return scene, shell


def run(substeps, contact_ccd, pair_cap, frames, n_along, n_around, margin, stiffness, friction, thickness):
    scene, shell = build(substeps, contact_ccd, pair_cap, n_along, n_around, margin, stiffness, friction, thickness)
    contact = scene.vbd_solver.contact
    contact.clear_toi()
    semi_axes = np.array(SEMI_AXES)
    grid_mb = contact.hash_buckets * contact.hash_cap * scene.vbd_solver._B * 4 * 4 / 2**20

    peak_pt = peak_ee = timed_frames = 0
    worst_depth = 0.0
    failed_frame = -1
    start = None
    for frame in range(frames):
        scene.step()
        if start is None:
            # the first step compiles the kernels; the clock starts after it
            torch.cuda.synchronize()
            start = time.perf_counter()
        else:
            timed_frames += 1
        diagnostics = scene.vbd_solver.contact_diagnostics()
        peak_pt = max(peak_pt, int(diagnostics.n_point_pairs[0]))
        peak_ee = max(peak_ee, int(diagnostics.n_edge_pairs[0]))
        positions = tensor_to_array(shell.get_positions())[0]
        worst_depth = max(worst_depth, inside_depth(positions, semi_axes).max())
        if bool(scene.vbd_solver.env_status().is_failed[0]):
            failed_frame = frame
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    diagnostics = scene.vbd_solver.contact_diagnostics()
    positions = tensor_to_array(shell.get_positions())[0]
    row = Row(
        contact_ccd=contact_ccd,
        substeps=substeps,
        pair_cap=pair_cap,
        frames_run=frames if failed_frame < 0 else failed_frame + 1,
        failed_frame=failed_frame,
        errno=int(scene.vbd_solver.env_status().errno[0]),
        peak_point_pairs=peak_pt,
        peak_edge_pairs=peak_ee,
        min_toi=float(diagnostics.min_toi[0]),
        rebuilds=int(diagnostics.rebuild_count[0]),
        depth_mm=1000.0 * worst_depth,
        travel_mm=1000.0 * (positions[:, 0].mean() - START_X),
        ms_per_substep=1000.0 * elapsed / max(1, timed_frames * substeps),
        grid_mb=grid_mb,
        cell_mm=1000.0 * contact.cell,
    )
    gs.destroy()
    return row


def report(rows, frames):
    free_flight = 1000.0 * LAUNCH_SPEED * frames * 2.5e-3
    header = (
        f"{'ccd':>4}{'sub':>5}{'cap':>8}{'frames':>8}{'fail@':>7}{'errno':>7}{'pt':>6}{'ee':>6}"
        f"{'toi':>9}{'rebuild':>9}{'depth':>8}{'travel':>8}{'ms/sub':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row.contact_ccd):>4}{row.substeps:>5}{row.pair_cap:>8}{row.frames_run:>8}{row.failed_frame:>7}"
            f"{row.errno:>7}{row.peak_point_pairs:>6}{row.peak_edge_pairs:>6}{row.min_toi:>9.4f}{row.rebuilds:>9}"
            f"{row.depth_mm:>8.3f}{row.travel_mm:>8.1f}{row.ms_per_substep:>9.3f}"
        )
    print(f"\ndepth and travel in mm; free flight over {frames} frames would be {free_flight:.0f} mm")
    print("errno bits: 128 cell, 256 pair, 512 sweep, 1024 nan, 2048 crossing, 4096 motion bound")
    print(f"grid {rows[0].grid_mb:.2f} MB at a {rows[0].cell_mm:.3f} mm cell")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sweep", "capacity"), default="sweep")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--along", type=int, default=40)
    parser.add_argument("--around", type=int, default=24)
    parser.add_argument("--margin", type=float, default=2e-4)
    parser.add_argument("--thickness", type=float, default=2e-4)
    parser.add_argument("--stiffness", type=float, default=1e5)
    parser.add_argument("--friction", type=float, default=0.3)
    parser.add_argument("--substeps", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--pair-caps", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.mode == "sweep":
        configurations = [(ccd, substeps, 65536) for ccd in (False, True) for substeps in args.substeps]
    else:
        configurations = [(False, 5, pair_cap) for pair_cap in args.pair_caps]

    rows = []
    for contact_ccd, substeps, pair_cap in configurations:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        rows.append(
            run(
                substeps=substeps,
                contact_ccd=contact_ccd,
                pair_cap=pair_cap,
                frames=args.frames,
                n_along=args.along,
                n_around=args.around,
                margin=args.margin,
                stiffness=args.stiffness,
                friction=args.friction,
                thickness=args.thickness,
            )
        )
    report(rows, args.frames)
    if args.out is not None:
        with open(args.out, "w") as handle:
            json.dump([row._asdict() for row in rows], handle, indent=2)


if __name__ == "__main__":
    main()
