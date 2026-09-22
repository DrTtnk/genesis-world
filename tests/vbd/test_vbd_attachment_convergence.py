"""More solver work must not make the attachment answer worse.

Astra's head36 hanging gate fails under load with contact and muscle removed, and the failure gets *worse* with
more sweeps: 10.674 mm of peak raw attachment gap at 12 sweeps, 11.475 mm at 48, 15.465 mm at 192, with the
number of attachments sitting at the stiffness cap rising 259 -> 322 -> 341 of 474. Sixty-four bit arithmetic
does not repair it and gravity-off stays at rest, so it is neither precision nor geometry.

Slow convergence improves with iterations. Something here scales with the iteration count in the wrong
direction, and the forward path has a candidate: `func_update_attachment_dual` advances the augmented
Lagrangian multiplier *and* the stiffness ramp once per sweep. An augmented Lagrangian multiplier update is an
outer-loop step, taken on a converged primal iterate. The gradient path already knows this -- see the comment
on the constrained step in vbd_solver.py, "a dual update on an unconverged iterate overshoots at the stiffness
cap and limit-cycles instead of converging" -- and does exact Uzawa. The forward path does not.

`test_vbd_attachment_stiffness.py` already shows one bone on pinned tissue holding to 0.062 mm, so a single
attachment at 10 sweeps is healthy. This scene adds what head36 has and that one does not: a chain, so that a
free body's block is loaded through tissue that another body is also pulling on, and head36's 0.25 ms substep.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array


SIDE = 0.01
SWEEPS = [12, 48, 192]


def _chain(n_iterations, dt=2.5e-3, substeps=10, gravity=(0.0, 0.0, -9.81)):
    """Pinned tissue, free bone, tissue, free bone: every attachment carries everything below it.

    The coupling gate needs at least one free joint, and a chain is the smallest arrangement in which a free
    body's 6x6 block is loaded through tissue that another block also pulls on, which is what the head is.
    The substep is head36's."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=gravity),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1e3),
        show_viewer=False,
    )
    upper = scene.add_entity(
        morph=gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(0.0, 0.0, 0.1), nobisect=False, maxvolume=5e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4),
    )
    upper_bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.01), pos=(0.0, 0.0, 0.0885)),
        material=gs.materials.Rigid(rho=1000.0),
    )
    lower = scene.add_entity(
        morph=gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(0.0, 0.0, 0.078), nobisect=False, maxvolume=5e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4),
    )
    lower_bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.01), pos=(0.0, 0.0, 0.0665)),
        material=gs.materials.Rigid(rho=1000.0),
    )
    upper_rest, lower_rest = tensor_to_array(upper.init_positions), tensor_to_array(lower.init_positions)
    upper.add_rigid_attachments([int(v) for v in np.argsort(upper_rest[:, 2])[:4]], upper_bone.links[0])
    lower.add_rigid_attachments([int(v) for v in np.argsort(-lower_rest[:, 2])[:4]], upper_bone.links[0])
    lower.add_rigid_attachments([int(v) for v in np.argsort(lower_rest[:, 2])[:4]], lower_bone.links[0])
    scene.build()
    pinned = np.zeros(upper.n_vertices, dtype=bool)
    pinned[np.argsort(-upper_rest[:, 2])[:4]] = True
    upper.set_pinned(pinned)
    return scene, (upper, lower), (upper_bone, lower_bone)


def _rotate(quat, local):
    """wxyz quaternion applied to a vector, as `qd_transform_by_quat_fast` does it in the kernel."""
    w, v = quat[0], quat[1:]
    return local + 2.0 * np.cross(v, np.cross(v, local) + w * local)


def _peak_attachment_gap(scene, tissues, bones):
    """The largest raw gap over every rigid attachment: the bound tissue point, less the point on its bone.

    The weights and the stored local offset are read from the solver's own records, so this measures exactly
    what the kernel binds, not a reconstruction of it."""
    attachment = scene.sim.vbd_solver.rigid_attachment
    verts_idx = attachment.info.verts.to_numpy()
    weights = attachment.info.weights.to_numpy()
    links = attachment.info.link.to_numpy()
    local = attachment.info.local_pos.to_numpy()
    positions = np.zeros((scene.sim.vbd_solver.n_vertices, 3))
    for tissue in tissues:
        positions[tissue.v_start : tissue.v_start + tissue.n_vertices] = tensor_to_array(tissue.get_positions())[0]
    pose = {bone.links[0].idx: (tensor_to_array(bone.get_pos()), tensor_to_array(bone.get_quat())) for bone in bones}
    worst = 0.0
    for i_a in range(attachment.n_attachments):
        point = weights[i_a] @ positions[verts_idx[i_a]]
        bone_pos, bone_quat = pose[int(links[i_a])]
        worst = max(worst, float(np.linalg.norm(point - bone_pos - _rotate(bone_quat, local[i_a]))))
    return worst


def _capped_fraction(scene):
    """How many attachments sit at the stiffness cap. In head36 this count is the cleanest signal in Astra's
    traces: it rises fastest at 192 sweeps, then 48, then 12, and the failure follows it."""
    solver = scene.sim.vbd_solver
    cap = solver._constraint_k_max_ratio * solver._k_start
    stiffness = solver.rigid_attachment.state.stiffness.to_numpy()[:, 0]
    return float(np.count_nonzero(stiffness >= cap * (1.0 - 1e-6))) / len(stiffness)


def _peak_over_run(n_iterations, n_steps=40):
    scene, tissues, bones = _chain(n_iterations)
    peak, capped = 0.0, 0.0
    for _ in range(n_steps):
        scene.step()
        peak = max(peak, _peak_attachment_gap(scene, tissues, bones))
        capped = max(capped, _capped_fraction(scene))
    print(f"    {n_iterations} sweeps: {100 * capped:.1f}% of attachments reached the stiffness cap")
    return peak


@pytest.mark.required
@pytest.mark.parametrize("n_iterations", SWEEPS)
def test_a_loaded_chain_holds_its_attachments(show_viewer, n_iterations):
    """The gate itself: a hanging chain must not let an attachment open by a visible amount."""
    peak = _peak_over_run(n_iterations)
    print(f"{n_iterations} sweeps: peak raw attachment gap {1000 * peak:.4f} mm")
    assert np.isfinite(peak)
    assert peak < 1e-4, f"an attachment opened by {1000 * peak:.3f} mm at {n_iterations} sweeps"


@pytest.mark.required
def test_more_sweeps_do_not_open_the_attachments_further(show_viewer):
    """The diagnosis. Whatever the absolute gap is, sixteen times the solver work must not enlarge it: an
    iteration that makes the answer worse is not converging to anything."""
    peaks = {n: _peak_over_run(n) for n in SWEEPS}
    for n, peak in peaks.items():
        print(f"{n} sweeps: peak raw attachment gap {1000 * peak:.4f} mm")
    best = peaks[SWEEPS[0]]
    for n in SWEEPS[1:]:
        assert peaks[n] <= 1.1 * best, (
            f"{n} sweeps opened {1000 * peaks[n]:.3f} mm against {1000 * best:.3f} mm at {SWEEPS[0]} sweeps: "
            "more solver work made the answer worse"
        )
