"""Routed Hill muscle-tendon units, ligaments and rotary restraints (gates A3 and A4).

The mathematics these tests hold the engine to is proved separately, on random inputs, against torch autograd
in `spikes/verify_avbd_mtu_math.py` of the application repository: the route gradient `dL/dp_i = u_(i-1) - u_i`,
the positive semidefinite route Hessian, the series tendon stiffness `k_T A' / (k_T + A')` through the
implicitly solved fibre, the closed-form fibre slope the kernel uses in place of the reference's central
difference, and the anchor Jacobians that carry the pull onto a link and onto the vertices of a tet.

These tests are about the engine, not the formulas: that the ported law gives the reference numbers, that the
state advances once per substep and not once per sweep, that the pull reaches every owner it should, and that
the reactions balance.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.engine.solvers.vbd_mtu import (
    EPS_REF,
    HillParameters,
    K,
    K_E,
    LinkAnchor,
    N,
    TAU_ACT,
    TAU_DEACT,
    TissueAnchor,
    W,
    W_PE,
    WorldAnchor,
)
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose


F_MAX = 250.0
L_OPT = 0.06
L_SLACK = 0.06
V_MAX = 0.3


def hill_parameters():
    return HillParameters(f_max=F_MAX, l_opt=L_OPT, l_slack=L_SLACK, v_max=V_MAX)


# --------------------------------------------------------------------------------------------------------
# An independent float64 reference of the same law, written from `docs/MUSCLE_MODEL.md`. It never calls the
# engine: it is what the engine must reproduce.
# --------------------------------------------------------------------------------------------------------


def reference_force_length(fibre):
    return math.exp(-(((fibre / L_OPT - 1.0) / W) ** 2))


def reference_force_velocity(velocity):
    x = velocity / V_MAX
    if x <= 0.0:
        return (1.0 + x) / (1.0 - x / K)
    return N - (N - 1.0) / (1.0 + K_E * x)


def reference_parallel_elastic(fibre):
    return F_MAX * max((fibre / L_OPT - 1.0) / W_PE, 0.0) ** 2


def reference_series_elastic(length, fibre):
    return F_MAX * max(length - fibre - L_SLACK, 0.0) / (EPS_REF * L_SLACK)


def reference_activation_step(activation, excitation, dt):
    tau = TAU_ACT if excitation >= activation else TAU_DEACT
    return activation + (1.0 - math.exp(-dt / tau)) * (excitation - activation)


def reference_fibre_step(fibre_previous, length, activation, dt, iterations=40):
    """Backward Euler on the tendon-fibre balance, by bisection so that it shares no code path with the
    engine's Newton iteration. The residual decreases in the fibre length, so the root is bracketed."""

    def residual(fibre):
        velocity = (fibre - fibre_previous) / dt
        return (
            reference_series_elastic(length, fibre)
            - activation * F_MAX * reference_force_length(fibre) * reference_force_velocity(velocity)
            - reference_parallel_elastic(fibre)
        )

    taut = length - L_SLACK
    lo, hi = 1e-4 * L_OPT, taut
    if residual(hi) > 0.0:
        return hi
    for _ in range(iterations + 60):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reference_run(length, excitations, dt):
    """Tension after every substep of a run at a fixed route length."""
    activation, fibre, tensions = 0.0, min(L_OPT, length - L_SLACK), []
    for excitation in excitations:
        activation = reference_activation_step(activation, excitation, dt)
        fibre = reference_fibre_step(fibre, length, activation, dt)
        tensions.append(reference_series_elastic(length, fibre))
    return activation, fibre, tensions


# --------------------------------------------------------------------------------------------------------


def tissue_only_scene(n_iterations=4, substeps=4, dt=2.5e-3):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-10.0, damping=2e-3),
        show_viewer=False,
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.5, 0.0), nobisect=False, maxvolume=4e-6),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3),
    )
    return scene, tissue


@pytest.mark.parametrize("stretch", [1.02, 1.15, 0.98])
def test_the_engine_law_matches_the_independent_reference_through_a_rise_and_a_fall(stretch, precision):
    """Gate A4 parity, both activation branches and the slack transition. The route is two world anchors, so
    its length is exactly known and the only state that moves is the muscle's own.

    The float64 budget below is the accepted A4 requirement and is never relaxed. Float32 accumulates the
    same law through hundreds of substeps of fibre Newton iterations; measured on the GPU float32 backend
    (2026-09-13) the worst relative tension error across these three stretches is 2.74e-5, at stretch=0.98
    (measured 1.98e-4 N against an expected 7.23 N). The float32 budget is 1.75 times that measured floor.
    """
    atol, rtol = (1e-8, 1e-6) if precision == "64" else (1e-6, 5e-5)
    scene, _ = tissue_only_scene()
    length = stretch * (L_OPT + L_SLACK)
    scene.vbd_solver.add_mtu(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((length, 0.0, 0.0))], hill_parameters()
    )
    scene.build(n_envs=2)
    substep_dt = scene.dt / scene.sim.substeps
    schedule = [0.9] * 12 + [0.1] * 12  # rise on TAU_ACT, fall on TAU_DEACT
    excitations = []
    tensions = []
    for excitation in schedule:
        scene.vbd_solver.set_excitation(torch.full((2, 1), excitation, device=gs.device))
        scene.step()
        excitations.extend([excitation] * scene.sim.substeps)
        tensions.append(float(scene.vbd_solver.mtu_state().tension[0, 0]))
    activation, _, reference_tensions = reference_run(length, excitations, substep_dt)
    assert_allclose(scene.vbd_solver.mtu_state().activation[0, 0], activation, tol=1e-6)
    for measured, expected in zip(tensions, reference_tensions[scene.sim.substeps - 1 :: scene.sim.substeps]):
        assert abs(measured - expected) <= atol + rtol * abs(expected)
    # The tendon carries no compression, so the fibre may never be longer than the route leaves for it. A
    # route shorter than l_opt + l_slack does not make the tendon slack: the fibre shortens instead and the
    # unit still pulls. Storing an inadmissible fibre length is what made the first version of this solver
    # read a fibre velocity out of its own initial condition (useful_knowledge.md, 2026-09-13).
    state = scene.vbd_solver.mtu_state()
    assert float(state.fibre_length[0, 0]) <= length - L_SLACK + 1e-12
    assert min(tensions) >= 0.0


def test_an_unexcited_unit_carries_no_tension_whatever_its_route_length():
    """The slack transition. With no excitation the fibre resists with its parallel element alone, which is
    zero below the optimum, so the balance settles exactly at the tendon's slack length and the unit pulls
    nothing. This is the only way a taut route can transmit zero."""
    scene, _ = tissue_only_scene()
    short = scene.vbd_solver.add_mtu(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((0.98 * (L_OPT + L_SLACK), 0.0, 0.0))], hill_parameters()
    )
    long = scene.vbd_solver.add_mtu(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((1.05 * (L_OPT + L_SLACK), 0.0, 0.0))], hill_parameters()
    )
    scene.build()
    for _ in range(20):
        scene.vbd_solver.set_excitation(torch.zeros(1, 2, device=gs.device))
        scene.step()
    tension = scene.vbd_solver.mtu_state().tension[0]
    assert float(tension[short]) == 0.0
    assert float(tension[long]) > 0.0  # past the optimum the parallel element alone stretches the tendon


def test_the_activation_does_not_depend_on_the_sweep_count():
    """Activation and the fibre state advance once per substep. If they rode the sweeps instead, the same
    command would give a different muscle at a different solve accuracy."""
    states = []
    for n_iterations in (4, 8):
        scene, _ = tissue_only_scene(n_iterations=n_iterations)
        scene.vbd_solver.add_mtu(
            [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((1.05 * (L_OPT + L_SLACK), 0.0, 0.0))], hill_parameters()
        )
        scene.build()
        for _ in range(10):
            scene.vbd_solver.set_excitation(torch.full((1, 1), 0.7, device=gs.device))
            scene.step()
        state = scene.vbd_solver.mtu_state()
        states.append((float(state.activation[0, 0]), float(state.fibre_length[0, 0]), float(state.tension[0, 0])))
    assert_allclose(states[0][0], states[1][0], tol=1e-9)
    assert_allclose(states[0][1], states[1][1], tol=1e-9)
    assert_allclose(states[0][2], states[1][2], tol=1e-9)


def test_a_route_reports_the_summed_length_of_its_segments_and_a_guide_bends_it():
    """The length of a routed unit is the polyline length, not the distance between its ends."""
    scene, _ = tissue_only_scene()
    straight = scene.vbd_solver.add_mtu(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((0.12, 0.0, 0.0))], hill_parameters()
    )
    bent = scene.vbd_solver.add_mtu(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((0.06, 0.03, 0.0)), WorldAnchor((0.12, 0.0, 0.0))],
        hill_parameters(),
    )
    scene.build()
    scene.step()
    length = scene.vbd_solver.mtu_state().route_length[0]
    assert_allclose(length[straight], 0.12, tol=1e-6)
    assert_allclose(length[bent], 2.0 * math.hypot(0.06, 0.03), tol=1e-6)


def hinge_scene(n_iterations=4, substeps=4, dt=2.5e-3, gravity=(0.0, 0.0, 0.0)):
    """A hinge bone with a small tissue strap attached to it: the articulated ownership path."""
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(world, "body", name="base")
    ET.SubElement(base, "geom", type="box", size=".01 .02 .01", pos="-.06 0 0", mass=".05")
    bone = ET.SubElement(base, "body", name="bone", pos="0 0 0")
    ET.SubElement(bone, "joint", name="hinge", axis="0 1 0", range="-.6 .6", damping=".002", armature="0")
    ET.SubElement(bone, "geom", type="box", size=".05 .01 .005", pos=".05 0 0", mass=".1")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=gravity),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-10.0, damping=2e-3),
        show_viewer=False,
    )
    skeleton = scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(root, encoding="unicode")), material=gs.materials.Rigid())
    strap = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.01, 0.006), pos=(0.06, 0.0, 0.008), nobisect=False, maxvolume=2e-6),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.3),
    )
    rest = tensor_to_array(strap.init_positions)
    strap.add_rigid_attachments(np.flatnonzero(rest[:, 2] < rest[:, 2].min() + 1e-5), skeleton.get_link("bone"))
    return scene, skeleton, strap


def test_two_antagonists_hold_the_hinge_and_the_stronger_one_wins():
    """Gate A4 antagonism. Equal excitation on a symmetric pair leaves the joint where it is; raising one
    turns the joint the way that muscle shortens."""
    scene, skeleton, _ = hinge_scene()
    base, bone = skeleton.get_link("base"), skeleton.get_link("bone")
    rest = L_OPT + L_SLACK
    # a flexor above the joint and an extensor below it, both with the same moment arm
    flexor = scene.vbd_solver.add_mtu(
        [LinkAnchor(base, (-0.02, 0.0, 0.01)), LinkAnchor(bone, (rest - 0.02, 0.0, 0.01))], hill_parameters()
    )
    extensor = scene.vbd_solver.add_mtu(
        [LinkAnchor(base, (-0.02, 0.0, -0.01)), LinkAnchor(bone, (rest - 0.02, 0.0, -0.01))], hill_parameters()
    )
    scene.build()
    for _ in range(40):
        scene.vbd_solver.set_excitation(torch.tensor([[0.5, 0.5]], device=gs.device))
        scene.step()
    balanced = float(skeleton.get_qpos()[0])
    assert abs(balanced) <= 2e-3
    for _ in range(40):
        excitation = torch.zeros(1, 2, device=gs.device)
        excitation[0, flexor], excitation[0, extensor] = 1.0, 0.1
        scene.vbd_solver.set_excitation(excitation)
        scene.step()
    # the flexor runs above the hinge axis, so shortening it lifts the far end of the bone: qpos falls
    assert float(skeleton.get_qpos()[0]) < balanced - 1e-2


def test_the_pull_on_a_link_anchor_turns_the_hinge_the_way_the_route_shortens():
    """One muscle only, so the direction of the torque is unambiguous, and the route must get shorter."""
    scene, skeleton, _ = hinge_scene()
    base, bone = skeleton.get_link("base"), skeleton.get_link("bone")
    rest = L_OPT + L_SLACK
    scene.vbd_solver.add_mtu(
        [LinkAnchor(base, (-0.02, 0.0, 0.012)), LinkAnchor(bone, (rest - 0.02, 0.0, 0.012))], hill_parameters()
    )
    scene.build()
    scene.step()
    first = float(scene.vbd_solver.mtu_state().route_length[0, 0])
    for _ in range(60):
        scene.vbd_solver.set_excitation(torch.ones(1, 1, device=gs.device))
        scene.step()
    assert float(scene.vbd_solver.mtu_state().route_length[0, 0]) < first - 1e-4
    assert float(skeleton.get_qpos()[0]) < -1e-2


def test_an_intermediate_guide_receives_the_resultant_of_its_two_segments():
    """Gate A4 guide reactions. A guide anchor is pulled along `u_(i-1) - u_i` times the tension; a guide on a
    straight route feels nothing, and a guide at a corner feels the bisector."""
    scene, _ = tissue_only_scene()
    parameters = hill_parameters()
    rest = L_OPT + L_SLACK
    straight = scene.vbd_solver.add_mtu(
        [
            WorldAnchor((0.0, 0.0, 0.0)),
            WorldAnchor((0.5 * 1.1 * rest, 0.0, 0.0)),
            WorldAnchor((1.1 * rest, 0.0, 0.0)),
        ],
        parameters,
    )
    corner = scene.vbd_solver.add_mtu(
        [
            WorldAnchor((0.0, 0.0, 0.0)),
            WorldAnchor((0.55 * rest, 0.0, 0.0)),
            WorldAnchor((0.55 * rest, 0.55 * rest, 0.0)),
        ],
        parameters,
    )
    scene.build()
    for _ in range(20):
        scene.vbd_solver.set_excitation(torch.ones(1, 2, device=gs.device))
        scene.step()
    forces = scene.vbd_solver.mtu_anchor_forces()[0]
    tension = scene.vbd_solver.mtu_state().tension[0]
    assert float(tension[straight]) > 0.0
    assert_allclose(torch.linalg.vector_norm(forces[straight, 1]), 0.0, tol=1e-6 * float(tension[straight]))
    # the corner guide is pulled along the inward bisector of the right angle, with magnitude T sqrt(2)
    pull = forces[corner, 1]
    assert_allclose(torch.linalg.vector_norm(pull), math.sqrt(2.0) * float(tension[corner]), tol=1e-5)
    assert_allclose(pull[0] / torch.linalg.vector_norm(pull), -math.sqrt(0.5), tol=1e-5)
    assert_allclose(pull[1] / torch.linalg.vector_norm(pull), math.sqrt(0.5), tol=1e-5)


def test_the_route_pull_on_the_anchors_sums_to_zero():
    """An MTU is an internal force. Whatever it does to the joint, the pulls it applies to its own anchors
    must add to nothing, or the unit is a thruster."""
    scene, _ = tissue_only_scene()
    rest = L_OPT + L_SLACK
    scene.vbd_solver.add_mtu(
        [
            WorldAnchor((0.0, 0.0, 0.0)),
            WorldAnchor((0.4 * rest, 0.2 * rest, 0.0)),
            WorldAnchor((0.8 * rest, 0.1 * rest, 0.3 * rest)),
            WorldAnchor((1.1 * rest, 0.0, 0.0)),
        ],
        hill_parameters(),
    )
    scene.build()
    for _ in range(20):
        scene.vbd_solver.set_excitation(torch.ones(1, 1, device=gs.device))
        scene.step()
    forces = scene.vbd_solver.mtu_anchor_forces()[0, 0]
    tension = float(scene.vbd_solver.mtu_state().tension[0, 0])
    assert tension > 0.0
    assert_allclose(forces.sum(dim=0), torch.zeros(3, device=forces.device), tol=1e-5 * tension)


def test_a_tissue_anchor_pulls_its_supporting_vertices_and_moves_them():
    """A route anchored inside the tissue distributes its pull by the barycentric weights, and the tissue
    actually follows it."""
    scene, tissue = tissue_only_scene()
    rest = tensor_to_array(tissue.init_positions)
    corner = int(np.argmax(rest[:, 0]))
    start = rest[corner].copy()
    anchor_length = 1.1 * (L_OPT + L_SLACK)
    scene.vbd_solver.add_mtu(
        [
            WorldAnchor((float(start[0]) + anchor_length, float(start[1]), float(start[2]))),
            TissueAnchor(tissue, (corner, corner, corner, corner), (1.0, 0.0, 0.0, 0.0)),
        ],
        hill_parameters(),
    )
    scene.build()
    for _ in range(40):
        scene.vbd_solver.set_excitation(torch.ones(1, 1, device=gs.device))
        scene.step()
    moved = tensor_to_array(tissue.get_state().pos)[0, corner]
    assert moved[0] > start[0] + 1e-4  # pulled towards the world anchor, which lies at greater x
    assert float(scene.vbd_solver.mtu_state().tension[0, 0]) > 0.0


def test_a_ligament_pulls_only_when_it_is_stretched_past_its_slack_length():
    """A ligament is the tension-only linear element of the interface: no fibre, no activation, no command."""
    scene, _ = tissue_only_scene()
    slack = scene.vbd_solver.add_ligament(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((0.04, 0.0, 0.0))], stiffness=2000.0, slack_length=0.05
    )
    taut = scene.vbd_solver.add_ligament(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((0.06, 0.0, 0.0))], stiffness=2000.0, slack_length=0.05
    )
    scene.build()
    scene.step()
    tension = scene.vbd_solver.mtu_state().tension[0]
    assert float(tension[slack]) == 0.0
    assert_allclose(tension[taut], 2000.0 * 0.01, tol=1e-4)


def test_a_rotary_restraint_returns_the_hinge_to_its_rest_angle():
    """The rotary restraint of the interface is a torque -k (q - q_rest) on one joint coordinate."""
    scene, skeleton, _ = hinge_scene()
    base, bone = skeleton.get_link("base"), skeleton.get_link("bone")
    rest = L_OPT + L_SLACK
    scene.vbd_solver.add_mtu(
        [LinkAnchor(base, (-0.02, 0.0, 0.012)), LinkAnchor(bone, (rest - 0.02, 0.0, 0.012))], hill_parameters()
    )
    scene.vbd_solver.add_rotary_restraint(dof=0, stiffness=4.0, rest_angle=0.0)
    scene.build()
    for _ in range(60):
        scene.vbd_solver.set_excitation(torch.ones(1, 1, device=gs.device))
        scene.step()
    pulled = float(skeleton.get_qpos()[0])
    for _ in range(120):
        scene.vbd_solver.set_excitation(torch.zeros(1, 1, device=gs.device))
        scene.step()
    released = float(skeleton.get_qpos()[0])
    assert pulled < -1e-2
    assert abs(released) < 0.25 * abs(pulled)


def test_two_environments_take_different_commands():
    """Gate A4 asks for at least two commands and two environments in one scene."""
    scene, _ = tissue_only_scene()
    scene.vbd_solver.add_mtu(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((1.08 * (L_OPT + L_SLACK), 0.0, 0.0))], hill_parameters()
    )
    scene.build(n_envs=2)
    for _ in range(20):
        scene.vbd_solver.set_excitation(torch.tensor([[1.0], [0.0]], device=gs.device))
        scene.step()
    state = scene.vbd_solver.mtu_state()
    assert float(state.tension[0, 0]) > float(state.tension[1, 0]) + 1.0
    assert float(state.activation[1, 0]) < 1e-3


def test_an_excitation_outside_the_unit_interval_is_rejected():
    scene, _ = tissue_only_scene()
    scene.vbd_solver.add_mtu(
        [WorldAnchor((0.0, 0.0, 0.0)), WorldAnchor((0.12, 0.0, 0.0))], hill_parameters()
    )
    scene.build()
    with pytest.raises(Exception, match="excitation"):
        scene.vbd_solver.set_excitation(torch.tensor([[1.5]], device=gs.device))
    with pytest.raises(Exception, match="shape"):
        scene.vbd_solver.set_excitation(torch.zeros(1, 2, device=gs.device))


def test_a_route_with_one_anchor_is_rejected():
    scene, _ = tissue_only_scene()
    with pytest.raises(Exception, match="two anchors"):
        scene.vbd_solver.add_mtu([WorldAnchor((0.0, 0.0, 0.0))], hill_parameters())
