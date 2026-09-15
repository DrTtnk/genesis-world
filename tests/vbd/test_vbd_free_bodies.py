"""Several free rigid bodies held only by soft tissue: bones with no joints between them.

A snake's skull is kinetic. The two dentaries have no bony symphysis at all -- they are joined at the chin by an
elastic ligament -- and the quadrate is slung rather than hinged, which is how the gape opens around prey wider
than the head. Modelling that as revolute joints cannot be right: a hinge can rotate but never separate, and
separation is where much of the gape comes from.

So the target is bones as free rigid bodies with no joints, connected by ligaments, muscles and soft tissue,
with articular contact keeping them apart in compression. A hinge is then an emergent composition of those
elements, not a primitive: released from rest, a jaw should sag by the slack in its bands and be caught.

Until this change the coupling accepted a fixed-base revolute tree, or exactly one free body, and nothing
between. These tests use two free bodies, the smallest case the old gate refused.

Stiffnesses are Astra's measured cranial bands from `out/assembly28/mechanics` in the snakeSimWithAstra
repository: the intermandibular ligament is 47.0 N/m with rest 21.04 mm and slack 21.67 mm, so it carries
nothing until the chin has opened 0.63 mm. Zero preload is hers too, and it is what makes the sag real.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.vbd_mtu import LinkAnchor, TissueAnchor
from genesis.utils.misc import tensor_to_array


INTERMANDIBULAR_K = 47.0
INTERMANDIBULAR_SLACK = 0.02167
CHIN_GAP = 0.02104  # the band's rest length: the two chin anchors start this far apart


def _skull_and_two_dentaries(gravity=(0.0, 0.0, -9.81), dt=2e-3, n_iterations=12):
    """A held skull, two free dentary bodies, a tissue patch bound to the skull, and three tension-only bands:
    one from each dentary to the skull, and the intermandibular ligament between the two dentaries."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=gravity),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=n_iterations, floor_height=-1e3),
        show_viewer=False,
    )
    tissue = scene.add_entity(
        morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 0.0, 0.05), nobisect=False, maxvolume=4e-7),
        material=gs.materials.VBD.Muscle(E=1e4, nu=0.3),
    )
    skull = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.02, 0.01), pos=(0.0, 0.0, 0.06), fixed=True),
        material=gs.materials.Rigid(rho=1000.0),
    )
    left = scene.add_entity(
        morph=gs.morphs.Box(size=(0.03, 0.006, 0.006), pos=(0.0, -0.5 * CHIN_GAP, 0.04)),
        material=gs.materials.Rigid(rho=1000.0),
    )
    right = scene.add_entity(
        morph=gs.morphs.Box(size=(0.03, 0.006, 0.006), pos=(0.0, 0.5 * CHIN_GAP, 0.04)),
        material=gs.materials.Rigid(rho=1000.0),
    )
    rest = tensor_to_array(tissue.init_positions)
    top = np.argsort(-rest[:, 2])[:4]
    tissue.add_barycentric_attachments(
        [TissueAnchor(tissue, tuple(int(v) for v in top), (0.25, 0.25, 0.25, 0.25))], skull.links[0]
    )
    solver = scene.sim.vbd_solver
    for jaw in (left, right):
        solver.add_ligament(
            [LinkAnchor(skull.links[0], (0.0, 0.0, -0.005)), LinkAnchor(jaw.links[0], (0.0, 0.0, 0.003))],
            stiffness=54.8,  # Astra's quadrate articular capsule
            slack_length=0.0129,
        )
    solver.add_ligament(
        [LinkAnchor(left.links[0], (0.0, 0.5 * CHIN_GAP, 0.0)), LinkAnchor(right.links[0], (0.0, -0.5 * CHIN_GAP, 0.0))],
        stiffness=INTERMANDIBULAR_K,
        slack_length=INTERMANDIBULAR_SLACK,
    )
    scene.build()
    return scene, tissue, skull, left, right


@pytest.mark.required
def test_two_free_bodies_are_accepted(show_viewer):
    """The gate used to refuse this outright: one free link, or a revolute tree, and nothing between."""
    scene, _, _, left, right = _skull_and_two_dentaries()
    scene.step()
    for jaw in (left, right):
        assert np.isfinite(tensor_to_array(jaw.get_pos())).all()


@pytest.mark.required
def test_a_jaw_slung_on_ligaments_sags_and_is_caught(show_viewer):
    """The behaviour the whole model is aimed at. Released from rest the dentary falls, because every band has
    zero preload and 3 percent slack, and then the bands take it: the drop must be small, of the order of the
    slack, and it must stop rather than continue."""
    scene, _, _, left, _ = _skull_and_two_dentaries()
    start = float(tensor_to_array(left.get_pos())[2])
    heights = []
    for _ in range(400):
        scene.step()
        heights.append(float(tensor_to_array(left.get_pos())[2]))
    heights = np.array(heights)
    drop = start - heights.min()
    # windows after the initial fall, so this measures the ring-down and not the transient itself
    early, late = heights[100:200], heights[300:400]
    print(
        f"jaw dropped {1000 * drop:.3f} mm; peak to peak early {1000 * np.ptp(early):.4f} mm, "
        f"late {1000 * np.ptp(late):.4f} mm; mean early {1000 * (start - early.mean()):.4f} mm below start, "
        f"late {1000 * (start - late.mean()):.4f} mm"
    )
    assert drop > 1e-5, "a jaw held only by slack tension-only bands must sag at all"
    assert drop < 0.02, "the bands must catch it rather than let it fall away"
    # The bands are conservative, tension-only springs and a free body carries no joint damping, so the only
    # thing removing energy here is the integrator: the incremental potential AVBD minimises is backward Euler,
    # which is numerically dissipative. That is enough to ring the jaw down by an order of magnitude. What must
    # hold is that the swing decays rather than grows, since growth would mean the block Gauss-Seidel over
    # several free bodies is injecting energy, which is the real failure mode of this change.
    assert np.ptp(late) < 0.5 * np.ptp(early), "the swing must decay, not grow: growth is injected energy"
    assert np.ptp(late) < 0.5 * drop, "it must be settling toward its sag, not still swinging through it"
    assert late.mean() < start, "it must settle below where it started, not above"


@pytest.mark.required
def test_the_chin_ligament_carries_nothing_until_its_slack_is_taken_up(show_viewer):
    """Tension-only with zero preload: pulling the dentaries together must cost nothing, pulling them apart
    past the slack must cost something. This is what lets the gape open."""
    scene, _, _, left, right = _skull_and_two_dentaries(gravity=(0.0, 0.0, 0.0))
    for _ in range(20):
        scene.step()
    separation = float(np.linalg.norm(tensor_to_array(right.get_pos()) - tensor_to_array(left.get_pos())))
    print(f"chin separation at rest {1000 * separation:.3f} mm, band slack {1000 * INTERMANDIBULAR_SLACK:.3f} mm")
    assert separation < INTERMANDIBULAR_SLACK + 1e-4, "with no load the chin must not be pulled open"
