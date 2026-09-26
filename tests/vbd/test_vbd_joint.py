"""Joints between rigid links: a six-axis spring in a frame fixed to each link at rest.

A snake skull's articulations were strung as capsules of tension-only ligament lines with bone-on-bone contact
for compression. Every line started at its slack point, a kink the block's Newton step crossed every substep,
and the load went through stiff penalty contact. A joint is one smooth element per articulation: translational
springs along three joint axes fixed to the first link, and alignment springs between each of those axes and its
twin fixed to the second link. Hinge, ball, slide, rigid and soft joints are the same element with some
stiffnesses zero: a hinge aligns its axis and leaves rotation about it free, a slide leaves translation along
its axis free, a ball aligns nothing. The closed-form blocks are checked against Torch autograd by
`spikes/verify_joint_blocks.py` in the snakeSim repository.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

DT = 1e-3
G = 9.81
SIZE = (0.04, 0.01, 0.01)


def _scene(gravity=(0.0, 0.0, -G), requires_grad=False):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=gravity, requires_grad=requires_grad),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            integrator=gs.integrator.approximate_implicitfast if requires_grad else gs.integrator.Euler,
        ),
        vbd_options=gs.options.VBDOptions(n_iterations=10, floor_height=-1e3),
        show_viewer=False,
    )


def _hand_over(scene, anchor):
    """VBD owns the free bodies only when some tissue is coupled to a rigid link."""
    patch = scene.add_entity(
        morph=gs.morphs.Box(size=(0.003, 0.003, 0.003), pos=(0.0, 0.2, 0.0), nobisect=False, maxvolume=4e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4),
    )
    patch.add_rigid_glue(np.arange(patch.n_vertices), anchor.links[0])


def _bar_on_a_post(kt, kr, axes=np.eye(3), gravity=(0.0, 0.0, -G)):
    """A 40 mm bar jointed at one end to a fixed post; its centre of mass sits 20 mm along +x from the joint."""
    scene = _scene(gravity)
    post = scene.add_entity(morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 0.2, 0.0), fixed=True),
                            material=gs.materials.Rigid(rho=1500.0))
    bar = scene.add_entity(morph=gs.morphs.Box(size=SIZE, pos=(0.02, 0.0, 0.0)), material=gs.materials.Rigid(rho=1500.0))
    _hand_over(scene, post)
    scene.sim.vbd_solver.add_rigid_joint(post.links[0], bar.links[0], centre=(0.0, 0.0, 0.0), axes=axes,
                                         translational_stiffness=kt, rotational_stiffness=kr)
    scene.build()
    return scene, bar


def _energy(bar):
    mass = 1500.0 * np.prod(SIZE)
    inertia = mass / 12 * np.diag([SIZE[1]**2 + SIZE[2]**2, SIZE[0]**2 + SIZE[2]**2, SIZE[0]**2 + SIZE[1]**2])
    R = gs.utils.geom.quat_to_R(tensor_to_array(bar.get_quat()))
    v, w = tensor_to_array(bar.get_vel()), tensor_to_array(bar.get_ang())
    return 0.5 * mass * v @ v + 0.5 * w @ (R @ inertia @ R.T) @ w + mass * G * tensor_to_array(bar.get_pos())[2]


def _joint_end(bar):
    """Where the bar's end at the joint is now."""
    R = gs.utils.geom.quat_to_R(tensor_to_array(bar.get_quat()))
    return tensor_to_array(bar.get_pos()) + R @ np.array([-0.02, 0.0, 0.0])


STIFF = 1e5


@pytest.mark.required
def test_a_joint_at_rest_carries_no_force():
    scene, bar = _bar_on_a_post((STIFF,) * 3, (STIFF,) * 3, gravity=(0.0, 0.0, 0.0))
    for _ in range(50):
        scene.step()
    assert np.abs(tensor_to_array(bar.get_pos()) - (0.02, 0.0, 0.0)).max() < 1e-12
    assert np.abs(tensor_to_array(bar.get_vel())).max() < 1e-12


@pytest.mark.required
def test_a_hinge_swings_about_its_axis_only_and_loses_energy():
    # the axis is the first joint axis, world y: gravity turns the bar about y
    scene, bar = _bar_on_a_post((STIFF,) * 3, (STIFF, 0.0, 0.0), axes=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]]).T)
    start = _energy(bar)
    energies, off_axis, ends = [], [], []
    for _ in range(300):
        scene.step()
        energies.append(_energy(bar))
        w = tensor_to_array(bar.get_ang())
        R = gs.utils.geom.quat_to_R(tensor_to_array(bar.get_quat()))
        world_w = R @ w
        off_axis.append(max(abs(world_w[0]), abs(world_w[2])) / max(abs(world_w[1]), 1e-9))
        ends.append(np.linalg.norm(_joint_end(bar)))
    angle = np.degrees(np.arccos(np.clip(gs.utils.geom.quat_to_R(tensor_to_array(bar.get_quat()))[:, 0] @ (1, 0, 0), -1, 1)))
    assert angle > 30.0, "gravity swings the bar down about the hinge"
    assert max(off_axis[5:]) < 1e-6, "and about the hinge axis only"
    assert max(ends) < 2e-4, "the jointed end stays on the post (within the spring's give under the load)"
    assert max(energies) <= start + 1e-12, "backward Euler on a hinge loses energy, never gains it"


@pytest.mark.required
def test_a_slide_moves_along_its_axis_only():
    # free translation along the first axis (world z), everything else held: gravity slides the bar straight down
    scene, bar = _bar_on_a_post((0.0, STIFF, STIFF), (STIFF,) * 3, axes=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]]).T)
    for _ in range(100):
        scene.step()
    t = 0.1
    p = tensor_to_array(bar.get_pos())
    assert p[2] == pytest.approx(-0.5 * G * t * t, rel=0.05), "it falls freely along the slide"
    assert abs(p[0] - 0.02) < 1e-9 and abs(p[1]) < 1e-9
    assert np.abs(tensor_to_array(bar.get_ang())).max() < 1e-9


@pytest.mark.required
def test_a_ball_joint_holds_the_centre_and_lets_every_rotation_go():
    scene, bar = _bar_on_a_post((STIFF,) * 3, (0.0, 0.0, 0.0))
    bar.set_dofs_velocity(np.array([0.0, 0.0, 0.0, 3.0, 0.0, 5.0]))
    for _ in range(100):
        scene.step()
    assert np.linalg.norm(_joint_end(bar)) < 2e-4
    R = gs.utils.geom.quat_to_R(tensor_to_array(bar.get_quat()))
    assert np.degrees(np.arccos(np.clip(R[:, 0] @ (1, 0, 0), -1, 1))) > 20.0, "the bar has turned away"


def test_a_joint_is_refused_between_one_link_and_itself_with_bad_axes_or_no_stiffness():
    scene = _scene()
    post = scene.add_entity(morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 0.2, 0.0), fixed=True),
                            material=gs.materials.Rigid(rho=1500.0))
    bar = scene.add_entity(morph=gs.morphs.Box(size=SIZE, pos=(0.02, 0.0, 0.0)), material=gs.materials.Rigid(rho=1500.0))
    solver = scene.sim.vbd_solver
    with pytest.raises(gs.GenesisException, match="two different links"):
        solver.add_rigid_joint(bar.links[0], bar.links[0], (0, 0, 0), np.eye(3), (1.0,) * 3, (1.0,) * 3)
    with pytest.raises(gs.GenesisException, match="right-handed orthonormal"):
        solver.add_rigid_joint(post.links[0], bar.links[0], (0, 0, 0), np.diag([1.0, 1.0, -1.0]), (1.0,) * 3, (1.0,) * 3)
    with pytest.raises(gs.GenesisException, match="stiffness"):
        solver.add_rigid_joint(post.links[0], bar.links[0], (0, 0, 0), np.eye(3), (0.0,) * 3, (0.0,) * 3)
    with pytest.raises(gs.GenesisException, match="stiffness"):
        solver.add_rigid_joint(post.links[0], bar.links[0], (0, 0, 0), np.eye(3), (-1.0, 1.0, 1.0), (1.0,) * 3)


def test_a_joint_is_refused_under_requires_grad():
    scene = _scene(requires_grad=True)
    post = scene.add_entity(morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0.0, 0.2, 0.0), fixed=True),
                            material=gs.materials.Rigid(rho=1500.0))
    bar = scene.add_entity(morph=gs.morphs.Box(size=SIZE, pos=(0.02, 0.0, 0.0)), material=gs.materials.Rigid(rho=1500.0))
    scene.add_entity(morph=gs.morphs.Box(size=(0.003, 0.003, 0.003), pos=(0.0, 0.3, 0.0), nobisect=False, maxvolume=4e-9),
                     material=gs.materials.VBD.Base(E=1e5, nu=0.4))
    scene.sim.vbd_solver.add_rigid_joint(post.links[0], bar.links[0], (0, 0, 0), np.eye(3), (1.0,) * 3, (1.0,) * 3)
    with pytest.raises(gs.GenesisException, match="adjoint"):
        scene.build()
