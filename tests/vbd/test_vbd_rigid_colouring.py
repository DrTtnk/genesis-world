"""Colours of the free bodies for the rigid block Gauss-Seidel.

Each free body owns one 6x6 block per sweep. Bodies that share an element (a joint, a muscle route, a glued
tetrahedron or rod segment across two links, or a contact pair this substep) must see each other's newest pose, so
they are solved one after the other; bodies that share none can be solved at once. The solver colours the bodies
once a substep, after the contact search, so that no two coupled bodies share a colour, and solves one colour at a
time with one thread a body. Colour by colour this is Gauss-Seidel in the order of the colours.
"""

import numpy as np
import pytest
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.vbd_mtu import HillParameters, LinkAnchor
from genesis.engine.solvers.vbd_rigid_attachment import func_attachment_link_system
from genesis.engine.solvers.vbd_rigid_colouring import func_attachment_link_system_entries, func_rigid_entry_terms
from genesis.utils.misc import tensor_to_array

from tests.vbd.test_vbd_initial_guess import _stacked_bones

DT = 1e-3
SIZE = (0.04, 0.01, 0.01)
STIFF = 1e5


def _scene(rigid_colour_cap=8):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=10, floor_height=-1e3, rigid_colour_cap=rigid_colour_cap),
        show_viewer=False,
    )


def _post(scene, pos):
    """A fixed post, with a glued patch so that VBD owns the free bodies."""
    post = scene.add_entity(morph=gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=pos, fixed=True),
                            material=gs.materials.Rigid(rho=1500.0))
    patch = scene.add_entity(
        morph=gs.morphs.Box(size=(0.003, 0.003, 0.003), pos=pos, nobisect=False, maxvolume=4e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4),
    )
    patch.add_rigid_glue(np.arange(patch.n_vertices), post.links[0])
    return post


def _bar(scene, pos):
    return scene.add_entity(morph=gs.morphs.Box(size=SIZE, pos=pos), material=gs.materials.Rigid(rho=1500.0))


def _joint(scene, a, b, centre):
    scene.sim.vbd_solver.add_rigid_joint(a.links[0], b.links[0], centre=centre, axes=np.eye(3),
                                         translational_stiffness=(STIFF,) * 3, rotational_stiffness=(STIFF,) * 3)


def _colours(scene, bodies):
    solver = scene.sim.vbd_solver
    slot = solver.rigid_attachment.free_slot.to_numpy()
    colour = solver.rigid_colouring.colour.to_numpy()[:, 0]
    return [int(colour[slot[body.links[0].idx]]) for body in bodies], int(solver.rigid_colouring.n_colours.to_numpy()[0])


def _chain(n, rigid_colour_cap=8):
    """n bars hanging end to end from a post, each jointed to the next."""
    scene = _scene(rigid_colour_cap)
    post = _post(scene, (0.0, 0.2, 0.0))
    bars = [_bar(scene, (0.02 + 0.04 * i, 0.0, 0.0)) for i in range(n)]
    _joint(scene, post, bars[0], (0.0, 0.0, 0.0))
    for i in range(n - 1):
        _joint(scene, bars[i], bars[i + 1], (0.04 * (i + 1), 0.0, 0.0))
    return scene, bars


@pytest.mark.required
def test_a_jointed_chain_takes_two_colours_and_no_joint_joins_one_colour():
    scene, bars = _chain(4)
    scene.build()
    scene.step()
    colours, n_colours = _colours(scene, bars)
    assert n_colours == 2
    assert all(colours[i] != colours[i + 1] for i in range(len(bars) - 1)), colours


@pytest.mark.required
def test_bodies_in_contact_take_different_colours_and_bodies_apart_share_one():
    scene, lower, upper = _stacked_bones()
    for _ in range(5):
        scene.step()
    colours, n_colours = _colours(scene, (lower, upper))
    assert n_colours == 2 and colours[0] != colours[1], colours

    scene = _scene()
    _post(scene, (0.0, 0.2, 0.0))
    apart = [_bar(scene, (0.1 * i, 0.0, 0.0)) for i in range(3)]
    scene.build()
    scene.step()
    colours, n_colours = _colours(scene, apart)
    assert n_colours == 1 and colours == [0, 0, 0]


@pytest.mark.required
def test_a_graph_that_needs_more_colours_than_the_cap_fails_loudly():
    scene = _scene(rigid_colour_cap=2)
    _post(scene, (0.0, 0.2, 0.0))
    bars = [_bar(scene, (0.1 * i, 0.0, 0.0)) for i in range(3)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        _joint(scene, bars[a], bars[b], (0.05 * (a + b), 0.0, 0.0))
    scene.build()
    scene.step()
    with pytest.raises(gs.GenesisException, match="rigid_colour_cap"):
        scene.sim.vbd_solver.check_errno()


@pytest.mark.required
def test_independent_chains_move_exactly_as_each_does_alone():
    """Two chains in one scene share colours, so their bodies are solved side by side; each must still move
    bit for bit as it does in a scene of its own."""

    def run(offsets):
        scene = _scene()
        chains = []
        for y in offsets:
            post = _post(scene, (0.0, y + 0.2, 0.0))
            bars = [_bar(scene, (0.02 + 0.04 * i, y, 0.0)) for i in range(3)]
            _joint(scene, post, bars[0], (0.0, y, 0.0))
            for i in range(2):
                _joint(scene, bars[i], bars[i + 1], (0.04 * (i + 1), y, 0.0))
            chains.append(bars)
        scene.build()
        for _ in range(30):
            scene.step()
        return [[np.concatenate([tensor_to_array(b.get_pos()), tensor_to_array(b.get_quat())]) for b in bars]
                for bars in chains]

    together = run((0.0, 0.5))
    alone = run((0.0,)) + run((0.5,))
    for chain_together, chain_alone in zip(together, alone):
        for pose_together, pose_alone in zip(chain_together, chain_alone):
            np.testing.assert_array_equal(pose_together, pose_alone)


@qd.kernel
def _both_blocks(f: int, solver: qd.template(), attachment: qd.template(), colouring: qd.template(),
                 serial: qd.types.ndarray(), entries: qd.types.ndarray()):
    for e, i_b in qd.ndrange(colouring.entry_cap, solver._B):
        if e < colouring.colour_entry_offset[colouring.cap, i_b]:
            func_rigid_entry_terms(f, e, i_b, solver, attachment, colouring)
    for i_f, i_b in qd.ndrange(attachment.n_free, solver._B):
        serial_force, serial_hessian = func_attachment_link_system(f, i_f, i_b, solver, attachment)
        entry_force, entry_hessian = func_attachment_link_system_entries(f, i_f, i_b, solver, attachment, colouring)
        for r in qd.static(range(6)):
            serial[i_b, i_f, r, 6] = serial_force[r]
            entries[i_b, i_f, r, 6] = entry_force[r]
            for c in qd.static(range(6)):
                serial[i_b, i_f, r, c] = serial_hessian[r, c]
                entries[i_b, i_f, r, c] = entry_hessian[r, c]


def _blocks(scene):
    solver = scene.sim.vbd_solver
    shape = (solver._B, solver.rigid_attachment.n_free, 6, 7)
    serial = torch.zeros(shape, dtype=gs.tc_float, device=gs.device)
    entries = torch.zeros_like(serial)
    _both_blocks(0, solver, solver.rigid_attachment, solver.rigid_colouring, serial, entries)
    return serial.cpu().numpy(), entries.cpu().numpy()


# The entry passes sum in the serial block's order, so the blocks agree to rounding, not always to the bit: summed
# where it is computed, a term's last multiply can be fused into the add (Quadrants compiles with fast math); read
# back from the entry buffer it cannot. Measured: at most one unit in the last place of the largest element.


@pytest.mark.required
def test_the_entry_passes_assemble_the_serial_contact_block_to_rounding():
    """Contact pairs between two free bones."""
    scene, lower, upper = _stacked_bones()
    for _ in range(5):
        scene.step()
    colouring = scene.sim.vbd_solver.rigid_colouring
    assert int(colouring.body_entry.to_numpy()[:, 0, 1].max()) > 0, "no contact entries to compare"
    serial, entries = _blocks(scene)
    np.testing.assert_allclose(entries, serial, rtol=0.0, atol=1e-14 * np.abs(serial).max())


@pytest.mark.required
def test_the_entry_passes_assemble_the_serial_muscle_block_to_rounding():
    """A muscle route across two free bars of a jointed chain."""
    scene, bars = _chain(3)
    scene.sim.vbd_solver.add_mtu(
        [LinkAnchor(bars[0].links[0], world_pos=(0.01, 0.0, 0.004)), LinkAnchor(bars[2].links[0], world_pos=(0.11, 0.0, 0.004))],
        HillParameters(f_max=2.0, l_opt=0.06, l_slack=0.03, v_max=0.5),
        activation0=1.0,
    )
    scene.build()
    for _ in range(5):
        scene.vbd_solver.set_excitation(torch.ones(1, 1, device=gs.device))
        scene.step()
    serial, entries = _blocks(scene)
    assert np.abs(serial).max() > 0.0
    np.testing.assert_allclose(entries, serial, rtol=0.0, atol=1e-14 * np.abs(serial).max())
