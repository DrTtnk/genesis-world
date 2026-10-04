"""Unwired CUDA cooperative contact assembly agrees with the serial rigid Gauss-Seidel block."""

import numpy as np
import pytest
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.vbd_contact import func_contact_link_terms
from genesis.engine.solvers.vbd_joint import func_joint_link_terms
from genesis.engine.solvers.vbd_rigid_attachment import func_solve_attachment_link
from genesis.engine.solvers.vbd_rigid_cooperative import contact_blocks_cooperative, sweep_rigid_contact_cooperative
from genesis.utils.misc import qd_to_torch


pytestmark = pytest.mark.precision("64")


@qd.kernel
def kernel_serial_contact_blocks(solver: qd.template(), attachment: qd.template(), force: qd.template(),
                                 hessian: qd.template()):
    for i_f, i_b in qd.ndrange(attachment.n_free, solver._B):
        i_l = attachment.free_info[i_f].link
        origin = attachment.link_state[i_f, i_b].pos
        force[i_f, i_b], hessian[i_f, i_b] = func_contact_link_terms(0, i_l, i_b, origin, solver, solver.contact)


@qd.kernel
def kernel_serial_rigid_sweep(solver: qd.template(), attachment: qd.template()):
    for i_b in range(solver._B):
        if not solver.env_failed[i_b]:
            for i_f in range(attachment.n_free):
                func_solve_attachment_link(0, i_f, i_b, solver, attachment)


@qd.kernel
def kernel_joint_wrenches(solver: qd.template(), attachment: qd.template(), force: qd.template()):
    for i_f, i_b in qd.ndrange(attachment.n_free, solver._B):
        i_l = attachment.free_info[i_f].link
        force[i_f, i_b], _ = func_joint_link_terms(i_l, i_b, attachment, solver.joints)


def _two_bones_on_plate(n_envs, linearization="iterate"):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=1, floor_height=-float("inf"), contact_margin=2e-4,
            contact_linearization=linearization,
        ),
        show_viewer=False,
    )
    plate = scene.add_entity(
        morph=gs.morphs.Box(size=(0.1, 0.06, 0.004), pos=(0.0, 0.0, -0.002), fixed=True),
        material=gs.materials.Rigid(rho=1500.0),
    )
    bones = [
        scene.add_entity(
            morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(x, 0.0, z)),
            material=gs.materials.Rigid(rho=1500.0),
        )
        for x, z in ((-0.015, 0.00165), (0.015, 0.00168))
    ]
    solver = scene.vbd_solver
    for bone in bones:
        solver.add_rigid_link(bone.links[0])
        solver.add_rigid_collider(bone.links[0], collision_group=1)
    solver.add_rigid_collider(plate.links[0], collision_group=0)
    solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.1, thickness=2e-4)
    solver.add_rigid_joint(
        bones[0].links[0], bones[1].links[0], (0.0, 0.0, 0.00165), np.eye(3),
        translational_stiffness=(1e3, 1e3, 1e3), rotational_stiffness=(1e-2, 1e-2, 1e-2),
    )
    scene.build(n_envs=n_envs)
    scene.step()
    assert not scene.vbd_solver.env_status().is_failed.any()
    contact = solver.contact_diagnostics()
    assert (contact.n_edge_pairs > 0).all(), "both bones must have real edge contacts"
    return scene


def _cuda_only():
    if gs.backend != gs.cuda:
        pytest.skip("CUDA-only cooperative prototype")


@pytest.mark.parametrize("linearization", ["iterate", "substep"])
def test_same_pose_contact_wrench_and_block_match_serial_assembler(linearization):
    _cuda_only()
    scene = _two_bones_on_plate(2, linearization)
    solver = scene.vbd_solver
    attachment = solver.rigid_attachment
    shape = (attachment.n_free, solver._B)
    serial_force = qd.Vector.field(6, dtype=gs.qd_float, shape=shape)
    serial_hessian = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=shape)
    cooperative_force = qd.Vector.field(6, dtype=gs.qd_float, shape=shape)
    cooperative_hessian = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=shape)
    kernel_serial_contact_blocks(solver, attachment, serial_force, serial_hessian)
    contact_blocks_cooperative(0, solver, attachment, cooperative_force, cooperative_hessian)
    force_ref = qd_to_torch(serial_force)
    force_got = qd_to_torch(cooperative_force)
    hessian_ref = qd_to_torch(serial_hessian)
    hessian_got = qd_to_torch(cooperative_hessian)
    print(f"{linearization} force max abs {float((force_got - force_ref).abs().max()):.3e}, "
          f"block max abs {float((hessian_got - hessian_ref).abs().max()):.3e}")
    torch.testing.assert_close(force_got, force_ref, rtol=1e-11, atol=1e-10)
    torch.testing.assert_close(hessian_got, hessian_ref, rtol=1e-11, atol=1e-7)
    assert torch.count_nonzero(force_ref) > 0


@pytest.mark.parametrize("linearization", ["iterate", "substep"])
def test_two_coupled_bodies_keep_serial_gauss_seidel_pose_order(linearization):
    _cuda_only()
    serial_scene = _two_bones_on_plate(0, linearization)
    cooperative_scene = _two_bones_on_plate(0, linearization)
    serial = serial_scene.vbd_solver
    cooperative = cooperative_scene.vbd_solver
    serial_attachment = serial.rigid_attachment
    cooperative_attachment = cooperative.rigid_attachment
    assert serial_attachment.n_free == cooperative_attachment.n_free == 2
    torch.testing.assert_close(qd_to_torch(serial_attachment.link_pose.pos),
                               qd_to_torch(cooperative_attachment.link_pose.pos), rtol=0.0, atol=1e-10)
    joint_force = qd.Vector.field(6, dtype=gs.qd_float, shape=(serial_attachment.n_free, serial._B))
    kernel_joint_wrenches(serial, serial_attachment, joint_force)
    assert torch.count_nonzero(qd_to_torch(joint_force)) > 0, "the second body must read a coupled joint"
    kernel_serial_rigid_sweep(serial, serial_attachment)
    sweep_rigid_contact_cooperative(0, cooperative, cooperative_attachment)
    for name in ("pos", "quat"):
        reference = qd_to_torch(getattr(serial_attachment.link_pose, name))
        actual = qd_to_torch(getattr(cooperative_attachment.link_pose, name))
        print(f"{linearization} link {name} max abs {float((actual - reference).abs().max()):.3e}")
        torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-11)
    rv_reference = qd_to_torch(serial.contact.rv_pos)
    rv_actual = qd_to_torch(cooperative.contact.rv_pos)
    torch.testing.assert_close(rv_actual, rv_reference, rtol=1e-11, atol=1e-11)


def test_cooperative_prototype_rejects_cpu_without_fallback():
    if gs.backend == gs.cuda:
        pytest.skip("CPU boundary")
    scene = _two_bones_on_plate(0)
    solver = scene.vbd_solver
    attachment = solver.rigid_attachment
    shape = (attachment.n_free, solver._B)
    force = qd.Vector.field(6, dtype=gs.qd_float, shape=shape)
    hessian = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=shape)
    with pytest.raises(gs.GenesisException, match="requires CUDA"):
        contact_blocks_cooperative(0, solver, attachment, force, hessian)
    with pytest.raises(gs.GenesisException, match="requires CUDA"):
        sweep_rigid_contact_cooperative(0, solver, attachment)
