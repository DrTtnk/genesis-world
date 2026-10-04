"""The rigid contact owner gate preserves the original per-pair wrench and block."""

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.vbd_contact import func_contact_link_terms, func_pair_terms
from genesis.utils.misc import qd_to_torch, tensor_to_array


@qd.func
def func_contact_link_terms_reference(f, i_l, i_b, origin, solver: qd.template(), contact: qd.template()):
    """The assembler before the owner check moved ahead of pair geometry."""
    force6 = qd.Vector.zero(gs.qd_float, 6)
    hessian6 = qd.Matrix.zero(gs.qd_float, 6, 6)
    base = contact.link_rv_offset[i_l]
    identity = qd.Matrix.identity(gs.qd_float, 3)
    for c in range(base, base + contact.link_active_n[i_l, i_b]):
        cv = contact.rv_cv[contact.link_active[c, i_b]]
        for slot in range(contact.cv_slot_offset[cv, i_b], contact.cv_slot_offset[cv + 1, i_b]):
            y, n, k, scale, slide, cvs, weights, own, curved = func_pair_terms(
                f, contact.cv_slot[slot, i_b], i_b, solver, contact
            )
            first = 4
            for j in qd.static(range(4)):
                if first == 4 and contact.cv_info[cvs[j]].kind == 1 and contact.cv_info[cvs[j]].owner == i_l:
                    first = j
            if curved and own == first:
                T = qd.Matrix.zero(gs.qd_float, 3, 6)
                for j in qd.static(range(4)):
                    if contact.cv_info[cvs[j]].kind == 1 and contact.cv_info[cvs[j]].owner == i_l:
                        r = contact.rv_pos[contact.cv_info[cvs[j]].ref, i_b] - origin
                        J = qd.Matrix.zero(gs.qd_float, 3, 6)
                        for row in qd.static(range(3)):
                            J[row, row] = 1.0
                        J[0, 4] = r[2]
                        J[0, 5] = -r[1]
                        J[1, 3] = -r[2]
                        J[1, 5] = r[0]
                        J[2, 3] = r[1]
                        J[2, 4] = -r[0]
                        T += weights[j] * J
                force6 += T.transpose() @ (-(y * n + scale * slide))
                nn = n.outer_product(n)
                hessian6 += T.transpose() @ (k * nn + scale * (identity - nn)) @ T
    return force6, hessian6


@qd.kernel
def kernel_compare_contact_link_terms(
    solver: qd.template(),
    contact: qd.template(),
    actual_force: qd.template(),
    actual_hessian: qd.template(),
    reference_force: qd.template(),
    reference_hessian: qd.template(),
):
    for i_l in range(contact.link_active_n.shape[0]):
        origin = gs.qd_vec3(0.0, 0.0, 0.0)
        actual_force[i_l], actual_hessian[i_l] = func_contact_link_terms(0, i_l, 0, origin, solver, contact)
        reference_force[i_l], reference_hessian[i_l] = func_contact_link_terms_reference(
            0, i_l, 0, origin, solver, contact
        )


def test_rigid_contact_owner_gate_preserves_wrench_and_curvature(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(
            n_iterations=10, floor_height=-1e3, contact_margin=2e-4, contact_k_max_ratio=4.0
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.1, -0.1, 0.08), camera_lookat=(0.0, 0.0, 0.0)
        ),
        show_viewer=show_viewer,
    )
    plate = scene.add_entity(
        morph=gs.morphs.Box(size=(0.06, 0.06, 0.004), pos=(0.0, 0.0, -0.002), fixed=True),
        material=gs.materials.Rigid(rho=1500.0),
    )
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(0.02, 0.01, 0.003), pos=(0.0, 0.0, 0.0018)),
        material=gs.materials.Rigid(rho=1500.0),
    )
    patch = scene.add_entity(
        morph=gs.morphs.Box(size=(0.003, 0.003, 0.003), pos=(0.025, 0.025, 0.0015), nobisect=False, maxvolume=4e-9),
        material=gs.materials.VBD.Muscle(E=1e5, nu=0.4, collision_group=2),
    )
    rest = tensor_to_array(patch.init_positions)
    patch.add_rigid_attachments([int(v) for v in np.argsort(rest[:, 2])[:4]], plate.links[0])
    solver = scene.vbd_solver
    solver.add_rigid_collider(plate.links[0], collision_group=0)
    solver.add_rigid_collider(bone.links[0], collision_group=1)
    solver.add_contact_rule(0, 1, stiffness=1e5, friction=0.1, thickness=2e-4)
    scene.build()

    for _ in range(15):
        scene.step()

    contact = solver.contact
    snapshot = contact.snapshot()
    assert int(snapshot.n_pt[0]) > 0
    assert int(snapshot.n_ee[0]) > 0
    triangles = qd_to_torch(contact.tri_cv)
    edges = qd_to_torch(contact.edge_cv)
    kind = qd_to_torch(contact.cv_info.kind)
    owner = qd_to_torch(contact.cv_info.owner)
    pt_cvs = torch.cat(
        (snapshot.pt_a[0, :snapshot.n_pt[0], None], triangles[snapshot.pt_b[0, :snapshot.n_pt[0]]]),
        dim=1,
    )
    ee_cvs = torch.cat(
        (edges[snapshot.ee_a[0, :snapshot.n_ee[0]]], edges[snapshot.ee_b[0, :snapshot.n_ee[0]]]),
        dim=1,
    )
    bone_idx = bone.links[0].idx
    plate_idx = plate.links[0].idx
    assert (kind[pt_cvs] == 1).all()
    assert (kind[ee_cvs] == 1).all()
    assert ((owner[pt_cvs] == plate_idx).sum(dim=1) >= 2).any()
    assert ((owner[ee_cvs] == bone_idx).sum(dim=1) >= 2).any()
    assert ((owner[ee_cvs] == plate_idx).sum(dim=1) >= 2).any()
    first_owner = torch.cat((owner[pt_cvs[:, 0]], owner[ee_cvs[:, 0]]))
    assert (first_owner == bone_idx).any()
    assert (first_owner == plate_idx).any()
    n_links = contact.link_active_n.shape[0]
    actual_force = qd.Vector.field(6, dtype=gs.qd_float, shape=n_links)
    actual_hessian = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=n_links)
    reference_force = qd.Vector.field(6, dtype=gs.qd_float, shape=n_links)
    reference_hessian = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=n_links)
    kernel_compare_contact_link_terms(
        solver, contact, actual_force, actual_hessian, reference_force, reference_hessian
    )
    torch.testing.assert_close(qd_to_torch(actual_force), qd_to_torch(reference_force), rtol=1e-12, atol=1e-12)
    # Branch placement changes a few last bits of the compiled matrix arithmetic.
    torch.testing.assert_close(qd_to_torch(actual_hessian), qd_to_torch(reference_hessian), rtol=1e-12, atol=1e-8)
    hessian = qd_to_torch(actual_hessian)
    assert torch.count_nonzero(hessian[bone_idx]) > 0
    assert torch.count_nonzero(hessian[plate_idx]) > 0
