"""Reciprocal tapered-rod contact with deformable triangle surfaces."""
import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch

pytestmark=pytest.mark.precision('64')

@pytest.mark.parametrize('kind', ['tet','shell'])
def test_surface_contact_reacts_and_replays(kind, show_viewer):
    scene=gs.Scene(
        sim_options=gs.options.SimOptions(dt=.0001,gravity=(0.,0.,0.)),
        vbd_options=gs.options.VBDOptions(n_iterations=16,floor_height=-float('inf'),contact_margin=.001),
        show_viewer=show_viewer,
    )
    vertices=np.array([[0.,-.02,.2],[0.,.02,.2],[0.,0.,.24],[-.02,0.,.22]])
    if kind=='tet':
        tissue=scene.add_entity(morph=gs.morphs.TetMesh(verts=vertices,elems=np.array([[0,2,1,3]]),
            faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])),
            material=gs.materials.VBD.Muscle(E=1e4,nu=.3,collision_group=2))
    else:
        tissue=scene.add_entity(morph=gs.morphs.TriMesh(verts=vertices[:3],faces=np.array([[0,1,2]])),
            material=gs.materials.VBD.Shell(E=1e4,nu=.3,thickness=.001,collision_group=2))
    rod=scene.add_entity(morph=gs.morphs.Rod(verts=np.array([[.0014,0.,.215],[.0014,0.,.225]]),
        frames=np.array([[1.,0.,0.,0.]]),radius=.001),material=gs.materials.VBD.Rod(collision_group=1))
    solver=scene.sim.vbd_solver
    solver.add_contact_rule(1,2,1.,0.,.0001)
    scene.build()
    snapshot=scene.get_state()
    before=solver.get_state(scene.sim.cur_substep_local).pos.clone()
    scene.step()
    after=solver.get_state(scene.sim.cur_substep_local).pos.clone()
    assert rod.get_positions()[0,:,0].mean() > .0014
    assert tissue.get_positions()[0,:,0].mean() < vertices[:tissue.n_vertices,0].mean()
    assert min(solver._rod_contacts[0].gaps()) > 0
    mass=qd_to_torch(solver.verts_info.mass)
    assert ((after-before)*mass[None,:,None]).sum(1).norm() < 1e-10
    scene.reset(snapshot)
    scene.step()
    torch.testing.assert_close(solver.get_state(scene.sim.cur_substep_local).pos,after,atol=1e-12,rtol=0)

    tissue.set_pinned([True] * tissue.n_vertices)
    target=vertices[:tissue.n_vertices].copy()
    target[:,0]+=.002
    tissue.set_pin_targets(target)
    with pytest.raises(ValueError,match='Prescribed surface motion exceeds collision clearance'):
        scene.step()


def test_surface_constraint_duals_use_accepted_motion(show_viewer):
    scene=gs.Scene(
        sim_options=gs.options.SimOptions(dt=.01,gravity=(0.,0.,0.)),
        vbd_options=gs.options.VBDOptions(n_iterations=1,floor_height=-float('inf'),contact_margin=.001),
        show_viewer=show_viewer,
    )
    vertices=np.array([[0.,-.02,.2],[0.,.02,.2],[0.,0.,.24],[-.02,0.,.22]])
    tissue=scene.add_entity(morph=gs.morphs.TetMesh(verts=vertices,elems=np.array([[0,2,1,3]]),
        faces=np.array([[0,1,2],[0,3,1],[1,3,2],[2,3,0]])),
        material=gs.materials.VBD.Muscle(E=100.,nu=.3,collision_group=2))
    tissue.add_distance_constraints([[0,3]],lo=.025,hi=.025)
    tissue.add_angle_constraints([[0,3,1,3]],60.,60.)
    rod=scene.add_entity(morph=gs.morphs.Rod(verts=np.array([[.0014,0.,.215],[.0014,0.,.225]]),
        frames=np.array([[1.,0.,0.,0.]]),radius=.001),material=gs.materials.VBD.Rod(collision_group=1))
    solver=scene.sim.vbd_solver
    solver.add_contact_rule(1,2,1.,0.,.0001)
    scene.build()
    rod.set_pinned([True,True])
    tissue.set_pinned([False,False,False,True])
    distance_k=qd_to_torch(solver.cons.k,transpose=True).clone()[0,0]
    angle_k=qd_to_torch(solver.acons.k,transpose=True).clone()[0,0]
    initial_lambda=qd_to_torch(solver.cons.lam_hi,transpose=True).clone()[0,0]
    initial_angle_lambda=qd_to_torch(solver.acons.lam_hi,transpose=True).clone()[0,0]
    angle_k0=qd_to_torch(solver.acons_info.k0).clone()[0]
    scene.step()
    x=tissue.get_positions()[0]
    # Conservative clipping keeps the largest surface move below 0.9 clearance.
    assert (x-x.new_tensor(vertices)).norm(dim=1).max() < .0003
    distance=(x[0]-x[3]).norm()
    u,v=x[0]-x[3],x[1]-x[3]
    cosine=(u*v).sum()/(u.norm()*v.norm())
    # The native substep decays the existing build/warm-start state before its sweep.
    expected_distance=.95*.99*initial_lambda+solver._constraint_dual_relaxation*torch.clamp(.99*distance_k,min=solver._k_start)*(distance-.025)
    expected_angle=.95*.99*initial_angle_lambda+solver._constraint_dual_relaxation*torch.maximum(.99*angle_k,angle_k0)*(cosine-.5)
    torch.testing.assert_close(qd_to_torch(solver.cons.lam_hi,transpose=True)[0,0],expected_distance,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(qd_to_torch(solver.acons.lam_hi,transpose=True)[0,0],expected_angle,rtol=1e-10,atol=1e-12)
