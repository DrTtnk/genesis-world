"""Native scene gates for finite-thickness rod contact."""
import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array


pytestmark = pytest.mark.precision("64")

def test_segment_contact_and_fixed_mesh_response(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001,gravity=(0.,0.,0.)),
        vbd_options=gs.options.VBDOptions(n_iterations=16,floor_height=-float('inf'),contact_margin=.002),
        viewer_options=gs.options.ViewerOptions(camera_pos=(.3,-.3,.4),camera_lookat=(.1,0.,.2)),
        show_viewer=show_viewer,
    )
    a = np.array([[-.02,0.,.2],[.02,0.,.2]])
    b = np.array([[0.,-.02,.207],[0.,.02,.207]])
    c = a+np.array([.2,0.,.014])
    rods=[]
    for vertices,frame,group in ((a,[2**-.5,0.,2**-.5,0.],1),(b,[2**-.5,-2**-.5,0.,0.],1),(c,[2**-.5,0.,2**-.5,0.],2)):
        rods.append(scene.add_entity(
            morph=gs.morphs.Rod(verts=vertices,frames=np.array([frame]),radius=.003),
            material=gs.materials.VBD.Rod(collision_group=group),
        ))
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(.02,.02,.02),pos=(.2,0.,.2),fixed=True),
        material=gs.materials.Rigid(),
    )
    solver=scene.sim.vbd_solver
    solver.add_contact_rule(1,1,100.,0.,.0001)
    solver.add_contact_rule(2,3,100.,0.,.0001)
    solver.add_rigid_collider(bone.links[0],3)
    scene.build()
    scene.step()
    pa,pb,pc=[tensor_to_array(rod.get_positions())[0] for rod in rods]
    assert pa[:,2].mean()<a[:,2].mean()
    assert pb[:,2].mean()>b[:,2].mean()
    assert pc[:,2].mean()>c[:,2].mean()
    assert abs((pa-a).mean(0)[2]+(pb-b).mean(0)[2])<1e-7
    assert min(solver._rod_contacts[0].gaps())>0
    snapshot=scene.get_state()
    scene.step()
    expected=solver.get_state(scene.sim.cur_substep_local)
    scene.reset(snapshot)
    scene.step()
    actual=solver.get_state(scene.sim.cur_substep_local)
    torch.testing.assert_close(actual.pos,expected.pos,atol=1e-12,rtol=0)
    for first,second in zip(actual.rod_states,expected.rod_states):
        for x,y in zip(first,second):
            torch.testing.assert_close(x,y,atol=1e-12,rtol=0)
    rods[0].set_pinned([True,True])
    rods[0].set_pin_targets(a+np.array([0.,0.,.03]))
    with pytest.raises(gs.GenesisException,match='Prescribed rod motion exceeds collision clearance'):
        scene.step()


def test_free_mesh_contact_reaction(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001,gravity=(0.,0.,0.)),
        rigid_options=gs.options.RigidOptions(enable_collision=False,integrator=gs.integrator.Euler),
        vbd_options=gs.options.VBDOptions(n_iterations=12,floor_height=-float('inf'),contact_margin=.002),
        viewer_options=gs.options.ViewerOptions(camera_pos=(.15,-.2,.3),camera_lookat=(0.,0.,.21)),
        show_viewer=show_viewer,
    )
    rest=np.array([[-.02,0.,.214],[.02,0.,.214]])
    rod=scene.add_entity(
        morph=gs.morphs.Rod(verts=rest,frames=np.array([[2**-.5,0.,2**-.5,0.]]),radius=.003),
        material=gs.materials.VBD.Rod(collision_group=1),
    )
    support=scene.add_entity(
        morph=gs.morphs.Box(size=(.001,.001,.001),pos=(-.02,0.,.214),fixed=True),
        material=gs.materials.Rigid(),
    )
    bone=scene.add_entity(
        morph=gs.morphs.Box(size=(.02,.02,.02),pos=(.005,0.,.2)),
        material=gs.materials.Rigid(),
    )
    rod.add_rigid_attachments([0],support.links[0])
    solver=scene.sim.vbd_solver
    solver.add_rigid_collider(bone.links[0],2)
    solver.add_contact_rule(1,2,100.,0.,.0001)
    scene.build()
    initial=scene.get_state()
    scene.step()
    assert bone.get_pos()[2]<.2
    assert tensor_to_array(rod.get_positions())[0,1,2]>.214
    assert min(solver._rod_contacts[0].gaps())>0
    expected=bone.get_pos().clone()
    scene.reset(initial)
    scene.step()
    torch.testing.assert_close(bone.get_pos(),expected,atol=1e-12,rtol=0)

    # Contact can end: an empty active set must contribute zero force/Hessian.
    for _ in range(15):
        scene.step()
    assert min(solver._rod_contacts[0].gaps()) > .002
