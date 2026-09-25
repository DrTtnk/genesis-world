"""Mixed material ownership and routed Hill loads on rod nodes."""
import numpy as np
import pytest
import torch

import genesis as gs
from genesis.engine.solvers.vbd_mtu import HillParameters, TissueAnchor
from genesis.utils.misc import tensor_to_array


pytestmark = pytest.mark.precision("64")

def test_rod_pulls_tissue_and_restores_state(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001, gravity=(0., 0., 0.)),
        vbd_options=gs.options.VBDOptions(n_iterations=48, floor_height=-float('inf')),
        viewer_options=gs.options.ViewerOptions(camera_pos=(.15, -.2, .3), camera_lookat=(0., 0., .22)),
        show_viewer=show_viewer,
    )
    # Tet first catches any accidental rod-local/global index equivalence.
    tissue = scene.add_entity(
        morph=gs.morphs.TetMesh(
            verts=np.array([[0.,0.,.2],[.005,0.,.2],[0.,.005,.2],[0.,0.,.205]]),
            elems=np.array([[0,1,2,3]]),
            faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]]),
        ),
        material=gs.materials.VBD.Muscle(E=1e4, nu=.3),
    )
    rest = np.array([[0.,0.,.205],[0.,0.,.245]])
    rod = scene.add_entity(
        morph=gs.morphs.Rod(verts=rest, frames=np.array([[1.,0.,0.,0.]]), radius=.003),
        material=gs.materials.VBD.Rod(collision_group=1),
    )
    solver = scene.sim.vbd_solver
    solver.add_tissue_attachment(TissueAnchor(rod,(0,0,0,0),(1.,0.,0.,0.)),
                                TissueAnchor(tissue,(0,1,2,3),(0.,0.,0.,1.)))
    bone = scene.add_entity(
        morph=gs.morphs.Box(size=(.002,.004,.01), pos=(.005,0.,.225), fixed=True),
        material=gs.materials.Rigid(),
    )
    solver.add_rigid_collider(bone.links[0], 2)
    solver.add_contact_rule(1, 2, 100., 0., .0001)
    solver.add_contact_rule(0, 2, 100., 0., .0001)
    scene.build()
    rod.set_pinned([False,True])
    # Preserve the 1 mm loading displacement with steps below the clearance bound.
    target=rest.copy();target[1,2]+=.0005
    rod.set_pin_targets(target)
    scene.step()
    target[1,2]+=.0005
    rod.set_pin_targets(target)
    initial=scene.get_state()
    scene.step()
    assert min(solver._rod_contacts[0].gaps()) > 0
    assert tissue.get_positions()[0,:,2].mean() > .20125
    assert (rod.get_positions()[0,0]-tissue.get_positions()[0,3]).norm() < 1e-5
    expected=solver.get_state(scene.sim.cur_substep_local)
    scene.reset(initial)
    scene.step()
    actual=solver.get_state(scene.sim.cur_substep_local)
    torch.testing.assert_close(actual.pos,expected.pos,rtol=0,atol=1e-12)


def test_hill_contracts_rod_with_one_force_owner(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001, gravity=(0.,0.,0.)),
        vbd_options=gs.options.VBDOptions(n_iterations=48, floor_height=-float('inf')),
        viewer_options=gs.options.ViewerOptions(camera_pos=(.15,-.2,.3), camera_lookat=(0.,0.,.22)),
        show_viewer=show_viewer,
    )
    rest=np.array([[0.,0.,.2],[0.,0.,.24]])
    rod=scene.add_entity(
        morph=gs.morphs.Rod(verts=rest, frames=np.array([[1.,0.,0.,0.]]), radius=.003),
        material=gs.materials.VBD.Rod(),
    )
    solver=scene.sim.vbd_solver
    solver.add_mtu([TissueAnchor(rod,(i,i,i,i),(1.,0.,0.,0.)) for i in range(2)],
                   HillParameters(f_max=.05,l_opt=.03,l_slack=.009,v_max=.3),activation0=1.)
    scene.build()
    solver.set_excitation([[1.]])
    scene.step()
    x=tensor_to_array(rod.get_positions())[0]
    assert np.linalg.norm(x[1]-x[0]) < .04
    np.testing.assert_allclose(x.mean(0),rest.mean(0),atol=1e-7,rtol=0)
    assert solver.mtu_state().tension[0,0]>0
    snapshot=scene.get_state()
    scene.step()
    expected=rod.get_positions().clone()
    expected_mtu=solver.mtu_state()
    scene.reset(snapshot)
    scene.step()
    torch.testing.assert_close(rod.get_positions(),expected,atol=1e-12,rtol=0)
    torch.testing.assert_close(solver.mtu_state().fibre_length,expected_mtu.fibre_length,atol=1e-12,rtol=0)


def test_transverse_bundle_transmits_shear(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001, gravity=(0.,0.,0.)),
        vbd_options=gs.options.VBDOptions(n_iterations=24, floor_height=-float('inf')),
        viewer_options=gs.options.ViewerOptions(camera_pos=(.15,-.2,.3), camera_lookat=(0.,0.,.22)),
        show_viewer=show_viewer,
    )
    rest=np.array([[0.,0.,.2],[0.,0.,.24]])
    rods=[]
    for offset in (0.,.01):
        rods.append(scene.add_entity(
            morph=gs.morphs.Rod(verts=rest+[offset,0.,0.], frames=np.array([[1.,0.,0.,0.]]), radius=.003),
            material=gs.materials.VBD.Rod(),
        ))
    solver=scene.sim.vbd_solver
    for i,j in ((0,0),(1,1),(0,1),(1,0)):
        solver.add_rod_bundle_link(rods[0],i,rods[1],j,100.,np.linalg.norm(rest[i]-rest[j]-[.01,0.,0.]))
    scene.build()
    rods[0].set_pinned([True,True])
    rods[0].set_pin_targets(rest+[0.,0.,.001])
    scene.step()
    assert rods[1].get_positions()[0,:,2].mean() > .22
    assert rods[1].get_positions()[0,:,2].mean() < .221
