"""Reference contact for tapered rods and rigid or deformable triangle surfaces.

The envelope is the convex hull of each segment's endpoint spheres. A logarithmic
barrier acts on surface clearance. Closest witnesses are convex minimizations;
Jacobian evaluation uses their envelope derivative. Exhaustive pair ownership
and conservative motion bounds make this a small-scene correctness reference.
"""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from scipy.optimize import minimize

import igl
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.vbd_rod import quat_matrix, retract
from genesis.utils.misc import qd_to_torch, tensor_to_array


class TaperedWitness(NamedTuple):
    weights_a: np.ndarray
    weights_b: np.ndarray
    gap: float
    lower_bound: float


def closest_tapered(a, ra, b, rb):
    """Return closest weights, gap estimate and a supporting-plane lower bound in metres."""
    matrix = np.column_stack((a[1]-a[0], -(b[1:]-b[0]).T))
    offset = a[0]-b[0]
    slope = np.r_[ra[1]-ra[0], rb[1:]-rb[0]]
    radius = ra[0]+rb[0]
    length_scale = max(np.linalg.norm(matrix,axis=0).max(),np.linalg.norm(offset),radius)
    matrix,offset,slope,radius = matrix/length_scale,offset/length_scale,slope/length_scale,radius/length_scale

    def objective(t):
        d = offset+matrix@t
        length = np.linalg.norm(d)
        if length == 0:
            return -radius-slope@t, -slope
        return length-radius-slope@t, matrix.T@(d/length)-slope

    width = matrix.shape[1]
    constraints = ()
    if len(b) == 3:
        constraints = ({'type':'ineq', 'fun':lambda t:1-t[1:].sum(),
                        'jac':lambda t:np.array([0.,-1.,-1.])},)
    result = minimize(objective, np.full(width, 1/width), jac=True, method='SLSQP',
                      bounds=[(0.,1.)]*width, constraints=constraints,
                      options={'ftol':1e-13,'maxiter':100})
    if not result.success:
        raise RuntimeError(f'Tapered contact witness failed: {result.message}')
    wa = np.array([1-result.x[0],result.x[0]])
    wb = np.r_[1-result.x[1:].sum(),result.x[1:]]
    value, gradient = objective(result.x)
    simplex_min = min(0., *gradient[1:]) if len(b)==3 else min(0., gradient[1])
    lower = value-gradient@result.x+min(0., gradient[0])+simplex_min
    return TaperedWitness(wa,wb,value*length_scale,lower*length_scale)


def contact_residual(gap, stiffness, margin):
    """Square-root barrier residual, with energy in joules and stiffness in N/m."""
    if bool((gap<=0).any()):
        raise ValueError('Rod contact surfaces intersect or have zero clearance.')
    active = gap < margin
    g = gap[active]
    return (2*stiffness)**.5*(margin-g)*torch.sqrt(-torch.log(g/margin))


@dataclass(frozen=True)
class RodContactPair:
    a: int
    segment_a: int
    b: int
    segment_b: int
    is_mesh: bool
    stiffness: float
    clearance: float


@dataclass(frozen=True)
class RodCollider:
    link: int
    vertices: torch.Tensor
    faces: np.ndarray


@dataclass(frozen=True)
class RodSurface:
    vertices: torch.Tensor
    faces: np.ndarray
    closed: bool


class RodContact:
    def __init__(self, solver, rules):
        self.solver = solver
        self.margin = max(rule[4] for rule in rules) if solver._contact_margin is None else solver._contact_margin
        if not 0 < self.margin < float("inf"):
            gs.raise_exception("Rod contact margin must be finite and positive.")
        self.models = solver._rod_models
        n_links = solver.sim.rigid_solver.n_links
        self.force = qd.Vector.field(6, dtype=gs.qd_float, shape=n_links) if n_links else ()
        self.hessian = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=n_links) if n_links else ()
        self.pairs = []
        self.colliders = []
        self.positions = []
        self.surface_positions = torch.empty((0, 3), dtype=gs.tc_float)
        self.surface_force = qd.Vector.field(3, dtype=gs.qd_float, shape=solver.n_vertices)
        self.surface_hessian = qd.Matrix.field(3, 3, dtype=gs.qd_float, shape=solver.n_vertices)
        self.surface_reference = qd.Vector.field(3, dtype=gs.qd_float, shape=solver.n_vertices)
        self.link_pos = torch.empty((0,3),dtype=gs.tc_float)
        self.link_quat = torch.empty((0,4),dtype=gs.tc_float)
        groups = [entity.material.collision_group for entity in solver._rod_entities]
        collider_groups = []
        for link,group,regions in solver._rigid_colliders:
            if regions is not None:
                gs.raise_exception('Rod reference contact requires whole closed collider surfaces.')
            vertices,faces = [],[]
            offset = 0
            for geom in link.geoms:
                local = gu.transform_by_trans_quat(geom.init_verts,geom.init_pos,geom.init_quat)
                vertices.append(local)
                faces.append(geom.init_faces+offset)
                offset += len(local)
            local = np.concatenate(vertices)
            triangles = np.concatenate(faces)
            # Each mesh edge must have two incident faces for an inside/outside test.
            edges = np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1)
            if not np.all(np.unique(edges,axis=0,return_counts=True)[1]==2):
                gs.raise_exception('Rod rigid colliders require closed manifold triangle meshes.')
            self.colliders.append(RodCollider(link.idx,torch.tensor(local,dtype=gs.tc_float),triangles))
            collider_groups.append(group)
        for entity in solver._entities:
            if entity in solver._rod_entities:
                continue
            faces = entity.tris if entity.n_triangles else igl.boundary_facets(entity.elems)[0]
            edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
            closed = bool(np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2))
            self.colliders.append(RodSurface(torch.arange(entity.v_start, entity.v_start + entity.n_vertices), faces, closed))
            collider_groups.append(entity.material.collision_group)
        seen = set()
        for ga,gb,k,friction,clearance in rules:
            key=tuple(sorted((ga,gb)))
            if key in seen:
                gs.raise_exception('Duplicate rod contact rule.')
            seen.add(key)
            if friction:
                gs.raise_exception('Rod reference contact currently supports frictionless normal response.')
            for ia,model in enumerate(self.models):
                for ib in range(ia,len(self.models)):
                    if tuple(sorted((groups[ia],groups[ib])))!=key:
                        continue
                    for sa in range(len(model.length)):
                        for sb in range(len(self.models[ib].length)):
                            if ia==ib and sb<=sa+1:
                                continue
                            self.pairs.append(RodContactPair(ia,sa,ib,sb,False,k,clearance))
                for ib,group in enumerate(collider_groups):
                    if tuple(sorted((groups[ia],group)))==key:
                        for sa in range(len(model.length)):
                            self.pairs.append(RodContactPair(ia,sa,ib,0,True,k,clearance))
        if not self.pairs:
            gs.raise_exception('Rod contact rules select no segment pairs.')

    def mesh(self, index):
        collider = self.colliders[index]
        if isinstance(collider, RodSurface):
            return self.surface_positions[collider.vertices]
        return collider.vertices@quat_matrix(self.link_quat[collider.link]).T+self.link_pos[collider.link]

    def witness(self, pair, positions, scales):
        a = tensor_to_array(positions[pair.a][pair.segment_a:pair.segment_a+2])
        ra = self.models[pair.a].radius*tensor_to_array(scales[pair.a][pair.segment_a:pair.segment_a+2])
        if not pair.is_mesh:
            b = tensor_to_array(positions[pair.b][pair.segment_b:pair.segment_b+2])
            rb = self.models[pair.b].radius*tensor_to_array(scales[pair.b][pair.segment_b:pair.segment_b+2])
            witness = closest_tapered(a,ra,b,rb)
            return witness.weights_a,witness.weights_b,witness.lower_bound-pair.clearance,np.array([pair.segment_b,pair.segment_b+1])
        mesh = tensor_to_array(self.mesh(pair.b))
        faces = self.colliders[pair.b].faces
        corners = mesh[faces]
        separation = np.maximum(np.maximum(corners.min(1)-a.max(0),a.min(0)-corners.max(1)),0)
        lower = np.linalg.norm(separation,axis=1)-ra.max()
        best = float('inf')
        best_result = ()
        lower_bound = float('inf')
        for index in np.argsort(lower):
            if lower[index]>=best:
                lower_bound = min(lower_bound,lower[index])
                break
            witness = closest_tapered(a,ra,corners[index],np.zeros(3))
            lower_bound = min(lower_bound,witness.lower_bound)
            if witness.gap<best:
                best = witness.gap
                best_result = witness.weights_a,witness.weights_b,faces[index]
        wa,wb,indices = best_result
        return wa,wb,lower_bound-pair.clearance,indices

    def gaps(self):
        scales = [model.scale for model in self.models]
        return [self.witness(pair,self.positions,scales)[2] for pair in self.pairs]

    def validate(self):
        if min(self.gaps())<=0:
            raise ValueError('Rod contact surfaces intersect or have zero clearance.')
        for pair in self.pairs:
            if pair.is_mesh and (isinstance(self.colliders[pair.b], RodCollider) or self.colliders[pair.b].closed):
                mesh = tensor_to_array(self.mesh(pair.b))
                points = tensor_to_array(self.positions[pair.a][pair.segment_a:pair.segment_a+2])
                winding = igl.winding_number(mesh,self.colliders[pair.b].faces,points)
                if np.any(np.abs(winding)>.5):
                    raise ValueError('A rod centreline lies inside a closed collider.')

    def begin(self, positions):
        self.surface_positions = positions.clone()
        self.positions = [positions[e.v_start:e.v_start+e.n_vertices] for e in self.solver._rod_entities]
        rigid = self.solver.sim.rigid_solver
        if self.solver.has_rigid_attachment:
            a = self.solver.rigid_attachment
            links = qd_to_torch(a.free_info.link)[:a.n_free]
            pos = qd_to_torch(a.link_state.pos, transpose=True)
            quat = qd_to_torch(a.link_state.quat, transpose=True)
            pos.copy_(qd_to_torch(a.link_state.previous_pos, transpose=True))
            quat.copy_(qd_to_torch(a.link_state.previous_quat, transpose=True))
            poses = qd_to_torch(a.link_pose.pos, transpose=True)
            quats = qd_to_torch(a.link_pose.quat, transpose=True)
            poses[0,links] = pos[0,:a.n_free]
            quats[0,links] = quat[0,:a.n_free]
            self.link_pos, self.link_quat = poses[0].clone(), quats[0].clone()
        elif rigid.is_active:
            self.link_pos = torch.tensor(tensor_to_array(rigid.get_links_pos()),dtype=gs.tc_float).reshape(-1,3)
            self.link_quat = torch.tensor(tensor_to_array(rigid.get_links_quat()),dtype=gs.tc_float).reshape(-1,4)
        self.validate()
        self.start_surface_positions = self.surface_positions.clone()
        self.start_positions = [p.clone() for p in self.positions]
        self.start_scales = [m.scale.clone() for m in self.models]
        self.start_link_pos = self.link_pos.clone()
        self.start_link_quat = self.link_quat.clone()

    def validate_sweep(self):
        """Certify positive clearance on the simultaneous substep interpolation."""
        end_pos, end_quat = self.link_pos, self.link_quat
        end_surface = self.surface_positions
        aligned = torch.where((end_quat*self.start_link_quat).sum(1,keepdim=True)<0,-end_quat,end_quat)
        bounds = []
        for pair in self.pairs:
            sa = slice(pair.segment_a,pair.segment_a+2)
            bound = (self.positions[pair.a][sa]-self.start_positions[pair.a][sa]).norm(dim=1).max()
            bound += self.models[pair.a].radius*(self.models[pair.a].scale[sa]-self.start_scales[pair.a][sa]).abs().max()
            if pair.is_mesh:
                collider = self.colliders[pair.b]
                if isinstance(collider, RodSurface):
                    bound += (end_surface[collider.vertices] - self.start_surface_positions[collider.vertices]).norm(dim=1).max()
                else:
                    i = collider.link
                    angle = 4*torch.asin(torch.clamp((aligned[i]-self.start_link_quat[i]).norm()/2,max=1.))
                    # Nlerp angular speed is bounded by 4*tan(angle/4), in radians per unit time.
                    bound += (end_pos[i]-self.start_link_pos[i]).norm()+4*torch.tan(angle/4)*collider.vertices.norm(dim=1).max()
            else:
                sb = slice(pair.segment_b,pair.segment_b+2)
                bound += (self.positions[pair.b][sb]-self.start_positions[pair.b][sb]).norm(dim=1).max()
                bound += self.models[pair.b].radius*(self.models[pair.b].scale[sb]-self.start_scales[pair.b][sb]).abs().max()
            bounds.append(float(bound))
        time = 0.
        try:
            for _ in range(self.solver._contact_ccd_iterations):
                positions = [a+time*(b-a) for a,b in zip(self.start_positions,self.positions)]
                scales = [a+time*(m.scale-a) for a,m in zip(self.start_scales,self.models)]
                self.link_pos = self.start_link_pos+time*(end_pos-self.start_link_pos)
                self.link_quat = self.start_link_quat+time*(aligned-self.start_link_quat)
                self.link_quat = self.link_quat/self.link_quat.norm(dim=1,keepdim=True)
                self.surface_positions = self.start_surface_positions + time * (end_surface - self.start_surface_positions)
                step = 1.-time
                for pair,bound in zip(self.pairs,bounds):
                    gap = self.witness(pair,positions,scales)[2]
                    if gap<=0:
                        raise RuntimeError('Rod contact swept path intersects; reduce the timestep.')
                    if bound:
                        step = min(step,.9*gap/bound)
                if step>=1.-time:
                    return
                if time+step==time:
                    break
                time += step
        finally:
            self.link_pos,self.link_quat = end_pos,end_quat
            self.surface_positions = end_surface
        raise RuntimeError('Rod contact swept clearance could not be certified; reduce the timestep.')

    def rigid_terms(self):
        forces = qd_to_torch(self.force)
        hessians = qd_to_torch(self.hessian)
        forces.zero_()
        hessians.zero_()
        for collider_index, collider in enumerate(self.colliders):
            if isinstance(collider, RodSurface):
                continue
            pairs = [p for p in self.pairs if p.is_mesh and p.b==collider_index]
            witnesses = [self.witness(p,self.positions,[m.scale for m in self.models]) for p in pairs]
            pos, quat = self.link_pos[collider.link], self.link_quat[collider.link]

            def residual(delta):
                mesh = collider.vertices@quat_matrix(retract(quat,delta[3:])).T+pos+delta[:3]
                values = []
                for pair,(wa,wb,_,indices) in zip(pairs,witnesses):
                    wa,wb = pos.new_tensor(wa),pos.new_tensor(wb)
                    a = (wa[:,None]*self.positions[pair.a][pair.segment_a:pair.segment_a+2]).sum(0)
                    b = (wb[:,None]*mesh[indices]).sum(0)
                    radius = self.models[pair.a].radius*(wa*self.models[pair.a].scale[pair.segment_a:pair.segment_a+2]).sum()
                    values.append(contact_residual(((a-b).norm()-radius-pair.clearance)[None],pair.stiffness,self.margin))
                return torch.cat(values) if values else pos.new_empty(0)

            delta = pos.new_zeros(6)
            r = residual(delta)
            if not r.numel():
                continue
            J = torch.autograd.functional.jacobian(residual,delta,vectorize=True)
            forces[collider.link] += -J.T@r
            hessians[collider.link] += J.T@J

    def accept_rigid_motion(self):
        a = self.solver.rigid_attachment
        poses = qd_to_torch(a.link_pose.pos,transpose=True)
        quats = qd_to_torch(a.link_pose.quat,transpose=True)
        proposed_pos,proposed_quat = poses[0].clone(),quats[0].clone()
        proposed_quat = torch.where((proposed_quat*self.link_quat).sum(1,keepdim=True)<0,-proposed_quat,proposed_quat)
        gaps = self.gaps()
        for power in range(24):
            fraction = 2.**(-power)
            pos = self.link_pos+fraction*(proposed_pos-self.link_pos)
            quat = self.link_quat+fraction*(proposed_quat-self.link_quat)
            quat = quat/quat.norm(dim=1,keepdim=True)
            is_safe = True
            for pair,gap in zip(self.pairs,gaps):
                if not pair.is_mesh:
                    continue
                collider = self.colliders[pair.b]
                if isinstance(collider, RodSurface):
                    continue
                index = collider.link
                angle = 4*torch.asin(torch.clamp((quat[index]-self.link_quat[index]).norm()/2,max=1.))
                bound = (pos[index]-self.link_pos[index]).norm()+angle*collider.vertices.norm(dim=1).max()
                if bool(bound>=.9*gap):
                    is_safe = False
                    break
            if is_safe:
                self.link_pos,self.link_quat = pos,quat
                poses[0].copy_(pos)
                quats[0].copy_(quat)
                links = qd_to_torch(a.free_info.link)[:a.n_free]
                state_pos = qd_to_torch(a.link_state.pos,transpose=True)
                state_quat = qd_to_torch(a.link_state.quat,transpose=True)
                state_pos[0,:a.n_free] = pos[links]
                state_quat[0,:a.n_free] = quat[links]
                return
        raise RuntimeError('Rod contact rigid displacement bound exhausted.')

    def surface_terms(self):
        """Assemble frozen-witness force and GN blocks at the current surface iterate."""
        forces = qd_to_torch(self.surface_force)
        hessians = qd_to_torch(self.surface_hessian)
        reference = qd_to_torch(self.surface_reference)
        forces.zero_()
        hessians.zero_()
        reference.copy_(self.surface_positions)
        for pair in self.pairs:
            if not pair.is_mesh or not isinstance(self.colliders[pair.b], RodSurface):
                continue
            collider = self.colliders[pair.b]
            wa, wb, _, indices = self.witness(pair, self.positions, [m.scale for m in self.models])
            wa, wb = self.surface_positions.new_tensor(wa), self.surface_positions.new_tensor(wb)
            a = (wa[:, None] * self.positions[pair.a][pair.segment_a:pair.segment_a + 2]).sum(0)
            vertices = collider.vertices[indices]
            b = (wb[:, None] * self.surface_positions[vertices]).sum(0)
            radius = self.models[pair.a].radius * (wa * self.models[pair.a].scale[pair.segment_a:pair.segment_a + 2]).sum()
            distance = (a - b).norm()
            gap = distance - radius - pair.clearance
            if gap <= 0:
                raise ValueError('Rod contact surface iterate intersects.')
            if gap >= self.margin:
                continue
            logarithm = -torch.log(gap / self.margin)
            factor = (2 * pair.stiffness) ** .5
            residual = factor * (self.margin - gap) * torch.sqrt(logarithm)
            derivative = factor * (-torch.sqrt(logarithm) - (self.margin - gap) / (2 * gap * torch.sqrt(logarithm)))
            jacobian = -derivative * wb[:, None] * (a - b) / distance
            forces.index_add_(0, vertices, -residual * jacobian)
            hessians.index_add_(0, vertices, jacobian[:, :, None] * jacobian[:, None, :])

    def accept_surface_motion(self, proposed, prescribed=False):
        """Bound surface displacement by clearance before committing native vertices."""
        fraction = 1.
        for pair in self.pairs:
            if not pair.is_mesh or not isinstance(self.colliders[pair.b], RodSurface):
                continue
            collider = self.colliders[pair.b]
            gap = self.witness(pair, self.positions, [m.scale for m in self.models])[2]
            if gap <= 0:
                raise ValueError('Rod contact surface iterate intersects.')
            bound = (proposed[collider.vertices] - self.surface_positions[collider.vertices]).norm(dim=1).max()
            if bound > 0:
                fraction = min(fraction, float(.9 * gap / bound))
        if prescribed and fraction < 1.:
            raise ValueError('Prescribed surface motion exceeds collision clearance; reduce its step.')
        proposed.copy_(self.surface_positions + fraction * (proposed - self.surface_positions))
        self.surface_positions = proposed.clone()

    def view(self,index):
        return RodContactView(self,index)


class RodContactView:
    def __init__(self,contact,index):
        self.contact = contact
        self.index = index
        self.pairs = [p for p in contact.pairs if p.a==index or (not p.is_mesh and p.b==index)]
        self.witnesses = []

    def refresh(self,x,s):
        positions = list(self.contact.positions)
        scales = [model.scale for model in self.contact.models]
        positions[self.index],scales[self.index] = x,s
        self.witnesses = [self.contact.witness(pair,positions,scales) for pair in self.pairs]

    def residual(self,x,s):
        c = self.contact
        positions = list(c.positions)
        scales = [model.scale for model in c.models]
        positions[self.index],scales[self.index] = x,s
        residuals = []
        for pair,(wa,wb,_,indices) in zip(self.pairs,self.witnesses):
            wa,wb = x.new_tensor(wa),x.new_tensor(wb)
            a = (wa[:,None]*positions[pair.a][pair.segment_a:pair.segment_a+2]).sum(0)
            radius = c.models[pair.a].radius*(wa*scales[pair.a][pair.segment_a:pair.segment_a+2]).sum()
            if pair.is_mesh:
                b = (wb[:,None]*c.mesh(pair.b)[indices]).sum(0)
            else:
                b = (wb[:,None]*positions[pair.b][indices]).sum(0)
                radius = radius+c.models[pair.b].radius*(wb*scales[pair.b][indices]).sum()
            gap = (a-b).norm()-radius-pair.clearance
            residuals.append(contact_residual(gap[None],pair.stiffness,c.margin))
        return torch.cat(residuals) if residuals else x.new_empty(0)

    def allowed(self,x,s,candidate_x,candidate_s):
        c = self.contact
        if len(self.witnesses) != len(self.pairs):
            raise RuntimeError('Refresh rod contact witnesses before testing a block step.')
        for pair, witness in zip(self.pairs, self.witnesses):
            gap = witness[2]
            if gap<=0:
                raise ValueError('Rod contact iterate intersects.')
            bound = x.new_tensor(0.)
            segments = [(pair.a,pair.segment_a)]
            if not pair.is_mesh:
                segments.append((pair.b,pair.segment_b))
            for index,segment in segments:
                if index==self.index:
                    sl = slice(segment,segment+2)
                    bound = bound+(candidate_x[sl]-x[sl]).norm(dim=-1).max()+c.models[index].radius*(candidate_s[sl]-s[sl]).abs().max()
            if bound>=.9*gap:
                return False
        return True
