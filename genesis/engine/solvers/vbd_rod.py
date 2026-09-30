"""Passive VIPER energies and a reference block-descent implementation.

Angles et al., VIPER (2019), equations 11-20 and 40. Segment terms use midpoint
quadrature; curvature uses interior dual cells. Translation uses endpoint mass
lumping. Cross-section inertia integrates the two material directors exactly.
The Torch implementation is a correctness reference for small rod-only scenes.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class RodParameters:
    density: float
    stretch_x: float
    stretch_y: float
    stretch_z: float
    volume: float
    surface_bend: float

    def __post_init__(self):
        values = (self.density, self.stretch_x, self.stretch_y, self.stretch_z, self.volume)
        if any(not 0 < value < float("inf") for value in values):
            raise ValueError("Rod density and stiffnesses must be finite and positive.")
        if not 0 <= self.surface_bend < float("inf"):
            raise ValueError("Rod surface bending stiffness must be finite and nonnegative.")


class RodAttachment(NamedTuple):
    vertices: torch.Tensor
    weights: torch.Tensor
    target: torch.Tensor
    stiffness: torch.Tensor

    def residual(self, x):
        point = (self.weights[:, :, None] * x[self.vertices]).sum(1)
        return (self.stiffness.sqrt()[:, None] * (point - self.target)).reshape(-1)


class RodState(NamedTuple):
    scale: torch.Tensor
    quat: torch.Tensor
    director_velocity: torch.Tensor


def quat_multiply(a, b):
    """Hamilton product of scalar-first unit quaternions."""
    return torch.cat(
        (
            a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True),
            a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.linalg.cross(a[..., 1:], b[..., 1:]),
        ),
        dim=-1,
    )


def quat_matrix(q):
    """Rotation matrix from a scalar-first unit quaternion."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def retract(q, delta):
    """World-frame Cayley rotation with the identity tangent derivative."""
    dq = torch.cat((torch.ones_like(delta[..., :1]), delta / 2), dim=-1)
    return quat_multiply(dq / torch.linalg.vector_norm(dq, dim=-1, keepdim=True), q)


class RodModel:
    def __init__(self, positions, quaternions, radius, parameters):
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("Rod positions must have shape (n >= 2, 3).")
        if quaternions.shape != (len(positions) - 1, 4):
            raise ValueError("Rod needs one scalar-first quaternion per segment.")
        if not bool(torch.isfinite(positions).all() & torch.isfinite(quaternions).all()):
            raise ValueError("Rod rest state must be finite.")
        if not 0 < radius < float("inf"):
            raise ValueError("Rod reference radius must be finite and positive.")
        if not torch.allclose(quaternions.square().sum(-1), torch.ones_like(quaternions[:, 0])):
            raise ValueError("Rod frames must be unit quaternions.")
        self.rest_pos = positions.clone()
        self.rest_quat = quaternions.clone()
        self.radius = radius
        self.parameters = parameters
        edge = positions[1:] - positions[:-1]
        self.length = torch.linalg.vector_norm(edge, dim=-1)
        if not bool((self.length > 0).all()):
            raise ValueError("Rod segments must have positive rest length.")
        self.rest_director = quat_matrix(quaternions)[..., 2]
        # A segment's direction carries the rounding of its end positions over its length, about eps |x| / L; in
        # float32 a half-millimetre segment 0.1 m from the origin is already 1e-5 off. Allow what the precision
        # can represent, and never less than the fixed bound this check had (allclose, atol = rtol = 1e-6).
        eps = torch.finfo(positions.dtype).eps
        reach = torch.maximum(positions[1:].abs().amax(-1), positions[:-1].abs().amax(-1))
        fixed = 1e-6 + 1e-6 * self.rest_director.abs()
        tol = torch.maximum(fixed, (8.0 * eps * reach / self.length)[:, None])
        if not bool(((edge / self.length[:, None] - self.rest_director).abs() <= tol).all()):
            raise ValueError("Rest frame third axis must align with its centreline segment.")
        self.dual_length = (self.length[1:] + self.length[:-1]) / 2
        self.area = torch.pi * radius**2
        self.moment = torch.pi * radius**4 / 4
        segment_mass = parameters.density * self.area * self.length
        self.mass = torch.cat((segment_mass[:1], segment_mass[:-1] + segment_mass[1:], segment_mass[-1:])) / 2
        self.rest_volume = self.area * self.length.sum()
        self.scale = torch.ones_like(positions[:, 0])
        self.quat = quaternions.clone()
        self.rest_curvature = self.curvature(quaternions)
        self.director_velocity = torch.zeros_like(self.directors(self.scale, self.quat))
        self.previous_directors = self.directors(self.scale, self.quat)
        self.predicted_directors = self.previous_directors.clone()
        self.predicted_pos = positions.clone()
        self.dt = 1.0
        # Rest geometry and material are immutable; these weights are constant
        # across every residual, Jacobian seed and backtracking trial.
        self.position_weight = self.mass.sqrt()[:, None]
        self.section_weight = (parameters.density * self.moment * self.length).sqrt()[:, None, None]
        self.stretch_weight = (self.area * parameters.stretch_z * self.length).sqrt()[:, None]
        self.radius_weight = (self.area * (parameters.stretch_x + parameters.stretch_y) * self.length).sqrt()
        self.radius_gradient_weight = (self.moment * (parameters.stretch_x + parameters.stretch_y) * self.length).sqrt()
        bending_weight = positions.new_tensor(
            (4 * self.moment * parameters.stretch_z,
             4 * self.moment * parameters.stretch_z,
             4 * self.moment * (parameters.stretch_x + parameters.stretch_y))
        )
        self.bending_weight = (self.dual_length[:, None] * bending_weight).sqrt()
        self.volume_weight = (self.area * parameters.volume * self.length).sqrt()[:, None]
        self.volume_bending_weight = (2 * self.moment * parameters.volume * self.dual_length).sqrt()[:, None]
        self.surface_weight = (self.moment * parameters.surface_bend * self.dual_length).sqrt()

    def curvature(self, q):
        conjugate = torch.cat((q[:-1, :1], -q[:-1, 1:]), dim=-1)
        relative = quat_multiply(conjugate, q[1:])
        # The shortest quaternion branch is invariant to either frame's sign.
        relative = torch.where(relative[:, :1] < 0, -relative, relative)
        return 2 * relative[:, 1:] / self.dual_length[:, None]

    def directors(self, s, q):
        return ((s[:-1] + s[1:]) / 2)[:, None, None] * quat_matrix(q)[..., :2]

    def axial_residual(self, x, s, q):
        tangent = (x[1:] - x[:-1]) / self.length[:, None]
        director = quat_matrix(q)[..., 2]
        mid = (s[:-1] + s[1:]) / 2
        return (
            self.stretch_weight * (tangent - director),
            self.volume_weight * (mid[:, None] ** 2 * tangent - director),
        )

    def elastic_residual(self, x, s, q):
        mid = (s[:-1] + s[1:]) / 2
        stretch, volume = self.axial_residual(x, s, q)
        curvature = self.curvature(q)
        bend = s[1:-1, None] * curvature - self.rest_curvature
        volume_bend = s[1:-1, None] ** 3 * curvature - self.rest_curvature
        residual = (
            stretch,
            self.radius_weight * (mid - 1),
            self.radius_gradient_weight * (s[1:] - s[:-1]) / self.length,
            self.bending_weight * bend,
            volume,
            self.volume_bending_weight * volume_bend[:, :2],
        )
        # Surface bending uses a length-normalized dual-cell second derivative.
        second = ((s[2:] - s[1:-1]) / self.length[1:] - (s[1:-1] - s[:-2]) / self.length[:-1]) / self.dual_length
        surface = self.surface_weight * second
        return torch.cat((*[part.reshape(-1) for part in residual], surface))

    def energy(self, x, s, q):
        return self.elastic_residual(x, s, q).square().sum() / 2

    def volume(self, x, s):
        return self.area * (((s[:-1] + s[1:]) / 2) ** 2 * torch.linalg.vector_norm(x[1:] - x[:-1], dim=-1)).sum()

    def residual(self, x, s, q, attachments=(), contacts=()):
        position = self.position_weight * (x - self.predicted_pos) / self.dt
        section = (
            self.section_weight
            * (self.directors(s, q) - self.predicted_directors)
            / self.dt
        )
        return torch.cat(
            (position.reshape(-1), section.reshape(-1), self.elastic_residual(x, s, q),
             *(attachment.residual(x) for attachment in attachments),
             *(contact.residual(x, s) for contact in contacts))
        )

    def vertex_residual(self, x, s, q, attachments=(), contacts=()):
        """Only rows with a nonzero position derivative, in original row order."""
        position = self.position_weight * (x - self.predicted_pos) / self.dt
        stretch, volume = self.axial_residual(x, s, q)
        return torch.cat(
            (position.reshape(-1), stretch.reshape(-1), volume.reshape(-1),
             *(attachment.residual(x) for attachment in attachments),
             *(contact.residual(x, s) for contact in contacts))
        )

    def incremental_energy(self, x):
        return self.residual(x, self.scale, self.quat).square().sum() / 2

    def begin(self, x, velocity, dt, gravity):
        self.dt = dt
        self.predicted_pos = x + dt * velocity + dt * dt * gravity
        self.previous_directors = self.directors(self.scale, self.quat)
        self.predicted_directors = self.previous_directors + dt * self.director_velocity

    def sweep(self, x, pinned, attachments=(), contacts=(), vertex_terms=()):
        """One sequential vertex, frame and coupled-scale Gauss-Newton sweep.

        Backtracking enforces descent and positive scale. A failed search raises;
        the iteration count remains the caller's finite-sweep accuracy choice.
        """
        x = x.clone()
        for index in range(len(x) + len(self.quat) + 1):
            is_vertex = index < len(x)
            is_scale = index == len(x) + len(self.quat)
            if is_vertex and bool(pinned[index]):
                continue
            width = len(self.scale) if is_scale else 3
            delta = x.new_zeros(width)

            def trial(d):
                if is_scale:
                    return x, self.scale + d, self.quat
                if is_vertex:
                    mask = torch.nn.functional.one_hot(torch.tensor(index, device=x.device), len(x)).to(x)
                    return x + mask[:, None] * d, self.scale, self.quat
                mask = torch.nn.functional.one_hot(torch.tensor(index - len(x), device=x.device), len(self.quat)).to(x)
                return x, self.scale, retract(self.quat, mask[:, None] * d)

            def block_residual(d):
                return self.residual(*trial(d), attachments, contacts)

            def differentiated_residual(d):
                residual = (
                    self.vertex_residual(*trial(d), attachments, contacts)
                    if is_vertex else block_residual(d)
                )
                return residual, residual

            for contact in contacts:
                contact.refresh(x, self.scale)
            # Reuse the primal from the Jacobian evaluation. Constant rows do
            # not contribute to a vertex block's gradient or GN curvature.
            J, r = torch.func.jacrev(differentiated_residual, has_aux=True)(delta)
            gradient = J.T @ r
            hessian = J.T @ J
            external_force = x.new_zeros(width)
            external_hessian = x.new_zeros((width, width))
            if is_vertex:
                for term in vertex_terms:
                    force, curvature = term(x, index)
                    external_force += force
                    external_hessian += curvature
            gradient -= external_force
            hessian += external_hessian
            direction = torch.linalg.solve(hessian, -gradient)
            # Cayley tangent steps change a unit quaternion by half the angular-step norm.
            # Compare each block in its stored coordinate metric before asking for resolvable descent.
            if is_scale:
                step_size = torch.linalg.vector_norm(direction)
                coordinate_scale = torch.linalg.vector_norm(self.scale)
            elif is_vertex:
                step_size = direction.abs().max()
                coordinate_scale = torch.maximum(x[index].abs().max(), self.length.max())
            else:
                step_size = torch.linalg.vector_norm(direction) / 2
                coordinate_scale = torch.linalg.vector_norm(self.quat[index - len(x)])
            if bool(step_size <= torch.finfo(x.dtype).eps * coordinate_scale):
                continue
            old_energy = block_residual(delta).square().sum() if is_vertex else r.square().sum()
            slope = (gradient * direction).sum()
            is_accepted = False
            for power in range(24):
                fraction = 2.0 ** (-power)
                candidate = trial(fraction * direction)
                if not bool((candidate[1] > 0).all()):
                    continue
                if any(not contact.allowed(x, self.scale, candidate[0], candidate[1]) for contact in contacts):
                    continue
                displacement = fraction * direction
                new_energy = self.residual(*candidate, attachments, contacts).square().sum()
                # Native Hill/tissue blocks are refreshed at each vertex. The local
                # quadratic has their exact current force and native PSD curvature.
                new_energy += -2 * external_force @ displacement + displacement @ external_hessian @ displacement
                if bool((candidate[1] > 0).all()) and bool(
                    new_energy <= old_energy + 1e-4 * fraction * slope + 32 * torch.finfo(x.dtype).eps * old_energy
                ):
                    x, self.scale, self.quat = (value.detach() for value in candidate)
                    is_accepted = True
                    break
            if not is_accepted:
                raise RuntimeError("VIPER block line search failed to decrease the incremental potential.")
        return x

    def end(self, x):
        self.director_velocity = (self.directors(self.scale, self.quat) - self.previous_directors) / self.dt
        if not bool(torch.isfinite(x).all() & torch.isfinite(self.scale).all() & torch.isfinite(self.quat).all()):
            raise RuntimeError("VIPER state is non-finite.")
        if not bool((self.scale > 0).all()):
            raise RuntimeError("VIPER radius collapsed.")
        axial = ((x[1:] - x[:-1]) * quat_matrix(self.quat)[..., 2]).sum(-1)
        if not bool((axial > 0).all()):
            raise RuntimeError("VIPER segment collapsed or reversed against its frame.")

    def get_state(self):
        return RodState(self.scale.clone(), self.quat.clone(), self.director_velocity.clone())

    def set_state(self, state):
        if (
            state.scale.shape != self.scale.shape
            or state.quat.shape != self.quat.shape
            or state.director_velocity.shape != self.director_velocity.shape
        ):
            raise ValueError("VIPER snapshot shape differs from its entity.")
        if not bool(
            torch.isfinite(state.scale).all()
            & torch.isfinite(state.quat).all()
            & torch.isfinite(state.director_velocity).all()
        ):
            raise ValueError("VIPER snapshot must be finite.")
        if not bool((state.scale > 0).all()) or not torch.allclose(
            state.quat.square().sum(-1), torch.ones_like(self.quat[:, 0])
        ):
            raise ValueError("VIPER snapshot requires positive scales and unit frames.")
        self.scale, self.quat, self.director_velocity = (value.clone() for value in state)
