"""Block derivative checks and pre-optimization serial-sweep oracle."""

import pytest
import torch

from genesis.engine.solvers.vbd_rod import RodAttachment, RodModel, RodParameters, quat_matrix, retract


@pytest.fixture(params=[("cpu", 193), ("cpu", 419), ("cuda", 193), ("cuda", 419)])
def rod_case(request):
    device, seed = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.zeros((5, 3), dtype=torch.float64, device=device)
    x[:, 2] = torch.arange(5, device=device) * 0.01
    q = torch.zeros((4, 4), dtype=x.dtype, device=device)
    q[:, 0] = 1
    model = RodModel(x, q, 0.001, RodParameters(1050, 1300, 1900, 2600, 4000, 900))
    model.scale += 0.03 * torch.randn(5, dtype=x.dtype, device=device, generator=generator)
    model.quat = retract(q, 0.05 * torch.randn((4, 3), dtype=x.dtype, device=device, generator=generator))
    x += 0.0001 * torch.randn(x.shape, dtype=x.dtype, device=device, generator=generator)
    model.begin(x, torch.zeros_like(x), 0.003, x.new_tensor([0.0, 0.0, -9.81]))
    attachment = RodAttachment(
        torch.tensor([[0, 1]], device=device),
        x.new_tensor([[0.7, 0.3]]),
        x[:1].clone(),
        x.new_tensor([20.0]),
    )
    return model, x, (attachment,)


def test_forward_ad_matches_reverse_ad_random_blocks(rod_case):
    model, x, attachments = rod_case
    for index in range(10):
        is_scale = index == 9
        is_vertex = index < 5
        delta = x.new_zeros(5 if is_scale else 3)

        def residual(d):
            if is_scale:
                return model.residual(x, model.scale + d, model.quat, attachments)
            if is_vertex:
                mask = torch.nn.functional.one_hot(torch.tensor(index, device=x.device), 5).to(x)
                return model.residual(x + mask[:, None] * d, model.scale, model.quat, attachments)
            mask = torch.nn.functional.one_hot(torch.tensor(index - 5, device=x.device), 4).to(x)
            return model.residual(x, model.scale, retract(model.quat, mask[:, None] * d), attachments)

        def with_aux(d):
            r = residual(d)
            return r, r

        jacrev, primal = torch.func.jacrev(with_aux, has_aux=True)(delta)
        reverse = torch.autograd.functional.jacobian(residual, delta, vectorize=True)
        torch.testing.assert_close(jacrev, reverse, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(primal, residual(delta), atol=0, rtol=0)
        forward = torch.autograd.functional.jacobian(residual, delta, vectorize=True, strategy="forward-mode")
        torch.testing.assert_close(forward, reverse, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(forward.T @ forward, reverse.T @ reverse, atol=1e-12, rtol=1e-12)


class ReverseRodModel(RodModel):
    """Pre-optimization serial reverse-AD oracle; test-only."""

    def elastic_residual(self, x, s, q):
        p = self.parameters
        mid = (s[:-1] + s[1:]) / 2
        tangent = (x[1:] - x[:-1]) / self.length[:, None]
        director = quat_matrix(q)[..., 2]
        curvature = self.curvature(q)
        bend = s[1:-1, None] * curvature - self.rest_curvature
        volume_bend = s[1:-1, None] ** 3 * curvature - self.rest_curvature
        bending_weight = x.new_tensor(
            (
                4 * self.moment * p.stretch_z,
                4 * self.moment * p.stretch_z,
                4 * self.moment * (p.stretch_x + p.stretch_y),
            )
        )
        residual = (
            (self.area * p.stretch_z * self.length).sqrt()[:, None] * (tangent - director),
            (self.area * (p.stretch_x + p.stretch_y) * self.length).sqrt() * (mid - 1),
            (self.moment * (p.stretch_x + p.stretch_y) * self.length).sqrt() * (s[1:] - s[:-1]) / self.length,
            (self.dual_length[:, None] * bending_weight).sqrt() * bend,
            (self.area * p.volume * self.length).sqrt()[:, None] * (mid[:, None] ** 2 * tangent - director),
            (2 * self.moment * p.volume * self.dual_length).sqrt()[:, None] * volume_bend[:, :2],
        )
        # Surface bending uses a length-normalized dual-cell second derivative.
        second = ((s[2:] - s[1:-1]) / self.length[1:] - (s[1:-1] - s[:-2]) / self.length[:-1]) / self.dual_length
        surface = (self.moment * p.surface_bend * self.dual_length).sqrt() * second
        return torch.cat((*[part.reshape(-1) for part in residual], surface))
    def residual(self, x, s, q, attachments=(), contacts=()):
        position = self.mass.sqrt()[:, None] * (x - self.predicted_pos) / self.dt
        section = (
            (self.parameters.density * self.moment * self.length).sqrt()[:, None, None]
            * (self.directors(s, q) - self.predicted_directors)
            / self.dt
        )
        return torch.cat(
            (position.reshape(-1), section.reshape(-1), self.elastic_residual(x, s, q),
             *(attachment.residual(x) for attachment in attachments),
             *(contact.residual(x, s) for contact in contacts))
        )

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

            for contact in contacts:
                contact.refresh(x, self.scale)
            r = block_residual(delta)
            J = torch.autograd.functional.jacobian(block_residual, delta, vectorize=True)
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
            # A step below relative coordinate precision is already converged;
            # Armijo decrease cannot be resolved for a rotated rest state there.
            coordinate_scale = (
                self.scale.abs().max()
                if is_scale
                else (torch.maximum(x[index].abs().max(), self.length.max()) if is_vertex else x.new_tensor(1.0))
            )
            if bool(direction.abs().max() <= torch.finfo(x.dtype).eps * coordinate_scale):
                continue
            old_energy = r.square().sum()
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


def test_cached_sweep_matches_original_oracle(rod_case):
    model, x, attachments = rod_case
    oracle = ReverseRodModel(model.rest_pos, model.rest_quat, model.radius, model.parameters)
    oracle.set_state(model.get_state())
    oracle.begin(x, torch.zeros_like(x), model.dt, x.new_tensor([0.0, 0.0, -9.81]))
    pinned = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    pinned[0] = True
    actual = model.sweep(x, pinned, attachments)
    expected = oracle.sweep(x, pinned, attachments)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(model.scale, oracle.scale, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(model.quat, oracle.quat, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(model.energy(actual, model.scale, model.quat),
                               oracle.energy(expected, oracle.scale, oracle.quat), atol=1e-15, rtol=1e-11)


def test_vertex_residual_reduction_preserves_gradient_and_curvature(rod_case):
    model, x, attachments = rod_case
    mask = torch.nn.functional.one_hot(torch.tensor(2, device=x.device), len(x)).to(x)
    delta = x.new_zeros(3)

    def full(d):
        return model.residual(x + mask[:, None] * d, model.scale, model.quat, attachments)

    def reduced(d):
        candidate = x + mask[:, None] * d
        tangent = (candidate[1:] - candidate[:-1]) / model.length[:, None]
        director = quat_matrix(model.quat)[..., 2]
        mid = (model.scale[:-1] + model.scale[1:]) / 2
        return torch.cat((
            (model.mass.sqrt()[:, None] * (candidate - model.predicted_pos) / model.dt).reshape(-1),
            (model.stretch_weight * (tangent - director)).reshape(-1),
            (model.volume_weight * (mid[:, None] ** 2 * tangent - director)).reshape(-1),
            *(attachment.residual(candidate) for attachment in attachments),
        ))

    torch.testing.assert_close(model.vertex_residual(x, model.scale, model.quat, attachments), reduced(delta), atol=0, rtol=0)
    full_j = torch.autograd.functional.jacobian(full, delta, vectorize=True)
    reduced_j = torch.autograd.functional.jacobian(reduced, delta, vectorize=True)
    torch.testing.assert_close(reduced_j.T @ reduced(delta), full_j.T @ full(delta), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(reduced_j.T @ reduced_j, full_j.T @ full_j, atol=1e-12, rtol=1e-12)
