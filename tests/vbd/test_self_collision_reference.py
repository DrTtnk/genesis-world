"""The contact between two vertices of the same body, checked against automatic differentiation.

Self-collision is the first contact in this solver where both sides are solver vertices, so unlike the
floor and the bolus it contributes an off-diagonal block as well as a diagonal one. The energy is a
quadratic penalty on the overlap of two spheres of radius half the thickness:

    E = k/2 (t - d)^2   for d < t,   d = |x_i - x_j|

Its exact Hessian is indefinite while the pair overlaps, which is why the solver takes the positive
semidefinite part, as it already does for contact and for the fibre term.
"""

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)
K, THICK = 3.0e3, 0.05


def energy(x_i, x_j):
    d = torch.linalg.norm(x_i - x_j)
    return 0.5 * K * torch.clamp(THICK - d, min=0.0) ** 2


def closed_forms(x_i, x_j):
    """Force on vertex i, the exact diagonal block, and the part of it the solver keeps."""
    e = x_i - x_j
    d = torch.linalg.norm(e)
    n = e / d
    nn = torch.outer(n, n)
    overlap = THICK - d
    force = K * overlap * n
    psd = K * nn
    exact = psd - K * overlap / d * (torch.eye(3) - nn)
    return force, exact, psd


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_pair_force_and_blocks_match_automatic_differentiation(seed):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    overlap = 0.3 + 0.5 * rng.random()  # a real overlap, where the exact block is indefinite
    x_i = torch.as_tensor(rng.normal(scale=0.1, size=3))
    x_j = x_i - torch.as_tensor(direction * THICK * (1.0 - overlap))

    xi = x_i.clone().requires_grad_(True)
    grad = torch.autograd.grad(energy(xi, x_j), xi, create_graph=True)[0]
    hess = torch.stack([torch.autograd.grad(grad[c], xi, retain_graph=True)[0] for c in range(3)])
    force, exact, psd = closed_forms(x_i, x_j)

    assert torch.allclose(-grad, force, atol=1e-9), (-grad, force)
    assert torch.allclose(hess, exact, atol=1e-7), (hess, exact)
    # the kept part is what makes the local solve well posed while the pair overlaps
    eigenvalues = torch.linalg.eigvalsh(exact)
    assert float(eigenvalues.min()) < 0.0, "the exact block should be indefinite during overlap"
    assert float(torch.linalg.eigvalsh(psd).min()) > -1e-9  # rank one, so two eigenvalues sit at roundoff


@pytest.mark.parametrize("seed", [0, 1])
def test_the_off_diagonal_block_is_the_negative_of_the_diagonal(seed):
    """The energy depends only on the difference, so d2E/dxi dxj = -d2E/dxi2. That is what lets the
    reverse sweep scatter to the partner with one sign flip."""
    rng = np.random.default_rng(seed + 10)
    x_i = torch.as_tensor(rng.normal(scale=0.1, size=3)).requires_grad_(True)
    x_j = (x_i.detach() + torch.as_tensor(rng.normal(size=3) * 0.01)).requires_grad_(True)
    grad_i = torch.autograd.grad(energy(x_i, x_j), x_i, create_graph=True)[0]
    mixed = torch.stack([torch.autograd.grad(grad_i[c], x_j, retain_graph=True)[0] for c in range(3)])
    same = torch.stack([torch.autograd.grad(grad_i[c], x_i, retain_graph=True)[0] for c in range(3)])
    assert torch.allclose(mixed, -same, atol=1e-7), (mixed, -same)


def test_no_force_when_the_pair_is_further_apart_than_the_thickness():
    x_i = torch.zeros(3, requires_grad=True)
    x_j = torch.tensor([THICK * 1.5, 0.0, 0.0])
    assert float(energy(x_i, x_j).detach()) == 0.0
    assert torch.allclose(torch.autograd.grad(energy(x_i, x_j), x_i)[0], torch.zeros(3))
