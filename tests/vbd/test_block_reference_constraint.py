"""A reference for one fused block update, in plain torch.

The forward does two things in a vertex's colour pass: it takes a Newton step for the vertex, then it
updates the multiplier and the stiffness of every constraint that vertex owns. The reverse sweep must
undo both, in the opposite order. Automatic differentiation through a plain torch version of the same
two steps gives the exact adjoint to build against, including the terms that are easy to forget: the
dependence of the step on the multiplier, and of the multiplier on the position.

The augmented Lagrangian of Giles et al. 2025, which the paper on reverse-sweep adjoints does not
cover, is what this file pins down.
"""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _double_precision():
    """Genesis sets the torch default dtype to float32 when it initialises, and that happens after
    this module is imported, so the dtype has to be set per test rather than once at import."""
    torch.set_default_dtype(torch.float64)


H_STEP, MASS, K_C, W, REST = 2.5e-4, 0.01, 5.0e4, 0.25, 0.30


def constraint(x_a, x_b):
    return torch.linalg.norm(x_a - x_b) - REST


def block_update(x_i, x_j, y_i, lam, k):
    """One owner vertex's fused update: its Newton step, then the dual update of the constraint it owns.

    The local energy is the inertia term plus the augmented Lagrangian of one equality constraint, so the
    local gradient and Hessian are exactly what the solver builds for that case.
    """
    e = x_i - x_j
    dist = torch.linalg.norm(e)
    n = e / dist
    mult = k * (dist - REST) + lam
    g = MASS / H_STEP**2 * (x_i - y_i) + mult * n
    H = MASS / H_STEP**2 * torch.eye(3) + k * torch.outer(n, n) + (mult.abs() / dist) * (torch.eye(3) - torch.outer(n, n))
    dx = -torch.linalg.solve(H, g)
    x_new = x_i + dx
    lam_new = lam + W * k * constraint(x_new, x_j)  # the dual update reads the position the step just produced
    return x_new, lam_new, dx


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_fused_block_update_has_the_adjoint_the_reverse_sweep_must_build(seed):
    """The three derivatives the reverse pass needs, against automatic differentiation."""
    rng = np.random.default_rng(seed)
    x_i = torch.as_tensor(rng.normal(scale=0.1, size=3))
    x_j = torch.as_tensor(rng.normal(scale=0.1, size=3) + np.array([REST, 0.0, 0.0]))
    y_i = x_i + torch.as_tensor(rng.normal(scale=1e-3, size=3))
    lam = torch.as_tensor(float(rng.normal(scale=10.0)))
    k = torch.as_tensor(K_C)
    xbar_new = torch.as_tensor(rng.normal(size=3))
    lambar_new = torch.as_tensor(float(rng.normal()))

    inputs = [t.clone().requires_grad_(True) for t in (x_i, x_j, y_i, lam)]
    x_new, lam_new, dx = block_update(*inputs, k)
    grads = torch.autograd.grad((xbar_new * x_new).sum() + lambar_new * lam_new, inputs)
    xbar_i, xbar_j, ybar_i, lambar = grads

    # what the reverse sweep computes, term by term, from the same quantities
    e = (x_i - x_j).detach()
    dist = torch.linalg.norm(e)
    n = e / dist
    mult = k * (dist - REST) + lam
    H = MASS / H_STEP**2 * torch.eye(3) + k * torch.outer(n, n) + (mult.abs() / dist) * (torch.eye(3) - torch.outer(n, n))

    # The dual update comes last in the forward, so it is undone first, and it reads the constraint at the
    # position the step just produced. Using the position the step started from is wrong by the step size
    # over the constraint length, a few percent here, and that is the ordering the kernel must follow.
    e_new = (x_new - x_j).detach()
    n_new = e_new / torch.linalg.norm(e_new)
    xbar_after_dual = xbar_new + lambar_new * W * k * n_new

    p = torch.linalg.solve(H.T, xbar_after_dual)
    print(f"seed={seed}: |xbar_i| {xbar_i.norm():.3e}, |xbar_j| {xbar_j.norm():.3e}, |ybar| {ybar_i.norm():.3e}, lambar {lambar:.3e}", flush=True)

    # the predictor enters only through the inertia term
    assert torch.allclose(ybar_i, MASS / H_STEP**2 * p, atol=1e-8), (ybar_i, MASS / H_STEP**2 * p)

    # The block's own outgoing adjoint is the identity branch, minus the exact local Jacobian of the
    # gradient, minus the Hessian tangent. A constraint's block depends on the vertex's own position,
    # through k n n^T and the |mult| / dist proxy, so unlike the elastic block its diagonal tangent does
    # not vanish. Both are taken from automatic differentiation here, to check the formula itself.
    def local_gradient(xx):
        e_ = xx - x_j
        d_ = torch.linalg.norm(e_)
        return MASS / H_STEP**2 * (xx - y_i) + (k * (d_ - REST) + lam) * e_ / d_

    def local_hessian_form(xx):
        e_ = xx - x_j
        d_ = torch.linalg.norm(e_)
        n_ = e_ / d_
        m_ = k * (d_ - REST) + lam
        H_ = MASS / H_STEP**2 * torch.eye(3) + k * torch.outer(n_, n_) + (m_.abs() / d_) * (torch.eye(3) - torch.outer(n_, n_))
        return p.detach().dot(H_ @ dx.detach())

    dg = torch.autograd.functional.jacobian(local_gradient, x_i.clone())
    tangent = torch.autograd.grad(local_hessian_form(x_i.clone().requires_grad_(True)), x_i.clone().requires_grad_(True), allow_unused=True)[0]
    xx = x_i.clone().requires_grad_(True)
    tangent = torch.autograd.grad(local_hessian_form(xx), xx)[0]
    predicted = xbar_after_dual - dg.T @ p - tangent
    assert torch.allclose(xbar_i, predicted, atol=1e-6 * max(float(xbar_i.norm()), 1.0)), (xbar_i, predicted)

    # the multiplier enters the step through the constraint force, dg/dlam = n, and survives the dual update
    assert torch.allclose(lambar, lambar_new - n.dot(p), atol=1e-8), (lambar, lambar_new - n.dot(p))
