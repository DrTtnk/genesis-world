"""A block-level reference for the solver's derivatives, in plain torch.

The solver builds its per-vertex gradient, its 3x3 block and the off-diagonal blocks from closed
forms. Here the same quantities come from automatic differentiation of one tetrahedron's energy, so
a wrong sign or a missing factor shows up in the term that carries it, not as a wrong trajectory ten
thousand vertices later. The reverse-sweep adjoint needs one further derivative, the Hessian tangent,
and its closed form is checked the same way.

No Genesis scene and no GPU: this is about the mathematics the kernels implement.
"""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _double_precision():
    """Genesis sets the torch default dtype to float32 when it initialises, and that happens after
    this module is imported, so the dtype has to be set per test rather than once at import."""
    torch.set_default_dtype(torch.float64)


MU, LAM = 3.0, 7.0  # lam is the solver's lam' = lam + mu
ALPHA = 1.0 + MU / LAM


def rest_frame(seed):
    """A random tetrahedron: its inverse rest shape B, its rest volume V, and the four vertex weights."""
    rng = np.random.default_rng(seed)
    rest = torch.as_tensor(rng.normal(size=(3, 4)))
    Dm = torch.stack([rest[:, i] - rest[:, 0] for i in (1, 2, 3)], dim=1)
    B = torch.linalg.inv(Dm)
    V = abs(float(torch.det(Dm))) / 6.0
    w = [-(B.T @ torch.ones(3))] + [B.T[:, r] for r in range(3)]  # dF = u w_j^T when vertex j moves by u
    return B, V, w, torch.as_tensor(rng.normal(size=(3, 4)))


def deformation(x, B):
    return torch.stack([x[:, i] - x[:, 0] for i in (1, 2, 3)], dim=1) @ B


def energy(x, B, V):
    """Stable neo-Hookean, the solver's form: mu/2 (|F|^2 - 3) + lam/2 (det F - alpha)^2."""
    F = deformation(x, B)
    return V * (0.5 * MU * ((F * F).sum() - 3.0) + 0.5 * LAM * (torch.det(F) - ALPHA) ** 2)


def cofactor(F):
    return torch.stack([torch.linalg.cross(F[:, 1], F[:, 2]), torch.linalg.cross(F[:, 2], F[:, 0]), torch.linalg.cross(F[:, 0], F[:, 1])], dim=1)


def skew(a):
    return torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def closed_forms(x, B, V, w, i, j):
    """What the solver computes: the vertex force, its diagonal block, and the block against vertex j."""
    F = deformation(x, B)
    cof = cofactor(F)
    J = torch.det(F)
    P = MU * F + LAM * (J - ALPHA) * cof
    q_i, q_j = cof @ w[i], cof @ w[j]
    g_i = V * (P @ w[i])  # the solver's `force` is -g_i
    H_ii = V * (MU * w[i].dot(w[i]) * torch.eye(3) + LAM * torch.outer(q_i, q_i))
    K_ij = -skew(F @ torch.linalg.cross(w[i], w[j]))
    H_ij = V * (MU * w[i].dot(w[j]) * torch.eye(3) + LAM * torch.outer(q_i, q_j) + LAM * (J - ALPHA) * K_ij)
    return g_i, H_ii, H_ij, q_i, K_ij


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_vertex_force_and_its_blocks_match_automatic_differentiation(seed):
    B, V, w, x = rest_frame(seed)
    x = x.requires_grad_(True)
    grad = torch.autograd.grad(energy(x, B, V), x, create_graph=True)[0]
    for i in range(4):
        for j in range(4):
            H = torch.stack([torch.autograd.grad(grad[c, i], x, retain_graph=True)[0][:, j] for c in range(3)])
            g_i, H_ii, H_ij, _, _ = closed_forms(x.detach(), B, V, w, i, j)
            assert torch.allclose(grad[:, i], g_i, atol=1e-10), (i, grad[:, i], g_i)
            expect = H_ii if i == j else H_ij
            assert torch.allclose(H, expect, atol=1e-9), (i, j, H - expect)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_hessian_tangent_matches_automatic_differentiation(seed):
    """The reverse sweep needs (d H_i / d x_j) contracted with a rank-one matrix. Only q_i depends on
    the state, so the closed form is (q_i . dx) K^T p + (q_i . p) K^T dx with K = -[F (w_i x w_j)]_x."""
    B, V, w, x = rest_frame(seed)
    rng = np.random.default_rng(seed + 100)
    p, dx = (torch.as_tensor(rng.normal(size=3)) for _ in range(2))
    for i in range(4):
        for j in range(4):
            xv = x.clone().requires_grad_(True)
            F = deformation(xv, B)
            q = cofactor(F) @ w[i]
            H_i = V * (MU * w[i].dot(w[i]) * torch.eye(3) + LAM * torch.outer(q, q))
            reference = torch.autograd.grad((H_i * torch.outer(p, dx)).sum(), xv)[0][:, j]
            _, _, _, q_i, K_ij = closed_forms(x, B, V, w, i, j)
            closed = V * LAM * (q_i.dot(dx) * (K_ij.T @ p) + q_i.dot(p) * (K_ij.T @ dx))
            assert torch.allclose(reference, closed, atol=1e-9), (i, j, reference, closed)
            if i == j:
                assert torch.allclose(closed, torch.zeros(3), atol=1e-12), "the tangent must vanish on the diagonal"
