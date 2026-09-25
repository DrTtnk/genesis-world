"""Pure tensor checks for VIPER rod-bundle distance links."""

import pytest
import torch

from genesis.engine.solvers.vbd_rod_bundle import RodBundleLinks


def test_bundle_links_preserve_energy_under_rigid_motion_and_balance_forces():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.0, 0.1], [0.2, 0.0, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    links = RodBundleLinks(
        torch.tensor([[0, 2], [1, 3]]),
        torch.tensor([0.15, 0.15], dtype=torch.float64),
        torch.tensor([3.0, 5.0], dtype=torch.float64),
    )
    energy = links.residual(positions).square().sum() / 2
    gradient, = torch.autograd.grad(energy, positions)
    torch.testing.assert_close(gradient[0] + gradient[2], torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(gradient[1] + gradient[3], torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(gradient.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)

    axis = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
    rotation = torch.linalg.matrix_exp(torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=torch.float64
    ))
    moved = positions.detach() @ rotation.T + torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(links.residual(moved).square().sum() / 2, energy.detach(), atol=1e-14, rtol=0)


def test_bundle_link_residual_passes_random_gradcheck():
    positions = torch.randn((7, 3), dtype=torch.float64, generator=torch.Generator().manual_seed(21))
    positions.requires_grad_()
    links = RodBundleLinks(
        torch.tensor([[0, 3], [1, 5], [2, 6], [4, 0]]),
        torch.tensor([0.4, 0.8, 1.1, 0.7], dtype=torch.float64),
        torch.tensor([1.2, 2.3, 0.7, 3.1], dtype=torch.float64),
    )
    assert torch.autograd.gradcheck(links.residual, (positions,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_bundle_links_reject_invalid_rest_data_and_indices():
    vertices = torch.tensor([[0, 1]])
    with pytest.raises(ValueError, match="must be positive"):
        RodBundleLinks(vertices, torch.tensor([0.0]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="one value per link"):
        RodBundleLinks(vertices, torch.tensor([1.0, 2.0]), torch.tensor([1.0]))

    links = RodBundleLinks(vertices, torch.tensor([1.0]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="outside"):
        links.residual(torch.zeros((1, 3)))
