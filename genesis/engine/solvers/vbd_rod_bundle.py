"""Torch energy terms for transverse links between VIPER rod vertices."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RodBundleLinks:
    """Elastic distance links between explicitly mapped vertices in a flattened rod bundle."""

    vertices: torch.Tensor
    rest_lengths: torch.Tensor
    stiffness: torch.Tensor

    def __post_init__(self):
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2 or self.vertices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Bundle vertices must be an integer tensor with shape (n_links, 2).")
        if self.rest_lengths.shape != (len(self.vertices),) or self.stiffness.shape != (len(self.vertices),):
            raise ValueError("Bundle rest lengths and stiffnesses must have one value per link.")
        if not bool(torch.isfinite(self.rest_lengths).all() & torch.isfinite(self.stiffness).all()):
            raise ValueError("Bundle rest lengths and stiffnesses must be finite.")
        if not bool(((self.rest_lengths > 0) & (self.stiffness > 0)).all()):
            raise ValueError("Bundle rest lengths and stiffnesses must be positive.")

    def residual(self, positions):
        """Return weighted distance errors for insertion into a rod residual vector."""
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Bundle positions must have shape (n_vertices, 3).")
        if not bool(((self.vertices >= 0) & (self.vertices < len(positions))).all()):
            raise ValueError("Bundle vertex index is outside the flattened position array.")
        delta = positions[self.vertices[:, 0]] - positions[self.vertices[:, 1]]
        return self.stiffness.sqrt() * (torch.linalg.vector_norm(delta, dim=-1) - self.rest_lengths)


@dataclass(frozen=True)
class RodBundleView:
    """One rod varies while the other bundle nodes keep their current iterate."""

    links: RodBundleLinks
    positions: torch.Tensor
    start: int

    def residual(self, x):
        positions = torch.cat((self.positions[:self.start], x, self.positions[self.start + len(x):]))
        return self.links.residual(positions)
