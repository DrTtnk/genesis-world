from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base


class Shell(Base):
    """
    Stable neo-Hookean shell material for the VBD solver: a thin sheet carrying triangles and, where two
    triangles share an edge, a bending stencil. No thickness dimension is resolved by the mesh.

    `E`, `nu` and `rho` keep the meaning they have in `Base`, and give the membrane its `mu`/`lam`. The
    mesh carries no volume, so `thickness` stands in for it: vertex mass is `thickness * rho * area` and
    `thickness` is also the shell's contact margin.

    An entity built from this material carries triangles and bending stencils only. It may not mix
    tetrahedra and triangles in one entity.

    Rayleigh damping (`VBDOptions.damping`) and gradients (`requires_grad`) are not implemented for shell
    elements yet: the solver raises rather than silently ignoring either.

    Parameters
    ----------
    thickness : float, optional
        Sheet thickness (m): sets vertex mass and the contact margin. Default is 1e-3.
    bending_stiffness : float, optional
        Weight of the curvature-change bending energy between two triangles sharing an edge. 0 disables
        bending. Default is 0.0.
    """

    thickness: PositiveFloat = 1e-3
    bending_stiffness: NonNegativeFloat = 0.0
