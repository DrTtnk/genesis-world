from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base


class Shell(Base):
    """
    Stable neo-Hookean shell material for the VBD solver: a thin sheet carrying triangles and, where two
    triangles share an edge, a bending stencil. No thickness dimension is resolved by the mesh.

    `E`, `nu` and `rho` keep the meaning they have in `Base`, and give the membrane its `mu`/`lam`. The
    mesh carries no volume, so `thickness` stands in for it: vertex mass is `thickness * rho * area`. It is
    the sheet's mechanical thickness and nothing else. The distance at which a contact pair activates is the
    `thickness` argument of `VBDSolver.add_contact_rule`, declared per pair of collision groups and measured
    between the two surfaces themselves, not per participant and not derived from this field. The two are
    independent on purpose: a 1.5 mm wall may need a 2 mm pair distance for the step size in use, and changing
    the pair distance must not change any mass.

    An entity built from this material carries triangles and bending stencils only. It may not mix
    tetrahedra and triangles in one entity.

    Rayleigh damping (`VBDOptions.damping`) applies to the membrane, whose rest Hessian is positive
    semidefinite at every Poisson ratio and zero on rigid motions. Bending is deliberately left undamped: its
    quadratic model is not invariant under a rotation of a curved rest shape, so its Hessian would brake a
    body that merely coils. Gradients (`requires_grad`) are not implemented for shell elements yet: the solver
    raises rather than silently ignoring them.

    Parameters
    ----------
    thickness : float, optional
        Mechanical sheet thickness (m): sets vertex mass, `thickness * rho * area`. It is not the contact
        distance; see `VBDSolver.add_contact_rule`. Default is 1e-3.
    bending_stiffness : float, optional
        Weight of the curvature-change bending energy between two triangles sharing an edge. 0 disables
        bending. Default is 0.0.
    """

    thickness: PositiveFloat = 1e-3
    bending_stiffness: NonNegativeFloat = 0.0
