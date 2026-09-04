from genesis.typing import PositiveFloat, PositiveInt

from .base import Base


class Muscle(Base):
    """
    Actuable stable neo-Hookean tetrahedral material for the VBD solver.

    Each tetrahedron may carry a unit fiber direction and a muscle group (see
    ``VBDEntity.set_muscle``). Actuation ``a`` in [0, 1] of a group shortens the rest shape of
    its tetrahedra along the fiber by the ratio ``s = 1 - a * gain`` at constant rest volume.

    Parameters
    ----------
    n_groups : int, optional
        Number of independent muscle groups. Default is 1.
    gain : float, optional
        Maximum fractional fiber contraction at ``a = 1``. Must be below 1. Default is 0.3.
    """

    n_groups: PositiveInt = 1
    gain: PositiveFloat = 0.3
