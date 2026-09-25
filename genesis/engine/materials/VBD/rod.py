"""Passive rod coefficients for small reference simulations."""

from genesis.typing import NonNegativeFloat, NonNegativeInt, PositiveFloat

from ..base import Material


class Rod(Material):
    """VIPER strain coefficients (Pa), density (kg/m^3), and surface bending (Pa m^2).

    Rod-only CPU scenes provide a numerical reference for passive mechanics.
    The separate axial, transverse and volume coefficients require calibration.
    """

    collision_group: NonNegativeInt = 0
    rho: PositiveFloat = 1000.0
    stretch_x: PositiveFloat = 1000.0
    stretch_y: PositiveFloat = 1000.0
    stretch_z: PositiveFloat = 10000.0
    volume: PositiveFloat = 1000000.0
    surface_bend: NonNegativeFloat = 0.0
