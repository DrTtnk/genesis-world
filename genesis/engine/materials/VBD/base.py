from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from genesis.typing import PositiveFloat, ValidFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.vbd_entity import VBDEntity


class Base(Material["VBDEntity"]):
    """
    The base class of VBD (Vertex Block Descent) materials: stable neo-Hookean tetrahedral solids.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa). Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Material density (kg/m³). Default is 1000.
    """

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0

    # Lamé parameters, computed in model_post_init, not user-specified.
    mu: ValidFloat = Field(default=0.0, exclude=True)
    lam: ValidFloat = Field(default=0.0, exclude=True)

    def model_post_init(self, context: Any) -> None:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
