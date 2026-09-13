"""Independent material references for the A5 tissue fixtures in `test_vbd_tissue.py`.

Each formula here is derived on paper from the stable neo-Hookean energy the VBD kernel
implements, `Psi = mu/2 (I_C - 3) + lam'/2 (J - alpha)^2` with `alpha = 1 + mu / lam'`, or from
classic linear beam theory. None of it calls solver code: it only reuses the material constants
(E, nu) an entity is built with. See `tests/vbd/test_vbd_tissue.py::test_lateral_stretch_closed_form_is_a_root_of_the_kernel_stress`
for the symbolic proof that the lateral-stretch formula is an exact root of the kernel's own
first Piola-Kirchhoff stress.
"""

import numpy as np


def lame_parameters(E, nu):
    """Shear modulus mu and lam' = lam + mu, matching `genesis.materials.VBD.Base`."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam + mu


def uniaxial_lateral_stretch(lambda_x, mu, lamp):
    """Traction-free lateral stretch of a homogeneous stable neo-Hookean bar held at axial
    stretch `lambda_x`.

    For F = diag(lambda_x, lambda_y, lambda_y) the kernel stress P = mu F + lam' (J - alpha) cof(F)
    gives P_yy = lambda_y * [lam' * lambda_x^2 * lambda_y^2 - lambda_x * (lam' + mu) + mu]. The
    traction-free root (lambda_y > 0) is:

        lambda_y^2 = alpha / lambda_x - mu / (lam' * lambda_x^2),   alpha = 1 + mu / lam'.
    """
    alpha = 1.0 + mu / lamp
    ly2 = alpha / lambda_x - mu / (lamp * lambda_x**2)
    if ly2 <= 0.0:
        raise ValueError("no positive traction-free lateral stretch at this axial stretch")
    return np.sqrt(ly2)


def clamped_guided_beam_shape(x_over_length, delta=1.0):
    """Transverse deflection at `x_over_length` = x / L of a beam clamped at x = 0 (y = y' = 0)
    and held at a guided support at x = L: translated by `delta`, rotation still zero (y(L) =
    delta, y'(L) = 0). With no distributed load in between, E I y'''' = 0; integrating four times
    with those four boundary conditions gives the cubic Hermite shape:

        y(x) = delta * x^2 (3 L - 2 x) / L^3,  i.e. y(t) = delta * t^2 (3 - 2 t), t = x / L.

    This shape depends only on geometry, not on E, I or the applied delta's magnitude: it is
    the right reference for a *displacement-controlled* bending fixture, sidestepping any
    question of what force a given delta requires.
    """
    t = x_over_length
    return delta * t**2 * (3.0 - 2.0 * t)


def annulus_volume(r_in, r_out, height):
    """Volume of a hollow cylindrical tube (the digestive-wall stand-in): pi (r_out^2 - r_in^2) h."""
    return np.pi * (r_out**2 - r_in**2) * height
