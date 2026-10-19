"""The fixed-wing tables as an airframe model: ``C + sum(delta C)``.

:class:`FixedWingAero` is a ``Vehicle.aero``: the rotors-off aircraft (:class:`~sudestada.aero.fixed.data.FixedWingData`) as coefficient
splines over ``(alpha, beta)``, plus each control surface's
increment over ``(alpha, delta)`` at its actual deflection, then the rate derivatives and the textbook 6-DOF build-up.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from sudestada.types.aero import AeroCoefficients
from sudestada.types.airframe import AerodynamicWrench, AirframeState
from sudestada.frames import FRD_FROM_FLU, body_from_wind, flow_angles
from sudestada.fits.splines import CoefficientSplines, smooth_clamp
from sudestada.aero.fixed.data import FixedWingData

BETA_CLAMP_EPS_DEG = 2.0


class FixedWingAero:
    """The clean airframe and the increments of the surfaces named ``surfaces`` (in deflection order; None: every
    surface in ``data``, in its order), at ``data.rho``.  ``beta`` is smoothly clamped to the clean grid, each
    ``delta`` clipped to its surface's grid.  It needs no vehicle: anything that wants the aircraft's coefficients
    (a trim, a stability check) can build one from a :class:`FixedWingData`."""

    def __init__(self, data: FixedWingData, surfaces: Optional[Sequence[str]] = None):
        self.data = data
        clean = data.clean
        self.reference, self.rates, self.rho = data.reference, clean.rate_derivatives, data.rho
        self._clean = CoefficientSplines(clean.alpha_grid, clean.beta_grid, clean.coefficients)
        self._beta_range = (clean.beta_grid[0], clean.beta_grid[-1])

        self.surfaces = tuple(surface.name for surface in data.surfaces) if surfaces is None else tuple(surfaces)
        self._surfaces = []
        for name in self.surfaces:
            increments = data.surface(name)
            grid = increments.delta_grid_deg
            self._surfaces.append(((grid[0], grid[-1]), CoefficientSplines(clean.alpha_grid, grid, increments.increments)))

    def coefficients(self, alpha, beta, deflections) -> AeroCoefficients[float]:
        """``C(alpha, beta)`` plus each surface's ``delta C(alpha, delta)`` (rad)."""

        alpha_deg = np.degrees(alpha)
        beta_deg = smooth_clamp(np.degrees(beta), *self._beta_range, BETA_CLAMP_EPS_DEG)
        c = self._clean(alpha_deg, beta_deg)
        for (delta_range, splines), delta in zip(self._surfaces, deflections):
            c = c + splines(alpha_deg, float(np.clip(np.degrees(delta), *delta_range)))
        return c

    def __call__(self, state: AirframeState) -> AerodynamicWrench:
        reference, w = self.reference, state.w
        # One frame conversion at the boundary: FRD flow angles and rates at the airframe origin (the MRP).
        speed, alpha, beta = flow_angles(state.airspeed)
        p, q, r = FRD_FROM_FLU @ w
        c = self.coefficients(alpha, beta, state.surface_deflections)
        qbar = 0.5 * self.rho * speed * speed
        area, span, cbar = reference.area, reference.span, reference.cbar

        force_frd = qbar * area * body_from_wind(alpha, beta) @ np.array([-c.CD, c.CY, -c.CL])
        moment_frd = qbar * area * np.array([span * c.Cl, cbar * c.Cm, span * c.Cn])

        # Rate terms, written multiplied out so nothing divides by V.
        d = self.rates
        scale = 0.25 * self.rho * speed * area
        force_frd[1] += scale * span * (d.CYp * p + d.CYr * r)
        force_frd[2] -= scale * cbar * d.CLq * q
        moment_frd[0] += scale * span * span * (d.Clp * p + d.Clr * r)
        moment_frd[1] += scale * cbar * cbar * d.Cmq * q
        moment_frd[2] += scale * span * span * (d.Cnp * p + d.Cnr * r)

        return AerodynamicWrench(FRD_FROM_FLU @ force_frd, FRD_FROM_FLU @ moment_frd)
