"""The rotors-off aircraft's data, typed: what an offline fit produces and :class:`~sudestada.aero.fixed.model.FixedWingAero` reads.  

* Clean-airframe coefficients are conventional aerodynamic coefficients in **FRD** axes, rate derivatives included,
  about the **airframe origin**: it is the moment reference point, and the flow angles are read there.
* Control-surface increments add to the clean coefficients in the same FRD frame, about the same origin, before the single
  frame conversion: ``dC[alpha, delta]`` on the clean alpha grid and a surface ``delta_grid_deg``, with
  ``dC(alpha, 0) = 0``; trailing-edge-down positive.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np

from sudestada.types.aero import AeroCoefficients, AeroReference, COEFFICIENT_NAMES, RateDerivatives


def _array(value, shape, name):
    array = np.array(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}, got {array.shape}")
    return array


def _unit(value, name):
    vector = _array(value, (3,), name)
    if abs(np.linalg.norm(vector) - 1.0) > 1e-6:
        raise ValueError(f"{name} must be a unit vector")
    return vector


def _grid(value, name, periodic=False):
    grid = np.array(value, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} must be a strictly increasing 1-D grid of at least 2 points")
    if periodic and not np.allclose(np.diff(grid), grid[1] - grid[0], rtol=1e-9, atol=1e-12):
        raise ValueError(f"periodic {name} must be uniform")
    return grid


def _tables(value: AeroCoefficients, shape, name) -> AeroCoefficients:
    """An :class:`AeroCoefficients` of tables, each checked to ``shape``."""

    return AeroCoefficients(*(_array(getattr(value, key), shape, f"{name}.{key}") for key in COEFFICIENT_NAMES))


def _set(record, **values):
    for key, value in values.items():
        object.__setattr__(record, key, value)


def _provenance(record, name):
    if not record.provenance:
        raise ValueError(f"{name}.provenance must say where the numbers came from")


@dataclass(frozen=True, kw_only=True)
class CleanData:
    """The rotors-off airframe: coefficient tables over ``(alpha_grid, beta_grid)`` (deg), bodies included, rate
    derivatives, and the ``trusted_alpha_deg`` band where the tables are solver data only (outside it they blend into a crude post-stall model; None: no band declared)."""

    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    coefficients: AeroCoefficients[np.ndarray]
    rate_derivatives: RateDerivatives
    provenance: Dict[str, object]
    trusted_alpha_deg: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        alpha = _grid(self.alpha_grid, "clean.alpha_grid", periodic=True)
        beta = _grid(self.beta_grid, "clean.beta_grid")
        if not all(np.isfinite(getattr(self.rate_derivatives, f.name)) for f in fields(RateDerivatives)):
            raise ValueError("clean.rate_derivatives must be finite")
        _set(self, alpha_grid=alpha, beta_grid=beta,
             coefficients=_tables(self.coefficients, (alpha.size, beta.size), "clean.coefficients"))
        if self.trusted_alpha_deg is not None:
            low, high = (float(value) for value in self.trusted_alpha_deg)
            if not low < high:
                raise ValueError("clean.trusted_alpha_deg must be [low, high] with low < high")
            _set(self, trusted_alpha_deg=(low, high))
        _provenance(self, "clean")


@dataclass(frozen=True, kw_only=True)
class SurfaceIncrements:
    """One control surface's effect on the whole aircraft: ``increments[alpha, delta]`` on the clean alpha grid and
    ``delta_grid_deg``, zero at zero deflection.  The surface itself (its actuator) is the vehicle's
    ``ControlSurface`` of the same ``name``."""

    name: str
    delta_grid_deg: np.ndarray
    increments: AeroCoefficients[np.ndarray]
    provenance: Dict[str, object]

    def __post_init__(self):
        grid = _grid(self.delta_grid_deg, f"{self.name}.delta_grid_deg")
        zero = np.flatnonzero(np.abs(grid) < 1e-9)
        if zero.size != 1:
            raise ValueError(f"{self.name}.delta_grid_deg must contain 0 deg")
        shape = np.shape(self.increments.CL)
        increments = _tables(self.increments, (shape[0] if shape else -1, grid.size), f"{self.name}.increments")
        if any(np.max(np.abs(getattr(increments, key)[:, zero[0]])) > 1e-12 for key in COEFFICIENT_NAMES):
            raise ValueError(f"{self.name}.increments must be zero at zero deflection")
        _set(self, delta_grid_deg=grid, increments=increments)
        _provenance(self, self.name)


def _check_surfaces(surfaces, clean):
    names = [surface.name for surface in surfaces]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate surface names in {names}")
    for surface in surfaces:
        if surface.increments.CL.shape[0] != clean.alpha_grid.size:
            raise ValueError(f"{surface.name}.increments must lie on the clean alpha grid")


@dataclass(frozen=True, kw_only=True)
class FixedWingData:
    """The rotors-off aircraft on its own: the reference it is normalised on, the air density it assumes, the clean
    airframe and the control-surface increments.  What :class:`~sudestada.aero.fixed.model.FixedWingAero` reads."""

    reference: AeroReference
    rho: float
    clean: CleanData
    surfaces: Tuple[SurfaceIncrements, ...] = ()

    def __post_init__(self):
        reference = self.reference
        if not all(value > 0.0 for value in (self.rho, reference.area, reference.span, reference.cbar)):
            raise ValueError("rho and the reference S, b and cbar must be positive")
        _set(self, rho=float(self.rho), surfaces=tuple(self.surfaces))
        _check_surfaces(self.surfaces, self.clean)

    def surface(self, name) -> SurfaceIncrements:
        return {surface.name: surface for surface in self.surfaces}[name]
