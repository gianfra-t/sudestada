"""The powered-aero model's data, typed: the rotors-off aircraft (:mod:`sudestada.aero.fixed.data`) plus the rotor
fits, the receivers their wakes wash and the (receiver, source) pairs.  Every record validates itself at construction,
and :class:`PoweredAeroData` checks that its records refer to each other consistently.  Serialization is
:mod:`sudestada.aero.interaction.io`'s.

* Receiver frames and immersion centroids are in the **vehicle body frame (FLU)**, origin at the airframe body origin,
  to match the ``AirframeState`` the vehicle hands the model.
* A receiver stores ``c_hat`` (chordwise, pointing aft), ``s_hat`` (spanwise, chosen by the pipeline) and the induction
  ``n_hat = cross(c_hat, s_hat)``: the *suction-side* normal, so a wing at positive incidence gives a positive angle of
  attack and upward lift.
* Gain polynomials are in ``(chi_deg, delta_t_deg)`` with the six monomials ``1, x, y, x^2, xy, y^2``.

Rotor placement, disks and surface actuators are the vehicle's (:class:`~sudestada.vehicle.Vehicle`); the vehicle's
builder picks each part's record by name (``data.rotor(name)``, ``data.surface(name)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple

import numpy as np

from sudestada.types.aero import AeroReference
from sudestada.types.units import Area, Length
from sudestada.aero.fixed.data import (
    CleanData, FixedWingData, SurfaceIncrements, _array, _grid, _provenance, _set, _unit,
)
from sudestada.aero.rotor.polynomial import RotorFit

GAIN_NAMES = ("k_w", "gF", "gm", "dalpha")
# B3-default: the published defaults; B4-fitted: calibrated (needs its report); assumed: set by hand, e.g. a bound.
GAIN_STATUSES = ("B3-default", "B4-fitted", "assumed")


@dataclass(frozen=True, kw_only=True)
class ReceiverData:
    """A lifting panel that rotor wakes wash: chordwise ``c_hat`` and spanwise ``s_hat`` unit axes (body FLU), chord,
    total area, and its 360-deg section polar ``cl, cd, cm`` over ``polar_alpha_grid`` (deg)."""

    name: str
    c_hat: np.ndarray
    s_hat: np.ndarray
    chord: float
    area_total: float
    polar_alpha_grid: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    provenance: Dict[str, object]

    def __post_init__(self):
        c_hat, s_hat = _unit(self.c_hat, f"{self.name}.c_hat"), _unit(self.s_hat, f"{self.name}.s_hat")
        if abs(np.dot(c_hat, s_hat)) > 1e-6:
            raise ValueError(f"{self.name}: c_hat and s_hat must be orthogonal")
        if not (self.chord > 0.0 and self.area_total > 0.0):
            raise ValueError(f"{self.name}: chord and area_total must be positive")
        grid = _grid(self.polar_alpha_grid, f"{self.name}.polar_alpha_grid", periodic=True)
        polar = {key: _array(getattr(self, key), grid.shape, f"{self.name}.{key}") for key in ("cl", "cd", "cm")}
        if np.any(polar["cd"] < -1e-9):
            raise ValueError(f"{self.name}: cd must be nonnegative")
        _set(self, c_hat=c_hat, s_hat=s_hat, polar_alpha_grid=grid, **polar)
        _provenance(self, self.name)

    @cached_property
    def n_hat(self) -> np.ndarray:
        return np.cross(self.c_hat, self.s_hat)


@dataclass(frozen=True, kw_only=True)
class PairData:
    """How rotor ``source``'s wake washes receiver ``receiver``: the immersed area and its centroid over the wake
    direction ``(theta_grid, psi_grid)`` (deg), the wake radius and decay, and the gain polynomials."""

    receiver: str
    source: str
    theta_grid: np.ndarray
    psi_grid: np.ndarray
    immersion_area: np.ndarray
    immersion_centroid: np.ndarray
    wake_radius: float
    wake_zeta: float
    gains: Dict[str, np.ndarray]
    gain_status: str
    calibration_report: Dict[str, object]

    def __post_init__(self):
        name = f"pair {self.receiver}<-{self.source}"
        theta = _grid(self.theta_grid, f"{name}.theta_grid")
        psi = _grid(self.psi_grid, f"{name}.psi_grid", periodic=True)
        if theta[0] < 0.0 or theta[-1] > 180.0 + 1e-9:
            raise ValueError(f"{name}.theta_grid must lie in [0, 180] deg")
        shape = (theta.size, psi.size)
        area = _array(self.immersion_area, shape, f"{name}.immersion_area")
        if np.any(area < 0.0):
            raise ValueError(f"{name}.immersion_area must be nonnegative")
        if not (self.wake_radius > 0.0 and self.wake_zeta >= 0.0):
            raise ValueError(f"{name}: wake radius must be positive and zeta nonnegative")
        if set(self.gains) != set(GAIN_NAMES):
            raise ValueError(f"{name}.gains must define {GAIN_NAMES}")
        if self.gain_status not in GAIN_STATUSES:
            raise ValueError(f"{name}.gain_status must be one of {GAIN_STATUSES}")
        if self.gain_status == "B4-fitted" and not self.calibration_report:
            raise ValueError(f"{name}: a B4-fitted pair must carry its calibration_report")
        _set(self, theta_grid=theta, psi_grid=psi, immersion_area=area,
             immersion_centroid=_array(self.immersion_centroid, shape + (3,), f"{name}.immersion_centroid"),
             gains={key: _array(self.gains[key], (6,), f"{name}.gains[{key}]") for key in GAIN_NAMES})


@dataclass(frozen=True, kw_only=True)
class PoweredAeroData:
    """The whole data file: reference geometry (``area``, ``span``, ``cbar``), the air density
    ``rho`` it assumes, the rotor fits, the clean airframe, the receivers, the (receiver, source) pairs and the
    control-surface increments.  ``meta`` says when and from what it was built.  ``fixed_wing`` is its rotors-off
    part."""

    meta: Dict[str, object]
    rho: float
    area: float
    span: float
    cbar: float
    rotors: Tuple[RotorFit, ...]
    clean: CleanData
    receivers: Tuple[ReceiverData, ...]
    pairs: Tuple[PairData, ...]
    surfaces: Tuple[SurfaceIncrements, ...] = ()

    def __post_init__(self):
        _set(self, rotors=tuple(self.rotors),
             receivers=tuple(self.receivers), pairs=tuple(self.pairs), surfaces=tuple(self.surfaces))
        self.fixed_wing  # noqa: B018 -- built once here, so the reference, rho and surfaces are checked
        for kind, records in (("rotor", self.rotors), ("receiver", self.receivers)):
            names = [record.name for record in records]
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {kind} names in {names}")
        receivers, sources = {r.name for r in self.receivers}, {r.name for r in self.rotors}
        if not self.pairs or {p.receiver for p in self.pairs} != receivers or {p.source for p in self.pairs} != sources:
            raise ValueError("every pair must name a known receiver and rotor, and every receiver and rotor a pair")
        by_name = {receiver.name: receiver for receiver in self.receivers}
        for receiver in self.receivers:  # mirrored panels: mirrored suction normals, equal size and polars
            other = by_name.get(receiver.name.replace("left", "right")) if "left" in receiver.name else None
            if other is None:
                continue
            if (not np.allclose(receiver.n_hat, other.n_hat * [1.0, -1.0, 1.0], atol=1e-6)
                    or abs(receiver.chord - other.chord) > 1e-9 or abs(receiver.area_total - other.area_total) > 1e-12
                    or not np.allclose(receiver.cl, other.cl, atol=1e-12)
                    or not np.allclose(receiver.cd, other.cd, atol=1e-12)):
                raise ValueError(f"mirrored receivers {receiver.name}/{other.name} differ")

    @property
    def reference(self) -> AeroReference:
        return AeroReference(area=Area(self.area), span=Length(self.span), cbar=Length(self.cbar))

    @cached_property
    def fixed_wing(self) -> FixedWingData:
        return FixedWingData(reference=self.reference, rho=self.rho, clean=self.clean, surfaces=self.surfaces)

    def receiver(self, name) -> ReceiverData:
        return {receiver.name: receiver for receiver in self.receivers}[name]

    def surface(self, name) -> SurfaceIncrements:
        return {surface.name: surface for surface in self.surfaces}[name]

    def rotor(self, name) -> RotorFit:
        return {rotor.name: rotor for rotor in self.rotors}[name]
