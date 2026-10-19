"""The powered-aero rotor: the rotor polynomial and its fit.

* :class:`PolynomialRotorModel` is a :class:`~sudestada.types.rotor.RotorModel`: the smooth
  forward-flight fit and, with RotorPy's terms only (:meth:`PolynomialRotorModel.rotorpy`), RotorPy's
  native rotor, kept for compatibility.
* :class:`RotorFit` is one fitted rotor as the data file stores it: the polynomial, the envelope it was fitted over
  (:class:`RotorEnvelope`), its residuals and where it came from.  ``RotorFit.model`` is that rotor's
  ``Rotor.model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

from sudestada.types.rotor import RotorInflow, RotorLoads
from sudestada.types.units import Force, Moment, Thrust, Torque
from sudestada.fits.splines import cross3

if TYPE_CHECKING:
    from sudestada.aero.interaction.data import PoweredAeroData


def _coefficient(rotorpy=None, fitted=True):
    """A polynomial coefficient, 0 unless given.  ``rotorpy`` names the same term in RotorPy's ``k_*`` params;
    ``fitted`` coefficients are the fit's."""

    return field(default=0.0, metadata={"rotorpy": rotorpy, "fitted": fitted})


@dataclass(frozen=True, kw_only=True)
class PolynomialRotorModel:
    """One rotor's dimensional polynomial, one coefficient per term (0 unless given):

    ``T = a0 W|W| + a1 W Vax + a2 Vax^2 + a3 Vip^2``, ``Q = b0 W|W| + b1 W Vax + b2 Vax^2 + b3 Vip^2``,
    hub force ``H = -(h0 W + h1 Vax) Vip - d (s0 W + s1 Vax) (Vip x n) - z0 W Vax n``,
    hub moment ``Mh = -(m0 W + m1 Vax) (Vip x n) + d (p0 W + p1 Vax) Vip``, with ``d`` the spin direction.

    It is the smooth forward-flight fit of the powered-aero model, and, with only the terms RotorPy has
    (:meth:`rotorpy`), RotorPy's native rotor.  ``z0`` is RotorPy's axial hub drag ``k_z``; the fit has no such
    term.

    ``p0, p1`` carry the p-factor moment about the in-plane flow direction and ``s0, s1`` the side force; both are
    spin-odd: they flip with ``d``.  
    """

    a0: float = _coefficient("k_eta")
    a1: float = _coefficient()
    a2: float = _coefficient()
    a3: float = _coefficient("k_h")
    b0: float = _coefficient("k_m")
    b1: float = _coefficient()
    b2: float = _coefficient()
    b3: float = _coefficient()
    h0: float = _coefficient("k_d")
    h1: float = _coefficient()
    z0: float = _coefficient("k_z", fitted=False)
    m0: float = _coefficient("k_flap")
    m1: float = _coefficient()
    s0: float = _coefficient()
    s1: float = _coefficient()
    p0: float = _coefficient()
    p1: float = _coefficient()

    @classmethod
    def rotorpy(cls, *, k_eta, k_m, k_d=0.0, k_z=0.0, k_h=0.0, k_flap=0.0) -> PolynomialRotorModel:
        """RotorPy's native rotor (``Multirotor``): ``T = k_eta W|W| + k_h |Vip|^2``, ``Q = k_m W|W|``,
        hub force ``-W (k_d Vip + k_z Vax n)``, hub moment ``-k_flap W (Vip x n)``.  Good at low speed.
        """

        values = {"k_eta": k_eta, "k_m": k_m, "k_d": k_d, "k_z": k_z, "k_h": k_h, "k_flap": k_flap}
        return cls(**{name: values[alias] for name, alias in ROTORPY_NAMES.items()})

    @classmethod
    def from_data(cls, data: PoweredAeroData, name: str) -> PolynomialRotorModel:
        """The powered-aero data file's fit of rotor ``name``."""

        fit = data.rotor(name).model
        return cls(**{coefficient.name: getattr(fit, coefficient.name) for coefficient in fields(cls)})

    def static(self) -> PolynomialRotorModel:
        """Only the static thrust and torque terms ``a0``, ``b0``: RotorPy's rotor with ``aero=False``."""

        return PolynomialRotorModel(a0=self.a0, b0=self.b0)

    def speed_for_thrust(self, thrust):
        """The speed that makes ``thrust`` with no inflow: ``T = a0 W|W|`` inverted (``cmd_motor_thrusts``)."""

        ratio = thrust / self.a0
        return float(np.sign(ratio) * np.sqrt(np.abs(ratio)))

    def __call__(self, inflow: RotorInflow) -> RotorLoads:
        w, v_ax, v_ip, axis, d = inflow.rotor_speed, inflow.v_axial, inflow.v_inplane, inflow.axis, inflow.direction
        # Omega*|Omega| (== Omega**2 for Omega >= 0): a reversed rotor brakes.
        signed_square = w * abs(w)
        v_ip2 = float(v_ip @ v_ip)
        v_ip_x_n = cross3(v_ip, axis)
        thrust = self.a0 * signed_square + self.a1 * w * v_ax + self.a2 * v_ax**2 + self.a3 * v_ip2
        torque = self.b0 * signed_square + self.b1 * w * v_ax + self.b2 * v_ax**2 + self.b3 * v_ip2
        hub_force = (-(self.h0 * w + self.h1 * v_ax) * v_ip - d * (self.s0 * w + self.s1 * v_ax) * v_ip_x_n
                     - self.z0 * w * v_ax * axis)
        hub_moment = -(self.m0 * w + self.m1 * v_ax) * v_ip_x_n + d * (self.p0 * w + self.p1 * v_ax) * v_ip
        return RotorLoads(thrust=Thrust(thrust), torque=Torque(torque), force=Force(thrust * axis + hub_force),
                          moment=Moment(d * torque * axis + hub_moment))


# The fitted coefficients (the data file's); the RotorPy ``k_*`` name of each term RotorPy has.
FITTED_ROTOR_COEFFICIENTS = tuple(coefficient.name for coefficient in fields(PolynomialRotorModel)
                                  if coefficient.metadata["fitted"])
ROTORPY_NAMES = {coefficient.name: coefficient.metadata["rotorpy"] for coefficient in fields(PolynomialRotorModel)
                 if coefficient.metadata["rotorpy"]}


def _bounds(value, name):
    low, high = (float(bound) for bound in value)
    if not (np.isfinite(low) and np.isfinite(high) and low <= high):
        raise ValueError(f"{name} must be finite [low, high] with low <= high")
    return low, high


@dataclass(frozen=True, kw_only=True)
class RotorEnvelope:
    """The box a rotor fit was fitted over: speed ``omega`` (rad/s), axial ``v_ax`` and in-plane ``v_ip`` airspeed
    (m/s), each ``(low, high)``.  Outside it the fit extrapolates; the simulation still runs."""

    omega: Tuple[float, float]
    v_ax: Tuple[float, float]
    v_ip: Tuple[float, float]

    def __post_init__(self):
        for name in ("omega", "v_ax", "v_ip"):
            object.__setattr__(self, name, _bounds(getattr(self, name), f"envelope.{name}"))

    def excess(self, omega, v_ax, v_ip) -> Dict[str, float]:
        """Signed distance outside each violated bound (negative below, positive above); empty inside."""

        result = {}
        for name, value in (("omega", omega), ("v_ax", v_ax), ("v_ip", v_ip)):
            low, high = getattr(self, name)
            if not low <= value <= high:
                result[name] = float(value - low if value < low else value - high)
        return result


@dataclass(frozen=True, kw_only=True)
class RotorFit:
    """One fitted rotor: its polynomial ``model``, the ``envelope`` it was fitted over, the fit's ``fit_rms``
    residuals per load and its ``provenance``.  ``name`` is the rotor it was fitted for."""

    name: str
    model: PolynomialRotorModel
    envelope: RotorEnvelope
    fit_rms: Dict[str, float]
    provenance: Dict[str, object]

    def __post_init__(self):
        if not self.name:
            raise ValueError("rotor fit needs a name")
        if not (self.model.a0 > 0.0 and self.model.b0 > 0.0):
            raise ValueError(f"rotor fit {self.name}: a0 and b0 must be positive")
        if not self.provenance:
            raise ValueError(f"rotor fit {self.name}: provenance must say where the numbers came from")
