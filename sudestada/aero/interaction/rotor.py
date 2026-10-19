"""The rotor's wake at the disk, what the interaction model reads: :class:`RotorWake`, from each rotor's thrust.

* ``v0`` — Glauert forward-flight momentum, solved with 12 damped fixed-point steps from the hover guess, driven by
  ``softplus(T)`` so a folded rotor gives ``v0 = 0`` smoothly;
* ``chi`` — wake skew at the disk, ``atan2(V_ip, V_ax + v0)``;
* ``d_hat`` — wake centreline unit vector, ``(U_h - v0 n) / |U_h - v0 n|``,
  finite at ``chi = 90 deg`` where the ``tan(chi)`` form is singular.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sudestada.types.units import PerRotor, PerRotorVector
from sudestada.fits.splines import softplus, smooth_max


# 0.1 g of thrust: far below the model's accuracy, and large enough
# that dv0/dT stays finite when a tumbling rotor crosses T = 0
# (a 1e-9 N corner forces Drake's variable step to zero).
THRUST_EPSILON_N = 1e-3
NORM_EPSILON_MPS = 1e-9
PICARD_STEPS = 12


@dataclasses.dataclass(frozen=True)
class RotorWake:
    """Each rotor's wake at the disk, from the vehicle's own rotor thrust.

    * ``thrust`` (N,): N, along the axis; ``omega`` (N,): rad/s;
    * ``v_ax`` (N,): hub airspeed along the axis; ``v_ip`` (N,): in-plane speed, m/s;
    * ``v0`` (N,): Glauert mean induced velocity, m/s; ``chi`` (N,): wake skew, rad;
    * ``wake_direction`` (N, 3): unit wake centreline, body FLU; ``axes`` (N, 3): thrust axes;
    * ``tilt`` (N,): angle of each rotor's joint (0: none), rad.
    """

    thrust: PerRotor
    omega: PerRotor
    v_ax: PerRotor
    v_ip: PerRotor
    v0: PerRotor
    chi: PerRotor
    wake_direction: PerRotorVector
    axes: PerRotorVector
    tilt: PerRotor

    @property
    def chi_deg(self) -> PerRotor:
        return PerRotor(np.degrees(self.chi))

    @property
    def tilt_deg(self) -> PerRotor:
        return PerRotor(np.degrees(self.tilt))


def induced_velocity(thrust, v_ax, v_ip, rho, area):
    """Glauert mean induced velocity, 12 fixed damped-Picard steps.

    ``T = 2 rho A v0 sqrt(V_ip^2 + (V_ax + v0)^2)``.  The thrust is passed
    through ``softplus`` so ``v0 -> 0`` smoothly as ``T -> 0``, and the result
    is smooth-maxed against zero (a heavily descending rotor is outside the
    calibrated envelope; the simulator must still run).
    """

    thrust = np.asarray(thrust, dtype=float)
    v_ax = np.asarray(v_ax, dtype=float)
    v_ip = np.asarray(v_ip, dtype=float)
    driven = softplus(thrust, THRUST_EPSILON_N)
    denom = 2.0 * rho * area
    v0 = np.sqrt(driven / denom)
    for _ in range(PICARD_STEPS):
        speed = np.sqrt(v_ip * v_ip + (v_ax + v0) ** 2 + NORM_EPSILON_MPS**2)
        target = driven / (denom * speed)
        v0 = 0.5 * (v0 + target)
    return smooth_max(v0, 0.0, NORM_EPSILON_MPS)


def wake_state_vectors(thrust, v_ax, v_ip, rho, area, flow, axis):
    """Vector wake state for arrays: ``(v0, chi, d_hat)`` with ``d_hat`` the
    smooth unit wake centreline ``(flow - v0 axis) / |flow - v0 axis|``."""

    flow = np.asarray(flow, dtype=float)
    axis = np.asarray(axis, dtype=float)
    v0 = induced_velocity(thrust, v_ax, v_ip, rho, area)
    chi = np.arctan2(np.asarray(v_ip, dtype=float), np.asarray(v_ax, dtype=float) + v0)
    numerator = flow - v0[:, None] * axis
    norm = np.sqrt(np.sum(numerator * numerator, axis=-1, keepdims=True) + NORM_EPSILON_MPS**2)
    direction = numerator / norm
    return v0, chi, direction
