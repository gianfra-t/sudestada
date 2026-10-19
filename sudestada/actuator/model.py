"""
What a part lets a controller command, and how the commanded value becomes the actual one.

A controllable part declares each of its controllable variables as an :class:`Actuator` field, named for the
variable: eg.: ``Rotor.speed``, ``JointDrive.rate`` and ``JointDrive.torque``, ``ControlSurface.deflection``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

from sudestada.types.units import Duration


class ActuatorModel(Protocol):
    """``d value / dt`` of an actuator given its command and its actual value."""

    def __call__(self, command: float, value: float) -> float: ...


@dataclass(frozen=True)
class FirstOrderLag:
    """``d value / dt = (command - value) / tau``, ``tau`` in s."""

    tau: Duration

    def __call__(self, command, value):
        return (command - value) / self.tau


@dataclass(frozen=True)
class Actuator:
    """A controllable variable: the command is clipped to ``limits`` and the actual value follows it through
    ``model``.  No model: the actual value is the command."""

    limits: Tuple[float, float] = (-np.inf, np.inf)
    model: Optional[ActuatorModel] = None

