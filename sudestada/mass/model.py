"""The mass-model contract: a vehicle's mass properties at joint angles ``q``.

A mass model gives the mass, COM and central inertia of what it models as a function of the joint angles
(:data:`~sudestada.skeleton.JointAngles`).  Output: body FLU, positions from the MRP (the
``sudestada.vehicle`` frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from sudestada.skeleton import JointAngles
from sudestada.types.units import InertiaTensor, Mass, Position


def parallel_axis(offset) -> InertiaTensor:
    """``|r|^2 I - r r^T``: the parallel-axis term per unit mass."""

    offset = np.asarray(offset, dtype=float)
    return InertiaTensor(np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))


@dataclass(frozen=True)
class MassProperties:
    """``mass`` (kg), ``com`` (body FLU, m, from the MRP) and central ``inertia`` (body FLU, kg m^2)."""

    mass: Mass
    com: Position
    inertia: InertiaTensor


def combine(lumps: Iterable[MassProperties]) -> MassProperties:
    """Lumps in one frame -> their total: ``m = sum m_i``, ``c = sum m_i c_i / m``,
    ``I = sum I_i + m_i parallel_axis(c_i - c)``.  None: zero mass at the origin."""

    lumps = [lump for lump in lumps if lump.mass > 0.0]
    if not lumps:
        return MassProperties(Mass(0.0), Position(np.zeros(3)), InertiaTensor(np.zeros((3, 3))))
    m = np.array([lump.mass for lump in lumps])
    c = np.array([lump.com for lump in lumps])
    com = m @ c / m.sum()
    inertia = np.sum([lump.inertia + lump.mass * parallel_axis(lump.com - com) for lump in lumps], axis=0)
    return MassProperties(Mass(float(m.sum())), Position(com), InertiaTensor(inertia))


class MassModel(Protocol):
    """The mass properties at joint angles ``q``."""

    def __call__(self, q: JointAngles) -> MassProperties: ...
