"""Typed vehicle description which includes the skeleton and the models over it.

A :class:`Vehicle` is a :class:`~sudestada.skeleton.Skeleton` (where the parts are and how they move), a mass
model over it, a :class:`Rotor` in each ``RotatingPart`` (by its name), the airframe's aerodynamic model with its
control surfaces, and how each joint is driven (by its name).  

What a controller commands is declared as :class:`~sudestada.actuator.model.Actuator` fields: ``Rotor.speed``,
 ``JointDrive.rate`` and ``JointDrive.torque``,``ControlSurface.deflection``.

Frames: body FLU, positions in metres from the MRP (the skeleton's root).  The MRP is the aerodynamic moment
reference point: ``aero`` reads the flow there and returns its wrench about it.  Inertias are central; axes are unit.

Example: ``Vehicle(skeleton, PartMasses(skeleton, ...), {"rotor_fl": Rotor(...), ...}, aero=..., surfaces=(...),
joints={"tilt_l": JointDrive(rate_limit=1.6)})``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from sudestada.mass.model import MassModel, MassProperties, combine
from sudestada.actuator.model import Actuator, FirstOrderLag
from sudestada.types.airframe import AirframeAeroModel
from sudestada.types.rotor import RotorModel
from sudestada.skeleton import JointAngles, RotatingPart, Skeleton, to_root
from sudestada.types.units import (
    AngularSpeed, Area, Axis, Duration, InertiaTensor, Length, Mass, MomentOfInertia, Position,
)


def axisymmetric_inertia(axis, spin_inertia: float, transverse_inertia: float) -> InertiaTensor:
    """``I_t (1 - a a^T) + J_p a a^T``: the central inertia of a body symmetric about the unit ``axis``."""

    axial = np.outer(axis, axis)
    return InertiaTensor(transverse_inertia * (np.eye(3) - axial) + spin_inertia * axial)


@dataclass(frozen=True, kw_only=True)
class ControlSurface:
    """A control surface: its ``deflection`` actuator (rad, trailing-edge-down positive).  Its aerodynamics are the
    airframe model's (``C + delta C``), which reads the actual deflection."""

    name: str
    deflection: Actuator


@dataclass(frozen=True, kw_only=True)
class JointDrive:
    """How a joint is driven: commanded in ``rate`` or ``torque``; a rate beyond ``rate_limit`` (rad/s) stops the run."""

    rate_limit: AngularSpeed
    rate: Actuator = Actuator(model=FirstOrderLag(Duration(0.05)))
    torque: Actuator = Actuator()


@dataclass(frozen=True, kw_only=True)
class Rotor:
    """What fills a ``RotatingPart``: its rigid body, its aerodynamic model and its motor.

    * ``model``: the rotor's loads from its inflow (:class:`~sudestada.types.rotor.RotorModel`)
    * ``direction``: +1 or -1, RotorPy's ``rotor_directions``;
    * ``speed``: the motor, rad/s: its speed limits and lag, e.g. ``Actuator((0.0, 2000.0), FirstOrderLag(0.05))``;
    * ``mass`` (kg), ``spin_inertia`` ``J_p`` about the axis and ``transverse_inertia`` ``I_t`` across it
      (kg*m^2): an axisymmetric rotor, its COM at the hub. 
    * ``radius`` (m): the disk, for models that need it (the powered-aero wake); ``None`` when unknown.
    """

    model: RotorModel
    direction: float
    speed: Actuator
    mass: Mass = Mass(0.0)
    spin_inertia: MomentOfInertia = MomentOfInertia(0.0)
    transverse_inertia: MomentOfInertia = MomentOfInertia(0.0)
    radius: Optional[Length] = None

    @property
    def area(self) -> Area:
        """Disk area ``pi r^2``; needs ``radius``."""

        assert self.radius is not None, "rotor has no radius"
        return Area(np.pi * self.radius**2)

    def inertia(self, axis) -> InertiaTensor:
        """Locked central inertia about the unit spin ``axis``, in that axis's frame."""

        return axisymmetric_inertia(axis, self.spin_inertia, self.transverse_inertia)


@dataclass(frozen=True)
class Vehicle:
    """The whole vehicle: the skeleton and the models over it."""

    skeleton: Skeleton
    mass: MassModel                                     # the parts; the rotors' masses are the rotors'
    rotors: Mapping[str, Rotor]                         # RotatingPart name -> the rotor in it
    aero: Optional[AirframeAeroModel] = None            # If we only want to model quadcopters at low speeds, airframe aero model is generally neglected.
    surfaces: Sequence[ControlSurface] = ()
    joints: Mapping[str, JointDrive] = field(default_factory=dict)   # joint name -> its drive

    def __post_init__(self):
        if set(self.rotors) != {part.name for part in self.skeleton.rotating}:
            raise ValueError(f"rotors {sorted(self.rotors)} do not fill the rotating parts "
                             f"{sorted(part.name for part in self.skeleton.rotating)}")
        object.__setattr__(self, "surfaces", tuple(self.surfaces))

    @property
    def num_rotors(self) -> int:
        return len(self.rotors)

    def rotating(self, name: str) -> RotatingPart:
        return next(part for part in self.skeleton.rotating if part.name == name)

    def joint_angles(self, angles) -> JointAngles:
        """Joint angles in skeleton order -> by joint name."""
        return dict(zip(self.skeleton.joints, (float(a) for a in angles)))

    def rotor_frames(self, q: JointAngles) -> dict[str, tuple[Position, Axis]]:
        """Each rotor's hub and unit thrust axis at ``q``, body FLU, in ``rotors`` order."""

        out = {}
        for name in self.rotors:
            part = self.rotating(name)
            F = to_root(self.skeleton, q, part.parent)
            out[name] = (F.point(part.hub), F.vector(part.axis))
        return out

    def mass_properties(self, q: JointAngles) -> MassProperties:
        """The parts (``mass``) plus each rotor at its hub, body FLU about the MRP."""

        rotors = [MassProperties(rotor.mass, hub, rotor.inertia(axis))
                  for rotor, (hub, axis) in zip(self.rotors.values(), self.rotor_frames(q).values())]
        return combine([self.mass(q), *rotors])

    def rotor_values(self, name) -> np.ndarray:
        """One numeric rotor field over the rotors, e.g. ``rotor_values("direction")``."""

        return np.array([getattr(rotor, name) for rotor in self.rotors.values()], dtype=float)

    def with_rotors(self, **values) -> Vehicle:
        """A copy with rotor fields replaced, each value one for all rotors or one per rotor (``rotors`` order):
        ``vehicle.with_rotors(model=PolynomialRotorModel.rotorpy(k_eta=1e-5, k_m=1e-7), mass=[0.01, 0.01, 0.005, 0.005])``."""

        def column(value):
            per_rotor = isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.num_rotors
            return list(value) if per_rotor else [value] * self.num_rotors

        columns = {name: column(value) for name, value in values.items()}
        return replace(self, rotors={
            name: replace(rotor, **{field: column[index] for field, column in columns.items()})
            for index, (name, rotor) in enumerate(self.rotors.items())})
