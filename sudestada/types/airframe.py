"""The airframe-aero contract: the airframe's wrench from its flight state.

An aero model is a function ``AirframeState -> AerodynamicWrench`` (:class:`AirframeAeroModel`): in, the airframe's motion,
the flow each part meets, its surface deflections and what its rotors are doing; out, one force and one moment about
the airframe origin, body FLU, or one wrench per part (:func:`wrenches`, :func:`total`).  

The mental model is rotors first, airframe second.  The state carries every rotor's inflow and its loads so a model of
the airframe *with* its rotors running needs nothing else; a model that ignores them is the rotors-off airframe, the default
(:class:`~sudestada.aero.fixed.model.FixedWingAero`).

Rotor-wing interaction is one model built on this (:class:`~sudestada.aero.interaction.powered.PoweredAeroModel`). 

In a dynamic simulator specifically, the vehicle evaluates each rotor's model, builds the :class:`AirframeState` from the rotors'
inflow and loads and every part's flow, calls the airframe's model (``Vehicle.aero``) and applies each wrench to its
part's body at the part's origin (the root's at the MRP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Tuple

import numpy as np

from sudestada.types.rotor import RotorInflow, RotorLoads
from sudestada.skeleton import ROOT, JointAngles, Frame
from sudestada.types.units import AngularVelocity, Force, Moment, PerSurface, Position, Velocity


@dataclass(frozen=True)
class PartFlow:
    """What a part meets, handed by the solver, body FLU: its ``frame`` (origin from the MRP, axes), the airspeed at
    its origin (its velocity minus the wind) and its angular velocity."""

    frame: Frame
    airspeed: Velocity
    w: AngularVelocity


@dataclass(frozen=True)
class AirframeState:
    """Where an airframe model is evaluate.

    * ``position``: the airframe origin, world (z up), m;
    * ``w``: airframe body rates, rad/s;
    * ``airspeed``: the airframe origin's velocity minus the wind, m/s;
    * ``q``: the joint angles, rad, by joint name;
    * ``surface_deflections``: actual deflections, rad, in ``Vehicle.surfaces`` order (what the controller/pilot commanded);
    * ``rotors``, ``rotor_loads``: each rotor's inflow and its model's loads, in ``Vehicle.rotors`` order;
    * ``parts``: each skeleton part's flow, by part name (the root's is the fields above).
    """

    position: Position
    w: AngularVelocity
    airspeed: Velocity
    q: JointAngles
    surface_deflections: PerSurface
    rotors: Tuple[RotorInflow, ...] = ()
    rotor_loads: Tuple[RotorLoads, ...] = ()
    parts: Mapping[str, PartFlow] = field(default_factory=dict)


@dataclass(frozen=True)
class AerodynamicWrench:
    """Force (N) and moment about the airframe origin (N*m) on the airframe, body FLU.  Wrenches add with ``+``."""

    force: Force
    moment: Moment

    def __add__(self, other: AerodynamicWrench) -> AerodynamicWrench:
        return AerodynamicWrench(Force(self.force + other.force), Moment(self.moment + other.moment))


class AirframeAeroModel(Protocol):
    """The airframe's wrench in ``state``: one on the root, about the MRP, or a mapping part name -> its wrench about
    that part's origin (its hinge), body FLU, the root's under ``"body"``."""

    def __call__(self, state: AirframeState) -> AerodynamicWrench | Mapping[str, AerodynamicWrench]: ...


def wrenches(result) -> Mapping[str, AerodynamicWrench]:
    """A model's result by part: a bare wrench is the root's."""
    return result if isinstance(result, Mapping) else {ROOT: result}


def total(state: AirframeState, result) -> AerodynamicWrench:
    """The whole airframe's wrench about the MRP: each part's moved there, ``M + r x F``, r its origin."""
    out = AerodynamicWrench(Force(np.zeros(3)), Moment(np.zeros(3)))
    for name, w in wrenches(result).items():
        r = np.zeros(3) if name == ROOT else state.parts[name].frame.origin
        out = out + AerodynamicWrench(w.force, Moment(w.moment + np.cross(r, w.force)))
    return out

