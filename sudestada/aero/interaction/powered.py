"""``PoweredAeroModel``: an airframe's aerodynamics with its rotors running and a model for wake interaction.

Starts from the base (:class:`~sudestada.aero.fixed.model.FixedWingAero`, clean airframe plus surfaces) plus
the rotor-on-surface interaction, rotor ``i``'s wake on receiver ``r``'s lifting surface.

Forces and moments are then obtained from the clean whole-aircraft coefficients plus each surface's increment, 
evaluated at the fixed moment reference point and transferred to the airframe origin, plus the interaction increments

Each interaction increment is the difference between the receiver load in the source rotor's wake and the freestream load
on the same immersed area (the key assumption of the model).

All functions are pure, so the simulator built on top of pydrake (github.com/gianfra-t/rotorpy) can work with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from sudestada.types.airframe import AerodynamicWrench, AirframeState
from sudestada.aero.fixed.model import FixedWingAero
from sudestada.aero.interaction.model import InteractionAero
from sudestada.aero.interaction.rotor import RotorWake, wake_state_vectors
from sudestada.aero.interaction.data import PoweredAeroData
from sudestada.skeleton import Skeleton, joints_above
from sudestada.types.units import PerRotor, PerRotorVector

if TYPE_CHECKING:
    from sudestada.vehicle import ControlSurface, Rotor


class PoweredAeroModel:
    """The fixed-wing wrench of ``surfaces`` (the airframe's, in order) plus the interaction
    increments of ``rotors`` (the vehicle's, in order).  Each pair's source rotor is the rotor of that name.

    The interaction tables take one tilt per rotor: the angle of the one joint between its rotating part and the
    root (none: 0).  A rotor under two joints is refused."""

    def __init__(self, data: PoweredAeroData, skeleton: Skeleton, rotors: Mapping[str, Rotor],
                 surfaces: Sequence[ControlSurface]):
        self.data = data
        self.fixed_wing = FixedWingAero(data.fixed_wing, [surface.name for surface in surfaces])
        self.rotor_index = {name: index for index, name in enumerate(rotors)}
        self.interaction = InteractionAero(data, self.rotor_index)
        self._areas = np.array([rotor.area for rotor in rotors.values()])
        parents = {part.name: part.parent for part in skeleton.rotating}
        self._joint = []
        for name in rotors:
            joints = joints_above(skeleton, parents[name])
            if len(joints) > 1:
                raise ValueError(f"{name}: under joints {joints}; the interaction tables take one tilt per rotor")
            self._joint.append(joints[0] if joints else None)

    def rotor_wake(self, state: AirframeState) -> RotorWake:
        """Each rotor's wake at the disk, from its own thrust."""

        axes = np.array([inflow.axis for inflow in state.rotors])
        hub_airspeeds = np.array([inflow.hub_airspeed for inflow in state.rotors])
        v_ax = np.array([inflow.v_axial for inflow in state.rotors])
        v_ip = np.array([np.linalg.norm(inflow.v_inplane) for inflow in state.rotors])
        thrust = np.array([loads.thrust for loads in state.rotor_loads])
        # Glauert momentum v0, skew chi and the wake centreline (U_h - v0 n) / |U_h - v0 n|:
        # dynamics-control/walkthroughs/powered-aero-model.md §3.
        v0, chi, wake_direction = wake_state_vectors(
            thrust, v_ax, v_ip, self.data.rho, self._areas, -hub_airspeeds, axes
        )
        return RotorWake(thrust=PerRotor(thrust), omega=PerRotor(np.array([inflow.rotor_speed for inflow in state.rotors])),
                         v_ax=PerRotor(v_ax), v_ip=PerRotor(v_ip), v0=PerRotor(v0), chi=PerRotor(chi),
                         wake_direction=PerRotorVector(wake_direction), axes=PerRotorVector(axes),
                         tilt=PerRotor(np.array([0.0 if joint is None else state.q[joint] for joint in self._joint])))

    def interaction_wrench(self, state: AirframeState) -> AerodynamicWrench:
        """The interaction increments alone."""

        return AerodynamicWrench(*self.interaction.wrench(state.airspeed, state.w, self.rotor_wake(state)))

    def __call__(self, state: AirframeState) -> AerodynamicWrench:
        return self.fixed_wing(state) + self.interaction_wrench(state)
