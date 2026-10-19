"""Physical quantities as types: the name carries the unit and the shape.

Each is a ``NewType``, so a static checker (pyright, configured in ``pyproject.toml``) tells them apart: a ``Velocity``
is not a ``Position``, and a bare array or float is neither until it is wrapped (``Velocity(v - wind)``). 
"""

from typing import NewType

import numpy as np
from numpy.typing import NDArray

# Scalars
Mass = NewType("Mass", float)                                # kg
Length = NewType("Length", float)                            # m
Area = NewType("Area", float)                                # m^2
Duration = NewType("Duration", float)                        # s
Speed = NewType("Speed", float)                              # m/s
AngularSpeed = NewType("AngularSpeed", float)                # rad/s
Thrust = NewType("Thrust", float)                            # N, along the rotor axis
Torque = NewType("Torque", float)                            # N*m, about the rotor axis
MomentOfInertia = NewType("MomentOfInertia", float)          # kg*m^2, about one axis

# Vectors and tensors, body frame unless a field says otherwise
Position = NewType("Position", NDArray[np.float64])          # (3,) m
Velocity = NewType("Velocity", NDArray[np.float64])          # (3,) m/s
AngularVelocity = NewType("AngularVelocity", NDArray[np.float64])  # (3,) rad/s
Axis = NewType("Axis", NDArray[np.float64])                  # (3,) unit vector
Force = NewType("Force", NDArray[np.float64])                # (3,) N
Moment = NewType("Moment", NDArray[np.float64])              # (3,) N*m
Rotation = NewType("Rotation", NDArray[np.float64])          # (3, 3) proper orthogonal, det +1
InertiaTensor = NewType("InertiaTensor", NDArray[np.float64])  # (3, 3) kg*m^2, symmetric
Quaternion = NewType("Quaternion", NDArray[np.float64])      # (4,) [x, y, z, w], scalar last

# One entry per part, in the vehicle's order
PerRotor = NewType("PerRotor", NDArray[np.float64])          # (N,) one value per rotor
PerRotorVector = NewType("PerRotorVector", NDArray[np.float64])  # (N, 3) one vector per rotor
PerSurface = NewType("PerSurface", NDArray[np.float64])      # (S,) one value per control surface
