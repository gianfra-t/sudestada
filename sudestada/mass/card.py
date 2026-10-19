"""What each part of the skeleton weighs, in that part's own frame.

:class:`PartMasses` is the vehicle's :class:`~sudestada.mass.model.MassModel`: items (:class:`RigidMass`) by name, by
part name, ``"body"`` included, each in its part's frame.  Called with the joint angles it moves every item with
:func:`~sudestada.skeleton.to_root` and sums them in body FLU about the MRP.  

Example (numbers made up), the nacelle drawn from its pivot:

    def nacelle():
        return {"shell": RigidMass(0.060, (0.02, 0.0, 0.00)), "motor": RigidMass(0.045, (0.00, 0.0, 0.03))}

    masses = PartMasses(skeleton, {
        "body": {"battery": RigidMass(0.330, (0.06, 0.0, -0.005), box(0.330, 0.105, 0.035, 0.030)),
                 **mirror("esc", RigidMass(0.020, (0.0, 0.08, 0.0)))},
        "nacelle_l": nacelle(),
        "nacelle_r": nacelle(),
    })
    masses({"tilt_l": 0.0, "tilt_r": 0.0}).inertia          # hover, body FLU
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from sudestada.mass.model import MassProperties, combine
from sudestada.skeleton import JointAngles, Skeleton, to_root
from sudestada.types.units import InertiaTensor, Mass, Position

MIRROR_Y = np.diag([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class RigidMass:
    mass: Mass
    com: Position                                                # in its part's frame
    inertia: InertiaTensor = field(default_factory=lambda: InertiaTensor(np.zeros((3, 3))))   # central, part's frame

    def __post_init__(self):
        object.__setattr__(self, "com", np.asarray(self.com, float))
        object.__setattr__(self, "inertia", np.asarray(self.inertia, float))


def mirror(name: str, item: RigidMass) -> dict[str, RigidMass]:
    """The item and its mirror image about y = 0 of its part's frame, by name."""
    return {name: item, f"{name} (mirror)": replace(item, com=Position(MIRROR_Y @ item.com),
                                                    inertia=InertiaTensor(MIRROR_Y @ item.inertia @ MIRROR_Y))}


def box(mass, lx, ly, lz) -> InertiaTensor:
    """Central inertia of a solid box with its edges along the part's axes."""
    return InertiaTensor(mass / 12 * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2]))


@dataclass(frozen=True)
class PartMasses:
    skeleton: Skeleton
    parts: Mapping[str, Mapping[str, RigidMass]]                 # part name -> item name -> item

    def __call__(self, q: JointAngles) -> MassProperties:
        return combine(self.lumps(q))

    def of(self, name: str) -> MassProperties:
        """Part ``name``'s items summed in its own frame (a multibody solver's body): zero when it has none."""
        return combine(MassProperties(item.mass, item.com, item.inertia) for item in self.parts.get(name, {}).values())

    def lumps(self, q: JointAngles) -> list[MassProperties]:
        """Each item at ``q``, body FLU: ``com -> F.point(com)``, ``inertia -> R I R^T``."""
        out = []
        for name, items in self.parts.items():
            F = to_root(self.skeleton, q, name)
            out += [MassProperties(item.mass, F.point(item.com),
                                   InertiaTensor(F.rotation @ item.inertia @ F.rotation.T)) for item in items.values()]
        return out

    @property
    def mass(self) -> float:
        return sum(item.mass for items in self.parts.values() for item in items.values())
