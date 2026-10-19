"""The kinematic skeleton: where the parts are and how they move, and nothing else.

A tree, as in URDF, MuJoCo or Drake: each :class:`Part` names its parent, and the joint belongs to the child.  The
root is ``"body"``: body FLU, origin at the MRP (the ``sudestada.vehicle`` frame).  It is the reference, so
there is nothing to declare for it.

Everything hung on a part (a mass, a station, a section) is given in that part's frame.  :func:`to_root` solves the
tree at joint angles ``q``: a part's frame in body axes, then ``frame.point(p)`` / ``frame.vector(v)``.

Example: a nacelle tilting about +y, its rotor on it.

    skeleton = Skeleton(
        parts=(Part("nacelle_l", parent="body", frame=Frame((0.13, 0.33, 0.0)), joint=Revolute("tilt_l", (0, 1, 0))),),
        rotating=(RotatingPart("rotor_fl", parent="nacelle_l", hub=(0.0, 0.0, 0.06), axis=(0, 0, 1)),))
    to_root(skeleton, {"tilt_l": np.radians(90)}, "nacelle_l").vector((0, 0, 1))    # -> +x: thrust forward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from sudestada.types.units import Axis, Position, Rotation

ROOT = "body"

JointAngles = Mapping[str, float]         # joint name -> angle, rad; a joint left out is at 0


def rotation(axis, angle) -> Rotation:
    """Rotation by ``angle`` about ``axis`` (right-hand rule). "https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula" """
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return Rotation(np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K)


@dataclass(frozen=True)
class Frame:
    """A frame in its parent: ``origin`` and ``rotation`` (its axes as columns, in the parent's axes)."""

    origin: Position = field(default_factory=lambda: Position(np.zeros(3)))
    rotation: Rotation = field(default_factory=lambda: Rotation(np.eye(3)))

    def __post_init__(self):
        object.__setattr__(self, "origin", np.asarray(self.origin, float))
        object.__setattr__(self, "rotation", np.asarray(self.rotation, float))

    def __matmul__(self, child: Frame) -> Frame:
        """``child``, given in this frame, in this frame's parent."""
        return Frame(self.point(child.origin), Rotation(self.rotation @ child.rotation))

    def inverse(self) -> Frame:
        return Frame(Position(-self.rotation.T @ self.origin), Rotation(self.rotation.T))

    def point(self, p) -> Position:
        """Points ``(3,)`` or ``(n, 3)`` of this frame, in the parent's."""
        return Position(np.asarray(p, float) @ self.rotation.T + self.origin)

    def vector(self, v) -> Axis:
        """Directions ``(3,)`` or ``(n, 3)`` of this frame, in the parent's: rotated only."""
        return Axis(np.asarray(v, float) @ self.rotation.T)

    def point_from_parent(self, p) -> Position:
        """Points ``(3,)`` or ``(n, 3)`` of the parent, in this frame: the inverse of :meth:`point`."""
        return Position((np.asarray(p, float) - self.origin) @ self.rotation)

    def vector_from_parent(self, v) -> Axis:
        """Directions ``(3,)`` or ``(n, 3)`` of the parent, in this frame: the inverse of :meth:`vector`."""
        return Axis(np.asarray(v, float) @ self.rotation)


@dataclass(frozen=True)
class Revolute:
    name: str                             # the DOF: its key in the joint angles
    axis: Axis                            # in the part's frame, through its origin


@dataclass(frozen=True)
class Part:
    name: str
    parent: str
    frame: Frame = Frame()                # in the parent's frame, at joint angle 0
    joint: Optional[Revolute] = None      # None: welded


@dataclass(frozen=True)
class RotatingPart:
    name: str
    parent: str
    hub: Position                         # in the parent's frame
    axis: Axis = field(default_factory=lambda: Axis(np.array([0.0, 0.0, 1.0])))   # spin and thrust, parent's frame

    def __post_init__(self):
        object.__setattr__(self, "hub", np.asarray(self.hub, float))
        object.__setattr__(self, "axis", np.asarray(self.axis, float) / np.linalg.norm(self.axis))


@dataclass(frozen=True)
class Skeleton:
    parts: Sequence[Part] = ()
    rotating: Sequence[RotatingPart] = ()

    def part(self, name: str) -> Part:
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(name)

    @property
    def joints(self) -> tuple[str, ...]:
        """The DOFs, in part order."""
        return tuple(part.joint.name for part in self.parts if part.joint is not None)


def local(part: Part, q: JointAngles) -> Frame:
    """The part in its parent, at its joint angle (left out of ``q``: 0)."""
    if part.joint is None:
        return part.frame
    return part.frame @ Frame(rotation=rotation(part.joint.axis, q.get(part.joint.name, 0.0)))


def to_root(skeleton: Skeleton, q: JointAngles, name: str) -> Frame:
    """The frame of part ``name`` in body axes at ``q``."""
    frame = Frame()
    while name != ROOT:
        part = skeleton.part(name)
        frame = local(part, q) @ frame
        name = part.parent
    return frame


def joints_above(skeleton: Skeleton, name: str) -> tuple[str, ...]:
    """The joints between part ``name`` and the root, nearest first."""
    out = []
    while name != ROOT:
        part = skeleton.part(name)
        if part.joint is not None:
            out.append(part.joint.name)
        name = part.parent
    return tuple(out)


def cad_axes(mrp) -> Frame:
    """CAD axes (+X aft, +Y starboard, +Z up) in body FLU, the body origin at ``mrp`` (CAD, m):
    ``p_flu = diag(-1, -1, 1) (p_cad - mrp)``."""
    R = np.diag([-1.0, -1.0, 1.0])
    return Frame(Position(-R @ np.asarray(mrp, float)), Rotation(R))
