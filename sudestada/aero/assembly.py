"""Aero assembly -> the aircraft as the air sees it, the single source every model reads its shape from.

Building blokcs:
* **lifting surfaces**: section stacks (leading edge, chord, twist, airfoil), mirrored about y = 0 or not, with
  their control surfaces (``Flap``); one that names a skeleton ``part`` moves with it (a tilting wing, a folding
  tip);
* **bodies**: station stacks (centre, normal, up, width, height, superellipse exponent): fuselage, nacelles;
* the **skeleton**, shared with the mass card and the vehicle, and ``q``, the joint angles the assembly is placed at.

Plus the reference the coefficients are normalised on (``AeroReference``: S, b, cbar) and the moment
reference point ``mrp``.

**Placement.**  An assembly is an :class:`AeroGeometry`: called with joint angles ``q`` it returns itself placed
there.  A component that names a ``part`` is given in that part's frame; one that names none is in the assembly's
axes.  Placing hands each component its frame in the assembly's axes (``placed``: ``to_root(part)`` seen from
``to_root(assembly.part)``; :meth:`Surface.strips`, :meth:`Body.placed_stations` and the emitters apply it).  The
result is the last step before the air (speed, alpha, beta, rates), which is the solver's operating point.

Axes: those of the skeleton part ``part``, which should be AeroSandbox's geometry axes: +X aft, +Y starboard, +Z up,
metres, origin at the nose tip (a welded part with frame ``skeleton.cad_axes(mrp)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Protocol

import numpy as np

from sudestada.types.aero import AeroReference
from sudestada.skeleton import ROOT, JointAngles, Frame, Skeleton, to_root
from sudestada.types.units import Area, Length

SIDES = ("right", "left", "both")


def _vec(value) -> np.ndarray:
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class Section:
    """One station of a lifting surface, in its part's frame: leading edge ``le`` (m), ``chord`` (m) along +X before
    twist, ``twist_deg`` about the leading edge, and the
    ``airfoil`` by name."""

    # TODO: allow any section shape: an arbitrary curve, from CST parameters or a point cloud for eg., not just a
    # named airfoil.
    le: tuple[float, float, float]
    chord: float
    twist_deg: float = 0.0
    airfoil: str = "naca0012"


@dataclass(frozen=True)
class Flap:
    """A trailing-edge control surface over panels ``panels = (first, last)`` (panel i joins sections i and i+1,
    ``last`` excluded), hinged at ``hinge`` of the local chord, on one ``side`` of a mirrored surface or ``both``.
    Its ``name`` is the deflection channel; trailing edge down positive."""

    name: str
    panels: tuple[int, int]
    hinge: float = 0.75
    side: str = "right"


@dataclass(frozen=True)
class Surface:
    """A lifting surface: ``sections`` root to tip (the right-hand side when ``symmetric``), in the frame of
    ``part`` (None: the assembly's axes).  ``placed`` is that frame in the assembly's axes, set by the assembly that
    holds it.  A mirrored surface's left side is the mirror image (y -> -y) of its placed right side."""

    name: str
    sections: tuple[Section, ...]
    symmetric: bool = True
    flaps: tuple[Flap, ...] = ()
    part: Optional[str] = None
    placed: Frame = field(default_factory=Frame, compare=False, repr=False)

    def point(self, p) -> np.ndarray:
        """Points as placed."""
        return self.placed.point(p)

    def vector(self, v) -> np.ndarray:
        """Directions as placed."""
        return self.placed.vector(v)

    def __post_init__(self):
        if len(self.sections) < 2:
            raise ValueError(f"{self.name}: a surface needs at least two sections")
        for flap in self.flaps:
            first, last = flap.panels
            if not 0 <= first < last <= len(self.sections) - 1:
                raise ValueError(f"{self.name}: flap {flap.name} panels {flap.panels} outside the surface")
            if flap.side not in SIDES or (flap.side != "right" and not self.symmetric):
                raise ValueError(f"{self.name}: flap {flap.name} side {flap.side!r}")

    # ---- strips: every integral below, and the mass card, use them ----
    def strips(self, per_panel: int = 50):
        """Midpoint strips of one side: leading edges (n, 3) in body axes, chords (n,), widths (n,) along the section
        line."""

        les, chords, widths = [], [], []
        for a, b in zip(self.sections[:-1], self.sections[1:]):
            t = (np.arange(per_panel) + 0.5) / per_panel
            le_a, le_b = _vec(a.le), _vec(b.le)
            les.append(le_a + np.outer(t, le_b - le_a))
            chords.append(a.chord + t * (b.chord - a.chord))
            widths.append(np.full(per_panel, np.linalg.norm((le_b - le_a)[1:]) / per_panel))
        return self.point(np.concatenate(les)), np.concatenate(chords), np.concatenate(widths)

    def chord_axis(self) -> np.ndarray:
        """The chord direction before twist (+X of its part), as placed."""

        return self.vector([1.0, 0.0, 0.0])

    def area(self) -> float:
        """Planform area along the sections, both sides when mirrored."""

        _, c, w = self.strips()
        return (2.0 if self.symmetric else 1.0) * float(c @ w)

    def span(self) -> float:
        """Tip-to-tip extent in y when mirrored; the length along the sections otherwise (a fin's height)."""

        if self.symmetric:
            return 2.0 * float(np.abs(self.point([s.le for s in self.sections])[:, 1]).max())
        return float(self.strips()[2].sum())

    def mac(self) -> float:
        _, c, w = self.strips()
        return float((c * c) @ w / (c @ w))

    def mac_c4(self) -> np.ndarray:
        """Quarter chord of the mean aerodynamic chord, body axes; on y = 0 when mirrored."""

        le, c, w = self.strips()
        point = (c * w) @ le / (c @ w) + 0.25 * self.mac() * self.chord_axis()
        if self.symmetric:
            point[1] = 0.0
        return point


@dataclass(frozen=True)
class Station:
    """One station of a body: ``center`` (m), unit ``normal`` (the body axis there), ``width`` and ``height`` (m),
    the superellipse ``shape`` exponent (2: ellipse, large: box), and ``up``, the direction ``height`` is measured
    along, all in its part's frame.  ``up`` None: +Z projected onto the station's plane (+X when the normal is along
    Z); :meth:`Body.placed_stations` turns it, so a section keeps its roll however it turns."""

    center: tuple[float, float, float]
    width: float
    height: float
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    shape: float = 2.0
    up: Optional[tuple[float, float, float]] = None

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit ``(normal, side, up)``: right-handed, ``side`` along the width."""

        n = _vec(self.normal) / np.linalg.norm(self.normal)
        for guess in ((self.up,) if self.up is not None else ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))):
            u = _vec(guess) - (_vec(guess) @ n) * n
            if np.linalg.norm(u) > 1e-9:
                u = u / np.linalg.norm(u)
                return n, np.cross(u, n), u
        raise ValueError(f"station at {self.center}: up {self.up} is along the normal")


@dataclass(frozen=True)
class Body:
    """A body: ``stations`` in the frame of ``part`` (None: the assembly's axes); ``placed`` that frame in the
    assembly's axes, set by the assembly."""

    name: str
    stations: tuple[Station, ...]
    part: Optional[str] = None
    placed: Frame = field(default_factory=Frame, compare=False, repr=False)

    def placed_stations(self) -> tuple[Station, ...]:
        """The stations as placed, each with its ``up`` fixed (a section keeps its roll however it turns)."""

        if self.part is None:
            return self.stations
        F, out = self.placed, []
        for st in self.stations:
            n, _, up = st.axes()
            out.append(replace(st, center=tuple(F.point(st.center)), normal=tuple(F.vector(n)), up=tuple(F.vector(up))))
        return tuple(out)


@dataclass(frozen=True)
class AeroAssembly:
    """``surfaces`` and ``bodies`` on the ``skeleton``, in the axes of its part ``part``, placed at the joint angles
    ``q`` (rad by joint name; left out: 0)."""

    name: str
    surfaces: tuple[Surface, ...]
    reference: AeroReference
    mrp: np.ndarray = field(repr=False)
    bodies: tuple[Body, ...] = ()
    skeleton: Skeleton = Skeleton()
    part: str = ROOT
    q: JointAngles = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "mrp", _vec(self.mrp))
        names = [p.name for p in (*self.surfaces, *self.bodies)]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")
        q = {joint: float(self.q.get(joint, 0.0)) for joint in self.skeleton.joints}
        axes = to_root(self.skeleton, q, self.part).inverse()

        def placed(component):
            if component.part is None:
                return replace(component, placed=Frame())
            return replace(component, placed=axes @ to_root(self.skeleton, q, component.part))

        object.__setattr__(self, "surfaces", tuple(placed(s) for s in self.surfaces))
        object.__setattr__(self, "bodies", tuple(placed(b) for b in self.bodies))

    def surface(self, name: str) -> Surface:
        return next(s for s in self.surfaces if s.name == name)

    def flaps(self) -> tuple[str, ...]:
        """Every deflection channel, in surface order."""

        return tuple(f.name for s in self.surfaces for f in s.flaps)

    def __call__(self, q: Optional[Mapping[str, float]] = None) -> "AeroAssembly":
        """The assembly with the joints at ``q`` (rad by joint name; left out: 0)."""

        return replace(self, q=dict(q or {}))


class AeroGeometry(Protocol):
    """The aircraft's aerodynamic geometry at joint angles ``q`` (the mass model's counterpart)."""

    def __call__(self, q: JointAngles) -> AeroAssembly: ...


def area_weighted_reference(surfaces) -> tuple[AeroReference, np.ndarray]:
    """One common choice for several lifting surfaces: S their total area, cbar the area-weighted MAC, b the largest
    span, the MRP the area-weighted mean of their MAC quarter chords."""

    surfaces = tuple(surfaces)
    areas = np.array([s.area() for s in surfaces])
    area = float(areas.sum())
    cbar = float(areas @ np.array([s.mac() for s in surfaces]) / area)
    mrp = areas @ np.array([s.mac_c4() for s in surfaces]) / area
    return AeroReference(area=Area(area), span=Length(max(s.span() for s in surfaces)), cbar=Length(cbar)), mrp
