"""Rotor geometry: the propeller's shape, the single source every rotor model reads it from.

The rotor counterpart of the aero assembly: data and geometric queries only.  A rotor is one blade definition
repeated ``n_blades`` times at equal azimuth, plus an optional hub (an ``aero_assembly.Body``, the spinner).

* **Blade sections** are stacked on their **leading edge**: each ``BladeSection`` puts its LE at ``(r, sweep,
  rake)``, the chord runs from there to the trailing edge at ``twist_deg`` from the disk plane, and the airfoil is
  scaled to the section's thickness ratio (the whole profile, camber included).  This is APC's PE0 definition
  (sweep and rake of the LE parting line, twist about the LE-TE datum), read as-is.
* **Airfoil blends**: a section may sit between two airfoils (``airfoil`` -> ``airfoil_to`` at ``blend``), as APC's
  transitions do; the profiles are mixed point by point before the thickness scaling.
* ``Blade.loft`` says how a CAD emitter interpolates between sections; the analysis readers (BEMT, lifting line)
  take the sections themselves.
* **No mass.**  Mass and inertias are the vehicle's ``Rotor``'s, defined or measured by whoever builds it.

Frame adoption:  the rotor's own, metres.  Hub at the origin, thrust along +Z, blade 0 along +X.  A right-handed rotor
(``hand = +1``) spins about +Z by the right-hand rule: its leading edges face the direction of travel, so blade 0's LE
faces +Y and its chord runs toward -Y and down.  The left-handed rotor is the mirror image in y (not a negated
twist); the blade is always defined right-handed.  Where the rotor sits on an aircraft is not part of it.

"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional

import numpy as np

from sudestada.aero.assembly import Body

PROFILE_POINTS = 61              # per surface, cosine spaced, the same at every section


@lru_cache(maxsize=None)
def _surfaces(name: str, n: int = PROFILE_POINTS):
    """Unit airfoil ``name`` as (x, upper, lower) on ``n`` cosine-spaced chord stations, and its max thickness."""

    import aerosandbox as asb

    coords = np.asarray(asb.Airfoil(name).coordinates, float)
    le = int(coords[:, 0].argmin())
    upper, lower = coords[:le + 1][::-1], coords[le:]            # both LE -> TE
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    yu, yl = np.interp(x, *upper.T), np.interp(x, *lower.T)
    return x, yu, yl, float((yu - yl).max())


@dataclass(frozen=True)
class BladeSection:
    """One station: ``r`` (m) from the axis; ``chord`` (m) LE to TE; ``twist_deg`` of the chord from the disk plane,
    nose up; ``thickness``, the max thickness / chord the airfoil is scaled to; ``airfoil`` by name, blending into
    ``airfoil_to`` by ``blend`` (0: all ``airfoil``); ``sweep`` (m, in the disk plane, + toward the direction of
    travel) and ``rake`` (m, + along the thrust axis) of the leading edge."""

    r: float
    chord: float
    twist_deg: float
    thickness: float
    airfoil: str
    airfoil_to: Optional[str] = None
    blend: float = 0.0
    sweep: float = 0.0
    rake: float = 0.0

    @property
    def le(self) -> np.ndarray:
        return np.array([self.r, self.sweep, self.rake])

    def profile(self) -> tuple[np.ndarray, np.ndarray]:
        """The scaled unit profile, (x, y) along and normal to the chord: upper surface TE -> LE, then the lower one
        LE -> TE (Selig order)."""

        x, yu, yl, t0 = _surfaces(self.airfoil)
        if self.airfoil_to is not None and self.blend:
            _, yu2, yl2, _ = _surfaces(self.airfoil_to)
            yu, yl = (1 - self.blend) * yu + self.blend * yu2, (1 - self.blend) * yl + self.blend * yl2
            t0 = float((yu - yl).max())
        s = self.thickness / t0
        return np.concatenate([x[::-1], x[1:]]), s * np.concatenate([yu[::-1], yl[1:]])

    def points(self) -> np.ndarray:
        """The section in the rotor frame (right-handed, blade 0), (n, 3), Selig order."""

        b = math.radians(self.twist_deg)
        u = np.array([0.0, -math.cos(b), -math.sin(b)])        # chord, LE -> TE
        n = np.array([0.0, -math.sin(b), math.cos(b)])         # profile normal, upper side
        x, y = self.profile()
        return self.le + self.chord * (np.outer(x, u) + np.outer(y, n))


@dataclass(frozen=True)
class Blade:
    """``sections`` root to tip (r increasing); ``loft``: "linear" (ruled between sections) or "spline"."""

    sections: tuple[BladeSection, ...]
    loft: str = "linear"

    def __post_init__(self):
        if len(self.sections) < 2:
            raise ValueError("a blade needs at least two sections")
        if any(b.r <= a.r for a, b in zip(self.sections, self.sections[1:])):
            raise ValueError("blade sections must be in increasing r")
        if self.loft not in ("linear", "spline"):
            raise ValueError(f"unknown loft {self.loft!r}")


@dataclass(frozen=True)
class Rotor:
    """``n_blades`` copies of ``blade`` at equal azimuth (blade k at ``2 pi k / n_blades`` about +Z), a ``hub`` and the
    ``hand`` (+1 right-handed, -1 its mirror)."""

    name: str
    blade: Blade
    n_blades: int
    hub: Optional[Body] = None
    hand: int = 1

    def __post_init__(self):
        if self.hand not in (1, -1):
            raise ValueError("hand is +1 or -1")

    @property
    def radius(self) -> float:
        return self.blade.sections[-1].r

    @property
    def direction(self) -> float:
        """``Rotor.direction`` for this rotor on a +Z thrust axis: the drag-torque couple on the parent
        opposes the spin, so a rotor spinning about +Z (``hand = +1``) has ``direction = -1``."""

        return float(-self.hand)

    def stations(self) -> dict[str, np.ndarray]:
        """Per section: ``r``, ``chord``, ``twist`` (rad), ``thickness``, ``sweep``, ``rake`` (m): what blade-element and
        lifting-line models read."""

        s = self.blade.sections
        return dict(r=np.array([x.r for x in s]), chord=np.array([x.chord for x in s]),
                    twist=np.radians([x.twist_deg for x in s]), thickness=np.array([x.thickness for x in s]),
                    sweep=np.array([x.sweep for x in s]), rake=np.array([x.rake for x in s]))

    def solidity(self) -> float:
        """Blade area over disk area, ``n_blades * integral c dr / (pi R^2)``."""

        st = self.stations()
        return float(self.n_blades * np.trapezoid(st["chord"], st["r"]) / (math.pi * self.radius**2))

    def mirrored(self) -> "Rotor":
        return replace(self, hand=-self.hand, name=f"{self.name} (mirrored)")

    def points(self, blade: int = 0) -> np.ndarray:
        """Every section of blade ``blade`` in the rotor frame, (sections, n, 3)."""

        p = np.stack([s.points() for s in self.blade.sections])
        if self.hand == -1:
            p = p * np.array([1.0, -1.0, 1.0])
        a = 2.0 * math.pi * blade / self.n_blades
        c, s = math.cos(a), math.sin(a)
        return p @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
