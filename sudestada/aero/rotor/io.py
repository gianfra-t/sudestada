"""Rotor files in and out: APC's PE0 geometry files, read into a :class:`~sudestada.aero.rotor.geometry.Rotor`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from sudestada.aero.assembly import Body, Station
from sudestada.aero.rotor.geometry import Blade, BladeSection, Rotor

INCH = 0.0254
APC_AIRFOILS = {"APC12": "naca4412"}   # APC's own names -> AeroSandbox's (the PE0 notes APC12 = NACA 4412)


def read_pe0(path) -> dict:
    """An APC PE0 file in SI: the station table (``r, chord, sweep, rake, thickness, twist_deg``) and its scalars
    (``radius, hub_radius, hub_transition, blades, airfoils`` [(r, name), ...], and APC's computed ``mass`` and spin
    ``inertia`` of the whole propeller)."""

    text = Path(path).read_text()
    rows = np.array([line.split() for line in text.splitlines()
                     if len(line.split()) == 14 and re.match(r"^\d", line.split()[0])], dtype=float)

    def scalar(key):
        match = re.search(rf"{re.escape(key)}\s*=?\s*([-\d.Ee+]+)", text)
        if match is None:
            raise ValueError(f"{path}: no {key}")
        return float(match.group(1))

    return dict(
        r=rows[:, 0] * INCH, chord=rows[:, 1] * INCH, sweep=rows[:, 5] * INCH, rake=rows[:, 6] * INCH,
        thickness=rows[:, 7], twist_deg=rows[:, 8],
        radius=scalar("RADIUS:") * INCH, hub_radius=scalar("HUBRAD:") * INCH,
        hub_transition=scalar("HUBTRA:") * INCH, blades=int(scalar("BLADES:")),
        airfoils=[(float(r) * INCH, APC_AIRFOILS.get(name, name.lower()))
                  for r, name in re.findall(r"AIRFOIL\d:\s*([\d.]+),\s*(\S+)", text)],
        mass=scalar("TOTAL WEIGHT (Kg)"), inertia=scalar("MOMENT OF INERTIA (Kg-M**2)"))


def from_pe0(path, name: Optional[str] = None, loft: str = "linear", hub_height: Optional[float] = None) -> Rotor:
    """The rotor of an APC PE0 file, right-handed.  Airfoils blend linearly in r over APC's transition.  PE0 gives the
    hub's radius but not its height: with ``hub_height`` (m) the hub is that cylinder about z = 0, otherwise none."""

    pe0 = read_pe0(path)
    (r0, first), (r1, second) = pe0["airfoils"][0], pe0["airfoils"][-1]
    sections = []
    for r, c, sw, rk, t, tw in zip(*(pe0[k] for k in ("r", "chord", "sweep", "rake", "thickness", "twist_deg"))):
        w = float(np.clip((r - r0) / (r1 - r0), 0.0, 1.0)) if r1 > r0 else float(r >= r1)
        sections.append(BladeSection(float(r), float(c), float(tw), float(t), first,
                                     second if w > 0.0 else None, w, float(sw), float(rk)))
    blade = Blade(tuple(sections), loft)

    hub = None if hub_height is None else Body("hub", tuple(
        Station((0.0, 0.0, z), 2 * pe0["hub_radius"], 2 * pe0["hub_radius"], normal=(0.0, 0.0, 1.0))
        for z in (-0.5 * hub_height, 0.5 * hub_height)))
    return Rotor(name or Path(path).stem.removesuffix("-PERF"), blade, pe0["blades"], hub)
