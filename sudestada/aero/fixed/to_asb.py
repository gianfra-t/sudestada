"""AeroSandbox emitter for the aero assembly: ``AeroAssembly`` -> ``asb.Airplane`` at joint angles and a deflection set.

The assembly's axes are AeroSandbox's geometry axes, so points pass through unchanged.  ``xyz_ref`` is the assembly's
MRP and ``s_ref, c_ref, b_ref`` its reference.

* **Mirrored surfaces** are emitted as two one-sided wings, both running toward +y (the left one tip to root), so a
  flap on one side deflects on its own.
* **Flaps.** AeroSandbox 4.2's VLM ignores ``ControlSurface`` objects; a deflection is carried by the section
  airfoils (``Airfoil.add_control_surface``: the camber line the VLM meshes).  Where the deflection changes at an
  interior section, a second section ``hinge_gap`` into the deflected panel carries the step.  Gaps of 5-10 mm make
  sliver panels and a near-singular VLM (``CL`` of order -100 without an error); 20 mm was clean on tl-1's chords.
* **Surfaces on a part.**  AeroSandbox's sections twist about the span only, so a surface whose part is turned
  about the span (a tilting wing) is emitted as moved leading edges plus that much twist; one turned about any other
  axis (a folding tip) raises.
* Bodies become ``asb.Fuselage``: drawn, and read by AeroBuildup, not by the VLM.  AeroSandbox orients a section's
  height by the projected +Z, not by the station's ``up``.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Mapping, Optional

import numpy as np

from sudestada.aero.assembly import AeroAssembly, Section, Surface
from sudestada.skeleton import Frame

HINGE_GAP_M = 0.02


@lru_cache(maxsize=None)
def _airfoil(name: str, deflection_deg: float = 0.0, hinge: float = 0.75):
    import aerosandbox as asb

    airfoil = asb.Airfoil(name)
    return airfoil if deflection_deg == 0.0 else airfoil.add_control_surface(deflection=deflection_deg,
                                                                              hinge_point_x=hinge)


def _unturned(surface: Surface) -> Surface:
    """The surface with its placement written into its sections (leading edges moved, twist added)."""

    if surface.part is None:
        return surface
    R = surface.placed.rotation
    if not (np.allclose(R[:, 1], [0.0, 1.0, 0.0]) and np.allclose(R[1], [0.0, 1.0, 0.0])):
        raise NotImplementedError(f"{surface.name}: AeroSandbox twists sections about the span only; this surface "
                                  "is turned about another axis")
    turn = float(np.degrees(np.arctan2(R[0, 2], R[0, 0])))     # about +Y: the trailing edge down, nose up
    sections = tuple(replace(s, le=tuple(surface.point(s.le)), twist_deg=s.twist_deg + turn) for s in surface.sections)
    return replace(surface, sections=sections, part=None, placed=Frame())


def _side(surface: Surface, side: str, deflections: Mapping[str, float], hinge_gap: float):
    """The sections of one side, root to tip, each with its (deflection deg, hinge)."""

    panels = len(surface.sections) - 1
    per_panel = [(0.0, 0.75)] * panels
    for flap in surface.flaps:
        if flap.side in (side, "both") and flap.name in deflections:
            for i in range(*flap.panels):
                per_panel[i] = (float(np.degrees(deflections[flap.name])), flap.hinge)
    out = []
    for i, section in enumerate(surface.sections):
        before = per_panel[i - 1] if i > 0 else per_panel[0]
        after = per_panel[i] if i < panels else per_panel[-1]
        out.append((section, before))
        if before != after:
            nxt = surface.sections[i + 1]
            le, le_next = np.asarray(section.le, float), np.asarray(nxt.le, float)
            t = hinge_gap / np.linalg.norm((le_next - le)[1:])
            if not 0.0 < t < 1.0:
                raise ValueError(f"{surface.name}: panel {i} is shorter than the hinge gap")
            out.append((Section(tuple(le + t * (le_next - le)), section.chord + t * (nxt.chord - section.chord),
                                section.twist_deg + t * (nxt.twist_deg - section.twist_deg), section.airfoil), after))
    return out


def _wing(name, sections):
    import aerosandbox as asb

    return asb.Wing(name=name, symmetric=False, xsecs=[
        asb.WingXSec(xyz_le=np.asarray(s.le, float), chord=s.chord, twist=s.twist_deg,
                     airfoil=_airfoil(s.airfoil, *deflection)) for s, deflection in sections])


def to_airplane(assembly: AeroAssembly, deflections: Optional[Mapping[str, float]] = None,
                q: Optional[Mapping[str, float]] = None, hinge_gap: float = HINGE_GAP_M,
                surfaces: Optional[tuple[str, ...]] = None):
    """``deflections`` (rad) by flap name; ``q`` (rad) by joint name (left out: 0); ``surfaces``: the names to emit
    (all)."""

    import aerosandbox as asb

    deflections = dict(deflections or {})
    unknown = set(deflections) - set(assembly.flaps())
    if unknown:
        raise ValueError(f"unknown flaps {sorted(unknown)}")
    placed = assembly(q)
    wings = []
    for surface in placed.surfaces:
        if surfaces is not None and surface.name not in surfaces:
            continue
        surface = _unturned(surface)
        right = _side(surface, "right", deflections, hinge_gap)
        if not surface.symmetric:
            wings.append(_wing(surface.name, right))
            continue
        wings.append(_wing(f"{surface.name}_right", right))
        left = [(Section((s.le[0], -s.le[1], s.le[2]), s.chord, s.twist_deg, s.airfoil), d)
                for s, d in _side(surface, "left", deflections, hinge_gap)]
        wings.append(_wing(f"{surface.name}_left", left[::-1]))
    fuselages = [asb.Fuselage(name=b.name, xsecs=[
        asb.FuselageXSec(xyz_c=np.asarray(st.center, float), xyz_normal=np.asarray(st.normal, float),
                         width=st.width, height=st.height, shape=st.shape) for st in b.placed_stations()])
        for b in placed.bodies]
    ref = assembly.reference
    return asb.Airplane(name=assembly.name, xyz_ref=assembly.mrp, wings=wings, fuselages=fuselages,
                        s_ref=ref.area, c_ref=ref.cbar, b_ref=ref.span)
