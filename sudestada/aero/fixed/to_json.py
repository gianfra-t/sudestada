"""JSON emitter for the aero assembly: the data file a solver outside Python reads the aircraft from.

The file is the assembly as it is, field for field, plus what a reader cannot get without AeroSandbox: every
airfoil's coordinates (unit chord, Selig order, from ``asb.Airfoil``).  Schema 3:

    {"schema": 3, "name": ..., "axes": "+X aft, +Y starboard, +Z up, m, rad", "part": ...,   # the axes' part
     "reference": {"area", "span", "cbar"}, "mrp": [x, y, z],
     "airfoils": {name: [[x, y], ...]},
     "surfaces": [{"name", "symmetric", "part",                               # part: name or null (the axes)
                   "sections": [{"le", "chord", "twist", "airfoil"}],          # in the part's frame; twist about the LE
                   "flaps": [{"name", "panels", "hinge", "side"}]}],          # panels: first, last (excluded)
     "bodies": [{"name", "part", "stations": [{"center", "width", "height", "normal", "up", "shape"}]}],
     "skeleton": [{"name", "parent", "origin", "rotation", "joint"}],         # joint: {"name", "axis"} or null
     "q": {joint: rad}}

A part's frame in its parent is ``origin, rotation`` (columns: its axes) times its joint's rotation about ``axis``;
the root is ``"body"``.

Panels and sections count from 0, as in the assembly.

    python -m sudestada.aero.fixed.to_json tl1_tandem out.json [point]   # from analysis/tl-1
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sudestada.aero.assembly import AeroAssembly

SCHEMA = 3


def _airfoil(name: str) -> list[list[float]]:
    import aerosandbox as asb

    return [[float(x), float(y)] for x, y in np.asarray(asb.Airfoil(name).coordinates)]


def to_dict(assembly: AeroAssembly) -> dict:
    ref = assembly.reference
    names = sorted({s.airfoil for surface in assembly.surfaces for s in surface.sections})
    return {
        "schema": SCHEMA, "name": assembly.name, "axes": "+X aft, +Y starboard, +Z up, m, rad", "part": assembly.part,
        "reference": {"area": ref.area, "span": ref.span, "cbar": ref.cbar},
        "mrp": [float(v) for v in assembly.mrp],
        "airfoils": {name: _airfoil(name) for name in names},
        "surfaces": [{
            "name": s.name, "symmetric": s.symmetric, "part": s.part,
            "sections": [{"le": [float(v) for v in c.le], "chord": c.chord, "twist": math.radians(c.twist_deg),
                          "airfoil": c.airfoil} for c in s.sections],
            "flaps": [{"name": f.name, "panels": list(f.panels), "hinge": f.hinge, "side": f.side} for f in s.flaps],
        } for s in assembly.surfaces],
        "bodies": [{
            "name": b.name, "part": b.part,
            "stations": [{"center": [float(v) for v in st.center], "width": st.width, "height": st.height,
                          "normal": st.axes()[0].tolist(), "up": st.axes()[2].tolist(), "shape": st.shape}
                         for st in b.stations],
        } for b in assembly.bodies],
        "skeleton": [{"name": p.name, "parent": p.parent, "origin": p.frame.origin.tolist(),
                      "rotation": p.frame.rotation.tolist(),
                      "joint": None if p.joint is None else {"name": p.joint.name,
                                                             "axis": [float(v) for v in p.joint.axis]}}
                     for p in assembly.skeleton.parts],
        "q": {name: float(a) for name, a in assembly.q.items()},
    }


def write(assembly: AeroAssembly, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_dict(assembly), indent=1) + "\n")
    return path


if __name__ == "__main__":
    import importlib
    import sys

    family, out = sys.argv[1], sys.argv[2]
    vehicle = importlib.import_module(f"{family}.vehicle")
    layout, _ = vehicle.design_point(*sys.argv[3:])
    write(layout.assembly(), out)
