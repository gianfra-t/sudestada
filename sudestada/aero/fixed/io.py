"""Fixed-wing data files: JSON in and out of :mod:`sudestada.aero.fixed.data`.  One JSON file per vehicle, written by
an offline fitting pipeline and read by the online model; its keys are the records' field names, and the records
validate themselves on the way in.  ``save_data`` writes any data record, powered-aero included."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from sudestada.types.aero import AeroCoefficients, AeroReference, RateDerivatives
from sudestada.aero.fixed.data import CleanData, FixedWingData, SurfaceIncrements


def _jsonable(value) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def to_dict(data) -> dict:
    return _jsonable(asdict(data))


def _clean_from_dict(clean: dict) -> CleanData:
    return CleanData(**{**clean, "coefficients": AeroCoefficients(**clean["coefficients"]),
                        "rate_derivatives": RateDerivatives(**clean["rate_derivatives"])})


def _surfaces_from_dict(surfaces) -> tuple[SurfaceIncrements, ...]:
    return tuple(SurfaceIncrements(**{**surface, "increments": AeroCoefficients(**surface["increments"])})
                 for surface in surfaces)


def fixed_wing_from_dict(payload: dict) -> FixedWingData:
    """Read a standalone master fixed-wing fit, with the same validation as a powered-aero record."""

    return FixedWingData(**{**payload, "reference": AeroReference(**payload["reference"]),
                           "clean": _clean_from_dict(payload["clean"]),
                           "surfaces": _surfaces_from_dict(payload.get("surfaces", ()))})


def save_data(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(data), indent=2) + "\n")


def load_fixed_wing_data(path) -> FixedWingData:
    """Load the standalone fixed-wing data exported by an offline fit pipeline."""

    return fixed_wing_from_dict(json.loads(Path(path).read_text()))
