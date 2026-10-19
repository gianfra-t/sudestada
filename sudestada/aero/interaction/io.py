"""Powered-aero data files: JSON in and out of :mod:`sudestada.aero.interaction.data` (writing is
:func:`sudestada.aero.fixed.io.save_data`)."""

from __future__ import annotations

import json
from pathlib import Path

from sudestada.aero.fixed.io import _clean_from_dict, _surfaces_from_dict
from sudestada.aero.interaction.data import PairData, PoweredAeroData, ReceiverData
from sudestada.aero.rotor.polynomial import PolynomialRotorModel, RotorEnvelope, RotorFit


def from_dict(payload: dict) -> PoweredAeroData:
    return PoweredAeroData(**{
        **payload,
        "rotors": [RotorFit(**{**rotor, "model": PolynomialRotorModel(**rotor["model"]),
                               "envelope": RotorEnvelope(**rotor["envelope"])})
                   for rotor in payload["rotors"]],
        "clean": _clean_from_dict(payload["clean"]),
        "receivers": [ReceiverData(**receiver) for receiver in payload["receivers"]],
        "pairs": [PairData(**pair) for pair in payload["pairs"]],
        "surfaces": _surfaces_from_dict(payload["surfaces"]),
    })


def load_data(path) -> PoweredAeroData:
    return from_dict(json.loads(Path(path).read_text()))
