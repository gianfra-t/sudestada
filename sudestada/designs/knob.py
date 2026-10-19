"""Knob: a design parameter that may change, as opposed to a fixed number or a hard requirement.

A field typed ``Knob[T]`` says, in the definition itself, that its value is a design choice: the ``value`` it has now
and the range ``[low, high]`` it may take (``None``: open on that side).  A plain field is fixed.  Nothing here
optimises; it marks what may move and how far, so the limits live next to the value.

:class:`AircraftData` is a definition of the aircraft made of knobs (and other definitions): a configuration's
layout, its mass layout.  A design point is one of them tweaked: built from the baseline, its knobs ``set``
(:meth:`Knob.set` refuses a value outside the range); :meth:`AircraftData.violations` walks every knob and
:meth:`AircraftData.values` reads them back.

A :class:`Selection` is the discrete knob: one of a named map of choices (a part, a model), picked by name
(:meth:`Selection.pick` refuses a name not in the map).  A choice may itself be a definition holding knobs and
case assumptions; optional bound discovery/selection belongs to :mod:`sudestada.designs.campaign`, above concrete fits.

Model uncertainty (a coefficient's bound corners) is not a knob: the design does not choose it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, ClassVar, Generic, Mapping, Optional, Protocol, TypeVar


class Ordered(Protocol):
    def __le__(self, other: Any, /) -> bool: ...


T = TypeVar("T", bound=Ordered)
V = TypeVar("V")


@dataclass
class Knob(Generic[T]):
    """One plain value and its range: never another knob or a definition (nesting is ``AircraftData``'s)."""

    value: T
    low: Optional[T] = None
    high: Optional[T] = None

    def __post_init__(self):
        for name in ("value", "low", "high"):
            _plain(getattr(self, name), name)

    @property
    def within(self) -> bool:
        return (self.low is None or self.low <= self.value) and (self.high is None or self.value <= self.high)

    def set(self, value: T) -> None:
        """Set the value; outside ``[low, high]`` raises."""

        _plain(value, "value")
        if (self.low is not None and not self.low <= value) or (self.high is not None and not value <= self.high):
            raise ValueError(f"{value} outside [{self.low}, {self.high}]")
        self.value = value


@dataclass
class Selection(Generic[V]):
    """``name``d choice among ``options`` (choice name -> value); the first is picked until :meth:`pick` says
    otherwise."""

    name: str
    options: Mapping[str, V]
    choice: str = ""  # empty: the first option

    def __post_init__(self):
        if not self.options:
            raise ValueError(f"{self.name}: nothing to select from")
        self.pick(self.choice or next(iter(self.options)))

    @property
    def value(self) -> V:
        return self.options[self.choice]

    def pick(self, choice: str) -> None:
        """Pick ``choice``; one not in ``options`` raises."""

        if choice not in self.options:
            raise ValueError(f"{self.name}: {choice!r} not one of {list(self.options)}")
        self.choice = choice


def _plain(value, name: str) -> None:
    if isinstance(value, (Knob, Selection, AircraftData)):
        raise TypeError(f"a knob's {name} is a plain value, not {type(value).__name__}")


class AircraftData:
    """A design definition: a dataclass whose fields are knobs, selections, other definitions, or fixed values.

    A selection counts as its picked value for design checks. Model uncertainty is handled by
    :mod:`sudestada.designs.campaign` at the case layer.
    """

    __dataclass_fields__: ClassVar[dict[str, Any]]  # every subclass is a dataclass

    def violations(self, prefix: str = "") -> list[str]:
        """Every knob outside its range, nested definitions' own checks included, as text."""

        out = []
        for f in fields(self):
            item, name = getattr(self, f.name), f"{prefix}{f.name}"
            if isinstance(item, Knob):
                if not item.within:
                    out.append(f"{name} = {item.value} outside [{item.low}, {item.high}]")
            elif isinstance(item, Selection) and isinstance(item.value, AircraftData):
                out += item.value.violations(f"{name}.")
            elif isinstance(item, AircraftData):
                out += item.violations(f"{name}.")
        return out

    def values(self) -> dict[str, Any]:
        """Every knob value and every selection's choice (with its value's knobs), nested as the definitions are."""

        out = {}
        for f in fields(self):
            item = getattr(self, f.name)
            if isinstance(item, Knob):
                out[f.name] = item.value
            elif isinstance(item, Selection):
                out[f.name] = item.choice
                if isinstance(item.value, AircraftData) and item.value.values():
                    out[f"{f.name}.{item.choice}"] = item.value.values()
            elif is_dataclass(item) and isinstance(item, AircraftData):
                out[f.name] = item.values()
        return out
