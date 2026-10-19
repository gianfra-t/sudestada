"""The aerodynamic coefficients types."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Generic, TypeVar

from sudestada.types.units import Area, Length

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class AeroCoefficients(Generic[T]):
    """The six whole-aircraft coefficients, FRD: lift ``CL``, drag ``CD`` and side force ``CY`` in wind axes, roll
    ``Cl``, pitch ``Cm`` and yaw ``Cn`` in body axes.  Increments add with ``+``.

    Generic in what each coefficient holds because the same six appear at every stage of the pipeline:
    ``AeroCoefficients[np.ndarray]`` is fitted data (one table per coefficient, on a grid),
    ``AeroCoefficients[RectSpline]`` the interpolants built on those tables, and ``AeroCoefficients[float]`` one flow
    condition read from them.  One type carries the names, the axes and ``+`` for all three, and the parameter says
    which one is held, so a type checker flags a table passed where a value is expected."""

    CL: T
    CD: T
    CY: T
    Cl: T
    Cm: T
    Cn: T

    def __add__(self, other: AeroCoefficients[T]) -> AeroCoefficients[T]:
        return AeroCoefficients(*(getattr(self, name) + getattr(other, name) for name in COEFFICIENT_NAMES))

    def map(self, f: Callable[[T], U]) -> AeroCoefficients[U]:
        """``f`` applied to each of the six."""
        return AeroCoefficients(*(f(getattr(self, name)) for name in COEFFICIENT_NAMES))


@dataclass(frozen=True)
class RateDerivatives:
    """Body-rate damping derivatives, FRD, per nondimensional rate ``p b / 2V``, ``q cbar / 2V``, ``r b / 2V``."""

    CYp: float
    CYr: float
    CLq: float
    Clp: float
    Clr: float
    Cmq: float
    Cnp: float
    Cnr: float


COEFFICIENT_NAMES = tuple(field.name for field in fields(AeroCoefficients))


@dataclass(frozen=True)
class AeroReference:
    """What the coefficients are normalised on: area ``S`` (m^2), span ``b`` and mean chord ``cbar`` (m).  They are
    taken about the airframe origin, which is the moment reference point; the flow is read there too."""

    area: Area
    span: Length
    cbar: Length
