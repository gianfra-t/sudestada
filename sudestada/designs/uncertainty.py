"""A descriptive marker for a value whose bounds are known but whose true value is not.

``Uncertainty(low, high, estimate)`` remembers bounds; ``Uncertainty({choices}, estimate=choice)`` remembers a
discrete set of alternatives. It does not do arithmetic, fit models or propagate bounds. Numeric bounds default
to their midpoint; categorical alternatives need an explicit estimate. Pessimism depends on the analysis.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")


@dataclass(frozen=True)
class Uncertainty(Generic[T]):
    low: T | set[T] | frozenset[T]
    high: T | None = None
    estimate: T | None = None

    def __post_init__(self):
        if isinstance(self.low, (set, frozenset)):
            if not self.low or self.high is not None:
                raise ValueError("a choice uncertainty needs a nonempty set and no high bound")
            object.__setattr__(self, "low", frozenset(self.low))
            if self.estimate is not None and self.estimate not in self.low:
                raise ValueError("the estimate must belong to the choice set")
        elif self.high is None:
            raise ValueError("a bounded uncertainty needs both low and high")

    def best_estimate(self) -> T:
        """The supplied estimate, or a numeric midpoint; categorical alternatives require an estimate."""

        if self.estimate is not None:
            return self.estimate
        if isinstance(self.low, frozenset) and len(self.low) == 1:
            return next(iter(self.low))
        if isinstance(self.low, (int, float)) and isinstance(self.high, (int, float)):
            return 0.5 * (self.low + self.high)
        raise ValueError(f"no best estimate between {self.low!r} and {self.high!r}: give one")

    @property
    def ends(self) -> tuple[T, ...]:
        if isinstance(self.low, frozenset):
            return tuple(sorted(self.low, key=repr))
        return cast(tuple[T, T], (self.low, self.high))  # numeric: __post_init__ checked both

    def contains(self, value: T) -> bool:
        """Membership between ordered bounds; categorical ends need not define an ordering."""

        if isinstance(self.low, frozenset):
            return value in self.low
        low, high = cast(tuple[Any, Any], (self.low, self.high))  # ordered bounds, by construction
        return low <= value <= high
