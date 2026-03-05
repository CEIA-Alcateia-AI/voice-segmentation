"""Shared domain types used across all ports."""

from dataclasses import dataclass

type Timestamp = tuple[float, float]
"""A (start, end) pair in seconds."""


@dataclass(frozen=True)
class Segment:
    """A contiguous span of audio represented as sample indices."""

    start: int
    end: int

    @property
    def duration_samples(self) -> int:
        """Number of samples in this segment."""
        return self.end - self.start
