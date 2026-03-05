"""Port: post-processing and rule enforcement."""

from typing import Protocol

from vas.ports.types import Segment, Timestamp


class AudioPostProcessor(Protocol):
    """Enforces duration rules on raw detected segments and converts them to timestamps.

    Implementations apply the business rules that are independent of the
    detection strategy: merging short segments, padding with overlap, and
    discarding segments that fall outside the hard duration limits.
    """

    def process(
        self, segments: list[Segment], audio_length_samples: int
    ) -> list[Timestamp]:
        """Apply duration rules and convert segments to timestamps.

        Args:
            segments (list[Segment]): Raw segments produced by a :class:`CoreProcessor`,
                with start/end in samples.
            audio_length_samples (int): Total length of the source audio in samples,
                used to clamp overlap padding at the boundaries.

        Returns:
            list[Timestamp]: (start, end) timestamps in seconds that pass all rules.

        """
        ...
