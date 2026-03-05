"""No-op post-processor - converts segments to timestamps with no rule enforcement."""

from vas.ports.types import Segment, Timestamp


class NoOpPostProcessor:
    """Post-processor that converts raw segments directly to timestamps in seconds."""

    def __init__(self, sample_rate: int) -> None:
        """Initialise with the audio sample rate used for unit conversion.

        Args:
            sample_rate (int): Sample rate in Hz.

        """
        self._sample_rate = sample_rate

    def process(
        self, segments: list[Segment], audio_length_samples: int
    ) -> list[Timestamp]:
        """Convert sample-based segments directly to second-based timestamps.

        Args:
            segments (list[Segment]): Raw segments with start/end in samples.
            audio_length_samples (int): Unused.

        Returns:
            list[Timestamp]: List of (start, end) timestamps in seconds.

        """
        return [
            (segment.start / self._sample_rate, segment.end / self._sample_rate)
            for segment in segments
        ]
