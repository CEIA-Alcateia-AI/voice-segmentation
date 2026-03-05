"""Duration-based post-processor - merge, overlap padding, and hard-limit filtering."""

from logging import getLogger

from vas._math import seconds_to_samples
from vas.ports.types import Segment, Timestamp
from vas.settings.duration import DurationSettings

logger = getLogger(__name__)


class DurationPostProcessor:
    """Enforces duration rules on raw detected segments."""

    def __init__(
        self,
        sample_rate: int,
        duration_settings: DurationSettings,
    ) -> None:
        """Initialise with the audio sample rate and duration configuration.

        Args:
            sample_rate (int): Sample rate in Hz, used to convert between samples and
                seconds.
            duration_settings (DurationSettings): Duration constraints and merge
                behaviour.

        """
        self._sample_rate = sample_rate
        self._duration = duration_settings

    def process(
        self, segments: list[Segment], audio_length_samples: int
    ) -> list[Timestamp]:
        """Apply merge, overlap padding, and hard-limit filtering.

        Args:
            segments (list[Segment]): Raw segments with start/end in samples.
            audio_length_samples (int): Total audio length in samples, used to clamp
                overlap padding at the end of the file.

        Returns:
            list[Timestamp]: List of validated (start, end) timestamps in seconds.

        """
        merged = self._merge_short_segments(list(segments))
        padded = self._apply_overlap(merged, audio_length_samples)
        return self._filter(padded)

    def _filter(self, segments: list[Segment]) -> list[Timestamp]:
        sample_rate = self._sample_rate
        timestamps: list[Timestamp] = []

        for segment in segments:
            duration = segment.duration_samples / sample_rate

            if duration < self._duration.hard_lower_limit:
                logger.debug(
                    "Discarding %.2f-%.2fs (%.2fs): below hard lower limit.",
                    segment.start / sample_rate,
                    segment.end / sample_rate,
                    duration,
                )
                continue

            if duration > self._duration.hard_upper_limit:
                logger.warning(
                    "Discarding %.2f-%.2fs (%.2fs): exceeds hard upper limit.",
                    segment.start / sample_rate,
                    segment.end / sample_rate,
                    duration,
                )
                continue

            timestamps.append((segment.start / sample_rate, segment.end / sample_rate))

        return timestamps

    def _apply_overlap(
        self, segments: list[Segment], audio_length_samples: int
    ) -> list[Segment]:
        if self._duration.overlap <= 0:
            return segments

        padding = seconds_to_samples(self._duration.overlap / 2, self._sample_rate)
        return [
            Segment(
                start=max(0, segment.start - padding),
                end=min(audio_length_samples, segment.end + padding),
            )
            for segment in segments
        ]

    def _merge_short_segments(self, segments: list[Segment]) -> list[Segment]:
        if not segments:
            return []

        sample_rate = self._sample_rate
        soft_min = int(self._duration.soft_lower_limit * sample_rate)
        hard_max = int(self._duration.hard_upper_limit * sample_rate)
        target = int(
            (self._duration.soft_lower_limit + self._duration.soft_upper_limit)
            / 2
            * sample_rate
        )
        max_gap = int(self._duration.maximum_merge_gap_duration * sample_rate)

        i = 0
        while i < len(segments):
            if segments[i].duration_samples >= soft_min:
                i += 1
                continue

            current = segments[i]
            left = segments[i - 1] if i > 0 else None
            right = segments[i + 1] if i < len(segments) - 1 else None

            can_left, score_left = False, float("inf")
            if left is not None:
                gap = current.start - left.end
                new_duration = current.end - left.start
                if new_duration <= hard_max and gap <= max_gap:
                    can_left, score_left = True, abs(new_duration - target)

            can_right, score_right = False, float("inf")
            if right is not None:
                gap = right.start - current.end
                new_duration = right.end - current.start
                if new_duration <= hard_max and gap <= max_gap:
                    can_right, score_right = True, abs(new_duration - target)

            if not can_left and not can_right:
                logger.debug(
                    (
                        "Segment %d is short but unmergeable. Keeping for hard-limit "
                        "filter."
                    ),
                    i,
                )
                i += 1
                continue

            if can_left and (not can_right or score_left <= score_right):
                logger.debug("Merging segment %d left into %d.", i, i - 1)
                segments[i - 1] = Segment(start=left.start, end=current.end)  # type: ignore[union-attr]
                segments.pop(i)
                i = max(0, i - 1)
            else:
                logger.debug("Merging segment %d right into %d.", i, i + 1)
                segments[i + 1] = Segment(start=current.start, end=right.end)  # type: ignore[union-attr]
                segments.pop(i)

        return segments
