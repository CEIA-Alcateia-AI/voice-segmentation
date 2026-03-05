"""Pydantic models for segmentation metadata."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class SegmentManifest(BaseModel):
    """Metadata for a single audio segment."""

    source_file: str = Field(description="Stem of the source audio file.")
    index: int = Field(description="Zero-based position in the segmentation sequence.")
    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")
    duration: float = Field(description="Segment duration in seconds (end - start).")
    output_file: str = Field(description="Path to the segment audio file.")


class SegmentEntry(BaseModel):
    """A compact row inside a :class:`RunManifest`."""

    index: int = Field(description="Zero-based position in the segmentation sequence.")
    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")
    duration: float = Field(description="Segment duration in seconds (end - start).")
    output_file: str = Field(description="Path to the segment audio file.")


class RunManifest(BaseModel):
    """Summary of a full segmentation run for a single source audio file."""

    source_file: str = Field(description="Stem of the source audio file.")
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="UTC timestamp of when the segmentation run completed.",
    )
    total_segments: int = Field(description="Total number of segments produced.")
    segments: list[SegmentEntry] = Field(
        description="Ordered list of all segments produced in this run."
    )

    @classmethod
    def from_segment_manifests(
        cls, source_file: str, manifests: list[SegmentManifest]
    ) -> "RunManifest":
        """Build a RunManifest from the per-segment manifests of a single run.

        Args:
            source_file (str): Stem of the source audio file.
            manifests (list[SegmentManifest]): Per-segment manifests in index order.

        Returns:
            RunManifest: A fully populated RunManifest instance.

        """
        return cls(
            source_file=source_file,
            total_segments=len(manifests),
            segments=[
                SegmentEntry(
                    index=manifest.index,
                    start=manifest.start,
                    end=manifest.end,
                    duration=manifest.duration,
                    output_file=manifest.output_file,
                )
                for manifest in manifests
            ],
        )

    @property
    def output_directory(self) -> Path:
        """Infer the output directory from the first segment's output file.

        Returns:
            Path: Parent directory of the first segment, or the current working
            directory if there are no segments.

        """
        if not self.segments:
            return Path()

        return Path(self.segments[0].output_file).parent
