"""File output configuration settings."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class FileType(StrEnum):
    """Supported audio output formats."""

    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"


class FileSettings(BaseModel):
    """Settings controlling where and how segmented audio files are written."""

    model_config = {"extra": "forbid"}

    output_directory: Path = Field(
        default=Path("output"),
        description="Root directory for all output files.",
    )
    output_in_subdirectory: bool = Field(
        default=True,
        description="Group all segments for a source file in a named subdirectory.",
    )
    output_segment_in_subdirectory: bool = Field(
        default=False,
        description="Place each segment in its own subdirectory.",
    )
    name_template: str = Field(
        default="{original_name}_segment_{segment_index}",
        description="Filename template for segment audio files.",
    )
    file_format: FileType = Field(
        default=FileType.WAV,
        description="Audio encoding for output segment files.",
    )
    generate_segment_manifest: bool = Field(
        default=True,
        description="Write a JSON sidecar alongside every segment.",
    )
    generate_run_manifest: bool = Field(
        default=True,
        description="Write a summary JSON for the full segmentation run.",
    )
    segment_manifest_template: str = Field(
        default="{original_name}_segment_{segment_index}",
        description="Filename template for per-segment manifest JSON files.",
    )
    run_manifest_template: str = Field(
        default="{original_name}_manifest",
        description="Filename template for the run-level manifest JSON file.",
    )
