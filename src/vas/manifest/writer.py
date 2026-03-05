"""Functions for writing manifest files to disk."""

from pathlib import Path

from vas.manifest.models import RunManifest, SegmentManifest


def write_segment_manifest(path: Path, manifest: SegmentManifest) -> None:
    """Serialise a :class:`~vas.manifest.models.SegmentManifest` to a JSON file.

    Args:
        path (Path): Destination path (parent directories are created if absent).
        manifest (SegmentManifest): The manifest to serialise.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(manifest.model_dump_json(indent=2))


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """Serialise a :class:`~vas.manifest.models.RunManifest` to a JSON file.

    Args:
        path (Path): Destination path (parent directories are created if absent).
        manifest (RunManifest): The manifest to serialise.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(manifest.model_dump_json(indent=2))
