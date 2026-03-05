"""Path and filename helpers for output file organisation."""

from pathlib import Path


def format_filename(original_name: str, index: int, template: str, suffix: str) -> str:
    """Render a filename from a template.

    Args:
        original_name (str): Stem of the source audio file.
        index (int): Zero-based segment index.
        template (str): Template string with optional ``{original_name}`` and
            ``{segment_index}`` placeholders.
        suffix (str): File extension *without* the leading dot (e.g. ``"wav"``).

    Returns:
        str: Rendered filename including extension.

    """
    return (
        template.format(original_name=original_name, segment_index=index) + f".{suffix}"
    )


def build_output_path(
    output_directory: Path,
    original_name: str,
    output_in_subdirectory: bool,
    output_segment_in_subdirectory: bool,
    segment_index: int,
) -> Path:
    """Compute the directory in which a segment (or its sidecar) should land.

    Args:
        output_directory (Path): Root output directory.
        original_name (str): Stem of the source audio file.
        output_in_subdirectory (bool): Group all segments in a named subdirectory.
        output_segment_in_subdirectory (bool): Place each segment in its own child
            subdirectory.
        segment_index (int): Zero-based segment index.

    Returns:
        Path: Directory path (not yet created on disk).

    """
    path = output_directory

    if output_in_subdirectory:
        path = path / original_name

        if output_segment_in_subdirectory:
            path = path / f"segment_{segment_index}"

    return path
