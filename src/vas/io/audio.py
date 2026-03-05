"""Local-disk audio loading and writing."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from numpy.typing import NDArray

from vas.settings.file import FileType

_SF_FORMAT: dict[FileType, tuple[str, str | None]] = {
    FileType.WAV: ("WAV", "PCM_16"),
    FileType.FLAC: ("FLAC", None),
    FileType.OGG: ("OGG", "VORBIS"),
}


def load_audio(path: Path) -> tuple[NDArray[np.float32], int]:
    """Load an audio file from disk at its native sample rate.

    The audio is always returned as a 1-D float32 array. Multi-channel files
    are averaged down to mono before returning.

    Args:
        path (Path): Path to the audio file.

    Returns:
        tuple[NDArray[np.float32], int]: A tuple containing the audio data as a
        1-D float32 array and the file's native sample rate in Hz.

    """
    audio: NDArray[Any]
    sample_rate: int
    audio, sample_rate = cast(
        "tuple[NDArray[Any], int]",
        sf.read(path, dtype="float32", always_2d=True),
    )

    # Mix down to mono by averaging channels
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    return audio.astype(np.float32), sample_rate


def write_segment(
    path: Path,
    audio: NDArray[Any],
    sample_rate: int,
    file_type: FileType = FileType.WAV,
) -> None:
    """Write an audio segment to disk.

    Args:
        path (Path): Destination file path (parent directories are created if absent).
        audio (NDArray[Any]): Audio data as a 1-D or 2-D float32 array.
        sample_rate (int): Sample rate in Hz.
        file_type (FileType): Output format (default: FileType.WAV).

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt, subtype = _SF_FORMAT[file_type]
    sf.write(path, audio, sample_rate, format=fmt, subtype=subtype)
