"""Testes para write_segments e funções auxiliares do writer."""

import json
from pathlib import Path

import numpy as np

from voice_segmentation.io.settings import IOSettings
from voice_segmentation.io.writer import write_segments
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray

SR = 16_000


def _audio(duration: float = 10.0) -> AudioArray:
    return (np.sin(np.linspace(0, duration * 440 * 2 * np.pi, int(SR * duration))) * 0.3).astype(
        np.float32
    )


def _settings(tmp_path: Path, **kwargs) -> IOSettings:
    defaults = {
        "output_folder": tmp_path,
        "create_run_subfolder": False,
        "create_segment_subfolders": False,
        "write_run_metadata": False,
        "write_segment_metadata": False,
        "write_segment_spectrograms": False,
        "redo_if_exists": True,
    }
    defaults.update(kwargs)
    return IOSettings(**defaults)


def _duration_settings() -> DurationSettings:
    return DurationSettings()


# ---------------------------------------------------------------------------
# Estrutura de diretórios
# ---------------------------------------------------------------------------


def test_creates_output_folder(tmp_path: Path):
    audio = _audio()
    segs = [(0.0, 5.0), (5.5, 10.0)]
    out = tmp_path / "out"
    io = _settings(out, create_run_subfolder=False)
    run_dir, results = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=segs,
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="TestSegmenter",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    assert run_dir.exists()
    assert len(results) == 2


def test_run_subfolder_uses_source_stem(tmp_path: Path):
    audio = _audio()
    io = _settings(
        tmp_path,
        create_run_subfolder=True,
        create_segment_subfolders=False,
        write_run_metadata=False,
        write_segment_metadata=False,
    )
    run_dir, _ = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0)],
        source_path=Path("my_audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    assert run_dir.name == "my_audio"


def test_run_subfolder_custom_name(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path, create_run_subfolder=True, run_name="custom_run")
    run_dir, _ = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    assert run_dir.name == "custom_run"


def test_segment_subfolders_created(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path, create_segment_subfolders=True)
    run_dir, results = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0), (5.5, 10.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    for seg in results:
        assert seg.path is not None
        assert seg.path.parent.is_dir()


def test_segment_audio_files_written(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path)
    _, results = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0), (5.5, 10.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    for seg in results:
        assert seg.path is not None
        assert seg.path.exists()
        assert seg.path.suffix == ".flac"


def test_wav_format(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path, audio_format="wav")
    _, results = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    assert results[0].path is not None
    assert results[0].path.suffix == ".wav"


# ---------------------------------------------------------------------------
# Metadados de segmento
# ---------------------------------------------------------------------------


def test_segment_metadata_written(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path, write_segment_metadata=True, create_segment_subfolders=True)
    run_dir, _ = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    meta_files = list(run_dir.rglob("metadata.json"))
    assert len(meta_files) == 1


def test_segment_metadata_contains_run_context(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path, write_segment_metadata=True, create_segment_subfolders=True)
    run_dir, _ = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0)],
        source_path=Path("my_src.flac"),
        io_settings=io,
        segmenter_name="MySegmenter",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    meta = json.loads(next(run_dir.rglob("metadata.json")).read_text())
    assert meta["source"]["filename"] == "my_src.flac"
    assert meta["pipeline"]["segmenter"] == "MySegmenter"
    assert "processed_at" in meta
    assert "start_s" in meta
    assert "end_s" in meta
    assert "rms_db" in meta


# ---------------------------------------------------------------------------
# run_metadata.json
# ---------------------------------------------------------------------------


def test_run_metadata_written(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path, write_run_metadata=True)
    run_dir, _ = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0), (5.5, 10.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    meta_path = run_dir / "run_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["result"]["segment_count"] == 2


def test_run_metadata_processed_at_matches_segments(tmp_path: Path):
    """processed_at deve ser o mesmo em run_metadata e em cada segmento."""
    audio = _audio()
    io = _settings(
        tmp_path,
        write_run_metadata=True,
        write_segment_metadata=True,
        create_segment_subfolders=True,
    )
    run_dir, _ = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0), (5.5, 10.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    run_ts = json.loads((run_dir / "run_metadata.json").read_text())["processed_at"]
    for seg_meta in run_dir.rglob("metadata.json"):
        seg_ts = json.loads(seg_meta.read_text())["processed_at"]
        assert seg_ts == run_ts


# ---------------------------------------------------------------------------
# redo_if_exists
# ---------------------------------------------------------------------------


def test_redo_if_exists_false_skips(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path, write_run_metadata=True, redo_if_exists=False)

    # Primeira execução
    run_dir, results_first = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    # Segunda execução com mesmo diretório e redo_if_exists=False
    _, results_second = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    assert results_second == []


def test_redo_if_exists_true_recreates(tmp_path: Path):
    audio = _audio()
    io = _settings(tmp_path, redo_if_exists=True)
    for _ in range(2):
        _, results = write_segments(
            audio=audio,
            sample_rate=SR,
            segments=[(0.0, 5.0)],
            source_path=Path("audio.flac"),
            io_settings=io,
            segmenter_name="X",
            segmenter_settings=None,
            duration_settings=_duration_settings(),
            processing_time_s=0.1,
        )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Espectrogramas
# ---------------------------------------------------------------------------


def test_spectrograms_written(tmp_path: Path):
    audio = _audio()
    io = _settings(
        tmp_path,
        write_segment_spectrograms=True,
        create_segment_subfolders=True,
    )
    run_dir, _ = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0), (5.5, 10.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    spectrograms = list(run_dir.rglob("spectrogram.png"))
    assert len(spectrograms) == 2


def test_spectrograms_recorded_in_run_metadata(tmp_path: Path):
    audio = _audio()
    io = _settings(
        tmp_path,
        write_segment_spectrograms=True,
        write_run_metadata=True,
        create_segment_subfolders=True,
    )
    run_dir, _ = write_segments(
        audio=audio,
        sample_rate=SR,
        segments=[(0.0, 5.0)],
        source_path=Path("audio.flac"),
        io_settings=io,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=_duration_settings(),
        processing_time_s=0.1,
    )
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "spectrogram" in meta["segments"][0]
