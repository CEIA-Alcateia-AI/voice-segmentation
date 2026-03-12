"""Testes de integração para Pipeline.run() — WebRTCPipeline e SilencePipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from voice_segmentation import (
    IOSettings,
    RunResult,
    SilencePipeline,
    WebRTCAggressiveness,
    WebRTCPipeline,
)
from voice_segmentation.exceptions import EmptySegmentationError
from voice_segmentation.types import AudioArray

SR = 16_000


def _speech_audio() -> AudioArray:
    """3 rafadas de seno (440 Hz) separadas por silêncio."""
    burst = (np.sin(np.linspace(0, 3 * 440 * 2 * np.pi, SR * 3)) * 0.6).astype(np.float32)
    gap = np.zeros(SR, dtype=np.float32)
    return np.concatenate([gap, burst, gap, burst, gap, burst, gap]).astype(np.float32)


def _silence_pipeline() -> SilencePipeline:
    return SilencePipeline(
        soft_lower=1.0,
        soft_upper=10.0,
        hard_lower=0.5,
        hard_upper=20.0,
        max_gap=0.5,
    )


# ---------------------------------------------------------------------------
# Retorno de RunResult — campos básicos
# ---------------------------------------------------------------------------


def test_run_returns_run_result():
    pipeline = _silence_pipeline()
    result = pipeline.run(_speech_audio(), sample_rate=SR)
    assert isinstance(result, RunResult)


def test_run_no_output_segments_have_no_path():
    pipeline = _silence_pipeline()
    result = pipeline.run(_speech_audio(), sample_rate=SR)
    assert result.output_dir is None
    for seg in result.segments:
        assert seg.path is None


def test_run_processing_time_positive():
    pipeline = _silence_pipeline()
    result = pipeline.run(_speech_audio(), sample_rate=SR)
    assert result.processing_time_s > 0.0


def test_run_segmenter_name_set():
    pipeline = _silence_pipeline()
    result = pipeline.run(_speech_audio(), sample_rate=SR)
    assert result.segmenter_name == "SilenceSegmenter"


def test_run_duration_settings_propagated():
    pipeline = SilencePipeline(soft_lower=3.0, soft_upper=15.0)
    result = pipeline.run(_speech_audio(), sample_rate=SR)
    assert result.duration_settings.soft_lower == pytest.approx(3.0)
    assert result.duration_settings.soft_upper == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Arquivo: require sample_rate for array input
# ---------------------------------------------------------------------------


def test_array_without_sample_rate_raises():
    pipeline = _silence_pipeline()
    with pytest.raises(ValueError, match="sample_rate"):
        pipeline.run(_speech_audio())  # type: ignore[call-arg]


def test_array_with_sample_rate_works():
    pipeline = _silence_pipeline()
    result = pipeline.run(_speech_audio(), sample_rate=SR)
    assert len(result.segments) >= 1


def test_source_name_appears_in_metadata(tmp_path: Path):
    pipeline = _silence_pipeline()
    result = pipeline.run(
        _speech_audio(),
        sample_rate=SR,
        output=str(tmp_path),
        write_run_metadata=True,
        source_name="my_recording",
    )
    assert result.output_dir is not None
    meta = json.loads((result.output_dir / "run_metadata.json").read_text())
    assert meta["source"]["filename"] == "my_recording"


# ---------------------------------------------------------------------------
# Saída para disco
# ---------------------------------------------------------------------------


def test_run_with_output_creates_files(tmp_path: Path):
    pipeline = _silence_pipeline()
    result = pipeline.run(_speech_audio(), sample_rate=SR, output=str(tmp_path))
    assert result.output_dir is not None
    assert result.output_dir.exists()
    for seg in result.segments:
        assert seg.path is not None
        assert seg.path.exists()


def test_run_segments_have_path_when_output_given(tmp_path: Path):
    pipeline = _silence_pipeline()
    result = pipeline.run(_speech_audio(), sample_rate=SR, output=str(tmp_path))
    for seg in result.segments:
        assert seg.path is not None


def test_run_with_output_subfolder_named_after_source(tmp_path: Path):
    pipeline = _silence_pipeline()
    import soundfile as sf

    audio_path = tmp_path / "my_audio.flac"
    sf.write(str(audio_path), _speech_audio(), SR)

    out_dir = tmp_path / "out"
    result = pipeline.run(str(audio_path), output=str(out_dir), create_run_subfolder=True)
    assert result.output_dir is not None
    assert result.output_dir.name == "my_audio"


def test_run_with_io_settings_object(tmp_path: Path):
    pipeline = _silence_pipeline()
    io = IOSettings(output_folder=tmp_path, write_run_metadata=True)
    result = pipeline.run(_speech_audio(), sample_rate=SR, io_settings=io)
    assert result.output_dir is not None
    assert (result.output_dir / "run_metadata.json").exists()


def test_io_settings_overrides_output_kwarg(tmp_path: Path):
    pipeline = _silence_pipeline()
    io = IOSettings(output_folder=tmp_path / "from_io")
    # output kwarg deve ser ignorado quando io_settings é passado
    result = pipeline.run(
        _speech_audio(),
        sample_rate=SR,
        output=str(tmp_path / "from_output"),
        io_settings=io,
    )
    assert result.output_dir is not None
    assert "from_io" in str(result.output_dir)


def test_run_write_segment_spectrograms(tmp_path: Path):
    pipeline = _silence_pipeline()
    result = pipeline.run(
        _speech_audio(),
        sample_rate=SR,
        output=str(tmp_path),
        write_segment_spectrograms=True,
        create_segment_subfolders=True,
    )
    assert result.output_dir is not None
    pngs = list(result.output_dir.rglob("spectrogram.png"))
    assert len(pngs) == len(result.segments)


# ---------------------------------------------------------------------------
# WebRTCPipeline
# ---------------------------------------------------------------------------


def test_webrtc_pipeline_no_output():
    pipeline = WebRTCPipeline(
        aggressiveness=WebRTCAggressiveness.QUALITY,
        soft_lower=1.0,
        soft_upper=10.0,
        hard_lower=0.5,
        hard_upper=20.0,
        max_gap=0.5,
    )
    audio = _speech_audio()
    try:
        result = pipeline.run(audio, sample_rate=SR)
        assert result.output_dir is None
        assert result.segmenter_name == "WebRTCSegmenter"
    except EmptySegmentationError:
        pytest.skip("WebRTC VAD não detectou fala com áudio sintético")


def test_webrtc_pipeline_with_output(tmp_path: Path):
    pipeline = WebRTCPipeline(
        aggressiveness=WebRTCAggressiveness.QUALITY,
        soft_lower=1.0,
        soft_upper=10.0,
        hard_lower=0.5,
        hard_upper=20.0,
        max_gap=0.5,
    )
    audio = _speech_audio()
    try:
        result = pipeline.run(audio, sample_rate=SR, output=str(tmp_path))
        assert result.output_dir is not None
        for seg in result.segments:
            assert seg.path is not None
    except EmptySegmentationError:
        pytest.skip("WebRTC VAD não detectou fala com áudio sintético")


# ---------------------------------------------------------------------------
# File source (Path/str)
# ---------------------------------------------------------------------------


def test_run_from_file_path(tmp_path: Path):
    import soundfile as sf

    audio_path = tmp_path / "audio.flac"
    sf.write(str(audio_path), _speech_audio(), SR)

    pipeline = _silence_pipeline()
    result = pipeline.run(audio_path)
    assert isinstance(result, RunResult)
    assert len(result.segments) >= 1


def test_run_from_file_str(tmp_path: Path):
    import soundfile as sf

    audio_path = tmp_path / "audio.flac"
    sf.write(str(audio_path), _speech_audio(), SR)

    pipeline = _silence_pipeline()
    result = pipeline.run(str(audio_path))
    assert len(result.segments) >= 1


# ---------------------------------------------------------------------------
# Reutilização da mesma instância com chamadas distintas
# ---------------------------------------------------------------------------


def test_pipeline_reuse_different_configs(tmp_path: Path):
    pipeline = _silence_pipeline()
    audio = _speech_audio()

    r1 = pipeline.run(audio, sample_rate=SR)
    r2 = pipeline.run(audio, sample_rate=SR, output=str(tmp_path / "run1"))
    r3 = pipeline.run(
        audio,
        sample_rate=SR,
        output=str(tmp_path / "run2"),
        write_segment_spectrograms=True,
        create_segment_subfolders=True,
    )

    assert r1.output_dir is None
    assert r2.output_dir is not None
    assert r3.output_dir is not None
    assert r2.output_dir != r3.output_dir
