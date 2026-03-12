"""Testes para SileroSegmenter."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from voice_segmentation.core.silero import SileroSegmenter, SileroSettings
from voice_segmentation.exceptions import EmptySegmentationError
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray

SR = 16_000


def _sine(duration: float, sr: int = SR) -> AudioArray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


def _silence(duration: float, sr: int = SR) -> AudioArray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _settings(**kwargs) -> DurationSettings:
    defaults = {
        "hard_lower": 0.5,
        "soft_lower": 1.0,
        "soft_upper": 10.0,
        "hard_upper": 30.0,
    }
    defaults.update(kwargs)
    return DurationSettings(**defaults)


# ---------------------------------------------------------------------------
# SileroSettings
# ---------------------------------------------------------------------------


def test_defaults():
    s = SileroSettings()
    assert s.threshold == 0.5
    assert s.min_speech_duration_ms == 250
    assert s.min_silence_duration_ms == 100
    assert s.speech_pad_ms == 30
    assert s.window_size_samples == 512


def test_threshold_bounds():
    with pytest.raises(ValidationError):
        SileroSettings(threshold=1.5)
    with pytest.raises(ValidationError):
        SileroSettings(threshold=-0.1)


# ---------------------------------------------------------------------------
# _prepare_audio
# ---------------------------------------------------------------------------


def test_prepare_audio_returns_tensor_at_16k():
    seg = SileroSegmenter()
    audio = _sine(1.0)
    tensor, sr_out = seg._prepare_audio(audio, SR)
    assert isinstance(tensor, torch.Tensor)
    assert sr_out == SR
    assert len(tensor) == len(audio)


def test_prepare_audio_resamples_to_16k():
    seg = SileroSegmenter()
    orig_sr = 44100
    t = np.linspace(0, 1.0, orig_sr, endpoint=False)
    audio_44k = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    tensor, sr_out = seg._prepare_audio(audio_44k, orig_sr)
    assert sr_out == 16000
    assert abs(len(tensor) - 16000) <= 100


def test_prepare_audio_keeps_8k():
    seg = SileroSegmenter()
    orig_sr = 8000
    audio_8k = _sine(1.0, sr=orig_sr)
    tensor, sr_out = seg._prepare_audio(audio_8k, orig_sr)
    assert sr_out == 8000
    assert len(tensor) == len(audio_8k)


# ---------------------------------------------------------------------------
# segment() — com mock de get_speech_timestamps
# ---------------------------------------------------------------------------


def test_segment_returns_sorted_segments():
    seg = SileroSegmenter()
    audio = _sine(10.0)
    fake_ts = [{"start": 0.5, "end": 3.0}, {"start": 4.0, "end": 7.5}]
    with patch("voice_segmentation.core.silero.get_speech_timestamps", return_value=fake_ts):
        result = seg.segment(audio, SR, _settings())
    starts = [s for s, _ in result]
    assert starts == sorted(starts)


def test_segment_maps_timestamps_to_tuples():
    seg = SileroSegmenter()
    audio = _sine(10.0)
    fake_ts = [{"start": 1.0, "end": 4.0}]
    with patch("voice_segmentation.core.silero.get_speech_timestamps", return_value=fake_ts):
        result = seg.segment(audio, SR, _settings())
    assert len(result) == 1
    s, e = result[0]
    assert pytest.approx(s, abs=0.1) == 1.0
    assert pytest.approx(e, abs=0.1) == 4.0


def test_segment_raises_empty_when_no_speech():
    seg = SileroSegmenter()
    audio = _silence(3.0)
    with (
        patch("voice_segmentation.core.silero.get_speech_timestamps", return_value=[]),
        pytest.raises(EmptySegmentationError),
    ):
        seg.segment(audio, SR, _settings())


def test_segment_passes_settings_to_vad():
    silero_settings = SileroSettings(threshold=0.7, speech_pad_ms=50)
    seg = SileroSegmenter(silero_settings)
    audio = _sine(5.0)
    fake_ts = [{"start": 0.5, "end": 4.0}]
    captured: dict = {}

    def fake_get_timestamps(tensor, model, **kwargs):
        captured.update(kwargs)
        return fake_ts

    with patch(
        "voice_segmentation.core.silero.get_speech_timestamps", side_effect=fake_get_timestamps
    ):
        seg.segment(audio, SR, _settings())

    assert captured["threshold"] == 0.7
    assert captured["speech_pad_ms"] == 50


def test_segment_resamples_before_inference():
    seg = SileroSegmenter()
    orig_sr = 44100
    t = np.linspace(0, 5.0, int(orig_sr * 5), endpoint=False)
    audio_44k = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    fake_ts = [{"start": 0.5, "end": 4.0}]

    call_sr: list[int] = []

    def fake_get_timestamps(tensor, model, sampling_rate, **kwargs):
        call_sr.append(sampling_rate)
        return fake_ts

    with patch(
        "voice_segmentation.core.silero.get_speech_timestamps", side_effect=fake_get_timestamps
    ):
        seg.segment(audio_44k, orig_sr, _settings())

    assert call_sr[0] == 16000


def test_segment_on_silent_audio_raises_empty():
    """Teste end-to-end com o modelo real: silêncio puro não deve produzir segmentos."""
    seg = SileroSegmenter(SileroSettings(threshold=0.5))
    audio = _silence(3.0)
    with pytest.raises(EmptySegmentationError):
        seg.segment(audio, SR, _settings())
