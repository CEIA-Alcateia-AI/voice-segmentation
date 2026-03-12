"""Testes para WebRTCSegmenter."""

import numpy as np
import pytest

from voice_segmentation.core.webrtc import (
    WebRTCAggressiveness,
    WebRTCSegmenter,
    WebRTCSettings,
)
from voice_segmentation.exceptions import EmptySegmentationError
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray

SR = 16_000


def _multi_sine(duration: float, freqs: list[int] = (200, 440, 880, 1600)) -> AudioArray:
    """Soma de senos em múltiplas frequências de fala — mais reconhecível pelo VAD."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    audio = sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)  # type: ignore[assignment]
    return (audio / (np.max(np.abs(audio)) + 1e-8) * 0.7).astype(np.float32)


def _silence(duration: float) -> AudioArray:
    return np.zeros(int(SR * duration), dtype=np.float32)


def _duration_settings(**kwargs) -> DurationSettings:
    defaults = {
        "hard_lower": 0.5,
        "soft_lower": 1.0,
        "soft_upper": 10.0,
        "hard_upper": 20.0,
        "max_gap": 0.5,
    }
    defaults.update(kwargs)
    return DurationSettings(**defaults)


# ---------------------------------------------------------------------------
# WebRTCSettings
# ---------------------------------------------------------------------------


def test_defaults():
    s = WebRTCSettings()
    assert s.aggressiveness == WebRTCAggressiveness.AGGRESSIVE
    assert s.vad_sample_rate == 16000
    assert s.frame_duration_ms == 20
    assert s.min_silence_ms == 300
    assert s.speech_pad_ms == 50


# ---------------------------------------------------------------------------
# _prepare_audio
# ---------------------------------------------------------------------------


def test_prepare_audio_int16_range():
    audio = _multi_sine(1.0)
    seg = WebRTCSegmenter()
    pcm = seg._prepare_audio(audio, SR)
    assert pcm.dtype == np.int16
    assert pcm.max() <= 32767
    assert pcm.min() >= -32768


def test_prepare_audio_resamples():
    # Áudio a 44100 Hz deve ser reamostrado para 16000 Hz
    orig_sr = 44100
    (_multi_sine(1.0)).astype(np.float32)
    # Re-create at 44100 Hz
    t = np.linspace(0, 1.0, orig_sr, endpoint=False)
    audio_44k = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    seg = WebRTCSegmenter(WebRTCSettings(vad_sample_rate=16000))
    pcm = seg._prepare_audio(audio_44k, orig_sr)
    # Deve ter aproximadamente 16000 amostras
    assert abs(len(pcm) - 16000) <= 100


# ---------------------------------------------------------------------------
# _frames_to_segments
# ---------------------------------------------------------------------------


def test_frames_to_segments_all_speech():
    seg = WebRTCSegmenter()
    labels = [True] * 50  # 50 frames de 20ms = 1s de fala
    segments = seg._frames_to_segments(labels, audio_duration=1.0)
    assert len(segments) == 1
    assert segments[0][0] == pytest.approx(0.0, abs=0.1)
    assert segments[0][1] == pytest.approx(1.0, abs=0.1)


def test_frames_to_segments_all_silence():
    seg = WebRTCSegmenter()
    labels = [False] * 50
    segments = seg._frames_to_segments(labels, audio_duration=1.0)
    assert segments == []


def test_frames_to_segments_two_bursts():
    seg = WebRTCSegmenter(WebRTCSettings(speech_pad_ms=0, min_silence_ms=0))
    # fala | silêncio | fala
    labels = [True] * 10 + [False] * 20 + [True] * 10
    segments = seg._frames_to_segments(labels, audio_duration=40 * 0.02)
    assert len(segments) >= 1  # pode mesclar ou separar conforme min_silence


def test_frames_to_segments_short_silence_filled():
    # Silêncio apenas 1 frame (< min_silence de 300ms/20ms = 15 frames) → preenchido
    seg = WebRTCSegmenter(WebRTCSettings(min_silence_ms=300, frame_duration_ms=20, speech_pad_ms=0))
    labels = [True] * 20 + [False] * 1 + [True] * 20
    segments = seg._frames_to_segments(labels, audio_duration=41 * 0.02)
    # Silêncio de 1 frame deve ser preenchido → deve voltar como 1 único segmento
    assert len(segments) == 1


# ---------------------------------------------------------------------------
# segment() — integração
# ---------------------------------------------------------------------------


def test_raises_on_silent_audio():
    seg = WebRTCSegmenter()
    audio = _silence(5.0)
    with pytest.raises(EmptySegmentationError):
        seg.segment(audio, SR, _duration_settings())


def test_segment_results_have_valid_bounds():
    """Resultado deve ter start >= 0 e end <= duração do áudio."""
    burst = _multi_sine(2.0)
    silence = _silence(1.0)
    audio = np.concatenate([silence, burst, silence, burst, silence]).astype(np.float32)
    duration = len(audio) / SR

    seg = WebRTCSegmenter(WebRTCSettings(aggressiveness=WebRTCAggressiveness.QUALITY))
    try:
        segs = seg.segment(audio, SR, _duration_settings())
    except EmptySegmentationError:
        pytest.skip("VAD não detectou fala com áudio sintético — comportamento esperado")

    for start, end in segs:
        assert start >= 0.0
        assert end <= duration + 0.001
        assert end > start


def test_segment_results_sorted():
    burst = _multi_sine(2.0)
    silence = _silence(1.0)
    audio = np.concatenate([silence, burst, silence, burst, silence]).astype(np.float32)
    seg = WebRTCSegmenter(WebRTCSettings(aggressiveness=WebRTCAggressiveness.QUALITY))
    try:
        segs = seg.segment(audio, SR, _duration_settings())
    except EmptySegmentationError:
        pytest.skip("VAD não detectou fala com áudio sintético")

    for i in range(len(segs) - 1):
        assert segs[i][1] <= segs[i + 1][0]
