"""Testes para SilenceSegmenter."""

import numpy as np
import pytest

from voice_segmentation.core.silence import SilenceSegmenter, SilenceSettings
from voice_segmentation.exceptions import EmptySegmentationError
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray

SR = 16_000


def _make_audio(*blocks: tuple[float, float]) -> AudioArray:
    """Concatena blocos de (amplitude, duration_s) formando áudio sintético."""
    parts = []
    for amplitude, duration in blocks:
        samples = int(SR * duration)
        if amplitude > 0:
            t = np.linspace(0, duration * 440 * 2 * np.pi, samples)
            parts.append((np.sin(t) * amplitude).astype(np.float32))
        else:
            parts.append(np.zeros(samples, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _settings(**kwargs) -> DurationSettings:
    defaults = {
        "hard_lower": 0.5,
        "soft_lower": 1.0,
        "soft_upper": 10.0,
        "hard_upper": 20.0,
        "max_gap": 0.3,
    }
    defaults.update(kwargs)
    return DurationSettings(**defaults)


def test_segments_alternating_loud_quiet():
    # silêncio|fala|silêncio|fala|silêncio → deve gerar 2 segmentos
    audio = _make_audio((0.0, 1.0), (0.8, 3.0), (0.0, 1.0), (0.8, 3.0), (0.0, 1.0))
    segmenter = SilenceSegmenter()
    segs = segmenter.segment(audio, SR, _settings())
    assert len(segs) >= 1
    for start, end in segs:
        assert end > start
        assert start >= 0.0
        assert end <= len(audio) / SR


def test_segments_cover_speech_region():
    # 1s silence + 4s speech + 1s silence
    audio = _make_audio((0.0, 1.0), (0.8, 4.0), (0.0, 1.0))
    segmenter = SilenceSegmenter()
    segs = segmenter.segment(audio, SR, _settings())
    # Ao menos um segmento deve cobrir a região de fala
    assert any(start < 3.0 and end > 2.0 for start, end in segs)


def test_raises_empty_on_silent_audio():
    audio = np.zeros(SR * 5, dtype=np.float32)
    segmenter = SilenceSegmenter()
    # Áudio totalmente silencioso levará a enforçamento de duração → EmptySegmentationError
    with pytest.raises(EmptySegmentationError):
        segmenter.segment(audio, SR, _settings())


def test_custom_silence_settings():
    audio = _make_audio((0.0, 0.5), (0.8, 4.0), (0.0, 0.5))
    settings = SilenceSettings(silence_percentile=0.10)
    segmenter = SilenceSegmenter(settings)
    segs = segmenter.segment(audio, SR, _settings())
    assert len(segs) >= 1


def test_segment_results_sorted():
    audio = _make_audio((0.0, 0.5), (0.8, 2.0), (0.0, 0.5), (0.8, 2.0), (0.0, 0.5))
    segmenter = SilenceSegmenter()
    segs = segmenter.segment(audio, SR, _settings())
    for i in range(len(segs) - 1):
        assert segs[i][1] <= segs[i + 1][0]
