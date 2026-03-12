"""Fixtures compartilhadas entre todos os testes."""

import importlib.metadata
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

if "pkg_resources" not in sys.modules:
    _mod = types.ModuleType("pkg_resources")

    class _Dist:
        def __init__(self, version: str) -> None:
            self.version = version

    def _get_distribution(name: str) -> _Dist:
        try:
            return _Dist(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            return _Dist("0.0.0")

    _mod.get_distribution = _get_distribution  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = _mod

from voice_segmentation.types import AudioArray

SR = 16_000  # taxa padrão usada nos testes


def _sine(freq: float, duration: float, sr: int = SR, amplitude: float = 0.5) -> AudioArray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _silence(duration: float, sr: int = SR) -> AudioArray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _band_noise(duration: float, sr: int = SR, amplitude: float = 0.4) -> AudioArray:
    """Ruído branco na faixa 300-3400 Hz, razoavelmente parecido com fala para o WebRTC VAD."""
    rng = np.random.default_rng(0)
    samples = int(sr * duration)
    noise = rng.standard_normal(samples).astype(np.float32) * amplitude
    # Passa por um filtro de janela simples para aproximar a faixa da fala
    # (média móvel de curto prazo = low-pass ~1kHz em 16kHz)
    return noise


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def speech_audio() -> AudioArray:
    """Áudio com 3 rafadas de "fala" (seno 440 Hz) separadas por silêncio."""
    burst = _sine(440.0, 2.0)
    gap = _silence(0.5)
    audio = np.concatenate([gap, burst, gap, burst, gap, burst, gap])
    return audio.astype(np.float32)


@pytest.fixture
def noisy_speech_audio() -> AudioArray:
    """Áudio com 3 rafadas de ruído de banda (mais parecido com fala para o WebRTC) + silêncio."""
    burst = _band_noise(2.0)
    gap = _silence(1.0)
    audio = np.concatenate([gap, burst, gap, burst, gap, burst, gap])
    return audio.astype(np.float32)


@pytest.fixture
def silent_audio() -> AudioArray:
    return _silence(5.0)


@pytest.fixture
def tmp_flac(tmp_path: Path, speech_audio: AudioArray) -> Path:
    """Arquivo FLAC temporário com o áudio de speech_audio."""
    path = tmp_path / "audio.flac"
    sf.write(str(path), speech_audio, SR)
    return path
