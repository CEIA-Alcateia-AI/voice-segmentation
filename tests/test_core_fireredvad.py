"""Testes para FireRedSegmenter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from voice_segmentation.core.fireredvad import FireRedSegmenter, FireRedSettings
from voice_segmentation.exceptions import EmptySegmentationError
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray

SR = 16_000
_FAKE_MODEL_DIR = Path("/fake/fireredvad/model")


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


def _make_segmenter(detect_return=None, model_dir=_FAKE_MODEL_DIR):
    """Cria um FireRedSegmenter com o modelo completamente mockado."""
    if detect_return is None:
        detect_return = ({"timestamps": [(1.0, 4.0)], "dur": 5.0}, None)

    mock_vad = MagicMock()
    mock_vad.detect.return_value = detect_return

    settings = FireRedSettings(model_dir=model_dir)
    with (
        patch("voice_segmentation.core.fireredvad.FireRedVad") as mock_cls,
        patch.object(FireRedSegmenter, "_ensure_weights"),
    ):
        mock_cls.from_pretrained.return_value = mock_vad
        seg = FireRedSegmenter(settings)
        seg._vad = mock_vad  # garante que o mock é usado em segment()
    return seg


# ---------------------------------------------------------------------------
# FireRedSettings
# ---------------------------------------------------------------------------


def test_settings_defaults():
    s = FireRedSettings(model_dir=_FAKE_MODEL_DIR)
    assert s.use_gpu is False
    assert s.smooth_window_size == 5
    assert s.speech_threshold == 0.4
    assert s.min_speech_frame == 20
    assert s.max_speech_frame == 2000
    assert s.min_silence_frame == 20
    assert s.merge_silence_frame == 0
    assert s.extend_speech_frame == 0
    assert s.chunk_max_frame == 30000


def test_settings_model_dir_stored_as_path():
    s = FireRedSettings(model_dir=Path("/some/path"))
    assert isinstance(s.model_dir, Path)


def test_settings_threshold_bounds():
    with pytest.raises(ValidationError):
        FireRedSettings(model_dir=_FAKE_MODEL_DIR, speech_threshold=1.5)
    with pytest.raises(ValidationError):
        FireRedSettings(model_dir=_FAKE_MODEL_DIR, speech_threshold=-0.1)


# ---------------------------------------------------------------------------
# segment() — com modelo mockado
# ---------------------------------------------------------------------------


def test_segment_returns_timestamps_as_segments():
    seg = _make_segmenter(
        detect_return=({"timestamps": [(0.5, 3.0), (4.0, 7.0)], "dur": 8.0}, None)
    )
    result = seg.segment(_sine(8.0), SR, _settings())
    assert len(result) == 2
    assert result[0][0] == pytest.approx(0.5, abs=0.05)
    assert result[0][1] == pytest.approx(3.0, abs=0.05)


def test_segment_raises_empty_when_no_timestamps():
    seg = _make_segmenter(detect_return=({"timestamps": [], "dur": 3.0}, None))
    with pytest.raises(EmptySegmentationError):
        seg.segment(_silence(3.0), SR, _settings())


def test_segment_writes_temp_wav_at_16k(tmp_path):
    """Verifica que o arquivo temporário gravado tem a taxa correta."""
    written_files: list[str] = []
    written_srs: list[int] = []

    original_write = sf.write

    def capture_write(path, data, samplerate, **kwargs):
        written_files.append(path)
        written_srs.append(samplerate)
        original_write(path, data, samplerate, **kwargs)

    seg = _make_segmenter()
    audio = _sine(2.0)

    with patch("voice_segmentation.core.fireredvad.sf.write", side_effect=capture_write):
        seg.segment(audio, SR, _settings())

    assert len(written_srs) == 1
    assert written_srs[0] == 16000


def test_segment_resamples_non_16k_audio():
    """Áudio a 44100 Hz é reamostrado antes de gravar o WAV temporário."""
    written_srs: list[int] = []

    original_write = sf.write

    def capture_write(path, data, samplerate, **kwargs):
        written_srs.append(samplerate)
        original_write(path, data, samplerate, **kwargs)

    seg = _make_segmenter()
    orig_sr = 44100
    t = np.linspace(0, 2.0, int(orig_sr * 2), endpoint=False)
    audio_44k = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    with patch("voice_segmentation.core.fireredvad.sf.write", side_effect=capture_write):
        seg.segment(audio_44k, orig_sr, _settings())

    assert written_srs[0] == 16000


def test_segment_deletes_temp_file_on_success(tmp_path):
    """O arquivo temporário é removido após a inferência (sem erros)."""
    created: list[str] = []

    import tempfile

    original_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(**kwargs):
        f = original_ntf(**kwargs)
        created.append(f.name)
        return f

    seg = _make_segmenter()

    with patch(
        "voice_segmentation.core.fireredvad.tempfile.NamedTemporaryFile", side_effect=tracking_ntf
    ):
        seg.segment(_sine(2.0), SR, _settings())

    for path in created:
        assert not Path(path).exists(), f"Arquivo temporário não foi removido: {path}"


def test_segment_deletes_temp_file_on_error():
    """O arquivo temporário é removido mesmo quando a inferência lança uma exceção."""
    created: list[str] = []

    import tempfile

    original_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(**kwargs):
        f = original_ntf(**kwargs)
        created.append(f.name)
        return f

    mock_vad = MagicMock()
    mock_vad.detect.side_effect = RuntimeError("falha de inferência")
    settings = FireRedSettings(model_dir=_FAKE_MODEL_DIR)

    with (
        patch("voice_segmentation.core.fireredvad.FireRedVad") as mock_cls,
        patch.object(FireRedSegmenter, "_ensure_weights"),
    ):
        mock_cls.from_pretrained.return_value = mock_vad
        seg = FireRedSegmenter(settings)
        seg._vad = mock_vad

    with patch(
        "voice_segmentation.core.fireredvad.tempfile.NamedTemporaryFile", side_effect=tracking_ntf
    ), pytest.raises(RuntimeError):
        seg.segment(_sine(2.0), SR, _settings())

    for path in created:
        assert not Path(path).exists(), f"Arquivo temporário não foi removido após erro: {path}"


# ---------------------------------------------------------------------------
# Auto-download de pesos
# ---------------------------------------------------------------------------


def test_ensure_weights_downloads_when_dir_missing(tmp_path):
    """_ensure_weights dispara o download quando o diretório não existe."""
    missing_dir = tmp_path / "VAD"
    with patch(
        "voice_segmentation.core.fireredvad.download_fireredvad_weights"
    ) as mock_download:
        mock_download.return_value = missing_dir
        FireRedSegmenter._ensure_weights(missing_dir)
        mock_download.assert_called_once_with(tmp_path, variant="VAD")


def test_ensure_weights_skips_download_when_weights_present(tmp_path):
    """_ensure_weights não baixa nada quando os pesos já estão presentes."""
    model_dir = tmp_path / "VAD"
    model_dir.mkdir()
    (model_dir / "model.pth.tar").write_bytes(b"fake")
    (model_dir / "cmvn.ark").write_bytes(b"fake")

    with patch(
        "voice_segmentation.core.fireredvad.download_fireredvad_weights"
    ) as mock_download:
        FireRedSegmenter._ensure_weights(model_dir)
        mock_download.assert_not_called()


def test_segmenter_auto_downloads_on_init(tmp_path):
    """FireRedSegmenter baixa os pesos automaticamente ao ser instanciado sem pesos."""
    model_dir = tmp_path / "VAD"
    settings = FireRedSettings(model_dir=model_dir)

    mock_vad = MagicMock()
    with (
        patch("voice_segmentation.core.fireredvad.FireRedVad") as mock_cls,
        patch(
            "voice_segmentation.core.fireredvad.download_fireredvad_weights"
        ) as mock_download,
    ):
        mock_cls.from_pretrained.return_value = mock_vad
        mock_download.return_value = model_dir
        FireRedSegmenter(settings)
        mock_download.assert_called_once()
