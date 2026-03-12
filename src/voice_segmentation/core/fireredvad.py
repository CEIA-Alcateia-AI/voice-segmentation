from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
import soundfile as sf

try:
    from fireredvad import FireRedVad, FireRedVadConfig

    _FIREREDVAD_AVAILABLE = True
except ImportError:
    _FIREREDVAD_AVAILABLE = False

from pydantic import BaseModel, Field

from voice_segmentation.exceptions import EmptySegmentationError
from voice_segmentation.post.duration import enforce_duration
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray, Segment

logger = logging.getLogger(__name__)

_FIRERED_SAMPLE_RATE: int = 16_000
_HF_REPO_ID: str = "FireRedTeam/FireRedVAD"
_VALID_VARIANTS = frozenset({"VAD", "Stream-VAD", "AED"})
_DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "voice_segmentation" / "fireredvad"


def download_fireredvad_weights(
    local_dir: Path | str | None = None,
    variant: Literal["VAD", "Stream-VAD", "AED"] = "VAD",
) -> Path:
    """Baixa os pesos do FireRed VAD do HuggingFace Hub para o diretório de cache local.

    Args:
        local_dir: Diretório base onde os pesos serão salvos. O subdiretório <variant>
            é criado automaticamente dentro dele. Quando omitido, usa
            ~/.cache/voice_segmentation/fireredvad.
        variant: Variante do modelo a baixar. "VAD" (padrão) para segmentação
            não-streaming; "Stream-VAD" para inferência em tempo real; "AED"
            para detecção de eventos de áudio.

    Returns:
        Caminho para o diretório do modelo, pronto para passar como model_dir a
        FireRedSegmenter ou FireRedPipeline.

    Raises:
        ImportError: Se huggingface_hub não estiver instalado.
        ValueError: Se variant não for válida.
    """
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant deve ser um de {sorted(_VALID_VARIANTS)!r}, não {variant!r}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub é necessário para baixar os pesos do FireRed VAD. "
            "Instale com: pip install 'voice-segmentation[fireredvad]'"
        ) from exc

    dest = Path(local_dir) if local_dir is not None else _DEFAULT_CACHE_DIR
    logger.info("Baixando FireRed VAD (%s) para %s …", variant, dest)
    snapshot_download(
        repo_id=_HF_REPO_ID,
        allow_patterns=[f"{variant}/*"],
        local_dir=str(dest),
    )
    model_dir = dest / variant
    logger.info("Download concluído: %s", model_dir)
    return model_dir


class FireRedSettings(BaseModel):
    """Configurações da estratégia de segmentação por FireRed VAD.

    O FireRed VAD exige uma pasta com os pesos do modelo, obtida via HuggingFace ou
    ModelScope. O áudio é convertido internamente para WAV 16 kHz 16-bit mono PCM antes
    da inferência.

    Attributes:
        model_dir: Caminho para o diretório contendo os pesos do modelo FireRed VAD.
        use_gpu: Usa GPU (CUDA) na inferência se disponível.
        smooth_window_size: Tamanho da janela de suavização de probabilidades.
        speech_threshold: Probabilidade mínima para classificar um frame como fala.
        min_speech_frame: Duração mínima de um segmento de fala em frames (10 ms/frame).
        max_speech_frame: Duração máxima de um segmento de fala em frames.
        min_silence_frame: Duração mínima de silêncio para separar dois segmentos em frames.
        merge_silence_frame: Silêncios menores que este valor são mesclados ao segmento.
        extend_speech_frame: Número de frames adicionados ao final de cada segmento.
        chunk_max_frame: Tamanho máximo do chunk processado por vez em frames.
    """

    model_dir: Path = Field(
        default_factory=lambda: _DEFAULT_CACHE_DIR / "VAD",
        description="Diretório com os pesos (cache em ~/.cache/voice_segmentation/fireredvad/VAD)",
    )
    use_gpu: bool = Field(
        default=False,
        description="Usar GPU na inferência",
    )
    smooth_window_size: int = Field(
        default=5,
        ge=1,
        description="Tamanho da janela de suavização de probabilidades",
    )
    speech_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Probabilidade mínima para classificar um frame como fala",
    )
    min_speech_frame: int = Field(
        default=20,
        ge=1,
        description="Duração mínima de fala em frames (1 frame = 10 ms)",
    )
    max_speech_frame: int = Field(
        default=2000,
        ge=1,
        description="Duração máxima de fala em frames antes de forçar um corte",
    )
    min_silence_frame: int = Field(
        default=20,
        ge=0,
        description="Duração mínima de silêncio para separar segmentos em frames",
    )
    merge_silence_frame: int = Field(
        default=0,
        ge=0,
        description="Silêncios menores que este valor (frames) são mesclados ao segmento",
    )
    extend_speech_frame: int = Field(
        default=0,
        ge=0,
        description="Frames adicionados ao final de cada segmento de fala",
    )
    chunk_max_frame: int = Field(
        default=30000,
        ge=1,
        description="Tamanho máximo do chunk processado de uma vez em frames",
    )


class FireRedSegmenter:
    """Segmentador baseado no FireRed Voice Activity Detector.

    O modelo é carregado uma única vez na inicialização. A inferência exige um arquivo WAV
    16 kHz 16-bit mono PCM temporário; o áudio é convertido automaticamente antes de cada
    chamada e o arquivo é removido ao final.
    """

    _REQUIRED_FILES: frozenset[str] = frozenset({"model.pth.tar", "cmvn.ark"})

    def __init__(self, fireredvad_settings: FireRedSettings) -> None:
        """Inicializa o segmentador e carrega os pesos do FireRed VAD.

        Se o diretório de pesos não existir ou estiver incompleto, os pesos são baixados
        automaticamente do HuggingFace Hub (FireRedTeam/FireRedVAD) sem necessidade de
        intervenção manual.

        Args:
            fireredvad_settings: Configurações do segmentador, incluindo o caminho para
                o diretório de pesos do modelo (baixado automaticamente se ausente).
        """
        if not _FIREREDVAD_AVAILABLE:
            raise ImportError(
                "fireredvad é necessário para FireRedSegmenter. "
                "Instale com: pip install 'voice-segmentation[fireredvad]'"
            )
        self.fireredvad_settings = fireredvad_settings
        self._ensure_weights(fireredvad_settings.model_dir)

        vad_config: Any = FireRedVadConfig(
            use_gpu=fireredvad_settings.use_gpu,
            smooth_window_size=fireredvad_settings.smooth_window_size,
            speech_threshold=fireredvad_settings.speech_threshold,
            min_speech_frame=fireredvad_settings.min_speech_frame,
            max_speech_frame=fireredvad_settings.max_speech_frame,
            min_silence_frame=fireredvad_settings.min_silence_frame,
            merge_silence_frame=fireredvad_settings.merge_silence_frame,
            extend_speech_frame=fireredvad_settings.extend_speech_frame,
            chunk_max_frame=fireredvad_settings.chunk_max_frame,
        )
        self._vad: Any = FireRedVad.from_pretrained(str(fireredvad_settings.model_dir), vad_config)
        logger.debug("FireRedSegmenter inicializado com %s", fireredvad_settings)

    @classmethod
    def _ensure_weights(cls, model_dir: Path) -> None:
        """Baixa os pesos automaticamente se o diretório estiver ausente ou incompleto."""
        missing = not model_dir.exists() or not all(
            (model_dir / f).exists() for f in cls._REQUIRED_FILES
        )
        if missing:
            logger.info(
                "Pesos do FireRed VAD não encontrados em '%s'. Baixando do HuggingFace Hub…",
                model_dir,
            )

            # resolve variant from the last path component (VAD / Stream-VAD / AED)
            variant = model_dir.name if model_dir.name in _VALID_VARIANTS else "VAD"
            local_dir = model_dir.parent if model_dir.name in _VALID_VARIANTS else model_dir
            download_fireredvad_weights(local_dir, variant=variant)  # type: ignore[arg-type]

    def segment(
        self,
        audio: AudioArray,
        sample_rate: int,
        settings: DurationSettings,
    ) -> list[Segment]:
        """Segmenta um sinal de áudio usando o FireRed VAD.

        O áudio é convertido internamente para WAV 16 kHz 16-bit mono PCM em um arquivo
        temporário antes da inferência. O arquivo é removido ao final mesmo em caso de erro.

        Args:
            audio: Sinal de áudio mono em float32.
            sample_rate: Taxa de amostragem do sinal em Hz.
            settings: Configurações de duração e mesclagem.

        Returns:
            Lista de segmentos (início, fim) em segundos, ordenada cronologicamente, com
            durações dentro dos limites configurados.

        Raises:
            EmptySegmentationError: Se nenhum segmento válido for produzido após o
                pós-processamento.
        """
        if sample_rate != _FIRERED_SAMPLE_RATE:
            logger.debug(
                "Reamostando de %d Hz para %d Hz para o FireRed VAD.",
                sample_rate,
                _FIRERED_SAMPLE_RATE,
            )
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_FIRERED_SAMPLE_RATE)

        audio_duration = len(audio) / _FIRERED_SAMPLE_RATE

        tmp_path_str: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path_str = tmp.name

            pcm = np.clip(audio, -1.0, 1.0)
            sf.write(tmp_path_str, pcm, _FIRERED_SAMPLE_RATE, subtype="PCM_16")

            result: dict[str, Any]
            result, _ = self._vad.detect(tmp_path_str)
        finally:
            if tmp_path_str is not None:
                Path(tmp_path_str).unlink(missing_ok=True)

        raw_timestamps: list[tuple[float, float]] = result.get("timestamps", [])
        segments: list[Segment] = list(raw_timestamps)

        if not segments:
            raise EmptySegmentationError("FireRed VAD não detectou fala no áudio.")

        return enforce_duration(segments, settings, audio_duration)
