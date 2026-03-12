from __future__ import annotations

import logging
from typing import Any

import librosa
from pydantic import BaseModel, Field

try:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    _SILERO_AVAILABLE = True
except ImportError:
    _SILERO_AVAILABLE = False

from voice_segmentation.exceptions import EmptySegmentationError
from voice_segmentation.post.duration import enforce_duration
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray, Segment

logger = logging.getLogger(__name__)

_SUPPORTED_SAMPLE_RATES: frozenset[int] = frozenset({8000, 16000})


class SileroSettings(BaseModel):
    """Configurações da estratégia de segmentação por Silero VAD.

    Attributes:
        threshold: Probabilidade mínima de fala para considerar um frame como voz.
            Valores maiores rejeitam mais frames como não-fala.
        min_speech_duration_ms: Duração mínima de uma região de fala em ms. Regiões menores
            são descartadas pelo modelo.
        min_silence_duration_ms: Duração mínima de silêncio em ms para separar dois segmentos
            de fala adjacentes.
        speech_pad_ms: Padding em ms adicionado antes e após cada região de fala detectada.
        window_size_samples: Tamanho da janela de análise do modelo em amostras.
            Use 512 para 16 kHz e 256 para 8 kHz.
    """

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probabilidade mínima de fala para classificar um frame como voz (0–1)",
    )
    min_speech_duration_ms: int = Field(
        default=250,
        ge=0,
        description="Duração mínima de fala em ms para manter um segmento",
    )
    min_silence_duration_ms: int = Field(
        default=100,
        ge=0,
        description="Duração mínima de silêncio em ms para separar segmentos de fala",
    )
    speech_pad_ms: int = Field(
        default=30,
        ge=0,
        description="Padding em ms adicionado nas bordas de cada região de fala",
    )
    window_size_samples: int = Field(
        default=512,
        gt=0,
        description="Tamanho da janela de análise em amostras (512 para 16 kHz, 256 para 8 kHz)",
    )


class SileroSegmenter:
    """Segmentador baseado no Silero Voice Activity Detector.

    O modelo é carregado uma única vez na inicialização e reutilizado em chamadas
    subsequentes a segment. O áudio é reamostrado internamente para 16 kHz se
    a taxa de amostragem original não for suportada (8 kHz ou 16 kHz).
    """

    def __init__(self, silero_settings: SileroSettings | None = None) -> None:
        """Inicializa o segmentador com as configurações do Silero VAD.

        Args:
            silero_settings: Configurações específicas do VAD. Se None, os valores padrão de
                SileroSettings são usados.
        """
        if not _SILERO_AVAILABLE:
            raise ImportError(
                "silero-vad e torch são necessários para SileroSegmenter. "
                "Instale com: pip install 'voice-segmentation[silero]'"
            )
        self.silero_settings = silero_settings or SileroSettings()
        self._model: Any = load_silero_vad()
        logger.debug("SileroSegmenter inicializado com %s", self.silero_settings)

    def _prepare_audio(self, audio: AudioArray, sample_rate: int) -> tuple[torch.Tensor, int]:
        """Reamostra o áudio para uma taxa suportada e converte para tensor PyTorch.

        O Silero VAD suporta apenas 8 kHz e 16 kHz. Qualquer outra taxa é reamostrada para
        16 kHz antes da inferência; o sinal original não é modificado.

        Args:
            audio: Sinal de áudio mono em float32.
            sample_rate: Taxa de amostragem original do sinal em Hz.

        Returns:
            Tupla (tensor float32, taxa_de_amostragem_efetiva).
        """
        if sample_rate not in _SUPPORTED_SAMPLE_RATES:
            target_sr = 16000
            logger.debug(
                "Reamostando de %d Hz para %d Hz para o Silero VAD.", sample_rate, target_sr
            )
            resampled: AudioArray = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=target_sr
            )
            return torch.from_numpy(resampled), target_sr

        return torch.from_numpy(audio.copy()), sample_rate

    def segment(
        self,
        audio: AudioArray,
        sample_rate: int,
        settings: DurationSettings,
    ) -> list[Segment]:
        """Segmenta um sinal de áudio usando o Silero VAD.

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
        tensor, sr = self._prepare_audio(audio, sample_rate)
        audio_duration = tensor.shape[0] / sr

        raw: list[dict[str, float]] = get_speech_timestamps(
            tensor,
            self._model,
            threshold=self.silero_settings.threshold,
            sampling_rate=sr,
            min_speech_duration_ms=self.silero_settings.min_speech_duration_ms,
            min_silence_duration_ms=self.silero_settings.min_silence_duration_ms,
            speech_pad_ms=self.silero_settings.speech_pad_ms,
            window_size_samples=self.silero_settings.window_size_samples,
            return_seconds=True,
        )

        segments: list[Segment] = [(d["start"], d["end"]) for d in raw]

        if not segments:
            raise EmptySegmentationError("Silero VAD não detectou fala no áudio.")

        return enforce_duration(segments, settings, audio_duration)
