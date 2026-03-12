"""Estratégia de segmentação baseada no WebRTC VAD."""

import logging
from enum import IntEnum
from typing import Literal

import librosa
import numpy as np
import webrtcvad
from pydantic import BaseModel, Field

from voice_segmentation.exceptions import EmptySegmentationError, SilenceDetectionError
from voice_segmentation.post.duration import enforce_duration
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray, Segment

logger = logging.getLogger(__name__)


class WebRTCAggressiveness(IntEnum):
    """Nível de agressividade do WebRTC VAD.

    Attributes:
        QUALITY: Mais permissivo - preserva mais fala, pode incluir mais ruido.
        LOW_BITRATE: Equilibrio leve para fala com ruido moderado.
        AGGRESSIVE: Filtragem robusta para ambientes barulhentos.
        VERY_AGGRESSIVE: Máxima filtragem - apenas fala muito clara e mantida.
    """

    QUALITY = 0
    LOW_BITRATE = 1
    AGGRESSIVE = 2
    VERY_AGGRESSIVE = 3


class WebRTCSettings(BaseModel):
    """Configurações da estratégia de segmentação por WebRTC VAD.

    O áudio é reamostrado internamente para vad_sample_rate antes da classificação; o sinal
    original não é modificado.

    Attributes:
        aggressiveness: Nível de agressividade do VAD (0 a 3). Valores maiores rejeitam ruido
            mais agressivamente.
        vad_sample_rate: Taxa de amostragem para o VAD em Hz. Deve ser 8000, 16000, 32000
            ou 48000.
        frame_duration_ms: Duração de cada frame classificado em milissegundos. Deve ser 10, 20
            ou 30.
        min_silence_ms: Duração mínima de silêncio continuo em ms para encerrar um segmento de
            fala. Pausas menores são ignoradas e a fala continua.
        speech_pad_ms: Padding adicionado antes e após cada região de fala em ms para evitar
            cortes abruptos.
    """

    aggressiveness: WebRTCAggressiveness = Field(
        default=WebRTCAggressiveness.AGGRESSIVE,
        description="Nível de agressividade do VAD (0=menor, 3=maior)",
    )
    vad_sample_rate: Literal[8000, 16000, 32000, 48000] = Field(
        default=16000,
        description="Taxa de amostragem interna do VAD em Hz",
    )
    frame_duration_ms: Literal[10, 20, 30] = Field(
        default=20,
        description="Duração de cada frame de análise em ms (10, 20 ou 30)",
    )
    min_silence_ms: int = Field(
        default=300,
        ge=0,
        description="Silêncio mínimo para encerrar um segmento de fala em ms",
    )
    speech_pad_ms: int = Field(
        default=50,
        ge=0,
        description="Padding adicionado nas bordas de cada região de fala em ms",
    )


class WebRTCSegmenter:
    """Segmentador baseado no WebRTC Voice Activity Detector."""

    def __init__(self, webrtc_settings: WebRTCSettings | None = None) -> None:
        """Inicializa o segmentador com as configurações do WebRTC VAD.

        Args:
            webrtc_settings: Configurações específicas do VAD. Se None, os valores padrão de
                WebRTCSettings sao usados.
        """
        self.webrtc_settings = webrtc_settings or WebRTCSettings()
        logger.debug("WebRTCSegmenter inicializado com %s", self.webrtc_settings)

    def _prepare_audio(self, audio: AudioArray, sample_rate: int) -> np.ndarray:
        """Reamostra e converte o áudio para PCM int16 na taxa do VAD.

        Args:
            audio: Sinal de áudio mono em float32.
            sample_rate: Taxa de amostragem original do sinal.

        Returns:
            Sinal em int16 reamostrado para vad_sample_rate.
        """
        vad_sr = self.webrtc_settings.vad_sample_rate

        if sample_rate != vad_sr:
            logger.debug("Reamostrando de %d Hz para %d Hz para o WebRTC VAD.", sample_rate, vad_sr)
            resampled: np.ndarray = librosa.resample(audio, orig_sr=sample_rate, target_sr=vad_sr)
        else:
            resampled = audio

        # Converte float32 [-1, 1] para int16
        pcm: np.ndarray = np.clip(resampled, -1.0, 1.0)
        return (pcm * 32767).astype(np.int16)

    def _classify_frames(self, pcm: np.ndarray) -> list[bool]:
        """Classifica cada frame como fala (True) ou silêncio (False).

        Args:
            pcm: Sinal em int16 na taxa de amostragem do VAD.

        Returns:
            Lista com um booleano por frame na ordem cronológica.
        """
        vad = webrtcvad.Vad(int(self.webrtc_settings.aggressiveness))
        sr = self.webrtc_settings.vad_sample_rate
        frame_samples = int(sr * self.webrtc_settings.frame_duration_ms / 1000)

        results: list[bool] = []
        for i in range(0, len(pcm) - frame_samples + 1, frame_samples):
            frame_bytes = pcm[i : i + frame_samples].tobytes()
            results.append(vad.is_speech(frame_bytes, sr))

        return results

    def _frames_to_segments(self, frame_labels: list[bool], audio_duration: float) -> list[Segment]:
        """Converte a sequência de labels de frame em segmentos de tempo.

        Args:
            frame_labels: Lista de booleanos (True = fala) por frame.
            audio_duration: Duração total do áudio em segundos, usado como limite superior
                do padding.

        Returns:
            Lista de segmentos (início, fim) em segundos.
        """
        frame_s = self.webrtc_settings.frame_duration_ms / 1000.0
        min_silence_frames = int(
            self.webrtc_settings.min_silence_ms / self.webrtc_settings.frame_duration_ms
        )
        pad_frames = int(
            self.webrtc_settings.speech_pad_ms / self.webrtc_settings.frame_duration_ms
        )

        # Suaviza: preenche silêncios curtos dentro de regiões de fala
        smoothed = list(frame_labels)
        i = 0
        while i < len(smoothed):
            if not smoothed[i]:
                # Conta o tamanho do bloco de silêncio
                j = i
                while j < len(smoothed) and not smoothed[j]:
                    j += 1
                silence_len = j - i
                # Se o bloco de silêncio é curto e há fala em ambos os lados, preenche
                if silence_len <= min_silence_frames and i > 0 and j < len(smoothed):
                    for k in range(i, j):
                        smoothed[k] = True
                i = j
            else:
                i += 1

        # Agrupa frames de fala em segmentos e aplica padding
        segments: list[Segment] = []
        in_speech = False
        seg_start = 0.0

        for idx, is_speech in enumerate(smoothed):
            if is_speech and not in_speech:
                in_speech = True
                seg_start = max(0.0, idx * frame_s - pad_frames * frame_s)
            elif not is_speech and in_speech:
                in_speech = False
                seg_end = min(audio_duration, (idx - 1) * frame_s + frame_s + pad_frames * frame_s)
                segments.append((seg_start, seg_end))

        # Fecha segmento aberto no final do áudio
        if in_speech:
            seg_end = min(
                audio_duration,
                len(smoothed) * frame_s + pad_frames * frame_s,
            )
            segments.append((seg_start, seg_end))

        return segments

    def segment(
        self,
        audio: AudioArray,
        sample_rate: int,
        settings: DurationSettings,
    ) -> list[Segment]:
        """Segmenta um sinal de áudio usando o WebRTC VAD.

        Args:
            audio: Sinal de áudio mono em float32.
            sample_rate: Taxa de amostragem do sinal em Hz.
            settings: Configurações de duração e mesclagem.

        Returns:
            Lista de segmentos (início, fim) em segundos, ordenada cronologicamente, com durações
            dentro dos limites configurados.

        Raises:
            SilenceDetectionError: Se a classificação do VAD falhar.
            EmptySegmentationError: Se nenhum segmento válido for produzido após o
                pós-processamento.
        """
        audio_duration = len(audio) / sample_rate

        try:
            pcm = self._prepare_audio(audio, sample_rate)
            frame_labels = self._classify_frames(pcm)
        except Exception as exc:
            raise SilenceDetectionError(str(exc)) from exc

        n_speech = sum(frame_labels)
        logger.debug(
            "WebRTC VAD classificou %d/%d frames como fala (%.1f%%).",
            n_speech,
            len(frame_labels),
            100 * n_speech / len(frame_labels) if frame_labels else 0,
        )

        raw_segments = self._frames_to_segments(frame_labels, audio_duration)
        result = enforce_duration(raw_segments, settings, audio_duration)

        if not result:
            logger.warning(
                "WebRTC VAD não produziu segmentos válidos. "
                "O áudio pode estar inteiramente silêncioso ou as configurações podem ser muito "
                "restritivas."
            )
            raise EmptySegmentationError(
                "Nenhum segmento válido após WebRTC VAD e pós-processamento. "
                "Tente ajustar aggressiveness, min_silence_ms ou os limites de duração."
            )

        return result
