import logging

import librosa
import numpy as np
from librosa.effects import split as librosa_split
from pydantic import BaseModel, Field

from voice_segmentation.exceptions import EmptySegmentationError, SilenceDetectionError
from voice_segmentation.post.duration import enforce_duration
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray, Segment

logger = logging.getLogger(__name__)


class SilenceSettings(BaseModel):
    """Configurações da estratégia de segmentação por silêncios.

    Attributes:
        silence_percentile: Fração (0 a 1) da distribuição de energia que define o limiar de
            silêncio. Valores mais altos tornam a detecção mais agressiva, tratando mais frames
            como silêncio.
        frame_length: Tamanho da janela de análise espectral em amostras. Deve ser potencia de 2
            para eficiencia da FFT.
        hop_length: Número de amostras entre janelas consecutivas. Determina a resolucao temporal
            da detecção.
        min_silence_duration: Duração mínima de silêncio entre dois intervalos nao-silênciosos para
            que sejam mantidos separados, em segundos. Gaps menores sao eliminados mesclando os
            intervalos adjacentes.
    """

    silence_percentile: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description=(
            "Percentil da distribuição de energia usado como limiar de silêncio (0 a 1). "
            "Valores maiores equivalem a detecção mais agressiva."
        ),
    )
    frame_length: int = Field(
        default=2048,
        gt=0,
        description="Tamanho da janela de análise em amostras",
    )
    hop_length: int = Field(
        default=512,
        gt=0,
        description="Passo entre janelas consecutivas em amostras",
    )
    min_silence_duration: float = Field(
        default=0.1,
        ge=0.0,
        description="Gap mínimo entre intervalos nao-silênciosos para mante-los separados (s)",
    )


class SilenceSegmenter:
    """Segmentador que usa períodos de silêncio como fronteiras entre segmentos."""

    def __init__(self, silence_settings: SilenceSettings | None = None) -> None:
        """Inicializa o segmentador com as configurações de silêncio.

        Args:
            silence_settings: Configurações específicas de detecção de silêncios. Se None, os
                valores padrão de SilenceSettings sao usados.
        """
        self.silence_settings = silence_settings or SilenceSettings()
        logger.debug("SilenceSegmenter inicializado com %s", self.silence_settings)

    def _compute_top_db(self, audio: AudioArray) -> float:
        """Calcula o limiar top_db relativo a distribuição de energia do sinal.

        Computa a energia RMS por frame em dB relativa ao pico do sinal e retorna a distância
        em dB entre o pico e o percentil silence_percentile da distribuição.

        Args:
            audio: Sinal de áudio mono em float32.

        Returns:
            Valor positivo de top_db a ser usado em librosa.effects.split.
        """
        rms: np.ndarray = librosa.feature.rms(
            y=audio,
            frame_length=self.silence_settings.frame_length,
            hop_length=self.silence_settings.hop_length,
        )[0]

        # Converte para dB relativo ao pico - valores em (-inf, 0]
        rms_db: np.ndarray = librosa.amplitude_to_db(rms, ref=float(np.max(rms)))

        threshold_db = float(np.percentile(rms_db, self.silence_settings.silence_percentile * 100))
        top_db = -threshold_db  # top_db é positivo: distância abaixo do pico

        logger.debug(
            (
                "Limiar relativo calculado - silence_percentile=%.2f -> threshold=%.1f dB -> "
                "top_db=%.1f dB"
            ),
            self.silence_settings.silence_percentile,
            threshold_db,
            top_db,
        )
        return top_db

    def segment(
        self,
        audio: AudioArray,
        sample_rate: int,
        settings: DurationSettings,
    ) -> list[Segment]:
        """Segmenta um sinal de áudio usando detecção de silêncios.

        Args:
            audio: Sinal de áudio mono em float32.
            sample_rate: Taxa de amostragem do sinal em Hz.
            settings: Configurações de duração e mesclagem.

        Returns:
            Lista de segmentos (início, fim) em segundos, ordenada cronologicamente, com durações
            dentro dos limites configurados.

        Raises:
            SilenceDetectionError: Se librosa.effects.split falhar.
            EmptySegmentationError: Se nenhum segmento válido for produzido após o
                pós-processamento.
        """
        top_db = self._compute_top_db(audio)

        try:
            raw_intervals: np.ndarray = librosa_split(
                y=audio,
                top_db=top_db,
                frame_length=self.silence_settings.frame_length,
                hop_length=self.silence_settings.hop_length,
            )
        except Exception as exc:
            raise SilenceDetectionError(str(exc)) from exc

        segments = self._merge_silence_gaps(raw_intervals, sample_rate)
        audio_duration = len(audio) / sample_rate
        result = enforce_duration(segments, settings, audio_duration)

        if not result:
            logger.warning(
                "Detecção de silêncios não produziu segmentos válidos. "
                "O áudio pode estar inteiramente silêncioso ou as configurações podem ser muito "
                "restritivas."
            )
            raise EmptySegmentationError(
                "Nenhum segmento válido após detecção de silêncios e pós-processamento. "
                "Tente ajustar silence_percentile, min_silence_duration ou os limites de duração."
            )

        return result

    def _merge_silence_gaps(
        self,
        intervals: np.ndarray,
        sample_rate: int,
    ) -> list[Segment]:
        """Converte intervalos em amostras para segundos, mesclando gaps curtos.

        Dois intervalos nao-silênciosos separados por um gap menor que
        silence_settings.min_silence_duration sao mesclados em um único segmento, evitando
        fragmentação excessiva por silêncios brevissimos.

        Args:
            intervals: Array de forma (N, 2) com [start, end] em amostras.
            sample_rate: Taxa de amostragem em Hz.

        Returns:
            Lista de segmentos (início, fim) em segundos.
        """
        if len(intervals) == 0:
            return []

        min_gap_samples = int(self.silence_settings.min_silence_duration * sample_rate)

        current_segment_start, current_segment_end = int(intervals[0][0]), int(intervals[0][1])
        result: list[Segment] = []

        for next_start_raw, next_end_raw in intervals[1:]:
            next_start, next_end = int(next_start_raw), int(next_end_raw)
            if next_start - current_segment_end < min_gap_samples:
                current_segment_end = next_end
            else:
                result.append(
                    (current_segment_start / sample_rate, current_segment_end / sample_rate)
                )
                current_segment_start, current_segment_end = next_start, next_end

        result.append((current_segment_start / sample_rate, current_segment_end / sample_rate))
        return result
