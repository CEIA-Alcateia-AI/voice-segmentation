"""Contratos estruturais (Protocols) da biblioteca."""

from typing import Protocol, Self, runtime_checkable

from voice_segmentation._settings import DurationSettings
from voice_segmentation._types import AudioArray, Segment


@runtime_checkable
class Segmenter(Protocol):
    """Contrato para qualquer estratégia de segmentação de voz."""

    def segment(
        self: Self,
        audio: AudioArray,
        sample_rate: int,
        settings: DurationSettings,
    ) -> list[Segment]:
        """Segmenta um sinal de áudio em intervalos de tempo.

        Args:
            audio: Sinal de áudio mono em float32.
            sample_rate: Taxa de amostragem do sinal, em Hz.
            settings: Configurações de duração e mesclagem.

        Returns:
            Lista de segmentos ``(início, fim)`` em segundos,
            ordenada cronologicamente, sem sobreposição.
        """
        ...
