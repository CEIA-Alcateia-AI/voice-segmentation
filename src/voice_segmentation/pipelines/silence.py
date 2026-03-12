"""Pipeline pre-configurado baseado em detecção de silêncio."""

from pydantic import BaseModel

from voice_segmentation.core.silence import SilenceSegmenter, SilenceSettings
from voice_segmentation.pipelines.base import Pipeline
from voice_segmentation.settings import DurationSettings


class SilencePipeline(Pipeline):
    """Pipeline de segmentação baseado em detecção de regiões de silêncio.

    Aceita parâmetros individuais diretamente no construtor. Um objeto ``SilenceSettings``
    completo pode ser passado para uso avançado e prevalece sobre os parâmetros individuais.

    Opções de saída (espectrogramas, metadados, formato de áudio, etc.) são configuradas
    por chamada em ``run()``, permitindo reutilizar a mesma instância com saídas distintas.

    Attributes:
        silence_settings: Configurações do detector de silêncio.
        duration_settings: Limites de duração e mesclagem. Herdado de :class:`Pipeline`.
    """

    def __init__(
        self,
        *,
        silence_percentile: float = 0.20,
        frame_length: int = 2048,
        hop_length: int = 512,
        min_silence_duration: float = 0.1,
        soft_lower: float = 1.0,
        soft_upper: float = 30.0,
        hard_lower: float = 0.5,
        hard_upper: float = 60.0,
        overlap: float = 0.0,
        max_gap: float = 0.0,
        silence_settings: SilenceSettings | None = None,
        duration_settings: DurationSettings | None = None,
    ) -> None:
        """Inicializa o pipeline de silêncio.

        Args:
            silence_percentile: Percentil (0–1) da distribuição de RMS usado como limiar de
                silêncio. Valores menores preservam mais fala.
            frame_length: Tamanho do frame para análise de RMS em amostras.
            hop_length: Deslocamento entre frames em amostras.
            min_silence_duration: Duração mínima de silêncio em segundos para separar segmentos.
            soft_lower: Duração mínima preferida para segmentos em segundos.
            soft_upper: Duração máxima preferida para segmentos em segundos.
            hard_lower: Duração mínima absoluta para segmentos em segundos.
            hard_upper: Duração máxima absoluta para segmentos em segundos.
            overlap: Sobreposição em segundos adicionada ao final de segmentos longos.
            max_gap: Gap máximo em segundos para mesclar segmentos adjacentes.
            silence_settings: Configurações completas do detector. Prevalece sobre os parâmetros
                de silêncio quando fornecido.
            duration_settings: Configurações completas de duração. Prevalece sobre os parâmetros
                de duração quando fornecido.
        """
        super().__init__(
            soft_lower=soft_lower,
            soft_upper=soft_upper,
            hard_lower=hard_lower,
            hard_upper=hard_upper,
            overlap=overlap,
            max_gap=max_gap,
            duration_settings=duration_settings,
        )
        self.silence_settings = silence_settings or SilenceSettings(
            silence_percentile=silence_percentile,
            frame_length=frame_length,
            hop_length=hop_length,
            min_silence_duration=min_silence_duration,
        )
        self._segmenter = SilenceSegmenter(self.silence_settings)

    @property
    def _segmenter_settings(self) -> BaseModel | None:
        return self.silence_settings
