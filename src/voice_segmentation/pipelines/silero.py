from pydantic import BaseModel

from voice_segmentation.core.silero import SileroSegmenter, SileroSettings
from voice_segmentation.pipelines.base import Pipeline
from voice_segmentation.settings import DurationSettings


class SileroPipeline(Pipeline):
    """Pipeline de segmentação baseado no Silero Voice Activity Detector.

    O modelo é carregado automaticamente. Aceita parâmetros individuais ou um objeto
    SileroSettings completo, que prevalece sobre os parâmetros individuais.

    Attributes:
        silero_settings: Configurações do VAD neural.
        duration_settings: Limites de duração e mesclagem. Herdado de Pipeline.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        window_size_samples: int = 512,
        soft_lower: float = 1.0,
        soft_upper: float = 30.0,
        hard_lower: float = 0.5,
        hard_upper: float = 60.0,
        overlap: float = 0.0,
        max_gap: float = 0.0,
        apply_post_processing: bool = True,
        silero_settings: SileroSettings | None = None,
        duration_settings: DurationSettings | None = None,
    ) -> None:
        """Inicializa o pipeline Silero.

        Args:
            threshold: Probabilidade mínima de fala para classificar um frame como voz.
            min_speech_duration_ms: Duração mínima de fala em ms para manter um segmento.
            min_silence_duration_ms: Duração mínima de silêncio em ms para separar segmentos.
            speech_pad_ms: Padding em ms adicionado nas bordas de cada região de fala.
            window_size_samples: Tamanho da janela de análise em amostras.
            soft_lower: Duração mínima preferida para segmentos em segundos.
            soft_upper: Duração máxima preferida para segmentos em segundos.
            hard_lower: Duração mínima absoluta para segmentos em segundos.
            hard_upper: Duração máxima absoluta para segmentos em segundos.
            overlap: Sobreposição em segundos adicionada ao final de segmentos longos.
            max_gap: Gap máximo em segundos para mesclar segmentos adjacentes.
            apply_post_processing: Quando False, retorna os segmentos brutos do detector sem
                aplicar mesclagem, divisão, overlap ou filtragem por limites de duração.
            silero_settings: Configurações completas do VAD. Prevalece sobre os parâmetros
                individuais do VAD quando fornecido.
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
            apply_post_processing=apply_post_processing,
            duration_settings=duration_settings,
        )
        self.silero_settings = silero_settings or SileroSettings(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            window_size_samples=window_size_samples,
        )
        self._segmenter = SileroSegmenter(self.silero_settings)

    @property
    def _segmenter_settings(self) -> BaseModel | None:
        return self.silero_settings
