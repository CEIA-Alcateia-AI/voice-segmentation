from typing import Literal

from pydantic import BaseModel

from voice_segmentation.core.webrtc import WebRTCAggressiveness, WebRTCSegmenter, WebRTCSettings
from voice_segmentation.pipelines.base import Pipeline
from voice_segmentation.settings import DurationSettings


class WebRTCPipeline(Pipeline):
    """Pipeline de segmentação baseado no WebRTC Voice Activity Detector.

    Aceita parâmetros individuais ou um objeto WebRTCSettings completo, que prevalece
    sobre os parâmetros individuais.

    Attributes:
        webrtc_settings: Configurações do VAD.
        duration_settings: Limites de duração e mesclagem. Herdado de Pipeline.
    """

    def __init__(
        self,
        *,
        aggressiveness: WebRTCAggressiveness = WebRTCAggressiveness.AGGRESSIVE,
        vad_sample_rate: Literal[8000, 16000, 32000, 48000] = 16000,
        frame_duration_ms: Literal[10, 20, 30] = 20,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 50,
        soft_lower: float = 1.0,
        soft_upper: float = 30.0,
        hard_lower: float = 0.5,
        hard_upper: float = 60.0,
        overlap: float = 0.0,
        max_gap: float = 0.0,
        apply_post_processing: bool = True,
        webrtc_settings: WebRTCSettings | None = None,
        duration_settings: DurationSettings | None = None,
    ) -> None:
        """Inicializa o pipeline WebRTC.

        Args:
            aggressiveness: Nível de agressividade do VAD.
            vad_sample_rate: Taxa de amostragem para o VAD em Hz.
            frame_duration_ms: Duração de cada frame em milissegundos.
            min_silence_ms: Silêncio mínimo em ms para encerrar um segmento de fala.
            speech_pad_ms: Padding em ms adicionado antes e após cada região de fala.
            soft_lower: Duração mínima preferida para segmentos em segundos.
            soft_upper: Duração máxima preferida para segmentos em segundos.
            hard_lower: Duração mínima absoluta para segmentos em segundos.
            hard_upper: Duração máxima absoluta para segmentos em segundos.
            overlap: Sobreposição em segundos adicionada ao final de segmentos longos.
            max_gap: Gap máximo em segundos para mesclar segmentos adjacentes.
            apply_post_processing: Quando False, retorna os segmentos brutos do detector sem
                aplicar mesclagem, divisão, overlap ou filtragem por limites de duração.
            webrtc_settings: Configurações completas do VAD. Prevalece sobre os parâmetros
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
        self.webrtc_settings = webrtc_settings or WebRTCSettings(
            aggressiveness=aggressiveness,
            vad_sample_rate=vad_sample_rate,
            frame_duration_ms=frame_duration_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._segmenter = WebRTCSegmenter(self.webrtc_settings)

    @property
    def _segmenter_settings(self) -> BaseModel | None:
        return self.webrtc_settings
