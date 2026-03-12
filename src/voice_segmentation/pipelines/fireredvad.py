from pathlib import Path

from pydantic import BaseModel

from voice_segmentation.core.fireredvad import FireRedSegmenter, FireRedSettings
from voice_segmentation.pipelines.base import Pipeline
from voice_segmentation.settings import DurationSettings


class FireRedPipeline(Pipeline):
    """Pipeline de segmentação baseado no FireRed Voice Activity Detector.

    Os pesos são baixados automaticamente se ausentes. Aceita parâmetros individuais ou um
    objeto FireRedSettings completo, que prevalece sobre os parâmetros individuais.

    Attributes:
        fireredvad_settings: Configurações do VAD.
        duration_settings: Limites de duração e mesclagem. Herdado de Pipeline.
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        use_gpu: bool = False,
        smooth_window_size: int = 5,
        speech_threshold: float = 0.4,
        min_speech_frame: int = 20,
        max_speech_frame: int = 2000,
        min_silence_frame: int = 20,
        merge_silence_frame: int = 0,
        extend_speech_frame: int = 0,
        chunk_max_frame: int = 30000,
        soft_lower: float = 1.0,
        soft_upper: float = 30.0,
        hard_lower: float = 0.5,
        hard_upper: float = 60.0,
        overlap: float = 0.0,
        max_gap: float = 0.0,
        fireredvad_settings: FireRedSettings | None = None,
        duration_settings: DurationSettings | None = None,
    ) -> None:
        """Inicializa o pipeline FireRed.

        Args:
            model_dir: Caminho para o diretório com os pesos do modelo FireRed VAD.
                Quando omitido, usa `~/.cache/voice_segmentation/fireredvad/VAD/` e
                baixa os pesos automaticamente do HuggingFace Hub se ainda não presentes.
            use_gpu: Usar GPU (CUDA) na inferência.
            smooth_window_size: Tamanho da janela de suavização de probabilidades.
            speech_threshold: Probabilidade mínima para classificar um frame como fala.
            min_speech_frame: Duração mínima de fala em frames (1 frame = 10 ms).
            max_speech_frame: Duração máxima de fala em frames antes de forçar corte.
            min_silence_frame: Duração mínima de silêncio para separar segmentos em frames.
            merge_silence_frame: Silêncios menores que este valor são mesclados ao segmento.
            extend_speech_frame: Frames adicionados ao final de cada segmento de fala.
            chunk_max_frame: Tamanho máximo do chunk processado de uma vez em frames.
            soft_lower: Duração mínima preferida para segmentos em segundos.
            soft_upper: Duração máxima preferida para segmentos em segundos.
            hard_lower: Duração mínima absoluta para segmentos em segundos.
            hard_upper: Duração máxima absoluta para segmentos em segundos.
            overlap: Sobreposição em segundos adicionada ao final de segmentos longos.
            max_gap: Gap máximo em segundos para mesclar segmentos adjacentes.
            fireredvad_settings: Configurações completas do VAD. Quando fornecido, prevalece
                sobre todos os parâmetros individuais, inclusive `model_dir`.
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
        self.fireredvad_settings = fireredvad_settings or FireRedSettings(
            **({"model_dir": Path(model_dir)} if model_dir is not None else {}),
            use_gpu=use_gpu,
            smooth_window_size=smooth_window_size,
            speech_threshold=speech_threshold,
            min_speech_frame=min_speech_frame,
            max_speech_frame=max_speech_frame,
            min_silence_frame=min_silence_frame,
            merge_silence_frame=merge_silence_frame,
            extend_speech_frame=extend_speech_frame,
            chunk_max_frame=chunk_max_frame,
        )
        self._segmenter = FireRedSegmenter(self.fireredvad_settings)

    @property
    def _segmenter_settings(self) -> BaseModel | None:
        return self.fireredvad_settings
