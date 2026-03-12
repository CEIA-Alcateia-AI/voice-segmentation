from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from voice_segmentation.settings import DurationSettings


@dataclass(frozen=True)
class SegmentResult:
    """Um único segmento de voz produzido pelo pipeline.

    Attributes:
        start: Início do segmento em segundos.
        end: Fim do segmento em segundos.
        path: Caminho do arquivo de áudio gravado, ou None quando run() foi chamado
            sem output.
    """

    start: float
    end: float
    path: Path | None = None

    @property
    def duration(self) -> float:
        """Duração do segmento em segundos."""
        return self.end - self.start


@dataclass(frozen=True)
class RunResult:
    """Resultado imutável produzido por um pipeline após segmentação.

    Attributes:
        segments: Lista de segmentos produzidos, com caminho opcional ao arquivo de áudio.
        processing_time_s: Tempo total de processamento em segundos.
        segmenter_name: Nome da classe do segmentador utilizada.
        segmenter_settings: Configurações do segmentador como instância Pydantic.
            None quando o segmentador não expõe configurações estruturadas.
        duration_settings: Configurações de duração aplicadas no pipeline.
        output_dir: Diretório onde os arquivos foram gravados, ou None quando run()
            foi chamado sem output.
    """

    segments: list[SegmentResult]
    processing_time_s: float
    segmenter_name: str
    segmenter_settings: BaseModel | None
    duration_settings: DurationSettings
    output_dir: Path | None = None
