import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from voice_segmentation.exceptions import EmptySegmentationError
from voice_segmentation.io.reader import load_audio
from voice_segmentation.io.settings import IOSettings
from voice_segmentation.io.writer import write_segments
from voice_segmentation.result import RunResult, SegmentResult
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray, Segment

logger = logging.getLogger(__name__)


@runtime_checkable
class Segmenter(Protocol):
    """Contrato que todo segmentador deve satisfazer.

    Implemente este protocolo para adicionar novas estratégias de segmentação
    compatíveis com Pipeline.
    """

    def segment(
        self,
        audio: AudioArray,
        sample_rate: int,
        settings: DurationSettings,
    ) -> list[Segment]:
        """Segmenta um sinal de áudio em intervalos de tempo.

        Args:
            audio: Sinal de áudio mono em float32.
            sample_rate: Taxa de amostragem do sinal em Hz.
            settings: Configurações de duração e mesclagem.

        Returns:
            Lista de segmentos (início, fim) em segundos, ordenada cronologicamente.
        """
        ...


class Pipeline(ABC):
    """Classe base para pipelines de segmentação de voz.

    Encapsula os limites de duração e fornece a implementação completa de `run()`.
    Subclasses devem atribuir `self._segmenter` e implementar `_segmenter_settings`.

    Attributes:
        duration_settings: Configurações de duração aplicadas em cada execução.
    """

    _segmenter: Segmenter  # atribuído no __init__ da subclasse

    def __init__(
        self,
        *,
        soft_lower: float = 1.0,
        soft_upper: float = 30.0,
        hard_lower: float = 0.5,
        hard_upper: float = 60.0,
        overlap: float = 0.0,
        max_gap: float = 0.0,
        duration_settings: DurationSettings | None = None,
    ) -> None:
        """Inicializa os parâmetros de duração.

        Args:
            soft_lower: Duração mínima preferida para segmentos em segundos.
            soft_upper: Duração máxima preferida para segmentos em segundos.
            hard_lower: Duração mínima absoluta para segmentos em segundos.
            hard_upper: Duração máxima absoluta para segmentos em segundos.
            overlap: Sobreposição em segundos adicionada ao final de segmentos longos.
            max_gap: Gap máximo em segundos para mesclar segmentos adjacentes.
            duration_settings: Configurações completas de duração. Prevalece sobre os parâmetros
                individuais quando fornecido.
        """
        self.duration_settings = duration_settings or DurationSettings(
            soft_lower=soft_lower,
            soft_upper=soft_upper,
            hard_lower=hard_lower,
            hard_upper=hard_upper,
            overlap=overlap,
            max_gap=max_gap,
        )

    @property
    @abstractmethod
    def _segmenter_settings(self) -> BaseModel | None:
        """Configurações do segmentador concreto, para inclusão nos metadados."""
        ...

    def run(
        self,
        source: Path | str | AudioArray,
        sample_rate: int | None = None,
        *,
        output: Path | str | None = None,
        io_settings: IOSettings | None = None,
        create_run_subfolder: bool = True,
        create_segment_subfolders: bool = False,
        write_run_metadata: bool = True,
        write_segment_metadata: bool = True,
        write_segment_spectrograms: bool = False,
        redo_if_exists: bool = True,
        audio_format: Literal["flac", "wav"] = "flac",
        run_name: str | None = None,
        segment_prefix: str | None = None,
        source_name: str | None = None,
    ) -> RunResult:
        """Executa a segmentação sobre um arquivo de áudio ou array.

        Args:
            source: Caminho do arquivo de áudio ou array numpy float32. Quando for array,
                `sample_rate` é obrigatório.
            sample_rate: Taxa de amostragem em Hz. Ignorado quando `source` é um caminho.
            output: Diretório de saída para gravar os segmentos. Ignorado quando
                `io_settings` é fornecido.
            io_settings: Configurações de escrita completas. Quando fornecido, prevalece sobre
                `output` e todos os parâmetros individuais de escrita.
            create_run_subfolder: Cria subdiretório por execução dentro de output.
            create_segment_subfolders: Cada segmento recebe seu próprio subdiretório.
            write_run_metadata: Grava run_metadata.json na raiz da execução.
            write_segment_metadata: Grava metadata.json individual por segmento.
            write_segment_spectrograms: Salva imagem PNG com três espectrogramas por segmento.
            redo_if_exists: Recria o diretório caso já exista.
            audio_format: Formato dos arquivos de áudio gravados.
            run_name: Nome fixo do subdiretório da execução.
            segment_prefix: Prefixo dos arquivos de segmento.
            source_name: Nome descritivo usado nos metadados quando `source` é um array numpy.
                Quando None, usa `"audio"`.

        Returns:
            RunResult com a lista de segmentos, métricas de execução e, quando output é
            fornecido, o caminho dos arquivos gravados e o diretório de saída.

        Raises:
            ValueError: Se `source` for um array e `sample_rate` não for fornecido.
            EmptySegmentationError: Se nenhum segmento válido for produzido.
        """
        pipeline_name = type(self).__name__

        if isinstance(source, str | Path):
            audio, sr = load_audio(source)
            source_path = Path(source)
        elif isinstance(source, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate é obrigatório quando source é um array numpy.")
            audio = source
            sr = sample_rate
            source_path = Path(source_name or "audio")
        else:
            raise TypeError(f"source deve ser Path, str ou ndarray, não {type(source).__name__!r}")

        duration = len(audio) / sr
        logger.info(
            "%s iniciando - duração: %.2fs | %d amostras @ %d Hz",
            pipeline_name,
            duration,
            len(audio),
            sr,
        )

        t_start = time.perf_counter()
        segments = self._segmenter.segment(audio, sr, self.duration_settings)
        processing_time_s = time.perf_counter() - t_start

        if not segments:
            raise EmptySegmentationError(
                f"{pipeline_name} não produziu segmentos. Verifique as configurações."
            )

        logger.info(
            "%s concluído - %d segmento(s) em %.3fs",
            pipeline_name,
            len(segments),
            processing_time_s,
        )

        resolved_io = io_settings or (
            IOSettings(
                output_folder=Path(output),
                create_run_subfolder=create_run_subfolder,
                create_segment_subfolders=create_segment_subfolders,
                write_run_metadata=write_run_metadata,
                write_segment_metadata=write_segment_metadata,
                write_segment_spectrograms=write_segment_spectrograms,
                redo_if_exists=redo_if_exists,
                audio_format=audio_format,
                run_name=run_name,
                segment_prefix=segment_prefix,
            )
            if output is not None
            else None
        )

        if resolved_io is None:
            segment_results = [SegmentResult(start=s, end=e) for s, e in segments]
            return RunResult(
                segments=segment_results,
                processing_time_s=processing_time_s,
                segmenter_name=type(self._segmenter).__name__,
                segmenter_settings=self._segmenter_settings,
                duration_settings=self.duration_settings,
                output_dir=None,
            )

        run_dir, segment_results = write_segments(
            audio=audio,
            sample_rate=sr,
            segments=segments,
            source_path=source_path,
            io_settings=resolved_io,
            segmenter_name=type(self._segmenter).__name__,
            segmenter_settings=self._segmenter_settings,
            duration_settings=self.duration_settings,
            processing_time_s=processing_time_s,
        )
        return RunResult(
            segments=segment_results,
            processing_time_s=processing_time_s,
            segmenter_name=type(self._segmenter).__name__,
            segmenter_settings=self._segmenter_settings,
            duration_settings=self.duration_settings,
            output_dir=run_dir,
        )
