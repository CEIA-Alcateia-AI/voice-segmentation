import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from voice_segmentation.io.settings import IOSettings
from voice_segmentation.io.spectrogram import _save_segment_spectrogram
from voice_segmentation.result import SegmentResult
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray, Segment

logger = logging.getLogger(__name__)


def _rms_db(audio: AudioArray) -> float:
    value = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return float(20.0 * np.log10(value)) if value > 0.0 else -float("inf")


def _peak_db(audio: AudioArray) -> float:
    value = float(np.max(np.abs(audio.astype(np.float64))))
    return float(20.0 * np.log10(value)) if value > 0.0 else -float("inf")


def _duration_stats(durations: list[float]) -> dict[str, float]:
    arr = np.array(durations, dtype=np.float64)
    return {
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "median_s": float(np.median(arr)),
    }


def _build_segment_record(
    index: int,
    start_s: float,
    end_s: float,
    audio_slice: AudioArray,
    sample_rate: int,
    relative_audio_path: str,
    run_context: dict[str, Any],
    relative_spectrogram_path: str | None = None,
) -> dict[str, Any]:
    seg_rms = _rms_db(audio_slice)
    seg_peak = _peak_db(audio_slice)
    record: dict[str, Any] = {
        "index": index,
        "start_s": round(start_s, 6),
        "end_s": round(end_s, 6),
        "duration_s": round(end_s - start_s, 6),
        "sample_rate": sample_rate,
        "samples": len(audio_slice),
        "rms_db": round(seg_rms, 3) if seg_rms != -float("inf") else None,
        "peak_db": round(seg_peak, 3) if seg_peak != -float("inf") else None,
        "file": relative_audio_path,
    }
    if relative_spectrogram_path is not None:
        record["spectrogram"] = relative_spectrogram_path
    record.update(run_context)
    return record


def _build_run_metadata(
    source_path: Path,
    audio_duration: float,
    sample_rate: int,
    segment_count: int,
    segmenter_name: str,
    segmenter_settings: BaseModel | None,
    duration_settings: DurationSettings,
    processing_time_s: float,
    processed_at: str,
    segment_records: list[dict[str, Any]],
) -> dict[str, Any]:
    durations = [r["duration_s"] for r in segment_records]
    total_speech = sum(durations)
    realtime_factor = audio_duration / processing_time_s if processing_time_s > 0 else None
    return {
        "version": "1",
        "source": {
            "path": str(source_path.resolve()),
            "filename": source_path.name,
            "duration_s": round(audio_duration, 6),
            "sample_rate": sample_rate,
        },
        "pipeline": {
            "segmenter": segmenter_name,
            "segmenter_settings": (
                segmenter_settings.model_dump() if segmenter_settings is not None else None
            ),
            "duration_settings": duration_settings.model_dump(),
        },
        "processed_at": processed_at,
        "result": {
            "segment_count": segment_count,
            "total_speech_s": round(total_speech, 6),
            "speech_coverage": (
                round(total_speech / audio_duration, 6) if audio_duration > 0 else 0.0
            ),
            "gap_s": round(audio_duration - total_speech, 6),
            "duration_stats": _duration_stats(durations) if durations else {},
        },
        "performance": {
            "processing_time_s": round(processing_time_s, 6),
            "realtime_factor": round(realtime_factor, 3) if realtime_factor is not None else None,
        },
        "segments": segment_records,
    }


def write_segments(
    audio: AudioArray,
    sample_rate: int,
    segments: list[Segment],
    source_path: Path | str,
    io_settings: IOSettings,
    segmenter_name: str,
    segmenter_settings: BaseModel | None,
    duration_settings: DurationSettings,
    processing_time_s: float,
) -> tuple[Path, list[SegmentResult]]:
    """Grava segmentos de áudio no diretório de saída e retorna os caminhos produzidos.

    Extrai cada fatia do sinal, grava no formato configurado e opcionalmente produz
    metadados e espectrogramas. Retorna sem gravar nada se o diretório já existir
    e redo_if_exists for False.

    Args:
        audio: Sinal de áudio completo mono float32.
        sample_rate: Taxa de amostragem do sinal em Hz.
        segments: Lista de segmentos (início, fim) em segundos.
        source_path: Caminho do arquivo de origem, usado nos metadados.
        io_settings: Configurações de caminho, formato e comportamento de escrita.
        segmenter_name: Nome da classe do segmentador para os metadados.
        segmenter_settings: Configurações do segmentador ou None.
        duration_settings: Configurações de duração para os metadados.
        processing_time_s: Tempo de processamento em segundos para os metadados.

    Returns:
        Tupla (diretório da execução, lista de SegmentResult com path preenchido).
    """
    source_path = Path(source_path)

    if io_settings.create_run_subfolder:
        run_name = io_settings.run_name or source_path.stem
        run_dir = io_settings.output_folder / run_name
    else:
        run_dir = io_settings.output_folder

    if run_dir.exists():
        if not io_settings.redo_if_exists:
            logger.warning("Diretório já existe e redo_if_exists=False - pulando: %s", run_dir)
            return run_dir, []
        logger.info("Recriando diretório existente: %s", run_dir)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    audio_duration = len(audio) / sample_rate
    ext = io_settings.audio_format
    subtype = "PCM_16" if ext == "wav" else None
    prefix = io_settings.segment_prefix or source_path.stem
    processed_at = datetime.now(UTC).isoformat()
    run_context: dict[str, Any] = {
        "source": {
            "path": str(source_path.resolve()),
            "filename": source_path.name,
            "duration_s": round(audio_duration, 6),
            "sample_rate": sample_rate,
        },
        "pipeline": {
            "segmenter": segmenter_name,
            "segmenter_settings": (
                segmenter_settings.model_dump() if segmenter_settings is not None else None
            ),
            "duration_settings": duration_settings.model_dump(),
        },
        "processed_at": processed_at,
    }
    segment_records: list[dict[str, Any]] = []
    segment_results: list[SegmentResult] = []

    for index, (segment_start, segment_end) in enumerate(segments):
        start_sample = int(segment_start * sample_rate)
        end_sample = min(int(segment_end * sample_rate), len(audio))
        audio_slice = audio[start_sample:end_sample]

        segment_name = f"{prefix}_segment_{index:03d}"

        if io_settings.create_segment_subfolders:
            segment_dir = run_dir / segment_name
            segment_dir.mkdir()
            audio_dest = segment_dir / f"{segment_name}.{ext}"
            metadata_dest = segment_dir / "metadata.json"
            relative_audio = f"{segment_name}/{segment_name}.{ext}"
        else:
            audio_dest = run_dir / f"{segment_name}.{ext}"
            metadata_dest = run_dir / f"{segment_name}_metadata.json"
            relative_audio = f"{segment_name}.{ext}"

        sf.write(str(audio_dest), audio_slice, sample_rate, subtype=subtype)

        relative_spectrogram: str | None = None
        if io_settings.write_segment_spectrograms:
            if io_settings.create_segment_subfolders:
                spec_dest = Path(segment_dir) / "spectrogram.png"
                relative_spectrogram = f"{segment_name}/spectrogram.png"
            else:
                spec_dest = Path(run_dir) / f"{segment_name}_spectrogram.png"
                relative_spectrogram = f"{segment_name}_spectrogram.png"
            _save_segment_spectrogram(audio_slice, sample_rate, spec_dest)

        record = _build_segment_record(
            index=index,
            start_s=segment_start,
            end_s=segment_end,
            audio_slice=audio_slice,
            sample_rate=sample_rate,
            relative_audio_path=relative_audio,
            run_context=run_context,
            relative_spectrogram_path=relative_spectrogram,
        )

        if io_settings.write_segment_metadata:
            metadata_dest.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        segment_records.append(record)
        segment_results.append(SegmentResult(start=segment_start, end=segment_end, path=audio_dest))

    if io_settings.write_run_metadata:
        run_meta = _build_run_metadata(
            source_path=source_path,
            audio_duration=audio_duration,
            sample_rate=sample_rate,
            segment_count=len(segments),
            segmenter_name=segmenter_name,
            segmenter_settings=segmenter_settings,
            duration_settings=duration_settings,
            processing_time_s=processing_time_s,
            processed_at=processed_at,
            segment_records=segment_records,
        )
        metadata_path = run_dir / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Metadados da execução gravados em: %s", metadata_path)

    logger.info(
        "Escrita concluída - %d segmento(s) gravado(s) em: %s",
        len(segments),
        run_dir,
    )
    return run_dir, segment_results
