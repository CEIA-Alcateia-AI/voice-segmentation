import logging
from pathlib import Path

import librosa
import numpy as np

from voice_segmentation.types import AudioArray

logger = logging.getLogger(__name__)


def load_audio(
    path: Path | str,
    target_sr: int | None = None,
    mono: bool = True,
) -> tuple[AudioArray, int]:
    """Carrega um arquivo de áudio e retorna o sinal normalizado em float32.

    Args:
        path: Caminho do arquivo de áudio. Suporta qualquer formato reconhecido pelo librosa
            (WAV, FLAC, MP3, OGG, entre outros).
        target_sr: Taxa de amostragem de destino em Hz. Se None, a taxa original é preservada.
        mono: Se True, converte sinais multicanal para mono via média dos canais.

    Returns:
        Tupla (audio, sample_rate) onde audio é float32 de forma (n_samples,).

    Raises:
        FileNotFoundError: Se o caminho não existir no sistema de arquivos.
        ValueError: Se o arquivo não puder ser interpretado como áudio.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {path}")

    logger.debug("Carregando áudio: %s | target_sr=%s | mono=%s", path.name, target_sr, mono)

    raw, sr = librosa.load(path, sr=target_sr, mono=mono)
    audio: AudioArray = raw.astype(np.float32)

    logger.debug("Carregado - %d amostras @ %d Hz (%.2fs)", len(audio), sr, len(audio) / sr)
    return audio, int(sr)
