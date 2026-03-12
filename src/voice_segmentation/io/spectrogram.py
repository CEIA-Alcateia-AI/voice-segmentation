"""Geração e persistência de espectrogramas de segmentos de áudio."""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from voice_segmentation.types import AudioArray

_N_FFT = 2048
_HOP_LENGTH = 512
_N_MELS = 128


def _save_segment_spectrogram(audio: AudioArray, sample_rate: int, dest: Path) -> None:
    """Computa e salva uma imagem PNG com três métodos de espectrograma do segmento.

    Os três painéis, compartilhando o eixo de tempo, são:

    - **STFT** — frequência linear, amplitude em dB.
    - **Mel** — escala mel perceptual, potência em dB.
    - **CQT** — Transformada Q-Constante, amplitude em dB; resolução logarítmica
      de frequência, ideal para fala e música.

    Args:
        audio: Fatia de áudio mono float32.
        sample_rate: Taxa de amostragem do sinal em Hz.
        dest: Caminho de destino da imagem .png.
    """
    y = audio.astype(np.float32)

    D_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP_LENGTH)), ref=np.max
    )
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=y, sr=sample_rate, n_fft=_N_FFT, hop_length=_HOP_LENGTH, n_mels=_N_MELS
        ),
        ref=np.max,
    )
    C_db = librosa.amplitude_to_db(
        np.abs(librosa.cqt(y, sr=sample_rate, hop_length=_HOP_LENGTH)), ref=np.max
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    img0 = librosa.display.specshow(
        D_db, sr=sample_rate, hop_length=_HOP_LENGTH, y_axis="linear", x_axis="time", ax=axes[0]
    )
    fig.colorbar(img0, ax=axes[0], format="%+.0f dB")
    axes[0].set_title("STFT")

    img1 = librosa.display.specshow(
        mel_db, sr=sample_rate, hop_length=_HOP_LENGTH, y_axis="mel", x_axis="time", ax=axes[1]
    )
    fig.colorbar(img1, ax=axes[1], format="%+.0f dB")
    axes[1].set_title("Mel")

    img2 = librosa.display.specshow(
        C_db, sr=sample_rate, hop_length=_HOP_LENGTH, y_axis="cqt_hz", x_axis="time", ax=axes[2]
    )
    fig.colorbar(img2, ax=axes[2], format="%+.0f dB")
    axes[2].set_title("CQT")

    fig.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
