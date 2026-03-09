"""Biblioteca de estratégias de segmentação de voz."""

from voice_segmentation._settings import DurationSettings
from voice_segmentation._types import AudioArray, Segment
from voice_segmentation.protocols import Segmenter

__all__ = [
    "AudioArray",
    "DurationSettings",
    "Segment",
    "Segmenter",
]
