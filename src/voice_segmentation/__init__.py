"""Biblioteca de estratégias de segmentação de voz."""

from voice_segmentation.exceptions import EmptySegmentationError, VoiceSegmentationError
from voice_segmentation.result import RunResult, SegmentResult

__all__ = [
    "EmptySegmentationError",
    "RunResult",
    "SegmentResult",
    "VoiceSegmentationError",
]
