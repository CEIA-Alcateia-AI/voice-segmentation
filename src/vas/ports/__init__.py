"""Ports package - structural contracts for the segmentation pipeline."""

from vas.ports.core_processor import CoreProcessor
from vas.ports.post_processor import AudioPostProcessor
from vas.ports.pre_processor import AudioPreProcessor
from vas.ports.types import Segment, Timestamp

__all__ = [
    "AudioPostProcessor",
    "AudioPreProcessor",
    "CoreProcessor",
    "Segment",
    "Timestamp",
]
