"""Biblioteca de estratégias de segmentação de voz."""

from voice_segmentation.core.silence import SilenceSettings
from voice_segmentation.core.webrtc import WebRTCAggressiveness, WebRTCSettings
from voice_segmentation.exceptions import EmptySegmentationError, VoiceSegmentationError
from voice_segmentation.io.settings import IOSettings
from voice_segmentation.pipelines.base import Pipeline, Segmenter
from voice_segmentation.pipelines.silence import SilencePipeline
from voice_segmentation.pipelines.webrtc import WebRTCPipeline
from voice_segmentation.result import RunResult, SegmentResult
from voice_segmentation.settings import DurationSettings

__all__ = [
    "DurationSettings",
    "EmptySegmentationError",
    "IOSettings",
    "Pipeline",
    "RunResult",
    "SegmentResult",
    "Segmenter",
    "SilencePipeline",
    "SilenceSettings",
    "VoiceSegmentationError",
    "WebRTCAggressiveness",
    "WebRTCPipeline",
    "WebRTCSettings",
]
