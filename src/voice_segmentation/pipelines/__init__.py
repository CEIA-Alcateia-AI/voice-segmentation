"""Pipelines pre-configurados prontos para uso."""

from voice_segmentation.pipelines.silence import SilencePipeline
from voice_segmentation.pipelines.webrtc import WebRTCPipeline

__all__ = [
    "SilencePipeline",
    "WebRTCPipeline",
]
