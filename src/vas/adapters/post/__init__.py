"""Post-processor adapters."""

from vas.adapters.post.duration import DurationPostProcessor
from vas.adapters.post.noop import NoOpPostProcessor

__all__ = ["DurationPostProcessor", "NoOpPostProcessor"]
