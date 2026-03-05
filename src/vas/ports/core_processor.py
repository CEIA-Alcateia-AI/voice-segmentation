"""Port: core segmentation strategy."""

from typing import Any, Protocol

from numpy.typing import NDArray

from vas.ports.types import Segment


class CoreProcessor(Protocol):
    """Detects speech segments in a prepared audio array.

    Implementations are the concrete segmentation strategies (VAD, energy,
    silence-based, model-based, etc.). They operate purely on audio data and
    return raw, sample-based segments with optionally lenient duration rules, leaving
    the enforcement of business rules to the post-processor.
    """

    def process(self, audio: NDArray[Any]) -> list[Segment]:
        """Detect segments in the audio array.

        Args:
            audio (NDArray): Prepared audio array.

        Returns:
            list[Segment]: List of detected segments with start/end expressed in
                samples.

        """
        ...
