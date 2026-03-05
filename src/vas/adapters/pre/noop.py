"""No-op pre-processor - passes audio through unchanged."""

from typing import Any

from numpy.typing import NDArray


class NoOpPreProcessor:
    """Pre-processor that returns the audio array untouched.

    Use this when the core processor can work directly on the raw audio
    and no preparation step is needed.
    """

    def process(self, audio: NDArray[Any]) -> NDArray[Any]:
        """Return the audio array unchanged.

        Args:
            audio (NDArray[Any]): Input audio array.

        Returns:
            NDArray[Any]: The same audio array, unmodified.

        """
        return audio
