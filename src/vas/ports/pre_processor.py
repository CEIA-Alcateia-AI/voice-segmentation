"""Port: audio pre-processing."""

from typing import Any, Protocol

from numpy.typing import NDArray


class AudioPreProcessor(Protocol):
    """Prepares a raw audio array before it is passed to the core processor.

    Implementations may perform any combination of resampling, normalisation,
    channel mixing, noise reduction, filtering, etc.
    A no-op implementation is valid when no preparation is required.
    """

    def process(self, audio: NDArray[Any]) -> NDArray[Any]:
        """Prepare an audio array for segmentation.

        Args:
            audio (NDArray): Raw input audio array.

        Returns:
            NDArray: Prepared audio array (may be the same object if no changes are
                needed).

        """
        ...
