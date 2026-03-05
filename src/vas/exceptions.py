"""Domain exceptions for the voice segmentation package."""


class InvalidTimestampError(ValueError):
    """Raised when a (start, end) timestamp fails validation."""

    def __init__(self, start: float, end: float, reason: str) -> None:
        """Initialise with the offending timestamp and a human-readable reason.

        Args:
            start (float): Start time in seconds.
            end (float): End time in seconds.
            reason (str): Description of why the timestamp is invalid.

        """
        self.start = start
        self.end = end
        self.reason = reason
        super().__init__(f"Invalid timestamp [{start:.3f}s, {end:.3f}s]: {reason}")


class SegmentationError(RuntimeError):
    """Raised when a core processor fails to produce segments."""

    def __init__(self, processor_name: str, reason: str) -> None:
        """Initialise with the processor's class name and a reason.

        Args:
            processor_name (str): Name of the processor that failed.
            reason (str): Description of the failure.

        """
        self.processor_name = processor_name
        self.reason = reason
        super().__init__(f"Processor '{processor_name}' failed: {reason}")
