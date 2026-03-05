"""Low-level numeric helpers used throughout the package."""


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert a duration in seconds to a sample count.

    Args:
        seconds (float): Duration in seconds.
        sample_rate (int): Sample rate in Hz.

    Returns:
        int: Equivalent number of samples (floored to the nearest integer).

    """
    return int(seconds * sample_rate)


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert a sample count to a duration in seconds.

    Args:
        samples (int): Number of samples.
        sample_rate (int): Sample rate in Hz.

    Returns:
        float: Equivalent duration in seconds.

    """
    return samples / sample_rate
