"""Audio configuration settings."""

from pydantic import Field, PositiveInt

from vas.settings.base import ProjectSettings


class AudioSettings(ProjectSettings):
    """Settings related to audio loading and processing."""

    model_config = ProjectSettings.model_config.copy()
    model_config["env_prefix"] = "VAS_AUDIO__"

    sample_rate_hz: PositiveInt = Field(
        default=16_000,
        description="Target sample rate in Hz.",
    )

    channels: PositiveInt = Field(
        default=1,
        description="Number of audio channels (1 = mono, 2 = stereo).",
    )
