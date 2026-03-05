"""Duration constraint settings for segment post-processing."""

from typing import Self

from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from vas.settings.base import ProjectSettings


class DurationSettings(ProjectSettings):
    """Settings that control how detected segments are filtered and merged."""

    model_config = ProjectSettings.model_config.copy()
    model_config["env_prefix"] = "VAS_DURATION__"

    hard_lower_limit: PositiveFloat = Field(
        default=0.5,
        description="Minimum segment duration in seconds (hard discard below this).",
    )

    soft_lower_limit: PositiveFloat = Field(
        default=3.0,
        description="Minimum desired segment duration in seconds (triggers merging).",
    )

    soft_upper_limit: PositiveFloat = Field(
        default=15.0,
        description=("Maximum desired segment duration in seconds (merge target "
        "ceiling).")
    )

    hard_upper_limit: PositiveFloat = Field(
        default=30.0,
        description="Maximum segment duration in seconds (hard discard above this).",
    )

    overlap: NonNegativeFloat = Field(
        default=0.2,
        description=(
            "Total overlap in seconds added symmetrically to each segment. "
            "Set to 0 to disable."
        ),
    )

    maximum_merge_gap_duration: NonNegativeFloat = Field(
        default=1.0,
        description=(
            "Maximum silence gap in seconds between two segments that can be merged."
        ),
    )

    @model_validator(mode="after")
    def _validate_limit_ordering(self) -> Self:
        """Ensure the four limits are in ascending order."""
        if not (
            self.hard_lower_limit
            <= self.soft_lower_limit
            <= self.soft_upper_limit
            <= self.hard_upper_limit
        ):
            error_message = (
                "Duration limits must satisfy: "
                "hard_lower_limit ≤ soft_lower_limit ≤ "
                "soft_upper_limit ≤ hard_upper_limit. "
                f"Got: {self.hard_lower_limit} / {self.soft_lower_limit} / "
                f"{self.soft_upper_limit} / {self.hard_upper_limit}."
            )
            raise ValueError(error_message)
        return self
