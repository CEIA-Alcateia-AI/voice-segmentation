"""Abstract base settings for the project."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseSettings):
    """Base settings for the project.

    All concrete settings classes should inherit from this class and define
    their own env_prefix in model_config.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        extra="ignore",
        use_enum_values=True,
    )
