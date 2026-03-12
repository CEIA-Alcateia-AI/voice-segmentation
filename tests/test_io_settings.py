"""Testes para IOSettings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from voice_segmentation.io.settings import IOSettings


def test_defaults():
    s = IOSettings(output_folder=Path("/tmp/out"))
    assert s.create_run_subfolder is True
    assert s.create_segment_subfolders is False
    assert s.write_run_metadata is True
    assert s.write_segment_metadata is True
    assert s.write_segment_spectrograms is False
    assert s.redo_if_exists is True
    assert s.audio_format == "flac"
    assert s.run_name is None
    assert s.segment_prefix is None


def test_audio_format_wav():
    s = IOSettings(output_folder=Path("/tmp"), audio_format="wav")
    assert s.audio_format == "wav"


def test_audio_format_invalid():
    with pytest.raises(ValidationError):
        IOSettings(output_folder=Path("/tmp"), audio_format="mp3")  # type: ignore[arg-type]


def test_custom_run_name():
    s = IOSettings(output_folder=Path("/tmp"), run_name="minha_execucao")
    assert s.run_name == "minha_execucao"
