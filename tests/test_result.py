"""Testes para SegmentResult e RunResult."""

from pathlib import Path

import pytest

from voice_segmentation.result import RunResult, SegmentResult
from voice_segmentation.settings import DurationSettings


def test_segment_result_duration():
    seg = SegmentResult(start=1.0, end=4.5)
    assert seg.duration == pytest.approx(3.5)


def test_segment_result_path_defaults_none():
    seg = SegmentResult(start=0.0, end=1.0)
    assert seg.path is None


def test_segment_result_with_path():
    p = Path("/tmp/seg.flac")
    seg = SegmentResult(start=0.0, end=2.0, path=p)
    assert seg.path == p


def test_segment_result_is_frozen():
    seg = SegmentResult(start=0.0, end=1.0)
    with pytest.raises((AttributeError, TypeError)):
        seg.start = 5.0  # type: ignore[misc]


def test_run_result_fields():
    ds = DurationSettings()
    segs = [SegmentResult(start=0.0, end=2.0), SegmentResult(start=3.0, end=5.0)]
    result = RunResult(
        segments=segs,
        processing_time_s=0.42,
        segmenter_name="TestSegmenter",
        segmenter_settings=None,
        duration_settings=ds,
    )
    assert len(result.segments) == 2
    assert result.processing_time_s == pytest.approx(0.42)
    assert result.segmenter_name == "TestSegmenter"
    assert result.segmenter_settings is None
    assert result.output_dir is None


def test_run_result_with_output_dir():
    ds = DurationSettings()
    result = RunResult(
        segments=[],
        processing_time_s=0.1,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=ds,
        output_dir=Path("/out"),
    )
    assert result.output_dir == Path("/out")


def test_run_result_is_frozen():
    ds = DurationSettings()
    result = RunResult(
        segments=[],
        processing_time_s=0.1,
        segmenter_name="X",
        segmenter_settings=None,
        duration_settings=ds,
    )
    with pytest.raises((AttributeError, TypeError)):
        result.processing_time_s = 99.0  # type: ignore[misc]
