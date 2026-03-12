"""Testes para DurationSettings."""

import pytest

from voice_segmentation.settings import DurationSettings


def test_defaults():
    s = DurationSettings()
    assert s.soft_lower == 1.0
    assert s.soft_upper == 30.0
    assert s.hard_lower == 0.5
    assert s.hard_upper == 60.0
    assert s.overlap == 0.0
    assert s.max_gap == 0.0


def test_custom_values():
    s = DurationSettings(soft_lower=5.0, soft_upper=20.0, hard_lower=2.0, hard_upper=30.0)
    assert s.soft_lower == 5.0
    assert s.hard_upper == 30.0


def test_hierarchy_hard_lower_exceeds_soft_lower():
    with pytest.raises(ValueError, match="hard_lower"):
        DurationSettings(hard_lower=5.0, soft_lower=3.0)


def test_hierarchy_soft_lower_exceeds_soft_upper():
    with pytest.raises(ValueError, match="soft_lower"):
        DurationSettings(soft_lower=20.0, soft_upper=10.0)


def test_hierarchy_soft_upper_exceeds_hard_upper():
    with pytest.raises(ValueError, match="soft_upper"):
        DurationSettings(soft_upper=40.0, hard_upper=30.0)


def test_equal_boundaries_allowed():
    # Limites iguais são válidos (ex: segmento de duração exata)
    DurationSettings(hard_lower=5.0, soft_lower=5.0, soft_upper=5.0, hard_upper=5.0)


def test_overlap_must_be_nonnegative():
    with pytest.raises(ValueError):
        DurationSettings(overlap=-0.1)


def test_max_gap_must_be_nonnegative():
    with pytest.raises(ValueError):
        DurationSettings(max_gap=-1.0)
