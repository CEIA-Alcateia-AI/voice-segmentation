"""Testes para o módulo de pós-processamento de duração."""

import pytest

from voice_segmentation.post.duration import (
    _apply_overlap,
    _filter_by_hard_limits,
    _merge_short_segments,
    _split_long_segments,
    enforce_duration,
)
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import Segment


def _settings(**kwargs) -> DurationSettings:
    defaults = {"hard_lower": 0.5, "soft_lower": 2.0, "soft_upper": 10.0, "hard_upper": 20.0}
    defaults.update(kwargs)
    return DurationSettings(**defaults)


# ---------------------------------------------------------------------------
# _merge_short_segments
# ---------------------------------------------------------------------------


def test_merge_empty():
    assert _merge_short_segments([], _settings()) == []


def test_merge_single_segment_already_long_enough():
    segs: list[Segment] = [(0.0, 5.0)]
    result = _merge_short_segments(segs, _settings())
    assert result == segs


def test_merge_short_into_right_neighbour():
    # Segmento de 0.5s (< soft_lower 2.0) deve ser mesclado com o seguinte
    segs: list[Segment] = [(0.0, 0.5), (1.0, 5.0)]
    settings = _settings(max_gap=1.0)
    result = _merge_short_segments(segs, settings)
    assert len(result) == 1
    assert result[0] == (0.0, 5.0)


def test_merge_short_into_left_neighbour():
    segs: list[Segment] = [(0.0, 5.0), (5.5, 6.0)]
    settings = _settings(max_gap=1.0)
    result = _merge_short_segments(segs, settings)
    assert len(result) == 1
    assert result[0] == (0.0, 6.0)


def test_merge_respects_max_gap():
    # Gap de 3s > max_gap de 0.5s: não deve mesclar
    segs: list[Segment] = [(0.0, 0.8), (3.8, 8.0)]
    settings = _settings(max_gap=0.5)
    result = _merge_short_segments(segs, settings)
    # O segmento curto fica sem vizinho compatível e passa para o filtro hard
    assert len(result) == 2


def test_merge_respects_hard_upper():
    # Mescla resultaria em 25s > hard_upper 20s: não deve mesclar
    segs: list[Segment] = [(0.0, 1.0), (1.5, 20.5)]
    settings = _settings(max_gap=1.0)
    result = _merge_short_segments(segs, settings)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _split_long_segments
# ---------------------------------------------------------------------------


def test_split_empty():
    assert _split_long_segments([], _settings()) == []


def test_split_short_segment_unchanged():
    segs: list[Segment] = [(0.0, 5.0)]
    assert _split_long_segments(segs, _settings()) == segs


def test_split_long_segment_into_equal_parts():
    # 20s com soft_upper=10s → 2 partes de 10s
    segs: list[Segment] = [(0.0, 20.0)]
    result = _split_long_segments(segs, _settings(soft_upper=10.0))
    assert len(result) == 2
    assert result[0] == pytest.approx((0.0, 10.0))
    assert result[1] == pytest.approx((10.0, 20.0))


def test_split_ceil_division():
    # 15s com soft_upper=10s → ceil(15/10)=2 partes de 7.5s
    segs: list[Segment] = [(0.0, 15.0)]
    result = _split_long_segments(segs, _settings(soft_upper=10.0))
    assert len(result) == 2
    for s, e in result:
        assert (e - s) == pytest.approx(7.5)


def test_split_drops_parts_below_hard_lower():
    # 4.1s com soft_upper=4.0 → 2 partes de 2.05s cada
    # Se hard_lower=2.5, ambas caem abaixo → lista vazia
    segs: list[Segment] = [(0.0, 4.1)]
    settings = _settings(soft_lower=2.5, soft_upper=4.0, hard_lower=2.5)
    result = _split_long_segments(segs, settings)
    # Nenhuma parte sobrevive pois 2.05 < hard_lower 2.5
    assert len(result) == 0


# ---------------------------------------------------------------------------
# _apply_overlap
# ---------------------------------------------------------------------------


def test_apply_overlap_zero():
    segs: list[Segment] = [(1.0, 3.0)]
    assert _apply_overlap(segs, _settings(overlap=0.0), 10.0) == segs


def test_apply_overlap_expands_both_sides():
    segs: list[Segment] = [(1.0, 3.0)]
    result = _apply_overlap(segs, _settings(overlap=1.0), 10.0)
    assert result[0] == pytest.approx((0.5, 3.5))


def test_apply_overlap_clamps_to_audio_bounds():
    segs: list[Segment] = [(0.1, 4.9)]
    result = _apply_overlap(segs, _settings(overlap=1.0), 5.0)
    assert result[0][0] == pytest.approx(0.0)
    assert result[0][1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _filter_by_hard_limits
# ---------------------------------------------------------------------------


def test_filter_keeps_valid_segments():
    segs: list[Segment] = [(0.0, 5.0), (6.0, 12.0)]
    settings = _settings(hard_lower=1.0, hard_upper=15.0)
    assert _filter_by_hard_limits(segs, settings) == segs


def test_filter_removes_below_hard_lower():
    segs: list[Segment] = [(0.0, 0.3), (1.0, 5.0)]
    settings = _settings(hard_lower=0.5)
    result = _filter_by_hard_limits(segs, settings)
    assert len(result) == 1
    assert result[0] == (1.0, 5.0)


def test_filter_removes_above_hard_upper():
    segs: list[Segment] = [(0.0, 5.0), (6.0, 30.0)]
    settings = _settings(hard_upper=20.0)
    result = _filter_by_hard_limits(segs, settings)
    assert len(result) == 1
    assert result[0] == (0.0, 5.0)


# ---------------------------------------------------------------------------
# enforce_duration — integração
# ---------------------------------------------------------------------------


def test_enforce_duration_empty():
    assert enforce_duration([], _settings(), 10.0) == []


def test_enforce_duration_typical():
    # 3 segmentos normais dentro dos limites → passam sem modificação
    segs: list[Segment] = [(0.0, 3.0), (4.0, 7.0), (8.0, 11.0)]
    settings = _settings(soft_lower=2.0, soft_upper=10.0, hard_lower=1.0, hard_upper=15.0)
    result = enforce_duration(segs, settings, 12.0)
    # Devem sobreviver todos os segmentos (podem ser mesclados ou divididos levemente)
    assert len(result) > 0
    for s, e in result:
        assert (e - s) >= settings.hard_lower
        assert (e - s) <= settings.hard_upper


def test_enforce_duration_all_too_short_without_merging():
    # Segmentos isolados de 0.3s, gap 5s entre eles → abaixo de soft_lower, sem vizinho
    # compatível pelo max_gap 0 → são descartados pelo filtro hard
    segs: list[Segment] = [(0.0, 0.3), (5.0, 5.3), (10.0, 10.3)]
    settings = DurationSettings(
        soft_lower=2.0, soft_upper=10.0, hard_lower=1.0, hard_upper=20.0, max_gap=0.0
    )
    result = enforce_duration(segs, settings, 12.0)
    assert result == []
