"""Aplicação de limites de duração sobre listas de segmentos."""

import logging
import math

from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import Segment

logger = logging.getLogger(__name__)


def _merge_short_segments(
    segments: list[Segment],
    settings: DurationSettings,
) -> list[Segment]:
    """Mescla segmentos abaixo do mínimo desejado com o melhor vizinho disponível.

    Um segmento abaixo de soft_lower é mesclado com o vizinho (esquerdo ou direito) que minimize
    a distância ao ponto médio do intervalo suave (soft_lower + soft_upper) / 2. A mesclagem só
    ocorre se o gap entre os segmentos respeitar max_gap e a duração resultante não superar
    hard_upper.

    Args:
        segments: Lista de segmentos (início, fim) em segundos, ordenada cronologicamente.
        settings: Configurações de duração e mesclagem.

    Returns:
        Nova lista de segmentos após a mesclagem dos segmentos curtos.
    """
    if not segments:
        return []

    result: list[Segment] = list(segments)
    target = (settings.soft_lower + settings.soft_upper) / 2.0

    i = 0
    while i < len(result):
        start, end = result[i]
        duration = end - start

        # Segmento já está dentro do intervalo desejado - avanca
        if duration >= settings.soft_lower:
            i += 1
            continue

        # Avalia possibilidade de mesclagem com o segmento anterior
        can_merge_left = False
        left_merge_score = float("inf")
        if i > 0:
            previous_segment_start, previous_segment_end = result[i - 1]
            gap_left = start - previous_segment_end
            merged_duration_left = end - previous_segment_start
            if gap_left <= settings.max_gap and merged_duration_left <= settings.hard_upper:
                can_merge_left = True
                left_merge_score = abs(merged_duration_left - target)

        # Avalia possibilidade de mesclagem com o segmento seguinte
        can_merge_right = False
        right_merge_score = float("inf")
        if i < len(result) - 1:
            next_segment_start, next_segment_end = result[i + 1]
            gap_right = next_segment_start - end
            merged_duration_right = next_segment_end - start
            if gap_right <= settings.max_gap and merged_duration_right <= settings.hard_upper:
                can_merge_right = True
                right_merge_score = abs(merged_duration_right - target)

        if not can_merge_left and not can_merge_right:
            # Segmento curto sem vizinho compatível - será tratado pelo filtro hard
            logger.debug(
                "Segmento %.2fs-%.2fs (%.2fs) abaixo do mínimo desejado sem vizinho compatível.",
                start,
                end,
                duration,
            )
            i += 1
            continue

        if can_merge_left and (not can_merge_right or left_merge_score <= right_merge_score):
            # Mescla com o segmento anterior (estende o fim do segmento anterior)
            previous_segment_start, _ = result[i - 1]
            result[i - 1] = (previous_segment_start, end)
            result.pop(i)
            logger.debug("Segmento %d mesclado a esquerda.", i)
            # Reavalia o segmento recentemente crescido (pode ainda estar abaixo do mínimo)
            i = max(0, i - 1)

        else:
            # Mescla com o segmento seguinte (recua o início do segmento seguinte)
            _, next_segment_end = result[i + 1]
            result[i + 1] = (start, next_segment_end)
            result.pop(i)
            logger.debug("Segmento %d mesclado a direita.", i)
            # i permanece: O segmento seguinte desceu para a posição i

    return result


def _split_long_segments(
    segments: list[Segment],
    settings: DurationSettings,
) -> list[Segment]:
    """Divide segmentos acima de soft_upper em partes iguais.

    Args:
        segments: Lista de segmentos (início, fim) em segundos.
        settings: Configurações de duração e mesclagem.

    Returns:
        Nova lista com cada segmento longo substituido pelas suas sub-partes.
    """
    result: list[Segment] = []
    for start, end in segments:
        duration = end - start
        if duration <= settings.soft_upper:
            result.append((start, end))
            continue

        number_of_parts = math.ceil(duration / settings.soft_upper)
        part_size = duration / number_of_parts
        logger.debug(
            "Dividindo %.2fs-%.2fs (%.2fs) em %d partes de ~%.2fs.",
            start,
            end,
            duration,
            number_of_parts,
            part_size,
        )
        for k in range(number_of_parts):
            part_start = start + k * part_size
            part_end = start + (k + 1) * part_size
            if (part_end - part_start) >= settings.hard_lower:
                result.append((part_start, part_end))

    return result


def _apply_overlap(
    segments: list[Segment],
    settings: DurationSettings,
    audio_duration: float,
) -> list[Segment]:
    """Adiciona padding de sobreposição as bordas de cada segmento.

    Cada segmento é expandido overlap / 2 segundos em cada direção, respeitando os limites
    [0, audio_duration]. Se overlap for zero, a lista é retornada sem modificação.

    Args:
        segments: Lista de segmentos (início, fim) em segundos.
        settings: Configurações de duração e mesclagem.
        audio_duration: Duração total do áudio em segundos, usada como limite superior do padding.

    Returns:
        Nova lista de segmentos com o padding aplicado.
    """
    if settings.overlap <= 0.0:
        return list(segments)

    pad = settings.overlap / 2.0
    return [(max(0.0, start - pad), min(audio_duration, end + pad)) for start, end in segments]


def _filter_by_hard_limits(
    segments: list[Segment],
    settings: DurationSettings,
) -> list[Segment]:
    """Descarta segmentos fora dos limites absolutos de duração.

    Args:
        segments: Lista de segmentos (início, fim) em segundos.
        settings: Configurações de duração e mesclagem.

    Returns:
        Nova lista contendo apenas segmentos dentro dos limites hard_lower e hard_upper.
    """
    result: list[Segment] = []
    for start, end in segments:
        duration = end - start
        if duration < settings.hard_lower:
            logger.debug(
                "Descartando %.2fs-%.2fs (%.2fs): abaixo do limite hard_lower (%.2fs).",
                start,
                end,
                duration,
                settings.hard_lower,
            )
            continue
        if duration > settings.hard_upper:
            logger.warning(
                "Descartando %.2fs-%.2fs (%.2fs): acima do limite hard_upper (%.2fs).",
                start,
                end,
                duration,
                settings.hard_upper,
            )
            continue
        result.append((start, end))
    return result


def enforce_duration(
    segments: list[Segment],
    settings: DurationSettings,
    audio_duration: float,
) -> list[Segment]:
    """Aplica o pipeline completo de limitação de duração sobre uma lista de segmentos.

    Etapas executadas em ordem:

    1. Mesclagem de curtos: Segmentos abaixo de soft_lower são mesclados com o melhor vizinho.
    2. Divisao de longos: Segmentos acima de soft_upper sao divididos em partes iguais.
    3. Overlap: Cada segmento é expandido overlap / 2 segundos em cada borda.
    4. Filtragem hard: Segmentos fora de [hard_lower, hard_upper] sao descartados.

    Args:
        segments: Lista de segmentos (início, fim) em segundos produzida por uma estratégia do
            core.
        settings: Configurações de duração e mesclagem.
        audio_duration: Duração total do áudio em segundos, necessária para limitar o padding.

    Returns:
        Lista de segmentos processados, ordenada cronologicamente.
    """
    merged = _merge_short_segments(segments, settings)
    split = _split_long_segments(merged, settings)
    overlapped = _apply_overlap(split, settings, audio_duration)
    return _filter_by_hard_limits(overlapped, settings)
