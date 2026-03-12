"""Configurações compartilhadas do pipeline de segmentação."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class DurationSettings(BaseModel):
    """Limites de duração e parâmetros de mesclagem de segmentos.

    Os limites seguem a hierarquia obrigatoria:
        - hard_lower <= soft_lower <= soft_upper <= hard_upper.

    Os limites soft (suaves) representam a duração desejada; os hard (rigidos) representam os
    extremos toleráveis. Segmentos fora dos limites hard sao descartados ou reparticionados pelo
    pós-processamento.

    Attributes:
        soft_lower: Duração mínima desejada de um segmento, em segundos.
        soft_upper: Duração máxima desejada de um segmento, em segundos.
        hard_lower: Duração mínima tolerável de um segmento em segundos. Segmentos abaixo deste
            valor sao descartados.
        hard_upper: Duração máxima tolerável de um segmento em segundos. Segmentos acima deste
            valor sao descartados.
        overlap: Sobreposição entre segmentos consecutivos, em segundos.
        max_gap: Intervalo máximo entre dois segmentos para que sejam mesclados em um único,
            em segundos.
    """

    soft_lower: float = Field(default=1.0, gt=0.0, description="Duração mínima desejada (s)")
    soft_upper: float = Field(default=30.0, gt=0.0, description="Duração máxima desejada (s)")
    hard_lower: float = Field(default=0.5, gt=0.0, description="Duração mínima tolerável (s)")
    hard_upper: float = Field(default=60.0, gt=0.0, description="Duração máxima tolerável (s)")
    overlap: float = Field(default=0.0, ge=0.0, description="Sobreposição entre segmentos (s)")
    max_gap: float = Field(
        default=0.0,
        ge=0.0,
        description="Gap máximo antes de mesclar dois segmentos (s)",
    )

    @model_validator(mode="after")
    def _validate_duration_hierarchy(self: Self) -> Self:
        """Válida a hierarquia hard_lower <= soft_lower <= soft_upper <= hard_upper."""
        if self.hard_lower > self.soft_lower:
            raise ValueError(
                f"hard_lower ({self.hard_lower}) deve ser <= soft_lower ({self.soft_lower})"
            )
        if self.soft_lower > self.soft_upper:
            raise ValueError(
                f"soft_lower ({self.soft_lower}) deve ser <= soft_upper ({self.soft_upper})"
            )
        if self.soft_upper > self.hard_upper:
            raise ValueError(
                f"soft_upper ({self.soft_upper}) deve ser <= hard_upper ({self.hard_upper})"
            )
        return self
