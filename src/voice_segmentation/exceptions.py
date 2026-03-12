class VoiceSegmentationError(Exception):
    """Classe base para todas as exceções da biblioteca."""


class SilenceDetectionError(VoiceSegmentationError):
    """Lançada quando a detecção de silêncios falha inesperadamente.

    Args:
        cause: Mensagem descrevendo a causa subjacente do erro.
    """

    def __init__(self, cause: str) -> None:
        """Inicializa com a causa do erro."""
        super().__init__(f"Falha na detecção de silêncios: {cause}")


class EmptySegmentationError(VoiceSegmentationError):
    """Lançada quando nenhum segmento válido é produzido após o processamento.

    Args:
        message: Mensagem explicando por que nenhum segmento foi gerado.
    """

    def __init__(self, message: str) -> None:
        """Inicializa com a mensagem de erro."""
        super().__init__(message)
