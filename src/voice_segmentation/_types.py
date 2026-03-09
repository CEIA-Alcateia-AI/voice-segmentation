"""Tipos fundamentais compartilhados por toda a biblioteca."""

import numpy as np
import numpy.typing as npt

# Áudio mono como array de floats 32-bit
type AudioArray = npt.NDArray[np.float32]

# Segmento de tempo: (início_segundos, fim_segundos)
type Segment = tuple[float, float]
