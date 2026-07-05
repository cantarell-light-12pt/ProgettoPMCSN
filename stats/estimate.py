"""
Stima intervallare condivisa: media e semi-ampiezza dell'intervallo di confidenza
sul campione delle medie per-replica, con la t di Student a n-1 gradi di liberta'
(come in 'estimate.c' di Leemis & Park).

Il quantile della t e' calcolato con scipy.stats (libreria testata): e' calcolo
statistico di post-processing, non fa parte del motore di simulazione.
"""
import math
from typing import List, Tuple

import numpy as np
from scipy import stats


def mean_ci(samples: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Args:
        samples: campione delle medie per-replica.
        confidence: livello di confidenza (default 0.95).

    Returns:
        (media, semi_ampiezza_IC). La semi-ampiezza e' NaN se n < 2.
        std e' campionaria (ddof=1), coerente con w = t * s / sqrt(n).
    """
    n = len(samples)
    mean = float(np.mean(samples))
    if n < 2:
        return mean, float("nan")
    std = float(np.std(samples, ddof=1))
    t = stats.t.ppf(0.5 + confidence / 2.0, df=n - 1)
    half = t * std / math.sqrt(n)
    return mean, half
