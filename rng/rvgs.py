"""
Generatori di variate casuali (Random Variate GeneratorS), porting Python
autonomo di 'rvgs.c' (Leemis & Park). Tutti costruiti sopra rngs.Random(),
cosi da consumare lo stream attualmente selezionato.

Prima di ogni chiamata il chiamante deve aver selezionato lo stream desiderato
tramite rngs.SelectStream(...), in modo da assegnare uno stream indipendente a
ciascuna sorgente stocastica del modello.
"""
import math
from typing import List, Sequence

from rng.rngs import Random


def Uniform(a: float, b: float) -> float:
    """Variate Uniforme continua su (a, b), con a < b."""
    return a + (b - a) * Random()


def Exponential(m: float) -> float:
    """
    Variate Esponenziale con media m > 0.
    Equivale a random.expovariate(1/m): per un tasso lambda usare Exponential(1/lambda).
    """
    return -m * math.log(1.0 - Random())


def Normal(m: float, s: float) -> float:
    """
    Variate Normale con media m e deviazione standard s > 0.
    Usa l'approssimazione accuratissima della idf normale di Odeh & Evans (1974),
    come nell'originale rvgs.c: una sola chiamata a Random() per variate.
    """
    p0 = 0.322232431088
    q0 = 0.099348462606
    p1 = 1.0
    q1 = 0.588581570495
    p2 = 0.342242088547
    q2 = 0.531103462366
    p3 = 0.204231210245e-1
    q3 = 0.103537752850
    p4 = 0.453642210148e-4
    q4 = 0.385607006340e-2

    u = Random()
    if u < 0.5:
        t = math.sqrt(-2.0 * math.log(u))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - u))
    p = p0 + t * (p1 + t * (p2 + t * (p3 + t * p4)))
    q = q0 + t * (q1 + t * (q2 + t * (q3 + t * q4)))
    if u < 0.5:
        z = (p / q) - t
    else:
        z = t - (p / q)
    return m + s * z


def Lognormal(a: float, b: float) -> float:
    """
    Variate Lognormale: exp(Normal(a, b)), dove a e b sono media e deviazione
    standard della normale sottostante. Equivale a random.lognormvariate(a, b).
    """
    return math.exp(a + b * Normal(0.0, 1.0))


def Bernoulli(p: float) -> int:
    """Variate di Bernoulli: 1 con probabilita p, 0 con probabilita 1 - p."""
    return 0 if Random() < (1.0 - p) else 1


def Empirical(values: Sequence[float], probs: Sequence[float]) -> float:
    """
    Campiona da una Probability Mass Function (PMF) empirica: values[i] con
    probabilita probs[i]. Replica la semantica cumulativa del precedente
    _sample_pmf (rand <= cumulata), consumando lo stream selezionato.
    """
    rand = Random()
    cumulative = 0.0
    for v, p in zip(values, probs):
        cumulative += p
        if rand <= cumulative:
            return v
    return values[-1]
