"""
Modulo contenente gli algoritmi online di Welford per il tracciamento
delle statistiche della simulazione in $O(1)$ di memoria.
"""


class StandardWelford:
    """
    Algoritmo di Welford standard per metriche discrete per-entità.
    Calcola la media e predispone il calcolo della varianza on-the-fly.
    """

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, value: float) -> None:
        """
        Aggiorna lo stimatore con un nuovo valore scalare.

        Args:
            value (float): Nuovo dato osservato (es. tempo di attesa di un job).
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2


class TimeWeightedWelford:
    """
    Algoritmo di Welford esteso per metriche di stato continuo (time-weighted).
    """

    def __init__(self) -> None:
        self.total_weight: float = 0.0
        self.mean: float = 0.0

    def update(self, value: float, weight: float) -> None:
        """
        Aggiorna la media ponderata usando il tempo come peso.

        Args:
            value (float): Il valore dello stato mantenuto (es. server occupati).
            weight (float): La durata temporale (delta t) dello stato.
        """
        if weight <= 0:
            return
        self.total_weight += weight
        delta = value - self.mean
        self.mean += (weight / self.total_weight) * delta