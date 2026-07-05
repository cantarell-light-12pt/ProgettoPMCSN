"""
Analisi del transitorio con il metodo di Welch.

Date N repliche indipendenti, si calcola per ogni "istante" (indice di campione
per le metriche di stato, indice di job per le metriche per-job) la media di
ensemble sulle repliche; opzionalmente si applica la media mobile di Welch per
lisciare la curva e si stima l'istante di fine transitorio (warm-up).

Le tracce sono liste di tuple prodotte dal simulatore:
- state_trace: (time, utilization, queue_len, servers_busy, feedback_len)
- job_trace:   (job_id, exit_time, wait_time, service_time, response_time)
"""
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def _ensemble(series: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    """
    Media di ensemble su repliche di lunghezza eventualmente diversa: le serie
    vengono troncate alla lunghezza minima comune, cosi ogni punto medio e'
    calcolato sullo stesso numero di repliche (approccio standard di Welch).

    Returns:
        (media_per_indice, lunghezza_comune)
    """
    if not series:
        return np.array([]), 0
    n = min(len(s) for s in series)
    if n == 0:
        return np.array([]), 0
    stacked = np.vstack([s[:n] for s in series])
    return stacked.mean(axis=0), n


def ensemble_state(state_traces: Sequence[Sequence[tuple]],
                   extractor: Callable[[tuple], float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Media di ensemble di una metrica di stato lungo la griglia temporale comune.

    Args:
        state_traces: una state_trace per replica.
        extractor:    funzione che estrae il valore scalare da una riga di traccia.

    Returns:
        (tempi, media_di_ensemble) sulla griglia temporale.
    """
    series = [np.array([extractor(row) for row in tr], dtype=float) for tr in state_traces]
    mean, n = _ensemble(series)
    times = np.array([row[0] for row in state_traces[0][:n]], dtype=float) if n else np.array([])
    return times, mean


def ensemble_job(job_traces: Sequence[Sequence[tuple]],
                 col: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Media di ensemble di una metrica per-job lungo l'indice di job.

    Le tracce sono ordinate per job_id (ordine di creazione/arrivo) e non per
    istante di uscita: ordinare per uscita introdurrebbe un bias di selezione
    (i job piu' brevi escono prima), simulando un falso transitorio.

    Returns:
        (indici_job a partire da 1, media_di_ensemble).
    """
    series = [np.array([row[col] for row in sorted(tr, key=lambda r: r[0])], dtype=float)
              for tr in job_traces]
    mean, n = _ensemble(series)
    idx = np.arange(1, n + 1, dtype=float) if n else np.array([])
    return idx, mean


def welch_moving_average(y: np.ndarray, w: int) -> np.ndarray:
    """
    Media mobile di Welch con finestra w (0 => nessun lisciamento).

    Per i > w:  media su [i-w, i+w] (2w+1 punti).
    Per i <= w: media su [0, 2i] (2i+1 punti, finestra ridotta ai bordi).
    """
    n = len(y)
    if w <= 0 or n == 0:
        return y.copy()
    smoothed = np.empty(n, dtype=float)
    for i in range(n):
        half = min(i, n - 1 - i, w)
        smoothed[i] = y[i - half:i + half + 1].mean()
    return smoothed


def detect_warmup(x: np.ndarray, y: np.ndarray,
                  tol: float = 0.05, tail_frac: float = 0.2,
                  edge: int = 0, k_sigma: float = 3.0) -> Optional[float]:
    """
    Stima euristica dell'istante/indice di fine transitorio.

    Si prende come riferimento la media della coda finale (ultimo tail_frac della
    regione affidabile) e si trova il primo punto a partire dal quale tutti i
    valori successivi restano entro una banda attorno al riferimento.

    La banda e' consapevole del rumore: max(tol relativa, k_sigma * dev.std della
    coda). Cosi un segnale a regime ma rumoroso (tipico delle metriche di stato a
    carico basso, con alta varianza rispetto alla media) viene riconosciuto come
    convergente, mentre un vero transitorio - che si discosta dal riferimento di
    molte deviazioni standard - resta correttamente rilevato.

    Args:
        edge: numero di punti finali da ignorare. La media mobile di Welch usa
            una finestra ridotta ai bordi, quindi gli ultimi ~w punti sono
            rumorosi: vanno esclusi dalla verifica (prassi standard: la curva di
            Welch si esamina su [1, m-w]).
        k_sigma: numero di deviazioni standard della coda che definisce la banda
            di fluttuazione a regime.

    Returns:
        Il valore di x (tempo o indice job) di inizio steady-state, oppure None se
        la serie non si stabilizza entro l'orizzonte simulato.
    """
    n = len(y)
    if n < 5:
        return None
    m = n - edge if 0 < edge < n - 4 else n   # regione affidabile [0, m)
    yr = y[:m]

    tail_start = max(1, int(m * (1.0 - tail_frac)))
    tail = yr[tail_start:]
    reference = tail.mean()
    sigma_tail = tail.std()
    band = max(tol * abs(reference), k_sigma * sigma_tail)

    within = np.abs(yr - reference) <= band
    # cerca il primo indice da cui in poi 'within' e' sempre True (nella regione affidabile)
    for i in range(m):
        if within[i:].all():
            return float(x[i])
    return None


def mser5(x: np.ndarray, y: np.ndarray, batch_size: int = 5) -> Optional[float]:
    """
    Regola MSER-5 (Marginal Standard Error Rule, batch 5) per stimare il troncamento
    del transitorio (White; Robinson; Cobb). Criterio piu' rigoroso dell'euristica.

    La serie viene raggruppata in batch da 'batch_size' (media per batch, riduce
    l'autocorrelazione), poi si sceglie il troncamento d che MINIMIZZA lo standard
    error della media troncata:

        MSER(d) = sum_{i>d} (z_i - z_bar(d))^2 / (m - d)^2

    dove z sono le medie di batch e z_bar(d) e' la media dei batch dopo d. La ricerca
    e' limitata alla prima meta' (non si tronca piu' di meta' serie); se il minimo
    cade sul bordo di tale regione, la serie non si e' stabilizzata -> None.

    Opera sulla media di ensemble (non sulla curva lisciata di Welch): il batching
    fornisce gia' la riduzione del rumore.

    Returns:
        L'ascissa (tempo o indice job) di inizio steady-state, oppure None.
    """
    n = len(y)
    if n < 4 * batch_size:
        return None
    m = n // batch_size
    b = np.array([y[j * batch_size:(j + 1) * batch_size].mean() for j in range(m)])
    xb = np.array([x[j * batch_size] for j in range(m)])

    d_limit = m // 2
    best_d, best_stat = 0, np.inf
    for d in range(0, d_limit + 1):
        retained = b[d:]
        k = len(retained)
        mean = retained.mean()
        stat = float(np.sum((retained - mean) ** 2) / (k * k))
        if stat < best_stat:
            best_stat = stat
            best_d = d
    if best_d >= d_limit:      # minimo sul bordo -> nessuna stabilizzazione
        return None
    return float(xb[best_d])
