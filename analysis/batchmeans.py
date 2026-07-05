"""
Analisi steady-state con il metodo BATCH MEANS (orizzonte infinito).

Una singola run lunga; si elimina il transitorio iniziale (`warmup`) e si divide il
resto in k batch di uguale DURATA. Per ogni batch si calcola la media della metrica;
i k valori di batch danno media + intervallo di confidenza (t di Student, k-1 gdl).
Le metriche time-average (utilizzazione, popolazioni) sono mediate sui campioni di
stato del batch; quelle per-job (servizio passed, risposta, attesa) sui job del batch.

Restituisce un bundle nella stessa forma delle repliche (liste a 1 elemento), cosi'
Verifica e Validazione girano invariate sulla run infinita.
"""
import os
from typing import Dict

import numpy as np

from config.settings import Config
from runner import simulate_once, write_traces, _write_csv
from stats.estimate import mean_ci
from analysis import plots


def _batch_means(values, times, warmup: float, T: float, k: int) -> np.ndarray:
    """Media della metrica per ciascun batch temporale; ritorna i batch non vuoti."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    if v.size == 0:
        return np.array([])
    mask = (t >= warmup) & (t <= T)
    v, t = v[mask], t[mask]
    idx = ((t - warmup) / (T - warmup) * k).astype(int)
    idx = np.clip(idx, 0, k - 1)
    sums = np.bincount(idx, weights=v, minlength=k)
    cnts = np.bincount(idx, minlength=k)
    nonempty = cnts > 0
    return sums[nonempty] / cnts[nonempty]


def run_batch_means(conf: Config, quiet: bool = False) -> Dict[str, list]:
    """Esegue la run infinita, calcola i batch means e ritorna il bundle per la V&V."""
    conf.max_time = conf.run_length
    if not quiet:
        print("\n=== STEADY-STATE (batch means, orizzonte infinito = "
              f"{conf.run_length:.0f}s, c={conf.c}, seed {conf.seed}) ===")

    sim = simulate_once(conf, conf.seed)
    write_traces(sim, conf.trace_dir)

    k, warmup, T = conf.num_batches, conf.warmup, conf.run_length
    st = sim.state_trace                       # (time, util, queue, servers, feedback)
    times_st = [r[0] for r in st]
    jt = sim.job_trace                         # (id, exit, wait, service, response)
    exit_t = [r[1] for r in jt]
    passed = [(r[0], r[3]) for r in sim.node_trace if r[1] == "passed"]  # (t, service_visit)

    # (chiave canonica, etichetta, array dei batch)
    batches = [
        ("utilization", "Utilizzazione",
         _batch_means([r[1] for r in st], times_st, warmup, T, k)),
        ("service_passed", "Tempo di servizio (passed) [s]",
         _batch_means([s for _, s in passed], [t for t, _ in passed], warmup, T, k)),
        ("response_time", "Tempo di risposta [s]",
         _batch_means([r[4] for r in jt], exit_t, warmup, T, k)),
        ("wait_time", "Tempo di attesa [s]",
         _batch_means([r[2] for r in jt], exit_t, warmup, T, k)),
        ("qlen", "Popolazione media in coda",
         _batch_means([r[2] for r in st], times_st, warmup, T, k)),
        ("syslen", "Popolazione media nel sistema",
         _batch_means([r[2] + r[3] for r in st], times_st, warmup, T, k)),
    ]

    rows = []
    estimates: Dict[str, tuple] = {}   # key -> (label, mean, half)
    batches_plot = []                  # (key, label, valori_batch, mean, half) per i pannelli
    if not quiet:
        print(f"  ({k} batch dopo warmup {warmup:.0f}s) - media +/- IC 95% (t di Student)")
        print(f"  {'Metrica':<34}{'#batch':>7}{'Media':>14}{'+/- IC':>13}{'IC rel':>9}")
    for key, label, b in batches:
        mean, half = mean_ci(list(b))
        rel = half / abs(mean) if mean else float("inf")
        estimates[key] = (label, mean, half)
        batches_plot.append((key, label, b, mean, half))
        # Diagramma diagnostico dei batch (valori dei batch + media + banda IC).
        plots.plot_batch_means(key, f"Batch means - {label}", label, b, mean, half,
                               conf.trace_dir)
        if not quiet:
            print(f"  {label:<34}{len(b):>7}{mean:>14.4f}{half:>13.4f}{rel:>8.2%}")
        rows.append([key, len(b), f"{mean:.6f}", f"{half:.6f}", f"{rel:.4f}"])

    path = os.path.join(conf.trace_dir, "batch_means_summary.csv")
    _write_csv(path, ["metric", "n_batches", "mean", "ci_half_width_95", "ci_rel"], rows)
    if not quiet:
        print(f"  Tabella + 6 diagrammi diagnostici (batchmeans_*.png) salvati in {conf.trace_dir}")

    return {
        "state_traces": [sim.state_trace],
        "job_traces": [sim.job_trace],
        "node_traces": [sim.node_trace],
        "batch_traces": [sim.batch_trace],
        "welford_results": [sim.results()],
        "estimates": estimates,
        "batches_plot": batches_plot,
    }
