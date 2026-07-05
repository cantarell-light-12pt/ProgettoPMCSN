"""
Analisi del transitorio con il METODO DELLE REPLICHE INDIPENDENTI (orizzonte finito).

Si esegue la simulazione una volta per ciascuno dei seed in config (5 repliche
indipendenti). Per ogni replica si calcola la MEDIA CUMULATA (media progressiva) di
utilizzazione, tempo di servizio dei soli job 'passed' e tempo di risposta; le 5 curve
sono sovrapposte sullo stesso grafico (una per metrica) per mostrare la convergenza al
regime e la variabilita' tra i seed.
"""
import os
from typing import List, Tuple

import numpy as np

from config.settings import Config
from runner import simulate_once, write_traces
from analysis import plots


def _cumavg(values) -> np.ndarray:
    """Media cumulata (progressiva) di una sequenza."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return a
    return np.cumsum(a) / np.arange(1, a.size + 1)


def run_transient(conf: Config) -> None:
    """Esegue le repliche a orizzonte finito e produce i 3 grafici del transitorio."""
    conf.max_time = conf.finite_horizon
    print("=== ANALISI DEL TRANSITORIO (repliche indipendenti, orizzonte "
          f"finito = {conf.finite_horizon:.0f}s, {len(conf.seeds)} seed) ===")

    util_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    svc_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    resp_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for seed in conf.seeds:
        sim = simulate_once(conf, seed)
        write_traces(sim, os.path.join(conf.trace_dir, "transient", f"seed{seed}"))
        label = f"seed {seed}"

        # Utilizzazione: media cumulata dei campioni di stato, asse x = tempo simulato.
        st = sim.state_trace  # (time, util, queue, servers, feedback)
        if st:
            t = np.array([r[0] for r in st], dtype=float)
            util_curves.append((label, t, _cumavg([r[1] for r in st])))

        # Tempo di servizio (SOLO passed): media cumulata vs indice del job passed.
        passed = [r[3] for r in sim.node_trace if r[1] == "passed"]  # service_visit
        if passed:
            svc_curves.append((label, np.arange(1, len(passed) + 1), _cumavg(passed)))

        # Tempo di risposta: media cumulata vs indice del job uscito (ordine di uscita).
        resp = [r[4] for r in sorted(sim.job_trace, key=lambda row: row[1])]
        if resp:
            resp_curves.append((label, np.arange(1, len(resp) + 1), _cumavg(resp)))

        print(f"  {label}: {len(sim.job_trace)} job usciti, {len(passed)} passed"
              + (f", utilizz. finale ~ {util_curves[-1][2][-1]:.4f}" if st else ""))

    p1 = plots.plot_transient_replications(
        "utilization", "Transitorio - Utilizzazione (media cumulata)",
        "Tempo simulato [s]", "Utilizzazione", util_curves, conf.trace_dir)
    p2 = plots.plot_transient_replications(
        "service_passed", "Transitorio - Tempo di servizio, job passed (media cumulata)",
        "Indice job passed", "Tempo di servizio [s]", svc_curves, conf.trace_dir)
    p3 = plots.plot_transient_replications(
        "response", "Transitorio - Tempo di risposta (media cumulata)",
        "Indice job (uscita)", "Tempo di risposta [s]", resp_curves, conf.trace_dir)
    print(f"  Grafici salvati: {p1}, {p2}, {p3}")
