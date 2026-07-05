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
from runner import simulate_once, write_traces, _write_csv
from stats.estimate import mean_ci
from analysis import plots


def _cumavg(values) -> np.ndarray:
    """Media cumulata (progressiva) di una sequenza."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return a
    return np.cumsum(a) / np.arange(1, a.size + 1)


def run_transient(conf: Config) -> dict:
    """
    Esegue le repliche a orizzonte finito e produce i 3 grafici del transitorio.

    Returns:
        dict {chiave_metrica: curves}, con curves = lista di (etichetta_seed, x, y),
        cosi il chiamante (experiment.py) puo' ricomporre i pannelli di confronto tra c.
    """
    conf.max_time = conf.finite_horizon
    print("=== ANALISI DEL TRANSITORIO (repliche indipendenti, orizzonte "
          f"finito = {conf.finite_horizon:.0f}s, {len(conf.seeds)} seed) ===")

    util_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    svc_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    resp_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    # Stima per replica = media cumulata finale (= media su tutto l'orizzonte finito).
    finals = {"utilization": {}, "service_passed": {}, "response": {}}

    for seed in conf.seeds:
        sim = simulate_once(conf, seed)
        write_traces(sim, os.path.join(conf.trace_dir, "transient", f"seed{seed}"))
        label = f"seed {seed}"

        # Utilizzazione: media cumulata dei campioni di stato, asse x = tempo simulato.
        st = sim.state_trace  # (time, util, queue, servers, feedback)
        if st:
            t = np.array([r[0] for r in st], dtype=float)
            y = _cumavg([r[1] for r in st])
            util_curves.append((label, t, y))
            finals["utilization"][seed] = float(y[-1])

        # Tempo di servizio (SOLO passed): media cumulata vs indice del job passed.
        passed = [r[3] for r in sim.node_trace if r[1] == "passed"]  # service_visit
        if passed:
            y = _cumavg(passed)
            svc_curves.append((label, np.arange(1, len(passed) + 1), y))
            finals["service_passed"][seed] = float(y[-1])

        # Tempo di risposta: media cumulata vs indice del job uscito (ordine di uscita).
        resp = [r[4] for r in sorted(sim.job_trace, key=lambda row: row[1])]
        if resp:
            y = _cumavg(resp)
            resp_curves.append((label, np.arange(1, len(resp) + 1), y))
            finals["response"][seed] = float(y[-1])

        print(f"  {label}: {len(sim.job_trace)} job usciti, {len(passed)} passed"
              + (f", utilizz. finale ~ {util_curves[-1][2][-1]:.4f}" if st else ""))

    _write_transient_summary(conf, finals)

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

    return {"utilization": util_curves, "service_passed": svc_curves, "response": resp_curves}


def _write_transient_summary(conf: Config, finals: dict) -> None:
    """
    CSV numerico dell'analisi del transitorio: per ogni metrica la stima di ciascuna
    replica (media cumulata finale = media sull'intero orizzonte finito), poi media e
    IC 95% (t di Student) sulle repliche. E' la forma numerica del metodo delle repliche
    indipendenti, complementare ai grafici.
    """
    metrics = [("utilization", "Utilizzazione"),
               ("service_passed", "Tempo di servizio passed [s]"),
               ("response", "Tempo di risposta [s]")]
    header = ["metrica"] + [f"seed_{s}" for s in conf.seeds] + ["media_repliche", "ic_half_95"]
    rows = []
    for key, label in metrics:
        vals = [finals[key].get(s) for s in conf.seeds]
        clean = [v for v in vals if v is not None]
        mean, half = mean_ci(clean)
        row = ([label]
               + [f"{v:.6f}" if v is not None else "" for v in vals]
               + [f"{mean:.6f}", f"{half:.6f}"])
        rows.append(row)
    path = os.path.join(conf.trace_dir, "transient_summary.csv")
    _write_csv(path, header, rows)
    print(f"  Riepilogo numerico transitorio -> {path}")
