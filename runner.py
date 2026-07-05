"""
Pipeline riusabile di una singola simulazione (per un dato config).

Esegue N repliche indipendenti, scrive le tracce per replica nella cartella
`conf.trace_dir`, produce l'analisi del transitorio (grafici Welch) e la tabella
numerica riassuntiva (media + IC 95%). Restituisce un "bundle" con tracce e
risultati, riutilizzato sia da main.py (che vi aggiunge Verifica e Validazione)
sia da experiment.py (che ripete la pipeline al variare del numero di serventi,
ciascuna in una cartella dedicata).
"""
import os
import csv
from typing import Dict, List, Sequence

from config.settings import Config
from core.simulator import TravisCISimulator
from rng import rngs
from stats.estimate import mean_ci
from analysis import welch, plots

# Definizione delle metriche di stato: (chiave, titolo, ylabel, extractor).
STATE_METRICS = [
    ("utilization", "Transitorio - Utilizzazione", "Utilizzazione",
     lambda row: row[1]),
    ("qlen", "Transitorio - Popolazione in coda", "Popolazione in coda [job]",
     lambda row: row[2]),
    ("syslen", "Transitorio - Popolazione nel sistema", "Popolazione nel sistema (coda+serventi) [job]",
     lambda row: row[2] + row[3]),
]

# Definizione delle metriche per-job: (chiave, titolo, ylabel, indice_colonna).
JOB_METRICS = [
    ("wait_time", "Transitorio - Tempo di attesa", "Tempo di attesa [s]", 2),
    ("service_time", "Transitorio - Tempo di servizio", "Tempo di servizio [s]", 3),
    ("response_time", "Transitorio - Tempo di risposta", "Tempo di risposta [s]", 4),
]


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence]) -> None:
    """Scrive una traccia su file CSV usando il modulo standard (nessuna dipendenza)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _run_replications(conf: Config, quiet: bool = False) -> Dict[str, list]:
    """Esegue le repliche, scrive le tracce in conf.trace_dir e raccoglie tutto."""
    # Le tracce CSV vanno in sottocartelle per tipo, cosi la cartella della
    # simulazione resta navigabile (grafici e tabella non sono sommersi dai CSV).
    state_dir = os.path.join(conf.trace_dir, "traces", "state")
    jobs_dir = os.path.join(conf.trace_dir, "traces", "jobs")
    node_dir = os.path.join(conf.trace_dir, "traces", "node")
    arrivals_dir = os.path.join(conf.trace_dir, "traces", "arrivals")
    for d in (state_dir, jobs_dir, node_dir, arrivals_dir):
        os.makedirs(d, exist_ok=True)
    rngs.PlantSeeds(conf.seed)

    state_traces: List[list] = []
    job_traces: List[list] = []
    node_traces: List[list] = []
    batch_traces: List[list] = []
    welford_results: List[dict] = []
    per_replica_results: dict = {k: [] for k in
                                 ("utilization", "service_time", "wait_time",
                                  "response_time", "qlen", "syslen")}

    # Numero di repliche adattivo: si continua finche' la semi-ampiezza relativa
    # dell'IC della metrica-obiettivo scende sotto la soglia, entro [min, max].
    r = 0
    while True:
        r += 1
        sim = TravisCISimulator(conf)
        sim.run()

        state_traces.append(sim.state_trace)
        job_traces.append(sim.job_trace)
        node_traces.append(sim.node_trace)
        batch_traces.append(sim.batch_trace)

        _write_csv(
            os.path.join(state_dir, f"state_rep{r:03d}.csv"),
            ["time", "utilization", "queue_len", "servers_busy", "feedback_len"],
            sim.state_trace,
        )
        _write_csv(
            os.path.join(jobs_dir, f"jobs_rep{r:03d}.csv"),
            ["job_id", "exit_time", "wait_time", "service_time", "response_time"],
            sim.job_trace,
        )
        _write_csv(
            os.path.join(node_dir, f"node_rep{r:03d}.csv"),
            ["completion_time", "fate", "wait_visit", "service_visit"],
            sim.node_trace,
        )
        _write_csv(
            os.path.join(arrivals_dir, f"arrivals_rep{r:03d}.csv"),
            ["arrival_time", "batch_size"],
            sim.batch_trace,
        )

        res = sim.results()
        welford_results.append(res)
        for key, value in res.items():
            per_replica_results[key].append(value)

        # Criterio di arresto: precisione raggiunta sulla metrica-obiettivo o cap.
        mean, half = mean_ci(per_replica_results[conf.target_metric])
        rel = half / abs(mean) if mean else float("inf")
        if not quiet:
            print(f"  Replica {r:>3} completata (job usciti: {len(sim.job_trace)}, "
                  f"servizi: {len(sim.node_trace)}); IC rel '{conf.target_metric}' = {rel:.2%}")
        if r >= conf.min_replications and (rel <= conf.target_rel_halfwidth
                                           or r >= conf.max_replications):
            break

    return {
        "state_traces": state_traces,
        "job_traces": job_traces,
        "node_traces": node_traces,
        "batch_traces": batch_traces,
        "welford_results": welford_results,
        "per_replica_results": per_replica_results,
    }


def run_simulation_suite(conf: Config, quiet: bool = False) -> Dict[str, list]:
    """
    Pipeline completa di una simulazione: repliche + tracce + transitorio + tabella,
    tutto nella cartella conf.trace_dir. Restituisce il bundle di tracce e risultati.
    """
    if not quiet:
        print(f"Avvio simulazione: c={conf.c}, repliche adattive "
              f"[min {conf.min_replications}, max {conf.max_replications}, "
              f"stop quando IC rel '{conf.target_metric}' <= {conf.target_rel_halfwidth:.1%}] "
              f"(Seed: {conf.seed}, Max Time: {conf.max_time}s) -> {conf.trace_dir}")

    bundle = _run_replications(conf, quiet=quiet)
    _transient_analysis(conf, bundle["state_traces"], bundle["job_traces"])
    _numeric_summary(conf, bundle["per_replica_results"])
    return bundle


def _transient_analysis(conf: Config, state_traces: List[list],
                        job_traces: List[list]) -> None:
    """
    Media di ensemble (Welch), stima del fine-transitorio con DUE metodi affiancati
    (euristica di Welch noise-aware e MSER-5) e salvataggio dei grafici.
    """
    print("\n=== ANALISI DEL TRANSITORIO (Welch vs MSER-5) ===")

    for key, title, ylabel, extractor in STATE_METRICS:
        x, raw = welch.ensemble_state(state_traces, extractor)
        if len(x) == 0:
            continue
        smoothed = welch.welch_moving_average(raw, conf.welch_window)
        w_welch = welch.detect_warmup(x, smoothed, edge=conf.welch_window)
        w_mser = welch.mser5(x, raw)
        markers = [("Welch", w_welch, "green"), ("MSER-5", w_mser, "purple")]
        path = plots.plot_metric(key, title, "Tempo simulato [s] (scala log)", ylabel,
                                 x, raw, smoothed, markers, conf.trace_dir,
                                 edge=conf.welch_window, logx=True)
        _report_warmups(key, [("Welch", w_welch), ("MSER-5", w_mser)], "s", path)

    for key, title, ylabel, col in JOB_METRICS:
        x, raw = welch.ensemble_job(job_traces, col)
        if len(x) == 0:
            print(f"  {key}: nessun job uscito, transitorio non calcolabile.")
            continue
        smoothed = welch.welch_moving_average(raw, conf.welch_window)
        w_welch = welch.detect_warmup(x, smoothed, edge=conf.welch_window)
        w_mser = welch.mser5(x, raw)
        markers = [("Welch", w_welch, "green"), ("MSER-5", w_mser, "purple")]
        path = plots.plot_metric(key, title, "Indice job (creazione)", ylabel,
                                 x, raw, smoothed, markers, conf.trace_dir,
                                 edge=conf.welch_window)
        _report_warmups(key, [("Welch", w_welch), ("MSER-5", w_mser)], "job", path)


def _report_warmups(key: str, methods, unit: str, path: str) -> None:
    parts = [f"{name}={'NON conv.' if val is None else f'{val:.0f} {unit}'}"
             for name, val in methods]
    print(f"  {key:<14}: " + " | ".join(parts) + f"   -> {path}")


def _numeric_summary(conf: Config, per_replica_results: dict) -> None:
    """Tabella numerica riassuntiva (media + IC 95%) sulle repliche, a stdout e CSV."""
    labels = {
        "utilization": "Utilizzazione media pool",
        "service_time": "Tempo medio di servizio [s]",
        "wait_time": "Tempo medio di attesa [s]",
        "response_time": "Tempo medio di risposta [s]",
        "qlen": "Popolazione media in coda [job]",
        "syslen": "Popolazione media nel sistema [job]",
    }

    n_used = len(per_replica_results[conf.target_metric])
    rows = []
    print(f"\n=== RISULTATI AGGREGATI SU {n_used} REPLICHE (IC 95%) ===")
    print(f"{'Metrica':<34}{'Media':>14}{'+/- IC':>14}{'IC rel':>10}")
    for key, label in labels.items():
        mean, half = mean_ci(per_replica_results[key])
        rel = half / abs(mean) if mean else float("inf")
        rows.append([label, f"{mean:.4f}", f"{half:.4f}", f"{rel:.4f}"])
        print(f"{label:<34}{mean:>14.4f}{half:>14.4f}{rel:>9.2%}")

    tmean, thalf = mean_ci(per_replica_results[conf.target_metric])
    trel = thalf / abs(tmean) if tmean else float("inf")
    met = trel <= conf.target_rel_halfwidth
    esito = "target raggiunto" if met else f"cap max_replications={conf.max_replications} raggiunto"
    print(f"-> Metrica-obiettivo '{conf.target_metric}': IC rel {trel:.2%} "
          f"(soglia {conf.target_rel_halfwidth:.1%}) con {n_used} repliche [{esito}]")

    _write_csv(os.path.join(conf.trace_dir, "summary.csv"),
               ["metric", "mean", "ci_half_width_95", "ci_rel"], rows)
    print(f"Tabella riassuntiva salvata in {os.path.join(conf.trace_dir, 'summary.csv')}")
