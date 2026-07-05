"""
Helper riusabili di simulazione: esecuzione di una singola run e scrittura tracce.
Usati da analysis/transient.py (repliche indipendenti, orizzonte finito) e
analysis/batchmeans.py (single long run, steady-state).
"""
import os
import csv
from typing import Sequence

from config.settings import Config
from core.simulator import TravisCISimulator
from rng import rngs


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence]) -> None:
    """Scrive un CSV usando il modulo standard (nessuna dipendenza esterna nel core)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def simulate_once(conf: Config, seed: int) -> TravisCISimulator:
    """
    Pianta il seed indicato ed esegue una singola run (conf.max_time gia' impostato
    dal chiamante = finite_horizon per il transitorio, run_length per i batch means).
    Ogni run e' una replica indipendente perche' PlantSeeds re-inizializza gli stream.
    """
    rngs.PlantSeeds(seed)
    sim = TravisCISimulator(conf)
    sim.run()
    return sim


def write_traces(sim: TravisCISimulator, out_dir: str) -> None:
    """Scrive le 4 tracce del sim in sottocartelle traces/{state,jobs,node,arrivals}/."""
    dirs = {t: os.path.join(out_dir, "traces", t)
            for t in ("state", "jobs", "node", "arrivals")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    _write_csv(os.path.join(dirs["state"], "state.csv"),
               ["time", "utilization", "queue_len", "servers_busy", "feedback_len"],
               sim.state_trace)
    _write_csv(os.path.join(dirs["jobs"], "jobs.csv"),
               ["job_id", "exit_time", "wait_time", "service_time", "response_time"],
               sim.job_trace)
    _write_csv(os.path.join(dirs["node"], "node.csv"),
               ["completion_time", "fate", "wait_visit", "service_visit"],
               sim.node_trace)
    _write_csv(os.path.join(dirs["arrivals"], "arrivals.csv"),
               ["arrival_time", "batch_size"],
               sim.batch_trace)
