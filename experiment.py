"""
Esperimento: studio del comportamento del sistema al variare del numero di serventi.

Carica i parametri di una singola simulazione da 'config.json' e la lista dei valori
di c da 'experiments.json' (file separato). Per ogni numero di serventi esegue N
repliche indipendenti, aggrega le metriche con media e IC 95% (t di Student) e produce
una tabella riassuntiva (stdout + CSV) e i grafici di ciascuna metrica in funzione di c.

Uso comune dei numeri casuali: per ogni valore di c si ripiantano gli stessi seed
(config.seed), cosi tutte le configurazioni partono dallo stesso stato del generatore
(confronto piu' equo e pienamente riproducibile).
"""
import os
import csv
from typing import Dict

from config.settings import Config, ExperimentConfig
from stats.estimate import mean_ci
from analysis import plots
from runner import run_simulation_suite

# (chiave metrica in sim.results(), etichetta tabella, ylabel grafico)
METRICS = [
    ("utilization", "Utilizzazione media pool", "Utilizzazione"),
    ("wait_time", "Tempo medio di attesa [s]", "Tempo di attesa [s]"),
    ("response_time", "Tempo medio di risposta [s]", "Tempo di risposta [s]"),
    ("service_time", "Tempo medio di servizio [s]", "Tempo di servizio [s]"),
    ("qlen", "Popolazione media in coda [job]", "Popolazione in coda [job]"),
    ("syslen", "Popolazione media nel sistema [job]", "Popolazione nel sistema [job]"),
]
METRIC_KEYS = [m[0] for m in METRICS]


def run_experiment(conf: Config, exp: ExperimentConfig) -> None:
    """
    Sweep sul numero di serventi. Per ogni c esegue l'intera pipeline di simulazione
    (tracce + transitorio + tabella) in una cartella dedicata, poi produce la tabella
    e i grafici comparativi al variare di c nella cartella radice dell'esperimento.
    """
    os.makedirs(exp.output_dir, exist_ok=True)
    print(f"Esperimento serventi: c in {exp.server_counts}, repliche adattive "
          f"(target IC rel '{conf.target_metric}' <= {conf.target_rel_halfwidth:.1%}, "
          f"seed base {conf.seed}).")

    # results[metric][c] = (mean, ci_half); reps[c] = numero di repliche adattive usate
    results: Dict[str, Dict[int, tuple]] = {k: {} for k in METRIC_KEYS}
    reps: Dict[int, int] = {}
    for c in exp.server_counts:
        print(f"\n########## SIMULAZIONE c = {c} ##########")
        conf.c = c
        conf.trace_dir = os.path.join(exp.output_dir, f"c{c}")
        bundle = run_simulation_suite(conf, quiet=True)
        reps[c] = len(bundle["per_replica_results"][conf.target_metric])
        for k in METRIC_KEYS:
            results[k][c] = mean_ci(bundle["per_replica_results"][k])

    print("\n########## CONFRONTO AL VARIARE DI c ##########")
    _print_table(exp, results, reps)
    _save_csv(exp, results, reps)
    _save_plots(exp, results)


def _print_table(exp: ExperimentConfig, results: Dict[str, Dict[int, tuple]],
                 reps: Dict[int, int]) -> None:
    """Tabella numerica riassuntiva (media +/- IC per ogni c)."""
    print("\n=== METRICHE AL VARIARE DEL NUMERO DI SERVENTI (media +/- IC 95%) ===")
    header = f"{'Metrica':<34}" + "".join(f"{'c=' + str(c):>20}" for c in exp.server_counts)
    print(header)
    print(f"{'(repliche adattive)':<34}" + "".join(f"{reps[c]:>20}" for c in exp.server_counts))
    for key, label, _ in METRICS:
        row = f"{label:<34}"
        for c in exp.server_counts:
            mean, half = results[key][c]
            row += f"{mean:>12.4f} +/-{half:>5.3f}"
        print(row)


def _save_csv(exp: ExperimentConfig, results: Dict[str, Dict[int, tuple]],
              reps: Dict[int, int]) -> None:
    path = os.path.join(exp.output_dir, "servers_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "servers", "mean", "ci_half_width_95", "n_replications"])
        for key, _, _ in METRICS:
            for c in exp.server_counts:
                mean, half = results[key][c]
                w.writerow([key, c, f"{mean:.6f}", f"{half:.6f}", reps[c]])
    print(f"\nTabella riassuntiva salvata in {path}")


def _save_plots(exp: ExperimentConfig, results: Dict[str, Dict[int, tuple]]) -> None:
    for key, label, ylabel in METRICS:
        means = [results[key][c][0] for c in exp.server_counts]
        cis = [results[key][c][1] for c in exp.server_counts]
        path = plots.plot_metric_vs_servers(
            key, f"{label} vs numero di serventi", ylabel,
            exp.server_counts, means, cis, exp.output_dir)
        print(f"  grafico -> {path}")


def main() -> None:
    """Carica le due configurazioni e avvia l'esperimento di sweep sui serventi."""
    try:
        conf = Config("config.json")
        exp = ExperimentConfig("experiments.json")
        run_experiment(conf, exp)
    except FileNotFoundError as fe:
        print(f"Errore critico: file di configurazione non trovato ({fe}).")
    except ValueError as ve:
        print(f"Errore di Validazione Parametri: {ve}")


if __name__ == "__main__":
    main()
