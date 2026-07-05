"""
Esperimento: studio del comportamento del sistema al variare del numero di serventi.

Carica i parametri da 'config.json' e la lista dei valori di c da 'experiments.json'
(file separato). Per OGNI numero di serventi esegue l'intera analisi:
  - TRANSITORIO col metodo delle repliche indipendenti (5 seed) -> 3 grafici di media
    cumulata (utilizzazione, servizio passed, risposta) nella cartella del c;
  - STEADY-STATE col metodo dei BATCH MEANS (single long run).
Infine produce una tabella riassuntiva e i grafici di ciascuna metrica (media + IC 95%)
in funzione di c.
"""
import os
import csv
from typing import Dict

from config.settings import Config, ExperimentConfig
from analysis import plots, batchmeans, transient

# (chiave metrica nei batch means, etichetta tabella, ylabel grafico)
METRICS = [
    ("utilization", "Utilizzazione", "Utilizzazione"),
    ("wait_time", "Tempo medio di attesa [s]", "Tempo di attesa [s]"),
    ("response_time", "Tempo medio di risposta [s]", "Tempo di risposta [s]"),
    ("service_passed", "Tempo di servizio passed [s]", "Tempo di servizio [s]"),
    ("qlen", "Popolazione media in coda [job]", "Popolazione in coda [job]"),
    ("syslen", "Popolazione media nel sistema [job]", "Popolazione nel sistema [job]"),
]
METRIC_KEYS = [m[0] for m in METRICS]

# Tipi di grafico del transitorio per le immagini composite (chiave, titolo, xlabel, ylabel).
TRANSIENT_GRID = [
    ("utilization", "Transitorio - Utilizzazione (media cumulata)",
     "Tempo simulato [s]", "Utilizzazione"),
    ("service_passed", "Transitorio - Tempo di servizio passed (media cumulata)",
     "Indice job passed", "Tempo di servizio [s]"),
    ("response", "Transitorio - Tempo di risposta (media cumulata)",
     "Indice job (uscita)", "Tempo di risposta [s]"),
]


def run_experiment(conf: Config, exp: ExperimentConfig) -> None:
    """
    Sweep sul numero di serventi. Per ogni c esegue transitorio (grafici) + batch means
    (stima steady-state) in una cartella dedicata; poi produce la tabella e i grafici
    comparativi al variare di c.
    """
    os.makedirs(exp.output_dir, exist_ok=True)
    print(f"Esperimento serventi: c in {exp.server_counts}. Per ogni c: transitorio "
          f"({len(conf.seeds)} seed) + batch means ({conf.num_batches} batch, seed {conf.seed}).")

    # results[metric][c] = (mean, ci_half); dati per le immagini composite tra c
    results: Dict[str, Dict[int, tuple]] = {k: {} for k in METRIC_KEYS}
    transient_data: Dict[int, dict] = {}   # c -> {metrica: curves}
    batch_data: Dict[int, dict] = {}       # c -> {metrica: (label, batch_values, mean, half)}
    for c in exp.server_counts:
        print(f"\n########## SIMULAZIONE c = {c} ##########")
        conf.c = c
        conf.trace_dir = os.path.join(exp.output_dir, f"c{c}")
        # 1. Transitorio (repliche indipendenti) -> 3 grafici nella cartella di c
        transient_data[c] = transient.run_transient(conf)
        # 2. Steady-state (batch means)
        bundle = batchmeans.run_batch_means(conf, quiet=True)
        batch_data[c] = {key: (label, b, mean, half)
                         for key, label, b, mean, half in bundle["batches_plot"]}
        for k in METRIC_KEYS:
            _, mean, half = bundle["estimates"][k]
            results[k][c] = (mean, half)
        print(f"  c={c}: transitorio (grafici) + batch means completati.")

    print("\n########## CONFRONTO AL VARIARE DI c ##########")
    _print_table(exp, results)
    _save_csv(exp, results)
    _save_plots(exp, results)
    _save_grid_plots(exp, transient_data, batch_data)


def _print_table(exp: ExperimentConfig, results: Dict[str, Dict[int, tuple]]) -> None:
    """Tabella numerica riassuntiva (media +/- IC per ogni c)."""
    print("\n=== METRICHE AL VARIARE DEL NUMERO DI SERVENTI (batch means, media +/- IC 95%) ===")
    header = f"{'Metrica':<34}" + "".join(f"{'c=' + str(c):>20}" for c in exp.server_counts)
    print(header)
    for key, label, _ in METRICS:
        row = f"{label:<34}"
        for c in exp.server_counts:
            mean, half = results[key][c]
            row += f"{mean:>12.4f} +/-{half:>5.3f}"
        print(row)


def _save_csv(exp: ExperimentConfig, results: Dict[str, Dict[int, tuple]]) -> None:
    path = os.path.join(exp.output_dir, "servers_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "servers", "mean", "ci_half_width_95"])
        for key, _, _ in METRICS:
            for c in exp.server_counts:
                mean, half = results[key][c]
                w.writerow([key, c, f"{mean:.6f}", f"{half:.6f}"])
    print(f"\nTabella riassuntiva salvata in {path}")


def _save_plots(exp: ExperimentConfig, results: Dict[str, Dict[int, tuple]]) -> None:
    for key, label, ylabel in METRICS:
        means = [results[key][c][0] for c in exp.server_counts]
        cis = [results[key][c][1] for c in exp.server_counts]
        path = plots.plot_metric_vs_servers(
            key, f"{label} vs numero di serventi", ylabel,
            exp.server_counts, means, cis, exp.output_dir)
        print(f"  grafico -> {path}")


def _save_grid_plots(exp: ExperimentConfig, transient_data: Dict[int, dict],
                     batch_data: Dict[int, dict]) -> None:
    """Immagini composite: 4 riquadri (uno per c) dello stesso tipo di grafico."""
    cs = exp.server_counts
    # Transitorio: un'immagine composita per tipo (utilizzazione, servizio passed, risposta)
    for key, title, xlabel, ylabel in TRANSIENT_GRID:
        panels = [(c, transient_data[c][key]) for c in cs]
        path = plots.plot_transient_grid(key, title, xlabel, ylabel, panels, exp.output_dir)
        print(f"  griglia transitorio -> {path}")
    # Batch means: un'immagine composita per metrica (diagramma diagnostico)
    for key, label, ylabel in METRICS:
        panels = [(c, batch_data[c][key][1], batch_data[c][key][2], batch_data[c][key][3])
                  for c in cs]
        path = plots.plot_batch_means_grid(key, f"Batch means - {label}", ylabel,
                                           panels, exp.output_dir)
        print(f"  griglia batch means -> {path}")


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
