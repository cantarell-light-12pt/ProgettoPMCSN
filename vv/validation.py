"""
Validazione del modello: confronto tra i dati prodotti dal simulatore (tracce) e i
dati reali del dataset TravisTorrent, ristretti al progetto target (sonar-java).

Tutti i valori lato simulazione sono MISURATI dalle tracce (dati realizzati dal
motore DES), mai letti dai parametri di input: cosi' la validazione controlla il
comportamento emergente del simulatore, non riflette banalmente config.json.

Quantita' confrontate (reale vs simulato-misurato):
  1. Distribuzione degli esiti (passed/failed/errored/canceled): frazioni dal
     node_trace vs frazioni di tr_status nel dataset.
  2. Probabilita' di feedback (job instradato alla revisione umana, cioe' esito
     failed o errored): misurata dal node_trace vs frazione reale.
  3. Tempo di servizio: service_visit del node_trace vs
     tr_log_setup_time + tr_log_buildduration (il modello calibra il 'test' sulla
     durata dell'intero build). Complessivo e per esito 'passed'.
  4. Dimensione batch: PMF empirica dei batch realizzati (batch_trace) vs PMF
     empirica reale (job per build).
  5. Tasso di arrivo dei build: arrivi realizzati / tempo simulato (batch_trace)
     vs stima dai timestamp reali.

Output: tabelle a stdout + CSV e grafici PNG nella cartella di output.
"""
import csv
import os
from collections import Counter
from typing import List, Sequence

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

OUTCOMES = ["passed", "failed", "errored", "canceled"]
# Le probabilita' di routing in config.json sono ora allineate per etichetta al dataset
# reale (failed/errored non piu' scambiati): il confronto degli esiti e' quindi diretto,
# per etichetta identica.


# ----------------------------- Riferimento reale ----------------------------- #

def load_real_reference(config) -> dict:
    """Estrae dal dataset le statistiche reali del progetto target."""
    # Le colonne '*_time'/'*duration' vanno lette come stringa: l'inferenza di
    # schema puo' dedurle intere dalla finestra iniziale e poi fallire sui float
    # (es. '36.69'); il cast a Float64 (strict=False) avviene dopo.
    df = (pl.scan_csv(config.dataset_path, null_values=["NA"],
                      schema_overrides={"tr_log_setup_time": pl.String,
                                        "tr_log_buildduration": pl.String})
          .filter(pl.col("gh_project_name") == config.target_project))

    # Esiti
    oc = df.group_by("tr_status").len().collect()
    total = int(oc["len"].sum())
    real_outcome = {row["tr_status"]: row["len"] / total for row in oc.iter_rows(named=True)}

    # Tempo di servizio reale = tr_log_buildduration, che gia' include setup + test
    # (confermato). Sommare anche il setup lo conterebbe DUE volte. Si confronta con il
    # service_visit simulato (setup + test del modello).
    sdf = df.select([
        pl.col("tr_log_buildduration").cast(pl.Float64, strict=False).alias("build"),
        pl.col("tr_status"),
    ]).collect()
    build = sdf["build"]
    service_all = build.fill_null(0.0).to_numpy()
    service_all = service_all[~np.isnan(service_all)]
    mask_passed = (sdf["tr_status"] == "passed") & build.is_not_null()
    service_passed = build.filter(mask_passed).to_numpy()

    # Batch: job per build
    batch = df.group_by("tr_build_id").len(name="n").collect()["n"].to_numpy()
    vals, counts = np.unique(batch, return_counts=True)
    real_batch_pmf = {int(v): c / len(batch) for v, c in zip(vals, counts)}

    # Tasso d'arrivo degli arrivi NUOVI: il simulatore genera solo arrivi esogeni
    # (non i ritorni dal feedback), quindi il riferimento reale conta i soli build
    # nuovi (push, non-PR) - la stessa base su cui e' calibrato lambda_ext.
    b = (df.filter(pl.col("gh_is_pr") == False)
         .select(["tr_build_id", "gh_build_started_at"])
         .unique(subset="tr_build_id").drop_nulls().collect())
    ts = b["gh_build_started_at"].str.to_datetime(strict=False).drop_nulls().sort()
    span = (ts.max() - ts.min()).total_seconds()
    real_arrival_rate = len(ts) / span if span > 0 else float("nan")

    return {
        "outcome": real_outcome,
        "feedback_prob": real_outcome.get("failed", 0.0) + real_outcome.get("errored", 0.0),
        "service_all": service_all,
        "service_passed": service_passed,
        "batch_pmf": real_batch_pmf,
        "arrival_rate": real_arrival_rate,
        "n_jobs": total,
        "n_builds": len(batch),
    }


# ----------------------------- Dati simulati ----------------------------- #

def sim_reference(node_traces: Sequence[Sequence[tuple]],
                  batch_traces: Sequence[Sequence[tuple]],
                  max_time: float) -> dict:
    """
    Statistiche MISURATE dalle tracce della simulazione, aggregate su tutte le
    repliche (pooled). Nessun valore proviene dai parametri di config.
    """
    nt = [r for tr in node_traces for r in tr]   # (t, fate, wait_visit, service_visit)
    n = len(nt)
    fate_counts = Counter(r[1] for r in nt)
    sim_outcome = {k: fate_counts.get(k, 0) / n for k in OUTCOMES}
    service_all = np.array([r[3] for r in nt], dtype=float)
    service_passed = np.array([r[3] for r in nt if r[1] == "passed"], dtype=float)

    # Probabilita' di feedback realizzata: frazione di completamenti instradati
    # alla revisione umana (fate in {failed, errored}).
    feedback_prob = sim_outcome.get("failed", 0.0) + sim_outcome.get("errored", 0.0)

    # PMF empirica dei batch realizzati e tasso di arrivo realizzato.
    batch_sizes = np.array([b for tr in batch_traces for (_, b) in tr], dtype=int)
    vals, counts = np.unique(batch_sizes, return_counts=True)
    sim_batch_pmf = {int(v): c / len(batch_sizes) for v, c in zip(vals, counts)}
    n_reps = len(batch_traces)
    sim_arrival_rate = len(batch_sizes) / (n_reps * max_time) if n_reps else float("nan")

    return {"outcome": sim_outcome, "feedback_prob": feedback_prob,
            "service_all": service_all, "service_passed": service_passed,
            "batch_pmf": sim_batch_pmf, "arrival_rate": sim_arrival_rate,
            "n": n, "n_arrivals": int(len(batch_sizes))}


# ----------------------------- Confronti ----------------------------- #

def _ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def _dist_row(label: str, real: np.ndarray, sim: np.ndarray) -> list:
    return [label,
            f"{np.mean(real):.2f}", f"{np.mean(sim):.2f}",
            f"{np.median(real):.2f}", f"{np.median(sim):.2f}",
            f"{np.quantile(real, 0.9):.2f}", f"{np.quantile(sim, 0.9):.2f}"]


def run(config, node_traces, batch_traces) -> None:
    """
    Esegue la validazione completa e produce tabelle e grafici. Tutti i valori
    'sim' sono misurati dalle tracce (node_trace, batch_trace), non dai parametri.
    """
    print("\n=== VALIDAZIONE (dati misurati dal simulatore vs dataset reale) ===")
    print(f"  Progetto target: {config.target_project}")

    real = load_real_reference(config)
    sim = sim_reference(node_traces, batch_traces, config.max_time)
    rows_csv: List[list] = []

    # --- 1. Esiti: frazioni realizzate vs reali, per etichetta --- #
    print("\n[1] Distribuzione esiti (frazioni misurate dal node_trace)")
    print(f"  {'esito':<10}{'reale':>10}{'sim':>10}{'|diff|':>10}")
    for k in OUTCOMES:
        rf, sf = real["outcome"].get(k, 0.0), sim["outcome"].get(k, 0.0)
        print(f"  {k:<10}{rf:>10.4f}{sf:>10.4f}{abs(rf - sf):>10.4f}")
        rows_csv.append(["outcome", k, f"{rf:.4f}", f"{sf:.4f}", f"{abs(rf - sf):.4f}"])

    # --- 2. Probabilita' di feedback (revisione umana) --- #
    print("\n[2] Probabilita' di feedback (esito failed/errored -> revisione umana)")
    rf, sf = real["feedback_prob"], sim["feedback_prob"]
    print(f"  reale = {rf:.4f}   sim (misurata) = {sf:.4f}   |diff| = {abs(rf - sf):.4f}")
    rows_csv.append(["feedback_prob", "failed+errored", f"{rf:.4f}", f"{sf:.4f}", f"{abs(rf - sf):.4f}"])

    # --- 3. Tempo di servizio --- #
    print("\n[3] Tempo di servizio [s]  (media / mediana / q90)")
    print(f"  {'insieme':<12}{'media R':>10}{'media S':>10}{'med R':>10}{'med S':>10}{'q90 R':>10}{'q90 S':>10}")
    for label, rk, sk in [("complessivo", "service_all", "service_all"),
                          ("passed", "service_passed", "service_passed")]:
        row = _dist_row(label, real[rk], sim[sk])
        print(f"  {row[0]:<12}{row[1]:>10}{row[2]:>10}{row[3]:>10}{row[4]:>10}{row[5]:>10}{row[6]:>10}")
        rows_csv.append(["service_" + label] + row[1:])

    # --- 4. Batch: PMF realizzata vs reale --- #
    print("\n[4] Dimensione batch (PMF misurata dai batch realizzati)")
    keys = sorted(set(sim["batch_pmf"]) | set(real["batch_pmf"]))
    print(f"  {'k':>4}{'reale':>10}{'sim':>10}{'|diff|':>10}")
    for k in keys:
        rf, sf = real["batch_pmf"].get(k, 0.0), sim["batch_pmf"].get(k, 0.0)
        print(f"  {k:>4}{rf:>10.4f}{sf:>10.4f}{abs(rf - sf):>10.4f}")
        rows_csv.append(["batch", str(k), f"{rf:.4f}", f"{sf:.4f}", f"{abs(rf - sf):.4f}"])

    # --- 5. Tasso d'arrivo: realizzato vs reale (soli arrivi nuovi) --- #
    print("\n[5] Tasso di arrivo dei build NUOVI [build/s]  (reale = push/non-PR)")
    rr, sr = real["arrival_rate"], sim["arrival_rate"]
    rel = abs(sr - rr) / rr if rr else float("nan")
    print(f"  reale = {rr:.3e}   sim (misurato, {sim['n_arrivals']} arrivi) = {sr:.3e}   err.rel. = {rel:.1%}")
    rows_csv.append(["arrival_rate", "build/s", f"{rr:.6e}", f"{sr:.6e}", f"{rel:.4f}"])

    _save_plots(config, real, sim)

    path = os.path.join(config.trace_dir, "validation_report.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sezione", "voce", "reale", "simulato", "diff"])
        w.writerows(rows_csv)
    print(f"\n  Report salvato in {path}")


def _save_plots(config, real: dict, sim: dict) -> None:
    """Grafici di validazione: esiti, ECDF tempo di servizio, PMF batch."""
    # Esiti reale vs sim (per etichetta)
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.arange(len(OUTCOMES))
    ax.bar(idx - 0.2, [real["outcome"].get(k, 0.0) for k in OUTCOMES], 0.4,
           label="Reale", color="tab:blue")
    ax.bar(idx + 0.2, [sim["outcome"].get(k, 0.0) for k in OUTCOMES], 0.4,
           label="Simulato", color="tab:orange")
    ax.set_xticks(idx); ax.set_xticklabels(OUTCOMES)
    ax.set_ylabel("Frazione"); ax.set_title("Validazione - Distribuzione esiti")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(config.trace_dir, "validation_outcomes.png"), dpi=120)
    plt.close(fig)

    # ECDF tempo di servizio (passed)
    fig, ax = plt.subplots(figsize=(8, 5))
    for arr, name, col in [(real["service_passed"], "Reale (passed)", "tab:blue"),
                           (sim["service_passed"], "Simulato (passed)", "tab:orange")]:
        if len(arr):
            xs, ys = _ecdf(arr)
            ax.plot(xs, ys, label=name, color=col, linewidth=1.8)
    ax.set_xlabel("Tempo di servizio [s]"); ax.set_ylabel("ECDF")
    ax.set_title("Validazione - Tempo di servizio (passed)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(config.trace_dir, "validation_service_ecdf.png"), dpi=120)
    plt.close(fig)

    # PMF batch: reale vs realizzata dalla simulazione
    keys = sorted(set(sim["batch_pmf"]) | set(real["batch_pmf"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.arange(len(keys))
    ax.bar(idx - 0.2, [real["batch_pmf"].get(k, 0.0) for k in keys], 0.4,
           label="Reale", color="tab:blue")
    ax.bar(idx + 0.2, [sim["batch_pmf"].get(k, 0.0) for k in keys], 0.4,
           label="Simulato (misurato)", color="tab:orange")
    ax.set_xticks(idx); ax.set_xticklabels(keys)
    ax.set_xlabel("Job per build"); ax.set_ylabel("Probabilita'")
    ax.set_title("Validazione - PMF dimensione batch")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(config.trace_dir, "validation_batch_pmf.png"), dpi=120)
    plt.close(fig)
