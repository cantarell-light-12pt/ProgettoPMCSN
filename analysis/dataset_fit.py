"""
Fit esponenziale dei tempi di inter-arrivo, calcolato in DUE modi alternativi.

Nel dataset ogni riga e' un JOB, e tutti i job della stessa build condividono lo
stesso 'gh_build_started_at': la serie dei diff job-per-job e' quindi dominata da
zeri (i job di una stessa build matrix arrivano insieme). Come si trattano quegli
zeri non e' un dettaglio tecnico, e' una scelta di modello. Qui le due alternative
sono rese esplicite e confrontate:

  A) UN ARRIVO PER BUILD - si prende un solo timestamp per build (il minimo) e si
     tengono i soli arrivi ESOGENI, cioe' la prima build di un commit o una build
     che segue una build "buona". Le build che ripartono dopo un fallimento sono
     traffico del feedback loop, che il simulatore genera gia' internamente:
     contarle sarebbe un doppio conteggio. E' la base su cui e' calibrato
     'traffic.lambda_ext' in config.json.

  B) JOB CONSECUTIVI, ZERI FORZATI A 1 s - si mantiene la serie job-per-job e ogni
     differenza nulla viene portata a 1 s invece di essere scartata. Nessun campione
     va perso, ma il processo risultante e' un flusso di arrivi singoli molto
     ravvicinati: e' il modello che si otterrebbe RINUNCIANDO agli arrivi a gruppi.

Il confronto e' diagnostico: la variante B non calibra nulla, serve a mostrare con
i numeri perche' il modello adotta arrivi a gruppi M^[X] e non arrivi singoli.

Esecuzione:
    .venv/bin/python -m analysis.dataset_fit
"""
import csv
import os

import numpy as np
import polars as pl
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")  # backend non interattivo: salva su file senza display
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# ----------------------------- Lettura del dataset ----------------------------- #

def carica_job(config) -> pl.DataFrame:
    """
    Job del progetto target con il timestamp di avvio della build in secondi.

    Le colonne '*_time'/'*duration' vanno lette come stringa (stesso motivo spiegato
    in vv/validation.py: l'inferenza di schema puo' dedurle intere dalla finestra
    iniziale e poi fallire sui float); qui non servono, ma l'override evita l'errore
    in lettura. Il drop_nulls e' limitato alle colonne effettivamente usate: un
    drop_nulls() globale scarterebbe job validi per colonne non pertinenti.
    """
    return (
        pl.scan_csv(config.dataset_path, null_values=["NA", ""],
                    schema_overrides={"tr_log_setup_time": pl.String,
                                      "tr_log_buildduration": pl.String})
        .filter(pl.col("gh_project_name") == config.target_project)
        .select(["tr_build_id", "git_trigger_commit", "gh_build_started_at", "tr_status"])
        .with_columns(
            (pl.col("gh_build_started_at").str.to_datetime(strict=False)
             .dt.timestamp("ms") / 1000.0).alias("ts")
        )
        .drop_nulls(subset=["tr_build_id", "git_trigger_commit", "ts"])
    ).collect()


def arrivi_esogeni(df_job: pl.DataFrame) -> tuple[np.ndarray, int]:
    """
    Timestamp ordinati delle sole build che sono arrivi NUOVI (traffico esogeno).

    Criterio: una build e' un nuovo arrivo se e' la prima del suo git_trigger_commit,
    oppure se la build precedente sullo stesso commit era "buona" (nessun job failed
    o errored). Altrimenti e' un riavvio, cioe' traffico del feedback loop.

    Funzione pubblica perche' riusata da vv/validation.py: il riferimento reale del
    tasso d'arrivo deve poggiare sulla STESSA popolazione della calibrazione.

    Returns:
        (timestamp degli arrivi esogeni ordinati, numero totale di build).
    """
    df_build = (
        df_job.group_by(["tr_build_id", "git_trigger_commit"])
        .agg([
            pl.col("ts").min().alias("build_start"),
            ((pl.col("tr_status") == "failed") | (pl.col("tr_status") == "errored"))
            .sum().alias("bad_jobs"),
        ])
        .with_columns((pl.col("bad_jobs") > 0).alias("is_bad_build"))
        # tr_build_id come ultimo criterio: il sort di polars non e' stabile e senza
        # spareggio due build dello stesso commit partite nello stesso secondo si
        # ordinerebbero in modo arbitrario, cambiando quale delle due viene marcata
        # come nuovo arrivo (risultato diverso a ogni esecuzione).
        .sort(["git_trigger_commit", "build_start", "tr_build_id"])
    )

    esogene = (
        df_build
        .with_columns(pl.col("is_bad_build").shift(1)
                      .over("git_trigger_commit").alias("prev_build_bad"))
        .filter(pl.col("prev_build_bad").is_null() | (pl.col("prev_build_bad") == False))
        .sort("build_start")
    )
    return esogene["build_start"].to_numpy(), df_build.height


# ----------------------------- Fit ----------------------------- #

def _fit_esponenziale(x: np.ndarray, nome: str) -> dict:
    """
    Fit di una esponenziale con origine bloccata in 0 e test di Kolmogorov-Smirnov.

    Con floc=0 la stima di massima verosimiglianza dello 'scale' coincide con la
    media campionaria, quindi lambda = 1/media.
    """
    x = np.asarray(x, dtype=float)
    loc, scale = stats.expon.fit(x, floc=0)
    d, p = stats.kstest(x, "expon", args=(loc, scale))
    media = float(x.mean())
    return {
        "nome": nome,
        "n": int(len(x)),
        "media": media,
        "mediana": float(np.median(x)),
        "cv": float(x.std() / media),
        "lambda": 1.0 / scale,
        "scale": scale,
        "D": d,
        "p": p,
        "campioni": x,
    }


# ----------------------------- Grafico ----------------------------- #

def _plot_varianti(fit_a: dict, fit_b: dict, out_dir: str) -> str:
    """Istogramma delle due serie con sovrapposta la densita' esponenziale stimata."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pannello A: troncato al 95o percentile, altrimenti la coda (fino a ~6 giorni)
    # comprime tutta la massa di probabilita' contro l'asse.
    xa = fit_a["campioni"]
    xa_plot = xa[xa < np.percentile(xa, 95)]
    axes[0].hist(xa_plot, bins=50, density=True, alpha=0.6,
                 color="#2980b9", edgecolor="white", label="Dati (fino al 95° perc.)")
    grid_a = np.linspace(0, xa_plot.max(), 300)
    axes[0].plot(grid_a, stats.expon.pdf(grid_a, 0.0, fit_a["scale"]), "k--", lw=2.2,
                 label=f"Exp($\\lambda$={fit_a['lambda']:.2e})\nKS D={fit_a['D']:.4f}")
    axes[0].set_title("A - Un arrivo per build (esogeni)")
    axes[0].set_xlabel("Inter-arrivo [s]")
    axes[0].set_ylabel("Densita'")
    axes[0].legend()

    # Pannello B: qui l'istogramma non funziona (l'86% della massa sta nel singolo
    # valore 1 s). Si confrontano invece ECDF empirica e CDF esponenziale stimata,
    # su ascisse logaritmiche: il salto verticale iniziale e' la distanza KS.
    xb = np.sort(fit_b["campioni"])
    ecdf = np.arange(1, len(xb) + 1) / len(xb)
    axes[1].step(xb, ecdf, where="post", color="#c0392b", lw=2.0, label="ECDF empirica")
    grid_b = np.logspace(0, np.log10(xb.max()), 400)
    axes[1].plot(grid_b, stats.expon.cdf(grid_b, 0.0, fit_b["scale"]), "k--", lw=2.2,
                 label=f"CDF Exp($\\lambda$={fit_b['lambda']:.2e})")
    # Segmento verticale sul punto di massimo scostamento: e' la statistica D.
    x_max = xb[np.argmax(np.abs(ecdf - stats.expon.cdf(xb, 0.0, fit_b["scale"])))]
    axes[1].vlines(x_max, stats.expon.cdf(x_max, 0.0, fit_b["scale"]),
                   np.searchsorted(xb, x_max, side="right") / len(xb),
                   color="#2c3e50", lw=2.5,
                   label=f"KS D={fit_b['D']:.4f}")
    axes[1].set_xscale("log")
    axes[1].set_title("B - Job consecutivi, diff nulle forzate a 1 s")
    axes[1].set_xlabel("Inter-arrivo [s] (scala log)")
    axes[1].set_ylabel("Probabilita' cumulata")
    axes[1].legend(loc="lower right")

    fig.suptitle("Inter-arrivi: due modi di derivarli dal dataset", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "interarrivi_varianti.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ----------------------------- Driver ----------------------------- #

def run(config) -> dict:
    """Calcola, stampa e salva il confronto fra le due varianti di inter-arrivo."""
    os.makedirs(config.trace_dir, exist_ok=True)

    df_job = carica_job(config)
    ts_esogeni, n_build = arrivi_esogeni(df_job)

    print("=" * 78)
    print("FIT DEGLI INTER-ARRIVI - CONFRONTO FRA DUE DERIVAZIONI")
    print("=" * 78)
    print(f"Progetto: {config.target_project}   ({config.dataset_path})")
    print(f"Job: {df_job.height}   Build: {n_build}   "
          f"Build esogene (nuovi arrivi): {len(ts_esogeni)}")

    # --- A) Un arrivo per build, soli arrivi esogeni --- #
    d_a = np.diff(ts_esogeni)
    zeri_a = int((d_a == 0).sum())
    fit_a = _fit_esponenziale(d_a[d_a > 0], "A: build esogene (zeri scartati)")

    # --- B) Job consecutivi, differenze nulle forzate a 1 s --- #
    d_b = np.diff(np.sort(df_job["ts"].to_numpy()))
    zeri_b = int((d_b == 0).sum())
    fit_b = _fit_esponenziale(np.where(d_b == 0, 1.0, d_b),
                              "B: job consecutivi (zeri -> 1 s)")

    print(f"\nDifferenze nulle - A: {zeri_a}/{len(d_a)} (build distinte partite nello "
          f"stesso secondo, scartate)")
    print(f"                   B: {zeri_b}/{len(d_b)} (job della stessa build, "
          f"portate a 1 s)")

    header = f"\n{'Variante':<34}{'n':>7}{'media [s]':>12}{'mediana':>10}" \
             f"{'CV':>7}{'lambda [1/s]':>15}{'KS D':>9}"
    print(header)
    print("-" * len(header.strip("\n")))
    for f in (fit_a, fit_b):
        print(f"{f['nome']:<34}{f['n']:>7}{f['media']:>12.2f}{f['mediana']:>10.1f}"
              f"{f['cv']:>7.2f}{f['lambda']:>15.3e}{f['D']:>9.4f}")

    print(f"\nlambda_ext adottato in config.json: {config.lambda_ext:.3e} "
          f"(dalla variante A)")
    print("La variante B non calibra il modello: il fit e' nettamente peggiore "
          "(CV e KS D molto\npiu' alti) perche' i job non arrivano uno alla volta ma "
          "a gruppi. E' la giustificazione\nquantitativa della scelta M^[X].")

    # --- Salvataggi --- #
    csv_path = os.path.join(config.trace_dir, "dataset_fit_interarrivi.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variante", "n", "media_s", "mediana_s", "cv", "lambda_1_su_s",
                    "ks_D", "ks_p"])
        for f in (fit_a, fit_b):
            w.writerow([f["nome"], f["n"], f"{f['media']:.4f}", f"{f['mediana']:.4f}",
                        f"{f['cv']:.4f}", f"{f['lambda']:.6e}", f"{f['D']:.6f}",
                        f"{f['p']:.6e}"])

    png_path = _plot_varianti(fit_a, fit_b, config.trace_dir)
    print(f"\nTabella salvata in: {csv_path}")
    print(f"Grafico salvato in: {png_path}")

    return {"A": fit_a, "B": fit_b, "n_build": n_build,
            "n_esogene": len(ts_esogeni), "n_job": df_job.height}


if __name__ == "__main__":
    from config.settings import Config

    run(Config("config.json"))
