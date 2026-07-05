"""
Generazione dei grafici del transitorio (matplotlib/seaborn).

Ogni metrica produce una figura con la media di ensemble grezza, la media mobile
di Welch e la linea verticale dell'istante di fine transitorio stimato (se esiste).
Le figure sono salvate come PNG nella cartella di output.
"""
import os

import matplotlib
matplotlib.use("Agg")  # backend non interattivo: salva su file senza display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")


def plot_metric(key: str, title: str, xlabel: str, ylabel: str,
                x: np.ndarray, raw: np.ndarray, smoothed: np.ndarray,
                markers, out_dir: str, edge: int = 0,
                logx: bool = False) -> str:
    """
    Disegna e salva il grafico del transitorio di una singola metrica.

    Args:
        markers: sequenza di (etichetta, valore_x_o_None, colore); per ogni stimatore
            del fine-transitorio disegna una linea verticale (se il valore non e' None).
        edge: numero di punti finali della media mobile da NON tracciare. Vicino
            al bordo destro la finestra di Welch si restringe e il valore diventa
            rumoroso (fino a coincidere col dato grezzo): tracciarli darebbe un
            falso "spike" sul bordo. Convenzione: la curva di Welch vive su [1, m-w].
        logx: se True usa scala logaritmica sull'asse x. Usata per l'asse tempo
            (orizzonte fino a 1e7 s): rende visibile la convergenza precoce che su
            scala lineare resterebbe schiacciata contro l'asse y.

    Returns:
        Il path del file PNG salvato.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    if logx:
        ax.set_xscale("log")
    ax.plot(x, raw, color="tab:blue", alpha=0.35, linewidth=1.0,
            label="Media di ensemble")
    m = len(smoothed) - edge if 0 < edge < len(smoothed) else len(smoothed)
    ax.plot(x[:m], smoothed[:m], color="tab:red", linewidth=1.8,
            label="Media mobile di Welch")

    any_marker = False
    for label, value, color in (markers or []):
        if value is not None:
            ax.axvline(value, color=color, linestyle="--", linewidth=1.3,
                       label=f"{label} ~ {value:.0f}")
            any_marker = True
    if not any_marker:
        ax.text(0.98, 0.02, "Nessuna convergenza nell'orizzonte",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="darkred")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()

    path = os.path.join(out_dir, f"transient_{key}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_metric_vs_servers(key: str, title: str, ylabel: str,
                           server_counts, means, cis, out_dir: str) -> str:
    """
    Grafico di una metrica in funzione del numero di serventi, con barre d'errore
    pari alla semi-ampiezza dell'IC 95%.

    Returns:
        Il path del file PNG salvato.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(server_counts, means, yerr=cis, marker="o", capsize=4,
                linewidth=1.8, color="tab:blue", label="Media +/- IC 95%")
    ax.set_xlabel("Numero di serventi (c)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(server_counts))
    ax.legend(loc="best")
    fig.tight_layout()

    path = os.path.join(out_dir, f"servers_{key}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
