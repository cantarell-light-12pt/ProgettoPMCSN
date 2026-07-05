"""
Generazione dei grafici (matplotlib/seaborn), salvati come PNG.
"""
import os

import matplotlib
matplotlib.use("Agg")  # backend non interattivo: salva su file senza display
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_transient_replications(key: str, title: str, xlabel: str, ylabel: str,
                                curves, out_dir: str) -> str:
    """
    Grafico del transitorio col metodo delle repliche indipendenti: le medie cumulate
    delle repliche (una curva per seed) sovrapposte sullo stesso grafico.

    Args:
        curves: sequenza di (etichetta, x, y) - una per replica (seed).

    Returns:
        Il path del file PNG salvato.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, x, y in curves:
        ax.plot(x, y, linewidth=1.3, alpha=0.85, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9, title="Replica (seed)")
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
