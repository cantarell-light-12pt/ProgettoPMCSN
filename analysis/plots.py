"""
Generazione dei grafici (matplotlib/seaborn), salvati come PNG.
"""
import os

import matplotlib
matplotlib.use("Agg")  # backend non interattivo: salva su file senza display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


def plot_batch_means(key: str, title: str, ylabel: str,
                     batch_values, mean: float, half: float, out_dir: str) -> str:
    """
    Diagramma diagnostico dei batch means: i valori dei singoli batch (b_1..b_k) in
    funzione dell'indice di batch, con la media complessiva e la banda +/- IC 95%.
    Serve a verificare visivamente che i batch siano stazionari (nessun trend residuo
    -> warm-up eliminato) e a mostrare la variabilita' alla base dell'intervallo.

    Returns:
        Il path del file PNG salvato.
    """
    b = np.asarray(batch_values, dtype=float)
    x = np.arange(1, len(b) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, b, marker="o", markersize=3, linewidth=1.0, color="tab:blue",
            label="Media di batch")
    ax.axhline(mean, color="tab:red", linewidth=1.8, label=f"Media = {mean:.4g}")
    if half == half and half > 0:   # not NaN
        ax.axhspan(mean - half, mean + half, color="tab:red", alpha=0.15,
                   label=f"IC 95% (+/- {half:.3g})")
    ax.set_title(title)
    ax.set_xlabel("Indice batch")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, f"batchmeans_{key}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _grid_axes(n: int):
    """Crea una griglia di sottografici (2 colonne) per n pannelli; ritorna (fig, axes_flat)."""
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows),
                             squeeze=False)
    return fig, axes.ravel()


def plot_transient_grid(key: str, title: str, xlabel: str, ylabel: str,
                        panels, out_dir: str) -> str:
    """
    Immagine composita: un riquadro per esperimento (valore di c), ciascuno con le curve
    di media cumulata delle repliche di quel c. panels = lista di (c, curves), con
    curves = lista di (etichetta_seed, x, y).

    Returns:
        Il path del file PNG salvato.
    """
    fig, axes = _grid_axes(len(panels))
    for ax, (c, curves) in zip(axes, panels):
        for label, x, y in curves:
            ax.plot(x, y, linewidth=1.0, alpha=0.85, label=label)
        ax.set_title(f"c = {c}", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, loc="best")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"grid_transient_{key}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_batch_means_grid(key: str, title: str, ylabel: str,
                          panels, out_dir: str) -> str:
    """
    Immagine composita: un riquadro per esperimento (valore di c) col diagramma
    diagnostico dei batch di quel c. panels = lista di (c, batch_values, mean, half).

    Returns:
        Il path del file PNG salvato.
    """
    fig, axes = _grid_axes(len(panels))
    for ax, (c, b, mean, half) in zip(axes, panels):
        bb = np.asarray(b, dtype=float)
        x = np.arange(1, len(bb) + 1)
        ax.plot(x, bb, linewidth=0.6, color="tab:blue")
        ax.axhline(mean, color="tab:red", linewidth=1.5)
        if half == half and half > 0:  # not NaN
            ax.axhspan(mean - half, mean + half, color="tab:red", alpha=0.15)
        ax.set_title(f"c = {c}  (media {mean:.4g})", fontweight="bold")
        ax.set_xlabel("Indice batch")
        ax.set_ylabel(ylabel)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"grid_batchmeans_{key}.png")
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
