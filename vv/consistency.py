"""
Verifica strutturale del modello computazionale.

Mentre vv/verification.py controlla la coerenza interna delle tracce prodotte dalla
configurazione nominale (validita' dei valori + legge di Little), questo modulo
verifica gli strati su cui quelle tracce poggiano, dal basso verso l'alto:

  [A] Generatore pseudo-casuale (rng/rngs.py). Test di consistenza standard di
      Leemis & Park sul Lehmer (seed noto -> valore noto dopo 10 000 estrazioni),
      disgiunzione dei seed dei 256 stream dopo PlantSeeds e incorrelazione degli
      11 stream effettivamente assegnati alle sorgenti del modello.

  [B] Generatori di variate (rng/rvgs.py). Per ciascuna sorgente stocastica del
      modello si confrontano i momenti empirici con quelli teorici e si esegue un
      test di adattamento: chi-quadro per le PMF discrete e per il meccanismo a
      rischi competitivi che sceglie l'esito, Kolmogorov-Smirnov per l'esponenziale
      degli inter-arrivi e per le quattro lognormali.

  [C] Caso degenere con soluzione analitica ESATTA. Ponendo p_passed = 1 (feedback
      loop disattivato) e c = 1, la rete si riduce a una coda M^[X]/G/1 con arrivi
      a batch, per la quale il tempo medio di attesa e' noto in forma chiusa
      (Pollaczek-Khinchine generalizzata agli arrivi a gruppi). Il confronto e'
      ripetuto su cinque livelli di utilizzazione, cosi da sollecitare il motore
      anche in regimi di traffico lontani da quello nominale.

  [D] Conservazione del flusso e fattore di visita sulla configurazione nominale:
      ogni job generato deve uscire o risultare ancora nel sistema, e il numero
      medio di passaggi dal nodo di servizio deve coincidere con 1/(1-q), dove q
      e' la probabilita' di rientro dal feedback loop.

Nessun controllo di questo modulo usa il dataset reale: e' verifica, non validazione.
"""
import csv
import json
import math
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from config.settings import Config
from core import simulator as sim_mod
from rng import rngs, rvgs
from runner import simulate_once, _write_csv
from stats.estimate import mean_ci


# --------------------------------------------------------------------------- #
# Momenti teorici delle distribuzioni di input
# --------------------------------------------------------------------------- #

def _pmf_moments(values: Sequence[float], probs: Sequence[float]) -> Tuple[float, float]:
    """Momento primo e secondo di una PMF discreta."""
    m1 = sum(v * p for v, p in zip(values, probs))
    m2 = sum(v * v * p for v, p in zip(values, probs))
    return m1, m2


def _lognormal_moments(a: float, b: float) -> Tuple[float, float]:
    """Momento primo e secondo di una Lognormale(a, b) (a, b della normale sottostante)."""
    return math.exp(a + b * b / 2.0), math.exp(2.0 * a + 2.0 * b * b)


def service_moments(conf: Config) -> Tuple[float, float]:
    """
    Momenti del tempo di servizio S = setup + test nel caso degenere (tutti i job
    'passed'), con setup e test indipendenti.
    """
    s1, s2 = _pmf_moments(conf.pmf_setup_values, conf.pmf_setup_probs)
    t1, t2 = _lognormal_moments(conf.mu_test, conf.sigma_test)
    return s1 + t1, s2 + 2.0 * s1 * t1 + t2


# --------------------------------------------------------------------------- #
# [A] Generatore pseudo-casuale
# --------------------------------------------------------------------------- #

def check_rng(cfg: dict) -> List[dict]:
    """
    Controlli sul Lehmer multi-stream. Restituisce una lista di record
    {nome, misurato, atteso, esito}.
    """
    n = cfg["n_samples"]
    streams = cfg["streams"]
    seed = cfg["seed"]
    # Criterio statistico anziche' soglia arbitraria: sotto l'ipotesi di
    # indipendenza il coefficiente di correlazione campionario ha errore standard
    # 1/sqrt(n), quindi z = |r| sqrt(n) e' approssimativamente normale standard.
    # Con 55 coppie + 11 autocorrelazioni si applica una correzione di Bonferroni:
    # z_tol = 4 corrisponde a un livello globale ben inferiore all'1%.
    z_tol = cfg["z_tol"]
    out: List[dict] = []

    # A1. Test di consistenza del libro: dallo stato x0 = 1 sullo stream 0, dopo
    # 10 000 estrazioni lo stato deve valere esattamente CHECK = 399268537.
    rngs.SelectStream(0)
    rngs.PutSeed(1)
    for _ in range(10000):
        rngs.Random()
    got = rngs.GetSeed()
    out.append({"nome": "Lehmer: stato dopo 10^4 estrazioni da x0=1",
                "misurato": str(got), "atteso": str(rngs.CHECK),
                "esito": "PASS" if got == rngs.CHECK else "FAIL"})

    # A2. Dopo PlantSeeds i 256 stream devono partire da stati tutti distinti
    # (nessuna sovrapposizione all'origine).
    rngs.PlantSeeds(seed)
    seeds = [rngs._seed[i] for i in range(rngs.STREAMS)]
    distinct = len(set(seeds))
    out.append({"nome": "PlantSeeds: stati iniziali distinti sui 256 stream",
                "misurato": str(distinct), "atteso": str(rngs.STREAMS),
                "esito": "PASS" if distinct == rngs.STREAMS else "FAIL"})

    # A3. Uniformita' marginale di ciascuno degli 11 stream usati dal motore
    # (Kolmogorov-Smirnov contro U(0,1)) e incorrelazione a coppie.
    rngs.PlantSeeds(seed)
    draws: Dict[int, np.ndarray] = {}
    for s in streams:
        rngs.SelectStream(s)
        draws[s] = np.fromiter((rngs.Random() for _ in range(n)), dtype=float, count=n)

    worst_p, worst_s = 1.0, None
    for s in streams:
        p = float(stats.kstest(draws[s], "uniform").pvalue)
        if p < worst_p:
            worst_p, worst_s = p, s
    out.append({"nome": f"KS uniformita' marginale (p-value minimo, stream {worst_s})",
                "misurato": f"p = {worst_p:.4f}", "atteso": "p > 0.05",
                "esito": "PASS" if worst_p > 0.05 else "FAIL"})

    max_r, pair = 0.0, None
    for i, si in enumerate(streams):
        for sj in streams[i + 1:]:
            r = abs(float(np.corrcoef(draws[si], draws[sj])[0, 1]))
            if r > max_r:
                max_r, pair = r, (si, sj)
    z = max_r * math.sqrt(n)
    out.append({"nome": f"Correlazione a coppie fra stream (massima, {pair})",
                "misurato": f"|r| = {max_r:.5f}  (z = {z:.2f})",
                "atteso": f"z = |r| sqrt(n) < {z_tol}",
                "esito": "PASS" if z < z_tol else "FAIL"})

    # A4. Autocorrelazione di lag 1 entro ogni stream usato.
    max_a, max_as = 0.0, None
    for s in streams:
        x = draws[s]
        a = abs(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
        if a > max_a:
            max_a, max_as = a, s
    z = max_a * math.sqrt(n - 1)
    out.append({"nome": f"Autocorrelazione lag-1 entro stream (massima, stream {max_as})",
                "misurato": f"|r1| = {max_a:.5f}  (z = {z:.2f})",
                "atteso": f"z = |r1| sqrt(n) < {z_tol}",
                "esito": "PASS" if z < z_tol else "FAIL"})

    return out


# --------------------------------------------------------------------------- #
# [B] Generatori di variate
# --------------------------------------------------------------------------- #

def _sample(stream: int, fn, n: int) -> np.ndarray:
    """Genera n variate dallo stream indicato, selezionandolo una sola volta."""
    rngs.SelectStream(stream)
    return np.fromiter((fn() for _ in range(n)), dtype=float, count=n)


def _moment_row(nome: str, x: np.ndarray, m1: float, m2: float, z_tol: float) -> dict:
    """
    Confronto fra momenti empirici e teorici.

    Il criterio di esito e' statistico e non una soglia relativa fissa: la media
    campionaria di n variate i.i.d. ha errore standard sigma/sqrt(n), quindi
    z = |media_emp - media_teor| sqrt(n) / sigma deve stare entro poche unita'.
    Una soglia relativa fissa sarebbe inadeguata perche' le sorgenti del modello
    hanno coefficienti di variazione che differiscono di un ordine di grandezza
    (da 0.33 della PMF dei batch a 9.8 del delay di feedback).
    La deviazione standard e' riportata a scopo informativo.
    """
    n = len(x)
    sd = math.sqrt(max(m2 - m1 * m1, 0.0))
    e_mean, e_sd = float(np.mean(x)), float(np.std(x, ddof=1))
    z = abs(e_mean - m1) * math.sqrt(n) / sd if sd > 0 else 0.0
    err = abs(e_mean - m1) / abs(m1) if m1 else 0.0
    return {"nome": nome, "n": n,
            "media_teor": m1, "media_emp": e_mean,
            "sd_teor": sd, "sd_emp": e_sd,
            "err_rel": err, "z": z, "esito": "PASS" if z <= z_tol else "FAIL"}


def _gof_row(nome: str, stat: float, pvalue: float, kind: str, alpha: float) -> dict:
    return {"nome": nome, "test": kind, "stat": stat, "pvalue": pvalue,
            "esito": "PASS" if pvalue > alpha else "FAIL"}


def check_variates(conf: Config, cfg: dict) -> Tuple[List[dict], List[dict]]:
    """
    Verifica che ogni sorgente stocastica del modello generi effettivamente la
    distribuzione richiesta. Restituisce (righe_momenti, righe_adattamento).
    """
    n = cfg["n_samples"]
    alpha = cfg["alpha"]
    tol = cfg["z_tol"]
    rngs.PlantSeeds(cfg["seed"])

    moments: List[dict] = []
    gof: List[dict] = []

    # B1. Inter-arrivo esogeno: Esponenziale di media 1/lambda.
    mean_ia = 1.0 / conf.lambda_ext
    x = _sample(sim_mod.STREAM_ARRIVAL, lambda: rvgs.Exponential(mean_ia), n)
    moments.append(_moment_row("Inter-arrivo  Exp(1/lambda)", x,
                               mean_ia, 2.0 * mean_ia ** 2, tol))
    gof.append(_gof_row("Inter-arrivo  Exp(1/lambda)",
                        *stats.kstest(x, "expon", args=(0.0, mean_ia)), "KS", alpha))

    # B2. Dimensione del batch: PMF empirica a 9 valori.
    bv, bp = conf.pmf_batch_values, conf.pmf_batch_probs
    x = _sample(sim_mod.STREAM_BATCH, lambda: rvgs.Empirical(bv, bp), n)
    m1, m2 = _pmf_moments(bv, bp)
    moments.append(_moment_row("Dimensione batch  Empirical", x, m1, m2, tol))
    gof.append(_gof_row("Dimensione batch  Empirical", *_chi2(x, bv, bp), "chi2", alpha))

    # B3. Tempo di setup: PMF empirica a 28 valori.
    sv, sp = conf.pmf_setup_values, conf.pmf_setup_probs
    x = _sample(sim_mod.STREAM_SETUP, lambda: rvgs.Empirical(sv, sp), n)
    m1, m2 = _pmf_moments(sv, sp)
    moments.append(_moment_row("Tempo di setup  Empirical", x, m1, m2, tol))
    gof.append(_gof_row("Tempo di setup  Empirical", *_chi2(x, sv, sp), "chi2", alpha))

    # B4-B6. Le tre lognormali di servizio.
    for nome, stream, mu, sg in [
        ("Servizio test (passed)  Lognormal", sim_mod.STREAM_TEST, conf.mu_test, conf.sigma_test),
        ("Servizio failed  Lognormal", sim_mod.STREAM_FAIL_SVC, conf.mu_failed, conf.sigma_failed),
        ("Servizio canceled  Lognormal", sim_mod.STREAM_CANC_SVC, conf.mu_canceled, conf.sigma_canceled),
    ]:
        x = _sample(stream, lambda mu=mu, sg=sg: rvgs.Lognormal(mu, sg), n)
        m1, m2 = _lognormal_moments(mu, sg)
        moments.append(_moment_row(nome, x, m1, m2, tol))
        gof.append(_gof_row(nome, *stats.kstest(x, "lognorm",
                                                args=(sg, 0.0, math.exp(mu))), "KS", alpha))

    # B7. Delay di feedback umano (in minuti).
    x = _sample(sim_mod.STREAM_DELAY,
                lambda: rvgs.Lognormal(conf.mu_delay, conf.sigma_delay), n)
    m1, m2 = _lognormal_moments(conf.mu_delay, conf.sigma_delay)
    moments.append(_moment_row("Delay feedback [min]  Lognormal", x, m1, m2, tol))
    gof.append(_gof_row("Delay feedback [min]  Lognormal",
                        *stats.kstest(x, "lognorm",
                                      args=(conf.sigma_delay, 0.0, math.exp(conf.mu_delay))),
                        "KS", alpha))

    # B8. Esito del job: meccanismo a rischi competitivi (una sola uniforme, quattro rami).
    # Si replica esattamente la logica di core.simulator._calculate_service.
    cuts = [conf.p_passed,
            conf.p_passed + conf.p_failed,
            conf.p_passed + conf.p_failed + conf.p_errored]
    rngs.SelectStream(sim_mod.STREAM_FATE)
    codes = np.fromiter(
        (0 if (u := rvgs.Uniform(0.0, 1.0)) <= cuts[0]
         else 1 if u <= cuts[1] else 2 if u <= cuts[2] else 3
         for _ in range(n)), dtype=float, count=n)
    probs_fate = [conf.p_passed, conf.p_failed, conf.p_errored, conf.p_canceled]
    gof.append(_gof_row("Esito job  rischi competitivi (4 rami)",
                        *_chi2(codes, [0.0, 1.0, 2.0, 3.0], probs_fate), "chi2", alpha))
    for i, (nome, p) in enumerate(zip(["passed", "failed", "errored", "canceled"], probs_fate)):
        f = float(np.mean(codes == i))
        sd = math.sqrt(p * (1 - p))
        z = abs(f - p) * math.sqrt(n) / sd
        moments.append({"nome": f"  frazione esito '{nome}'", "n": n,
                        "media_teor": p, "media_emp": f,
                        "sd_teor": sd, "sd_emp": math.sqrt(f * (1 - f)),
                        "err_rel": abs(f - p) / p if p else 0.0,
                        "z": z, "esito": "PASS" if z <= tol else "FAIL"})

    # B9. Le tre Bernoulli del modello (retry e modo di errore).
    for nome, stream, p in [
        ("Retry dopo failed  Bernoulli", sim_mod.STREAM_RETRY, conf.p_retry_failed),
        ("Retry dopo errored  Bernoulli", sim_mod.STREAM_RETRY, conf.p_retry_errored),
        ("Errore precoce  Bernoulli", sim_mod.STREAM_ERR_MODE, conf.p_errored_early),
    ]:
        x = _sample(stream, lambda p=p: float(rvgs.Bernoulli(p)), n)
        moments.append(_moment_row(nome, x, p, p, tol))

    return moments, gof


def _chi2(sample: np.ndarray, values: Sequence[float],
          probs: Sequence[float]) -> Tuple[float, float]:
    """Test chi-quadro di adattamento di un campione discreto alla PMF (values, probs)."""
    n = len(sample)
    cnt = Counter(sample.tolist())
    obs = np.array([cnt.get(float(v), 0) for v in values], dtype=float)
    exp = np.array(probs, dtype=float) * n
    stat = float(np.sum((obs - exp) ** 2 / exp))
    df = len(values) - 1
    return stat, float(stats.chi2.sf(stat, df))


# --------------------------------------------------------------------------- #
# [C] Caso degenere M^[X]/G/1 con soluzione analitica esatta
# --------------------------------------------------------------------------- #

def mx_g1_theory(lam: float, ex: float, ex2: float,
                 es: float, es2: float) -> Dict[str, float]:
    """
    Soluzione esatta della coda M^[X]/G/1 (arrivi a batch di Poisson, servizio
    generale, un servente, disciplina FIFO).

    Il tempo di attesa di un cliente si scompone in due contributi indipendenti:

      1. il lavoro non smaltito trovato nel sistema dal batch cui appartiene. Per
         la PASTA questo e' il workload stazionario di una M/G/1 in cui ogni
         "cliente" e' l'intero batch, con richiesta di servizio S_B = somma di X
         servizi:
            E[V] = lambda * E[S_B^2] / (2 (1 - rho)),
            E[S_B^2] = E[X] E[S^2] + E[X(X-1)] E[S]^2;

      2. il servizio dei compagni di batch serviti prima di lui. Campionando un
         cliente a caso, la taglia del suo batch e' distorta dalla dimensione
         (P(X*=k) = k p_k / E[X]) e la sua posizione e' uniforme nel batch, da cui
            E[N_prima] = (E[X^2]/E[X] - 1) / 2.

    Quindi  E[Wq] = E[V] + E[S] (E[X^2]/E[X] - 1) / 2,  con rho = lambda E[X] E[S].
    Per X == 1 la formula degenera nella Pollaczek-Khinchine classica.
    """
    rho = lam * ex * es
    esb2 = ex * es2 + (ex2 - ex) * es * es
    e_v = lam * esb2 / (2.0 * (1.0 - rho))
    e_batch = es * (ex2 / ex - 1.0) / 2.0
    wq = e_v + e_batch
    lam_job = lam * ex
    return {"rho": rho, "Wq": wq, "Wq_workload": e_v, "Wq_batch": e_batch,
            "Lq": lam_job * wq, "W": wq + es, "L": lam_job * (wq + es)}


def _degenerate_config(conf: Config, cfg: dict, lam: float,
                       run_length: float) -> Config:
    """Applica gli override del caso degenere a una copia superficiale della Config."""
    import copy
    d = copy.copy(conf)
    for k, v in cfg["overrides"].items():
        setattr(d, k, v)
    total = d.p_passed + d.p_failed + d.p_errored + d.p_canceled
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Override degenere incoerente: le probabilita' sommano a {total}")
    d.lambda_ext = lam
    d.run_length = run_length
    d.max_time = run_length
    d.warmup = cfg["warmup_fraction"] * run_length
    d.num_batches = cfg["num_batches"]
    d.sampling_interval = run_length / cfg["state_samples"]
    return d


def check_analytic(conf: Config, cfg: dict) -> List[dict]:
    """
    Per ciascun livello di utilizzazione richiesto esegue una run del motore nella
    configurazione degenere e confronta le stime (batch means, IC 95%) con la
    soluzione esatta della M^[X]/G/1.
    """
    from analysis.batchmeans import _batch_means

    ex, ex2 = _pmf_moments(conf.pmf_batch_values, conf.pmf_batch_probs)
    es, es2 = service_moments(conf)
    rows: List[dict] = []

    for rho in cfg["rho_targets"]:
        lam = rho / (ex * es)
        run_length = cfg["target_jobs"] / (lam * ex) / (1.0 - cfg["warmup_fraction"])
        d = _degenerate_config(conf, cfg, lam, run_length)
        th = mx_g1_theory(lam, ex, ex2, es, es2)

        sim = simulate_once(d, cfg["seed"])
        st, jt = sim.state_trace, sim.job_trace
        t_st = [r[0] for r in st]
        t_ex = [r[1] for r in jt]
        k, w, T = d.num_batches, d.warmup, d.run_length

        measured = {
            "Wq": _batch_means([r[2] for r in jt], t_ex, w, T, k),
            "W":  _batch_means([r[4] for r in jt], t_ex, w, T, k),
            "Lq": _batch_means([r[2] for r in st], t_st, w, T, k),
            "L":  _batch_means([r[2] + r[3] for r in st], t_st, w, T, k),
            "rho": _batch_means([r[1] for r in st], t_st, w, T, k),
        }
        for key in ("rho", "Wq", "W", "Lq", "L"):
            mean, half = mean_ci(list(measured[key]))
            teor = th[key]
            err = abs(mean - teor) / abs(teor) if teor else float("nan")
            coperto = abs(mean - teor) <= half
            rows.append({
                "rho_target": rho, "grandezza": key, "teorico": teor,
                "simulato": mean, "ic_half": half, "err_rel": err,
                "n_job": len(jt),
                "esito": "PASS" if (coperto or err <= cfg["rel_tol"]) else "FAIL",
                "ic_copre": "si" if coperto else "no",
            })
        print(f"  rho={rho:.1f}: {len(jt)} job serviti, "
              f"Wq teor {th['Wq']:.1f}s vs sim {mean_ci(list(measured['Wq']))[0]:.1f}s")

    return rows


# --------------------------------------------------------------------------- #
# [D] Conservazione del flusso e fattore di visita
# --------------------------------------------------------------------------- #

def check_flow(conf: Config, bundle: dict, cfg: dict) -> List[dict]:
    """
    Bilancio dei job e fattore di visita sulla configurazione nominale, misurati
    dalle tracce della run steady-state.
    """
    rows: List[dict] = []
    arrivi = sum(b for tr in bundle["batch_traces"] for (_, b) in tr)
    uscite = sum(len(tr) for tr in bundle["job_traces"])
    visite = sum(len(tr) for tr in bundle["node_traces"])

    # D1. Conservazione: nessun job puo' essere creato o distrutto dal motore.
    # I job non ancora usciti a fine run sono quelli in coda, in servizio o in feedback.
    residui = arrivi - uscite
    rows.append({"nome": "Conservazione: arrivi - uscite = job ancora nel sistema",
                 "misurato": f"{arrivi} - {uscite} = {residui}",
                 "atteso": ">= 0 e trascurabile rispetto agli arrivi",
                 "err_rel": residui / arrivi if arrivi else float("nan"),
                 "esito": "PASS" if 0 <= residui and residui / arrivi < cfg["rel_tol"] else "FAIL"})

    # D2. Fattore di visita: q = P(rientro in coda dopo il feedback).
    q = conf.p_failed * conf.p_retry_failed + conf.p_errored * conf.p_retry_errored
    v_teor = 1.0 / (1.0 - q)
    v_mis = visite / uscite if uscite else float("nan")
    err = abs(v_mis - v_teor) / v_teor
    rows.append({"nome": "Fattore di visita al nodo: v = 1/(1-q)",
                 "misurato": f"{v_mis:.5f} ({visite} visite / {uscite} job)",
                 "atteso": f"{v_teor:.5f} (q = {q:.5f})",
                 "err_rel": err,
                 "esito": "PASS" if err <= cfg["rel_tol"] else "FAIL"})

    # D3. Tasso di arrivo al nodo: lambda_nodo = lambda * E[X] * v.
    ex, _ = _pmf_moments(conf.pmf_batch_values, conf.pmf_batch_probs)
    x_teor = conf.lambda_ext * ex * v_teor
    x_mis = visite / (len(bundle["node_traces"]) * conf.max_time)
    err = abs(x_mis - x_teor) / x_teor
    rows.append({"nome": "Throughput del nodo: X = lambda E[X] v",
                 "misurato": f"{x_mis:.6e} job/s", "atteso": f"{x_teor:.6e} job/s",
                 "err_rel": err,
                 "esito": "PASS" if err <= cfg["rel_tol"] else "FAIL"})

    # D4. Utilizzazione attesa: rho = X * E[S] / c, con E[S] media pesata sui quattro
    # esiti (l'errore precoce consuma solo una frazione uniforme del setup).
    s1, _ = _pmf_moments(conf.pmf_setup_values, conf.pmf_setup_probs)
    t1, _ = _lognormal_moments(conf.mu_test, conf.sigma_test)
    f1, _ = _lognormal_moments(conf.mu_failed, conf.sigma_failed)
    c1, _ = _lognormal_moments(conf.mu_canceled, conf.sigma_canceled)
    es_mix = (conf.p_passed * (s1 + t1)
              + conf.p_failed * (s1 + f1)
              + conf.p_canceled * (s1 + c1)
              + conf.p_errored * (conf.p_errored_early * 0.5 * s1
                                  + (1.0 - conf.p_errored_early) * (s1 + t1)))
    rho_teor = x_teor * es_mix / conf.c
    rho_mis = float(np.mean([w["utilization"] for w in bundle["welford_results"]]))
    err = abs(rho_mis - rho_teor) / rho_teor
    rows.append({"nome": "Utilizzazione: rho = X E[S] / c",
                 "misurato": f"{rho_mis:.6f}", "atteso": f"{rho_teor:.6f} (E[S]={es_mix:.2f}s)",
                 "err_rel": err,
                 "esito": "PASS" if err <= cfg["rel_tol"] else "FAIL"})

    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(conf: Config, bundle: dict, cfg_path: str = "verification.json") -> None:
    """Esegue i quattro blocchi di verifica strutturale, stampa e salva i report."""
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== VERIFICA STRUTTURALE ===")

    print("\n[A] Generatore pseudo-casuale di Lehmer")
    rng_rows = check_rng(cfg["rng"])
    for r in rng_rows:
        print(f"  [{r['esito']:<4}] {r['nome']:<58} {r['misurato']:>22}  (atteso {r['atteso']})")
    _write_csv(os.path.join(out_dir, "rng_report.csv"),
               ["nome", "misurato", "atteso", "esito"],
               [[r["nome"], r["misurato"], r["atteso"], r["esito"]] for r in rng_rows])

    print(f"\n[B] Generatori di variate ({cfg['variates']['n_samples']} campioni per sorgente)")
    moments, gof = check_variates(conf, cfg["variates"])
    print(f"  {'Sorgente':<40}{'media teor':>13}{'media emp':>13}"
          f"{'sd teor':>13}{'sd emp':>13}{'err.rel':>9}{'z':>7}")
    for r in moments:
        print(f"  [{r['esito']:<4}] {r['nome']:<34}{r['media_teor']:>13.4f}{r['media_emp']:>13.4f}"
              f"{r['sd_teor']:>13.4f}{r['sd_emp']:>13.4f}{r['err_rel']:>8.2%}{r['z']:>7.2f}")
    print(f"\n  {'Test di adattamento':<40}{'tipo':>8}{'statistica':>14}{'p-value':>10}")
    for r in gof:
        print(f"  [{r['esito']:<4}] {r['nome']:<34}{r['test']:>8}{r['stat']:>14.5f}{r['pvalue']:>10.4f}")
    _write_csv(os.path.join(out_dir, "variates_moments.csv"),
               ["sorgente", "n", "media_teorica", "media_empirica", "sd_teorica",
                "sd_empirica", "err_rel", "z", "esito"],
               [[r["nome"], r["n"], f"{r['media_teor']:.6f}", f"{r['media_emp']:.6f}",
                 f"{r['sd_teor']:.6f}", f"{r['sd_emp']:.6f}", f"{r['err_rel']:.6f}",
                 f"{r['z']:.3f}", r["esito"]] for r in moments])
    _write_csv(os.path.join(out_dir, "variates_gof.csv"),
               ["sorgente", "test", "statistica", "p_value", "esito"],
               [[r["nome"], r["test"], f"{r['stat']:.6f}", f"{r['pvalue']:.6f}", r["esito"]]
                for r in gof])

    print("\n[C] Caso degenere M^[X]/G/1: simulazione vs soluzione analitica esatta")
    an_rows = check_analytic(conf, cfg["analytic"])
    print(f"\n  {'rho':>5}{'grandezza':>11}{'teorico':>14}{'simulato':>14}"
          f"{'+/- IC 95%':>13}{'err.rel':>9}{'IC copre':>10}")
    for r in an_rows:
        print(f"  [{r['esito']:<4}]{r['rho_target']:>5.1f}{r['grandezza']:>11}"
              f"{r['teorico']:>14.4f}{r['simulato']:>14.4f}{r['ic_half']:>13.4f}"
              f"{r['err_rel']:>8.2%}{r['ic_copre']:>10}")
    _write_csv(os.path.join(out_dir, "analytic_report.csv"),
               ["rho_target", "grandezza", "teorico", "simulato", "ic_half_95",
                "err_rel", "ic_copre", "n_job", "esito"],
               [[r["rho_target"], r["grandezza"], f"{r['teorico']:.6f}",
                 f"{r['simulato']:.6f}", f"{r['ic_half']:.6f}", f"{r['err_rel']:.6f}",
                 r["ic_copre"], r["n_job"], r["esito"]] for r in an_rows])

    print("\n[D] Conservazione del flusso e fattore di visita (configurazione nominale)")
    flow_rows = check_flow(conf, bundle, cfg["flow"])
    for r in flow_rows:
        print(f"  [{r['esito']:<4}] {r['nome']:<52}")
        print(f"         misurato: {r['misurato']}   atteso: {r['atteso']}   "
              f"err.rel. {r['err_rel']:.2%}")
    _write_csv(os.path.join(out_dir, "flow_report.csv"),
               ["nome", "misurato", "atteso", "err_rel", "esito"],
               [[r["nome"], r["misurato"], r["atteso"], f"{r['err_rel']:.6f}", r["esito"]]
                for r in flow_rows])

    tutti = rng_rows + moments + gof + an_rows + flow_rows
    falliti = [r for r in tutti if r["esito"] != "PASS"]
    print(f"\n  -> Verifica strutturale: {len(tutti) - len(falliti)}/{len(tutti)} controlli superati")
    for r in falliti:
        print(f"     FAIL: {r.get('nome', r.get('grandezza'))}")
    print(f"  Report salvati in {out_dir}/")
