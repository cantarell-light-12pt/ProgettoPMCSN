"""
Verifica interna del simulatore a partire dalle tracce.

Due livelli:
1. Controlli di validita' sui valori raccolti (tempi non negativi, coerenza di
   response >= wait+service, stato entro i limiti fisici, esiti ammessi).
2. Legge di Little sul nodo coda+serventi, con scomposizione coda / serventi:
   L (numero medio nel nodo, misurato dal processo di stato) deve uguagliare
   X * W (throughput per tempo medio di residenza, misurato dai tempi delle
   visite). Le due misure sono indipendenti: la loro coincidenza verifica la
   correttezza della contabilita' degli eventi.

Tutte le funzioni operano sulle tracce in memoria (liste per replica), che sono
la stessa cosa persistita nei CSV di output.
"""
import csv
import os
from typing import List, Sequence

EPS = 1e-6
VALID_FATES = {"passed", "failed", "errored", "canceled"}


def check_values(job_traces: Sequence[Sequence[tuple]],
                 state_traces: Sequence[Sequence[tuple]],
                 node_traces: Sequence[Sequence[tuple]],
                 c: int) -> List[tuple]:
    """
    Esegue i controlli di validita' sui valori, aggregando tutte le repliche.

    Returns:
        Lista di tuple (nome_controllo, n_violazioni, n_totale).
    """
    jt = [r for tr in job_traces for r in tr]      # (id, exit, wait, service, response)
    nt = [r for tr in node_traces for r in tr]     # (t, fate, wait_visit, service_visit)
    st = [r for tr in state_traces for r in tr]    # (t, util, qlen, servers, feedback)

    checks: List[tuple] = [
        ("job: wait_time >= 0",
         sum(1 for r in jt if r[2] < -EPS), len(jt)),
        ("job: service_time >= 0",
         sum(1 for r in jt if r[3] < -EPS), len(jt)),
        ("job: response_time >= 0",
         sum(1 for r in jt if r[4] < -EPS), len(jt)),
        ("job: response >= wait+service",
         sum(1 for r in jt if r[4] < r[2] + r[3] - EPS), len(jt)),
        ("node: wait_visit >= 0",
         sum(1 for r in nt if r[2] < -EPS), len(nt)),
        ("node: service_visit >= 0",
         sum(1 for r in nt if r[3] < -EPS), len(nt)),
        ("node: fate ammesso",
         sum(1 for r in nt if r[1] not in VALID_FATES), len(nt)),
        ("state: 0 <= utilization <= 1",
         sum(1 for r in st if r[1] < -EPS or r[1] > 1.0 + EPS), len(st)),
        ("state: 0 <= servers_busy <= c",
         sum(1 for r in st if r[3] < 0 or r[3] > c), len(st)),
        ("state: queue_len >= 0",
         sum(1 for r in st if r[2] < 0), len(st)),
        ("state: feedback_len >= 0",
         sum(1 for r in st if r[4] < 0), len(st)),
    ]
    return checks


def _rel_err(measured: float, expected: float) -> float:
    """Errore relativo di 'measured' rispetto a 'expected'."""
    denom = abs(expected) if abs(expected) > EPS else EPS
    return abs(measured - expected) / denom


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def check_little(state_traces: Sequence[Sequence[tuple]],
                 node_traces: Sequence[Sequence[tuple]],
                 welford_results: Sequence[dict],
                 max_time: float, c: int) -> List[dict]:
    """
    Verifica la legge di Little sul nodo (coda+serventi) e sulle sue componenti,
    mediando le grandezze sulle repliche.

    Per ogni replica:
      - lato L (numero medio nel nodo): media temporale esatta di Welford
        (syslen=coda+serventi, qlen=coda, utilization*c=serventi) e, in parallelo,
        media campionata dallo state_trace (per mostrare che la traccia approssima
        l'integrale temporale);
      - lato X*W (throughput * residenza): dal node_trace, X=n_visite/max_time,
        Wq=media(wait_visit), Ws=media(service_visit), W=Wq+Ws.

    Returns:
        dict con chiave 'little' (lista di confronti L vs X*W) e 'sampling' (confronto
        informativo traccia-campionata vs media esatta di Welford).
    """
    L_node, Lq, Ls, L_node_sampled = [], [], [], []
    XW_node, XWq, XWs = [], [], []

    for st, nt, wf in zip(state_traces, node_traces, welford_results):
        # Lato L: medie temporali esatte (Welford)
        L_node.append(wf["syslen"])
        Lq.append(wf["qlen"])
        Ls.append(wf["utilization"] * c)

        # Lato L: media campionata dallo state_trace (queue+servers)
        if st:
            L_node_sampled.append(_avg([r[2] + r[3] for r in st]))

        # Lato X*W: dai tempi di residenza delle visite al nodo
        n = len(nt)
        X = n / max_time
        if n:
            Wq = _avg([r[2] for r in nt])
            Ws = _avg([r[3] for r in nt])
        else:
            Wq = Ws = 0.0
        XW_node.append(X * (Wq + Ws))
        XWq.append(X * Wq)
        XWs.append(X * Ws)

    def entry(name, L_side, XW_side):
        m_L, m_XW = _avg(L_side), _avg(XW_side)
        return {"name": name, "L": m_L, "XW": m_XW, "rel_err": _rel_err(m_XW, m_L)}

    little = [
        entry("Nodo (coda+serventi):  L = X*(Wq+Ws)", L_node, XW_node),
        entry("Coda:                  Lq = X*Wq", Lq, XWq),
        entry("Serventi (utilizz.):   Ls = X*Ws", Ls, XWs),
    ]
    # Accuratezza (informativa): quanto la media campionata dallo state_trace
    # approssima l'integrale temporale esatto di Welford. Non e' un test di Little.
    sampling = {
        "name": "L nodo: campionato (traccia) vs esatto (Welford)",
        "L": _avg(L_node), "XW": _avg(L_node_sampled),
        "rel_err": _rel_err(_avg(L_node_sampled), _avg(L_node)),
    }
    return {"little": little, "sampling": sampling}


def run(job_traces, state_traces, node_traces, welford_results, config) -> None:
    """
    Esegue Verifica completa: controlli valori + legge di Little, stampa il report
    e salva 'verification_report.csv' nella cartella di output.
    """
    print("\n=== VERIFICA (dalle tracce) ===")

    print("\n[1] Controlli di validita' sui valori raccolti")
    value_checks = check_values(job_traces, state_traces, node_traces, config.c)
    all_ok = True
    for name, viol, tot in value_checks:
        ok = (viol == 0)
        all_ok = all_ok and ok
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:<34} violazioni: {viol}/{tot}")
    print(f"  -> Esito controlli valori: {'PASS' if all_ok else 'FAIL'}")

    print("\n[2] Legge di Little (nodo coda + {} serventi)".format(config.c))
    little_res = check_little(state_traces, node_traces, welford_results,
                              config.max_time, config.c)
    little = little_res["little"]
    sampling = little_res["sampling"]
    little_ok = True
    print(f"  {'Relazione':<42}{'L (stato)':>12}{'X*W (visite)':>14}{'err.rel.':>10}")
    for r in little:
        ok = r["rel_err"] <= config.little_tol
        little_ok = little_ok and ok
        print(f"  [{'OK ' if ok else 'FAIL'}] {r['name']:<38}{r['L']:>12.4f}{r['XW']:>14.4f}{r['rel_err']:>9.2%}")
    print(f"  -> Legge di Little: {'PASS' if little_ok else 'FAIL'} (tolleranza {config.little_tol:.1%})")
    print(f"  (info) {sampling['name']}: esatto={sampling['L']:.4f} "
          f"campionato={sampling['XW']:.4f} scarto={sampling['rel_err']:.1%} "
          f"[accuratezza campionamento, non un test di Little]")

    # Salvataggio CSV
    path = os.path.join(config.trace_dir, "verification_report.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sezione", "voce", "valore1", "valore2", "esito"])
        for name, viol, tot in value_checks:
            w.writerow(["value_check", name, viol, tot, "PASS" if viol == 0 else "FAIL"])
        for r in little:
            esito = "PASS" if r["rel_err"] <= config.little_tol else "FAIL"
            w.writerow(["little", r["name"], f"{r['L']:.6f}", f"{r['XW']:.6f}", esito])
        w.writerow(["sampling_info", sampling["name"],
                    f"{sampling['L']:.6f}", f"{sampling['XW']:.6f}",
                    f"scarto={sampling['rel_err']:.4f}"])
    print(f"  Report salvato in {path}")
