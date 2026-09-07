"""
Entrypoint principale del Software di Simulazione (singola configurazione).

Flusso:
  1. Analisi del TRANSITORIO col metodo delle repliche indipendenti (orizzonte finito):
     5 run con 5 seed diversi, grafici delle medie cumulate (utilizzazione, servizio
     dei job passed, tempo di risposta).
  2. Analisi STEADY-STATE col metodo dei BATCH MEANS (orizzonte infinito): single long
     run, tabella media + IC 95% (t di Student).
  3. Verifica sulla run a orizzonte infinito: controlli di validita' dei valori e
     legge di Little sulle tracce (vv/verification.py), piu' verifica strutturale
     del modello computazionale (vv/consistency.py): generatore di Lehmer,
     generatori di variate, caso degenere M^[X]/G/1 con soluzione analitica esatta,
     conservazione del flusso.
  4. Validazione rispetto al dataset reale.
"""
from config.settings import Config
from analysis import transient, batchmeans
from vv import verification, validation, consistency


def main() -> None:
    """Carica la configurazione validata e innesca transitorio + batch means + V&V."""
    try:
        conf = Config("config.json")
        # 1. Transitorio (repliche indipendenti, orizzonte finito)
        transient.run_transient(conf)
        # 2. Steady-state (batch means, orizzonte infinito)
        bundle = batchmeans.run_batch_means(conf)
        # 3. Verifica sulla run infinita: tracce + struttura del modello
        verification.run(bundle["job_traces"], bundle["state_traces"],
                         bundle["node_traces"], bundle["welford_results"], conf)
        consistency.run(conf, bundle)
        # 4. Validazione rispetto al dataset reale
        validation.run(conf, bundle)
    except FileNotFoundError:
        print("Errore critico: File 'config.json' non trovato. Impossibile avviare il DES.")
    except ValueError as ve:
        print(f"Errore di Validazione Parametri: {ve}")


if __name__ == "__main__":
    main()
