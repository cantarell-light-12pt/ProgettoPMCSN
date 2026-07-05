"""
Entrypoint principale del Software di Simulazione (singola configurazione).

Flusso:
  1. Analisi del TRANSITORIO col metodo delle repliche indipendenti (orizzonte finito):
     5 run con 5 seed diversi, grafici delle medie cumulate (utilizzazione, servizio
     dei job passed, tempo di risposta).
  2. Analisi STEADY-STATE col metodo dei BATCH MEANS (orizzonte infinito): single long
     run, tabella media + IC 95% (t di Student).
  3. Verifica (legge di Little, controlli valori) e Validazione (vs dataset reale) sulla
     run a orizzonte infinito.
"""
from config.settings import Config
from analysis import transient, batchmeans
from vv import verification, validation


def main() -> None:
    """Carica la configurazione validata e innesca transitorio + batch means + V&V."""
    try:
        conf = Config("config.json")
        # 1. Transitorio (repliche indipendenti, orizzonte finito)
        transient.run_transient(conf)
        # 2. Steady-state (batch means, orizzonte infinito)
        bundle = batchmeans.run_batch_means(conf)
        # 3. Verifica e Validazione sulla run infinita
        verification.run(bundle["job_traces"], bundle["state_traces"],
                         bundle["node_traces"], bundle["welford_results"], conf)
        validation.run(conf, bundle)
    except FileNotFoundError:
        print("Errore critico: File 'config.json' non trovato. Impossibile avviare il DES.")
    except ValueError as ve:
        print(f"Errore di Validazione Parametri: {ve}")


if __name__ == "__main__":
    main()
