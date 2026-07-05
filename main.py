"""
Entrypoint principale del Software di Simulazione (singola configurazione).

Esegue la pipeline completa di una simulazione (repliche, tracce, analisi del
transitorio con Welch, tabella numerica) tramite runner.run_simulation_suite, poi
aggiunge Verifica e Validazione sulla base delle tracce estratte. I risultati
finiscono nella cartella indicata da config.json (output.trace_dir).
"""
from config.settings import Config
from runner import run_simulation_suite
from vv import verification, validation


def main() -> None:
    """Carica la configurazione validata e innesca simulazione + V&V."""
    try:
        conf = Config("config.json")
        bundle = run_simulation_suite(conf)
        # Verifica e Validazione sulla base delle tracce estratte.
        verification.run(bundle["job_traces"], bundle["state_traces"],
                         bundle["node_traces"], bundle["welford_results"], conf)
        validation.run(conf, bundle["node_traces"], bundle["batch_traces"])
    except FileNotFoundError:
        print("Errore critico: File 'config.json' non trovato. Impossibile avviare il DES.")
    except ValueError as ve:
        print(f"Errore di Validazione Parametri: {ve}")


if __name__ == "__main__":
    main()
