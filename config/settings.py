"""
Modulo per il caricamento, il parsing e la validazione formale dei parametri.
"""
import json
from typing import List


class Config:
    """
    Classe proxy per la configurazione del sistema letta da un JSON su filesystem.
    Isola le logiche di I/O dal motore di simulazione.
    """

    def __init__(self, filepath: str) -> None:
        """
        Inizializza e valida i parametri.

        Args:
            filepath (str): Path assoluto o relativo al file di configurazione JSON.

        Raises:
            ValueError: Se la somma delle probabilità di routing devia da 1.0.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.seed: int = data["simulation"]["seed"]
        # max_time e' impostato dal driver prima di ogni run (= finite_horizon per il
        # transitorio, = run_length per la run batch means). Default: run_length.
        self.max_time: float = 0.0

        self.c: int = data["architecture"]["c"]
        self.lambda_ext: float = data["traffic"]["lambda_ext"]

        self.pmf_batch_values: List[float] = data["traffic"]["pmf_batch"]["values"]
        self.pmf_batch_probs: List[float] = data["traffic"]["pmf_batch"]["probs"]

        self.p_passed: float = data["routing"]["p_passed"]
        self.p_failed: float = data["routing"]["p_failed"]
        self.p_errored: float = data["routing"]["p_errored"]
        self.p_canceled: float = data["routing"]["p_canceled"]
        # Errored bimodale: prob. di errore PRECOCE (rottura nel setup) vs tardivo (build completo).
        self.p_errored_early: float = data["routing"]["p_errored_early"]

        # Validazione strutturale dei Rischi Competitivi
        total_p = self.p_passed + self.p_failed + self.p_errored + self.p_canceled
        if abs(total_p - 1.0) > 1e-6:
            raise ValueError(f"Constraint Violato: Le probabilità di routing sommano a {total_p}, devono sommare a 1.0")
        if not (0.0 <= self.p_errored_early <= 1.0):
            raise ValueError(
                f"Constraint Violato: 'routing.p_errored_early' deve essere in [0,1], trovato {self.p_errored_early}"
            )

        self.pmf_setup_values: List[float] = data["service_setup"]["pmf_setup"]["values"]
        self.pmf_setup_probs: List[float] = data["service_setup"]["pmf_setup"]["probs"]

        self.mu_test: float = data["service_test"]["mu_test"]
        self.sigma_test: float = data["service_test"]["sigma_test"]
        # Servizio di failed/canceled: lognormale calibrata sulle rispettive buildduration
        # reali (netto setup), invece del troncamento uniforme p_cut (che sottostimava).
        self.mu_failed: float = data["service_failed"]["mu"]
        self.sigma_failed: float = data["service_failed"]["sigma"]
        self.mu_canceled: float = data["service_canceled"]["mu"]
        self.sigma_canceled: float = data["service_canceled"]["sigma"]
        if self.sigma_failed <= 0 or self.sigma_canceled <= 0:
            raise ValueError("Constraint Violato: 'sigma' di service_failed/service_canceled deve essere > 0")

        self.mu_delay: float = data["human_feedback"]["mu_delay"]
        self.sigma_delay: float = data["human_feedback"]["sigma_delay"]
        self.p_retry_failed: float = data["human_feedback"]["p_retry_failed"]
        self.p_retry_errored: float = data["human_feedback"]["p_retry_errored"]

        # Parametri operativi (non del modello): output tracce.
        self.trace_dir: str = data["output"]["trace_dir"]
        self.sampling_interval: float = data["output"]["sampling_interval"]

        # Transitorio con repliche indipendenti (orizzonte finito).
        self.finite_horizon: float = data["transient"]["finite_horizon"]
        self.seeds: List[int] = data["transient"]["seeds"]

        # Steady-state con batch means (orizzonte "infinito", single long run).
        self.run_length: float = data["batch_means"]["run_length"]
        self.num_batches: int = data["batch_means"]["num_batches"]
        self.warmup: float = data["batch_means"]["warmup"]

        # Validazione dei parametri operativi
        if not self.trace_dir:
            raise ValueError("Constraint Violato: 'output.trace_dir' non puo' essere vuoto")
        if self.sampling_interval <= 0:
            raise ValueError(
                f"Constraint Violato: 'output.sampling_interval' deve essere > 0, trovato {self.sampling_interval}"
            )
        if self.finite_horizon <= 0:
            raise ValueError(
                f"Constraint Violato: 'transient.finite_horizon' deve essere > 0, trovato {self.finite_horizon}"
            )
        if not self.seeds:
            raise ValueError("Constraint Violato: 'transient.seeds' non puo' essere vuoto")
        if self.run_length <= 0:
            raise ValueError(
                f"Constraint Violato: 'batch_means.run_length' deve essere > 0, trovato {self.run_length}"
            )
        if self.num_batches < 2:
            raise ValueError(
                f"Constraint Violato: 'batch_means.num_batches' deve essere >= 2 (serve per l'IC), trovato {self.num_batches}"
            )
        if not (0.0 <= self.warmup < self.run_length):
            raise ValueError(
                f"Constraint Violato: 'batch_means.warmup' deve essere in [0, run_length), trovato {self.warmup}"
            )

        # Parametri operativi di Verifica e Validazione.
        self.dataset_path: str = data["validation"]["dataset_path"]
        self.target_project: str = data["validation"]["target_project"]
        self.little_tol: float = data["verification"]["little_tol"]

        if not self.dataset_path:
            raise ValueError("Constraint Violato: 'validation.dataset_path' non puo' essere vuoto")
        if not self.target_project:
            raise ValueError("Constraint Violato: 'validation.target_project' non puo' essere vuoto")
        if self.little_tol <= 0:
            raise ValueError(
                f"Constraint Violato: 'verification.little_tol' deve essere > 0, trovato {self.little_tol}"
            )


class ExperimentConfig:
    """
    Configurazione dell'esperimento di sweep sul numero di serventi, letta da un
    file JSON SEPARATO da config.json (che contiene solo i parametri di una singola
    simulazione). Qui vivono solo i parametri specifici dell'esperimento.
    """

    def __init__(self, filepath: str) -> None:
        """
        Args:
            filepath: path del file JSON dell'esperimento.

        Raises:
            ValueError: se la lista dei serventi e' vuota o contiene valori < 1,
                oppure se 'output_dir' e' vuoto.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.server_counts: List[int] = data["server_counts"]
        self.output_dir: str = data["output_dir"]

        if not self.server_counts:
            raise ValueError("Constraint Violato: 'server_counts' non puo' essere vuoto")
        if any(int(c) < 1 for c in self.server_counts):
            raise ValueError(
                f"Constraint Violato: ogni valore in 'server_counts' deve essere >= 1, trovato {self.server_counts}"
            )
        if not self.output_dir:
            raise ValueError("Constraint Violato: 'output_dir' non puo' essere vuoto")