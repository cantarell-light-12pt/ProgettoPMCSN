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

        self.max_time: float = data["simulation"]["max_time"]
        self.seed: int = data["simulation"]["seed"]

        self.c: int = data["architecture"]["c"]
        self.lambda_ext: float = data["traffic"]["lambda_ext"]

        self.pmf_batch_values: List[float] = data["traffic"]["pmf_batch"]["values"]
        self.pmf_batch_probs: List[float] = data["traffic"]["pmf_batch"]["probs"]

        self.p_passed: float = data["routing"]["p_passed"]
        self.p_failed: float = data["routing"]["p_failed"]
        self.p_errored: float = data["routing"]["p_errored"]
        self.p_canceled: float = data["routing"]["p_canceled"]

        # Validazione strutturale dei Rischi Competitivi
        total_p = self.p_passed + self.p_failed + self.p_errored + self.p_canceled
        if abs(total_p - 1.0) > 1e-6:
            raise ValueError(f"Constraint Violato: Le probabilità di routing sommano a {total_p}, devono sommare a 1.0")

        self.pmf_setup_values: List[float] = data["service_setup"]["pmf_setup"]["values"]
        self.pmf_setup_probs: List[float] = data["service_setup"]["pmf_setup"]["probs"]

        self.mu_test: float = data["service_test"]["mu_test"]
        self.sigma_test: float = data["service_test"]["sigma_test"]

        self.mu_delay: float = data["human_feedback"]["mu_delay"]
        self.sigma_delay: float = data["human_feedback"]["sigma_delay"]
        self.p_retry_failed: float = data["human_feedback"]["p_retry_failed"]
        self.p_retry_errored: float = data["human_feedback"]["p_retry_errored"]

        # Parametri operativi (non del modello): output tracce e analisi transitorio.
        self.trace_dir: str = data["output"]["trace_dir"]
        self.sampling_interval: float = data["output"]["sampling_interval"]
        self.welch_window: int = data["analysis"]["welch_window"]

        # Numero di repliche ADATTIVO: si aggiungono repliche finche' la semi-ampiezza
        # relativa dell'IC della metrica-obiettivo scende sotto la soglia, entro [min, max].
        self.min_replications: int = data["analysis"]["min_replications"]
        self.max_replications: int = data["analysis"]["max_replications"]
        self.target_metric: str = data["analysis"]["target_metric"]
        self.target_rel_halfwidth: float = data["analysis"]["target_rel_halfwidth"]

        # Validazione dei parametri operativi
        if not self.trace_dir:
            raise ValueError("Constraint Violato: 'output.trace_dir' non puo' essere vuoto")
        if self.sampling_interval <= 0:
            raise ValueError(
                f"Constraint Violato: 'output.sampling_interval' deve essere > 0, trovato {self.sampling_interval}"
            )
        if self.welch_window < 0:
            raise ValueError(
                f"Constraint Violato: 'analysis.welch_window' deve essere >= 0, trovato {self.welch_window}"
            )
        valid_metrics = {"utilization", "service_time", "wait_time",
                         "response_time", "qlen", "syslen"}
        if self.min_replications < 2:
            raise ValueError(
                f"Constraint Violato: 'analysis.min_replications' deve essere >= 2 (serve per l'IC), trovato {self.min_replications}"
            )
        if self.max_replications < self.min_replications:
            raise ValueError(
                f"Constraint Violato: 'analysis.max_replications' ({self.max_replications}) deve essere >= min_replications ({self.min_replications})"
            )
        if self.target_metric not in valid_metrics:
            raise ValueError(
                f"Constraint Violato: 'analysis.target_metric' deve essere uno di {sorted(valid_metrics)}, trovato '{self.target_metric}'"
            )
        if self.target_rel_halfwidth <= 0:
            raise ValueError(
                f"Constraint Violato: 'analysis.target_rel_halfwidth' deve essere > 0, trovato {self.target_rel_halfwidth}"
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