"""
Modulo di definizione delle entità di simulazione.
"""
from dataclasses import dataclass

@dataclass
class Job:
    """
    Rappresenta un singolo processo di build/test (Job CI/CD).
    Contiene gli attributi temporali utili alle statistiche di Welford.
    """
    id: int
    creation_time: float
    queue_enter_time: float = 0.0
    service_start_time: float = 0.0
    total_wait_time: float = 0.0
    total_service_time: float = 0.0
    current_fate: str = ""