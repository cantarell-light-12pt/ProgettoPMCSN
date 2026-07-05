# File: core/__init__.py
"""
Package del motore di simulazione ad eventi discreti (DES).

Espone le entità di dominio (Job) e il motore principale (TravisCISimulator).
Questa astrazione permette di disaccoppiare l'entrypoint dall'organizzazione
interna dei file del core (models.py e simulator.py).
"""

from .models import Job
from .simulator import TravisCISimulator

__all__ = [
    "Job",
    "TravisCISimulator"
]