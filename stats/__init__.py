# File: stats/__init__.py
"""
Package per il tracciamento statistico della simulazione.

Espone esclusivamente le implementazioni dell'Algoritmo di Welford
per l'aggiornamento online delle metriche (standard e time-weighted),
mantenendo nascosta la logica interna del modulo welford.py.
"""

from .welford import StandardWelford, TimeWeightedWelford

__all__ = [
    "StandardWelford",
    "TimeWeightedWelford"
]