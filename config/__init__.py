# File: config/__init__.py
"""
Package per la gestione della configurazione architetturale.

Espone l'oggetto Config che esegue il parsing e la validazione dei parametri,
incapsulando la logica di lettura dal file system definita in settings.py.
"""

from .settings import Config

__all__ = [
    "Config"
]