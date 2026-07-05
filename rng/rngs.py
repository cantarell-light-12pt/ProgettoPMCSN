"""
Generatore di Lehmer multi-stream (Park & Miller / Leemis & Park).

Porting Python autonomo di 'rngs.c'. Espone un'unica istanza globale di stato
(come la libreria C originale) con 256 stream indipendenti generati per
"jumping" a partire da un seed piantato una sola volta.

Costanti standard del libro:
    modulo   m = 2147483647   (2^31 - 1, primo di Mersenne)
    moltipl. a = 48271
    jump     A = 22925        (per generare i seed dei 256 stream)
"""
from typing import List

MODULUS: int = 2147483647
MULTIPLIER: int = 48271
CHECK: int = 399268537
STREAMS: int = 256
A256: int = 22925          # jump multiplier: A256 = MULTIPLIER^(m/STREAMS) mod m
DEFAULT: int = 123456789   # seed di default iniziale

# Stato globale: un seed corrente per ciascuno dei 256 stream.
_seed: List[int] = [DEFAULT] * STREAMS
_stream: int = 0
_initialized: bool = False


def Random() -> float:
    """
    Restituisce un numero pseudo-casuale distribuito uniformemente in (0, 1),
    avanzando lo stato dello stream attualmente selezionato.
    Usa la scomposizione di Schrage per evitare overflow (fedele all'originale C).
    """
    global _seed
    Q: int = MODULUS // MULTIPLIER
    R: int = MODULUS % MULTIPLIER

    t: int = MULTIPLIER * (_seed[_stream] % Q) - R * (_seed[_stream] // Q)
    if t > 0:
        _seed[_stream] = t
    else:
        _seed[_stream] = t + MODULUS
    return _seed[_stream] / MODULUS


def SelectStream(index: int) -> None:
    """Seleziona lo stream attivo (0..STREAMS-1) per le successive chiamate a Random()."""
    global _stream
    _stream = index % STREAMS


def PutSeed(seed: int) -> None:
    """
    Imposta il seed dello stream corrente. Un valore negativo usa l'orologio di
    sistema; 0 usa il seed di default. Comportamento fedele a rngs.c.
    """
    global _seed
    if seed < 0:
        import time
        seed = int(time.time()) % MODULUS
    elif seed == 0:
        seed = DEFAULT
    _seed[_stream] = seed % MODULUS


def GetSeed() -> int:
    """Restituisce il seed corrente dello stream attivo."""
    return _seed[_stream]


def PlantSeeds(seed: int) -> None:
    """
    Inizializza tutti i 256 stream a partire da un unico seed, distanziandoli
    con il moltiplicatore di jump A256. Da chiamare una sola volta.
    """
    global _seed, _stream, _initialized
    Q: int = MODULUS // A256
    R: int = MODULUS % A256

    saved_stream: int = _stream
    _stream = 0
    PutSeed(seed)
    _initialized = True
    x0: int = _seed[0]

    for j in range(1, STREAMS):
        x: int = A256 * (x0 % Q) - R * (x0 // Q)
        if x > 0:
            x0 = x
        else:
            x0 = x + MODULUS
        _seed[j] = x0

    _stream = saved_stream


def is_initialized() -> bool:
    """True se PlantSeeds e' stato invocato almeno una volta."""
    return _initialized
