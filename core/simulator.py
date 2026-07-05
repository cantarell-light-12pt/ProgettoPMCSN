# ==========================================
# FILE: core/simulator.py
# ==========================================
"""
Core engine DES per la gestione degli eventi stocastici,
privo di collisioni temporali e stalli logici.

Il generatore pseudo-casuale e' il Lehmer multi-stream di Leemis & Park
(package rng): ogni sorgente stocastica del modello usa uno stream dedicato,
per garantire indipendenza e riproducibilita'. I seed vanno piantati una sola
volta dal driver tramite rng.rngs.PlantSeeds; il simulatore NON li reimposta,
cosi repliche successive consumano gli stream in sequenza (repliche indipendenti).
"""
import heapq
from collections import deque
from typing import List, Tuple, Any, Dict

from rng import rngs
from rng import rvgs
from core.models import Job
from config.settings import Config
from stats.welford import StandardWelford, TimeWeightedWelford

# Assegnazione di uno stream indipendente a ciascuna sorgente stocastica.
STREAM_ARRIVAL: int = 0   # inter-arrivo esogeno (Exponential)
STREAM_BATCH: int = 1     # dimensione del batch (Empirical)
STREAM_FATE: int = 2      # esito del job: passed/failed/errored/canceled (Uniform)
STREAM_SETUP: int = 3     # tempo di setup (Empirical)
STREAM_TEST: int = 4      # tempo di test (Lognormal)
STREAM_PCUT: int = 5      # troncamento anticipato p_cut (Uniform)
STREAM_DELAY: int = 6     # delay di feedback umano (Lognormal)
STREAM_RETRY: int = 7     # decisione di retry (Bernoulli)
STREAM_ERR_MODE: int = 8  # modo di errore: precoce (setup) vs tardivo (build) (Bernoulli)
STREAM_FAIL_SVC: int = 9  # servizio dei job failed (Lognormal calibrata)
STREAM_CANC_SVC: int = 10 # servizio dei job canceled (Lognormal calibrata)


class TravisCISimulator:
    """
    Rete di code M^[X]/G/c implementata tramite Event Calendar e heapq.

    Ogni istanza rappresenta una singola replica: stato, calendario eventi e
    stimatori Welford sono reinizializzati nel costruttore, mentre gli stream
    del generatore (globali) proseguono dalla replica precedente.
    """
    def __init__(self, config: Config) -> None:
        """
        Inizializza lo stato del simulatore, l'Event Calendar, il tracking
        metriche Welford e i buffer delle tracce.

        Args:
            config (Config): Impostazioni di sistema validate lette dal file esterno.
        """
        self.config = config

        self.time: float = 0.0
        self.last_event_time: float = 0.0
        self.events: List[Tuple[float, int, str, Any]] = []
        self.event_counter: int = 0

        self.queue: deque = deque()
        self.servers_busy: int = 0
        self.jobs_in_feedback: int = 0
        self.job_id_counter: int = 0

        # Welford tracking persistente e per-job
        self.stat_utilization = TimeWeightedWelford()
        self.stat_qlen = TimeWeightedWelford()
        self.stat_syslen = TimeWeightedWelford()

        self.stat_service_time = StandardWelford()
        self.stat_wait_time = StandardWelford()
        self.stat_response_time = StandardWelford()

        # Tracce: stato campionato su griglia fissa e osservazioni per-job.
        self.state_trace: List[Tuple[float, float, int, int, int]] = []
        self.job_trace: List[Tuple[int, float, float, float, float]] = []
        # Traccia a livello di nodo: una riga per ogni completamento di servizio
        # (END_SERVICE), che copre TUTTI i job/esiti anche quelli poi bloccati nel
        # feedback. Serve a Verifica (legge di Little per-visita) e Validazione (esiti,
        # tempo di servizio). Righe: (completion_time, fate, wait_visita, service_visita).
        self.node_trace: List[Tuple[float, str, float, float]] = []
        # Traccia degli arrivi esogeni: una riga per evento ARRIVAL (Matrix Build).
        # Serve alla Validazione per misurare DALLA SIMULAZIONE la PMF dei batch e il
        # tasso di arrivo (dati realizzati, non parametri di input).
        self.batch_trace: List[Tuple[float, int]] = []
        self.next_sample_time: float = self.config.sampling_interval

    def _schedule(self, event_time: float, event_type: str, *args) -> None:
        """
        Inserisce un evento nel calendario rispettando l'ordinamento temporale.
        """
        heapq.heappush(self.events, (event_time, self.event_counter, event_type, args))
        self.event_counter += 1

    def _emit_samples(self, up_to: float) -> None:
        """
        Registra i campioni di stato sulla griglia temporale fissa che cadono in
        (next_sample_time, up_to]. Lo stato del sistema e' costante tra due eventi
        consecutivi, quindi ogni campione riflette lo stato corrente pre-evento.
        Questi campioni alimentano la media cumulata del transitorio e i batch means.
        """
        while self.next_sample_time <= up_to:
            utilization = self.servers_busy / self.config.c
            self.state_trace.append((
                self.next_sample_time,
                utilization,
                len(self.queue),
                self.servers_busy,
                self.jobs_in_feedback,
            ))
            self.next_sample_time += self.config.sampling_interval

    def _update_time_stats(self, current_time: float) -> None:
        """
        Processa il delta-temporale (dt) tra l'evento precedente e quello attuale
        per calcolare gli stati continui tramite Welford Time-Weighted.
        """
        dt = current_time - self.last_event_time
        if dt > 0:
            self.stat_utilization.update(self.servers_busy / self.config.c, dt)
            self.stat_qlen.update(len(self.queue), dt)
            sys_pop = len(self.queue) + self.servers_busy
            self.stat_syslen.update(sys_pop, dt)
        self.last_event_time = current_time

    def _calculate_service(self) -> Tuple[float, str]:
        """
        Determina il destino finale del task e modula l'interruzione anticipata
        usando un Rischio Competitivo basato su variabile Uniforme.
        """
        rngs.SelectStream(STREAM_FATE)
        rand_fate = rvgs.Uniform(0.0, 1.0)
        if rand_fate <= self.config.p_passed:
            fate = "passed"
        elif rand_fate <= self.config.p_passed + self.config.p_failed:
            fate = "failed"
        elif rand_fate <= self.config.p_passed + self.config.p_failed + self.config.p_errored:
            fate = "errored"
        else:
            fate = "canceled"

        rngs.SelectStream(STREAM_SETUP)
        t_setup = rvgs.Empirical(self.config.pmf_setup_values, self.config.pmf_setup_probs)
        rngs.SelectStream(STREAM_TEST)
        t_test = rvgs.Lognormal(self.config.mu_test, self.config.sigma_test)
        rngs.SelectStream(STREAM_PCUT)
        p_cut = rvgs.Uniform(0.0, 1.0)

        if fate == "errored":
            # Errored bimodale: con prob p_errored_early rompe PRESTO (durante il setup,
            # servizio corto); altrimenti ha girato l'intero build e poi e' andato in
            # errore (servizio ~ come un passed). Riproduce la coda lunga reale.
            rngs.SelectStream(STREAM_ERR_MODE)
            if rvgs.Bernoulli(self.config.p_errored_early) == 1:
                return p_cut * t_setup, fate
            return t_setup + t_test, fate
        elif fate == "passed":
            return t_setup + t_test, fate
        elif fate == "failed":
            # Servizio calibrato sulle buildduration reali dei failed (girano quasi tutto
            # il build, non meta' come col troncamento p_cut).
            rngs.SelectStream(STREAM_FAIL_SVC)
            return t_setup + rvgs.Lognormal(self.config.mu_failed, self.config.sigma_failed), fate
        else:  # canceled
            rngs.SelectStream(STREAM_CANC_SVC)
            return t_setup + rvgs.Lognormal(self.config.mu_canceled, self.config.sigma_canceled), fate

    def _try_start_service(self) -> None:
        """
        Tenta di avviare l'elaborazione dei job in coda se vi sono runners disponibili.
        Previene i colli di bottiglia logici e i deadlock.
        """
        while self.queue and self.servers_busy < self.config.c:
            job: Job = self.queue.popleft()
            job.total_wait_time += (self.time - job.queue_enter_time)

            self.servers_busy += 1
            job.service_start_time = self.time

            service_duration, fate = self._calculate_service()
            job.current_fate = fate

            self._schedule(self.time + service_duration, "END_SERVICE", job)

    def _job_exit(self, job: Job) -> None:
        """
        Smaltisce definitivamente un job calcolando metriche Standard Welford
        e registrando l'osservazione nella traccia per-job.
        """
        response_time = self.time - job.creation_time
        self.stat_response_time.update(response_time)
        self.stat_service_time.update(job.total_service_time)
        self.stat_wait_time.update(job.total_wait_time)

        self.job_trace.append((
            job.id,
            self.time,
            job.total_wait_time,
            job.total_service_time,
            response_time,
        ))

    def _handle_arrival(self) -> None:
        """
        Elabora l'arrivo esogeno (Matrix Build) e schedula l'arrivo futuro.
        """
        rngs.SelectStream(STREAM_BATCH)
        batch_size = int(rvgs.Empirical(self.config.pmf_batch_values, self.config.pmf_batch_probs))
        self.batch_trace.append((self.time, batch_size))
        for _ in range(batch_size):
            self.job_id_counter += 1
            job = Job(id=self.job_id_counter, creation_time=self.time)
            job.queue_enter_time = self.time
            self.queue.append(job)

        self._try_start_service()
        rngs.SelectStream(STREAM_ARRIVAL)
        inter_arrival = rvgs.Exponential(1.0 / self.config.lambda_ext)
        self._schedule(self.time + inter_arrival, "ARRIVAL")

    def _handle_end_service(self, job: Job) -> None:
        """
        Libera il server. Instrada i job rotti nel feedback loop e quelli buoni all'uscita.
        Converte esplicitamente il tempo Lognormale da minuti a secondi.
        """
        # Traccia di nodo: il job lascia il nodo (coda+serventi) a questo END_SERVICE.
        # Attesa e servizio di QUESTA visita (queue_enter_time non e' ancora stato
        # sovrascritto da un eventuale re-inserimento in coda post-feedback).
        wait_visit = job.service_start_time - job.queue_enter_time
        service_visit = self.time - job.service_start_time
        self.node_trace.append((self.time, job.current_fate, wait_visit, service_visit))

        self.servers_busy -= 1
        job.total_service_time += (self.time - job.service_start_time)

        if job.current_fate in ("passed", "canceled"):
            self._job_exit(job)
        else:
            self.jobs_in_feedback += 1

            # La distribuzione Lognormale restituisce un valore temporale espresso in minuti
            rngs.SelectStream(STREAM_DELAY)
            delay_minutes = rvgs.Lognormal(self.config.mu_delay, self.config.sigma_delay)

            # Normalizzazione al dominio temporale del motore DES (secondi)
            delay_seconds = delay_minutes * 60.0

            self._schedule(self.time + delay_seconds, "END_FEEDBACK", job)

        self._try_start_service()

    def _handle_end_feedback(self, job: Job) -> None:
        """
        Processa il termine della verifica log da parte dell'operatore umano.
        """
        self.jobs_in_feedback -= 1
        retry_prob = self.config.p_retry_failed if job.current_fate == "failed" else self.config.p_retry_errored

        rngs.SelectStream(STREAM_RETRY)
        if rvgs.Bernoulli(retry_prob) == 1:
            job.queue_enter_time = self.time
            self.queue.append(job)
            self._try_start_service()
        else:
            self._job_exit(job)

    def run(self) -> None:
        """
        Esegue il Master Event Loop avanzando nel tempo finche' il limite non
        viene raggiunto, campionando lo stato sulla griglia fissa.
        """
        rngs.SelectStream(STREAM_ARRIVAL)
        inter_arrival = rvgs.Exponential(1.0 / self.config.lambda_ext)
        self._schedule(inter_arrival, "ARRIVAL")

        while self.events:
            ev_time, _, ev_type, args = heapq.heappop(self.events)

            if ev_time > self.config.max_time:
                self._emit_samples(self.config.max_time)
                self._update_time_stats(self.config.max_time)
                self.time = self.config.max_time
                break

            self._emit_samples(ev_time)
            self._update_time_stats(ev_time)
            self.time = ev_time

            if ev_type == "ARRIVAL":
                self._handle_arrival()
            elif ev_type == "END_SERVICE":
                self._handle_end_service(*args)
            elif ev_type == "END_FEEDBACK":
                self._handle_end_feedback(*args)

    def results(self) -> Dict[str, float]:
        """
        Restituisce le metriche finali stimate da Welford per questa replica.
        """
        return {
            "utilization": self.stat_utilization.mean,
            "service_time": self.stat_service_time.mean,
            "wait_time": self.stat_wait_time.mean,
            "response_time": self.stat_response_time.mean,
            "qlen": self.stat_qlen.mean,
            "syslen": self.stat_syslen.mean,
        }

    def print_welford_metrics(self) -> None:
        """
        Visualizza a standard output i KPI tracciati da Welford in secondi o entità.
        """
        print("=== RISULTATI SIMULAZIONE ===")
        print(f"-> Utilizzazione media pool (c={self.config.c}): {self.stat_utilization.mean:.4f}")
        print(f"-> Tempo medio di servizio (per job):   {self.stat_service_time.mean:.2f} s")
        print(f"-> Tempo medio attesa in coda:          {self.stat_wait_time.mean:.2f} s")
        print(f"-> Tempo medio di risposta sistema:     {self.stat_response_time.mean:.2f} s")
        print(f"-> Popolazione media in coda:           {self.stat_qlen.mean:.2f} jobs")
        print(f"-> Popolazione media nel sistema:       {self.stat_syslen.mean:.2f} jobs")
