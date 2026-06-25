import functools
import heapq
import operator
import tempfile
import threading
import time
import typing
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from qiskit import qpy
from qiskit.circuit import QuantumCircuit as QiskitCircuit
from qiskit.providers import JobV1 as QiskitJob

from ..base import Exceptions
from ..interfaces import Backend, Circuit
from ..util import Log
from ..util.translation import CircuitBackendTranslations
from ..util.visualization import plot_job_batch

if TYPE_CHECKING:
    from ..backends import BackendCircuitBin
    from ..parallelizer import ParallelizedBackend


class JobResult:
    def __init__(self, counts: dict[str, int] | None):
        self.completed = counts is not None
        self.counts = counts

    def get_counts(self):
        return self.counts

    @classmethod
    def empty(cls):
        return cls(None)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.counts)})"


class BatchedJobResult:
    def __init__(self, results: Sequence[JobResult]):
        self.results = results

    def get_counts(self, index: int | None = None):
        if index is None:
            return [result.counts for result in self.results]
        return self.results[index].counts

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __repr__(self):
        return repr(self.results)


class JobTiming:
    created: float
    placed: float | None
    submitted: float | None
    completed: float | None
    requested: float | None

    def __init__(self):
        self.created = time.time()
        self.placed = None
        self.submitted = None
        self.completed = None
        self.requested = None

    @property
    def as_tuple(self):
        return self.created, self.placed, self.submitted, self.completed, self.requested


class ParallelizerJob:
    """
    A single parallelized circuit execution.
    """

    def __init__(
        self,
        backend: "ParallelizedBackend",
        circuit: Circuit,
    ):
        self.id = uuid.uuid4()
        self.backend = backend
        self.circuit = circuit
        # Take a second reference of the original circuit for comparisons later
        self.original_circuit = circuit
        self.bin: BackendCircuitBin | None = None
        self.completion_requested: bool = False
        self.completed = threading.Event()
        self.cancelled: bool = False
        self.remote_job: QiskitJob | None = None
        self.remote_backend: Backend | None = None
        self.counts: dict[str, int] | None = None
        self.timing = JobTiming()

    @functools.cached_property
    def translations(self) -> CircuitBackendTranslations:
        return CircuitBackendTranslations.generate(
            self.circuit,
            self.backend.remote_backends,
            **self.backend.parent.translation_kwargs,
        )

    def _find_bin_layout(self, kwarg_contraints: dict[str, Any] = {}):
        Log.debug(
            f"![FINDING LAYOUT] Determining layout for |{self.circuit.num_qubits}-qubit| "
            "circuit.",
        )

        optimal_backends = self.translations.optimal_backends_for(self.circuit)
        translations = {
            backend: self.translations.get(self.circuit, backend) for backend in optimal_backends
        }

        candidate_placements: list[tuple[Any, "BackendCircuitBin", Circuit]] = []
        max_candidates = self.backend.packer.max_candidates or float("inf")

        for candidate_bin in self.backend.manager.best_bins(translations, kwarg_contraints):
            Log.debug(
                (
                    f"Trying bin on backend |'{candidate_bin.backend.name}'| with "
                    f"${candidate_bin.size} circuit$."
                ),
            )

            blocked = self.backend.packer.blocked(candidate_bin)
            translated = translations[candidate_bin.backend]
            completed_layout = self.backend.packer.find_layout(
                candidate_bin,
                translated,
                blocked,
            )
            if completed_layout is not None:
                completed_circuit = translated.with_layout(completed_layout)
                cost = self.backend.packer.evaluate(candidate_bin, completed_circuit)
                key = (
                    # makes currently empty bins inferior
                    candidate_bin.is_empty,
                    # actual cost, sorted as descending
                    -cost,
                    # and a dummy tie breaker
                    len(candidate_placements),
                )
                Log.debug(f"Cost of placement candidate is |{cost}|.")
                heapq.heappush(
                    candidate_placements,
                    (key, candidate_bin, completed_circuit),
                )
                if len(candidate_placements) >= max_candidates:
                    Log.debug(f"Maximum candidate limit (|{max_candidates}|) reached.")
                    break
        else:
            Log.debug("All bin candidates exhausted.")

        Log.debug(f"Candidates (${len(candidate_placements)} item$):")
        for key, bin, _ in candidate_placements:
            Log.debug(f" - Candidate with key |{key}| in bin with ${bin.size} circuit$.")

        if len(candidate_placements) > 0:
            key, bin, circuit = candidate_placements[0]
            Log.debug(f"![FOUND LAYOUT] Bin and layout with key |{key}| found for circuit.")
            return bin, circuit

        Log.fail("No suitable bins found for circuit:")
        Log.fail(lambda: f"{self.original_circuit.unwrap().draw(fold=-1, idle_wires=False)}")

        circuit_file = tempfile.NamedTemporaryFile("wb", delete=False)
        qpy.dump(self.original_circuit.unwrap(), circuit_file)  # type: ignore
        circuit_file.close()
        Log.info(f"Circuit serialized to |{circuit_file.name}|.")

        raise Exceptions.CircuitBackendCompatibility("circuit could not be placed on any backend")

    def place(self, kwargs: dict[str, Any] = {}):
        with self.backend.manager.lock:
            assert not self.is_placed, "circuit was attempted to be placed twice"
            bin, self.circuit = self._find_bin_layout(kwargs)
            bin.place(self, kwargs)
            self.bin = bin
            self.timing.placed = time.time()
            Log.debug(
                (
                    f"|{self.circuit.num_qubits}-qubit| circuit translated and placed on backend "
                    f"|'{bin.backend.name}'|."
                ),
            )

    @property
    def is_placed(self):
        return self.bin is not None

    @property
    def is_submitted(self):
        return self.remote_job is not None

    @property
    def is_ready(self):
        return self.completed.is_set()

    @property
    def layout(self):
        return self.circuit.layout

    @property
    def metadata(self):
        return self.circuit.metadata

    def cancel(self):
        if self.is_ready:
            return
        self.cancelled = True
        if self.remote_job is not None:
            self.remote_job.cancel()
        # TODO: remove from bin if the job has not been submitted yet?

    def request_completion(self):
        """
        Request this job to finish as soon as possible. Effectively, forces job submission even if
        the parallelization is not optimal. If the job has already completed, does nothing.
        """

        if self.timing.requested is None:
            self.timing.requested = time.time()
        if self.is_ready:
            return
        Log.info("Job completion requested!")
        self.completion_requested = True
        self.backend.manager.tick()

    def result(self, block: bool = True, request_completion: bool = True):
        """
        Retrieves the result of this job, blocking until it becomes available. Alternatively, if
        `block` is set to False, returns None immediately if there are no results yet.
        """

        if not self.is_ready and request_completion:
            self.request_completion()
            if block:
                Log.debug("Blocking until job completion.")
                self.completed.wait()
                Log.debug("Blocking wait finished!")
        if not self.is_ready:
            return JobResult.empty()
        assert self.counts is not None
        return JobResult(self.counts)

    def mark_submitted(
        self,
        remote_backend: Backend,
        remote_job: QiskitJob,
        at_time: float | None = None,
    ):
        self.timing.submitted = at_time or time.time()
        self.remote_backend = remote_backend
        self.remote_job = remote_job

    def mark_completed(self, counts: dict[str, int], at_time: float | None = None):
        self.timing.completed = at_time or time.time()
        self.counts = counts
        self.completed.set()

    def __hash__(self):
        return hash(self.id)


class ParallelizerJobBatch:
    """
    A batch of parallelized circuit executions. Not necessarily executed together - the contained
    jobs were just submitted together by user code.
    """

    def __init__(
        self,
        jobs: Sequence[ParallelizerJob],
        kwargs: dict[str, Any],
        sync_wait: float = 0.2,
    ):
        self.jobs = list(jobs)
        self.kwargs = kwargs
        self.sync_wait = sync_wait

    def place_all(self, sort: bool = True):
        """
        Places all circuits in this batch onto available backends.

        Args:
            sort:
                If True (default), circuits will be heuristically sorted to achieve a more optimal
                packing. Currently, this means sorting by

                 1. descending layout size (number of qubits with defined layout),
                 2. ascending number of connected components in the circuit topology, divided by the
                    number of qubits for a proportional metric, and
                 3. descending number of qubits,

                with priority in the listed order.
        """

        if sort:
            jobs = sorted(
                self.jobs,
                key=lambda job: (
                    -job.circuit.layout.size,
                    job.circuit.num_connected_components / job.circuit.num_qubits,
                    -job.circuit.num_qubits,
                ),
            )
        else:
            jobs = self.jobs

        for job in jobs:
            Log.debug(
                (
                    f"Placing circuit with ${job.circuit.num_qubits} qubit$ and layout of size "
                    f"|{job.circuit.layout.size}|."
                ),
            )
            job.place(self.kwargs)

    def request_completion(self):
        """
        Requests for all jobs in this batch to complete as soon as possible.
        """

        for job in self.jobs:
            job.request_completion()

    def result(self, block: bool = True, sync_wait: float | None = None):
        """
        Retrives the result(s) of this batch, blocking until they are available. Alternatively, if
        `block` is set to False, returns some or no results immediately depending on availability.
        The returned object is effectively a snapshot, so if some results are not available, this
        method must be called again at a later time.

        Args:
            block:
                If True, blocks until results are ready. If False, return (possibly) partial
                results immediately.
            sync_wait:
                Time to wait (in seconds) before requesting for results from the underlying
                backends. In multi-threaded applications, this can be very beneficial if several
                threads submit jobs and request results in parallel. Leaving this as None will use
                a shared pre-set value for the whole parallelized backend.
        """

        for job in self.jobs:
            if job.timing.requested is None:
                job.timing.requested = time.time()
        if not self.is_ready:
            time.sleep(sync_wait or self.sync_wait)
        if block:
            for job in self.jobs:
                job.request_completion()
        return BatchedJobResult([job.result(block, request_completion=block) for job in self.jobs])

    @property
    def is_ready(self):
        return all(job.is_ready for job in self.jobs)

    @property
    def id(self):
        """
        A pseudo-id for this batch, computed as a combination of the contained jobs ids.
        """

        return uuid.UUID(
            int=functools.reduce(operator.xor, (job.id.int for job in self.jobs)),
            version=4,
        )

    @property
    def remote_ids(self):
        return {
            job: job.remote_job.job_id() if job.remote_job is not None else None
            for job in self.jobs
        }

    def cancel(self):
        for job in self.jobs:
            job.cancel()

    def draw(self, *args, **kwargs):
        """
        Plots this job's chosen layout(s) on the backend(s). Requires Matplotlib to be installed.
        See `qc_parallelizer.util.visualization.plot_job_batch` for details.
        """

        return plot_job_batch(self, *args, **kwargs)

    def __getitem__(self, index: int | slice | Circuit | QiskitCircuit):
        if isinstance(index, int | slice):
            return self.jobs[index]
        if isinstance(index, QiskitCircuit):
            index = Circuit(index)
        if isinstance(index, Circuit):
            try:
                return next(job for job in self.jobs if job.original_circuit.hash() == index.hash())
            except StopIteration as exc:
                raise LookupError("the given circuit was not found in this job batch") from exc
        raise TypeError(f"'{index}' is not a valid index into a job batch")

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self):
        return len(self.jobs)

    def __or__(self, other: "ParallelizerJobBatch"):
        return ParallelizerJobBatch(
            self.jobs + other.jobs,
            self.kwargs | other.kwargs,
            max(self.sync_wait, other.sync_wait),
        )
