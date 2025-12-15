import uuid
import heapq
import functools
import threading
import operator
from typing import Sequence, Any, TYPE_CHECKING

from qiskit.providers import JobV1 as QiskitJob
from qiskit.circuit import QuantumCircuit as QiskitCircuit

from ..util import Log
from ..base import Exceptions
from ..util.translation import CircuitBackendTranslations
from ..interfaces import Circuit, Backend

if TYPE_CHECKING:
    from ..parallelizer import ParallelizedBackend
    from ..backends import BackendCircuitBin

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

class ParallelizerJob:
    """
    A single parallelized circuit execution.
    """

    def __init__(self, backend: "ParallelizedBackend", circuit: Circuit):
        self.id = uuid.uuid4()
        self.backend = backend
        self.circuit = circuit
        self.original_circuit = circuit # for reference
        self.completion_requested: bool = False
        self.completed = threading.Event()
        self.remote_job: QiskitJob | None = None
        self.remote_backend: Backend | None = None
        self.counts: dict[str, int] | None = None

    @functools.cached_property
    def translations(self) -> CircuitBackendTranslations:
        return CircuitBackendTranslations.generate(
            self.circuit,
            self.backend.remote_backends,
        )

    def _find_bin_layout(self):
        Log.debug(
            f"![FINDING LAYOUT] Determining layout for |{self.circuit.num_qubits}-qubit| "
            "circuit."
        )

        optimal_backends = self.translations.optimal_backends_for(self.circuit)
        translations = {
            backend: self.translations.get(self.circuit, backend)
            for backend in optimal_backends
        }

        candidate_placements: list[tuple[Any, "BackendCircuitBin", Circuit]] = []
        max_candidates = self.backend.packer.max_candidates or float("inf")

        for candidate_bin in self.backend.manager.best_bins(translations):
            Log.debug((
                f"Trying bin on backend |'{candidate_bin.backend.name}'| with "
                f"${candidate_bin.size} circuit$."
            ))

            blocked = self.backend.packer.blocked(candidate_bin)
            translated = translations[candidate_bin.backend]
            completed_layout = self.backend.packer.find_layout(
                candidate_bin,
                translated,
                blocked,
            )
            if completed_layout is not None:
                completed_circuit = translated.with_layout(completed_layout)
                rating = self.backend.packer.evaluate(candidate_bin, completed_circuit)
                Log.debug(f"Received rating |{rating}| for placement candidate.")
                heapq.heappush(candidate_placements, (-rating, candidate_bin, completed_circuit))
                if len(candidate_placements) >= max_candidates:
                    break

        if len(candidate_placements) > 0:
            Log.debug("![FOUND LAYOUT] Bin and layout found for circuit.")
            _, bin, circuit = candidate_placements[0]
            return bin, circuit

        Log.fail("No suitable bins found for circuit!")
        raise Exceptions.CircuitBackendCompatibility("circuit could not be placed on any backend")

    def place(self):
        bin, self.circuit = self._find_bin_layout()
        bin.place(self)
        Log.debug((
            f"|{self.circuit.num_qubits}-qubit| circuit translated and placed on backend "
            f"|'{bin.backend.name}'|."
        ))

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

    def request_completion(self):
        """
        Request this job to finish as soon as possible. Effectively, forces job submission even if
        the parallelization is not optimal. If the job has already completed, does nothing.
        """

        if self.is_ready:
            return
        self.completion_requested = True
        self.backend.manager.tick()

    def result(self, block: bool = True):
        """
        Retrieves the result of this job, blocking until it becomes available. Alternatively, if
        `block` is set to False, returns None immediately if there are no results yet.
        """

        if not self.is_ready:
            self.request_completion()
            if block:
                Log.debug(f"Blocking until job completion.")
                self.completed.wait()
        if not self.is_ready:
            return JobResult.empty()
        assert self.counts is not None
        return JobResult(self.counts)

    def mark_submitted(self, remote_backend: Backend, remote_job: QiskitJob):
        self.remote_backend = remote_backend
        self.remote_job = remote_job

    def mark_completed(self, counts: dict[str, int]):
        self.counts = counts
        self.completed.set()

    def __hash__(self):
        return hash(self.id)

class ParallelizerJobBatch:
    """
    A batch of parallelized circuit executions. Not necessarily executed together - the contained
    jobs were just submitted together by user code.
    """

    def __init__(self, jobs: Sequence[ParallelizerJob]):
        self.jobs = list(jobs)

    def place_all(self, sort: bool = True):
        jobs = sorted(self.jobs, key=lambda job: (
            -job.circuit.layout.size,
            job.circuit.num_connected_components / job.circuit.num_qubits,
            -job.circuit.num_qubits,
        )) if sort else self.jobs

        for job in jobs:
            Log.debug((
                f"Placing circuit with ${job.circuit.num_qubits} qubit$ and layout of size "
                f"|{job.circuit.layout.size}|."
            ))
            job.place()

    def result(self, block: bool = True):
        """
        Retrives the result(s) of this batch, blocking until they are available. Alternatively, if
        `block` is set to False, returns some or no results immediately depending on availability.
        The returned object is effectively a snapshot, so if some results are not available, this
        method must be called again at a later time.
        """

        if block:
            for job in self.jobs:
                job.request_completion()
        return BatchedJobResult([job.result(block) for job in self.jobs])

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

    def draw(self, figsize=None):
        """
        Plots this job's chosen layout(s) on the backend(s). Requires Matplotlib to be installed.

        Returns:
            A Matplotlib Figure. If used in a notebook, this function takes care of closing unwanted
            duplicates that may be displayed automatically.
        """

        try:
            import matplotlib.pyplot
            import qiskit.visualization.utils
        except ImportError as exc:
            raise RuntimeError("missing optional dependencies") from exc

        relevant_bins = {
            bin
            for job in self.jobs
            for bin in job.backend.manager.bins
        }
        job_bins = {
            bin: [job for job in self.jobs if job in bin]
            for bin in relevant_bins
            if not bin.is_empty
        }

        fig, axs = matplotlib.pyplot.subplots(
            ncols=len(job_bins),
            dpi=100,
            figsize=figsize,
            squeeze=False,
        )

        def get_qubit_colors(qubit_indices, num_qubits):
            indices = [None] * num_qubits
            for i, qubits in enumerate(qubit_indices):
                for q in qubits:
                    indices[q] = i
            color_table = ["#dd0000", "#008800", "#0000bb", "#0099aa", "#aa00aa", "#aa9900"]
            return [color_table[i % 6] if i is not None else "grey" for i in indices]

        def get_coupler_colors(circuit_couplers, backend_couplers):
            return ["black" if (a, b) in circuit_couplers else "grey" for a, b in backend_couplers]

        for i, (bin, jobs) in enumerate(job_bins.items()):
            qubit_indices = [job.layout.pindices for job in jobs]
            all_couplers = [
                edge
                for job in jobs
                for edge in (
                    (job.layout.v2p[a], job.layout.v2p[b])
                    for a, b in job.circuit.get_edges()
                )
            ]
            qubit_colors = get_qubit_colors(qubit_indices, bin.backend.num_qubits)
            coupler_colors = get_coupler_colors(all_couplers, bin.backend.edges)
            qiskit.visualization.plot_coupling_map(
                num_qubits=bin.backend.num_qubits,
                qubit_coordinates=None,
                coupling_map=bin.backend.edges,
                ax=axs[0, i],
                planar=False,
                qubit_size=80,
                font_size=24,
                line_width=25,
                qubit_color=qubit_colors,
                line_color=coupler_colors,
            )

        matplotlib.pyplot.tight_layout()
        qiskit.visualization.utils.matplotlib_close_if_inline(fig)
        return fig

    def __getitem__(self, index: int | slice | Circuit | QiskitCircuit):
        if isinstance(index, int | slice):
            return self.jobs[index]
        if isinstance(index, QiskitCircuit):
            index = Circuit(index)
        if isinstance(index, Circuit):
            try:
                return next(
                    job for job in self.jobs
                    if job.original_circuit.hash() == index.hash()
                )
            except StopIteration as exc:
                raise LookupError("the given circuit was not found in this job batch") from exc
        raise TypeError(f"'{index}' is not a valid index into a job batch")
