import functools
import itertools
import operator
from collections import Counter
from collections.abc import Sequence

import qiskit
import qiskit.providers
import qiskit.result

from . import binmanager, packers, postprocessing
from .base import Exceptions, Types
from .generic import circuittools
from .generic.layouts import CircuitWithLayout, IndexedLayout
from .generic.logging import Log


class ParallelJob:
    """
    A wrapper class for parallel execution jobs. This is intentionally not a subclass of the actual
    `Job` class since it merely tracks the underlying `Job` instances - however, in addition to the
    extended functionality for parallelization, it also provides a proxy interface into the
    underlying `Job` object that has the same attributes and methods. Note that the returned values
    are array-like objects (you may index or iterate them), since there can be multiple jobs
    involved - for instance, to get the first job ID, you can use `parallel_job.job_id()[0]`.

    Users should not construct instances of this class themselves since the post-processor expects
    certain fixed structure in the circuit metadata. Instead, instances of this class are returned
    by the `execute()` function.
    """

    class ParallelizationInfo:
        """
        A metadata class for information on how execution was parallelized.
        """

        def __init__(
            self,
            jobs: Sequence[tuple[Types.Backend, qiskit.QuantumCircuit, qiskit.providers.JobV1]],
            circuit_dict: dict[Types.Backend, Sequence[qiskit.QuantumCircuit]],
        ):
            self.jobs = jobs
            self.backends = list(circuit_dict.keys())
            self.circuits = circuit_dict

        @property
        def num_backends(self) -> int:
            return len(self.backends)

        @property
        def num_circuits(self) -> int:
            return sum(len(circuit_list) for circuit_list in self.circuits.values())

        @property
        def min_circuits_per_backend(self) -> int:
            return min(len(circuit_list) for circuit_list in self.circuits.values())

        @property
        def max_circuits_per_backend(self) -> int:
            return max(len(circuit_list) for circuit_list in self.circuits.values())

        @property
        def avg_circuits_per_backend(self) -> float:
            return self.num_circuits / self.num_backends

        @property
        def num_jobs(self) -> int | None:
            if self.jobs is None:
                return None
            return len(self.jobs)

        def for_circuit(
            self,
            circuit: qiskit.QuantumCircuit,
        ) -> tuple[qiskit.providers.Job, Types.Backend]:
            for backend, host_circuit, job in self.jobs:
                for hosted_circuit in host_circuit.metadata["_hosted_circuits"]:
                    if hosted_circuit["name"] == circuit.name:
                        return (job, backend)
            raise Exceptions.MissingInformation("the given circuit was not a part of this job")

    def __init__(
        self,
        jobs: Sequence[tuple[Types.Backend, qiskit.QuantumCircuit, qiskit.providers.JobV1]],
        circuits: dict[Types.Backend, Sequence[qiskit.QuantumCircuit]],
    ):
        self._jobs = jobs
        self._circuits = circuits
        self._raw_results: Sequence[qiskit.result.result.Result] | None = None
        self._results: Sequence[Types.Result] | None = None

    def _fetch_results(self):
        if self._raw_results is None:
            # `job.result()` is abstract and has no type, so we ignore type checking here
            self._raw_results = [job.result() for _, _, job in self._jobs]  # type: ignore
        if self._results is None:
            Log.debug(f"Got |{len(self._raw_results)}| results.")
            results = itertools.chain(
                # The type checker thinks that `self._raw_results` can be None here, so we ignore
                *(postprocessing.split_results(res) for res in self._raw_results),  # type: ignore
            )
            # Now we have a list of (index, result) tuples, so we sort by index and only keep result
            self._results = [result for _, result in sorted(results, key=operator.itemgetter(0))]

    def results(self) -> Sequence[Types.Result]:
        """
        Returns results for each circuit in the parallel execution. After calling this for the first
        time, the results are cached, so subsequent calls are very lightweight.
        """

        self._fetch_results()
        return self._results  # type: ignore (the previous call ensures that this is not None)

    @functools.cached_property
    def info(self) -> ParallelizationInfo:
        return self.ParallelizationInfo(self._jobs, self._circuits)

    class ListAttributeProxy:
        """
        A small wrapper around a list of attribute values for a list of objects. This facilitates
        accessing properties of the underlying objects in case both properties and methods need to
        be accessed.

        For example, let `foo` be an instance of this class that wraps objects and a property of
        them that might be a regular attribute *or* a method (known to the user, but not the
        program). Interface-wise, you can treat `foo` as if it were a regular array of attribute
        values, *or* call `foo()` to call the associated method on each instance and receive a
        list of return values.

        This class exists because, for example, Qiskit's `Job` objects have certain attributes that
        are actually implemented as methods, like `job.job_id()`. In the `ParallelJob` class, there
        are several underlying job objects, so an array is returned instead - but returning an
        actual array would break things, since the caller expects some attributes (like `.job_id`)
        to the callable.
        """

        def __init__(self, jobs, attr):
            self.jobs = jobs
            self.attr = attr

        def __call__(self, *args, **kwargs):
            return [getattr(job, self.attr)(*args, **kwargs) for job in self.jobs]

        def __getitem__(self, index):
            return [getattr(job, self.attr) for job in self.jobs][index]

        def __iter__(self):
            for job in self.jobs:
                yield getattr(job, self.attr)

        def __len__(self):
            return len(self.jobs)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise KeyError(f"attribute {attr} may not be accessed")
        return ParallelJob.ListAttributeProxy([job for _, _, job in self._jobs], attr)


def rearrange(
    circuits: Sequence[qiskit.QuantumCircuit | tuple[qiskit.QuantumCircuit, Types.Layout]],
    backends: Types.Backend | Sequence[Types.Backend] | Sequence[tuple[Types.Backend, float]],
    allow_ooe: bool = True,
    packer: packers.PackerBase = packers.Defaults.Fast(),
) -> dict[Types.Backend, Sequence[qiskit.QuantumCircuit]]:
    """
    (Re)arranges a list of circuits into larger host circuits in preparation for parallel execution.
    Basically, this function combines multiple circuits into merged, wider circuits. How the
    circuits are combined depends on their connectivity and other parameters.

    Multiple backends may also be passed, but this feature is mutually exclusive with passing layout
    information for circuits.

    Args:
        circuits:
            A list of QuantumCircuit objects or (QuantumCircuit, Layout) tuples. The given circuit
            objects are copied and not modified.
        backends:
            A Backend object, a list of Backend objects, or a list of (Backend, cost) tuples. The
            circuits will be rearranged onto these backends. In the third case, costs can be used to
            prioritize some backend by assigning lower cost to it. Backends with no associated cost
            are assumed to have unit cost (1).
        allow_ooe:
            Controls circuit ordering. Passing False forces the parallelizer to set circuits
            strictly in order - that is, a circuit that appears earlier in the list of circuits will
            be executed latest at the same time as a circuit that appears later. By default, the
            parallelizer can reorder circuits for efficiency.
        packer:
            The circuit packer to use for placing circuits onto backends. Must be a subclass of
            `packers.PackerBase` (see the `packers` module for more information). Two defaults are
            currently defined: `packers.Defaults.Fast` and `packers.Defaults.Optimal`.

    Returns:
        A dict with backends as keys and lists of packed circuits as values. You may inspect its
        contents manually, with `describe()`, or the visualization submodule, but you should not
        modify its contents. The dict is passable as is to `execute()`.
    """

    has_multiple_backends = False
    if isinstance(backends, Sequence):
        if len(backends) > 1:
            has_multiple_backends = True
    else:
        backends = [backends]

    has_layout_information = any(isinstance(circ, Sequence) for circ in circuits)

    if has_layout_information and has_multiple_backends:
        raise Exceptions.ParameterError(
            "circuit layouts and multiple backends may not be specified simultaneously",
        )

    def normalize_backend(backend) -> tuple[Types.Backend, float]:
        if isinstance(backend, Types.Backend):
            return backend, 1
        if isinstance(backend, Sequence) and len(backend) == 2:
            return tuple(backend)
        raise Exceptions.ParameterError(
            f"expected sequence of backends or (backend, cost) tuples, got '{type(backend)}'",
        )

    backends = [normalize_backend(backend) for backend in backends]

    def normalize_circuit(
        circuit: qiskit.QuantumCircuit | tuple[qiskit.QuantumCircuit, Types.Layout],
        index: int,
    ):
        """
        There are three parts to normalization:
         1. Layout normalization to an `IndexedLayout` object.
         2. Idle qubit removal (and according adjustment of the layout).
         3. Embedding the circuit's original index into its metadata.
        """

        if isinstance(circuit, qiskit.QuantumCircuit):
            layout = IndexedLayout()
        else:
            if isinstance(circuit, Sequence) and len(circuit) == 2:
                circuit, layout = circuit
            else:
                raise Exceptions.ParameterError(
                    (
                        f"expected sequence of circuits or (circuit, layout) tuples, got "
                        f"'{type(circuit)}'"
                    ),
                )
            layout = IndexedLayout.from_layout(layout, circuit)
        bare_circuit, layout = circuittools.remove_idle_qubits(circuit, layout)
        if bare_circuit.num_qubits == 0:
            # The circuit has no active qubits :(
            return None
        bare_circuit = bare_circuit.copy()
        bare_circuit.metadata = {
            "original_metadata": bare_circuit.metadata,
            "index": index,
        }
        return CircuitWithLayout(bare_circuit, layout)

    indexed_circuits = []
    circuit_index = 0
    for circuit in circuits:
        if normalized := normalize_circuit(circuit, circuit_index):
            indexed_circuits.append(normalized)
            circuit_index += 1

    Log.info(
        (
            f"Attempting to rearrange and distribute {len(indexed_circuits)} circuit(s) "
            f"onto {len(backends)} backend(s)."
        ),
    )

    backend_bins = binmanager.CircuitBinManager(backends, packer)
    backend_bins.place(indexed_circuits, allow_ooe)

    Log.info("Circuit rearranging succeeded.")

    return backend_bins.realize()


def execute(
    circuits: Sequence[qiskit.QuantumCircuit]
    | dict[Types.Backend, Sequence[qiskit.QuantumCircuit]],
    backends: Types.Backend
    | Sequence[Types.Backend]
    | Sequence[tuple[Types.Backend, float]]
    | None = None,
    rearrange_args: dict = {},
    run_args: dict = {},
) -> ParallelJob:
    """
    Executes a list of parallel circuits. If the circuits have not yet been rearranged for parallel
    execution, they are rearranged. If you wish to call just one function to do everything for you,
    this is that function.

    Args:
        circuits:
            A list of circuits. See `rearrange()` for details.
        backends:
            A backend or list of backends. See `rearrange()` for details.
        rearrange_args:
            A dict that is passed as kwargs to `rearrange()`, if calling it is necessary.
        run_args:
            A dict that is passed as kwargs to `backend.run()`. Use this to set execution
            properties like shot count.

    Returns:
        A `ParallelJob` object that wraps the underlying job information. Call `.results()` on this
        object to block and retrieve the results. The object behaves just like a traditional job
        object, but deals with several jobs at once.
    """

    if isinstance(circuits, Sequence):
        if backends is None:
            raise Exceptions.MissingParameter(
                "backends must be provided if the input is a list of circuits",
            )
        Log.debug("Provided circuits have not yet been rearranged. Rearranging.")
        circuits = rearrange(circuits, backends, **rearrange_args)

    # TODO: combine circuits with same measured qubits into batches to reduce job count
    job_args = [
        (backend, circuit)
        for backend, circuit_list in circuits.items()
        for circuit in circuit_list
        if len(circuit_list) > 0
    ]
    Log.info(f"Submitting |{len(job_args)}| jobs.")

    jobs: list[tuple[Types.Backend, qiskit.QuantumCircuit, qiskit.providers.JobV1]] = [
        (backend, circuit, backend.run(circuit, **run_args)) for backend, circuit in job_args
    ]  # type: ignore
    Log.info(f"Submitted |{len(jobs)}| jobs.")

    # TODO: jobs and circuits contain duplicate/redundant info
    return ParallelJob(jobs, circuits)


def describe(rearranged: dict[Types.Backend, Sequence[qiskit.QuantumCircuit]], color: bool = True):
    """
    Returns a description of parallelized circuits. The result is a nicely formatted string that
    contains a list of backends and statistics for the number of circuits per each backend. The
    formatted string contains newlines and is intended to be printed as is.

    Args:
        rearranged:
            A dict of rearranged circuits, as returned by `rearrange()`.

        color:
            Controls colored output. Enabled by default, set to False to disable.
    """

    class Color:
        Reset = "\033[0m" if color else ""
        Cyan = "\033[96m" if color else ""

    def fmt(qty, what, color):
        return f"{color or ''}{qty} {what}{'s' if qty != 1 else ''}{Color.Reset}"

    description = [f"{fmt(len(rearranged), 'backend', Color.Cyan)}"]
    for backend, circuits in rearranged.items():
        hosted_circuits = [circuit.metadata["hosted_circuits"] for circuit in circuits]
        lens = Counter(len(c) for c in hosted_circuits)
        lens_str = ", ".join(f"{count}x {size}-circuit" for size, count in lens.items())
        description.extend(
            [
                (
                    " - "
                    f"{backend.name} ({fmt(backend.num_qubits, 'qubit', Color.Cyan)}, "
                    f"{fmt(len(circuits), 'host circuit', Color.Cyan)})"
                ),
                ("   " f"With {Color.Cyan}{lens_str} hosts{Color.Reset}"),
            ],
        )

    return "\n".join(description)
