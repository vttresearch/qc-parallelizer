import collections
import functools
import itertools
import operator
import warnings

import qiskit
import qiskit.circuit
import qiskit.compiler
import qiskit.providers
import qiskit.qobj
import qiskit.result
import qiskit.result.models
import qiskit.transpiler
from vtt_quantumutils.common import circuittools, layouts
from vtt_quantumutils.parallelizer import packing, postprocessing
from vtt_quantumutils.parallelizer.base import Exceptions, Types


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
            jobs: list[tuple[Types.Backend, qiskit.QuantumCircuit, qiskit.providers.Job]] | None,
            circuit_dict: dict[Types.Backend, list[qiskit.QuantumCircuit]],
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
            circuit: qiskit.QuantumCircuit
        ) -> tuple[qiskit.providers.Job, Types.Backend]:
            for backend, host_circuit, job in self.jobs:
                for hosted_circuit in host_circuit.metadata["_hosted_circuits"]:
                    if hosted_circuit["name"] == circuit.name:
                        return (job, backend)
            raise Exceptions.MissingInformation("the given circuit was not a part of this job")

    def __init__(
        self,
        jobs: list[tuple[Types.Backend, qiskit.QuantumCircuit, qiskit.providers.Job]],
        circuits: dict[Types.Backend, list[qiskit.QuantumCircuit]],
    ):
        self._jobs = jobs
        self._circuits = circuits
        self._raw_results: list[qiskit.result.result.Result] | None = None
        self._results: list[qiskit.result.result.Result] | None = None

    def _fetch_results(self):
        if self._raw_results is None:
            self._raw_results = [job.result() for _, _, job in self._jobs]
        if self._results is None:
            results = itertools.chain(
                *(postprocessing.split_results(res) for res in self._raw_results),
            )
            # Now we have a list of (index, result) tuples, so we sort by index and only keep result
            self._results = [result for _, result in sorted(results, key=operator.itemgetter(0))]

    def results(self) -> list[Types.Result]:
        """
        Returns results for each circuit in the parallel execution. After calling this for the first
        time, the results are cached, so subsequent calls are very lightweight.
        """

        self._fetch_results()
        return self._results

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

        def __iter__(self, index):
            for job in self.jobs:
                yield getattr(job, self.attr)

        def __len__(self):
            return len(self.jobs)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            return None
        return ParallelJob.ListAttributeProxy([job for _, _, job in self._jobs], attr)


def rearrange(
    circuits: list[qiskit.QuantumCircuit | tuple[qiskit.QuantumCircuit, Types.Layout]],
    backends: Types.Backend | list[Types.Backend],
    allow_ooe: bool = True,
    packpol: packing.PackingPolicyBase = packing.DefaultPackingPolicy(),
    transpiler_seed: int = 0,
) -> dict[Types.Backend, list[qiskit.QuantumCircuit]]:
    """
    (Re)arranges a list of circuits into larger host circuits in preparation for parallel execution.
    Basically, this function combines as many circuits into new, wider circuits, as the provided
    backend(s) natively support.

    Multiple backends may also be passed, but this feature is mutually exclusive with passing layout
    information for circuits.

    `allow_ooe` (allow out-of-order execution) controls circuit ordering. Setting it to False forces
    the parallelizer to set circuits strictly in order - that is, a circuit that appears earlier in
    the list of circuits will be executed latest at the same time as a circuit that appears later.
    By default, it is True, which allows the parallelizer to reorder circuits for a more optimal
    packing.

    `packpol` is a circuit packing policy. Please refer to the packing module for more information.
    `transpiler_seed` is passed onward to transpiler passes that can work with seeds for consistent
    results. By default, it is fixed - use some value from `random` for variance in results.

    The resulting combined circuits will contain metadata about the original circuits, which the
    user should not touch.
    """

    has_multiple_backends = False
    if isinstance(backends, (list, tuple)):
        if len(backends) > 1:
            has_multiple_backends = True
    else:
        backends = [backends]

    has_layout_information = any(not isinstance(circ, qiskit.QuantumCircuit) for circ in circuits)

    if has_layout_information and has_multiple_backends:
        raise Exceptions.ParameterConflict(
            "circuit layouts and multiple backends may not be specified simultaneously",
        )

    def prune_and_normalize_layout_information(
        circuit: qiskit.QuantumCircuit | tuple[qiskit.QuantumCircuit, Types.Layout],
    ) -> tuple[qiskit.QuantumCircuit, layouts.QILayout]:
        layout = layouts.QILayout()
        if not isinstance(circuit, qiskit.QuantumCircuit):
            circuit, layout = circuit
        return circuittools.remove_idle_qubits(circuit, layout)

    # Add index to circuits for restoring original order after execution
    indexed_circuits = [
        (index, *circuit)
        for index, circuit in enumerate(
            prune_and_normalize_layout_information(circuit) for circuit in circuits
        )
    ]

    if allow_ooe:
        # Sort circuits based on two factors. In the order of precedence,
        #  1. circuits with more layout information and
        #  2. larger circuits
        # come first.
        indexed_circuits.sort(key=lambda circuit: (-circuit[2].size, -circuit[1].num_qubits))

    backend_bins = packing.CircuitBinManager(backends, packpol)

    for index, circuit, layout in indexed_circuits:
        backend_bins.place(circuit, layout, {"index": index}, transpiler_seed=transpiler_seed)

    return backend_bins.realize()


def execute(
    circuits: list[qiskit.QuantumCircuit] | dict[Types.Backend, list[qiskit.QuantumCircuit]],
    backends: Types.Backend | list[Types.Backend] | None = None,
    rearrange_args: dict = {},
    run_args: dict = {},
) -> ParallelJob:
    """
    Executes a list of parallel circuits. If the circuits have not yet been rearranged for parallel
    execution, they are rearranged. If you wish to call just one function to do everything for you,
    this is that function.

    Returns a `ParallelJob` object that wraps the underlying job information. Call `.results()` on
    this object to block and retrieve the results. The object behaves just like a traditional job
    object, but deals with several jobs at once.

    The parameters `rearrange_args` and `run_args` allow passing kwargs to the underlying calls to
    `rearrange()` and `backend.run()`, respectively.
    """

    if isinstance(circuits, list):
        if backends is None:
            raise Exceptions.MissingParameter(
                "backends must be provided if the input is a list of circuits",
            )
        circuits = rearrange(circuits, backends, **rearrange_args)

    # TODO: combine circuits with same measured qubits for less jobs
    jobs = [
        (backend, circuit, backend.run(circuit, **run_args))
        for backend, circuit_list in circuits.items()
        for circuit in circuit_list
        if len(circuit_list) > 0
    ]

    # TODO: jobs and circuits contain duplicate/redundant info
    return ParallelJob(jobs, circuits)


def describe(rearranged: dict[Types.Backend, list[qiskit.QuantumCircuit]], color: bool = True):
    """
    Returns a description of parallelized circuits. The result is a nicely formatted string that
    contains a list of backends and statistics for the number of circuits per each backend. The
    formatted string contains newlines and is intended to be printed as is.
    """

    class Color:
        Reset = "\033[0m" if color else ""
        Cyan = "\033[96m" if color else ""

    def fmt(qty, what, color):
        return f"{color or ''}{qty} {what}{'s' if qty != 1 else ''}{Color.Reset}"

    description = [f"{fmt(len(rearranged), 'backend', Color.Cyan)}"]
    for backend, circuits in rearranged.items():
        hosted_circuits = [circuit.metadata["_hosted_circuits"] for circuit in circuits]
        lens = [len(c) for c in hosted_circuits]
        description.extend(
            [
                (
                    " - "
                    f"{backend.name} ({fmt(backend.num_qubits, 'qubit', Color.Cyan)}, "
                    f"{fmt(len(circuits), 'circuit', Color.Cyan)})"
                ),
                (
                    "   "
                    f"Min {Color.Cyan}{min(lens)}{Color.Reset} / "
                    f"avg {Color.Cyan}{sum(lens) / len(hosted_circuits):.1f}{Color.Reset} / "
                    f"max {Color.Cyan}{max(lens)}{Color.Reset} "
                    "hosted circuits per physical circuit"
                ),
            ],
        )

    return "\n".join(description)
