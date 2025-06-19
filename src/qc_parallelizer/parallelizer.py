import typing
from collections import Counter
from collections.abc import Sequence

import qiskit
import qiskit.providers

from . import packers
from .base import Exceptions, Types
from .bins.manager import CircuitBinManager
from .extensions import Backend, Circuit, ensure_sequence, isnestedinstance
from .results.paralleljob import ParallelJob
from .util import Log


class InputTypes:
    Circuit = qiskit.QuantumCircuit | tuple[qiskit.QuantumCircuit, Types.Layout] | Circuit
    Circuits = Circuit | Sequence[Circuit]
    Backend = qiskit.providers.BackendV2 | tuple[qiskit.providers.BackendV2, int | float] | Backend
    Backends = Backend | Sequence[Backend]


def rearrange(
    circuits: InputTypes.Circuits,
    backends: InputTypes.Backends,
    allow_ooe: bool = True,
    packer: packers.PackerBase = packers.Defaults.Fast(),
) -> dict[Backend, Sequence[qiskit.QuantumCircuit]]:
    """
    (Re)arranges a list of circuits into larger host circuits in preparation for parallel execution.
    Basically, this function combines multiple circuits into merged, wider circuits. How the
    circuits are combined depends on their connectivity and other parameters.

    Multiple backends may also be passed, but this feature is mutually exclusive with passing layout
    information for circuits.

    Args:
        circuits:
            A list of QuantumCircuit objects or (QuantumCircuit, Layout) tuples. The given circuit
            objects are copied and the originals are not modified.
        backends:
            A backend object, a list of backend objects, or a list of (backend, cost) tuples. The
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

    # First, some military-grade input type checking and normalization.

    circuits = ensure_sequence(circuits, InputTypes.Circuit)

    def normalize_circuit(circuit):
        if isinstance(circuit, Circuit):
            return circuit
        if isinstance(circuit, qiskit.QuantumCircuit):
            return Circuit(circuit, clone=True)
        if isnestedinstance(circuit, tuple[qiskit.QuantumCircuit, Types.Layout]):
            return Circuit(*circuit, clone=True)
        assert False, "unreachable code"

    normalized_circuits = []
    for circuit in circuits:
        if normalized := normalize_circuit(circuit):
            if normalized.num_qubits > 0:
                normalized.metadata = {
                    "original_metadata": normalized.metadata,
                    "index": len(normalized_circuits),
                }
                normalized_circuits.append(normalized)
    circuits = normalized_circuits

    # Then the same dance for backends.

    backends = ensure_sequence(backends, InputTypes.Backend)

    def normalize_backend(backend):
        if isinstance(backend, Backend):
            return backend
        if isinstance(backend, qiskit.providers.BackendV2):
            return Backend(backend, 1)
        if isnestedinstance(backend, tuple[qiskit.providers.BackendV2, int | float]):
            return Backend(*backend)
        assert False, "unreachable code"

    backends = [normalize_backend(backend) for backend in backends]

    if any(circuit.layout.size > 0 for circuit in circuits) and len(backends) > 1:
        raise Exceptions.ParameterError(
            "circuit layouts and multiple backends may not be specified simultaneously",
        )

    Log.info(
        (
            f"Attempting to rearrange and distribute {len(circuits)} circuit(s) onto "
            f"{len(backends)} backend(s)."
        ),
    )

    backend_bins = CircuitBinManager(backends, packer)
    backend_bins.place(circuits, allow_ooe)

    Log.info("Circuit rearranging succeeded.")

    return backend_bins.realize()


def execute(
    circuits: InputTypes.Circuits | dict[Backend, Sequence[qiskit.QuantumCircuit]],
    backends: InputTypes.Backends | None = None,
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

    if not isnestedinstance(circuits, dict[Backend, Sequence[qiskit.QuantumCircuit]]):
        if backends is None:
            raise Exceptions.MissingParameter(
                "backends must be provided if the given circuits have not been rearranged",
            )
        Log.debug("Provided circuits have not yet been rearranged. Rearranging.")
        circuits = rearrange(
            typing.cast(InputTypes.Circuits, circuits),
            backends,
            **rearrange_args,
        )
    elif backends is not None:
        raise Exceptions.ParameterError(
            "backends were provided but circuits are already rearranged",
        )

    circuits = typing.cast(dict[Backend, Sequence[qiskit.QuantumCircuit]], circuits)

    # TODO: batch execute circuits whenever possible
    job_args = [
        (backend, circuit)
        for backend, circuit_list in circuits.items()
        for circuit in circuit_list
        if len(circuit_list) > 0
    ]
    Log.info(f"Submitting |{len(job_args)}| jobs.")

    jobs = [(circuit, backend.run(circuit, **run_args)) for backend, circuit in job_args]
    Log.info(f"Submitted |{len(jobs)}| jobs.")

    return ParallelJob(jobs)


def describe(rearranged: dict[Backend, Sequence[qiskit.QuantumCircuit]], color: bool = True):
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
