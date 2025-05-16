import functools
import heapq
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any

import qiskit
import qiskit.transpiler

from . import transpiling
from .base import Exceptions, Types
from .generic import backendtools, circuittools, layouts


class CircuitBin:
    """
    Internal class for a single circuit "bin", or a list of circuits with a bin-wide layout table
    from the backend's physical indices to qubits in each circuit.
    """

    def __init__(self, backend: Types.Backend):
        self.backend = backend
        self.tracked_circuits: list[tuple[qiskit.QuantumCircuit, Any, layouts.QILayout]] = []
        self.phys_assignments: dict[int, tuple[int, int] | None] = {
            i: None for i in range(backend.num_qubits)
        }
        # Note! Here, `phys_assignments` maps from physical indices to (circuit index, virtual
        # index) pairs. Other mappings used in this class inversely map from virtual to physical
        # indices. Also note that virtual indices are unique only within one circuit, so they are
        # made bin-unique by storing them in a tuple with the circuit index.

    @property
    def circuits(self):
        return list(self)

    @property
    def size(self):
        return len(self.tracked_circuits)

    @property
    def label(self):
        backend_id = f"{hash(self.backend.name):03x}"[-3:]
        return f"{backend_id}-{self.backend.num_qubits}qb-{len(self.tracked_circuits)}c"

    @functools.cached_property
    def backend_neighbor_sets(self) -> list[set[int]]:
        return backendtools.get_neighbor_sets(self.backend)

    @property
    def num_free(self):
        return len(self.get_free_indices())

    def get_defined_layout(self) -> dict[int, int]:
        return {k: v for k, v in self.phys_assignments.items() if v is not None}

    def get_free_indices(self) -> set[int]:
        return {k for k, v in self.phys_assignments.items() if v is None}

    def get_taken_indices(self) -> set[int]:
        return {k for k, v in self.phys_assignments.items() if v is not None}

    def backend_has_edge(self, from_, to) -> bool:
        edges = self.backend.coupling_map.get_edges()
        # Checking both ways might not be necessary, but the logic should short-circuit if the edge
        # exists, which makes this perform worse only if the edge is not present
        return (from_, to) in edges or (to, from_) in edges

    def backend_distance(self, from_, to) -> int:
        return self.backend.coupling_map.distance(from_, to)

    def compatible(self, circuit: qiskit.QuantumCircuit, initial_layout: layouts.QILayout):
        if self.num_free < circuit.num_qubits:
            return False
        if initial_layout.size > 0:
            taken = self.get_taken_indices()
            for p in initial_layout.p2v:
                if p in taken:
                    return False
        return True

    def place(
        self,
        circuit: qiskit.QuantumCircuit,
        layout: layouts.QILayout,
        metadata=None,
    ):
        if not self.compatible(circuit, layout):
            return False
        self.tracked_circuits.append((circuit, metadata, layout))
        for virt_index, phys_index in layout.v2p.items():
            self.phys_assignments[phys_index] = (len(self.tracked_circuits) - 1, virt_index)
        return True

    def realize(self) -> qiskit.QuantumCircuit:
        return circuittools.combine_for_backend(self.circuits, self.backend, name=self.label)

    def __getitem__(self, index) -> tuple[qiskit.QuantumCircuit, Any, layouts.QILayout]:
        return self.tracked_circuits[index]

    def __iter__(self):
        for i in range(self.size):
            yield self[i]


class PackingPolicyBase(ABC):
    """
    A base class for a circuit packing policy.

    A packing policy defines four things:
    1. The number of packing trials, as an `int` in a propery called `num_trials`.
    2. Maximum number of bins per backend, also as an `int`, called `max_bins_per_backend`. May be
       `None` for no limit.
    3. Which backend qubits are allowed to be used, based on the current bin state.
       Implemented as a method called `blocked`, and returns a `set[int]`.
    4. How good a proposed circuit/layout combination is for the backend, also based on the state.
       Implemented as a method called `evaluate`, and returns a `tuple[Any, bool]`.

    The two methods are given a `PackingState` object. See the source for more information on what
    is available in the state object and what exactly should be returned.
    """

    class PackingState:
        OriginalTranspiledPair = namedtuple("OriginalTranspiledPair", "original transpiled")

        def __init__(
            self,
            original_circuit: qiskit.QuantumCircuit,
            original_layout: layouts.QILayout,
            circuit_bin: CircuitBin | None = None,
            transpiled_circuit: qiskit.QuantumCircuit | None = None,
            transpiled_layout: layouts.QILayout | None = None,
        ):
            self.bin = circuit_bin
            self.transpiled = transpiled_circuit is not None
            self.circuits = self.OriginalTranspiledPair(original_circuit, transpiled_circuit)
            self.layouts = self.OriginalTranspiledPair(original_layout, transpiled_layout)

        def with_bin(self, circuit_bin: CircuitBin):
            return self.__class__(
                original_circuit=self.circuits.original,
                original_layout=self.layouts.original,
                circuit_bin=circuit_bin,
                transpiled_circuit=self.circuits.transpiled,
                transpiled_layout=self.layouts.transpiled,
            )

        def with_transpiled(self, circuit: qiskit.QuantumCircuit, layout: layouts.QILayout):
            return self.__class__(
                original_circuit=self.circuits.original,
                original_layout=self.layouts.original,
                circuit_bin=self.bin,
                transpiled_circuit=circuit,
                transpiled_layout=layout,
            )

        @functools.cached_property
        def original_depth(self):
            return self.circuits.original.depth()

        @functools.cached_property
        def transpiled_depth(self):
            if not self.transpiled:
                return None
            return self.circuits.transpiled.depth()

        @functools.cached_property
        def delta_depth(self):
            if not self.transpiled:
                return None
            return self.transpiled_depth - self.original_depth

    num_trials: int
    max_bins_per_backend: int

    @abstractmethod
    def blocked(self, state: PackingState) -> set[int]:
        """Returns a set of physical indices to be blocked on the backend."""

    @abstractmethod
    def evaluate(self, state: PackingState) -> tuple[Any, bool]:
        """
        Evaluates the current state and returns a tuple that contains
        1. a sortable object that indicates how good the current state is, and
        2. a boolean that is `True` if the current state is an optimum, which stops the search.
        """


class DefaultPackingPolicy(PackingPolicyBase):
    num_trials = 1
    max_bins_per_backend = None

    def __init__(self, allow_nb: bool = True):
        self.allow_nb = allow_nb

    def blocked(self, state: PackingPolicyBase.PackingState):
        taken = state.bin.get_taken_indices()
        if not self.allow_nb:
            nb_sets = state.bin.backend_neighbor_sets
            neighbors = set()
            for qb in taken:
                neighbors |= nb_sets[qb]
            return taken | neighbors
        return taken

    def evaluate(self, state: PackingPolicyBase.PackingState):
        return (state.delta_depth,), state.delta_depth <= 0


class CircuitBinManager:
    def __init__(self, backends: list[Types.Backend], packpol: PackingPolicyBase):
        self.backends = backends
        self.bins: list[CircuitBin] = []
        self.packpol = packpol

    def candidate_bins(self, circuit: qiskit.QuantumCircuit, initial_layout: layouts.QILayout):
        # Ensure that each backend has at least one empty bin (up to `max_bins`)
        backend_bin_empty = {backend: True for backend in self.backends}
        bins_per_backend = {backend: 0 for backend in self.backends}
        for circuit_bin in self.bins:
            bins_per_backend[circuit_bin.backend] += 1
            if circuit_bin.size == 0:
                backend_bin_empty[circuit_bin.backend] = False

        max_bins = self.packpol.max_bins_per_backend or float("inf")
        for backend, empty in backend_bin_empty.items():
            if empty and bins_per_backend[backend] < max_bins:
                self.bins.append(CircuitBin(backend))

        # Then yield each bin, first non-empty bins in increasing size order followed by empty ones
        for circuit_bin in sorted(self.bins, key=lambda cb: (cb.size == 0, cb.size)):
            if circuit_bin.compatible(circuit, initial_layout):
                yield circuit_bin

    def place(
        self,
        circuit: qiskit.QuantumCircuit,
        initial_layout: layouts.QILayout,
        metadata: Any,
        transpiler_seed: int,
    ):
        """
        Inserts a circuit into a managed bin, or raises an exception if not possible.
        """

        state = PackingPolicyBase.PackingState(
            original_circuit=circuit,
            original_layout=initial_layout,
        )
        bin_heap: list[tuple[Any, CircuitBin, qiskit.QuantumCircuit, layouts.QILayout]] = []

        candidates = list(self.candidate_bins(circuit, initial_layout))
        trial_params = (
            (trial, index, circuit_bin, state.with_bin(circuit_bin))
            for index, circuit_bin in enumerate(candidates)
            for trial in range(self.packpol.num_trials)
        )

        error_bins = set()
        for trial, index, circuit_bin, bin_state in trial_params:
            if circuit_bin in error_bins:
                continue
            try:
                transp = transpiling.transpile_to_layout(
                    circuit,
                    circuit_bin.backend,
                    initial_layout.with_blocked(self.packpol.blocked(bin_state)),
                    hash((trial, index, transpiler_seed)) & 0xFFFFFFFF,
                )
            except qiskit.transpiler.TranspilerError:
                error_bins.add(circuit_bin)
            else:
                sort_order, is_optimum = self.packpol.evaluate(bin_state.with_transpiled(*transp))
                heapq.heappush(bin_heap, (*sort_order, index, trial, circuit_bin, *transp))
                if is_optimum:
                    break

        # We now have a heap of circuit bins where the first item is the best option for the given
        # circuit. So, if there are any bins in the heap...
        if len(bin_heap) > 0:
            # Place the circuit in the first one.
            *_, circuit_bin, transpiled, transp_layout = bin_heap[0]
            assert circuit_bin.place(
                transpiled,
                transp_layout,
                metadata,
            ), "this assertion should never fail - the packing policy is probably incoherent"
        else:
            # Otherwise, there were no suitable bins for the circuit. Report this as an exception.
            raise Exceptions.CircuitBackendCompatibility(
                "could not place circuit in any bin",
            )

    def realize(self):
        backend_circuits = {}
        for circuit_bin in self.bins:
            if circuit_bin.size > 0:
                backend_circuits.setdefault(circuit_bin.backend, []).append(circuit_bin.realize())
        return backend_circuits
