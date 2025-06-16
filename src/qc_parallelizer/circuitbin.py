import functools

import qiskit

from .base import Exceptions, Types
from .generic import backendtools, circuittools
from .generic.layouts import CircuitWithLayout


class CircuitBin:
    """
    Internal class for a single circuit "bin". Each bin wraps one backend and a list of circuits
    that can/will be placed onto the backend as one circuit, along with a bin-wide layout that
    defines physical-virtual qubit mappings.
    """

    def __init__(self, backend: Types.Backend):
        self.backend = backend
        self.circuits: list[CircuitWithLayout] = []
        self.phys_assignments: dict[int, tuple[int, int] | None] = {
            i: None for i in range(backend.num_qubits)
        }
        # Note! Here, `phys_assignments` maps from physical indices to (circuit index, virtual
        # index) pairs. Other mappings used in this class inversely map from virtual to physical
        # indices. Also note that virtual indices are unique only within one circuit, so they are
        # made bin-unique by storing them in a tuple with the circuit index.

    @property
    def size(self):
        """The number of circuits placed in this bin."""
        return len(self.circuits)

    @property
    def label(self):
        r"""
        A label computed from the bin's contents. It is currently of the form
        ```
        23a-15qb-3c
         \    \   `--> number of circuits in bin
          \    `--> number of physical qubits (backend qubit count)
           `---> backend ID
        ```
        """
        backend_id = f"{hash(self.backend.name):03x}"[-3:]
        return f"{backend_id}-{self.backend.num_qubits}qb-{len(self.circuits)}c"

    @functools.cached_property
    def backend_neighbor_sets(self) -> list[set[int]]:
        return backendtools.get_neighbor_sets(self.backend)

    @property
    def num_free(self):
        return len(self.free_indices)

    @property
    def layout(self) -> dict[int, tuple[int, int]]:
        """
        Currently defined physical-to-virtual qubit mappings. The returned dict has physical indices
        as keys and (circuit index, virtual qubit index) tuples as values.
        """
        return {k: v for k, v in self.phys_assignments.items() if v is not None}

    @property
    def free_indices(self) -> set[int]:
        return {k for k, v in self.phys_assignments.items() if v is None}

    @property
    def taken_indices(self) -> set[int]:
        return {k for k, v in self.phys_assignments.items() if v is not None}

    @functools.cache
    def backend_has_edge(self, from_, to) -> bool:
        """
        Returns True if the backend has a coupler between the given qubit indices. This does not
        respect directed couplers, so each coupler is considered to go both ways.
        """
        edges = self.backend.coupling_map.get_edges()
        return (from_, to) in edges or (to, from_) in edges

    @functools.cache
    def backend_coupling_distance(self, from_, to) -> int:
        return self.backend.coupling_map.distance(from_, to)

    @functools.cached_property
    def backend_edges(self) -> list[tuple[int, int]]:
        return backendtools.get_edges(self.backend, bidir=True)

    @functools.cached_property
    def backend_edges_unique(self) -> list[tuple[int, int]]:
        return backendtools.get_edges(self.backend, bidir=False)

    def compatible(self, circuit: CircuitWithLayout) -> bool:
        """
        Checks if the given circuit and layout are compatible with the bin in its current state.
        This is only the case if there are enough free qubits and all physical qubits of the layout
        are still free.
        """
        if self.num_free < circuit.circuit.num_qubits:
            return False
        if circuit.layout.size > 0:
            taken = self.taken_indices
            for p in circuit.layout.p2v:
                if p in taken:
                    return False
        return True

    def place(self, circuit: CircuitWithLayout):
        """
        Attempts to place circuit into this bin. Returns True on success, False on failure (if the
        circuit is not compatible).
        """
        if not self.compatible(circuit):
            raise Exceptions.CircuitBackendCompatibility(
                "could not place circuit in bin - this is most likely a bug",
            )
        circuit_index = len(self.circuits)
        self.circuits.append(circuit)
        for virt_index, phys_index in circuit.layout.v2p.items():
            self.phys_assignments[phys_index] = (circuit_index, virt_index)

    def realize(self) -> qiskit.QuantumCircuit:
        return circuittools.combine_for_backend(self.circuits, self.backend, name=self.label)

    def __getitem__(self, index) -> CircuitWithLayout:
        return self.circuits[index]

    def __iter__(self):
        for i in range(self.size):
            yield self[i]
