import warnings

import qiskit
from qc_parallelizer.base import Exceptions, Types
from qc_parallelizer.extensions import Backend, Circuit


class CircuitBin:
    """
    Internal class for a single circuit "bin". Each bin wraps one backend and a list of circuits
    that can/will be placed onto the backend as one circuit, along with a bin-wide layout that
    defines physical-virtual qubit mappings.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.circuits: list[Circuit] = []
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

    def compatible(self, circuit: Circuit) -> bool:
        """
        Checks if the given circuit and layout are compatible with the bin in its current state.
        This is only the case if there are enough free qubits and all physical qubits of the layout
        are still free.
        """
        if self.num_free < circuit.num_qubits:
            return False
        if circuit.layout.size > 0:
            taken = self.taken_indices
            for p in circuit.layout.p2v:
                if p in taken:
                    return False
        return True

    def place(self, circuit: Circuit):
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
        host_circuit = qiskit.QuantumCircuit(
            self.backend.num_qubits,
            metadata={"hosted_circuits": []},
            name=self.label,
        )

        for index, subcircuit in enumerate(self.circuits):
            creg_mapping = {}
            for old_reg in subcircuit.cregs:
                new_reg = qiskit.ClassicalRegister(old_reg.size, name=f"circ{index}.{old_reg.name}")
                for i in range(old_reg.size):
                    creg_mapping[old_reg[i]] = new_reg[i]
                host_circuit.add_register(new_reg)

            qreg_indices = {
                virt_qubit: subcircuit.index_of(virt_qubit) for virt_qubit in subcircuit.qubits
            }

            qreg_mapping = {
                virt_qubit: host_circuit.qubits[subcircuit.layout.v2p[qreg_indices[virt_qubit]]]
                for virt_qubit in subcircuit.qubits
            }

            couplers = [
                (
                    min(subcircuit.layout.v2p[a], subcircuit.layout.v2p[b]),
                    max(subcircuit.layout.v2p[a], subcircuit.layout.v2p[b]),
                )
                for a, b in subcircuit.get_edges()
            ]

            host_circuit.metadata["hosted_circuits"].append(
                {
                    "name": subcircuit.name,
                    "metadata": subcircuit.metadata,
                    "qubits": subcircuit.layout.to_physical_list(),
                    "couplers": couplers,
                    "registers": {
                        "clbit": {"sizes": {f"{reg.name}": reg.size for reg in subcircuit.cregs}},
                    },
                },
            )

            for operation, qubits, clbits in subcircuit.operations:
                missing_qubit = any(qubit not in qreg_mapping for qubit in qubits)
                missing_clbit = any(clbit not in creg_mapping for clbit in clbits)
                if missing_qubit or missing_clbit:
                    warnings.warn(
                        (
                            f"Operation '{operation.name}' skipped while merging circuits. Some of "
                            f"its operands ({qubits if missing_qubit else clbits}) were not found "
                            f"in the register mapping "
                            f"({qreg_mapping if missing_qubit else creg_mapping})."
                        ),
                    )
                    continue
                host_circuit.append(
                    operation,
                    [qreg_mapping[qubit] for qubit in qubits],
                    [creg_mapping[clbit] for clbit in clbits],
                )

        return host_circuit

    def __getitem__(self, index) -> Circuit:
        return self.circuits[index]

    def __iter__(self):
        for i in range(self.size):
            yield self[i]
