import warnings
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.circuit import ClassicalRegister

from ..interfaces import Circuit, Backend

if TYPE_CHECKING:
    from ..jobs.job import ParallelizerJob

class BackendCircuitBin:
    """
    A backend circuit "bin" is a collection of circuits that are intended to be executed on some
    backend in parallel. An optimal parallelization is comparable to a good solution to the bin
    packing problem, hence the name.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.jobs: list["ParallelizerJob"] = []
        self.cregs: dict[ParallelizerJob, list[ClassicalRegister]] = {}
        self._final: bool = False

    @property
    def size(self):
        """The number of circuits placed in this bin."""
        return len(self.jobs)

    @property
    def label(self):
        r"""
        A label computed from the bin's contents. It is currently of the form
        ```
        23a-15qb-3c
         \    \   `--> number of circuits in bin
          \    `--> number of physical qubits (backend qubit count)
           `---> pseudorandom backend ID
        ```
        """
        backend_id = f"{hash(self.backend.name):03x}"[-3:]
        return f"{backend_id}-{self.backend.num_qubits}qb-{len(self.jobs)}c"

    @property
    def num_free(self):
        return len(self.free_indices)

    @property
    def num_taken(self):
        return len(self.taken_indices)

    @property
    def is_empty(self):
        return self.size == 0

    @property
    def is_full(self):
        return self.num_taken == self.backend.num_qubits

    @property
    def frac_taken(self):
        return self.num_taken / self.backend.num_qubits

    @property
    def num_free_couplers(self):
        """
        The number of currently free couplers. A coupler is free if both end qubits are free.
        """
        free = self.free_indices
        return sum(
            1 for a, b in self.backend.edges
            if a in free and b in free
        )

    @property
    def num_taken_couplers(self):
        """
        The number of currently taken couplers. A coupler is taken if either or both of its end
        qubits are taken.
        """
        taken = self.taken_indices
        return sum(
            1 for a, b in self.backend.edges
            if a in taken or b in taken
        )

    @property
    def _layouts(self):
        return (
            job.circuit.layout for job in self.jobs
        )

    @property
    def free_indices(self):
        return set(range(self.backend.num_qubits)) - self.taken_indices

    @property
    def taken_indices(self):
        qubits: set[int] = set()
        for layout in self._layouts:
            qubits |= layout.pindices
        return qubits

    def compatible(self, circuit: Circuit) -> bool:
        """
        Checks if the given circuit and layout are compatible with the bin in its current state.
        This is only the case if
        - there are enough free qubits,
        - there are enough free couplers (without actually considering the topology), and
        - all physical qubits of the layout are still free.
        """
        if self.num_free < circuit.num_qubits:
            return False
        if self.num_free_couplers < circuit.num_couplers:
            return False
        if circuit.layout.size > 0:
            taken = self.taken_indices
            for p in circuit.layout.pindices:
                if p in taken:
                    return False
        return True

    def place(self, job: "ParallelizerJob"):
        assert job.circuit.is_complete_layout, "attempted to place circuit with incomplete layout"
        assert not self.is_final, "attempted to place circuit in finalized bin"
        self.jobs.append(job)

    def finalize(self):
        """
        Finalizes or locks changes in the bin. This is intended to prevent modifications after a bin
        has been selected for submission or already subitted. Finalization as an action cannot be
        undone and afterwards the bin is effectively read-only.
        """

        assert not self.is_final, "attempted to finalize an already finalized bin"
        self._final = True

    @property
    def is_final(self):
        return self._final

    def to_circuit(self):
        """
        Converts this bin and contained jobs into a physical circuit. This records information about
        classical measurement registers in the resulting circuit and consequently has to finalize
        the bin.
        """

        if not self.is_final:
            self.finalize()

        host_circuit = QiskitCircuit(
            self.backend.num_qubits,
            name=self.label,
        )

        for index, job in enumerate(self.jobs):
            creg_mapping = {}
            for old_reg in job.circuit.cregs:
                new_reg = ClassicalRegister(old_reg.size, name=f"{index}:{old_reg.name}")
                for i in range(old_reg.size):
                    creg_mapping[old_reg[i]] = new_reg[i]
                host_circuit.add_register(new_reg)
                if job not in self.cregs:
                    self.cregs[job] = []
                self.cregs[job].append(new_reg)

            qreg_indices = {
                virt_qubit: job.circuit.index_of(virt_qubit)
                for virt_qubit in job.circuit.qubits
            }

            qreg_mapping = {
                virt_qubit: host_circuit.qubits[job.circuit.layout.v2p[qreg_indices[virt_qubit]]]
                for virt_qubit in job.circuit.qubits
            }

            for operation, qubits, clbits in job.circuit.operations:
                missing_qubit = any(qubit not in qreg_mapping for qubit in qubits)
                missing_clbit = any(clbit not in creg_mapping for clbit in clbits)
                if missing_qubit or missing_clbit:
                    # This could also be a `Log.warn()` call, but it should be clear to the user
                    # regardless of their logging settings.
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

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self):
        return len(self.jobs)

    def __contains__(self, job: "ParallelizerJob"):
        return job in self.jobs
