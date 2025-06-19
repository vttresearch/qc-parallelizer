import json
import warnings
from typing import Any

import qiskit
import qiskit.circuit

from qc_parallelizer.base import Exceptions
from qc_parallelizer.util import IndexedLayout


class Circuit:
    """
    Wrapper class around Qiskit's QuantumCircuit with support for layouts (IndexedLayout, not the
    internal TranspilerLayout).

    Call `.unwrap()` to extract the underlying circuit object.
    """

    _circuit: qiskit.QuantumCircuit
    _layout: IndexedLayout

    def __init__(
        self,
        circuit: qiskit.QuantumCircuit,
        layout: Any = None,
        clone: bool = False,
    ):
        self._circuit = circuit
        if clone:
            self._circuit = self._circuit.copy()
        self._layout = IndexedLayout.from_layout(layout, circuit)
        if circuit.layout:
            if self._layout.size > 0:
                warnings.warn("Layout provided, but circuit has existing layout.")
            self._layout = IndexedLayout.from_circuit(circuit)
        self._normalize_circuit()
        self._validate_layout()

    def unwrap(self):
        return self._circuit

    def _normalize_circuit(self):
        self._remove_idle()

    def _remove_idle(
        self,
    ):
        gate_count = self.count_gates()
        idle_indices = {index for index, qubit in enumerate(self.qubits) if gate_count[qubit] == 0}
        if len(idle_indices) == 0:
            return

        active_qubits = [
            qubit for index, qubit in enumerate(self.qubits) if index not in idle_indices
        ]

        qreg_mapping = {}
        new_qreg = qiskit.QuantumRegister(len(active_qubits))
        for new_index, old_qubit in enumerate(active_qubits):
            qreg_mapping[old_qubit] = new_qreg[new_index]

        new_circuit = qiskit.QuantumCircuit(
            new_qreg,
            *self._circuit.cregs,
            name=self.name,
            global_phase=self._circuit.global_phase,
            metadata=self._circuit.metadata,
        )

        for operation, qubits, clbits in self._circuit.data:
            new_qubits = [qreg_mapping[qubit] for qubit in qubits if qubit in qreg_mapping]
            if len(new_qubits) != len(qubits):
                # Some operations need to be adjusted. Currently, this is only the case for
                # barriers, since they "operate" on registers, but they do not affect a qubit's
                # activity. So, if we encounter a barrier that was placed partially on active
                # qubits, we lower the qubit count. This does not seem to have any side effects.

                operation = operation.copy()
                operation.num_qubits = len(new_qubits)
            new_circuit.append(operation, new_qubits, clbits)

        self._circuit = new_circuit

        # Since we are dealing with indices, which, inherently, keep pointing at the same index even
        # if the underlying array shifts, we must iterate in decreasing order to not invalidate
        # later indices.
        for index in sorted(idle_indices, reverse=True):
            if index in self._layout.pindices:
                self._layout.remove(phys=index, decrement_keys=True)

    def _validate_layout(self):
        circuit_indices = set(range(self.num_qubits))
        try:
            assert self._layout.vindices.issubset(circuit_indices)
        except AssertionError as error:
            raise Exceptions.InvalidLayout() from error

    def with_layout(self, layout: Any):
        return type(self)(self._circuit, layout)

    @property
    def name(self):
        return self._circuit.name

    @property
    def qubits(self):
        return self._circuit.qubits

    @property
    def cregs(self):
        return self._circuit.cregs

    @property
    def qregs(self):
        return self._circuit.qregs

    @property
    def num_qubits(self):
        return self._circuit.num_qubits

    @property
    def num_clbits(self):
        return self._circuit.num_clbits

    @property
    def num_connected_components(self):
        return self._circuit.num_connected_components()

    @property
    def num_nonlocal_gates(self):
        return self._circuit.num_nonlocal_gates()

    @property
    def depth(self):
        return self._circuit.depth()

    @property
    def metadata(self):
        return self._circuit.metadata

    @metadata.setter
    def metadata(self, value):
        self._circuit.metadata = value

    @property
    def layout(self):
        return self._layout

    @property
    def operations(self):
        return self._circuit.data

    def count_gates(self):
        """
        Returns gate counts for each qubit in the given circuit, including measurements, but
        excluding barriers. For example, for a circuit that looks like this:

        ```
             ┌───┐          ┌─┐
        q_0: ┤ H ├───────■──┤M├───
             ├───┤┌───┐┌─┴─┐└╥┘┌─┐
        q_1: ┤ X ├┤ H ├┤ X ├─╫─┤M├
             ├───┤└┬─┬┘└───┘ ║ └╥┘
        q_2: ┤ H ├─┤M├───────╫──╫─
             └───┘ └╥┘       ║  ║
        c: 3/═══════╩════════╩══╩═
                    2        0  1
        ```

        The returned counts would look like this:

        ```
        {
            "q_0": 3,
            "q_1": 4,
            "q_2": 2,
        }
        ```
        """

        gate_count = {qubit: 0 for qubit in self.qubits}
        for operation, qubits, _ in self._circuit.data:
            if operation.name == "barrier":
                continue
            for qubit in qubits:
                gate_count[qubit] += 1
        return gate_count

    def index_of(self, bit: qiskit.circuit.Qubit | qiskit.circuit.Bit) -> int:
        """
        Returns the index of the given qubit or classical bit in the circuit. The index can be used
        to refer to the bit in the circuit through the `.qubits` or `.clbits` arrays.
        """
        return self._circuit.find_bit(bit=bit).index

    def get_neighbor_sets(self) -> list[set[int]]:
        """
        Returns a list of sets of indices that represent all neighbors that qubits have in the given
        circuit. For non-transpiled circuits, this may also mean other than physically immediate
        neighbors - as a counterexample, a qubit that interacts with every other qubit will have all
        other qubits in its neighbor set, even if that is physically impossible.

        From another point of view, each set represents which qubits the corresponding qubit
        interacts with during the execution of the circuit. Barriers are not counted as interaction.

        For example, for the circuit below,

        ```
        q_0: ──■────■────■──
               │  ┌─┴─┐  │
        q_1: ──■──┤ X ├──┼──
             ┌─┴─┐└───┘  │
        q_2: ┤ X ├───────┼──
             └───┘     ┌─┴─┐
        q_3: ──────────┤ X ├
                       └───┘
        ```

        this function returns

        ```
        [{1, 2, 3}, {0, 2}, {0, 1}, {0}]
        ```
        """

        neighbors = [set() for _ in range(self.num_qubits)]
        for operation, qubits, _ in self._circuit.data:
            if operation.name == "barrier":
                continue
            qubit_indices = [self.index_of(qb) for qb in qubits]
            for i, qb_i in enumerate(qubit_indices):
                for qb_j in qubit_indices[i + 1 :]:
                    neighbors[qb_i].add(qb_j)
                    neighbors[qb_j].add(qb_i)
        return neighbors

    def get_edges(self, bidir: bool = False) -> set[tuple[int, int]]:
        """
        Returns a set of edges, which correspond to the edge's qubits interacting at some point in
        the circuit. Each edge is represented by a tuple of two ints, each int representing the
        virtual index of the two qubits that it joins.

        Args:
            bidir: If set to True, edges in both directions are returned. By default, each edge is
                listed just once, with the two indices in ascending order.
        """

        edges = set()
        for from_, nb_set in enumerate(self.get_neighbor_sets()):
            for to in nb_set:
                a, b = min(from_, to), max(from_, to)
                edges.add((a, b))
                if bidir:
                    edges.add((b, a))
        return edges

    def hash(self, meta: bool = True) -> int:
        """
        Returns an integer hash for the given circuit. Two circuits have the same hash if they have
        - the same number of qubits,
        - the same number of classical bits,
        - the same operations, in the same order, with equal (qubit and classical bit) operands,
        - the same global phase,
        - the same name*, and
        - the same metadata*.

        Set `meta` to False to ignore items with an asterisk.

        Note: Python hashes strings with a random seed, so these hashes are consistent **only**
        within the same session.
        """
        operations = tuple(
            [
                (
                    operation.name,
                    tuple([self.index_of(q) for q in qubits]),
                    tuple([self.index_of(c) for c in clbits]),
                )
                for operation, qubits, clbits in self._circuit.data
            ],
        )
        ophash = hash(
            (
                self.num_qubits,
                self.num_clbits,
                operations,
                self._circuit.global_phase,
            ),
        )
        if not meta:
            return ophash
        metahash = hash((self.name, json.dumps(self.metadata, sort_keys=True, ensure_ascii=True)))
        return hash((ophash, metahash))
