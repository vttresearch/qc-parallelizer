import qiskit
from qc_parallelizer.extensions import Backend, Circuit
from qc_parallelizer.util import IndexedLayout


class TestCircuitExtension:
    def test_idle_qubit_removal(self):
        circuit = qiskit.QuantumCircuit(5)
        layout = IndexedLayout.from_trivial(5)
        circuit.cx(0, 1)
        # Leave a gap here, qubit 2 is not touched
        circuit.cx(3, 4)

        pruned = Circuit(circuit, layout)

        assert len(pruned.qregs) == 1 and pruned.qregs[0].size == 4
        assert pruned.num_qubits == 4
        assert pruned.layout.v2p == {0: 0, 1: 1, 2: 3, 3: 4}

    def test_get_neighbor_sets(self):
        circuit = qiskit.QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(3, 4)
        assert Circuit(circuit).get_neighbor_sets() == [{1, 2}, {0}, {0}, {4}, {3}]

    def test_circuit_hash(self):
        circuit1 = qiskit.QuantumCircuit(2, name="a")
        circuit2 = qiskit.QuantumCircuit(2, name="a")
        circuit1.cx(0, 1)
        circuit2.cx(0, 1)

        assert Circuit(circuit1).hash() == Circuit(circuit2).hash()

        circuit1.h(0)

        assert Circuit(circuit1).hash() != Circuit(circuit2).hash()

        circuit3 = qiskit.QuantumCircuit(2, name="b")
        circuit3.cx(0, 1)
        circuit3.h(0)

        assert Circuit(circuit1).hash() != Circuit(circuit3).hash()
        assert Circuit(circuit1).hash(meta=False) == Circuit(circuit3).hash(meta=False)
