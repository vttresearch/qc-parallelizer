import qiskit
from qc_parallelizer.generic import circuittools, layouts


class TestLayouts:
    def test_layout_basics(self):
        layout = layouts.IndexedLayout(v2p={0: 0, 1: 2, 2: 4})
        assert layout.size == 3
        for i in range(3):
            assert layout.v2p[i] == i * 2
            assert layout.p2v[i * 2] == i

        layout.remove(virt=2)
        assert layout.size == 2
        assert len(layout.p2v) == 2 and len(layout.v2p) == 2


class TestCircuitTools:
    def test_count_gates(self):
        circuit = qiskit.QuantumCircuit(2)
        # An empty circuit should have no operations
        assert circuittools.count_gates(circuit) == {
            circuit.qubits[0]: 0,
            circuit.qubits[1]: 0,
        }
        circuit.h(0)
        # Adding a single-qubit gate should increment one count only
        assert circuittools.count_gates(circuit) == {
            circuit.qubits[0]: 1,
            circuit.qubits[1]: 0,
        }
        circuit.cx(0, 1)
        # Adding a two-qubit gate should increment both counts
        assert circuittools.count_gates(circuit) == {
            circuit.qubits[0]: 2,
            circuit.qubits[1]: 1,
        }
        circuit.barrier(0, 1)
        # And a barrier should not affect the counts at all
        assert circuittools.count_gates(circuit) == {
            circuit.qubits[0]: 2,
            circuit.qubits[1]: 1,
        }

    def test_remove_idle_qubits(self):
        circuit = qiskit.QuantumCircuit(5)
        layout = layouts.IndexedLayout.from_trivial(5)
        circuit.cx(0, 1)
        # Leave a gap here, qubit 2 is not touched
        circuit.cx(3, 4)
        pruned, pruned_layout = circuittools.remove_idle_qubits(circuit, layout=layout)
        assert len(pruned.qregs) == 1 and pruned.qregs[0].size == 4
        assert pruned.num_qubits == 4
        assert pruned_layout.v2p == {0: 0, 1: 1, 2: 3, 3: 4}

    def test_get_neighbor_sets(self):
        circuit = qiskit.QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(3, 4)
        nb_sets = circuittools.get_neighbor_sets(circuit)
        assert nb_sets == [{1, 2}, {0}, {0}, {4}, {3}]

    def test_circuit_hash(self):
        circuit1 = qiskit.QuantumCircuit(2, name="a")
        circuit2 = qiskit.QuantumCircuit(2, name="a")
        circuit1.cx(0, 1)
        circuit2.cx(0, 1)

        assert circuittools.circuit_hash(circuit1) == circuittools.circuit_hash(circuit2)

        circuit1.h(0)

        assert circuittools.circuit_hash(circuit1) != circuittools.circuit_hash(circuit2)

        circuit3 = qiskit.QuantumCircuit(2, name="b")
        circuit3.cx(0, 1)
        circuit3.h(0)

        assert circuittools.circuit_hash(circuit1) != circuittools.circuit_hash(circuit3)
        assert circuittools.circuit_hash(circuit1, meta=False) == circuittools.circuit_hash(
            circuit3,
            meta=False,
        )
