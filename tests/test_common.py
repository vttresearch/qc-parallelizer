import pytest
import qiskit
from qc_parallelizer.generic import backendtools, circuittools, generic, layouts

from .utils import build_circuit_list, fake_20qb_backend


class TestLayouts:
    def test_layout_basics(self):
        layout = layouts.QILayout(v2p={0: 0, 1: 2, 2: 4})
        assert layout.size == 3
        for i in range(3):
            assert layout.v2p[i] == i * 2
            assert layout.p2v[i * 2] == i

        layout = layout.with_blocked({0, 2})
        assert layout.is_blocked(0) and layout.is_blocked(2)
        assert layout.get_blocked() == {0, 2}
        assert 0 not in layout.p2v and 2 not in layout.p2v

        layout.remove(virt=2)
        assert layout.size == 2
        assert len(layout.p2v) == 0 and len(layout.v2p) == 0


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
        layout = layouts.QILayout.from_trivial(5)
        circuit.cx(0, 1)
        # Leave a gap here, qubit 2 is not touched
        circuit.cx(3, 4)
        pruned, pruned_layout = circuittools.remove_idle_qubits(circuit, layout=layout)
        assert len(pruned.qregs) == 1 and pruned.qregs[0].size == 4
        assert pruned.num_qubits == 4
        assert pruned_layout.v2p == {0: 0, 1: 1, 2: 3, 3: 4}
