import warnings

import pytest
import qiskit
from qc_parallelizer import packers, parallelizer
from qc_parallelizer.extensions import Backend, Circuit
from qc_parallelizer.util import IndexedLayout, translation

from .utils import build_circuit_list, fake_5qb_backend, fake_20qb_backend, fake_54qb_backend


class TestParallelizer:
    @pytest.mark.parametrize(
        ["circuits", "expected_rearranged_len"],
        [
            # A circuit with 30 qubits should not fit at all
            (build_circuit_list("1 h 30"), None),
            # Two circuits with 15 each should result in two circuits
            # TODO!!! Why does this hang?
            (build_circuit_list("2 h 15"), {fake_20qb_backend: 2}),
            # But two circuits with 15 qubits but only 5 used should be placed into just one circuit
            (build_circuit_list("2 partial 15 5"), {fake_20qb_backend: 1}),
        ],
    )
    def test_idle_qubit_removal(self, circuits, expected_rearranged_len):
        should_fail = expected_rearranged_len is None
        try:
            with warnings.catch_warnings(record=True) as w:
                rearranged = parallelizer.rearrange(circuits, fake_20qb_backend)
                assert (len(w) == 1) == should_fail
        except parallelizer.Exceptions.CircuitBackendCompatibility:
            assert should_fail
        else:
            assert not should_fail
            assert expected_rearranged_len == {
                backend: len(circuits) for backend, circuits in rearranged.items()
            }


class TestTranspiling:
    def test_translating(self):
        circuit = qiskit.QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.ccx(0, 1, 2)

        translated = translation.translate_for_backend(
            circuit,
            fake_20qb_backend,
        )
        assert translated is not None

        for operation, qubits, _ in translated.operations:
            assert len(qubits) <= 2
            assert operation.name in fake_20qb_backend.operation_names


class TestPackers:
    def test_packer_evaluate(self):
        bin = packers.CircuitBin(fake_5qb_backend)
        packer = packers.PackerBase()

        circuit1 = qiskit.QuantumCircuit(5)
        circuit1.cx(0, list(range(1, 5)))
        layout1 = IndexedLayout(
            v2p={
                0: 2,
                1: 0,
                2: 1,
                3: 3,
                4: 4,
            },
        )

        circuit2 = qiskit.QuantumCircuit(2)
        circuit2.cx(0, 1)
        layout2 = IndexedLayout(
            v2p={
                0: 0,
                1: 2,
            },
        )

        score1 = packer.evaluate(bin, Circuit(circuit1, layout1))
        score2 = packer.evaluate(bin, Circuit(circuit2, layout2))

        # Both of these use the same number of couplers
        assert score1 == score2

    @pytest.mark.parametrize(
        ["circuit", "expect_failure"],
        [
            # A GHZ circuit with up to five qubits fits on the backend's topology.
            (build_circuit_list("ghz 4"), False),
            (build_circuit_list("ghz 5"), False),
            (build_circuit_list("ghz 6"), True),
            (build_circuit_list("ghz 7"), True),
            # Grids should work up to 7x5.
            (build_circuit_list("grid 3 3"), False),
            (build_circuit_list("grid 7 5"), False),
            # Larger should fail.
            (build_circuit_list("grid 7 6"), True),
        ],
    )
    def test_packer_find_layout(self, circuit, expect_failure):
        circuit = Circuit(circuit)
        bin = packers.CircuitBin(fake_54qb_backend)
        packer = packers.Defaults.Fast()
        layout = packer.find_layout(
            bin,
            circuit,
            set(),
        )
        if layout is None:
            assert expect_failure, "layout failed when it should have succeeded"
        else:
            assert not expect_failure, "layout succeeded when it should have failed"

            for from_, to in circuit.get_edges():
                phys_edge = tuple(sorted((layout.v2p[from_], layout.v2p[to])))
                assert phys_edge in fake_54qb_backend.edges
