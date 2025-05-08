import warnings

import pytest
import qiskit
import qiskit.transpiler
from qc_parallelizer import parallelizer, transpiling
from qc_parallelizer.generic import backendtools, circuittools, generic

from .utils import build_circuit_list, fake_20qb_backend


class TestParallelizer:
    @pytest.mark.parametrize(
        ["circuits", "expected_rearranged_len"],
        [
            # A circuit with 30 qubits should not fit at all
            (build_circuit_list("1 partial 30 30"), None),
            # Two circuits with 15 each should result in two circuits
            (build_circuit_list("2 partial 15 15"), {fake_20qb_backend: 2}),
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


class TestLayouts:
    @pytest.mark.parametrize(
        ["circuit", "expect_failure"],
        [(build_circuit_list("ghz 5"), False), (build_circuit_list("ghz 50"), True)],
    )
    def test_layout_completion_single_circuit(self, circuit, expect_failure):
        try:
            circuit, layout = transpiling.transpile_to_layout(circuit, fake_20qb_backend)
        except qiskit.transpiler.TranspilerError:
            assert expect_failure
        else:
            assert not expect_failure

            phys_edges = generic.get_edges(backendtools.get_neighbor_sets(fake_20qb_backend))
            virt_edges = generic.get_edges(circuittools.get_neighbor_sets(circuit))
            for from_, to in virt_edges:
                phys_edge = tuple(sorted((layout.v2p[from_], layout.v2p[to])))
                # These two assertions should be equivalent, but verify just in case
                assert phys_edge in phys_edges
                assert fake_20qb_backend.coupling_map.distance(*phys_edge) == 1
