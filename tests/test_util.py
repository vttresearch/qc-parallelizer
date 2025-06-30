import pytest
import qiskit
from qc_parallelizer.util import IndexedLayout


class TestUtils:
    @pytest.mark.parametrize(
        ["input", "v2p"],
        [
            ({0: 1, 3: 4}, {0: 1, 3: 4}),
            ([1, 2, 3, 4], {0: 1, 1: 2, 2: 3, 3: 4}),
            ({}, {}),
            (None, {}),
        ],
    )
    def test_layout_parsing(self, input, v2p: dict[int, int]):
        circuit = qiskit.QuantumCircuit(4)
        parsed = IndexedLayout.from_layout(input, circuit)

        assert parsed.v2p == v2p
