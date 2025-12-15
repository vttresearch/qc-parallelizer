"""
A collection of packers that utilize the VF2++ layout algorithm.
"""

import heapq
import time
from typing import Any

import rustworkx

from ..backends import BackendCircuitBin
from ..interfaces import Circuit
from ..util import IndexedLayout, Log

from .base import PackerBase

class VF2Base(PackerBase):
    id_order: bool
    call_limit: int | None

    def __init__(self, id_order: bool = False, call_limit: int | None = 50_000_000, **kwargs):
        """
        Args:
            id_order:
                If True, qubits are considered in order by their index. If set to False, a
                heuristic order is used instead. See the
                [VF2++ paper](https://www.sciencedirect.com/science/article/pii/S0166218X18300829)
                for more details.
            call_limit:
                Sets a limit on the number of states that the VF2++ algorithm is allowed to
                explore. None indicates no limit.
        """
        super().__init__(**kwargs)
        self.id_order = id_order
        self.call_limit = call_limit

    def layout_generator(self, bin: BackendCircuitBin, circuit: Circuit, blocked: set[int]):
        phys = rustworkx.PyGraph(multigraph=False)
        phys.add_nodes_from(range(bin.backend.num_qubits))
        phys.add_edges_from_no_data(list(bin.backend.edges_bidir))
        virt = rustworkx.PyGraph(multigraph=False)
        virt.add_nodes_from(range(circuit.num_qubits))
        virt.add_edges_from_no_data(list(circuit.get_edges(bidir=True)))

        Log.debug(
            lambda: (
                f"Generated ${phys.num_nodes()} node$ and ${phys.num_edges()} edge$ for "
                f"physical graph, and ${virt.num_nodes()} node$ and ${virt.num_edges()} edge$ "
                f"for virtual graph."
            ),
        )

        def matcher(p, v) -> bool:
            if p in blocked:
                return False
            if v in circuit.layout.v2p:
                return p == circuit.layout.v2p[v]
            return True

        mapping_generator = rustworkx.vf2_mapping(
            phys,
            virt,
            node_matcher=matcher,
            subgraph=True,
            induced=self.min_intra_distance != 0,
            call_limit=self.call_limit,
            id_order=self.id_order,
        )

        Log.debug(f"VF2++ generator created with `id_order` = |{self.id_order}|.")

        return (dict(mapping) for mapping in mapping_generator)

class NonOptimizing(VF2Base):
    """
    Finds any valid layout. Very efficient, but results in possibly non-optimal packings.
    """

    def find_layout(self, bin: BackendCircuitBin, circuit: Circuit, blocked: set[int]):
        if circuit.num_qubits > bin.backend.num_qubits - len(blocked):
            return None
        Log.debug("Invoking VF2++ to determine the first valid layout.")
        try:
            layout = next(self.layout_generator(bin, circuit, blocked))
            Log.debug("Layout found.")
            return IndexedLayout(p2v=layout)
        except StopIteration:
            Log.warn("No layout found.")
            return None

class Minimizing(VF2Base):
    """
    Finds layouts that minimize unused couplers. Slower than the NonOptimizing version, but
    produces more optimal packings.
    """

    timeout: int | None = None

    def __init__(self, timeout: int | None = 2000, **kwargs):
        """
        Args:
            timeout:
                Defines a maximum runtime, in milliseconds, for evaluating different solutions.
                Set to None for no timeout.
        """
        super().__init__(**kwargs)
        self.timeout = timeout

    def find_layout(self, bin: BackendCircuitBin, circuit: Circuit, blocked):
        if circuit.num_qubits > bin.backend.num_qubits - len(blocked):
            return None
        Log.debug("Invoking VF2++ and iterating over results to find optimal layout.")

        solution_heap: list[tuple[Any, int, IndexedLayout]] = []

        start = time.time()

        def timed_out():
            return self.timeout is not None and time.time() - start >= self.timeout / 1000

        for i, layout in enumerate(self.layout_generator(bin, circuit, blocked)):
            # Check only every 16k iterations to reduce function calls
            if i > 0 and i & 0x3FFF == 0:
                Log.debug(
                    (
                        f"Discovered ${len(solution_heap)} options$ with leading score "
                        f"|{-solution_heap[0][0]}|."
                    ),
                )
                if timed_out():
                    Log.warn("Search interrupted by timeout.")
                    break
            layout = IndexedLayout(p2v=layout)
            score = self.evaluate(bin, circuit.with_layout(layout))
            heapq.heappush(solution_heap, (-score, i, layout))

        Log.debug(f"Found ${len(solution_heap)} possible placement$.")

        if len(solution_heap) > 0:
            *_, best = solution_heap[0]
            return best
        return None

__all__ = (
    "NonOptimizing",
    "Minimizing",
)
