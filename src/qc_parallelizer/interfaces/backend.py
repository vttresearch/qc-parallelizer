import functools
import threading
import typing
from collections.abc import Callable
from typing import Any

import qiskit.providers
import qiskit.transpiler
from qiskit.providers import JobV1 as QiskitJob
from qiskit.result import Result as QiskitJobResult
from qiskit.transpiler import Target as QiskitBackendTarget


class Backend:
    """
    Wrapper class for Qiskit's `BackendV2` class to provide additional short-hand functionality.
    """

    _backend: qiskit.providers.BackendV2
    _cost: float

    def __init__(self, backend: qiskit.providers.BackendV2, cost: float = 1):
        self._backend = backend
        self._cost = cost

    def unwrap(self):
        return self._backend

    @property
    def cost(self):
        return self._cost

    @property
    def name(self):
        return self._backend.name

    @property
    def num_qubits(self):
        return self._backend.num_qubits

    @property
    def num_couplers(self):
        return len(self.edges)

    @property
    def operation_names(self):
        return self._backend.operation_names

    @property
    def target(self):
        return typing.cast(QiskitBackendTarget, self._backend.target)

    @functools.cached_property
    def coupling_map(self) -> qiskit.transpiler.CouplingMap:
        return self._backend.coupling_map

    @functools.cached_property
    def neighbor_sets(self) -> list[set[int]]:
        """
        Returns sets of physical neighbors in the backend's topology. Returns a list where each
        index corresponds to a physical qubit and the set at that index to the physical neighbors
        of that qubit.
        """

        edges = self.coupling_map.get_edges()
        neighbors = [set() for _ in range(self.num_qubits)]
        for from_, to in edges:
            neighbors[from_].add(to)
            neighbors[to].add(from_)
        return neighbors

    @functools.cache
    def get_edges(self, bidir: bool = False) -> set[tuple[int, int]]:
        """
        Returns a set of couplers, or edges, in the backend. Each edge is represented by a tuple of
        two ints, each int representing the physical index of the two qubits that it joins.

        Args:
            bidir: If set to True, edges in both directions are returned. By default, each edge is
                listed just once, with the two indices in ascending order.
        """

        # We could simply return the edges from the coupling map, but it is internally represented
        # as a directed graph with two edges per coupler - in most cases. So, to be safe, we process
        # the edge list ourselves and either make it bidirectional or not.

        raw_edges = self.coupling_map.get_edges()
        edge_set = set()
        for a, b in raw_edges:
            a, b = min(a, b), max(a, b)
            edge_set.add((a, b))
            if bidir:
                edge_set.add((b, a))
        return edge_set

    @functools.cached_property
    def edges(self):
        """
        The set of edges in the backend with one edge per coupler. The two indices in each edge are
        always in increasing order.
        """
        return self.get_edges(bidir=False)

    @functools.cached_property
    def edges_bidir(self):
        """
        The set of edges in the backend with two edges, one per direction, per coupler.
        """
        return self.get_edges(bidir=True)

    @functools.cached_property
    def arch_hash(self) -> int:
        """
        Computes an integer hash of the given backend's architecture. Two backends with the same
        number of qubits, coupling map, and supported operations should hash to the same value. This
        can be used to see if two backends are equivalent (enough) as transpilation targets, to for
        example reduce the number of times a circuit is transpiled.

        Note: Python hashes strings with a random seed, so these hashes are consistent **only**
        within the same session.
        """
        return hash(
            (
                self.num_qubits,
                tuple(sorted(self.coupling_map.get_edges())),
                tuple(sorted(self.operation_names)),
            ),
        )

    def run(
        self,
        *args,
        callback: Callable[[QiskitJob, QiskitJobResult], Any] | None = None,
        **kwargs,
    ):
        """
        Executes a circuit on this backend. This is a thin wrapper around the real backend's
        `.run()` method with most args and kwargs left intact.

        Params:
            callback:
                An optional callback function that will be called when the job finishes. A new
                thread is started to wait for job completition. This thread is also responsible for
                invoking the callback, so please ensure that operations carried out by the callback
                are thread-safe.

                The callback function receives two arguments: the remote Qiskit job and its result.
        """

        job = typing.cast(QiskitJob, self._backend.run(*args, **kwargs))
        if callable(callback):
            callback_thread = threading.Thread(
                target=lambda: callback(job, job.result()),
                name=f"JobCallback<{job.job_id()}>",
                daemon=True,
            )
            callback_thread.start()
        return job
