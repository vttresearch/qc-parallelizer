import qiskit.providers


def get_neighbor_sets(backend: qiskit.providers.BackendV2) -> list[set[int]]:
    """
    Returns sets of physical neighbors in the backend's topology.
    """

    edges = backend.coupling_map.get_edges()
    neighbors = [set() for _ in range(backend.num_qubits)]
    for from_, to in edges:
        neighbors[from_].add(to)
        neighbors[to].add(from_)
    return neighbors


def get_edges(backend: qiskit.providers.BackendV2, bidir: bool = False) -> list[tuple[int, int]]:
    """
    Returns a list of couplers, or edges, in the backend. Each edge is represented by a tuple of two
    ints, each int representing the physical index of the two qubits that it joins.

    Args:
        bidir: If set to True, edges in both directions are returned. By default, each edge is
               listed just once, with the two indices in ascending order.
    """
    raw_edges = backend.coupling_map.get_edges()  # these may be bidirectional
    edge_set = set()
    for a, b in raw_edges:
        a, b = min(a, b), max(a, b)
        edge_set.add((a, b))
        if bidir:
            edge_set.add((b, a))
    return list(edge_set)


def arch_hash(backend: qiskit.providers.BackendV2) -> int:
    """
    Computes an integer hash of the given backend's architecture. Two backends with the same number
    of qubits, coupling map, and supported operations should hash to the same value. This can be
    used to see if two backends are equivalent (enough) as transpilation targets, to for example
    reduce the number of times a circuit is transpiled.

    Note: Python hashes strings with a random seed, so these hashes are consistent **only** within
    the same session.
    """
    return hash(
        (
            backend.num_qubits,
            tuple(sorted(backend.coupling_map.get_edges())),
            tuple(sorted(backend.operation_names)),
        ),
    )
