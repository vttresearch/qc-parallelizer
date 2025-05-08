def get_connected_qubit_sets(
    neighbors: list[set[int]],
) -> list[set[int]]:
    """
    Returns a list of sets of connected qubits. Qubits are separated into two different sets if
    there is no path along the gates/couplers between them.

    The `neighbors` parameter can be computed with `get_neighbot_sets` from either `circuit_tools`
    or `backend_tools`, depending on which you are working with.
    """

    not_seen = set(range(len(neighbors)))
    connected_sets = []

    while len(not_seen) > 0:
        seen = set()
        search_set = {not_seen.pop()}
        while len(search_set) > 0:
            qubit_index = search_set.pop()
            seen.add(qubit_index)
            for nb in neighbors[qubit_index]:
                if nb in not_seen:
                    not_seen.remove(nb)
                    search_set.add(nb)
        connected_sets.append(seen)

    return connected_sets


def get_edges(
    neighbors: list[set[int]],
) -> set[tuple[int, int]]:
    """
    Returns a set of edges between qubits. If there is a gate (virtual) or coupler (physical)
    between two qubits, there is an edge between them.

    Edges are not duplicated for both directions, but only one edge is returned per edge with the
    indices sorted in ascending order.
    """

    edges = set()
    for from_, nb_set in enumerate(neighbors):
        for to in nb_set:
            edges.add((min(from_, to), max(from_, to)))
    return edges
