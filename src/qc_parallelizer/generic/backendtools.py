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
