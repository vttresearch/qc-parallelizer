from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot
import qiskit
import qiskit.visualization.utils
from qc_parallelizer.extensions import Backend, Circuit


def plot_placements(
    rearranged: dict[Backend, Sequence[qiskit.QuantumCircuit]],
    figsize=None,
):
    """
    Plots sets of rearranged circuits on the backends that they were placed on.

    Args:
        rearranged:
            A dict of rearranged circuits, as returned by `rearrange()`.
        figsize:
            The figure size. Passed directly to Matplotlib.

    Returns:
        A Matplotlib Figure. If used in a notebook, this function takes care of closing unwanted
        duplicates that may be displayed automatically.
    """

    def get_qubit_colors(qubit_indices, num_qubits):
        indices = [None] * num_qubits
        for i, qubits in enumerate(qubit_indices):
            for q in qubits:
                indices[q] = i
        color_table = ["#dd0000", "#008800", "#0000bb", "#0099aa", "#aa00aa", "#aa9900"]
        return [color_table[i % 6] if i is not None else "grey" for i in indices]

    def get_coupler_colors(coupler_lists, edges):
        all_couplers = {c for circuit_couplers in coupler_lists for c in circuit_couplers}
        return ["black" if (a, b) in all_couplers else "grey" for a, b in edges]

    nrows, ncols = len(rearranged), max(len(clist) for clist in rearranged.values())
    fig, axs = matplotlib.pyplot.subplots(
        nrows=nrows,
        ncols=ncols,
        dpi=100,
        figsize=figsize,
        squeeze=False,
    )
    for i, (backend, circuits) in enumerate(rearranged.items()):
        for j, circuit in enumerate(circuits):
            qubit_indices = [h["qubits"] for h in circuit.metadata["hosted_circuits"]]
            coupler_lists = [h["couplers"] for h in circuit.metadata["hosted_circuits"]]
            qubit_colors = get_qubit_colors(qubit_indices, backend.num_qubits)
            coupler_colors = get_coupler_colors(coupler_lists, backend.edges)
            qiskit.visualization.plot_coupling_map(
                num_qubits=backend.num_qubits,
                qubit_coordinates=None,
                coupling_map=backend.edges,
                ax=axs[i, j],
                planar=False,
                qubit_size=80,
                font_size=24,
                line_width=25,
                qubit_color=qubit_colors,
                line_color=coupler_colors,
            )
        for j in range(len(circuits), ncols):
            axs[i, j].remove()
    matplotlib.pyplot.tight_layout()
    qiskit.visualization.utils.matplotlib_close_if_inline(fig)
    return fig
