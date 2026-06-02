import iqm.qiskit_iqm as iqm
import matplotlib.pyplot as plt
import qiskit.providers
import qiskit.transpiler
import qiskit.visualization
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from qc_parallelizer.util import IndexedLayout


def plot_circuits(
    circuits: list[qiskit.QuantumCircuit] | list[list[qiskit.QuantumCircuit]],
    figsize=None,
    **kwargs,
):
    """
    Plots a (1D or 2D) list of circuits in a grid.
    """

    if len(circuits) == 0:
        raise ValueError("no circuits to plot")

    # Force conversion to 2D array if 1D array was passed
    circuit_table: list[list[qiskit.QuantumCircuit]]
    if isinstance(circuits[0], qiskit.QuantumCircuit):
        circuit_table = [circuits]  # type: ignore
    else:
        circuit_table = circuits  # type: ignore

    nrows, ncols = len(circuit_table), max(len(row) for row in circuit_table)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, dpi=300, figsize=figsize)
    circuit_index = 0
    for i, row in enumerate(circuit_table):
        for j, circuit in enumerate(row):
            ax = (
                axs[i, j]
                if nrows > 1 and ncols > 1
                else axs[j]
                if ncols > 1
                else axs[i]
                if nrows > 1
                else axs
            )
            circuit.draw(output="mpl", style="clifford", ax=ax, **kwargs)
            ax.set_title(circuit.name or f"Circuit {circuit_index}")
            circuit_index += 1
    plt.tight_layout()
    plt.show()


def plot_histograms(counts: list[dict], figsize=None, **kwargs):
    """
    Plots a list of result histogram dicts, side by side.
    """

    fig, axs = plt.subplots(nrows=1, ncols=len(counts), dpi=300, figsize=figsize)
    for i, count_dict in enumerate(counts):
        ax = axs[i] if len(counts) > 1 else axs
        qiskit.visualization.plot_histogram(count_dict, ax=ax, **kwargs)
        ax.set_title(f"Histogram {i}")
    plt.tight_layout()
    plt.show()


def enable_logging(name: str):
    """
    Enables logging for a logger with `name`, at the DEBUG level, with a sensible log formatter.
    """

    import logging

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)-32s | %(levelname)-5s | %(message)s"),
    )
    logger.addHandler(handler)


def patch_qiskit_iqm():
    """
    Patches a Qiskit instruction name for the 'cc_prx' (classically-controlled PRX) gate. The
    patched value is the value for the 'prx' gate. So, if you use the cc variant, expect broken
    behaviour.

    If not patched, this will cause an exception to the raised when running on fake IQM backends.
    """

    table = iqm.iqm_backend.IQM_TO_QISKIT_GATE_NAME
    table["cc_prx"] = table["prx"]
