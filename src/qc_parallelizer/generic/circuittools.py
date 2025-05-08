import warnings
from typing import Any

import qiskit

from . import layouts


def count_gates(circuit: qiskit.QuantumCircuit):
    """
    Returns gate counts for each qubit in the given circuit, including measurements, but excluding
    barriers. For example, for a circuit that looks like this:

    ```
         ┌───┐          ┌─┐
    q_0: ┤ H ├───────■──┤M├───
         ├───┤┌───┐┌─┴─┐└╥┘┌─┐
    q_1: ┤ X ├┤ H ├┤ X ├─╫─┤M├
         ├───┤└┬─┬┘└───┘ ║ └╥┘
    q_2: ┤ H ├─┤M├───────╫──╫─
         └───┘ └╥┘       ║  ║
    c: 3/═══════╩════════╩══╩═
                2        0  1
    ```

    The returned counts would look like this:

    ```
    {q_0: 3, q_1: 4, q_2: 2}
    ```
    """

    gate_count = {qubit: 0 for qubit in circuit.qubits}
    for operation, qubits, _ in circuit.data:
        if operation.name == "barrier":
            continue
        for qubit in qubits:
            gate_count[qubit] += 1
    return gate_count


def remove_idle_qubits(
    circuit: qiskit.QuantumCircuit,
    layout: layouts.QILayout | None = None,
    allow_layout_replacement: bool = True,
):
    """
    Removes idle qubits from a circuit. An idle qubit is a qubit that no operations, including
    measurements, touch during the execution of the circuit. A layout for the circuit may also be
    provided via the `layout` parameter, which will be updated to not contain the idle qubits.

    Registers are immutable, so the operation is done by reconstructing the circuit gate by gate
    with a reduced set of registers. Register names are lost in the process, and all remaining
    active qubits are placed in a new quantum register.

    If the circuit contains layout information, which is the case after transpilation, this function
    will read that information to determine original register names. The resulting pruned circuit
    will not contain layout information, since removing qubits violates some assumptions that other
    functionality in Qiskit relies on. If you wish to force this function to ignore the layout data,
    though, set `allow_layout_replacement` to `False` - this will use original register names, but
    not the embedded layout.
    """

    if circuit.layout is not None and allow_layout_replacement:
        layout = layouts.QILayout.from_circuit(circuit)

    gate_count = count_gates(circuit)
    idle_indices = {index for index, qubit in enumerate(circuit.qubits) if gate_count[qubit] == 0}
    if len(idle_indices) == 0:
        if layout is None:
            return circuit
        return circuit, layout

    active_qubits = [
        qubit for index, qubit in enumerate(circuit.qubits) if index not in idle_indices
    ]

    qreg_mapping = {}
    new_qreg = qiskit.QuantumRegister(len(active_qubits))
    for new_index, old_qubit in enumerate(active_qubits):
        qreg_mapping[old_qubit] = new_qreg[new_index]

    new_circuit = qiskit.QuantumCircuit(
        new_qreg,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )

    for operation, qubits, clbits in circuit.data:
        new_qubits = [qreg_mapping[qubit] for qubit in qubits if qubit in qreg_mapping]
        if len(new_qubits) != len(qubits):
            # Some operations need to be adjusted. Currently, this is only the case for barriers,
            # since they "operate" on registers, but they do not affect a qubit's activity. So, if
            # we encounter a barrier that was placed partially on active qubits, we lower the qubit
            # count. This does not seem to have any side effects.

            operation = operation.copy()
            operation.num_qubits = len(new_qubits)
        new_circuit.append(operation, new_qubits, clbits)

    if layout is not None:
        new_layout = layout.copy()

        # Since we are dealing with indices, which, inherently, keep pointing at the same index even
        # if the underlying array shifts, we must iterate in decreasing order to not invalidate
        # later indices
        modified = False
        for index in sorted(idle_indices, reverse=True):
            if index in new_layout.vkeys:
                new_layout.remove(virt=index, decrement_keys=True)
                modified = True

        # If the layout was not modified, the new object is discarded and the old layout is returned
        # instead - this avoids unnecessary copies in memory
        return new_circuit, (new_layout if modified else layout)

    return new_circuit


def pad_to_width(circuit: qiskit.QuantumCircuit, width: int, in_place=True):
    """
    Pads a circuit with unused qubits to the specified width. The newly added padding qubits will be
    placed in a new single quantum register, called "padding".
    """

    num_padding = width - circuit.num_qubits
    if num_padding > 0:
        if not in_place:
            circuit = circuit.copy()
        circuit.add_register(qiskit.QuantumRegister(num_padding, name="padding"))
    return circuit


def get_neighbor_sets(circuit: qiskit.QuantumCircuit) -> list[set[int]]:
    """
    Returns a list of sets of indices that represent all neighbors that qubits have in the given
    circuit. For non-transpiled circuits, this may also mean other than physically immediate
    neighbors - as a counterexample, a qubit that interacts with every other qubit will have all
    other qubits in its neighbor set, even if that is physically impossible.

    From another point of view, each set represents which qubits the corresponding qubit interacts
    with during the execution of the circuit. Barriers are not counted as interaction.

    For example, for the circuit below,

    ```
    q_0: ──■────■────■──
           │  ┌─┴─┐  │
    q_1: ──■──┤ X ├──┼──
         ┌─┴─┐└───┘  │
    q_2: ┤ X ├───────┼──
         └───┘     ┌─┴─┐
    q_3: ──────────┤ X ├
                   └───┘
    ```

    this function returns

    ```
    [{1, 2, 3}, {0, 2}, {0, 1}, {0}]
    ```
    """

    neighbors = [set() for _ in range(circuit.num_qubits)]
    for operation, qubits, _ in circuit.data:
        if operation.name == "barrier":
            continue
        qubit_indices = [circuit.find_bit(qb).index for qb in qubits]
        for i, qb_i in enumerate(qubit_indices):
            for qb_j in qubit_indices[i + 1 :]:
                neighbors[qb_i].add(qb_j)
                neighbors[qb_j].add(qb_i)
    return neighbors


def combine_for_backend(
    circuits: list[tuple[qiskit.QuantumCircuit, Any, layouts.QILayout]],
    backend: qiskit.providers.BackendV2,
    name: str | None = None,
):
    """
    This function combines multiple circuits into a larger host circuit in a backend-aware manner.
    It requires a list of circuits, each of which must be represented by a (circuit, metadata,
    layout) triple. The returned circuit's width is the same as the backend size.

    The second element of the triple (metadata) is embedded into the circuit's "internal metadata".
    This allows for internal metadata to be stored, such as the index of the circuit in some
    original ordering, while preserving the actual metadata of the circuit.

    The resulting circuit will contain metadata about the hosted circuits that the user should not
    touch.
    """

    host_circuit = qiskit.QuantumCircuit(
        backend.num_qubits,
        metadata={"_hosted_circuits": []},
        name=name or f"{backend.num_qubits}-qubit host",
    )

    for index, (subcircuit, metadata, layout) in enumerate(circuits):
        creg_mapping = {}
        for old_reg in subcircuit.cregs:
            new_reg = qiskit.ClassicalRegister(old_reg.size, name=f"circ{index}.{old_reg.name}")
            for i in range(old_reg.size):
                creg_mapping[old_reg[i]] = new_reg[i]
            host_circuit.add_register(new_reg)

        host_circuit.metadata["_hosted_circuits"].append(
            {
                "name": subcircuit.name,
                "original_metadata": subcircuit.metadata,
                "internal_metadata": metadata,
                "registers": {
                    "clbit": {"sizes": {f"{reg.name}": reg.size for reg in subcircuit.cregs}},
                },
            },
        )

        qreg_indices = {
            virt_qubit: subcircuit.find_bit(virt_qubit).index for virt_qubit in subcircuit.qubits
        }
        qreg_mapping = {
            virt_qubit: host_circuit.qubits[layout[qreg_indices[virt_qubit]]]
            for virt_qubit in subcircuit.qubits
            if qreg_indices[virt_qubit] in layout
        }

        for operation, qubits, clbits in subcircuit.data:
            if any(qubit not in qreg_mapping for qubit in qubits):
                print(f"oh noes! instruction '{operation.name}' skipped")  # TODO: warn
                continue
            host_circuit.append(
                operation,
                [qreg_mapping[qubit] for qubit in qubits],
                [creg_mapping[clbit] for clbit in clbits],
            )

    return host_circuit
