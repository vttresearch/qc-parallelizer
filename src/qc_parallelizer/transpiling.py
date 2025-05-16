import qiskit
import qiskit.dagcircuit
import qiskit.providers
import qiskit.transpiler
import qiskit.transpiler.passes
import z3

from qiskit.transpiler.preset_passmanagers.common import generate_translation_passmanager

from .base import Exceptions, Types
from .generic import circuittools, generic, layouts


class LogPass(qiskit.transpiler.basepasses.AnalysisPass):
    """
    A debug pass that just prints a custom message and some variables. Useful for inspecting the
    pass manager's progress.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__()

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> qiskit.dagcircuit.DAGCircuit:
        from qiskit.converters.dag_to_circuit import dag_to_circuit

        print("V" * 32)
        print("LogPass", self.message)
        print("dag has", dag.num_qubits(), "qubits:")
        print(dag_to_circuit(dag))
        print("property set is", self.property_set)
        print("A" * 32)
        return dag


class StashLayout(qiskit.transpiler.basepasses.AnalysisPass):
    """
    A custom AnalysisPass that removes the normal layout property and moves it into a "stash". This
    is to avoid processing or applying the layout to the circuit in the pass manager. Without this,
    visualizing a circuit with a partial layout fails, for example.

    SabreLayout is a special case, because it generates a physical layout that is likely to reorder
    qubits and invalidate indices. However, since the generated layout is linear (i.e. a "trivial
    layout"), we can just regenerate a trivial layout for it. The `trivial_layout_condition` can be
    used to specify a condition (a predicate function, with signature `PropertySet -> bool`) that
    determines how the layout should be handled - if the predicate returns a truthy value, a trivial
    layout is generated.

    After the pass has finished, the stashed layout can be retrieved with `.get()`.
    """

    stashed_layout: layouts.QILayout | None = None

    def __init__(self, trivial_layout_condition: bool):
        self.trivial_layout_condition = trivial_layout_condition
        super().__init__()

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> qiskit.dagcircuit.DAGCircuit:
        if self.property_set["layout"]:
            if self.trivial_layout_condition(self.property_set):
                layout = layouts.QILayout.from_trivial(dag.num_qubits())
            else:
                layout = layouts.QILayout.from_property_set(self.property_set, dag)
            self.stashed_layout = layout
            del self.property_set["layout"]
        return dag

    def get(self):
        return self.stashed_layout


class AssertLayoutSuccessOrRaise(qiskit.transpiler.basepasses.AnalysisPass):
    """
    A simple AnalysisPass that asserts that the result of an earlier CheckMap pass was successful.
    If the mapping is not valid, this pass raises an `InvalidLayout` exception.
    """

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> qiskit.dagcircuit.DAGCircuit:
        if self.property_set["is_swap_mapped"] == False:
            reason = self.property_set["check_map_msg"]
            raise Exceptions.InvalidLayout(
                "layout check failed - is the provided layout compatible with the backend?"
                + (f" (reason: {reason})" if reason else ""),
            )


# Unused, but left in for reference in case it's needed later.
"""
class CompleteLayout(qiskit.transpiler.basepasses.TransformationPass):
    def __init__(self, target: qiskit.transpiler.Target):
        self.target = target
        super().__init__()

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> qiskit.dagcircuit.DAGCircuit:
        print(self.target.num_qubits, "qubits")
        if self.property_set["layout"]:
            layout: qiskit.transpiler.Layout = self.property_set["layout"]
            unmapped_qubits = {qubit for qubit in dag.qubits if qubit not in layout}
            free_phys = set(range(self.target.num_qubits)) - set(layout.get_physical_bits().keys())
            if len(free_phys) < len(unmapped_qubits):
                raise Exceptions.CircuitBackendCompatibility(
                    "target does not contain enough qubits to complete layout",
                )
            for qubit in unmapped_qubits:
                layout.add(qubit, free_phys.pop())
            if len(free_phys) > 0:
                unused = qiskit.QuantumRegister(len(free_phys), "unused")
                for phys, virt in zip(unused, free_phys):
                    layout.add(phys, virt)
            self.property_set["layout"] = layout
        return dag
"""


def adjust_indices(indices: set[int], blocked: set[int]) -> tuple[dict[int, int], dict[int, int]]:
    """
    Small helper for adjusting qubit indices based on a set of blocked qubits. Returns a `dict` of
    the mapped indices and an inverse mapping.

    This is best explained visually:
    ```
    0 [blocked] -> None  ,----> 0
                        /
    1 -----------------'  ,---> 1
                         /
    2 [blocked] -> None /   ,-> 2
                       /   /
    3 ----------------'   /
                         /
    4 ------------------'
    ```
    """

    def adjust(index):
        if index in blocked:
            return None
        blocked_below = len({other for other in blocked if other < index})
        return index - blocked_below

    mapping = {index: adjust(index) for index in indices}
    return mapping, {b: a for a, b in mapping.items() if b is not None}


def determine_layout(
    circuit: qiskit.QuantumCircuit,
    layout: layouts.QILayout,
    backend: Types.Backend,
    allow_swap: bool = True,
    transpiler_seed: int | None = None,
) -> tuple[qiskit.QuantumCircuit, layouts.QILayout]:
    """
    Attempts to find a layout for the given circuit that is compatible with the given backend while
    respecting the given layout.

    If a "perfect" layout is found, that is, if all qubits can be mapped within all constraints
    without requiring any SWAP gates, the circuit is returned unmodified along with the discovered
    layout.

    If such a layout cannot be found, Qiskit's implementation of SABRE is used to compute a mostly
    optimal layout with SWAPs inserted to fit coupling constraints. This respects blocked qubits,
    but disregards layout information otherwise. So, if it is crucial that the layout is used, set
    `allow_swap` to `False`. This will cause an exception to be thrown instead.
    """

    phys_coupling = backend.coupling_map.get_edges()
    virt_coupling = generic.get_edges(circuittools.get_neighbor_sets(circuit))
    blocked = layout.get_blocked()

    # This list holds the physical indices of each virtual qubit in the circuit.
    placements = [z3.Int(f"vqb{v}") for v in range(circuit.num_qubits)]

    solver = z3.Optimize()

    # All physical indices must be unique. No two virtual qubits can be placed on the same physical
    # qubit.
    solver.add(z3.Distinct(*placements))

    # All indices must fit on the physical backend. Formally, each index must be in [0, num_qubits).
    for index in placements:
        solver.add(index >= 0, index < backend.num_qubits)

    # The given layout must be respected.
    for v, p in layout.v2p.items():
        solver.add(placements[v] == p)

    # For each blocked physical qubit, no virtual qubit can be placed there.
    for b in blocked:
        solver.add(*[p != b for p in placements])

    # If two qubits couple virtually, they must also couple physically.
    for va, vb in virt_coupling:
        solver.add(
            z3.Or(
                *[z3.And(placements[va] == pa, placements[vb] == pb) for pa, pb in phys_coupling],
            ),
        )

    # And lastly, an optimization heuristic: we want to minimize the number of used couplers. This
    # reduces "wasted" couplers and should push circuits into the backend's corners, leaving more
    # space for other circuits.
    coupler_usage = sum(
        z3.Or(*[z3.Or(p == a, p == b) for p in placements]) for a, b in phys_coupling
    )
    #solver.minimize(coupler_usage)

    #print("Solving... ", end="", flush=True)
    result = solver.check()
    #print(result)
    if result == z3.sat:
        # A perfect layout was found, so we return it.
        model = solver.model()
        return circuit, layouts.QILayout(
            v2p={v: model[placement].as_long() for v, placement in enumerate(placements)},
        )
    elif not allow_swap:
        # Otherwise, if SWAPs are not allowed, report failure. This hijacks Qiskit's own exception
        # class, but it is fit for this purpose.
        raise qiskit.transpiler.TranspilerError(
            "circuit does not fit backend with given layout",
        )

    # Now, if SWAPs were allowed, we have to introduce SWAPs to fit constraints. Instead of
    # reinventing the wheel, we rely on Qiskit's implementation of SABRE. This requires constructing
    # a limited view of the coupling map with blocked qubits removed.

    reduced_phys_qubits, _ = adjust_indices(range(backend.num_qubits), blocked)
    reduced_coupling_map = [
        (reduced_phys_qubits[a], reduced_phys_qubits[b])
        for a, b in phys_coupling
        if a not in blocked and b not in blocked
    ]
    sabre_passman = qiskit.transpiler.PassManager(
        qiskit.transpiler.passes.SabreLayout(
            coupling_map=qiskit.transpiler.CouplingMap(
                couplinglist=reduced_coupling_map,
            ),
            seed=transpiler_seed,
        ),
    )
    swapped_circuit = sabre_passman.run(circuit)

    # Now we have a circuit whose virtual qubits map 1-to-1 to the reduced backend's physical
    # qubits. Thus, we generate a trivial layout for it, and we're done!

    return swapped_circuit, layouts.QILayout.from_trivial(swapped_circuit.num_qubits)


def transpile_to_layout(
    circuit: qiskit.QuantumCircuit,
    backend: qiskit.providers.BackendV2,
    initial_layout: layouts.QILayout | None = None,
    transpiler_seed: int | None = None,
) -> tuple[qiskit.QuantumCircuit, layouts.QILayout]:
    """
    Determines a layout for the given circuit on the given backend, while respecting the layout
    given in `initial_layout` (a `QILayout` object, NOT Qiskit's `Layout`). This allows for certain
    qubits to be blocked before determining the layout. Due to limitations in Qiskit, the layout
    must otherwise (so, for other than blocked qubits) be empty OR completed for the whole circuit.

    Note that if the circuit contains gates that operate on three qubits or more, they are
    transpiled into equivalent gates with at most two qubits. Because of this, and because some
    stages may reorder qubits, the returned layout is only valid for the returned circuit, not the
    original one. Furthermore, quantum register names are lost (in most cases) during this process,
    and the returned circuit contains only a single quantum register for all physical qubits.

    The parameter `transpiler_seed` is passed to transpiler stages that accept such a parameter. It
    can be used to produce consistent results.
    ```
    """

    if not isinstance(initial_layout, layouts.QILayout):
        initial_layout = layouts.QILayout(initial_layout or {})

    blocked = initial_layout.get_blocked()
    reduced_mapping, _ = adjust_indices(range(backend.num_qubits), blocked)
    reduced_coupling_map = [
        (reduced_mapping[a], reduced_mapping[b])
        for a, b in backend.coupling_map.get_edges()
        if a not in blocked and b not in blocked
    ]
    target = qiskit.transpiler.Target.from_configuration(
        basis_gates=backend.operation_names,
        num_qubits=backend.num_qubits - len(blocked),
        coupling_map=qiskit.transpiler.CouplingMap(
            couplinglist=reduced_coupling_map,
        ),
    )

    transpiled = qiskit.transpiler.PassManager(
        [
            qiskit.transpiler.passes.Unroll3qOrMore(target=target),
            *generate_translation_passmanager(
                target=backend.target,
            ).passes()[0]["passes"],
        ],
    ).run(circuit)

    result, layout = determine_layout(
        transpiled,
        initial_layout,
        backend,
        allow_swap=False,
        transpiler_seed=transpiler_seed,
    )

    return result, layout
