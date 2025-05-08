import qiskit
import qiskit.dagcircuit
import qiskit.providers
import qiskit.transpiler
import qiskit.transpiler.passes
import qiskit.transpiler.preset_passmanagers.common
import qiskit.circuit.equivalence_library as equivalence_library

# One name is exceptionally imported directly into globals since the fully qualified name is just
# too damn long :D
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
import qiskit.transpiler.preset_passmanagers
from vtt_quantumutils.common import circuittools, layouts
from vtt_quantumutils.parallelizer.base import Exceptions, Types


class StashLayout(qiskit.transpiler.basepasses.AnalysisPass):
    """
    A custom AnalysisPass that removes the normal layout property and moves it into a "stash". This
    is to avoid processing or applying the layout to the circuit in the passmanager. Without this,
    visualizing a circuit with a partial layout fails, for example.

    SabreLayout is a special case, because it generates a physical layout that is likely to reorder
    qubits and invalidate indices. However, since the generated layout is linear (i.e. a "trivial
    layout"), we can just regenerate a trivial layout for it. The `trivial_layout_condition` can be
    used to specify a condition (predicate, `PropertySet -> bool`) that determines how the layout
    should be handled - if the predicate returns a truthy value, a trivial layout is generated.

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


class CheckLayoutOrRaise(qiskit.transpiler.basepasses.AnalysisPass):
    """
    A simple AnalysisPass that checks the result of an earlier CheckMap pass. If the mapping is not
    valid, this pass raises an `InvalidLayout` exception.
    """

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> qiskit.dagcircuit.DAGCircuit:
        if self.property_set["is_swap_mapped"] == False:
            reason = self.property_set["check_map_msg"]
            raise Exceptions.InvalidLayout(
                "layout check failed - is the provided layout compatible with the backend?"
                + (f" (reason: {reason})" if reason else ""),
            )


class CompleteLayout(qiskit.transpiler.basepasses.TransformationPass):
    def __init__(self, target: qiskit.transpiler.Target):
        self.target = target
        super().__init__()

    def run(self, dag: qiskit.dagcircuit.DAGCircuit) -> qiskit.dagcircuit.DAGCircuit:
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
            self.property_set["layout"] = layout
        return dag


def transpile_to_layout(
    circuit: qiskit.QuantumCircuit,
    base_backend: qiskit.providers.BackendV2,
    initial_layout: layouts.QILayout | None = None,
    transpiler_seed: int | None = None,
) -> tuple[qiskit.QuantumCircuit, layouts.QILayout]:
    """
    Determines a layout for the given circuit on the given backend, while respecting the layout
    given in `initial_layout` (a `QILayout` object, NOT Qiskit's `Layout`). This allows for certain
    qubits to be blocked before determining the layout. Due to limitations in Qiskit, the layout
    must otherwise (so, for other than blocked qubits) be empty or completed for the whole circuit.

    Note that if the circuit contains gates that operate on three qubits or more, they are
    transpiled into equivalent gates with at most two qubits. Because of this, and because some
    stages may reorder qubits, the returned layout is only valid for the returned circuit, not the
    original one. Furthermore, quantum register names are lost (in most cases) during this process,
    and the returned circuit contains only a single quantum register for all active qubits (note:
    and only for active qubits - idle qubits are removed).

    The parameter `transpiler_seed` is passed to transpiler stages that accept such a parameter. It
    can be used to produce consistent results.

    TODO! Missing info about gate translation.

    Internally, the following transpiler pass manager structure is used:
    ```text
              +----------------+      .--------------.          +-----------+     +-------------+
    START --> | Unroll3qOrMore | --> (  Full layout?  ) YES --> | SetLayout | --> | ApplyLayout |
              +----------------+      '--------------'          +-----------+     +-------------+
                                             NO                                            V
                                             V            +---------------------+     +----------+
                                       +-----------+      | CheckLayoutOrRaise* | <-- | CheckMap |
                                       | VF2Layout |      +---------------------+     +----------+
                                       +-----------+                 |
                                             V                       V
             +-------------+          .-------------.         +--------------+
             | SabreLayout | <-- YES (  VF2 failed?  ) NO --> | StashLayout* | --> FINISH
             +-------------+          '-------------'         +--------------+
                    |                                                A
                    '------------------------------------------------'

    * = Custom transpilation pass. Other passes are built into Qiskit.
    ```
    """

    if not isinstance(initial_layout, layouts.QILayout):
        initial_layout = layouts.QILayout(initial_layout or {})

    # Here we remove the blocked indices from both the total number of qubits and the coupling map.
    # However, this isn't as simple as just removing the indices, as the transpiler does not
    # support indices with gaps in the middle. Instead, we present a subset of the real backend to
    # the transpiler *as if* it were the whole backend by removing and adjusting indices to end up
    # with the reduced allowed part of the backend. The only complication is that the result that
    # the transpiler returns is incomplete and does not contain qubits outside of the allowed
    # subset. So, we must re-introduce the qubits into the layout, and adjust indices again.

    def remove_disallowed_indices(coupling_list):
        disallowed = initial_layout.get_blocked()

        all_indices = {i for a, b in coupling_list for i in (a, b)}
        indices_below = {i: len({j for j in disallowed if j < i}) for i in all_indices}
        new_indices = {i: None if i in disallowed else (i - indices_below[i]) for i in all_indices}
        return [
            (a, b)
            for a, b in [(new_indices[a], new_indices[b]) for a, b in coupling_list]
            if a is not None and b is not None
        ]

    coupling_map = qiskit.transpiler.CouplingMap(
        couplinglist=remove_disallowed_indices(base_backend.coupling_map.get_edges()),
    )

    target = qiskit.transpiler.Target.from_configuration(
        basis_gates=base_backend.operation_names,
        num_qubits=base_backend.num_qubits - len(initial_layout.get_blocked()),
        coupling_map=coupling_map,
    )

    def vf2_failed(property_set):
        reason = property_set["VF2Layout_stop_reason"]
        return reason is not None and reason is not VF2LayoutStopReason.SOLUTION_FOUND

    def vf2_succeeded(property_set):
        return not vf2_failed(property_set)

    passman = qiskit.transpiler.PassManager(
        qiskit.transpiler.passes.Unroll3qOrMore(target=target),
    )

    if initial_layout.size == circuit.num_qubits:
        # A full layout was specified - apply it and check validity
        passman.append(
            [
                qiskit.transpiler.passes.SetLayout(initial_layout.to_qiskit_layout(circuit)),
                qiskit.transpiler.passes.ApplyLayout(),
                qiskit.transpiler.passes.CheckMap(coupling_map=coupling_map),
                CheckLayoutOrRaise(),
            ],
        )
    elif initial_layout.size > 0:
        # A partial layout was specified - not supported currently
        raise Exceptions.InvalidLayout(
            "partial layouts are not supported - a layout must specify mappings for zero or all "
            "qubits in the circuit",
        )
    else:
        # No layout was specified - compute one
        passman.append(
            [
                qiskit.transpiler.passes.VF2Layout(
                    seed=transpiler_seed,
                    target=target,
                ),
                qiskit.transpiler.ConditionalController(
                    qiskit.transpiler.passes.SabreLayout(
                        # SabreLayout cannot automatically determine properties from a target, so
                        # the coupling map must be given separately here
                        coupling_map=coupling_map,
                        seed=transpiler_seed,
                    ),
                    condition=vf2_failed,
                ),
            ],
        )

    # TODO: research this more
    passman += qiskit.transpiler.preset_passmanagers.common.generate_translation_passmanager(
        target=base_backend.target,
    )

    passman.append(layout_stash := StashLayout(trivial_layout_condition=vf2_failed))

    transpiled, layout = passman.run(circuit), layout_stash.get()

    # This re-introduces the removed indices and adjusts other indices as needed - after this, the
    # layout is valid for the whole backend
    layout.insert_blocked_indices(initial_layout.get_blocked())

    transpiled.name = circuit.name
    return circuittools.remove_idle_qubits(transpiled, layout=layout)
