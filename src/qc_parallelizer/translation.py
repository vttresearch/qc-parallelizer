import itertools
from collections.abc import Callable, Sequence
from typing import Literal

import qiskit
import qiskit.dagcircuit
import qiskit.providers
import qiskit.transpiler
import rustworkx
import z3

from .base import Exceptions, Types
from .generic import backendtools, circuittools, generic
from .generic.layouts import CircuitWithLayout, IndexedLayout
from .generic.logging import Log


def translate_for_backend(
    circuit: CircuitWithLayout | qiskit.QuantumCircuit,
    backend: Types.Backend,
    optimization_level: int = 1,
    **pm_kwargs,
) -> CircuitWithLayout | None:
    """
    Translates or unrolls a circuit for the given backend. Three-qubit or larger gates will be
    translated to multiple up-to-two-qubit gates, after which all gates will be translated to only
    operations that the backend natively supports.

    Args:
        pm_kwargs:
            Passed to PassManagerConfig. Allows configuring translation method, for example.

    Returns:
        Translated circuit, or None if the circuit could not be translated.
    """

    if isinstance(circuit, qiskit.QuantumCircuit):
        circuit = CircuitWithLayout(circuit, None)

    Log.debug(
        f"Translating |{circuit.circuit.num_qubits}-qubit| circuit for backend |'{backend.name}'|.",
    )

    # We create a fake target with couplers exactly where they are needed by the circuit. This way
    # the circuit is transpiled in place with no routing/layout.

    target = qiskit.transpiler.Target.from_configuration(
        basis_gates=backend.operation_names,
        num_qubits=backend.num_qubits,
        coupling_map=qiskit.transpiler.CouplingMap(
            couplinglist=generic.get_edges(circuittools.get_neighbor_sets(circuit.circuit)),
        )
        if circuit.circuit.num_nonlocal_gates() > 0
        else None,
        # For some reason circuits with only 1-qubit gates fail without this correction.
    )

    pm = qiskit.transpiler.generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        target=target,
    )

    # These have to be manually removed, but this seems to work fine.
    pm.layout = None
    pm.routing = None

    try:
        return CircuitWithLayout(pm.run(circuit.circuit), circuit.layout)
    except (qiskit.transpiler.TranspilerError, KeyError) as error:
        Log.warn(f"Could not translate circuit for backend due to error |'{error}'|.")
        return None


class CircuitBackendTranslations:
    """
    Represents a list of circuits translated against a list of backends.
    """

    translation_table: dict[int, dict[int, CircuitWithLayout]]
    arch_backends: dict[int, list[qiskit.providers.BackendV2]]
    optimal_backends: dict[int, list[Types.Backend]]

    def __init__(
        self,
        trans_table: dict[int, dict[int, CircuitWithLayout]],
        backend_arches: dict[Types.Backend, int],
        arch_backends: dict[int, list[Types.Backend]],
        optimal_backends: dict[int, list[Types.Backend]],
    ):
        self.translation_table = trans_table
        self.backend_arches = backend_arches
        self.arch_backends = arch_backends
        self.optimal_backends = optimal_backends

    @classmethod
    def generate(
        cls,
        circuits: Sequence[CircuitWithLayout],
        backends: Sequence[Types.Backend],
        **kwargs,
    ):
        """
        Translates a list of circuits for a list of backends.

        Instead of blindly translating each circuit against each backend, a reduced set of unique
        backend architectures is determined first. This is based on the backends' architecture
        hashes - see `backendtools.arch_hash()` for details. Circuits are also analyzed for
        duplicates, and each kind of circuit is transpiled only once.

        See the `CircuitBackendTranslations` class for more information on the returned object.
        """

        circuit_hashes = [circuittools.circuit_hash(circuit.circuit) for circuit in circuits]

        backend_arches = {backend: backendtools.arch_hash(backend) for backend in backends}
        arch_backends: dict[int, list[Types.Backend]] = {}
        for backend, arch in backend_arches.items():
            arch_backends.setdefault(arch, []).append(backend)

        Log.debug(
            lambda: (
                f"Reduced {len(backend_arches.keys())} backend(s) to "
                f"{len(set(backend_arches.values()))} unique architecture(s)."
            ),
        )

        def translation_helper(backend: Types.Backend):
            """
            Generates translations for the circuit against the given backend. Filters out failed
            translations, which would otherwise be None.
            """
            translations = {
                circuit_hash: translate_for_backend(
                    circuit,
                    backend,
                    **kwargs,
                )
                for circuit, circuit_hash in zip(circuits, circuit_hashes)
            }
            return {
                circuit_hash: translation
                for circuit_hash, translation in translations.items()
                if translation is not None
            }

        translation_table = {
            # Translate circuits for one (first) backend of the architecture
            arch: translation_helper(backends[0])
            for arch, backends in arch_backends.items()
        }

        translated_circuit_set = {
            circuit_hash
            for circuit_translations in translation_table.values()
            for circuit_hash in circuit_translations.keys()
        }

        if len(translated_circuit_set) != len(circuits):
            raise Exceptions.CircuitBackendCompatibility(
                "could not translate circuit for any backend",
            )

        optimal_depths = {
            circuit_hash: min(
                trans_circuits[circuit_hash].circuit.depth()
                for trans_circuits in translation_table.values()
                if circuit_hash in trans_circuits
            )
            for circuit_hash in translated_circuit_set
        }

        optimal_arches = {
            circuit_hash: [
                arch
                for arch, trans_circuits in translation_table.items()
                if circuit_hash in trans_circuits
                if trans_circuits[circuit_hash].circuit.depth() == optimal_depths[circuit_hash]
            ]
            for circuit_hash in translated_circuit_set
        }

        optimal_backends = {
            circuit_hash: [
                backend for bh in optimal_arches[circuit_hash] for backend in arch_backends[bh]
            ]
            for circuit_hash in translated_circuit_set
        }

        return cls(translation_table, backend_arches, arch_backends, optimal_backends)

    def get(
        self,
        circuit: CircuitWithLayout,
        backend: qiskit.providers.BackendV2,
    ) -> CircuitWithLayout:
        """
        Returns the translation for the given circuit against the given backend.
        """

        backend_hash = backendtools.arch_hash(backend)
        circuit_hash = circuittools.circuit_hash(circuit.circuit)
        return self.translation_table[backend_hash][circuit_hash]

    def optimal_backends_for(self, circuit: CircuitWithLayout) -> list[qiskit.providers.BackendV2]:
        circuit_hash = circuittools.circuit_hash(circuit.circuit)
        return self.optimal_backends.get(circuit_hash, [])

    def __str__(self):
        return (
            f"{self.__class__.__name__}< "
            f"{len(self.optimal_backends)} circuit(s) "
            f"against {len(self.arch_backends)} architecture(s) "
            ">"
        )
