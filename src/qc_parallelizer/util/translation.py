from collections.abc import Sequence

import qiskit
import qiskit.providers
import qiskit.transpiler

from qc_parallelizer.base import Exceptions
from qc_parallelizer.extensions import Backend, Circuit
from qc_parallelizer.util import Log


def translate_for_backend(
    circuit: Circuit | qiskit.QuantumCircuit,
    backend: Backend | qiskit.providers.BackendV2,
    optimization_level: int = 1,
    **pm_kwargs,
) -> Circuit | None:
    """
    Translates a circuit for the given backend. Three-qubit or larger gates will be translated to
    multiple up-to-two-qubit gates, after which all gates will be translated to only operations that
    the backend natively supports.

    Args:
        pm_kwargs:
            Passed to PassManagerConfig. Allows configuring translation method, for example.

    Returns:
        Translated circuit, or None if the circuit could not be translated.
    """

    if isinstance(circuit, qiskit.QuantumCircuit):
        circuit = Circuit(circuit)

    Log.debug(
        f"Translating |{circuit.num_qubits}-qubit| circuit for backend |'{backend.name}'|.",
    )

    # We create a fake target with couplers exactly where they are needed by the circuit. This way
    # the circuit is transpiled in place with no routing/layout.

    target = qiskit.transpiler.Target.from_configuration(
        basis_gates=backend.operation_names,
        num_qubits=backend.num_qubits,
        coupling_map=qiskit.transpiler.CouplingMap(
            couplinglist=circuit.get_edges(),
        )
        if circuit.num_nonlocal_gates > 0
        else None,
        # For some reason circuits with only 1-qubit gates fail without this correction.
    )

    pm = qiskit.transpiler.generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend.unwrap() if isinstance(backend, Backend) else backend,
        target=target,
        **pm_kwargs,
    )

    # These have to be manually removed, but this seems to work fine.
    pm.layout = None
    pm.routing = None

    try:
        return Circuit(pm.run(circuit.unwrap()), circuit.layout)
    except (qiskit.transpiler.TranspilerError, KeyError) as error:
        Log.warn(f"Could not translate circuit for backend due to error |'{error}'|.")
        return None


class CircuitBackendTranslations:
    """
    Represents a list of circuits translated against a list of backends.
    """

    translation_table: dict[int, dict[int, Circuit]]
    arch_backends: dict[int, list[Backend]]
    optimal_backends: dict[int, list[Backend]]

    def __init__(
        self,
        trans_table: dict[int, dict[int, Circuit]],
        backend_arches: dict[Backend, int],
        arch_backends: dict[int, list[Backend]],
        optimal_backends: dict[int, list[Backend]],
    ):
        self.translation_table = trans_table
        self.backend_arches = backend_arches
        self.arch_backends = arch_backends
        self.optimal_backends = optimal_backends

    @classmethod
    def generate(
        cls,
        circuits: Sequence[Circuit],
        backends: Sequence[Backend],
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

        circuit_hashes = [circuit.hash() for circuit in circuits]

        backend_arches = {backend: backend.arch_hash for backend in backends}
        arch_backends: dict[int, list[Backend]] = {}
        for backend in backends:
            arch_backends.setdefault(backend.arch_hash, []).append(backend)

        Log.debug(
            lambda: (
                f"Reduced {len(backend_arches.keys())} backend(s) to "
                f"{len(set(backend_arches.values()))} unique architecture(s)."
            ),
        )

        def translation_helper(backend: Backend):
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
                trans_circuits[circuit_hash].depth
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
                if trans_circuits[circuit_hash].depth == optimal_depths[circuit_hash]
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
        circuit: Circuit,
        backend: Backend,
    ) -> Circuit:
        """
        Returns the translation for the given circuit against the given backend.
        """

        return self.translation_table[backend.arch_hash][circuit.hash()]

    def optimal_backends_for(self, circuit: Circuit) -> list[Backend]:
        """
        Returns optimal backends for circuit. In a set of backends, the backends that can natively
        execute the circuit with the lowest transpiled depth are considered optimal.
        """

        return self.optimal_backends.get(circuit.hash(), [])

    def __str__(self):
        return (
            f"{self.__class__.__name__}< "
            f"{len(self.optimal_backends)} circuit(s) "
            f"against {len(self.arch_backends)} architecture(s) "
            ">"
        )
