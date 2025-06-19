import heapq
from collections.abc import Generator, Sequence
from typing import Any

import qiskit
from qc_parallelizer.base import Exceptions
from qc_parallelizer.extensions import Backend, Circuit
from qc_parallelizer.packers import PackerBase
from qc_parallelizer.util import Log
from qc_parallelizer.util.translation import CircuitBackendTranslations

from .circuitbin import CircuitBin


class CircuitBinManager:
    backends: Sequence[Backend]
    bins: dict[Backend, list[CircuitBin]]
    packer: PackerBase

    def __init__(
        self,
        backends: Sequence[Backend],
        packer: PackerBase,
    ):
        self.backends = backends
        self.bins = {backend: [] for backend in backends}
        self.packer = packer

    def generate_candidate_bins(
        self,
        optimal_backends: dict[Backend, Circuit],
    ) -> Generator[tuple[CircuitBin, Circuit], None, None]:
        candidates: list[CircuitBin] = []
        max_bins = self.packer.max_bins_per_backend or float("inf")

        Log.debug("Generating candidate bins for circuit.")
        Log.debug(f"There are |{len(optimal_backends)}| optimal backend(s).")

        for backend, translated in optimal_backends.items():
            bins = self.bins[backend]

            Log.debug(
                (
                    f"Backend |'{backend.name}'| has ${len(bins)} bin$ with sizes "
                    f"|{[bin.size for bin in bins]}|."
                ),
            )

            has_empty = any(bin.size == 0 for bin in bins)
            if (len(bins) == 0 or not has_empty) and len(bins) < max_bins:
                bins.append(CircuitBin(backend))
                Log.debug(f"Added new empty bin for backend |'{backend.name}'|.")

            for bin in bins:
                if bin.compatible(translated):
                    candidates.append(bin)

        Log.debug(f"Found ${len(candidates)} candidate bin$ for circuit.")

        # Sort bins based on
        # 1. how many bins the backend already has, multiplied by the backend's cost
        # 2. the bin size (taking empty bins last)

        for bin in sorted(
            candidates,
            key=lambda cb: (
                len(self.bins[cb.backend]) * cb.backend.cost,
                cb.size == 0,
                cb.size,
            ),
        ):
            yield bin, optimal_backends[bin.backend]

    def place_circuit(
        self,
        circuit: Circuit,
        translations: CircuitBackendTranslations,
    ):
        optimal_backends = {
            backend: translations.get(circuit=circuit, backend=backend)
            for backend in translations.optimal_backends_for(circuit)
        }

        Log.debug("Attempting to place next circuit in bin set.")

        candidate_placements: list[tuple[Any, CircuitBin, Circuit]] = []
        max_candidates = self.packer.max_candidates or float("inf")

        for candidate_bin, translated in self.generate_candidate_bins(optimal_backends):
            Log.debug("Attempting to place circuit in bin.")

            blocked = self.packer.blocked(candidate_bin)
            assert candidate_bin.taken_indices.issubset(blocked), "qubits cannot overlap"

            completed_layout = self.packer.find_layout(candidate_bin, translated, blocked)

            if completed_layout is None:
                Log.warn(f"Transpilation failed for one bin with |{candidate_bin.backend.name}|.")
                continue

            Log.debug("Bin is suitable for circuit.")
            completed_circuit = translated.with_layout(completed_layout)

            rating = self.packer.evaluate(candidate_bin, completed_circuit)
            Log.debug(f"Got rating |{rating}| for placement.")
            heapq.heappush(candidate_placements, (-rating, candidate_bin, completed_circuit))

            if len(candidate_placements) >= max_candidates:
                Log.debug(f"Maximum candidate count |{max_candidates}| reached!")
                break

        if len(candidate_placements) > 0:
            Log.debug(f"Found |{len(candidate_placements)}| possible placements.")
            _, bin, circuit = candidate_placements[0]
            bin.place(circuit)
            return

        Log.fail("No compatible backends were found!")

        raise Exceptions.CircuitBackendCompatibility(
            "could not place circuit on any backend",
        )

    def place(
        self,
        circuits: list[Circuit],
        allow_ooe: bool,
    ):
        if allow_ooe:
            Log.info("`allow_ooe` is |True|. Sorting circuits.")

            # Sort circuits based on two factors. In the order of precedence,
            #  1. circuits with more layout information,
            #  2. circuits with more connectivity, and
            #  3. larger circuits
            # come first.
            circuits.sort(
                key=lambda circuit: (
                    -circuit.layout.size,
                    circuit.num_connected_components / circuit.num_qubits,
                    -circuit.num_qubits,
                ),
            )

            Log.debug(f"Sorted order: |{[c.metadata['index'] for c in circuits]}|")
            Log.debug(f"Qubit counts: |{[c.num_qubits for c in circuits]}|")
            Log.debug(f"Layout sizes: |{[c.layout.size for c in circuits]}|")

        Log.debug("Generating circuit-backend translation table.")
        translations = CircuitBackendTranslations.generate(
            circuits,
            self.backends,
        )
        Log.debug("Translation table generated. Placing circuits in bins.")
        for index, circuit in enumerate(circuits):
            Log.debug(f"![NEXT CIRCUIT] Placing circuit |{index}|.")
            self.place_circuit(circuit, translations)
            Log.debug(
                lambda: (
                    f"![CIRCUIT DONE] Circuit |{index}| placed. Currently have "
                    f"${sum(len(bins) for bins in self.bins.values())} bin$ with "
                    f"|{[[bin.size for bin in bins] for bins in self.bins.values()]}| circuits "
                    f"each."
                ),
            )

    def realize(self) -> dict[Backend, Sequence[qiskit.QuantumCircuit]]:
        """
        Realizes or constructs the host circuits from bins that this manager tracks.
        """

        Log.debug(
            lambda: (
                f"Realizing ${sum(len(bins) for bins in self.bins.values())} bin$ with "
                f"|{[[bin.size for bin in bins] for bins in self.bins.values()]}| circuits each."
            ),
        )

        backend_circuits = {}
        for backend, bins in self.bins.items():
            for bin in bins:
                if bin.size > 0:
                    backend_circuits.setdefault(backend, []).append(bin.realize())

        Log.debug(
            lambda: (
                f"Realized |{[len(c) for c in backend_circuits.values()]}| circuits for "
                f"${len(backend_circuits.keys())} backend$."
            ),
        )

        return backend_circuits
