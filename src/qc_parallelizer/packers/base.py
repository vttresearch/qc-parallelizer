import warnings
from typing import Any

from ..base import Exceptions
from ..backends import BackendCircuitBin
from ..interfaces import Circuit
from ..util import IndexedLayout


class PackerBase:
    """
    Base class for packers that all packers should inherit. Any packer implementation must provide
    at least `find_layout()`, and may provide `evaluate()` and `blocked()`.
    """

    # The min distance control parameters follow this logic:
    # - "intranational": inside or within a nation
    #   --> "intracircuit": between qubits of one circuit
    # - "international": among or between nations
    #   --> "intercircuit": between qubits of different circuits

    min_intra_distance: int
    min_inter_distance: int
    max_candidates: int | None

    def __init__(
        self,
        min_intra_distance: int = 0,
        min_inter_distance: int = 0,
        max_candidates: int | None = 1,
    ):
        """
        Args:
            min_intra_distance:
                Sets the minimum distance between physical placements of two qubits **from the same
                circuit** that do not share any gates. If 0, circuits can be packed as densely as
                possible. If 1, qubits can be placed next to each other only if they share gates.
                Other values are currently not supported.
            min_inter_distance:
                Sets the minimum distance between physical placements of two qubits **from different
                circuits**. Setting this to 0 achieves the densest packing, but may introduce
                unwanted crosstalk. Setting this to 1 forces "padding" qubits to be left between
                circuits. Greater values are also accepted, but possibly with diminishing returns.
            max_candidates:
                Controls how many backend bin candidates are considered before picking the best
                option. Only one, the heuristically best candidate, is considered by default.
        """
        if min_intra_distance not in [0, 1]:
            raise Exceptions.ParameterError(
                f"min. intra-circuit distance must be either 0 or 1 (got {min_intra_distance})",
            )
        if min_inter_distance < min_intra_distance:
            warnings.warn(
                (
                    "Setting the min. inter-circuit distance lower than the min. intra-circuit "
                    "distance may lead to undesired behavior. If you understand why, you can "
                    "ignore this warning."
                ),
            )
        self.min_intra_distance = min_intra_distance
        self.min_inter_distance = min_inter_distance
        self.max_candidates = max_candidates

    def blocked(
        self,
        bin: BackendCircuitBin,
    ) -> set[int]:
        """
        Determines the set of blocked backend qubits. When placing subsequent circuits, no virtual
        qubits can be placed on the blocked set.
        """
        blocked = bin.taken_indices
        for _ in range(self.min_inter_distance):
            # On each iteration, expand `blocked` with each blocked qubits' neighbors.
            current, blocked = blocked, blocked.copy()
            for taken in current:
                blocked |= bin.backend.neighbor_sets[taken]
        return blocked

    def evaluate(
        self,
        bin: BackendCircuitBin,
        circuit: Circuit,
    ) -> Any:
        """
        Evaluates how "good" the packing is in the given bin after placing the given circuit. The
        return value should be comparable with other values of the same type (so, `int`, `float`,
        and tuples with the two types work best). Values that compare as greater represent better
        packings.

        The default implementation counts the number of couplers that the given circuit would
        "consume", including those that are at the edge of the circuit.
        """
        taken_qubits = bin.taken_indices
        circuit_qubits = circuit.layout.pindices
        blocked_couplers = {
            (a, b) for a, b in bin.backend.edges if a in taken_qubits or b in taken_qubits
        }
        circuit_couplers = {
            (a, b) for a, b in bin.backend.edges if a in circuit_qubits or b in circuit_qubits
        }
        return -len(circuit_couplers - blocked_couplers)

    def find_layout(
        self,
        bin: BackendCircuitBin,
        circuit: Circuit,
        blocked: set[int],
        /,
    ) -> IndexedLayout | None:
        """
        Finds a layout for the given circuit in a backend with an optional layout and blocked
        qubits. This cannot modify the circuit, but must return a valid layout for it as it is, or
        None if no layout can be found.
        """
        raise NotImplementedError()
