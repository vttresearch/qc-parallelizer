import heapq
import itertools
import re
import subprocess
import time
import warnings
from typing import Any

import rustworkx
import z3

from .base import Exceptions
from .bins.circuitbin import CircuitBin
from .extensions import Circuit
from .util import IndexedLayout, Log


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
    max_bins_per_backend: int | None
    max_candidates: int | None

    def __init__(
        self,
        min_intra_distance: int = 0,
        min_inter_distance: int = 0,
        max_bins_per_backend: int | None = None,
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
            max_bins_per_backend:
                Controls how many bins (= host circuits) can be created per backend. This is
                proportional to the total execution duration. Set to None for no limit.
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
        self.max_bins_per_backend = max_bins_per_backend
        self.max_candidates = max_candidates

    def blocked(
        self,
        bin: CircuitBin,
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
        bin: CircuitBin,
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
        bin: CircuitBin,
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


class SMTPackers:
    """
    A collection of packers that use the Z3 SMT solver/optimizer to determine layouts.
    """

    class _Base(PackerBase):
        timeout: int | None = None
        max_attempts: int
        seed: int

        def __init__(
            self,
            timeout: int | None = 2000,
            max_attempts: int = 4,
            seed: int = 0,
            **kwargs,
        ):
            """
            Args:
                timeout:
                    Solver timeout in milliseconds, per attempt. The total timeout is roughly
                    `timeout * max_attempts`. Set to None for no timeout.
                max_attempts:
                    The maximum number of times to retry if the solver returns an indecisive result.
                    This is the case if the solver is interrupted by the timeout.
                seed:
                    A random seed. Set to a constant by default, but a random value may be provided
                    for random results. In some cases, a good seed can reduce execution times by
                    one or two orders of magnitude.
            """
            super().__init__(**kwargs)
            self.timeout = timeout
            self.max_attempts = max_attempts
            self.seed = seed

        def optimize(
            self,
            solver: z3.Optimize,
            placements: list[list[z3.BoolRef]],
            bin: CircuitBin,
            blocked: set[int],
            /,
        ):
            raise NotImplementedError()

        @staticmethod
        def _log_find_conflicting(constraints: list[tuple[list, str]]):
            """
            Little helper that finds the smallest conflicting set of constraints. No-op if debug
            logging is not enabled. All conflicting sets of the smallest size are reported.
            """
            if not Log.enabled(Log.LogLevel.DBUG):
                return
            found_conflict_for_count = False
            for count in range(1, len(constraints) + 1):
                for csets in itertools.combinations(constraints, count):
                    temp_solver = z3.Solver()
                    for cset, _ in csets:
                        temp_solver.add(cset)
                    result = temp_solver.check()
                    if result != z3.sat:
                        found_conflict_for_count = True
                        names = [kind for _, kind in csets]
                        Log.debug(f"Found conflict between constraint lists |{names}|.")
                if found_conflict_for_count:
                    return
            Log.debug("No constraint conflicts found.")

        def find_layout(self, *args):
            Log.debug(f"Attempting SMT layout up to |{self.max_attempts}| times.")
            for attempt in range(self.max_attempts):
                Log.debug(f"![ATTEMPT] Running attempt |{attempt + 1}/{self.max_attempts}|.")
                try:
                    layout = self._find_layout(*args, seed=abs(hash((self.seed, attempt))))
                except Exceptions.CircuitBackendCompatibility:
                    break
                if layout is not None:
                    Log.debug("Found solution!")
                    return layout
            Log.debug(f"No solution found after |{self.max_attempts}| attempts.")
            return None

        def _find_layout(
            self,
            bin: CircuitBin,
            circuit: Circuit,
            blocked: set[int],
            seed: int,
        ):
            if circuit.num_qubits > bin.backend.num_qubits - len(blocked):
                return None

            Log.debug("Attempting constraint-based layout.")

            solver = z3.Optimize()

            if self.timeout is not None:
                solver.set(timeout=self.timeout)
                Log.debug(f"Set solver timeout to |{self.timeout} ms|.")
            else:
                Log.debug("No timeout defined for solver.")

            placements = [
                [z3.Bool(f"{v}on{p}") for v in range(circuit.num_qubits)]
                for p in range(bin.backend.num_qubits)
            ]
            Log.debug(f"Defined |{len(placements)}x{len(placements[0])}| boolean constants.")

            solver.add(
                base_constraints := (
                    [
                        # At most one virtual qubit should be placed on this physical qubit.
                        z3.AtMost(*v_placements, 1)
                        for v_placements in placements
                    ]
                    + [
                        # Each virtual qubit should be placed on exactly one physical qubit.
                        # Sadly there is no z3.Equals, so we use a close alternative.
                        # PbEq expects variables and weights, so we wrap each boolean with a unit
                        # weight.
                        z3.PbEq([(p[v], 1) for p in placements], 1)
                        for v in range(circuit.num_qubits)
                    ]
                ),
            )

            # The given layout must be respected.
            solver.add(
                layout_constraints := [
                    placements[p][v] == True for v, p in circuit.layout.v2p.items()
                ],
            )

            # For each blocked physical qubit, no virtual qubit can be placed there.
            solver.add(blocking_constraints := [v == False for b in blocked for v in placements[b]])

            # If two qubits couple virtually, they must also couple physically.
            virt_edges = circuit.get_edges()
            coupling_constraints = [
                z3.Or(
                    z3.And(placements[pa][va] == True, placements[pb][vb] == True)
                    for pa, pb in bin.backend.edges_bidir
                )
                for va, vb in virt_edges
            ]

            if self.min_intra_distance != 0:
                # If two qubits don't couple virtually, they cannot couple physically.
                for va in range(circuit.num_qubits):
                    for vb in range(va + 1, circuit.num_qubits):
                        if (va, vb) not in virt_edges:
                            coupling_constraints.append(
                                z3.And(
                                    z3.Implies(
                                        placements[pa][va] == True,
                                        placements[pb][vb] == False,
                                    )
                                    for pa, pb in bin.backend.edges
                                ),
                            )

            solver.add(coupling_constraints)

            # Add optimization expressions, if any are available.
            self.optimize(solver, placements, bin, blocked)

            Log.debug("Generating model s-expression.")
            sexpr = solver.sexpr()

            Log.debug(
                lambda: (
                    f"Generated |{sexpr.count('assert ')} hard|, "
                    f"|{sexpr.count('assert-soft')} soft|, |{sexpr.count('pbeq')} pbeq|, "
                    f"|{sexpr.count('at-most')} at-most|, |{sexpr.count('at-least')} "
                    f"at-least|, |{sexpr.count('minimize')} minimize|, and "
                    f"|{sexpr.count('maximize')} maximize| expressions."
                ),
            )

            Log.debug("Checking solver model.")

            # We could call solver.check() and solver.model(), but due to Python's implementation
            # details, the random seed is effectively disregarded. See for example this comment:
            # https://github.com/Z3Prover/z3/issues/6679#issuecomment-1503123308
            # Below, we invoke Z3 in a separate process. Instead of file I/O, as recommended in the
            # comment, we send the solver s-expression directly into the child process' stdin.

            # TODO: If no random seed is provided, we could call the Python interface to avoid
            # process spawning and I/O overhead.

            try:
                z3_proc = subprocess.run(
                    [
                        "z3",
                        "-model",
                        "-in",
                        f"smt.random-seed={seed}",
                        f"sat.random-seed={seed}",
                        f"timeout={self.timeout or 0}",
                    ],
                    input=sexpr.encode("utf-8"),
                    stdout=subprocess.PIPE,
                )
            except Exception as error:
                raise RuntimeError("could not run z3 process") from error

            # The process outputs two things: the result, and the solution. The code below extracts
            # them. If there is no solution (so result is either "unsat" or "unknown"), the
            # extracted solution string (`model_str`) will be empty.

            try:
                result, model_str = z3_proc.stdout.decode("utf-8").split("\n", 1)
            except Exception as error:
                raise RuntimeError("could not parse result from z3 process output") from error

            Log.debug(f"Solver finished with result |'{result}'|.")
            if result == "sat":
                # Now we know that there is a valid solution in the string. However, it is formatted
                # as an s-expression with Bool function definitions, which we must parse. An example
                # of a single-variable solution would be the following:
                # (
                #   (define-fun |variable name| () Bool
                #    true)
                # )
                # The regex below extracts variable names and values with its match groups.

                pattern = re.compile(
                    r"\(define-fun\s+\|([^\|\\]+)\|\s+\(\)\s+Bool\s+(true|false)\)",
                )

                # All that then remains is restructuring a list of ("VonP", "true" | "false") pairs
                # into a dict with V as keys and P as values, with only the true-valued pairs.

                try:
                    model = {
                        int(name.split("on", 1)[0]): int(name.split("on", 1)[1])
                        for name, val in pattern.findall(model_str)
                        if val == "true"
                    }
                except Exception as error:
                    raise RuntimeError("could not parse model from z3 process output") from error

                return IndexedLayout(v2p=model)
            elif result == "unsat":
                Log.debug("Model is unsatisfiable! Checking for conflicting constraint sets.")

                self._log_find_conflicting(
                    [
                        (base_constraints, "base"),
                        (layout_constraints, "layout"),
                        (blocking_constraints, "blocking"),
                        (coupling_constraints, "coupling"),
                    ],
                )

                raise Exceptions.CircuitBackendCompatibility()

            return None

    class NonOptimizing(_Base):
        """
        Finds any suitable layout, even if not optimal.
        """

        def optimize(self, *args):
            pass  # No optimization.

    class Minimizing(_Base):
        """
        SMT packer that minimizes the number of used couplers directly.
        """

        def optimize(self, solver, placements, bin: CircuitBin, blocked):
            Log.debug("Using coupler usage minimization.")

            phys_qubit_used = [z3.AtLeast(*v_placements, 1) for v_placements in placements]
            coupler_used = [
                z3.Or(phys_qubit_used[a], phys_qubit_used[b]) for a, b in bin.backend.edges_bidir
            ]
            solver.minimize(z3.Sum(coupler_used))

    class SoftConstraining(_Base):
        """
        Imposes soft constraints on all qubits with penalties proportional to the qubits'
        connectivities. In other words, using a more connected qubit costs more.
        """

        def optimize(self, solver, placements, bin: CircuitBin, blocked):
            Log.debug("Using weighted soft constraints.")

            phys_qubit_used = [z3.AtLeast(*v_placements, 1) for v_placements in placements]
            for p_index, p_used in enumerate(phys_qubit_used):
                # This physical qubit can be used, but with a penalty that depends on how many
                # non-blocked qubits it couples with.
                solver.add_soft(
                    p_used == False,
                    str(len(bin.backend.neighbor_sets[p_index] - blocked)),
                )


class VF2Packers:
    """
    A collection of packers that utilize the VF2++ layout algorithm.
    """

    class _Base(PackerBase):
        id_order: bool
        call_limit: int | None

        def __init__(self, id_order: bool = False, call_limit: int | None = 50_000_000, **kwargs):
            """
            Args:
                id_order:
                    If True, qubits are considered in order by their index. If set to False, a
                    heuristic order is used instead. See the
                    [VF2++ paper](https://www.sciencedirect.com/science/article/pii/S0166218X18300829)
                    for more details.
                call_limit:
                    Sets a limit on the number of states that the VF2++ algorithm is allowed to
                    explore. None indicates no limit.
            """
            super().__init__(**kwargs)
            self.id_order = id_order
            self.call_limit = call_limit

        def layout_generator(self, bin: CircuitBin, circuit: Circuit, blocked: set[int]):
            phys = rustworkx.PyGraph(multigraph=False)
            phys.add_nodes_from(range(bin.backend.num_qubits))
            phys.add_edges_from_no_data(list(bin.backend.edges_bidir))
            virt = rustworkx.PyGraph(multigraph=False)
            virt.add_nodes_from(range(circuit.num_qubits))
            virt.add_edges_from_no_data(list(circuit.get_edges(bidir=True)))

            Log.debug(
                lambda: (
                    f"Generated ${phys.num_nodes()} node$ and ${phys.num_edges()} edge$ for "
                    f"physical graph, and ${virt.num_nodes()} node$ and ${virt.num_edges()} edge$ "
                    f"for virtual graph."
                ),
            )

            def matcher(p, v) -> bool:
                if p in blocked:
                    return False
                if v in circuit.layout.v2p:
                    return p == circuit.layout.v2p[v]
                return True

            mapping_generator = rustworkx.vf2_mapping(
                phys,
                virt,
                node_matcher=matcher,
                subgraph=True,
                induced=self.min_intra_distance != 0,
                call_limit=self.call_limit,
                id_order=self.id_order,
            )

            Log.debug(f"VF2++ generator created with `id_order` = |{self.id_order}|.")

            return (dict(mapping) for mapping in mapping_generator)

    class NonOptimizing(_Base):
        """
        Finds any valid layout. Very efficient, but results in possibly non-optimal packings.
        """

        def find_layout(self, bin: CircuitBin, circuit: Circuit, blocked: set[int]):
            if circuit.num_qubits > bin.backend.num_qubits - len(blocked):
                return None
            Log.debug("Invoking VF2++ to determine the first valid layout.")
            try:
                layout = next(self.layout_generator(bin, circuit, blocked))
                Log.debug("Layout found.")
                return IndexedLayout(p2v=layout)
            except StopIteration:
                Log.warn("No layout found.")
                return None

    class Minimizing(_Base):
        """
        Finds layouts that minimize unused couplers. Slower than the NonOptimizing version, but
        produces more optimal packings.
        """

        timeout: int | None = None

        def __init__(self, timeout: int | None = 2000, **kwargs):
            """
            Args:
                timeout:
                    Defines a maximum runtime, in milliseconds, for evaluating different solutions.
                    Set to None for no timeout.
            """
            super().__init__(**kwargs)
            self.timeout = timeout

        def find_layout(self, bin: CircuitBin, circuit: Circuit, blocked):
            if circuit.num_qubits > bin.backend.num_qubits - len(blocked):
                return None
            Log.debug("Invoking VF2++ and iterating over results to find optimal layout.")

            solution_heap: list[tuple[Any, int, IndexedLayout]] = []

            start = time.time()

            def timed_out():
                return self.timeout is not None and time.time() - start >= self.timeout / 1000

            for i, layout in enumerate(self.layout_generator(bin, circuit, blocked)):
                # Check only every 16k iterations to reduce function calls
                if i > 0 and i & 0x3FFF == 0:
                    Log.debug(
                        (
                            f"Discovered ${len(solution_heap)} options$ with leading score "
                            f"|{-solution_heap[0][0]}|."
                        ),
                    )
                    if timed_out():
                        Log.warn("Search interrupted by timeout.")
                        break
                layout = IndexedLayout(p2v=layout)
                score = self.evaluate(bin, circuit.with_layout(layout))
                heapq.heappush(solution_heap, (-score, i, layout))

            Log.debug(f"Found ${len(solution_heap)} possible placement$.")

            if len(solution_heap) > 0:
                *_, best = solution_heap[0]
                return best
            return None


class Defaults:
    Optimizing = VF2Packers.Minimizing
    Fast = VF2Packers.NonOptimizing
