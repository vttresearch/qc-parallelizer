"""
A collection of packers that use the Z3 SMT solver/optimizer to determine layouts.
"""

import itertools
import re
import subprocess

import z3

from ..base import Exceptions
from ..backends import BackendCircuitBin
from ..interfaces import Circuit
from ..util import IndexedLayout, Log

from .base import PackerBase

class SMTBase(PackerBase):
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
        bin: BackendCircuitBin,
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
        Log.debug(f"Attempting SMT layout up to ${self.max_attempts} time$.")
        for attempt in range(self.max_attempts):
            Log.debug(f"![SMT LAYOUT ATTEMPT] Running attempt |{attempt + 1}/{self.max_attempts}|.")
            try:
                layout = self._find_layout(*args, seed=abs(hash((self.seed, attempt))))
            except Exceptions.CircuitBackendCompatibility:
                break
            if layout is not None:
                Log.debug("Found solution!")
                return layout
        Log.debug(f"No solution found after ${self.max_attempts} attempt$.")
        return None

    def _find_layout(
        self,
        bin: BackendCircuitBin,
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
        # TODO: For cases where a process must be spawned, the process could be pre-loaded, waiting
        # for standard input. Please benchmark and see if process spawn overhead is significant
        # before doing this, though.

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

            def parse_name(name: str):
                a, b = name.split("on", 1)
                return int(a), int(b)

            try:
                model = dict(
                    parse_name(name)
                    for name, val in pattern.findall(model_str)
                    if val == "true"
                )
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

class NonOptimizing(SMTBase):
    """
    Finds any suitable layout, even if not optimal.
    """

    def optimize(self, *args):
        pass  # No optimization.

class Minimizing(SMTBase):
    """
    SMT packer that minimizes the number of used couplers directly.
    """

    def optimize(self, solver, placements, bin: BackendCircuitBin, blocked):
        Log.debug("Using coupler usage minimization.")

        phys_qubit_used = [z3.AtLeast(*v_placements, 1) for v_placements in placements]
        coupler_used = [
            z3.Or(phys_qubit_used[a], phys_qubit_used[b]) for a, b in bin.backend.edges_bidir
        ]
        solver.minimize(z3.Sum(coupler_used))

class SoftConstraining(SMTBase):
    """
    Imposes soft constraints on all qubits with penalties proportional to the qubits'
    connectivities. In other words, using a more connected qubit costs more.
    """

    def optimize(self, solver, placements, bin: BackendCircuitBin, blocked):
        Log.debug("Using weighted soft constraints.")

        phys_qubit_used = [z3.AtLeast(*v_placements, 1) for v_placements in placements]
        for p_index, p_used in enumerate(phys_qubit_used):
            # This physical qubit can be used, but with a penalty that depends on how many
            # non-blocked qubits it couples with.
            solver.add_soft(
                p_used == False,
                str(len(bin.backend.neighbor_sets[p_index] - blocked)),
            )

__all__ = (
    "NonOptimizing",
    "Minimizing",
    "SoftConstraining",
)
