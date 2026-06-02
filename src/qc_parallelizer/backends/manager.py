import collections
import typing
from collections.abc import Sequence
from typing import Any

from qiskit.providers import JobV1 as QiskitJob
from qiskit.result import Result as QiskitJobResult

from ..interfaces import Backend, Circuit
from ..jobs import ParallelizerJob
from ..util import Log
from . import BackendCircuitBin


class ManagedBackend:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.num_runs = 0
        self.bins: list[BackendCircuitBin] = []

    @property
    def num_nonempty_bins(self):
        return sum(1 for bin in self.bins if bin.size > 0)


class BackendManager:
    def __init__(self):
        self.remote_backends: dict[Backend, ManagedBackend] = {}

    def register(self, backends: Sequence[Backend]):
        as_set = set(backends)
        new = as_set - set(self.remote_backends)
        Log.debug(f"Processing ${len(new)} new backend$.")
        for backend in new:
            Log.debug(f"Registered backend |'{backend.name}'| with ${backend.num_qubits} qubit$.")
            self.remote_backends[backend] = ManagedBackend(backend)

    def _ensure_empty_available(self):
        for backend in self.remote_backends:
            if any(bin.size == 0 for bin in self[backend].bins):
                continue
            bin = BackendCircuitBin(backend)
            self[backend].bins.append(bin)
            Log.debug(f"Added new empty bin for backend |'{backend.name}'|.")

    @property
    def bins(self):
        self._ensure_empty_available()
        return [bin for backend in self.remote_backends.values() for bin in backend.bins]

    def best_bins(
        self,
        backend_translations: dict[Backend, Circuit],
        kwarg_constraints: dict[str, Any],
    ):
        compatible = [
            bin
            for bin in self.bins
            if bin.backend in backend_translations
            if bin.compatible(backend_translations[bin.backend], kwarg_constraints)
        ]
        return sorted(
            compatible,
            key=lambda bin: (
                # this forces empty bins to be considered last
                bin.size == 0,
                # ...and then the actual sorting criteria
                (self[bin.backend].num_runs + self[bin.backend].num_nonempty_bins)
                * bin.backend.cost,
                bin.frac_taken,
            ),
        )

    def tick(self, auto_exec: bool = True):
        """
        Ticks, or updates, this manager. This means collecting ready bins, removing them from
        tracking, converting the contained circuits into host circuits, submitting the host
        circuits, and launching threads to collect results after jobs complete.
        """

        ready_bins = (
            bin
            for bin in self.bins
            if bin.size > 0
            if any(job.completion_requested for job in bin) or (bin.is_full and auto_exec)
        )

        for bin in ready_bins:
            self[bin.backend].bins.remove(bin)
            host_circuit = bin.to_circuit()

            Log.info(f"Submitting job to backend |'{bin.backend.name}'|...")
            Log.debug(lambda: f"Job kwargs: {bin.kwargs}")
            Log.debug(lambda: f"Circuit:\n{host_circuit.draw(idle_wires=False)}")

            remote_job = bin.backend.run(
                host_circuit,
                **bin.kwargs,
                callback=lambda job, result, bin=bin: self._remote_job_completed(
                    job,
                    result,
                    bin,
                ),
            )
            Log.info(f"![JOB SUBMITTED] Job submitted with id |'{remote_job.job_id()}'|.")

            for job in bin:
                job.mark_submitted(bin.backend, remote_job)
            self[bin.backend].num_runs += 1

    def _remote_job_completed(
        self,
        remote_job: QiskitJob,
        result: QiskitJobResult,
        bin: BackendCircuitBin,
    ):
        Log.info(f"![JOB COMPLETE] Job |'{remote_job.job_id()}'| completed!")
        if sum(len(cregs) for cregs in bin.cregs.values()) == 0:
            Log.info("Job has no cregs and thus no results to retrieve.")
            for job in bin:
                job.mark_completed({})
            return

        counts = typing.cast(dict[str, int], result.get_counts())

        sample_key = next(iter(counts.keys()))
        # Currently, only bitstring keys are supposed. Hexadecimal is also possible to encounter
        # but TODO.
        if not isinstance(sample_key, str) or not all(char in "01 " for char in sample_key):
            raise ValueError(f"counts with keys of form '{sample_key}' can not be parsed")

        """
        The counts keys are strings of bits separated by spaces to delimit the classical registers
        in the circuit. For example, given a batch with two jobs, one containing three- and one-bit
        classical registers, and the other containing two- and one-bit classical registers, a
        possible (reverse*) key would be "011 0 10 1".

        Thus we need to determine, for each encountered key, which bits belong to which (job's)
        classical registers. Luckily this mapping is fixed for all keys, so we can precompute a
        table of index bounds (represented in practice by `slice` objects). In English, each entry
        in this table represents the portion of a counts key that belongs to the corresponding job.
        In the example above, the table would look like this:

           JOB  | INDICES
          ------|---------
          job 1 | [0, 5)   --> "011 0"
          job 2 | [6, 10)  --> "10 1"

        Then, for each counts key / bitstring, we iterate this table, slice the bitstring to get
        just one job's bits, and update a corresponding isolated counts object.

        *) Qiskit follows a bit ordering convention that results in bits being read in reverse. In
           some contexts this probably makes sense, but for practical purposes, we need to handle
           these strings in reverse.
        """

        clbit_bounds: dict[ParallelizerJob, slice] = {}
        end = -1
        for job, cregs in bin.cregs.items():
            num_regs, num_bits = len(cregs), sum(creg.size for creg in cregs)

            # `end` points to the char before this job's first bit, so...
            # - `start` is computed as `end + 1`, which is the first char, and
            # - `end` is moved to `start` plus the number of bits (= chars) plus the number of
            #   spaces in between them, which is one less than the number of registers in this job
            start, end = end + 1, end + 1 + num_bits + num_regs - 1

            # To avoid having to reverse, slice, and unreverse, we can "pre-reverse" the slice.
            # Note the special case handling for `start == 0`, where the end becomes None to index
            # everything until the last bit. If left as zero, this would produce empty slices since
            # `slice(-end, 0) == slice(0)` for `end >= 0`.
            clbit_bounds[job] = slice(-end, -start if start > 0 else None)

        split_counts = {job: collections.Counter() for job in bin}
        for bitstring, count in counts.items():
            for job, bounds in clbit_bounds.items():
                split_counts[job][bitstring[bounds]] += count

        for job in bin:
            job.mark_completed(dict(split_counts[job]))

    def __getitem__(self, backend: Backend):
        return self.remote_backends[backend]
