from typing_extensions import assert_never
import typing
from typing import Sequence, TYPE_CHECKING
from numbers import Real

from qiskit.result import Result as QiskitResult
from qiskit.providers import (
    BackendV2 as QiskitBackend,
    JobV1 as QiskitJob,
    JobStatus as QiskitJobStatus,
    Options as QiskitBackendOptions,
)
from qiskit.result.models import (
    ExperimentResult as QiskitExperimentResult,
    ExperimentResultData as QiskitExperimentResultData,
)
from qiskit.circuit import Instruction as QiskitInstruction, QuantumCircuit as QiskitCircuit
from qiskit.transpiler import (
    InstructionProperties as QiskitInstructionProperties,
    Target as QiskitBackendTarget,
)

from ..base import Types, InputTypes
from . import Circuit, Backend
from ..util.typing import ensure_sequence, isnestedinstance

if TYPE_CHECKING:
    from qc_parallelizer.parallelizer import ParallelizedBackend
    from qc_parallelizer.jobs.job import ParallelizerJobBatch

# TODO: should the assert_never calls below be raise TypeError instead?

def convert_to_backend_list(backends: InputTypes.Backends) -> list[Backend]:
    """
    Converts a backend or list of backends into a list of internal Backend objects.
    """

    backends = ensure_sequence(backends, InputTypes.Backend)

    def normalize_backend(backend):
        if isinstance(backend, Backend):
            return backend
        if isinstance(backend, QiskitBackend):
            return Backend(backend, 1)
        if isnestedinstance(backend, tuple[QiskitBackend, Real]):
            return Backend(*backend)
        assert_never(backend)

    return [normalize_backend(backend) for backend in backends]

def convert_to_circuit_list(circuits: InputTypes.Circuits) -> list[Circuit]:
    """
    Converts a circuit or list of circuit into a list of internal Circuit objects.
    """

    circuits = ensure_sequence(circuits, InputTypes.Circuit)

    def normalize_circuit(circuit):
        if isinstance(circuit, Circuit):
            return circuit
        if isinstance(circuit, QiskitCircuit):
            return Circuit(circuit, clone=True)
        if isnestedinstance(circuit, tuple[QiskitCircuit, Types.Layout]):
            return Circuit(*circuit, clone=True)
        assert_never(circuit)

    return [normalize_circuit(circuit) for circuit in circuits]

class ParallelizedQiskitJobAdapter(QiskitJob):
    def __init__(
        self,
        backend: "ParallelizedQiskitBackendAdapter",
        job_batch: "ParallelizerJobBatch",
    ):
        super().__init__(backend, job_batch.id.hex)
        self._backend = backend
        self.par_job = job_batch

    def submit(self):
        # Parallelized jobs are automatically submitted when appropriate, so this method does
        # nothing. If this triggered a forced/asap execution, in the worst case, no parallelization
        # would take place as all jobs would run immediately.
        pass

    def result(self):
        exp_results = [
            QiskitExperimentResult(
                None,
                True,
                QiskitExperimentResultData(counts=result.counts),
            )
            for result in self.par_job.result(block=True)
        ]
        return QiskitResult(
            self._backend.name,
            self._backend.version,
            None,
            self.par_job.id,
            True,
            exp_results,
        )

    def cancel(self):
        # TODO? Cancelling jobs that have not been submitted yet is easy, otherwise not sure.
        raise NotImplementedError

    def status(self):
        if self.par_job.is_ready:
            return QiskitJobStatus.DONE
        if any(job.is_submitted for job in self.par_job.jobs):
            return QiskitJobStatus.RUNNING
        return QiskitJobStatus.QUEUED

class ParallelizedQiskitBackendAdapter(QiskitBackend):
    """
    An adapter/wrapper for parallelization instances that pretend to be real Qiskit backends.
    """

    def __init__(self, backend: "ParallelizedBackend"):
        super().__init__(name=self.__class__.__name__)
        self.backend = backend

    def run(self, run_input, **options): # type: ignore
        return ParallelizedQiskitJobAdapter(
            self,
            self.backend.run(run_input, **options),
        )

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls): # type: ignore
        return QiskitBackendOptions() # empty, but required for compatibility

    @property
    def target(self): # type: ignore
        return build_merged_target(self.backend.remote_backends)

def build_merged_target(backends: Sequence[Backend]):
    """
    Builds a merged targed from several backends. The merged target is a union of all provided
    backends. Since qubits from different backends cannot couple, the resulting coupling map is not
    connected.

    Instructions and their properties are preserved. Only qubit indices ("qargs") are modified.
    """

    instructions: dict[str, tuple[
        QiskitInstruction,
        dict[tuple[int, ...], QiskitInstructionProperties]
    ]] = {}

    qubit_cumulative_offset = 0
    for backend in backends:
        # Qiskit's abstract classes (BackendV2) don't play well with static type checking, so we
        # have to hold the type checker's hand here.
        assert backend.target is not None
        for instruction, qargs in typing.cast(
            Sequence[tuple[QiskitInstruction, tuple[int, ...]]],
            backend.target.instructions
        ):
            # The Instruction class is not hashable, so we use its repr instead.
            key = repr(instruction)
            if key not in instructions:
                instructions[key] = (instruction, {})
            _, props = instructions[key]
            shifted_qargs = tuple(q + qubit_cumulative_offset for q in qargs)
            props[shifted_qargs] = backend.target[instruction.name][qargs]
        qubit_cumulative_offset += backend.num_qubits

    target = QiskitBackendTarget()
    for instruction, qargs in instructions.values():
        target.add_instruction(instruction, qargs)

    return target
