import warnings
from collections.abc import Sequence

from pennylane import numpy as np
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.measurements import CountsMP, ExpectationMP, VarianceMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane_qiskit.converter import circuit_to_qiskit, mp_to_pauli
from pennylane_qiskit.remote import RemoteDevice
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

from ..util import Log
from ..util.translation import translate_for_backend

"""
Copied and modified from [1]. In essence, the `_execute_*` methods were split into two, a "prepare"
and a "finish" step, and the `execute` method was modified to first call the former for all circuits
before calling the latter. This allows jobs to be submitted in parallel.

[1]: https://github.com/PennyLaneAI/pennylane-qiskit/blob/master/pennylane_qiskit/qiskit_device.py
"""


class ParallelizableRemoteDevice(RemoteDevice):
    def execute(
        self,
        circuits: QuantumTape | Sequence[QuantumTape],
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]
        if len(circuits) == 0:
            return []

        for circuit in circuits:
            for meas in circuit.measurements:
                if isinstance(meas, CountsMP):
                    message = (
                        "A circuit with a CountsMP measurement was detected. This generates "
                        "circuits with unnecessarily many measurements (in observed cases, on all "
                        "qubits). If this causes issues, consider using other measurements instead."
                    )
                    warnings.warn(message)
                    Log.warn(message)

        session, jobs = self._session, []

        for circ in circuits:
            if circ.shots and len(circ.shots.shot_vector) > 1:
                raise ValueError(
                    f"Setting shot vector {circ.shots.shot_vector} is not supported for "
                    f"{self.name}. Please use a single integer instead when specifying the number "
                    "of shots.",
                )
            if isinstance(circ.measurements[0], (ExpectationMP, VarianceMP)) and getattr(
                circ.measurements[0].obs,
                "pauli_rep",
                None,
            ):
                prepare_fn, finish_fn = self._prepare_estimator, self._finish_estimator
            else:
                prepare_fn, finish_fn = self._prepare_sampler, self._finish_sampler
            jobs.append((circ, session, *prepare_fn(circ, session), finish_fn))

        return [finish_fn(*job) for *job, finish_fn in jobs]

    def _execute_sampler(self, *args):
        raise NotImplementedError

    def _execute_estimator(self, *args):
        raise NotImplementedError

    def _prepare_sampler(self, circuit, session):
        Log.debug(
            f"`circuit.num_wires` = |{circuit.num_wires}|, `self.num_wires` = |{self.num_wires}|",
        )
        qcirc = [circuit_to_qiskit(circuit, circuit.num_wires, diagonalize=True, measure=True)]
        sampler = Sampler(mode=session) if session else Sampler(mode=self.backend)
        compiled_circuits = self.compile_circuits(qcirc)
        sampler.options.update(**self._kwargs)  # type: ignore
        Log.debug("Invoking Sampler!")
        return sampler.run(
            compiled_circuits,
            shots=circuit.shots.total_shots if circuit.shots.total_shots else None,
        ), compiled_circuits

    def _finish_sampler(self, circuit, session, job, compiled_circuits):
        Log.debug(f"Result: {job.result()}")
        result = job.result()[0]
        classical_register_name = compiled_circuits[0].cregs[0].name
        Log.debug(f"Classical register: |'{classical_register_name}'|")
        self._current_job = getattr(result.data, classical_register_name)
        self._samples = self.generate_samples(0)
        Log.debug(f"Samples: {self._samples}")
        res = [
            mp.process_samples(self._samples, wire_order=self.wires) for mp in circuit.measurements
        ]
        single_measurement = len(circuit.measurements) == 1
        res = (res[0],) if single_measurement else tuple(res)
        return res

    def _prepare_estimator(self, circuit, session):
        qcirc = [circuit_to_qiskit(circuit, self.num_wires, diagonalize=False, measure=False)]
        estimator = Estimator(mode=session) if session else Estimator(mode=self.backend)
        compiled_circuits = self.compile_circuits(qcirc)
        pauli_observables = [
            mp_to_pauli(mp, min(self.num_wires, compiled_circuits[0].num_qubits))
            for mp in circuit.measurements
        ]
        compiled_observables = [
            op.apply_layout(compiled_circuits[0].layout) for op in pauli_observables
        ]
        estimator.options.update(**self._kwargs)  # type: ignore
        circ_and_obs = [(compiled_circuits[0], compiled_observables)]
        if Log.level.value >= Log.LogLevel.DBUG.value:
            with Log.lock:
                Log.debug("Invoking Estimator!")
                Log.debug("Compiled observables:")
                for obs in compiled_observables:
                    Log.debug(f"{obs}")
        return (
            estimator.run(
                circ_and_obs,
                precision=np.sqrt(  # type: ignore
                    1 / circuit.shots.total_shots,
                )
                if circuit.shots
                else None,
            ),
        )

    def _finish_estimator(self, circuit, session, job):
        result = job.result()
        self._current_job = result
        result = self._process_estimator_job(circuit.measurements, result)
        return result

    def compile_circuits(self, circuits):
        compiled_circuits = []
        transpile_args = self._transpile_args

        for i, circuit in enumerate(circuits):
            compiled_circ = translate_for_backend(
                circuit,
                self.compile_backend,
                **transpile_args,
            )
            assert compiled_circ is not None, "could not translate circuit"
            compiled_circ = compiled_circ.unwrap()

            compiled_circ.name = f"circ{i}"
            compiled_circuits.append(compiled_circ)

        return compiled_circuits
