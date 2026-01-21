from collections.abc import Sequence

from . import packers
from .backends import BackendManager
from .base import InputTypes
from .interfaces import (
    Backend,
    ParallelizedQiskitBackendAdapter,
    convert_to_backend_list,
    convert_to_circuit_list,
)
from .jobs import ParallelizerJob, ParallelizerJobBatch


class ParallelizedBackend:
    """
    A parallelization "backend" with a predefined set of backends. This works like a Qiskit backend
    and provides a `.run()` method for executing a list of circuits.

    Instances of this class also support the context management protocol, allowing for something
    like
    ```
    with par.across(backends) as par_backend:
        par_backend.run(...)
    ```

    Instances of this class may also be converted into a Qiskit `BackendV2` object to fully emulate
    a real backend with the `.as_qiskit_backend` property.
    """

    def __init__(self, parent: "Parallelizer", backends: Sequence[Backend], auto_exec: bool):
        self.parent = parent
        self.remote_backends = backends
        self.auto_exec = auto_exec

    def run(self, circuit_inputs: InputTypes.Circuits):
        circuits = convert_to_circuit_list(circuit_inputs)
        batch = ParallelizerJobBatch([ParallelizerJob(self, circuit) for circuit in circuits])
        batch.place_all()
        self.manager.tick(self.auto_exec)
        return batch

    @property
    def num_qubits(self):
        return sum(backend.num_qubits for backend in self.remote_backends)

    @property
    def num_backends(self):
        return len(self.remote_backends)

    @property
    def backend_utilization(self):
        """
        A dictionary of backend utilization, measured as the number of circuits submitted.
        """

        return {backend: self.manager[backend].num_runs for backend in self.remote_backends}

    @property
    def packer(self):
        return self.parent.packer

    @property
    def manager(self):
        return self.parent.manager

    @property
    def as_qiskit_backend(self):
        """
        Returns an emulated native Qiskit backend that parallelizes automatically.
        """

        return ParallelizedQiskitBackendAdapter(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class Parallelizer:
    def __init__(
        self,
        packer: packers.PackerBase = packers.Defaults.Fast(),
    ):
        """
        Constructs a parallelizer instance, which captures settings for parallelization. Backends
        are specified separately. See the argument descriptions for details.

        The optimal choice of parameters depends on the application. There is no definitive guide
        and experimentation is recommended. The defaults are sensible and should work as a good
        starting point.

        Args:
            packer:
                A circuit packer instance. This is responsible for placing circuits onto backends,
                which includes deciding which qubits are used and how closely circuits can lay to
                each other.
        """

        self.packer = packer
        self.manager = BackendManager()

    def across(
        self,
        backend_inputs: InputTypes.Backends,
        auto_exec: bool = True,
    ):
        """
        Creates a parallelization instance across a set of backends. The resulting object works like
        a Qiskit backend and supports the usual `.run()` method - only that the given circuits will
        automatically parallelize across all available backends.

        Args:
            backend_inputs:
                A Qiskit backend object or a sequence thereof. Each object may also be a tuple of a
                backend and a numeric type, where the latter represents the backend's weight/cost.
                Cost is considered when there are several suitable candidates for executing a
                circuit, with lower cost being more favourable. If not provided, a default cost of
                1 is used.

            auto_exec:
                By default, when a backend gets completely occupied, the combined circuit is
                submitted automatically. Passing False here disables this behaviour.
        """

        backends = convert_to_backend_list(backend_inputs)
        self.manager.register(backends)
        return ParallelizedBackend(self, backends, auto_exec)
