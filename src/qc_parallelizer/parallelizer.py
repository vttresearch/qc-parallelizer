import threading
from collections import deque
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
from .util import Log


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

    max_history_len: int = 1000

    def __init__(
        self,
        parent: "Parallelizer",
        backends: Sequence[Backend],
        auto_exec: bool,
        sync_wait: float,
    ):
        self.parent = parent
        self.remote_backends = backends
        self.auto_exec = auto_exec
        self.sync_wait = sync_wait
        self.history: deque[ParallelizerJobBatch] = deque()
        self.lock = threading.RLock()

    def run(self, circuit_inputs: InputTypes.Circuits, **kwargs):
        circuits = convert_to_circuit_list(circuit_inputs)
        Log.info(f"Received ${len(circuits)} circuit$ for execution.")
        batch = ParallelizerJobBatch(
            [ParallelizerJob(self, circuit) for circuit in circuits],
            kwargs,
            sync_wait=self.sync_wait,
        )
        batch.place_all()
        self.manager.tick(self.auto_exec)
        with self.lock:
            self.history.append(batch)
            while len(self.history) > self.max_history_len:
                self.history.popleft()
        return batch

    @property
    def num_qubits(self):
        """The total number of qubits across all available backends."""
        return sum(backend.num_qubits for backend in self.remote_backends)

    @property
    def max_backend_qubits(self):
        """The largest number of qubits in any single available backend."""
        return max(backend.num_qubits for backend in self.remote_backends)

    @property
    def num_backends(self):
        return len(self.remote_backends)

    @property
    def backend_utilization(self):
        """
        A dictionary of backend utilization, measured as the number of circuits submitted.
        """

        return {
            backend.unwrap(): self.manager[backend].num_runs for backend in self.remote_backends
        }

    @property
    def packer(self):
        return self.parent.packer

    @property
    def manager(self):
        return self.parent.manager

    @property
    def as_qiskit_backend(self):
        """
        An emulated native Qiskit backend that parallelizes automatically.
        """

        return ParallelizedQiskitBackendAdapter(self)

    @property
    def as_pennylane_device(self):
        """
        An emulated PennyLane device that parallelizes automatically. In short, this is a slightly
        modified `"qiskit.remote"` device with the `.as_qiskit_backend` adapter as the remote
        device.
        """

        try:
            from .interfaces.pennylane import ParallelizableRemoteDevice
        except ImportError as err:
            raise RuntimeError("missing dependencies for PennyLane") from err

        return ParallelizableRemoteDevice(
            wires=self.max_backend_qubits,
            backend=self.as_qiskit_backend,
            optimization_level=self.parent.translation_kwargs.get("optimization_level", 1),
        )

    @property
    def timeline(self):
        """
        A dict of ordered sequences of jobs on each available backend. Together, these form the
        "timeline" of executed jobs or job batches.

        The return value might look like this:
        ```
        {
            <backend 1>: [(timing, job), (timing, job), ...],
            <backend 2>: [(timing, job), ...],
            None: [(partial timing, job), ...]
        }
        ```
        where `timing` is a 5-tuple of creation, placement, submission, completion and completion
        request timestamps. Each key of the dictionary corresponds to a backend, with the exception
        of `None`, which represents jobs that have not been assigned a backend yet.

        This can be visualized with `qc_parallelizer.util.visualization.plot_timeline`.
        """

        events: dict[
            Backend | None,
            list[
                tuple[
                    tuple[float, float | None, float | None, float | None, float | None],
                    ParallelizerJob,
                ]
            ],
        ] = {backend: [] for backend in self.remote_backends} | {None: []}
        for batch in self.history:
            for job in batch:
                point = job.timing.as_tuple, job
                if job.remote_backend is not None:
                    events[job.remote_backend].append(point)
                else:
                    events[None].append(point)
        for k in events:
            events[k] = sorted(events[k], key=lambda ev: ev[0])

        return events

    def cancel_all(self):
        """
        Cancels all submitted jobs.
        """

        for batch in self.history:
            batch.cancel()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class Parallelizer:
    def __init__(
        self,
        packer: packers.PackerBase = packers.Defaults.Fast(),
        translation_kwargs: dict = {},
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
            translation_kwargs:
                A dictionary of keyword arguments to be passed to the circuit translation stage that
                translates circuits (pre-layout) for each backend. Most notably, this allows setting
                `optimization_level`.
        """

        self.packer = packer
        self.manager = BackendManager()
        self.translation_kwargs = translation_kwargs

    def across(
        self,
        backend_inputs: InputTypes.Backends,
        auto_exec: bool = True,
        sync_wait: float = 0.2,
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
                If True (default), jobs are automatically submitted when the associated host circuit
                uses all qubits on the backend. If False, no submission happens unless job results
                are explicitly requested.

                Generally, when enabled, this should have a tiny positive impact on performance. It
                can be useful to disable this for testing purposes or other situations where one
                wishes to place circuits but not submit any jobs.

            sync_wait:
                A delay (in seconds) to wait before requesting job results from the underlying
                backend(s). In multi-threaded applications, this can be very beneficial if several
                threads submit jobs and request results in parallel. Set to zero to disable. When
                retrieving results of individual jobs, an overridden value can also be passed as a
                keyword argument of the same name.
        """

        backends = convert_to_backend_list(backend_inputs)
        self.manager.register(backends)
        return ParallelizedBackend(
            self,
            backends,
            auto_exec,
            sync_wait,
        )
