import itertools
import operator
from collections.abc import Sequence

import qiskit
import qiskit.providers
from qc_parallelizer.base import Types
from qc_parallelizer.util import Log

from .postprocessing import split_results


class ParallelJob:
    """
    A wrapper class for parallel execution jobs. This is intentionally not a subclass of the actual
    `Job` class since it merely tracks the underlying `Job` instances - however, in addition to the
    extended functionality for parallelization, it also provides a proxy interface into the
    underlying `Job` object that has the same attributes and methods. Note that the returned values
    are array-like objects (you may index or iterate them), since there can be multiple jobs
    involved - for instance, to get the first job ID, you can use `parallel_job.job_id()[0]`.

    Users should not construct instances of this class themselves since the post-processor expects
    certain fixed structure in the circuit metadata. Instead, instances of this class are returned
    by the `execute()` function.
    """

    _jobs: Sequence[tuple[qiskit.QuantumCircuit, qiskit.providers.JobV1]]
    _raw_results: Sequence[Types.Result]
    _results: Sequence[Types.Result]

    def __init__(
        self,
        jobs: Sequence[tuple[qiskit.QuantumCircuit, qiskit.providers.JobV1]],
    ):
        self._jobs = jobs
        self._raw_results = []
        self._results = []

    def _fetch_results(self):
        if len(self._raw_results) == 0:
            self._raw_results = [job.result() for *_, job in self._jobs]
        if len(self._results) == 0:
            Log.debug(f"Got |{len(self._raw_results)}| results.")
            results = itertools.chain(
                *(split_results(res) for res in self._raw_results),
            )
            self._results = [result for _, result in sorted(results, key=operator.itemgetter(0))]

    def results(self) -> Sequence[Types.Result]:
        """
        Returns results for each circuit in the parallel execution. After calling this for the first
        time, the results are cached, so subsequent calls are very lightweight.
        """

        self._fetch_results()
        return self._results

    class ListAttributeProxy:
        """
        A small wrapper around a list of attribute values for a list of objects. This facilitates
        accessing properties of the underlying objects in case both properties and methods need to
        be accessed.

        For example, let `foo` be an instance of this class that wraps objects and a property of
        them that might be a regular attribute *or* a method (known to the user, but not the
        program). Interface-wise, you can treat `foo` as if it were a regular array of attribute
        values, *or* call `foo()` to call the associated method on each instance and receive a
        list of return values.

        This class exists because, for example, Qiskit's `Job` objects have certain attributes that
        are actually implemented as methods, like `job.job_id()`. In the `ParallelJob` class, there
        are several underlying job objects, so an array is returned instead - but returning an
        actual array would break things, since the caller expects some attributes (like `.job_id`)
        to the callable.
        """

        def __init__(self, jobs, attr):
            self.jobs = jobs
            self.attr = attr

        def __call__(self, *args, **kwargs):
            return [getattr(job, self.attr)(*args, **kwargs) for job in self.jobs]

        def __getitem__(self, index):
            return [getattr(job, self.attr) for job in self.jobs][index]

        def __iter__(self):
            for job in self.jobs:
                yield getattr(job, self.attr)

        def __len__(self):
            return len(self.jobs)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise KeyError(f"attribute {attr} may not be accessed")
        return ParallelJob.ListAttributeProxy([job for *_, job in self._jobs], attr)
