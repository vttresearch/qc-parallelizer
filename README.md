# Quantum Circuit Parallelizer

A Python module for optimally combining and distributing quantum circuits. See the
[included notebooks](./notebooks/) for examples and more information. In summary, the motivation
for this module comes from frequent underutilization of increasingly large quantum processors. The
module processes several independent circuits into a smaller set of wider, combined circuits, which
it then runs on available backends in parallel. All of this happens behind the scenes, so ideally
the user can treat the module's functionality as a parallelized drop-in replacement for Qiskit's
`backend.run()`.

Here is a brief and basic example:

```python
# Define or load a number of circuits.
from qiskit import QuantumCircuit
circuits = [QuantumCircuit(...), QuantumCircuit(...), ...]

# Define backends for circuit execution. These can be any Qiskit-compatible backend objects, but
# here we define two simulators that mimic IQM's 5-qubit Adonis architechture.
import iqm.qiskit_iqm as iqm
backends = [iqm.IQMFakeAdonis(), iqm.IQMFakeAdonis()]

# Parallelize and execute. This call will
#  1. determine how to combine the circuits and for which backends, and
#  2. submit jobs to the backends.
import qc_parallelizer as parallelizer
job = parallelizer.execute(circuits, backends=backends)

# Fetch and handle results. This plots the first circuit's result histogram, for example.
results = job.results()
qiskit.visualization.plot_histogram(result[0].get_counts())
# Information on the parallelization and underlying jobs is also available.
print(f"On average, {job.info.avg_circuits_per_backend} circuits were placed per backend.")
print("Job IDs:")
for job_id in job.job_id():
    print(f" - {job_id}")
```

For an operational overview, see this diagram:

![](./notebooks/parallelizer-full.drawio.png)

## Development setup

For the following commands, a virtual environment or equivalent isolation is recommended. This can
be done with Conda, for example, with
```bash
conda create --name parallelizer python=3.10 pip # note the Python version!
conda activate parallelizer
```

The package can then be installed from a local copy of the directory by running

```bash
pip install -e .
```

from the repository root. If you additionally wish to run tests or the provided notebook(s), you
can install dependencies for those with

```bash
pip install .[tests]
# and/or
pip install .[notebooks]
```

## Testing

Running all tests is as simple as installing the required dependencie (`.[tests]`) and running

```bash
pytest
```

from the repository root. Additionally, there is a benchmarking script in the `tests/` directory.

## Authors

- **Henri Ahola** - henri.ahola@vtt.fi
