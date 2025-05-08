# Quantum Circuit Parallelizer

A Python module for optimally combining and distributing quantum circuits. See the
[included notebooks](./notebooks/) for examples and more information. For an operational overview,
see the diagram below.

![](./notebooks/parallelizer-full.drawio.png)

## Setup

For the following commands, a virtual environment or equivalent isolation is recommended.

The package can be installed from a local copy of the directory by running

```bash
pip install .
```

from the repository root. However, if you wish to run tests or the provided notebook(s), you must
install additional dependencies with

```bash
pip install .[tests]
# and/or
pip install .[notebooks]
```

## Testing

Running all tests is as simple as running

```bash
pytest
```

from the repository root. Additionally, there is a benchmarking script in the `tests/` directory.

## Authors

- **Henri Ahola** - henri.ahola@vtt.fi
