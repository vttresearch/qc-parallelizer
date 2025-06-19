"""
Base fixtures and other general definitions that are used in multiple tests.
"""

import itertools
from inspect import signature

import qiskit
from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from qc_parallelizer.extensions import Backend

fake_5qb_backend = Backend(IQMFakeAdonis())
fake_20qb_backend = Backend(IQMFakeApollo())
fake_54qb_backend = Backend(IQMFakeAphrodite())


def fake_54qb_backend_cluster(n: int):
    return [IQMFakeAphrodite() for _ in range(n)]


def build_ghz_circuit(n: int):
    """Builds an n-qubit GHZ circuit. For n = 2, this is the Bell state circuit."""

    circuit = qiskit.QuantumCircuit(n)
    circuit.h(0)
    if n > 1:
        circuit.cx(0, list(range(1, n)))
    return circuit


def build_partially_used_circuit(num_qubits: int, num_used: int):
    """Builds a circuit with a gate on a subset of the qubits, leaving the rest unused."""

    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.h(list(range(num_used)))
    return circuit


def build_grid_circuit(width: int, height: int):
    circuit = qiskit.QuantumCircuit(width * height)
    for i in range(width - 1):
        for j in range(height):
            a = i + j * width
            b = i + 1 + j * width
            circuit.cx(a, b)
    for j in range(height - 1):
        for i in range(width):
            a = i + j * width
            b = i + (j + 1) * width
            circuit.cx(a, b)
    return circuit


def build_line_circut(n: int):
    circuit = qiskit.QuantumCircuit(n)
    circuit.h(0)
    for i in range(n - 1):
        circuit.cx(i, i + 1)
    return circuit


def build_circuit_list(circuits_string: str, force_list: bool = False):
    """
    Builds circuits based on a very simple language. Syntax is
    ```
    [count] type[ param[ param[...]] [[count] type ...]
    ```
    where brackets denote optional values and spaces are literal spaces. See source for supported
    types.
    """

    def generate(func, params, count):
        def try_int(value):
            try:
                return int(value)
            except ValueError:
                return value

        params = [try_int(param) for param in params]
        return [func(*params) for _ in range(count)]

    factory_func_table = {
        "ghz": build_ghz_circuit,
        "partial": build_partially_used_circuit,
        "star": lambda: build_ghz_circuit(5),
        "h": lambda n: build_partially_used_circuit(n, n),
        "grid": build_grid_circuit,
        "line": build_line_circut,
    }
    circuit_list, explicit_count_specified = [], False
    instruction_stream = iter(circuits_string.split(" "))
    for circuit_type in instruction_stream:
        try:
            count = int(circuit_type)
            circuit_type = next(instruction_stream)
            explicit_count_specified = True
        except ValueError:
            count = 1

        factory_func = factory_func_table[circuit_type]
        num_params = len(signature(factory_func).parameters)
        params = list(itertools.islice(instruction_stream, num_params))

        circuit_list.extend(generate(factory_func, params, count))
    if not explicit_count_specified and not force_list and len(circuit_list) == 1:
        return circuit_list[0]
    return circuit_list
