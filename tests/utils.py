"""
Base fixtures and other general definitions that are used in multiple tests.
"""

import itertools
from inspect import signature
from typing import Any

import iqm.qiskit_iqm as iqm
import qiskit

fake_20qb_backend = iqm.IQMFakeApollo()


def build_ghz_circuit(n):
    """Builds an n-qubit GHZ circuit. For n = 2, this is the Bell state circuit."""

    circuit = qiskit.QuantumCircuit(n)
    circuit.h(0)
    if n > 1:
        circuit.cx(0, list(range(1, n)))
    return circuit


def build_partially_used_circuit(num_qubits, num_used):
    """Builds a circuit with a gate on a subset of the qubits, leaving the rest unused."""

    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.h(list(range(num_used)))
    return circuit


def build_circuit_list(circuits_string: str):
    """
    Builds circuits based on a very simple language. Syntax is essentially
    ```
    [count] type[ param[ param[...]] [[count] type ...]
    ```
    where brackets denote optional values and spaces are literal spaces.

    Currently supported types are `ghz`, `partial`, `star`, and `h`. See source for more info.
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
    if not explicit_count_specified and len(circuit_list) == 1:
        return circuit_list[0]
    return circuit_list
