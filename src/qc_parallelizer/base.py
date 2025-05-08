import qiskit
import qiskit.circuit
import qiskit.providers
import qiskit.result
import qiskit.transpiler


class Types:
    Layout = list | dict | qiskit.transpiler.Layout
    Backend = qiskit.providers.BackendV2
    Result = qiskit.result.result.Result
    Qubit = qiskit.circuit.Qubit


class Exceptions:
    class MissingParameter(Exception):
        """A required parameter or part of a parameter is missing."""

    class MissingInformation(Exception):
        """
        Required information is missing from data. For example, a circuit object was passed that
        does not contain required metadata.
        """

    class ParameterConflict(Exception):
        """Two or more passed parameters or parts of parameters are in mutual conflict."""

    class CircuitBackendCompatibility(Exception):
        """One or more given circuits cannot be executed on any given backends."""

    class InvalidLayout(Exception):
        """
        One or more given circuit layouts are not valid. This can be due to incompleteness or
        duplicate/overlapping definitions.
        """
