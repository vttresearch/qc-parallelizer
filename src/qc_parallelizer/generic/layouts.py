import heapq
import itertools
from typing import Any

import qiskit
import qiskit.dagcircuit
import qiskit.transpiler


def layout_to_dict(
    layout: qiskit.transpiler.Layout | list | dict | None,
    circuit: qiskit.QuantumCircuit,
) -> dict[int, int]:
    """
    Given a qubit layout of basically any form that Qiskit understands, this function normalizes it
    into a dictionary that contains virtual-physical index mappings. That is, keys represent the
    virtual qubit indices in the circuit, and values represent the physical qubit indices in the
    backend.

    Note that if the layout cannot be resolved, an empty dict (`{}`) is returned, not None.
    """

    if isinstance(layout, list):
        # The Qiskit parser does not have information about registers, so we have to handle this
        # case separately
        # TODO: a list of something else, like Qubit objects, might also be given
        layout = qiskit.transpiler.Layout.from_intlist(layout, *circuit.qregs)
    else:
        # If not a list, we let the default parser do its thing - this handles Layout objects,
        # dicts, and other formats
        layout = qiskit.compiler.transpiler._parse_initial_layout(layout)
    # If there is no layout, return an empty mapping
    if layout is None:
        return {}
    # Finally, return a mapping from virtual indices to physical indices
    return {
        circuit.find_bit(qubit).index: physical_index
        for qubit, physical_index in layout.get_virtual_bits().items()
    }


class QILayout:
    """
    Class for representing one-to-one qubit index mappings between virtual circuit indices and
    physical backend indices. Resembles Qiskit's Layout class, but has a restricted and improved
    set of features.

    Note on terminology: in this context, a layout contains a mapping. While the two terms might be
    used interchangeably, "layout" refers to a one-to-one mapping that is specifically used to map
    physical qubits to virtual qubits or vice versa, while "mapping" refers to the dictionary
    instance(s) that represent it. At least in this class.
    """

    @classmethod
    def from_layout(
        cls,
        layout: qiskit.transpiler.Layout | list | dict | None,
        circuit: qiskit.QuantumCircuit,
    ):
        return cls(v2p=layout_to_dict(layout, circuit))

    @classmethod
    def from_circuit(cls, circuit: qiskit.QuantumCircuit):
        initial_layout = circuit.layout.initial_virtual_layout()
        return cls(
            p2v={
                p: circuit.layout.input_qubit_mapping[v]
                for p, v in initial_layout.get_physical_bits().items()
            },
        )

    @classmethod
    def from_property_set(
        cls,
        property_set: dict[str, Any],
        dag: qiskit.dagcircuit.DAGCircuit | None = None,
    ):
        if property_set["original_qubit_indices"] and property_set["layout"]:
            return cls(
                p2v={
                    p: property_set["original_qubit_indices"][v]
                    for p, v in property_set["layout"].get_physical_bits().items()
                },
            )
        elif property_set["layout"] and dag is not None:
            return cls(
                p2v={
                    p: dag.find_bit(qubit).index
                    for p, qubit in property_set["layout"].get_physical_bits().items()
                },
            )
        return cls()

    @classmethod
    def from_trivial(cls, num_qubits: int):
        return cls(v2p={i: i for i in range(num_qubits)})

    def __init__(self, v2p: dict[int, int] | None = None, p2v: dict[int, int] | None = None):
        if v2p is None and p2v is None:
            self._v2p: dict[int, int] = {}
            self._p2v: dict[int, int] = {}
        elif v2p is not None:
            self._v2p: dict[int, int] = v2p
            self._p2v: dict[int, int] = {}
            for k, v in v2p.items():
                if v is not None:
                    self._p2v[v] = k
        elif p2v is not None:
            self._v2p: dict[int, int] = {}
            self._p2v: dict[int, int] = p2v
            for k, v in p2v.items():
                if v is not None:
                    self._v2p[v] = k
        else:
            raise ValueError("only up to one mapping may be provided as an initializer")

    @property
    def size(self):
        return len(self._v2p)

    @property
    def vkeys(self):
        return self._v2p.keys()

    @property
    def pkeys(self):
        return self._p2v.keys()

    @property
    def v2p(self):
        return {v: p for v, p in self._v2p.items() if self._p2v[p] is not None}

    @property
    def p2v(self):
        return {p: v for p, v in self._p2v.items() if v is not None}

    def copy(self):
        return QILayout(p2v={**self._p2v})

    def add(self, virt: int, phys: int):
        self._v2p[virt] = phys
        self._p2v[phys] = virt

    def remove(
        self,
        virt: int | None = None,
        phys: int | None = None,
        decrement_keys: bool = False,
    ):
        if virt is not None:
            phys = self._v2p[virt]
        elif phys is not None:
            virt = self._p2v[phys]
        if virt in self._v2p:
            del self._v2p[virt]
        if phys in self._p2v:
            del self._p2v[phys]
        if decrement_keys:
            new_v2p = {}
            for other_virt in list(self._v2p.keys()):
                if other_virt > virt:
                    phys = self._v2p[other_virt]
                    del self._v2p[other_virt]
                    new_virt = other_virt - 1
                    new_v2p[new_virt] = phys
                    self._p2v[phys] = new_virt
            self._v2p = {**self._v2p, **new_v2p}

    def block(self, phys: int | set[int]):
        """
        Blocks a set of physical indices. The index will be reported as blocked when calling
        `{is, get}_blocked` and will not appear on `v2p` or `p2v`.
        """

        if isinstance(phys, int):
            phys = {phys}
        for i in phys:
            self._p2v[i] = None

    def with_entry(self, virt: int, phys: int):
        return QILayout(p2v={**self._p2v, phys: virt})

    def with_blocked(self, phys: int | set[int]):
        copy = self.copy()
        copy.block(phys)
        return copy

    def is_blocked(self, phys: int):
        return phys in self._p2v and self._p2v[phys] is None

    def get_blocked(self):
        return {p for p, v in self._p2v.items() if v is None}

    def insert_blocked_indices(self, phys_set: set[int]):
        """
        Similar to `block`, but increments other physical indices as if these qubits were inserted
        into the backend.
        """
        new_phys_indices = {p: p for p in self._p2v.keys()}

        for b in sorted(phys_set):
            for p in new_phys_indices:
                if new_phys_indices[p] >= b:
                    new_phys_indices[p] += 1

        for p, v in sorted(self._p2v.items(), reverse=True):
            del self._p2v[p]
            self._p2v[new_phys_indices[p]] = v
        for v, p in list(self._v2p.items()):
            self._v2p[v] = new_phys_indices[p]
        for p in phys_set:
            self._p2v[p] = None

    def to_qiskit_layout(self, circuit: qiskit.QuantumCircuit) -> qiskit.transpiler.Layout:
        """
        Converts to a Qiskit Layout object. Discards blocked indices.
        """
        return qiskit.transpiler.Layout({circuit.qubits[v]: p for v, p in self._v2p.items()})

    def __repr__(self):
        return f"QILayout(p2v={self._p2v.__repr__()})"

    @staticmethod
    def _p2v_sort_key(item):
        p, v = item
        return (v is None, p)

    def __str__(self):
        return (
            "{"
            + ", ".join(
                (f"!p_{p}" if v is None else f"v_{v} ~ p_{p}")
                for p, v in sorted(self._p2v.items(), key=self._p2v_sort_key)
            )
            + "}"
        )

    def _iqmstr(self):
        return (
            "{"
            + ", ".join(
                (f"!QB{p + 1}" if v is None else f"QB{p + 1}: {v}")
                for p, v in sorted(self._p2v.items(), key=self._p2v_sort_key)
            )
            + "}"
        )
