from collections.abc import Mapping
from typing import Any

import qiskit
import qiskit.dagcircuit
import qiskit.transpiler

from qc_parallelizer.extensions.typing import isnestedinstance


def layout_to_dict(
    layout: Any,
    circuit: qiskit.QuantumCircuit,
) -> dict[int, int]:
    """
    Given a qubit layout of basically any form, this function normalizes it into a dictionary that
    contains virtual-physical index mappings. That is, keys represent the virtual qubit indices in
    the circuit, and values represent the physical qubit indices in the backend.

    Note that if the layout cannot be resolved, an empty dict (`{}`) is returned, not None.
    """

    if layout is None:
        return {}
    if isinstance(layout, IndexedLayout):
        return layout.v2p
    if isnestedinstance(layout, dict[int, int]):
        return layout

    if isinstance(layout, dict):
        layout = qiskit.transpiler.Layout(layout)
    elif isnestedinstance(layout, list[int]):
        layout = qiskit.transpiler.Layout.from_intlist(layout, *circuit.qregs)
    else:
        raise TypeError(f"invalid layout format (got '{layout}')")

    return {
        circuit.find_bit(qubit).index: physical_index
        for qubit, physical_index in layout.get_virtual_bits().items()
    }


def adjust_indices(
    indices: set[int],
    blocked: set[int],
) -> tuple[dict[int, int | None], dict[int, int]]:
    """
    Small helper for adjusting qubit indices based on a set of blocked qubits. Returns a `dict` of
    the mapped indices and an inverse mapping.

    This is best explained visually:
    ```
    0 [blocked] -> None  ,----> 0
                        /
    1 -----------------'  ,---> 1
                         /
    2 [blocked] -> None /   ,-> 2
                       /   /
    3 ----------------'   /
                         /
    4 ------------------'
    ```
    """

    def adjust(index):
        if index in blocked:
            return None
        blocked_below = len({other for other in blocked if other < index})
        return index - blocked_below

    mapping = {index: adjust(index) for index in indices}
    return mapping, {b: a for a, b in mapping.items() if b is not None}


class IndexedLayout:
    """
    Class for representing one-to-one qubit index mappings between virtual circuit indices and
    physical backend indices. Resembles Qiskit's Layout class, but has a restricted and improved
    set of features.

    Note on terminology: in this context, a layout contains a mapping. While the two terms might be
    used interchangeably, "layout" refers to a one-to-one mapping that is specifically used to map
    physical qubits to virtual qubits or vice versa, while "mapping" refers to the dictionary
    instance(s) that represent it. At least in this class.
    """

    _v2p: dict[int, int]
    _p2v: dict[int, int]

    @classmethod
    def from_layout(
        cls,
        layout: Any,
        circuit: qiskit.QuantumCircuit,
    ):
        return cls(v2p=layout_to_dict(layout, circuit))

    @classmethod
    def from_circuit(cls, circuit: qiskit.QuantumCircuit):
        if circuit.layout is None:
            return cls()
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
                    if v in property_set["original_qubit_indices"]
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

    def __init__(self, v2p: Mapping[int, int] | None = None, p2v: Mapping[int, int] | None = None):
        """
        Constructs a layout from either a virtual-physical mapping, a physical-virtual mapping, or
        no mapping at all.
        """

        if v2p is None and p2v is None:
            self._v2p = {}
            self._p2v = {}
        elif v2p is not None:
            self._v2p = dict(v2p)
            self._p2v = {p: v for v, p in v2p.items()}
        elif p2v is not None:
            self._v2p = {v: p for p, v in p2v.items()}
            self._p2v = dict(p2v)
        else:
            raise ValueError("only up to one mapping may be provided as an initializer")
        self._validate()

    def _validate(self):
        try:
            assert len(self.vindices) == len(self.pindices)
        except AssertionError as error:
            raise IndexError("mapping is not bijective") from error

    @property
    def size(self):
        """
        Returns the size of the layout, which is the number of mapped qubits.
        """

        return len(self.v2p)

    @property
    def vindices(self):
        return set(self._v2p.keys())

    @property
    def pindices(self):
        return set(self._p2v.keys())

    @property
    def v2p(self):
        return self._v2p

    @property
    def p2v(self):
        return self._p2v

    def copy(self):
        return type(self)(p2v={**self._p2v})

    def add(self, virt: int, phys: int):
        self._v2p[virt] = phys
        self._p2v[phys] = virt

    def remove(
        self,
        virt: int | None = None,
        phys: int | None = None,
        decrement_keys: bool = False,
    ):
        if virt is None and phys is None:
            raise ValueError("both `virt` and `phys` cannot be None")
        elif virt is not None:
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
                if other_virt > virt:  # type: ignore # virt is known to be non-None at this point
                    phys = self._v2p[other_virt]
                    del self._v2p[other_virt]
                    new_virt = other_virt - 1
                    new_v2p[new_virt] = phys
                    self._p2v[phys] = new_virt
            self._v2p = {**self._v2p, **new_v2p}

    def with_entry(self, virt: int, phys: int):
        return type(self)(p2v={**self._p2v, phys: virt})

    def map(self, mapping: dict[int, int]):
        """
        Maps the layout's physical indices according to `mapping`.
        """
        return type(self)(p2v={mapping[p]: v for p, v in self._p2v.items()})

    def to_qiskit_layout(self, circuit: qiskit.QuantumCircuit) -> qiskit.transpiler.Layout:
        """
        Converts to a Qiskit Layout object.
        """
        return qiskit.transpiler.Layout({circuit.qubits[v]: p for v, p in self._v2p.items()})

    def to_physical_list(self) -> list[int | None]:
        """
        Converts to a list of physical qubit indices. Gaps, if present, are filled with None.
        """
        v2p = self.v2p
        num_virtual = max(v2p.keys()) + 1
        return [v2p[v] if v in v2p else None for v in range(num_virtual)]

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
                f"v_{v} ~ p_{p}" for p, v in sorted(self._p2v.items(), key=self._p2v_sort_key)
            )
            + "}"
        )

    def _iqmstr(self):
        return (
            "{"
            + ", ".join(
                f"QB{p + 1}: {v}" for p, v in sorted(self._p2v.items(), key=self._p2v_sort_key)
            )
            + "}"
        )


class CircuitWithLayout:
    circuit: qiskit.QuantumCircuit
    layout: IndexedLayout

    def __init__(self, circuit: qiskit.QuantumCircuit, layout: Any):
        self.circuit = circuit
        self.layout = IndexedLayout.from_layout(layout, circuit)

    @property
    def full_layout(self) -> bool:
        """
        `True` if the layout covers all virtual qubits, `False` otherwise.
        """
        return self.layout.size == self.circuit.num_qubits

    @property
    def layout_fraction(self) -> float:
        """
        A float in the range [0, 1] that indicates how many of the circuit's qubits have a mapping
        defined. For example, if 3/4 of the qubits are mapped, this property is `0.75`.
        """
        return self.layout.size / self.circuit.num_qubits
