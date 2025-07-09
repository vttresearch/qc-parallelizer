# Parallelizer notebooks

This folder contains example notebooks for demonstration and documentation purposes.

## General

- **[Introduction](./parallel-circuits-introduction.ipynb)**

  This covers the motivation, basic idea, and usage of the package. If the project is new to you,
  start with this notebook.

## Feature-specific

- **[Layouts](./circuit-layouts.ipynb)**

  Details on how layout information can be passed to the parallelizer. This is practical if certain
  circuit qubits need to be forced onto certain physical qubits.

- **[Packers](./circuit-packers.ipynb)**

  A list and documentation of circuit packers. Packers control the physical placement of circuits,
  considering the backend's topology and positioning relative to other circuits.
