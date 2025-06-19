import collections
import itertools

import qiskit
import qiskit.qobj
import qiskit.result.models
from qc_parallelizer.base import Exceptions, Types


def split_results(result: Types.Result):
    """
    Post-processes results from a parallel execution by splitting the result into smaller Result
    objects. This is only required if the results were not already processed, e.g. in case you, for
    whatever reason, get hold of the raw Result object from the call to `backend.run()`.

    Users must not call this on arbirary results that do not contain metadata about parallel
    execution.
    """

    # Here we define an inner generator function that can yield split results from inner loop
    # iterations somewhat more elegantly than by e.g. repeatedly appending to a list.

    def _generate_split_results(host_result: Types.Result):
        for experiment_result in host_result.results:
            if not hasattr(experiment_result.data, "counts"):
                continue

            # Different backends return slightly different data formats
            # TODO: expand this as necessary, or look for some method in the Result class that would
            # return the metadata format-agnostically
            if hasattr(experiment_result.header, "metadata"):
                # IBM's simulator backends have metadata here
                metadata = experiment_result.header.metadata
            elif hasattr(experiment_result.data, "metadata"):
                # And IQM's backends have it here
                metadata = experiment_result.data.metadata
            else:
                raise Exceptions.MissingInformation(
                    "cannot access circuit metadata in Result object",
                )

            try:
                original_circuits = metadata["hosted_circuits"]
                original_clreg_sizes = [
                    list(circuit["registers"]["clbit"]["sizes"].values())
                    for circuit in original_circuits
                ]
                indices: list[int] = [
                    original_circuit["metadata"]["index"] for original_circuit in original_circuits
                ]
            except KeyError as error:
                raise Exceptions.MissingInformation(
                    "parallelization metadata is not accessible and is probably missing (key: "
                    f"{error}) - the provided results must be returned from a parallelized "
                    "execution",
                ) from error

            parse_key = _determine_key_parser(experiment_result.data.counts, original_clreg_sizes)

            total_counts = [collections.Counter() for _ in original_circuits]
            for k, count in experiment_result.data.counts.items():
                for i, key in enumerate(parse_key(k)):
                    # Index from the back to please Qiskit
                    total_counts[-i - 1][key] += count

            for index, original_circuit, counts in zip(indices, original_circuits, total_counts):
                data = qiskit.result.models.ExperimentResultData(counts=dict(counts))
                exp_result = qiskit.result.models.ExperimentResult(
                    experiment_result.shots,
                    experiment_result.success,
                    data,
                    experiment_result.meas_level,
                    header=qiskit.qobj.QobjExperimentHeader(
                        # Place metadata here, the "correct" place
                        # (This is where you would actually find it)
                        metadata=original_circuit["metadata"]["original_metadata"],
                    ),
                )
                yield (
                    index,
                    qiskit.result.result.Result(
                        host_result.backend_name,
                        host_result.backend_version,
                        host_result.qobj_id,
                        host_result.job_id,
                        host_result.success,
                        [exp_result],
                        host_result.date,
                        host_result.status,
                        qiskit.qobj.QobjHeader(
                            # Also place metadata here, the "sensible" place
                            # (This is where you would expect to find it)
                            metadata=original_circuit["metadata"]["original_metadata"],
                        ),
                        **host_result._metadata,
                    ),
                )

    return list(_generate_split_results(result))


def _split_bitstring(bitstring: str, register_sizes: list[list]):
    """
    Splits a bitstring into an array of smaller bitstrings, based on the given register sizes. This
    is aware of multi-register configurations, which will be placed in the same string, but
    separated by a space.

    `register_sizes` should be an array of arrays, one array per circuit and one array element per
    register in that circuit. This is intended for situations where measurement bitstrings from
    multiple circuits are merged together.

    For example, if the register sizes are `[[1], [2], [1, 1]]`, the bitstring `"00101"` is split
    into `["0 0", "10", "1"]`. Note how register sizes are interpreted in reverse - this is to
    comply with Qiskit's conventions.
    """

    # This relies on some iterator magic - first, we get an iterator to the bitstring
    it = iter(bitstring)

    return [
        # Then, for each of the size lists, which specify values for `num_bits`, we take the next
        # elements from the iterator - the iterator is required (for elegance) because it remembers
        # the previous index
        " ".join("".join(itertools.islice(it, num_bits)) for num_bits in sizes)
        # And we iterate backwards due to Qiskit's indexing
        for sizes in register_sizes[::-1]
    ]


def _determine_key_parser(counts, register_sizes):
    """
    Returns a parser function for the counts keys, based on a sample key from the counts object as
    well as the register sizes.

    For example, if the register sizes are `[[1], [2], [2]]`, possible input strings are
        - `"101"`
        - `"00101"`
        - `"00 10 1"`
        - `"0x5"`

    and the output we want for all of these is
        - `["00", "10", "1"]`

    This also takes into account multiple classical registers per circuit, in which case the
    keys will include spaces.

    This function exists for dynamically determining a parser for counts keys, when there are
    potentially a large number of keys to be processed. It should be more efficient to call the
    function returned by this function than to check the key format on each loop iteration.
    """

    # Grab the first key from `counts`
    sample_key = next(iter(counts))

    if sample_key.startswith("0x"):
        # Hexadecimal, so we parse hex -> int -> bin -> spaced

        total_bits = sum(itertools.chain(*register_sizes))
        format_string = f"{{:0{total_bits}b}}"

        # Now we have a format string like "{:0Nb}", where N is the number of bits
        # This formats an integer as a zero-padded bit string of length N
        def parse_key(key):
            as_int = int(key, 16)
            as_bin = format_string.format(as_int)
            return _split_bitstring(as_bin, register_sizes)

    elif " " in sample_key:
        # Probably spaced out binary, so remove spaces and respace to ensure correct formatting

        def parse_key(key):
            return _split_bitstring(key.replace(" ", ""), register_sizes)

    else:
        # Already bin, but no spaces, so we just need to space it

        def parse_key(key):
            return _split_bitstring(key, register_sizes)

    return parse_key
