import itertools
import typing
from datetime import datetime

if typing.TYPE_CHECKING:
    from ..backends import BackendCircuitBin
    from ..jobs import ParallelizerJobBatch


def plot_timeline(
    timeline: dict,
    collapse_ratio: float | None = 0.2,
    figsize: tuple[float, float] = (16, 6),
    min_duration: float = 0.0,
):
    """
    Plots a parallelization timeline, as returned by `ParallelizedBackend.timeline`. Requires
    Matplotlib to be installed.

    Args:
        collapse_ratio:
            Regions with long idling that last longer than this ratio of the total time window are
            collapsed from the figure. For example, if set to 0.2, and a job takes over 20% of the
            total timeline to execute, its full duration will not be displayed. Set to None to
            disable.
        min_duration:
            Timeline events shorter than this (in seconds) are extended to last this long. This is
            useful for ensuring that all events are visible, regardless of length.
    """

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import qiskit.visualization.utils
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import TABLEAU_COLORS
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise RuntimeError("missing optional dependencies") from exc

    request_times = [
        (req, job) for events in timeline.values() for (*_, req), job in events if req is not None
    ]

    all_times = sorted(
        [t for events in timeline.values() for times, _ in events for t in times if t is not None],
    )
    min_time, max_time = all_times[0], all_times[-1]
    total_time_window = max_time - min_time

    collapsible = []
    if collapse_ratio is not None:
        for a, b in itertools.pairwise(all_times):
            if (b - a) / total_time_window >= collapse_ratio:
                collapsible.append((a, b))

    # First, split the timeline into three or more "tracks": in preparation, waiting, and one per
    # backend for in execution.

    track_keys = [
        "prep",
        *(
            job.bin.backend
            for events in timeline.values()
            for _, job in events
            if job.bin is not None
        ),
    ]
    boxes = {k: [] for k in track_keys}
    for events in timeline.values():
        if len(events) == 0:
            continue
        for (a, b, c, d, _), job in events:
            preparation = a, b
            waiting = b, c
            executing = c, d
            if None not in preparation:
                boxes["prep"].append((preparation, (job, False)))
            if None not in waiting:
                boxes[job.bin.backend].append((waiting, (job, True)))
            if None not in executing:
                boxes[job.bin.backend].append((executing, (job, False)))

    # Then, rearrange each track so that overlapping boxes are positioned in parallel instead of on
    # top of each other.

    for track, intervals in boxes.items():
        sorted_events = sorted(
            (
                ev
                for (a, b), job in intervals
                for ev in ((a, True, (a, b), job), (b - 1e-5, False, (a, b), job))
            ),
            key=lambda ev: (ev[0], ev[1]),
        )

        box_buffer, lanes, max_lanes = [], {}, 0
        boxes[track] = []
        for _, is_start, box, job in sorted_events:
            if is_start:
                lanes[job] = next(i for i in itertools.count() if i not in lanes.values())
                max_lanes = max(max(lanes.values()) + 1, max_lanes)
                box_buffer.append((box, lanes[job], job))
            else:
                del lanes[job]
                if len(lanes) == 0:
                    for box, lane, job in box_buffer:
                        w = 0.9 / max_lanes
                        r = 0.9 / 2 - w / 2
                        boxes[track].append(
                            (
                                *box,
                                w,
                                (r * 2.0 * (lane / (max_lanes - 1) - 0.5)) if max_lanes > 1 else 0,
                                job,
                            ),
                        )
                    box_buffer = []
                    max_lanes = 0

    color_table, color_cycle = {}, itertools.cycle(TABLEAU_COLORS.values())

    def color_for(job):
        if job not in color_table:
            color_table[job] = next(color_cycle)
        return color_table[job]

    # Lastly, convert each track into a list of rectangles with colors determined from the default
    # Matplotlib color cycle based on the job.

    verts, colors = [], []
    for i, (track, intervals) in enumerate(boxes.items()):
        for a, b, w, offset, (job, tint) in intervals:
            if b - a < min_duration:
                b = a + min_duration
            x, y = (
                mdates.date2num(datetime.fromtimestamp(a)),
                mdates.date2num(datetime.fromtimestamp(b)),
            )
            v = [
                (x, i + offset - w / 2),
                (y, i + offset - w / 2),
                (y, i + offset + w / 2),
                (x, i + offset + w / 2),
            ]
            colors.append((color_for(job), 0.3 if tint else 1.0))
            verts.append(v)

    # Then figure out where, if anywhere, to collapse the x-axis.

    xlims = [min_time]
    for a, b in collapsible:
        xlims.append(a)
        xlims.append(b)
    xlims.append(max_time)
    xlims = [(xlims[i], xlims[i + 1]) for i in range(0, len(xlims), 2)]

    # And finally draw it all.

    fig, axs = plt.subplots(figsize=figsize, ncols=len(collapsible) + 1, squeeze=False)

    for i, (ax, xlim) in enumerate(zip(axs[0], xlims)):
        length = max(xlim[1] - xlim[0], 1e-3)
        padded_xlim = xlim[0] - length * 0.05, xlim[1] + length * 0.05

        ax.vlines(
            [mdates.date2num(datetime.fromtimestamp(t)) for t, _ in request_times],
            ymin=-0.5,
            ymax=len(boxes) - 0.5,
            colors=[color_for(job) for _, job in request_times],
            alpha=0.6,
            zorder=-2,
        )

        ax.axhline(
            y=0.5,
            color="black",
            linestyle="dashed",
            alpha=0.3,
            zorder=-1,
        )

        bars = PolyCollection(verts, facecolors=colors)
        ax.add_collection(bars)

        ax.set_xlim(*(mdates.date2num(datetime.fromtimestamp(t)) for t in padded_xlim))
        ax.set_ylim(-0.5, len(boxes) - 0.5)

        num_ticks = 5 if len(collapsible) == 0 else (3 if len(collapsible) <= 2 else 2)
        ax.set_xticks(
            [
                mdates.date2num(datetime.fromtimestamp(t))
                for t in [
                    padded_xlim[0] + (padded_xlim[1] - padded_xlim[0]) * i / (num_ticks - 1)
                    for i in range(num_ticks)
                ]
            ],
        )
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: mdates.num2date(x).strftime("%T.%f")[:-5]),
        )
        ticklabels = ax.get_xticklabels()
        ticklabels[0].set_horizontalalignment("left")
        ticklabels[-1].set_horizontalalignment("right")

        if i == 0:
            ax.set_yticks(range(len(boxes)))
            ax.set_yticklabels(
                [
                    {"prep": "in preparation", "wait": "waiting"}.get(k, None) or k.name
                    for k in boxes.keys()
                ],
            )
            ax.spines[["top", "right"]].set_visible(False)
        else:
            ax.set_yticks([])
            ax.spines[["left", "top", "right"]].set_visible(False)
            ax.set_ylabel("- - - cut - - -", labelpad=1.5 * figsize[0] / (len(collapsible) + 1))

    fig.tight_layout()
    qiskit.visualization.utils.matplotlib_close_if_inline(fig)
    return fig


def plot_job_batch(
    job_batch: "ParallelizerJobBatch",
    circuit_colors: list[str] = [
        "#dd0000",
        "#008800",
        "#0000bb",
        "#0099aa",
        "#aa00aa",
        "#aa9900",
    ],
    active_coupler_color: str = "black",
    idle_qubit_color: str = "grey",
    idle_coupler_color: str = "grey",
    qubit_size: int = 80,
    coupler_width: int = 25,
    font_size: int | None = None,
    dpi: int = 100,
    figsize=None,
    use_iqm_labeling: bool = False,
):
    """
    Plots the given job's chosen layout(s) on the backend(s). Requires Matplotlib to be installed.
    Required dependencies can be installed with `pip install qc_parallelizer[visualization]`.

    Args:
        circuit_colors:
            A list of color strings that will be cycled through to color different circuits in each
            bin.
        active_coupler_color:
            A color string for coloring active couplers.
        idle_qubit_color:
            A color string for coloring idle qubits (from this job's perspective).
        idle_coupler_color:
            A color string for coloring idle couplers (from this job's perspective).
        use_iqm_labeling:
            If True, qubits will be labeled by IQM convention (QB1, QB2, etc.). Otherwise, the
            labels are simply the raw qubit indices.

    Returns:
        A Matplotlib Figure. If used in a notebook, this function takes care of closing unwanted
        duplicates that may be displayed automatically.
    """

    try:
        import matplotlib.pyplot as plt
        import qiskit.visualization.utils
    except ImportError as exc:
        raise RuntimeError("missing optional dependencies") from exc

    relevant_bins = {job.bin for job in job_batch.jobs}

    # Do a little dance to make the type checker happy
    assert None not in relevant_bins
    relevant_bins = typing.cast(set["BackendCircuitBin"], relevant_bins)

    job_bins = {
        bin: [job for job in job_batch.jobs if job in bin]
        for bin in relevant_bins
        if not bin.is_empty
    }

    fig, axs = plt.subplots(
        ncols=len(job_bins),
        dpi=dpi,
        figsize=figsize,
        squeeze=False,
    )

    def get_qubit_colors(qubit_indices, num_qubits):
        indices = [None] * num_qubits
        for i, qubits in enumerate(qubit_indices):
            for q in qubits:
                indices[q] = i
        return [
            circuit_colors[i % len(circuit_colors)] if i is not None else idle_qubit_color
            for i in indices
        ]

    def get_coupler_colors(circuit_couplers, backend_couplers):
        return [
            active_coupler_color if (a, b) in circuit_couplers else idle_coupler_color
            for a, b in backend_couplers
        ]

    for i, (bin, jobs) in enumerate(job_bins.items()):
        qubit_indices = [job.layout.pindices for job in jobs]
        all_couplers = [
            edge
            for job in jobs
            for edge in (
                (job.layout.v2p[a], job.layout.v2p[b]) for a, b in job.circuit.get_edges(bidir=True)
            )
        ]
        qubit_colors = get_qubit_colors(qubit_indices, bin.backend.num_qubits)
        coupler_colors = get_coupler_colors(all_couplers, bin.backend.edges)
        qiskit.visualization.plot_coupling_map(
            num_qubits=bin.backend.num_qubits,
            qubit_coordinates=None,
            coupling_map=bin.backend.edges,
            ax=axs[0, i],
            planar=False,
            qubit_size=qubit_size,
            font_size=font_size if font_size is not None else (18 if use_iqm_labeling else 24),
            line_width=coupler_width,
            qubit_color=qubit_colors,
            line_color=coupler_colors,
            qubit_labels=[f"QB{i + 1}" for i in range(bin.backend.num_qubits)]
            if use_iqm_labeling
            else None,
        )

    fig.tight_layout()
    qiskit.visualization.utils.matplotlib_close_if_inline(fig)
    return fig
