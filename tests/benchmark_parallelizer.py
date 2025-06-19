"""
Little script for benchmarking the parallelizer. Runs a fixed set of benchmarks, first some fixed-
size tests with a timer and then some parametrized tests until a time limit is reached.

Results are saved into /tmp (or equivalent, whatever tempfile provides) for comparison with later
results. If earlier results are present, they are loaded and automatically compared.

The code might not be the most readable kind but it works and produces pretty output :)
"""

import argparse
import cProfile
import functools
import json
import os
import pstats
import shutil
import sys
import tempfile
import time

import qc_parallelizer as parallelizer
from qc_parallelizer import packers
from utils import build_circuit_list, fake_54qb_backend_cluster

# Enable/disable ANSI colors depending on output type (or user preference)
if os.isatty(0) and "--no-color" not in sys.argv:

    class Color:  # type: ignore
        enabled = True
        Reset = "\033[0m"
        Grey = "\033[38;5;245m"
        Red = "\033[91m"
        Green = "\033[92m"
        Magenta = "\033[95m"
        Cyan = "\033[96m"
        BgBlue = "\033[104m\033[30m"
        BgWhite = "\033[107m\033[30m"
        Bold = "\033[1m"

        @staticmethod
        def BgGrey(level: float):
            return f"\033[48;5;{232 + level * 23:.0f}m\033[30m"
else:

    class DummyNoColorMetaclass(type):
        def __getattr__(self, attr):
            # No colors for you
            return ""

        @staticmethod
        def BgGrey(level: float):
            return ""

    class Color(metaclass=DummyNoColorMetaclass):
        enabled = False


save_filename = os.path.join(tempfile.gettempdir(), "parallelizer-bm-hist.json")
history = {}
benchmarks_run = 0


def load_history():
    global history
    try:
        with open(save_filename) as f:
            history = json.load(f)
    except FileNotFoundError:
        pass


def save_history():
    with open(save_filename, "w") as f:
        json.dump(history, f)


def print_title(title: str):
    if Color.enabled:
        width = shutil.get_terminal_size().columns
        remain = width - (len(title) + 2)
        print(
            f"{Color.BgWhite} {title} "
            f"{''.join(Color.BgGrey(1 - i / (remain * 1.5)) + ' ' for i in range(remain))}"
            f"{Color.Reset}",
        )
    else:
        print("=", title, "=" * (61 - len(title)))


def init_benchmark():
    current_benchmark = benchmarks_run + 1
    next_on_new_line = True

    if Color.enabled:
        print(f"{Color.BgWhite} Benchmark {current_benchmark} {Color.Reset}")
    else:
        print(f"[ Benchmark {current_benchmark} ]")

    def prefix_print(*args, color=Color.Reset, **kwargs):
        nonlocal next_on_new_line
        assert set(kwargs.keys()) <= {"end", "flush"}
        lines = " ".join(args).split("\n")
        for index, line in enumerate(lines):
            if next_on_new_line or index > 0:
                if Color.enabled:
                    print(
                        (
                            f"{Color.BgGrey(max(1 - index * 0.1, 0.2))}"
                            f" {current_benchmark} {Color.Reset} "
                        ),
                        end="",
                    )
                else:
                    print(f"[ {current_benchmark} ] ", end="")
            print(color, end="")
            if index == len(lines) - 1:
                print(line + Color.Reset, **kwargs)
            else:
                print(line + Color.Reset)
        next_on_new_line = "end" not in kwargs

    return prefix_print


def get_packer(name: str):
    return eval(f"packers.{name}")


def run_single(circuits_string: str, backends, packer: str, show: bool = False):
    global benchmarks_run
    print = init_benchmark()

    print(f"Generating {Color.Cyan}'{circuits_string}'{Color.Reset}... ", end="", flush=True)
    circuits = build_circuit_list(circuits_string, force_list=True)
    print(f"rearranging {Color.Cyan}{len(circuits)} circuits{Color.Reset}...")

    packer_ = get_packer(packer)
    start = time.time()
    rearranged = parallelizer.rearrange(circuits, backends, packer=packer_)
    duration = time.time() - start

    print(
        (
            f"==> {Color.Magenta}{duration:.2f} seconds{Color.Reset} "
            f"({Color.Magenta}{len(circuits) / duration:.2f} circ/s{Color.Reset}, "
            f"{Color.Magenta}{duration / len(circuits):.2f} s/circ{Color.Reset}, "
            f"{Color.Magenta}{sum(c.num_qubits for c in circuits) / duration:.2f} "
            f"qb/s{Color.Reset})"
        ),
        end="",
    )
    if previous := history.get(circuits_string, None):
        perc = round(100 * duration / previous)
        color = Color.Green if perc <= 100 else Color.Red
        print(f" ~ {color}{perc}%{Color.Reset} of last run")
    else:
        print()
    print(f"\nResult:\n{parallelizer.describe(rearranged, color=False)}", color=Color.Grey)

    if show:
        import matplotlib.pyplot as plt

        parallelizer.visualization.plot_placements(rearranged)
        plt.show()

    benchmarks_run += 1
    history[circuits_string] = duration


def profile_multiple(
    circuits_string: list[str],
    backends: list,
    packer: str,
    filename: str = "parallelizer.prof",
):
    circuit_lists = [build_circuit_list(cs) for cs in circuits_string]
    packer_ = get_packer(packer)
    pr = cProfile.Profile()
    print("Profiling...")
    for circs in circuit_lists:
        pr.runcall(parallelizer.rearrange, circs, backends, packer=packer_)
    print("Done.")
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").dump_stats(filename)
    print(f"Saved results in {filename}.")


def search_for_maximum(gen_circuits_string, time_limit: float, backends, packer: str):
    global benchmarks_run
    print = init_benchmark()

    label = gen_circuits_string("N")
    print(
        (
            f"Running {Color.Cyan}'{label}'{Color.Reset} for at most "
            f"{Color.Cyan}{time_limit}{Color.Reset} seconds..."
        ),
        end="",
        flush=True,
    )

    packer_ = get_packer(packer)

    @functools.cache
    def get_duration(index):
        circuits = build_circuit_list(gen_circuits_string(index))
        start = time.time()
        parallelizer.rearrange(circuits, backends, packer=packer_)
        return time.time() - start

    def search():
        """
        Runs binary search for the last count that does not exceed the time limit. First finds an
        upper bound by doubling the count until the time limit is exceeded, then narrows down to the
        exact count.
        """
        left, right = 1, 1
        while get_duration(right) < time_limit:
            right *= 2
            print(".", end="", flush=True)
        print(f" {Color.Grey}halfway there{Color.Reset}...", end="", flush=True)
        while True:
            middle = (left + right) // 2
            if middle == left:
                return left
            duration = get_duration(middle)
            if duration >= time_limit:
                right = middle
            else:
                left = middle
            print(".", end="", flush=True)

    count = search()
    print(f"\n==> {Color.Magenta}{count} circuits{Color.Reset}", end="")
    if previous := history.get(label, None):
        perc = round(100 * count / previous)
        color = Color.Green if perc >= 100 else Color.Red
        print(f" ~ {color}{perc}%{Color.Reset} of last run")
    else:
        print()

    benchmarks_run += 1
    history[label] = count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-p", "--profile", action="store_true")
    parser.add_argument("-s", "--show", action="store_true")
    parser.add_argument("-c", "--circuits", nargs="+")
    parser.add_argument("-n", "--num-qpus", type=int, default=2)
    parser.add_argument("--packer", type=str, default="Defaults.Fast()")
    args = parser.parse_args()

    backends = fake_54qb_backend_cluster(args.num_qpus)

    if args.log:
        parallelizer.Log.set_level("debug")

    if args.profile:
        circuits = args.circuits or ["20 star", "20 h 1"]
        profile_multiple(circuits, backends, args.packer)
    else:
        load_history()
        if args.circuits:
            for circuits in args.circuits:
                run_single(circuits, backends, args.packer, show=args.show)
        else:
            print_title("Single timed runs (lower is better)")
            run_single("1000 partial 20 10", backends, args.packer)
            run_single("1000 star", backends, args.packer)
            run_single("500 partial 20 10 500 star", backends, args.packer)
            run_single("500 star 500 partial 20 10", backends, args.packer)
            run_single("1000 h 1", backends, args.packer)
            run_single("1000 ghz 2", backends, args.packer)
            print_title("Time-limited runs (higher is better)")
            search_for_maximum(lambda i: f"{i} star", 1, backends, args.packer)
            search_for_maximum(lambda i: f"{i} ghz 2", 1, backends, args.packer)
            search_for_maximum(lambda i: f"{i} h 1", 1, backends, args.packer)
        print_title(f"{benchmarks_run} benchmark(s) finished!")
        save_history()
        print(f"\n{Color.Grey}Results have been saved into {save_filename}.{Color.Reset}")
