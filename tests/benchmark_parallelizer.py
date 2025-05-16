"""
Little script for benchmarking the parallelizer. Runs a fixed set of benchmarks, first some fixed-
size tests with a timer and then some parametrized tests until a time limit is reached.

Results are saved into /tmp (or equivalent, whatever tempfile provides) for comparison with later
results. If earlier results are present, they are loaded and automatically compared.

The code might not be the most readable kind but it works and produces pretty output :)
"""

import functools
import json
import os
import shutil
import sys
import tempfile
import time

from qc_parallelizer import parallelizer
from utils import build_circuit_list, fake_20qb_backend

# Enable/disable ANSI colors depending on output type (or user preference)
if os.isatty(0) and "--no-color" not in sys.argv:

    class Color:
        enabled = True
        Reset = "\033[0m"
        Grey = "\033[38;5;245m"
        Red = "\033[91m"
        Green = "\033[92m"
        Magenta = "\033[95m"
        Cyan = "\033[96m"
        BgBlue = "\033[104m\033[30m"
        BgWhite = "\033[107m\033[30m"

        @staticmethod
        def BgGrey(level: float):
            return f"\033[48;5;{232 + level * 23:.0f}m\033[30m"
else:

    class DummyNoColorMetaclass(type):
        def __getattr__(self, attr):
            # No colors for you
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
                print(line, **kwargs)
            else:
                print(line)
        next_on_new_line = "end" not in kwargs

    return prefix_print


def run_single(circuits_string: str):
    global benchmarks_run
    print = init_benchmark()

    print(f"Generating {Color.Cyan}'{circuits_string}'{Color.Reset}... ", end="", flush=True)
    circuits = build_circuit_list(circuits_string)
    print(f"rearranging {Color.Cyan}{len(circuits)} circuits{Color.Reset}...")

    start = time.time()
    rearranged = parallelizer.rearrange(circuits, fake_20qb_backend)
    duration = time.time() - start

    print(f"==> {Color.Magenta}{duration:.2f} seconds{Color.Reset}", end="")
    if previous := history.get(circuits_string, None):
        perc = round(100 * duration / previous)
        color = Color.Green if perc <= 100 else Color.Red
        print(f" ~ {color}{perc}%{Color.Reset} of last run")
    else:
        print()
    print(f"\nResult:\n{parallelizer.describe(rearranged, color=False)}", color=Color.Grey)

    benchmarks_run += 1
    history[circuits_string] = duration


def search_for_maximum(gen_circuits_string, time_limit: float):
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

    @functools.cache
    def get_duration(index):
        circuits = build_circuit_list(gen_circuits_string(index))
        start = time.time()
        parallelizer.rearrange(circuits, fake_20qb_backend)
        return time.time() - start

    def search():
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
    load_history()
    print_title("Single timed runs (lower is better)")
    run_single("10 partial 20 10")
    run_single("10 star")
    run_single("5 partial 20 10 5 star")
    run_single("5 star 5 partial 20 10")
    run_single("10 h 1")
    run_single("10 ghz 2")
    print_title("Time-limited runs (higher is better)")
    search_for_maximum(lambda i: f"{i} star", 1.0)
    search_for_maximum(lambda i: f"{i} ghz 2", 1.0)
    search_for_maximum(lambda i: f"{i} h 1", 1.0)
    print_title(f"{benchmarks_run} benchmarks finished!")
    save_history()
    print(f"\n{Color.Grey}Results have been saved into {save_filename}.{Color.Reset}")
