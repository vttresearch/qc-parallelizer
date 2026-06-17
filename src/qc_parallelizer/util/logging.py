import inspect
import logging
import os
import re
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Literal, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


class ANSICodes:
    Reset = "\033[0m"

    class Weight:
        Reset = "\033[22m"
        Bold = "\033[1m"

    class FgColor:
        Reset = "\033[39m"
        Grey = "\033[38;5;245m"
        Black = "\033[90m"
        Red = "\033[91m"
        Green = "\033[92m"
        Yellow = "\033[93m"
        Blue = "\033[94m"
        Magenta = "\033[95m"
        Cyan = "\033[96m"
        White = "\033[97m"

    class BgColor:
        Reset = "\033[49m"
        Red = "\033[101m"
        Yellow = "\033[103m"
        Blue = "\033[104m"
        White = "\033[107m"


def colorize_bg(text: str, color: str):
    return f"{getattr(ANSICodes.BgColor, color.capitalize())}{text}{ANSICodes.BgColor.Reset}"


def colorize_fg(text: str, color: str):
    return f"{getattr(ANSICodes.FgColor, color.capitalize())}{text}{ANSICodes.FgColor.Reset}"


def bold(text: str):
    return f"{ANSICodes.Weight.Bold}{text}{ANSICodes.Weight.Reset}"


class Log:
    """
    Custom logger class in lieu of Python's built-in `logging` class. Adds some nice features:
    - Logged messages can be regular strings _or_ lambdas. In the latter case, they are not
      evaluated if the message is not logged.
    - Messages can be highlighted in different ways:
      - Wrapping any part in vertical bars ("like |this|") will color that part differently.
      - Wrapping an integer followed by text in singular form in dollar symbols ("like $4 apple$")
        will color that part differently _and_ append an 's' when the plural form is appropriate.
      - Adding "![label]" to the start of a message will display that label in bright colors.
    """

    class LogLevel(Enum):
        NONE = 0
        FAIL = 10
        WARN = 20
        INFO = 30
        DBUG = 40

    Message = Callable[[], str] | str

    _max_caller_length: int = 30
    _min_stack_depths: dict[threading.Thread, int] = {}
    _color_table: dict[str, dict[str, str]] = {}
    _thread_ids: dict[int, int] = {}
    level: LogLevel = LogLevel.NONE
    color: bool = True
    force_builtin: bool = False
    lock: threading.RLock = threading.RLock()

    @classmethod
    def min_stack_depth(cls, current_depth: int):
        thread = threading.current_thread()
        if thread not in cls._min_stack_depths:
            depth = 100
        else:
            depth = cls._min_stack_depths[thread]
        depth = min(depth, current_depth)
        cls._min_stack_depths[thread] = depth
        return depth

    @classmethod
    def _msgformatter(cls, match: re.Match[str]):
        if not cls.color:
            return match.group(1)
        return colorize_fg(match.group(1), "cyan")

    @classmethod
    def _msgformatter_plural(cls, match: re.Match[str]):
        result = f"{match.group(1)} {match.group(2)}"
        if abs(int(match.group(1))) != 1:
            if result.endswith("y"):
                result = result[:-1] + "ies"
            else:
                result += "s"
        if not cls.color:
            return result
        return colorize_fg(result, "magenta")

    @classmethod
    def _labelformatter(cls, match: re.Match[str]):
        if not cls.color:
            return f"[ {match.group(1)} ]"

        return colorize_fg(colorize_bg(bold(f" {match.group(1)} "), "yellow"), "black")

    @classmethod
    def _formatlevel(cls, level: LogLevel):
        if not cls.color:
            return level.name.upper()
        levelcolor = {
            cls.LogLevel.DBUG: "white",
            cls.LogLevel.INFO: "blue",
            cls.LogLevel.WARN: "yellow",
            cls.LogLevel.FAIL: "red",
        }[level]
        return colorize_fg(
            (
                f"{ANSICodes.Weight.Bold if cls.color else ''}"
                f"{level.name.upper()}"
                f"{ANSICodes.Weight.Reset if cls.color else ''}"
            ),
            levelcolor,
        )

    @classmethod
    def _color_for(cls, name: str, namespace: str):
        if namespace not in cls._color_table:
            cls._color_table[namespace] = {}
        if name not in cls._color_table[namespace]:
            avail = ["cyan", "green", "magenta", "blue", "red", "yellow"]
            cls._color_table[namespace][name] = avail[len(cls._color_table[namespace]) % len(avail)]
        return cls._color_table[namespace][name]

    @classmethod
    def _get_thread_num(cls):
        thread_id = threading.get_ident()
        if thread_id not in cls._thread_ids:
            if thread_id == threading.main_thread().ident:
                cls._thread_ids[thread_id] = 0
            else:
                cls._thread_ids[thread_id] = max(cls._thread_ids.values(), default=0) + 1
        return cls._thread_ids[thread_id]

    @classmethod
    def _formatcontext(cls, filename: str, lineno: int, stackdepth: int):
        thread_str = f"[T{cls._get_thread_num()}]"
        lineno_str = str(lineno)
        adjusted_stackdepth = stackdepth - cls.min_stack_depth(stackdepth)
        if adjusted_stackdepth == 0:
            stackdepth_str = ""
        else:
            stackdepth_str = "~" * (adjusted_stackdepth - 1) + ">"
        total_len = len(thread_str) + len(filename) + len(lineno_str) + len(stackdepth_str)
        cls._max_caller_length = max(cls._max_caller_length, total_len)
        if not cls.color:
            return (
                f"{thread_str}{stackdepth_str} {filename}:{lineno_str}"
                f"{' ' * (cls._max_caller_length - total_len)}"
            )
        return (
            f"{bold(colorize_fg(thread_str, color='grey'))}"
            f"{colorize_fg(stackdepth_str, color='grey')} "
            f"{colorize_fg(filename, cls._color_for(filename, 'filename'))}"
            f"{colorize_fg(':', 'grey')}"
            f"{lineno_str}"
            f"{' ' * (cls._max_caller_length - total_len)}"
        )

    @classmethod
    def enabled(cls, level: LogLevel | Literal["debug", "info", "warn", "fail"]):
        if isinstance(level, str):
            match level:
                case "debug":
                    level = cls.LogLevel.DBUG
                case "info":
                    level = cls.LogLevel.INFO
                case "warn":
                    level = cls.LogLevel.WARN
                case "fail":
                    level = cls.LogLevel.FAIL
        return cls.level.value <= level.value

    @classmethod
    def set_level(cls, level: Literal["debug", "info", "warn", "fail"]):
        match level:
            case "debug":
                cls.level = cls.LogLevel.DBUG
            case "info":
                cls.level = cls.LogLevel.INFO
            case "warn":
                cls.level = cls.LogLevel.WARN
            case "fail":
                cls.level = cls.LogLevel.FAIL

    @classmethod
    def debug(cls, msg: Message, **kwargs):
        cls.log(cls.LogLevel.DBUG, msg, **kwargs)

    @classmethod
    def info(cls, msg: Message, **kwargs):
        cls.log(cls.LogLevel.INFO, msg, **kwargs)

    @classmethod
    def warn(cls, msg: Message, **kwargs):
        cls.log(cls.LogLevel.WARN, msg, **kwargs)

    @classmethod
    def fail(cls, msg: Message, **kwargs):
        cls.log(cls.LogLevel.FAIL, msg, **kwargs)

    @classmethod
    def log(cls, level: LogLevel, msg: Message, strip_stack: int = 0):
        """
        Formats and logs the message to stderr. If the `force_builtin` boolean is set, sends the
        message to `logging` instead. See the class docstring for formatting guide.
        """

        with cls.lock:
            if level.value > cls.level.value:
                return

            # Convert lambda to string, if necessary
            msg_str = msg() if callable(msg) else msg
            # Colorize vertical bars
            msg_str = re.sub(r"\|([^\|]+)\|", cls._msgformatter, msg_str)
            # Colorize (and pluralize, if needed) dollar signs
            msg_str = re.sub(r"\$([^ \$]+) ([^\$]+)\$", cls._msgformatter_plural, msg_str)
            # Format labels
            msg_str = re.sub(r"^\!\[([^\]]+)\]", cls._labelformatter, msg_str)

            if cls.force_builtin:
                logger = logging.getLogger(__name__)
                logger.log(
                    {
                        cls.LogLevel.DBUG: logging.DEBUG,
                        cls.LogLevel.INFO: logging.INFO,
                        cls.LogLevel.WARN: logging.WARN,
                        cls.LogLevel.FAIL: logging.ERROR,
                    }[level],
                    msg_str,
                )
                return

            stack = inspect.stack()
            caller = inspect.getframeinfo(stack[2 + strip_stack][0])
            context_str = cls._formatcontext(
                os.path.basename(caller.filename),
                caller.lineno,
                len(stack) - 2 - strip_stack,
            )

            date_str = datetime.today().strftime("%Y-%m-%d %H:%M:%S.%f")
            level_str = cls._formatlevel(level)
            sep = colorize_fg(" | ", "grey")
            for row in msg_str.split("\n"):
                sys.stderr.write(
                    f"{date_str}{sep}{context_str}{sep}{level_str}{sep}{row}{ANSICodes.Reset}\n",
                )
            sys.stderr.flush()

    @classmethod
    def debug_dump(cls):
        """
        Dumps process ID, thread ID/name, and stack listing with files, line numbers and current
        functions names.
        """

        if cls.level.value != cls.LogLevel.DBUG.value:
            return

        def format_frame(frame):
            info = inspect.getframeinfo(frame[0])
            filename = info.filename
            for base in sys.path:
                if base and filename.startswith(base):
                    loc = base.rsplit("/", 1)[-1]
                    file = filename.removeprefix(base).strip("/")
                    if file.startswith("site-packages/"):
                        file = file.removeprefix("site-packages/")
                        loc = "site-packages"
                    filename = f"{loc}/{file}"
                    break
            return f"{filename}:{info.lineno} in |{info.function}|()"

        stack = inspect.stack()[1:][::-1]
        lines = [
            "![DEBUG DUMP]",
            (
                f"Process |{os.getpid()}|, thread |'{threading.current_thread().name}'|/"
                f"|{threading.get_ident()}|/|{threading.get_native_id()}|"
            ),
            "Stack:",
            *(f"{i:3d}: {format_frame(frame)}" for i, frame in enumerate(stack)),
        ]
        cls.debug("\n".join(lines), strip_stack=1)

    @classmethod
    def trace(cls, func: Callable[P, T]) -> Callable[P, T]:
        """
        A function decorator that logs (with debug level) when the function is entered and left.
        """

        # TODO: use TLS to store information that we are already in a traced call and that nested
        # calls should not be logged!
        # threadLocal = threading.local() etc

        def wrapped(*args, **kwargs):
            cls.debug(f"> Entering |{func.__qualname__}|!", strip_stack=1)
            ret = func(*args, **kwargs)
            cls.debug(f"< Left |{func.__qualname__}|!", strip_stack=1)
            return ret

        return wrapped
