import inspect
import logging
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Literal


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
    _min_stack_depth: int = 100
    _color_table: dict[str, dict[str, str]] = {}
    level: LogLevel = LogLevel.NONE
    color: bool = True
    force_builtin: bool = False

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
            existing = set(cls._color_table[namespace].values())
            cls._color_table[namespace][name] = next(
                color
                for color in ["cyan", "green", "magenta", "blue", "red", "yellow"]
                if color not in existing
            )
        return cls._color_table[namespace][name]

    @classmethod
    def _formatcontext(cls, filename: str, lineno: int, stackdepth: int):
        lineno_str = str(lineno)
        cls._min_stack_depth = min(cls._min_stack_depth, stackdepth)
        adjusted_stackdepth = stackdepth - cls._min_stack_depth
        if adjusted_stackdepth == 0:
            stackdepth_str = ""
        elif adjusted_stackdepth == 1:
            stackdepth_str = "> "
        else:
            stackdepth_str = "~" * (adjusted_stackdepth - 1) + "> "
        total_len = len(filename) + len(lineno_str) + len(stackdepth_str)
        cls._max_caller_length = max(cls._max_caller_length, total_len)
        if not cls.color:
            return (
                f"{stackdepth_str}{filename}:{lineno_str}"
                f"{' ' * (cls._max_caller_length - total_len)}"
            )
        return (
            f"{colorize_fg(stackdepth_str, color='grey')}"
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
    def debug(cls, msg: Message):
        cls.log(cls.LogLevel.DBUG, msg)

    @classmethod
    def info(cls, msg: Message):
        cls.log(cls.LogLevel.INFO, msg)

    @classmethod
    def warn(cls, msg: Message):
        cls.log(cls.LogLevel.WARN, msg)

    @classmethod
    def fail(cls, msg: Message):
        cls.log(cls.LogLevel.FAIL, msg)

    @classmethod
    def log(cls, level: LogLevel, msg: Message):
        """
        Formats and logs the message to stderr. If the `force_builtin` boolean is set, sends the
        message to `logging` instead. See the class docstring for formatting guide.
        """

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
        caller = inspect.getframeinfo(stack[2][0])
        context_str = cls._formatcontext(
            os.path.basename(caller.filename),
            caller.lineno,
            len(stack),
        )

        date_str = datetime.today().strftime("%Y-%m-%d %H:%M:%S.%f")
        level_str = cls._formatlevel(level)
        sep = colorize_fg(" | ", "grey")
        for row in msg_str.split("\n"):
            sys.stderr.write(
                f"{date_str}{sep}{context_str}{sep}{level_str}{sep}{row}{ANSICodes.Reset}\n",
            )
