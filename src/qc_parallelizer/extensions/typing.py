from collections.abc import Mapping, Sequence
from types import UnionType
from typing import Any, TypeVar, cast, get_args, get_origin


def isnestedinstance(obj: Any, typ: type | UnionType):
    """
    Like `isinstance()`, but supports nested and parametrized types. For example,
    ```
    >>> isnestedinstance({"foo": {"bar", 1.5}, "baz": {4.0}}, dict[str, set[str | float]])
    True
    >>> isnestedinstance({"foo": {1, 2, 3, "bar"}}, dict[str, set[int]])
    False
    ```

    **Does not cover all types**. Raises a `TypeError` if an unrecognized type is detected.
    """

    origin = get_origin(typ)
    if origin is None:
        return isinstance(obj, typ)

    if origin is UnionType:
        subtyps = get_args(typ)
        return any(isnestedinstance(obj, subtyp) for subtyp in subtyps)

    if origin is tuple:
        if not isinstance(obj, tuple):
            return False
        args = get_args(typ)
        if len(args) == 2 and args[1] == Ellipsis:
            return all(isnestedinstance(item, args[0]) for item in obj)
        elif Ellipsis in args:
            raise TypeError(f"unsupported use of Ellipsis in tuple type {args}")
        else:
            return all(isnestedinstance(item, subtyp) for item, subtyp in zip(obj, args))
    if origin in [list, set, Sequence]:
        if not isinstance(obj, Sequence | set):
            return False
        return all(isnestedinstance(item, get_args(typ)[0]) for item in obj)

    if origin in [dict, Mapping]:
        if not isinstance(obj, Mapping):
            return False
        keytyp, valuetyp = get_args(typ)
        return all(isnestedinstance(key, keytyp) for key in obj.keys()) and all(
            isnestedinstance(value, valuetyp) for value in obj.values()
        )

    raise TypeError("unsupported type")


def typestr(obj: Any) -> str:
    """
    Returns a textual description of given object's type. For example,
    ```
    >>> typestr({0: "foo", "bar": "baz"})
    'dict[int | str, str]'
    ```

    In other words, the string returned for a given object should be a valid type hint for that
    object.
    """

    if isinstance(obj, str):
        return str.__name__
    if isinstance(obj, tuple):
        types = [typestr(item) for item in obj]
        return f"{type(obj).__name__}[{', '.join(types)}]"
    if isinstance(obj, Sequence | set):
        types = {typestr(item) for item in obj}
        return f"{type(obj).__name__}[{' | '.join(types)}]"
    if isinstance(obj, Mapping):
        keytypes = {typestr(key) for key in obj.keys()}
        valuetypes = {typestr(key) for key in obj.values()}
        return f"{type(obj).__name__}[{' | '.join(keytypes)}, {' | '.join(valuetypes)}]"
    return type(obj).__name__


T = TypeVar("T")


def ensure_sequence(obj: T | Sequence[T], typ: type | UnionType) -> Sequence[T]:
    if isnestedinstance(obj, typ):
        return [obj]  # type: ignore
    elif isnestedinstance(obj, Sequence[typ]):
        return obj  # type: ignore
    raise TypeError(f"expected '{typ}' or a sequence thereof, got '{typestr(obj)}'")
