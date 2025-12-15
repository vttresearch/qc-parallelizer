from .base import PackerBase
from . import vf2, smt

class Defaults:
    Optimizing = vf2.Minimizing
    Fast = smt.NonOptimizing

__all__ = (
    "Defaults",
    "vf2",
    "smt",
    "PackerBase",
)
