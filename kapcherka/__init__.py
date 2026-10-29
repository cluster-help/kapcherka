"""Капчерка — распознавание капчи сайтов судов РФ."""
from .solver import Result, Solver, default_solver, preprocess, solve

__version__ = "1.0.0"
__all__ = ["Result", "Solver", "default_solver", "preprocess", "solve", "__version__"]
