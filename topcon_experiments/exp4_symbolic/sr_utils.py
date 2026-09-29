"""Shared PySR configuration for fast validation vs full runs."""

from __future__ import annotations

from pathlib import Path

from pysr import PySRRegressor

from topcon_experiments.config import (
    OUTPUT_ROOT,
    SR_BATCHING,
    SR_MAXSIZE,
    SR_MODEL_SELECTION,
    SR_NITERATIONS,
    SR_PARSIMONY,
    SR_POPULATIONS,
    SR_POPULATION_SIZE,
)

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"

# Extended basic operators; inv uses Julia definition + sympy mapping
SR_BINARY_OPERATORS = ["+", "-", "*", "/"]
SR_UNARY_OPERATORS = [
    "exp",
    "log",
    "sqrt",
    "square",
    "abs",
    "inv(x) = 1/x",
]
SR_EXTRA_SYMPY_MAPPINGS = {"inv": lambda x: 1/x}
SR_NESTED_CONSTRAINTS = {
    "log": {"log": 0, "exp": 0},
    "exp": {"exp": 0},
    "sqrt": {"sqrt": 0},
    "square": {"square": 1},
    "inv": {"inv": 0},
    "abs": {"abs": 1},
}


def build_pysr_regressor(checkpoint_name: str, *, maxsize: int | None = None) -> PySRRegressor:
    sr_out = EXP4_OUT / "pysr_checkpoints" / checkpoint_name
    sr_out.mkdir(parents=True, exist_ok=True)
    return PySRRegressor(
        populations=SR_POPULATIONS,
        population_size=SR_POPULATION_SIZE,
        binary_operators=SR_BINARY_OPERATORS,
        unary_operators=SR_UNARY_OPERATORS,
        extra_sympy_mappings=SR_EXTRA_SYMPY_MAPPINGS,
        nested_constraints=SR_NESTED_CONSTRAINTS,
        maxsize=maxsize or SR_MAXSIZE,
        model_selection=SR_MODEL_SELECTION,
        parsimony=SR_PARSIMONY,
        niterations=SR_NITERATIONS,
        batching=SR_BATCHING,
        verbosity=1,
        progress=True,
        temp_equation_file=str(sr_out / "equations.csv"),
        tempdir=str(sr_out),
        delete_tempfiles=True,
    )
