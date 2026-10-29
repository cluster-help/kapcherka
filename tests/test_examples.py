import json
from pathlib import Path

import pytest

from kapcherka import Solver, solve

EX = Path(__file__).resolve().parent.parent / "examples"
ANSWERS = json.loads((EX / "answers.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def solver():
    return Solver()


@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_example(solver, name):
    assert solver.solve(EX / name).text == ANSWERS[name]


def test_input_kinds(solver):
    from PIL import Image
    import numpy as np

    p = EX / "digits_1.png"
    want = ANSWERS["digits_1.png"]
    assert solve(p.read_bytes()) == want
    assert solve(str(p)) == want
    assert solve(Image.open(p)) == want
    assert solve(np.asarray(Image.open(p).convert("RGB"))) == want


def test_confidence_range(solver):
    r = solver.solve(EX / "mirovye_1.jpg")
    assert 0.0 <= r.confidence <= 1.0
