"""Tests de parse_variant (mapeo nombre-de-variante -> prompt + guardrail)."""

import sys
from pathlib import Path

import pytest

_JUDGE_DIR = Path(__file__).resolve().parents[2] / "04_distillation" / "judge_calibration"
if str(_JUDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_JUDGE_DIR))

from calibrate_judge import parse_variant  # noqa: E402


@pytest.mark.parametrize("name,expected", [
    ("v2", ("v2", False)),
    ("v4", ("v4", False)),
    ("v1", ("v1", False)),
    ("v3", ("v3", False)),
    ("v2g", ("v2", True)),
    ("v4g", ("v4", True)),
    ("v3g", ("v3", True)),
])
def test_parse_variant(name, expected):
    assert parse_variant(name) == expected


@pytest.mark.parametrize("bad", ["v9", "v9g", "x", "gg", "vg"])
def test_parse_variant_rejects_unknown(bad):
    with pytest.raises(ValueError):
        parse_variant(bad)
