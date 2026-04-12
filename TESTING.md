# Automated Testing Guide

This document defines the conventions the entire team must follow when writing automated tests for the project.

---

## File structure

Each branch must have its own `tests/` folder at the root of the repository:

```
tests/
├── conftest.py
├── __init__.py
└── unit/
    ├── __init__.py
    ├── test_<module1>.py
    └── test_<module2>.py
```

- One test file per module being tested.
- The file name always starts with `test_` followed by the module name.

---

## Naming conventions

### Classes

Group tests for the same function or method in a class that starts with `Test`:

```python
class TestFunctionName:
    def test_happy_path_case(self):
        ...

    def test_alternative_case(self):
        ...
```

### Test functions

The name must describe exactly what is being verified:

```python
# Good
def test_returns_zero_for_empty_text(self):
def test_removes_script_tags(self):
def test_preserves_lines_that_dont_match(self):

# Bad
def test_1(self):
def test_function(self):
def test_ok(self):
```

---

## Minimum required per functionality

Each implemented function or method must have **at least 2 tests**:

| Test | What it verifies |
|------|-----------------|
| Happy path | Valid and normal input -> correct result |
| Alternative flow | Empty, invalid, or edge-case input -> does not crash or returns the expected value |

```python
class TestClassification:
    def test_clean_text_is_high(self):  # happy path
        assert classify_score(90) == "HIGH"

    def test_empty_text_does_not_raise(self):  # alternative flow
        result = compute_quality_score("")
        assert isinstance(result["quality_score"], int)
```

---

## Internal structure of a test

Follow the **Arrange -> Act -> Assert** pattern:

```python
def test_removes_source_line(self):
    # Arrange: prepare the input
    body = "Congreso de la República\nARTÍCULO 1. Contenido real."

    # Act: call the function
    result = remove_metadata_lines(body, source="Congreso de la República", subtipo="")

    # Assert: verify the result
    assert "Congreso de la República" not in result
    assert "ARTÍCULO 1. Contenido real." in result
```

---

## Using fixtures

If several tests in the same file need the same object, use a pytest fixture instead of repeating it:

```python
import pytest

@pytest.fixture
def cleaner():
    return DocumentCleaner()

class TestDocumentCleaner:
    def test_something(self, cleaner):
        result = cleaner.clean_text("texto")
        assert result is not None
```

---

## Using parametrize

When the same test applies to multiple values, use `@pytest.mark.parametrize` instead of copying the test:

```python
@pytest.mark.parametrize("score, expected", [
    (90, "HIGH"),
    (75, "MEDIUM"),
    (60, "LOW"),
    (30, "DEFECTIVE"),
])
def test_classify_score(self, score, expected):
    assert classify_score(score) == expected
```

---

## How to run the tests

From the project root with the virtual environment activated:

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single file
python -m pytest tests/unit/test_quality_metrics.py -v

# View code coverage
python -m pytest tests/ --cov --cov-report=term-missing
```

---

## What NOT to do

- Do not test functions that only do pure I/O (reading/writing files, real HTTP requests).
- Do not write tests that depend on local files on your machine.
- Do not leave tests with generic names like `test_1` or `test_ok`.
- Do not put complex logic inside a test; if the arrange section is too long, use a fixture.
