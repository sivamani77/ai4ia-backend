"""Unit tests for app.db.utils.normalize_value."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from app.db.utils import normalize_value


# ---------------------------------------------------------------------------
# Primitives — pass-through unchanged
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_none(self):
        assert normalize_value(None) is None

    def test_string(self):
        assert normalize_value("hello") == "hello"

    def test_integer(self):
        assert normalize_value(42) == 42

    def test_float(self):
        assert normalize_value(3.14) == 3.14

    def test_bool_true(self):
        assert normalize_value(True) is True

    def test_bool_false(self):
        assert normalize_value(False) is False


# ---------------------------------------------------------------------------
# Decimal → float
# ---------------------------------------------------------------------------

class TestDecimal:
    def test_decimal_whole(self):
        result = normalize_value(Decimal("10"))
        assert result == 10.0
        assert isinstance(result, float)

    def test_decimal_fraction(self):
        result = normalize_value(Decimal("3.14159"))
        assert isinstance(result, float)
        assert abs(result - 3.14159) < 1e-5

    def test_decimal_zero(self):
        assert normalize_value(Decimal("0")) == 0.0

    def test_decimal_negative(self):
        assert normalize_value(Decimal("-99.9")) == pytest.approx(-99.9)


# ---------------------------------------------------------------------------
# datetime / date → ISO string
# ---------------------------------------------------------------------------

class TestDatetime:
    def test_naive_datetime(self):
        dt = datetime(2026, 1, 15, 12, 30, 0)
        result = normalize_value(dt)
        assert result == "2026-01-15T12:30:00"

    def test_aware_datetime(self):
        dt = datetime(2026, 6, 25, 0, 0, 0, tzinfo=timezone.utc)
        result = normalize_value(dt)
        assert "2026-06-25" in result
        assert "00:00:00" in result

    def test_date_object(self):
        d = date(2026, 3, 10)
        result = normalize_value(d)
        assert result == "2026-03-10"


# ---------------------------------------------------------------------------
# Mapping (dict-like) → recursive normalization
# ---------------------------------------------------------------------------

class TestMapping:
    def test_plain_dict_passthrough(self):
        assert normalize_value({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    def test_dict_with_decimal(self):
        result = normalize_value({"amount": Decimal("9.99")})
        assert result == {"amount": 9.99}

    def test_dict_with_datetime(self):
        dt = datetime(2026, 1, 1, 0, 0, 0)
        result = normalize_value({"created": dt})
        assert result == {"created": "2026-01-01T00:00:00"}

    def test_nested_dict(self):
        inner = {"ts": datetime(2026, 1, 1), "val": Decimal("5")}
        result = normalize_value({"outer": inner})
        assert result == {"outer": {"ts": "2026-01-01T00:00:00", "val": 5.0}}

    def test_integer_keys_coerced_to_string(self):
        result = normalize_value({1: "one", 2: "two"})
        assert result == {"1": "one", "2": "two"}

    def test_empty_dict(self):
        assert normalize_value({}) == {}


# ---------------------------------------------------------------------------
# List / tuple → list with recursive normalization
# ---------------------------------------------------------------------------

class TestSequence:
    def test_list_passthrough(self):
        assert normalize_value([1, 2, 3]) == [1, 2, 3]

    def test_tuple_becomes_list(self):
        result = normalize_value((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_list_with_decimal(self):
        result = normalize_value([Decimal("1.5"), Decimal("2.5")])
        assert result == [1.5, 2.5]

    def test_list_with_datetime(self):
        dt = datetime(2026, 1, 1)
        result = normalize_value([dt, "string"])
        assert result == ["2026-01-01T00:00:00", "string"]

    def test_nested_list(self):
        result = normalize_value([[Decimal("1"), "x"], [2, datetime(2026, 1, 1)]])
        assert result == [[1.0, "x"], [2, "2026-01-01T00:00:00"]]

    def test_empty_list(self):
        assert normalize_value([]) == []


# ---------------------------------------------------------------------------
# Pydantic model → dict → recursively normalized
# ---------------------------------------------------------------------------

class TestPydanticModel:
    def test_model_dump_called(self):
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str
            score: float

        obj = MyModel(name="test", score=1.5)
        result = normalize_value(obj)
        assert result == {"name": "test", "score": 1.5}

    def test_model_with_nested_decimal(self):
        from pydantic import BaseModel

        class Inner(BaseModel):
            value: float

        class Outer(BaseModel):
            inner: Inner

        obj = Outer(inner=Inner(value=3.14))
        result = normalize_value(obj)
        assert isinstance(result, dict)
        # model_dump gives {"inner": {"value": 3.14}}
        assert result["inner"]["value"] == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Complex / deeply nested
# ---------------------------------------------------------------------------

class TestComplex:
    def test_deeply_nested(self):
        data = {
            "level1": {
                "level2": [
                    {"ts": datetime(2026, 1, 1), "amount": Decimal("42.0")}
                ]
            }
        }
        result = normalize_value(data)
        assert result["level1"]["level2"][0]["ts"] == "2026-01-01T00:00:00"
        assert result["level1"]["level2"][0]["amount"] == 42.0

    def test_mixed_list_and_dict(self):
        data = [
            {"date": date(2026, 6, 25)},
            [Decimal("1"), Decimal("2")],
            "raw string",
            None,
        ]
        result = normalize_value(data)
        assert result[0] == {"date": "2026-06-25"}
        assert result[1] == [1.0, 2.0]
        assert result[2] == "raw string"
        assert result[3] is None
