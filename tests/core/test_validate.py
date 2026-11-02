import pytest

from columnmap.core.validate import field_valid, valid_value


@pytest.mark.parametrize("value, typ, ok", [
    ("2024-11-03", "date", True),
    ("03/11/2024", "date", False),                     # not ISO
    ("2024-02-30", "date", False),                     # no such day
    ("15016899.16", "number", True),
    ("nan", "number", False),
    ("abc", "number", False),
    ("POL-1", "string", True),
    ("   ", "string", False),
    (None, "string", False),                           # blank cell
    ("", "date", False),
])
def test_value_checks_follow_the_template_type(value, typ, ok):
    assert valid_value(value, typ) is ok


def test_a_field_is_valid_only_when_every_cell_passes():
    assert field_valid(["2024-01-01", "2024-02-02"], "date")
    assert not field_valid(["2024-01-01", None], "date")
    assert not field_valid(["2024-01-01", "soon"], "date")
