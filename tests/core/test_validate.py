import pytest

from columnmap.core.validate import field_valid, valid_value


@pytest.mark.parametrize(
    "value, typ, ok",
    [
        ("2024-11-03", "date", True),
        ("03/11/2024", "date", False),  # not ISO
        ("2024-02-30", "date", False),  # no such day
        ("15016899.16", "number", True),
        ("nan", "number", False),
        ("abc", "number", False),
        ("POL-1", "string", True),
        ("   ", "string", False),
        (None, "string", False),  # blank cell
        ("", "date", False),
    ],
)
def test_value_checks_follow_the_template_type(value, typ, ok):
    assert valid_value(value, typ) is ok


def test_blank_cells_are_allowed_but_any_filled_cell_must_pass_the_type():
    assert field_valid(["2024-01-01", "2024-02-02"], "date")
    assert field_valid(["2024-01-01", None, ""], "date")  # source gaps stay blank
    assert not field_valid(
        ["2024-01-01", "soon"], "date"
    )  # a filled cell with the wrong type blocks the field
    assert not field_valid([None, ""], "date")  # nothing filled at all
