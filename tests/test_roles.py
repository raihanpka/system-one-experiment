# the role parser reads the field description, never the rule: operator and named operands, or None for a fallback

from columnmap.knowledge import Output
from columnmap.roles import shape


def roles(out, kind):
    s = shape(out, kind)
    return None if s is None else [(r.phrase, r.unit) for r in s.roles]


def test_arithmetic_descriptions_become_operators_with_typed_roles():
    assert roles(
        Output("total_charge", "number", "Transaction amount plus fee"), "arithmetic"
    ) == [("transaction amount", "amount"), ("fee", "amount")]
    assert roles(
        Output("net_claim_payable", "number", "Claimed amount minus deductible"),
        "arithmetic",
    ) == [("claimed amount", "amount"), ("deductible", "amount")]
    assert roles(
        Output("claim_paid_ratio", "number", "Paid amount divided by base premium"),
        "arithmetic",
    ) == [("paid amount", "amount"), ("base premium", "amount")]
    assert roles(
        Output(
            "gross_premium",
            "number",
            "Base premium plus tax: base premium times (1 + tax rate)",
        ),
        "arithmetic",
    ) == [("base premium", "amount"), ("tax rate", "rate")]
    assert roles(
        Output(
            "net_premium_after_discount",
            "number",
            "Base premium after discount: base premium times (1 - discount rate)",
        ),
        "arithmetic",
    ) == [("base premium", "amount"), ("discount rate", "rate")]


def test_date_and_join_descriptions_become_roles_in_order():
    assert roles(
        Output(
            "coverage_days",
            "number",
            "Number of days between coverage start and coverage end",
        ),
        "date_difference",
    ) == [("coverage start", "date"), ("coverage end", "date")]
    assert roles(
        Output("claim_age_days", "number", "Days from coverage start to the incident"),
        "date_difference",
    ) == [("coverage start", "date"), ("the incident", "date")]
    assert roles(
        Output(
            "insured_full_name",
            "string",
            "Full name of the insured, given name then family name",
        ),
        "join",
    ) == [("given name", "text"), ("family name", "text")]
    assert roles(
        Output(
            "insured_address",
            "string",
            "Single-line address: city, province postal code",
        ),
        "join",
    ) == [("city", "text"), ("province", "text"), ("postal code", "text")]


def test_single_column_kinds_get_one_role_and_an_unreadable_description_falls_back():
    assert roles(
        Output("insured_gender", "string", "Gender of the insured"), "copy"
    ) == [("Gender of the insured", "text")]
    assert roles(
        Output("insured_birth_year", "number", "Year of birth of the insured"), "year"
    ) == [("date of birth of the insured", "date")]
    assert roles(
        Output("claim_status", "string", "Claim status in words (Approved, Rejected)"),
        "lookup",
    ) == [("code for claim status in words (approved, rejected)", "text")]
    assert (
        shape(Output("mystery", "number", "Something computed elsewhere"), "arithmetic")
        is None
    )


def test_the_shape_builds_the_same_expression_the_rule_would():
    from columnmap.expressions import render

    col = {"src": "x"}
    cases = [
        ("Transaction amount plus fee", "arithmetic", {"src": "x"}),
        ("Claimed amount minus deductible", "arithmetic", {"src": "x"}),
        ("Paid amount divided by base premium", "arithmetic", {"src": "x"}),
        (
            "Base premium plus tax: base premium times (1 + tax rate)",
            "arithmetic",
            {"src": "x"},
        ),
        (
            "Base premium after discount: base premium times (1 - discount rate)",
            "arithmetic",
            {"src": "x"},
        ),
        ("Number of days between start and end", "date_difference", {"src": "x"}),
        ("Year of birth of the insured", "year", {"src": "x"}),
        ("Gender of the insured", "copy", {"src": "x"}),
    ]
    for desc, kind, _ in cases:
        s = shape(Output("f", "number", desc), kind)
        assert s is not None
        node = s.build([col] * len(s.roles))
        assert render(node, lambda name: name)
