# the input schema is extracted from names and values only; a partner schema is never required

from column_mapping.schema import extract


def test_extract_reads_type_unit_samples_nulls_and_distinct():
    rows = [
        {"claim_number": "CLM-1", "fee": "10.50", "rate": "0.11", "day": "2024-01-01"},
        {"claim_number": "CLM-2", "fee": "20.00", "rate": "0.12", "day": "2024-02-01"},
        {"claim_number": "", "fee": "30.00", "rate": "0.13", "day": "2024-03-01"},
    ]
    s = extract(rows, sample_k=3)
    assert s.rows == 3 and s.names() == ["claim_number", "fee", "rate", "day"]
    assert (
        s.column("claim_number").unit == "id"
        and s.column("claim_number").null_rate == 0.3333
    )
    assert s.column("fee").unit == "amount" and s.column("fee").type == "number"
    assert s.column("rate").unit == "rate"
    assert s.column("day").unit == "date" and s.column("day").type == "date"
    assert s.column("fee").samples == ("10.50", "20.00", "30.00")
    assert s.column("claim_number").distinct == 2
    assert s.units()["fee"] == "amount"


def test_a_small_or_empty_file_does_not_fail():
    assert extract([]).columns == ()
    one = extract([{"x": "1"}])
    assert one.rows == 1 and one.column("x").samples == ("1",)
    assert one.column("x").null_rate == 0.0
