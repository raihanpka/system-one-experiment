from columnmap.core.candidates import candidates
from columnmap.core.dsl import safe_evaluate
from columnmap.core.reference import covering, load_tables
from columnmap.core.template import load_template
from columnmap.dataset.export import write_reference

TABLE = {"A": "Approved", "P": "Pending", "R": "Rejected"}


def test_a_table_covers_a_column_only_when_every_filled_value_is_a_code():
    assert covering(["A", "P", "", "R"], [TABLE]) == [TABLE]
    assert covering(["A", "X"], [TABLE]) == []
    assert covering(["", " "], [TABLE]) == []


def test_a_covered_code_column_gets_a_lookup_candidate_that_translates_the_codes():
    cands = candidates([("status", ["A", "P", "R"])], tables=[TABLE])
    lookup = next(c for c in cands if c.get("op") == "lookup")
    assert [safe_evaluate(lookup, {"status": v}) for v in "APR"] == [
        "Approved",
        "Pending",
        "Rejected",
    ]
    assert not any(
        c.get("op") == "lookup"
        for c in candidates([("status", ["A", "X"])], tables=[TABLE])
    )


def test_export_writes_the_rule_map_as_a_reference_table_the_loader_reads_back(
    tmp_path,
):
    t = load_template()
    write_reference(t, tmp_path)
    tables = load_tables(tmp_path / "reference")
    assert TABLE in tables
