import json

from column_mapping.bench.store import Store


def test_roundtrip_resume_rounding_and_readable_layout(tmp_path):
    p = tmp_path / "result.json"
    s = Store(p, meta={"model": "m"})
    assert not s.has("mapping", "easy_identical-000")
    rec = {
        "case_id": "easy_identical-000",
        "level": 0,
        "matrix": {
            "colA": {"out1": -1.987654321, "out2": 2.0},
            "colB": {"out1": -2.0, "out2": -1.0},
        },
    }
    s.add("mapping", rec)

    data = json.loads(p.read_text())  # plain JSON, valid after every add
    assert data["meta"] == {"model": "m"}
    assert data["mapping"][0]["matrix"]["colA"]["out1"] == -1.9877

    text = p.read_text()
    assert (
        text.count("\n") > 8 and '"colA": {"out1"' in text
    )  # one input column per line
    assert len([l for l in text.splitlines() if l.strip().startswith('"colA"')]) == 1

    again = Store(p)  # new process: resume
    assert again.has("mapping", "easy_identical-000") and not again.has(
        "mapping", "zzz"
    )
    assert (
        again.rows("mapping", only={"easy_identical-000"})
        and again.rows("mapping", only={"zzz"}) == []
    )
    again.add("mapping", {**rec, "case_id": "medium_synonym-000"})
    assert [r["case_id"] for r in Store(p).rows("mapping")] == [
        "easy_identical-000",
        "medium_synonym-000",
    ]
    assert Store(p).data["meta"] == {"model": "m"}
