import json

from columnmap.core.store import Store


def test_roundtrip_resume_rounding_and_readable_layout(tmp_path):
    p = tmp_path / "result.json"
    s = Store(p, meta={"model": "m"})
    assert not s.has("mapping", "name", "identical-000")
    rec = {"case_id": "identical-000", "level": 0, "cond": "name",
           "matrix": {"colA": {"out1": {"p": 0.123456789, "d": -1.987654321}, "out2": {"p": 0.9, "d": 2.0}},
                      "colB": {"out1": {"p": 0.1, "d": -2.0}, "out2": {"p": 0.2, "d": -1.0}}},
           "kind": {"out1": {"choice": "concat", "p": {"concat": 0.5, "lookup": 0.5}}}}
    s.add("mapping", "name", rec)

    data = json.loads(p.read_text())                              # plain JSON, valid after every add
    assert data["meta"] == {"model": "m"}
    assert data["mapping"]["name"][0]["matrix"]["colA"]["out1"] == {"p": 0.1235, "d": -1.9877}

    text = p.read_text()
    assert text.count("\n") > 15 and '"colA": {"out1"' in text      # structure on separate lines, one input column per line
    assert len([l for l in text.splitlines() if l.strip().startswith('"colA"')]) == 1

    again = Store(p)                                               # new process: resume
    assert again.has("mapping", "name", "identical-000") and not again.has("mapping", "samples", "identical-000")
    assert again.rows("mapping", "name", only={"identical-000"}) and again.rows("mapping", "name", only={"zzz"}) == []
    again.add("mapping", "name", {**rec, "case_id": "synonym-000"})
    assert [r["case_id"] for r in Store(p).rows("mapping", "name")] == ["identical-000", "synonym-000"]
    assert Store(p).data["meta"] == {"model": "m"}
