import json

import yaml

from columnmap.core.paths import ROOT


def load():
    return yaml.safe_load(
        (ROOT / "ontology" / "insurance.yaml").read_text(encoding="utf-8")
    )


def test_every_output_field_has_a_kind_a_type_and_a_description():
    fields = load()["fields"]
    assert len(fields) == len(
        json.loads((ROOT / "data" / "template.json").read_text())["outputs"]
    )
    assert all(
        f["kind"] and f["type"] in ("date", "number", "string") and f["description"]
        for f in fields
    )


def test_relations_point_only_to_declared_source_concepts():
    d = load()
    sources = {s["name"] for s in d["sources"]}
    assert all(set(f["derived_from"]) <= sources for f in d["fields"])


def test_synthetic_column_names_are_not_in_the_ontology():
    # the generator's per-level column names are the answer key of the samples; they must never reach the ontology
    text = (ROOT / "ontology" / "insurance.yaml").read_text(encoding="utf-8")
    for leaked in ("pol_no", "contract_ref", "nomor_polis", "COV_UNIQ_KEY"):
        assert leaked not in text
