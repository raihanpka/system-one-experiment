import hashlib
import json
import re

import pytest

from columnmap.core.paths import ROOT
from columnmap.core.template import load_template
from columnmap.dataset.export import sample_folder
from columnmap.dataset.generate import generate
from columnmap.dataset.samples import DEFAULT, SAMPLES, word
from columnmap.main import main

def _no_model():
    raise AssertionError("the model must not start for a rejected sample")


CASES = {c["case_id"]: c for c in generate(load_template(), 1, 1234)}


def test_four_samples_ordered_by_difficulty():
    assert list(SAMPLES) == list("abcd") and DEFAULT == "abcd"
    assert [sample_folder(x) for x in SAMPLES] == ["a_easy_identical", "b_medium_synonym", "c_technical_terms", "d_hardest_noisy"]
    assert [SAMPLES[x][0] for x in SAMPLES] == ["easy_identical-000", "medium_synonym-000", "technical_terms-000", "hardest_noisy-000"]
    assert word("technical_terms-000") == "technical_terms"


def test_only_the_four_sample_kinds_are_generated():
    assert sorted({c["level"] for c in CASES.values()}) == [0, 3, 5, 6]
    assert set(CASES) == {SAMPLES[x][0] for x in SAMPLES}


def test_data_of_the_renamed_baseline_cases_is_identical():
    meta = json.loads((ROOT / "baselines" / "v1" / "data_hash.json").read_text())["sha256"]
    for old, new in (("coded-000", "technical_terms-000"), ("mixed-000", "hardest_noisy-000")):
        case = CASES[new]
        digest = hashlib.sha256(json.dumps({"columns": case["columns"], "rows": case["rows"]}, sort_keys=True).encode()).hexdigest()
        assert digest == meta[old]


@pytest.mark.parametrize("letters", ["e", "f", "abz"])
def test_removed_samples_are_rejected(letters, tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--samples", letters, "--root", str(tmp_path), "--data", str(tmp_path)], decider_factory=_no_model)
    assert "samples must be letters from abcd" in capsys.readouterr().err


def test_no_old_sample_names_in_active_code_or_docs():
    old = (r"\bcoded-", r"\bmixed-", r"\bstyle-000", r"\babbreviation-000", r"\bc_coded", r"\bd_mixed", r"\be_style\b", r"\bf_abbreviation", r"\bg_indonesian")
    for p in [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"), ROOT / "README.md", ROOT / "Makefile"]:
        if p.name in ("test_samples.py", "test_baseline.py", "baselines.py"):
            continue
        text = p.read_text(encoding="utf-8")
        assert not [w for w in old if re.search(w, text)], f"{p.name} still has an old sample name"
