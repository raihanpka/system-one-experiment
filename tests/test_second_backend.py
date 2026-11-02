# G1: a second backend runs the whole pipeline, records its own identity, and gets its own calibration key
import json

from columnmap.calibrate import setup_key
from columnmap.main import main
from tests.fakes import Generic


class SecondBackend(Generic):
    run_meta = {"model": "second-model", "backend": "fake", "quant": "fp16"}


def test_second_backend_runs_end_to_end_and_its_key_is_not_the_first_backends_key(tmp_path):
    root, data = tmp_path / "r", tmp_path / "d"
    main(["--samples", "ac", "--conds", "name", "--root", str(root), "--data", str(data), "--split", "dev"], decider_factory=SecondBackend)
    run = next(p for p in root.iterdir() if p.is_dir())
    meta = json.loads((run / "result.json").read_text())["meta"]
    assert meta["model"] == "second-model" and meta["backend"] == "fake" and meta["quant"] == "fp16"
    assert (run / "report" / "summary.md").exists()
    second = setup_key(meta, "options", "name", "field")
    first = setup_key({**meta, "model": "Jev-Style-2B-Decision-v3", "quant": "Q8_0"}, "options", "name", "field")
    assert second != first
