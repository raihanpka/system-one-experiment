# a second backend runs the whole pipeline and the run records its identity
import json

from column_mapping.bench.cli import main
from tests.bench.fakes import Generic


class SecondBackend(Generic):
    run_meta = {"model": "second-model", "backend": "fake", "quant": "fp16"}


def test_second_backend_runs_end_to_end_and_records_its_own_identity(tmp_path):
    root, data = tmp_path / "r", tmp_path / "d"
    main(
        [
            "--samples",
            "ac",
            "--root",
            str(root),
            "--data",
            str(data),
            "--split",
            "dev",
        ],
        decider_factory=SecondBackend,
    )
    run = next(p for p in root.iterdir() if p.is_dir())
    meta = json.loads((run / "result.json").read_text())["meta"]
    assert meta["model"] == "second-model" and meta["backend"] == "fake"
    summary = (run / "report" / "summary.md").read_text()
    assert "second-model" in summary
