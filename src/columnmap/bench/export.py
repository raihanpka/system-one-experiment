# CSV files of the template and of the seven named samples: data/ is the single source of truth

from __future__ import annotations

from pathlib import Path

from ..config import DATA
from ..expressions import render
from ..knowledge import Knowledge
from ..policy import Policy
from ..prompts import DEFAULT_TEMPLATES
from .csvio import write_csv as _write
from .samples import SAMPLES, word
from .template import knowledge_from_template, load_template


def write_template(t: dict, out) -> None:
    out = Path(out)
    _write(out / "template_header.csv", [o["name"] for o in t["outputs"]], [])
    _write(
        out / "template_columns.csv",
        ["name", "type", "kind", "description", "rule"],
        [
            [
                o["name"],
                o["type"],
                o["kind"],
                o["description"],
                render(o["rule"], lambda s: s),
            ]
            for o in t["outputs"]
        ],
    )


def write_case(case: dict, d) -> None:
    d, w = Path(d), word(case["case_id"])
    names = [k["name"] for k in case["columns"]]
    _write(d / f"input_{w}.csv", names, [[r[n] for n in names] for r in case["rows"]])
    _write(
        d / f"expected_{w}.csv",
        [o["name"] for o in case["outputs"]],
        [
            [
                ""
                if not o["available"] or o["expected"][i] is None
                else o["expected"][i]
                for o in case["outputs"]
            ]
            for i in range(len(case["rows"]))
        ],
    )
    name_of = {k["src"]: k["name"] for k in case["columns"] if k["src"]}
    rows = [
        [
            o["name"],
            o["kind"],
            o["available"],
            "|".join(o["sources"]),
            render(o["rule"], name_of.get) if o["available"] else "",
        ]
        for o in case["outputs"]
    ]
    rows += [["", "decoy", False, k["name"], ""] for k in case["columns"] if k["decoy"]]
    rows += [
        [
            "",
            "anomaly",
            False,
            f"{a['column']}@row{a['row'] + 1}",
            f"{a['kind']}: {a['value']!r}",
        ]
        for a in case["anomalies"]
    ]
    _write(
        d / f"ground_truth_{w}.csv",
        ["output", "kind", "available", "input_columns", "expression"],
        rows,
    )


def sample_folder(letter: str) -> str:
    return f"{letter}_{word(SAMPLES[letter][0])}"


# data/: the template and one folder per sample (input, expected output, ground truth). Deterministic
def write_samples(
    t: dict, cases: dict, letters: str, data_dir, model: str | None = None
) -> None:
    data_dir = Path(data_dir)
    write_template(t, data_dir)
    base = knowledge_from_template(t)
    # the folder is self-contained: outputs, reference tables, prompt book, and policy
    Knowledge(
        base.outputs,
        base.tables,
        tuple(sorted(DEFAULT_TEMPLATES.items())),
        Policy(model=model),
    ).save(data_dir / "knowledge")
    for x in letters:
        write_case(cases[SAMPLES[x][0]], data_dir / "samples" / sample_folder(x))


if __name__ == "__main__":
    from ..config import load_settings
    from .generate import generate

    cfg = load_settings()
    t = load_template()
    cases = {c["case_id"]: c for c in generate(t, 1, cfg["run"]["seed"])}
    write_samples(
        t,
        cases,
        "".join(SAMPLES),
        DATA,
        model=cfg["open_router"]["model"],
    )
    print(
        f"template and {len(SAMPLES)} samples written under data/ (input, expected, ground truth per sample)"
    )
