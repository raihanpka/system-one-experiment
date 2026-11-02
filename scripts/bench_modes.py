# Compare the two mapping modes on the model: wall time and per-column top-1 (about 5 minutes)

import sys, time

from columnmap.core.paths import VENDOR

sys.path.insert(0, str(VENDOR))
from jev_style_decision_gguf import JevStyleDecisionGGUF
from columnmap.core.prompts import input_state, samples
from columnmap.core.template import load_template
from columnmap.dataset.generate import generate

T = load_template()
desc = {s["id"]: s["desc"] for s in T["sources"] + T["decoys"]}
cases = {c["case_id"]: c for c in generate(T, 1, 1234)}
m = JevStyleDecisionGGUF(model_dir=VENDOR, quant="Q8_0")

def run(case_id, cond):
    c = cases[case_id]; outs = [o["name"] for o in c["outputs"]]
    truth = {k["name"]: {o["name"] for o in c["outputs"] if o["available"] and k["name"] in o["sources"]} for k in c["columns"]}
    res = {}
    for mode in ("pairs", "options"):
        t0 = time.time(); hit = n = 0
        for k in c["columns"]:
            st = input_state(k, cond, None, samples(c["rows"], k["name"]) if cond == "samples" else ())
            if mode == "pairs":
                qs = [{"t": "noul", "ins": f"This input column provides the data for the output field '{o}'.", "crit": None} for o in outs]
                sc = [r["scores"]["true"] - r["scores"]["false"] for r in m.score_many(st, qs)]
                sc = dict(zip(outs, sc))
            else:
                r = m.decide(st, "Which output fields does this input column provide data for?", options={o: None for o in outs}, qtype="choice")
                sc = r["scores"]
            if truth[k["name"]]:
                n += 1; hit += max(sc, key=sc.get) in truth[k["name"]]
        res[mode] = (round(time.time() - t0, 1), hit, n)
    return res

for cid, cond in (("coded-000", "name"), ("mixed-000", "samples")):
    r = run(cid, cond)
    print(f"{cid} ({cond}): pairs {r['pairs'][0]}s top1 {r['pairs'][1]}/{r['pairs'][2]} | options {r['options'][0]}s top1 {r['options'][1]}/{r['options'][2]}", flush=True)
m.close()
