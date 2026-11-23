"""P09 S4 analysis: key-point recall per speaker (natural experiment) and judge agreement.

Recall of a speaker in one set of minutes = mean over the speaker's key points (keypoints.json: human summary
sentences credited by the summlinks) of written = 1, partial = 0.5, not_written = 0 (strict: written only).
1. judge agreement: gpt-4o-mini (primary) vs claude-haiku-5-5 on the same minutes (point level: exact label,
   written-vs-not; Cohen's kappa).
2. manual-transcript minutes: recall ~ non-native + role + sex (+ speech share), meeting-clustered SEs, per model
   and prompt; and the human summary as reference (recall 1 by construction, so the gap is to the LLM itself).
3. ASR minus manual: change of a speaker's recall against their WER (slope per 10 WER points, 90% CI),
   the same estimand as null_bounds.py but on recall.
Output: results/kp_natural.json, results/kp_natural_rows.csv
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = WORKSPACE_ROOT
KP = json.load(open(f"{ROOT}/data/keypoints.json"))
M = json.load(open(f"{ROOT}/data/meetings.json"))
W = pd.read_csv(f"{ROOT}/results/speaker_wer.csv")
W["nonnative"] = (W.native != "English").astype(int)
VAL = {"written": 1.0, "partial": 0.5, "not_written": 0.0}


def load(judge):
    f = f"{ROOT}/results/kp_natural_{judge}.jsonl"
    if not os.path.exists(f):
        return None
    rows = []
    for l in open(f):
        r = json.loads(l)
        if r["labels"] is None:
            continue
        for p, lab in zip(KP[r["meeting"]], r["labels"]):
            for a in p["agents"]:
                rows.append({"model": r["model"], "prompt": r["prompt"], "cond": r["cond"], "meeting": r["meeting"], "agent": a,
                             "kid": p["kid"], "label": lab})
    return pd.DataFrame(rows)


def kappa(a, b):
    a, b = list(a), list(b)
    po = np.mean([x == y for x, y in zip(a, b)])
    cats = set(a) | set(b)
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


out = {}
A = load("gpt-4o-mini"); H = load("claude-haiku-5-5")
if A is not None and H is not None:
    J = A.merge(H, on=["model", "prompt", "cond", "meeting", "agent", "kid"], suffixes=("_mini", "_haiku")).dropna()
    J = J.drop_duplicates(["model", "prompt", "cond", "meeting", "kid"])
    out["agreement"] = {"n_points": len(J), "exact": float((J.label_mini == J.label_haiku).mean()),
                        "written_vs_not": float(((J.label_mini == "written") == (J.label_haiku == "written")).mean()),
                        "kappa": kappa(J.label_mini, J.label_haiku),
                        "share_written_mini": float((J.label_mini == "written").mean()), "share_written_haiku": float((J.label_haiku == "written").mean())}
for judge, D in (("gpt-4o-mini", A), ("claude-haiku-5-5", H)):
    if D is None:
        continue
    D = D.assign(v=D.label.map(VAL), strict=(D.label == "written").astype(float)).dropna(subset=["v"])
    R = D.groupby(["model", "prompt", "cond", "meeting", "agent"])[["v", "strict"]].mean().reset_index()
    sp = {}
    for mid, m in M.items():
        w = {}
        for s in m["segments"]:
            w[s["agent"]] = w.get(s["agent"], 0) + len(s["manual"].split())
        tot = sum(w.values())
        for a, n in w.items():
            sp[(mid, a)] = n / tot if tot else np.nan
    R["speech_share"] = [sp.get((m, a)) for m, a in zip(R.meeting, R.agent)]
    R = R.merge(W[["meeting", "agent", "role", "sex", "nonnative", "wer"]], on=["meeting", "agent"], how="left")
    R.to_csv(f"{ROOT}/results/kp_natural_rows_{judge}.csv", index=False)
    res = {}
    for (model, prompt), g in R[R.cond == "manual"].groupby(["model", "prompt"]):
        g = g.dropna(subset=["v", "role", "sex"])
        f = smf.ols("v ~ nonnative + C(role, Treatment('ME')) + C(sex) + speech_share", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.meeting)[0]})
        res[f"{model}|{prompt}|manual"] = {"n": len(g), "mean_recall": float(g.v.mean()), "mean_strict": float(g.strict.mean()),
                                           **{k: [float(f.params[k]), float(f.bse[k]), float(f.pvalues[k])] for k in f.params.index if k != "Intercept"}}
    P = R.pivot_table(index=["model", "prompt", "meeting", "agent"], columns="cond", values="v").reset_index()
    if {"manual", "asr"} <= set(P.columns):
        P = P.dropna(subset=["manual", "asr"]).assign(d=lambda x: x.asr - x.manual)
        P = P.merge(W[["meeting", "agent", "wer", "nonnative"]], on=["meeting", "agent"])
        P["wer10"] = P.wer * 10
        for (model, prompt), g in P.groupby(["model", "prompt"]):
            f = smf.ols("d ~ wer10", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.meeting)[0]})
            b, se = float(f.params.wer10), float(f.bse.wer10)
            res[f"{model}|{prompt}|asr_minus_manual"] = {"n": len(g), "mean_change": float(g.d.mean()), "slope_per10": b,
                                                         "ci90": [b - 1.645 * se, b + 1.645 * se], "p": float(f.pvalues.wer10),
                                                         "nonnative_minus_native": float(g[g.nonnative == 1].d.mean() - g[g.nonnative == 0].d.mean())}
    out[judge] = res
json.dump(out, open(f"{ROOT}/results/kp_natural.json", "w"), indent=1, default=float)
print(json.dumps(out.get("agreement"), indent=1))
for judge in ("gpt-4o-mini", "claude-haiku-5-5"):
    for k, v in (out.get(judge) or {}).items():
        short = {x: (np.round(y, 3) if isinstance(y, float) else [round(z, 3) for z in y] if isinstance(y, list) else y) for x, y in v.items()}
        print(judge, k, short)
