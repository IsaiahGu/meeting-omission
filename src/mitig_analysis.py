"""P09 S6 / S9 analysis: mitigations, and simulated diarisation errors.

For each minutes text: the target's embedding-attribution share (as in dense_analysis.py), key-point recall
(keypoints.py, gpt-4o-mini judge), whether the minutes say the speaker was hard to make out, word count and
number of calls; S8 labels (fabric.py) when present. Everything relative to the same model's minutes of the
manual transcript.
  S6: per (model, condition, method M0-M4): change in share and recall vs manual, share of the M0 loss recovered,
      "unclear" mentions, ungrounded / misattributed rate for the target.
  S9: per (model, condition): change in share and recall; association with the speaker's speech share, the share of
      their segments next to a turn change, and native language.
python mitig_analysis.py mitig|diar   Output: results/mitig_effects.csv | results/diar_effects.csv (+ _rows.csv)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from embed import encode  # noqa: E402

ROOT = WORKSPACE_ROOT
M = json.load(open(f"{ROOT}/data/meetings.json"))
T = json.load(open(f"{ROOT}/data/noise_targets.json"))
KP = json.load(open(f"{ROOT}/data/keypoints.json"))
TASK = "Given a sentence from meeting minutes, retrieve the meeting utterance that supports it"
UNCLEAR = re.compile(r"unclear|inaudible|hard to (?:hear|make out|follow|understand)|could not be (?:heard|understood|made out)|"
                     r"garbled|difficult to (?:hear|understand|follow)|not (?:clearly )?audible|poor audio|audio quality", re.I)
rng = np.random.default_rng(0)
which = sys.argv[1]


def split_sents(t):
    parts = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"[*#]+", " ", t))
    return [p.strip(" -•\t") for p in parts if len(p.strip(" -•\t").split()) >= 4]


texts = {}
for f in glob.glob(f"{ROOT}/results/{which}_summaries_*.jsonl"):
    model = f.split("_summaries_")[1][:-6]
    for l in open(f):
        r = json.loads(l)
        if not r.get("error") and r["text"].strip():
            texts[(model, r["meeting"], r["cond"], r["prompt"])] = r["text"]
# references: plain minutes of the manual transcript and (for S6) of the degraded transcript (M0)
for f in glob.glob(f"{ROOT}/results/dense_summaries_*.jsonl"):
    model = f.split("dense_summaries_")[1][:-6]
    for l in open(f):
        r = json.loads(l)
        if not r.get("error") and r["text"].strip() and (r["cond"] == "manual" or which == "mitig"):
            texts.setdefault((model, r["meeting"], r["cond"], "M0" if r["cond"] != "manual" else "minutes"), r["text"])
for model in {k[0] for k in texts}:
    fn = f"{ROOT}/results/summaries_{model}.jsonl"
    if os.path.exists(fn):
        for l in open(fn):
            r = json.loads(l)
            if r["cond"] == "manual" and r["prompt"] == "minutes" and r["text"].strip():
                texts.setdefault((model, r["meeting"], "manual", "minutes"), r["text"])
keys = list(texts)
sents = [split_sents(texts[k]) for k in keys]
Q = encode("qwen3-emb", [x for ss in sents for x in ss], "query", qwen_task=TASK, bs=64).astype(np.float32)
utt = {}
for mid in T:
    segs = [s for s in M[mid]["segments"] if len(s["manual"].split()) >= 4]
    utt[mid] = (np.array([s["agent"] for s in segs]), encode("qwen3-emb", [s["manual"] for s in segs], "doc", cache_key=f"p09_utt_{mid}").astype(np.float32))
share, pos = {}, 0
for k, ss in zip(keys, sents):
    agents, U = utt[k[1]]
    q = Q[pos:pos + len(ss)]; pos += len(ss)
    sh = {a: 0.0 for a in M[k[1]]["speakers"]}
    if len(ss):
        sim = q @ U.T
        for i in range(len(ss)):
            top = np.argsort(-sim[i])[:3]
            w = np.maximum(sim[i][top], 1e-6); w = w / w.sum()
            for j, wj in zip(top, w):
                sh[agents[j]] += wj / len(ss)
    share[k] = sh
REC = {}
for f in (glob.glob(f"{ROOT}/results/kp_natural_gpt-4o-mini.jsonl") + glob.glob(f"{ROOT}/results/kp_dense_gpt-4o-mini.jsonl")
          + glob.glob(f"{ROOT}/results/kp_{'mitig' if which == 'mitig' else 'diar'}_gpt-4o-mini.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if "natural" in f and not (r["cond"] == "manual" and r["prompt"] == "minutes"):
            continue
        if r["labels"] is None or r["meeting"] not in KP:
            continue
        a = T[r["meeting"]]
        pts = [x for p, x in zip(KP[r["meeting"]], r["labels"]) if a in p["agents"]]
        if pts:
            REC[(r["model"], r["meeting"], r["cond"], "M0" if (r["prompt"] == "minutes" and r["cond"] != "manual" and which == "mitig") else r["prompt"])] = \
                np.mean([1.0 if x == "written" else 0.5 if x == "partial" else 0.0 for x in pts])
FAB = {}
for f in glob.glob(f"{ROOT}/results/fab_{which}_claude-haiku-5-5.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["is_target"] and r["label"]:
            FAB.setdefault((r["model"], r["meeting"], r["cond"], r["prompt"]), []).append(r["label"])
rows = []
for k in keys:
    model, mid, cond, prompt = k
    if cond == "manual":
        continue
    k0 = (model, mid, "manual", "minutes")
    if k0 not in share:
        continue
    a = T[mid]
    fab = FAB.get(k, [])
    rows.append({"model": model, "meeting": mid, "cond": cond, "method": prompt, "target": a,
                 "nonnative": int(M[mid]["speakers"][a].get("native") != "English"), "role": M[mid]["speakers"][a]["role"],
                 "share0": share[k0][a], "d_share": share[k][a] - share[k0][a],
                 "recall0": REC.get(k0, np.nan), "d_recall": REC.get(k, np.nan) - REC.get(k0, np.nan),
                 "unclear_mention": bool(UNCLEAR.search(texts[k])), "words": len(texts[k].split()),
                 "target_ungrounded": np.mean([x == "ungrounded" for x in fab]) if fab else np.nan,
                 "target_misattributed": np.mean([x == "other_speaker" for x in fab]) if fab else np.nan})
D = pd.DataFrame(rows)
D.to_csv(f"{ROOT}/results/{which}_rows.csv", index=False)


def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    bs = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


eff = []
for (model, cond, method), g in D.groupby(["model", "cond", "method"]):
    r = {"model": model, "cond": cond, "method": method, "n": len(g), "share0": float(g.share0.mean()),
         "unclear_mention": float(g.unclear_mention.mean()), "words": float(g.words.mean()),
         "target_ungrounded": float(g.target_ungrounded.mean()), "target_misattributed": float(g.target_misattributed.mean())}
    for col in ("d_share", "d_recall"):
        r[col], r[col + "_lo"], r[col + "_hi"] = boot(g[col])
    eff.append(r)
E = pd.DataFrame(eff)
if which == "mitig":  # share of the plain-minutes (M0) loss that a method recovers
    base = E[E.method == "M0"].set_index(["model", "cond"])
    for col in ("d_share", "d_recall"):
        E[f"recovered_{col}"] = [1 - getattr(r, col) / base.loc[(r.model, r.cond), col] if (r.model, r.cond) in base.index and base.loc[(r.model, r.cond), col] < 0 else np.nan
                                 for r in E.itertuples()]
else:  # S9: who loses most when labels are wrong
    dd = pd.read_csv(f"{ROOT}/results/diar_dose.csv")
    sp = {}
    for mid, m in M.items():
        w = {}
        for s in m["segments"]:
            w[s["agent"]] = w.get(s["agent"], 0) + len(s["manual"].split())
        sp[mid] = w[T[mid]] / max(1, sum(w.values())) if mid in T else np.nan
    D2 = D.merge(dd[["meeting", "cond", "edge_share", "words_moved"]], on=["meeting", "cond"], how="left")
    D2["speech_share"] = D2.meeting.map(sp)
    fits = {}
    for model, g in D2.groupby("model"):
        g = g.dropna(subset=["d_share", "speech_share", "edge_share"])
        f = smf.ols("d_share ~ words_moved + speech_share + edge_share + nonnative", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.meeting)[0]})
        fits[model] = {k: [float(f.params[k]), float(f.pvalues[k])] for k in f.params.index}
    json.dump(fits, open(f"{ROOT}/results/diar_fits.json", "w"), indent=1)
    print(json.dumps(fits, indent=1))
E.to_csv(f"{ROOT}/results/{which}_effects.csv", index=False)
pd.set_option("display.width", 250)
print(E.round(4).to_string(index=False))
