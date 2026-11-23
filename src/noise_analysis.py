"""P09 extension step 3: does a speaker lose share in the minutes when only their transcript degrades?

Share = embedding attribution as in attribute.py (each summary sentence -> top-3 manual utterances,
weighted by similarity). Readouts per (model, prompt, meeting, condition), all relative to the
existing manual-transcript minutes of the same model and prompt:
  d_share    target's share minus its manual-transcript share
  d_mention  the same for explicit "Speaker X" mentions
Dose = the target's WER in that condition (manual_retest: 0; asr_target: clean-audio Whisper WER).
Reported per model and prompt: mean change per condition (meeting bootstrap), the slope of d_share
on WER (meeting-clustered SEs), the native/non-native WER gap at each SNR, and whether non-native
targets lose more at the same SNR.
Output: results/noise_rows.csv, results/noise_effects.csv, results/noise_summary.json
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import re
import sys

import jiwer
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from embed import encode  # noqa: E402

ROOT = WORKSPACE_ROOT
norm = EnglishTextNormalizer(json.load(open(f"{WORKSPACE_ROOT}/models/whisper-large-v3/normalizer.json")))
M = json.load(open(f"{ROOT}/data/meetings.json"))
T = json.load(open(f"{ROOT}/data/noise_targets.json"))
W1 = pd.read_csv(f"{ROOT}/results/speaker_wer.csv")
TASK = "Given a sentence from meeting minutes, retrieve the meeting utterance that supports it"
rng = np.random.default_rng(0)

# ---- dose: target WER per condition
recs = []
for p in sorted(glob.glob(f"{ROOT}/results/noise_asr_s*.jsonl")):
    for line in open(p):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
NZ = pd.DataFrame(recs).drop_duplicates(["audio_id", "snr"])
wr = []
for (mid, snr), g in NZ.groupby(["meeting_id", "snr"]):
    refs = [norm(x) for x in g.ref]; hyps = [norm(x) for x in g.hyp]
    keep = [i for i, x in enumerate(refs) if x.strip()]
    if keep:
        wr.append({"meeting": mid, "cond": f"snr{snr:g}", "wer": jiwer.wer([refs[i] for i in keep], [hyps[i] for i in keep])})
w1 = W1.set_index(["meeting", "agent"]).wer
WT = pd.concat([pd.DataFrame(wr),
                pd.DataFrame([{"meeting": m, "cond": "asr_target", "wer": w1.get((m, a), np.nan)} for m, a in T.items()]),
                pd.DataFrame([{"meeting": m, "cond": "manual_retest", "wer": 0.0} for m in T])], ignore_index=True)

# ---- summaries: noise conditions + manual references
S = []
for f in glob.glob(f"{ROOT}/results/noise_summaries_*.jsonl"):
    model = f.split("noise_summaries_")[1][:-6]
    S += [{**json.loads(l), "model": model} for l in open(f)]
S = [s for s in S if not s.get("error") and s["text"].strip()]  # failed or empty calls are not summaries
models = sorted({s["model"] for s in S})
ref = {}
for model in models:
    for l in open(f"{ROOT}/results/summaries_{model}.jsonl"):
        r = json.loads(l)
        if r["cond"] == "manual" and r["text"].strip():
            ref[(model, r["prompt"], r["meeting"])] = r["text"]


def split_sents(t):
    parts = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"[*#]+", " ", t))
    return [p.strip(" -•\t") for p in parts if len(p.strip(" -•\t").split()) >= 4]


texts = {}
for s in S:
    texts[(s["model"], s["prompt"], s["meeting"], s["cond"])] = s["text"]
    k0 = (s["model"], s["prompt"], s["meeting"])
    if k0 in ref:
        texts[(*k0, "manual")] = ref[k0]
keys = list(texts)
sents = [split_sents(texts[k]) for k in keys]
flat = [x for ss in sents for x in ss]
Q = encode("qwen3-emb", flat, "query", qwen_task=TASK).astype(np.float32)
utt = {}
for mid in T:
    segs = [s for s in M[mid]["segments"] if len(s["manual"].split()) >= 4]
    utt[mid] = (np.array([s["agent"] for s in segs]),
                encode("qwen3-emb", [s["manual"] for s in segs], "doc", cache_key=f"p09_utt_{mid}").astype(np.float32))
share, mention = {}, {}
pos = 0
for k, ss in zip(keys, sents):
    mid = k[2]
    agents, U = utt[mid]
    q = Q[pos:pos + len(ss)]; pos += len(ss)
    sh = {a: 0.0 for a in M[mid]["speakers"]}; me = {a: 0.0 for a in M[mid]["speakers"]}
    if len(ss):
        sim = q @ U.T
        for i in range(len(ss)):
            top = np.argsort(-sim[i])[:3]
            w = np.maximum(sim[i][top], 1e-6); w = w / w.sum()
            for j, wj in zip(top, w):
                sh[agents[j]] += wj / len(ss)
            ms = set(re.findall(r"Speaker ([A-D])", ss[i])) & set(me)
            for a in ms:
                me[a] += 1 / len(ms)
    mt = sum(me.values())
    share[k] = sh
    mention[k] = {a: (v / mt if mt else np.nan) for a, v in me.items()}

rows = []
for k in keys:
    model, prompt, mid, cond = k
    if cond == "manual" or (model, prompt, mid, "manual") not in share:
        continue
    a = T[mid]
    sp = M[mid]["speakers"][a]
    k0 = (model, prompt, mid, "manual")
    rows.append({"model": model, "prompt": prompt, "meeting": mid, "cond": cond, "target": a, "role": sp["role"],
                 "nonnative": int(sp.get("native") != "English"), "share0": share[k0][a], "share": share[k][a],
                 "d_share": share[k][a] - share[k0][a], "d_mention": mention[k][a] - mention[k0][a]})
D = pd.DataFrame(rows).merge(WT, on=["meeting", "cond"], how="left")
D.to_csv(f"{ROOT}/results/noise_rows.csv", index=False)


def boot(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    bs = rng.choice(x, (2000, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


eff = []
for (model, prompt, cond), g in D.groupby(["model", "prompt", "cond"]):
    e, lo, hi = boot(g.d_share)
    me_, mlo, mhi = boot(g.d_mention)
    eff.append({"model": model, "prompt": prompt, "cond": cond, "n": len(g), "wer": g.wer.mean(), "share0": g.share0.mean(),
                "d_share": e, "lo": lo, "hi": hi, "rel_change": e / g.share0.mean(),
                "d_mention": me_, "mention_lo": mlo, "mention_hi": mhi})
E = pd.DataFrame(eff)
E.to_csv(f"{ROOT}/results/noise_effects.csv", index=False)
print(E.round(4).to_string(index=False))

out = {"n_meetings": len(T), "target_nonnative_share": float(np.mean([M[m]["speakers"][a].get("native") != "English" for m, a in T.items()]))}
# WER by condition and nativeness (accent x noise)
WN = WT.assign(nonnative=WT.meeting.map(lambda m: int(M[m]["speakers"][T[m]].get("native") != "English")))
gap = {}
for cond, g in WN.groupby("cond"):
    nn, na = g[g.nonnative == 1].wer.dropna().to_numpy(), g[g.nonnative == 0].wer.dropna().to_numpy()
    bs = [rng.choice(nn, len(nn)).mean() - rng.choice(na, len(na)).mean() for _ in range(2000)]
    gap[cond] = {"wer_native": float(na.mean()), "wer_nonnative": float(nn.mean()), "gap": float(nn.mean() - na.mean()),
                 "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_native": len(na), "n_nonnative": len(nn)}
out["wer_by_cond"] = gap
# dose-response: slope of d_share on WER, and non-native x WER, per model and prompt
fits = {}
for (model, prompt), g in D.groupby(["model", "prompt"]):
    g = g.dropna(subset=["wer", "d_share"])
    cl = {"groups": pd.factorize(g.meeting)[0]}
    f1 = smf.ols("d_share ~ wer", g).fit(cov_type="cluster", cov_kwds=cl)
    f2 = smf.ols("d_share ~ wer * nonnative", g).fit(cov_type="cluster", cov_kwds=cl)
    ref = "manual_retest" if (g.cond == "manual_retest").any() else "asr_target"  # the merge runs have no retest
    f3 = smf.ols(f"d_share ~ C(cond, Treatment('{ref}')) * nonnative", g).fit(cov_type="cluster", cov_kwds=cl)
    fits[f"{model}|{prompt}"] = {
        "slope_per_10pt_wer": float(f1.params["wer"] * 0.1), "slope_lo": float(f1.conf_int().loc["wer", 0] * 0.1),
        "slope_hi": float(f1.conf_int().loc["wer", 1] * 0.1), "p": float(f1.pvalues["wer"]),
        "nonnative_x_wer": float(f2.params["wer:nonnative"]), "p_nonnative_x_wer": float(f2.pvalues["wer:nonnative"]),
        "cond_x_nonnative": {k: [float(v), float(f3.pvalues[k])] for k, v in f3.params.items() if ":nonnative" in k},
        "retest_sd": float(g[g.cond == "manual_retest"].d_share.std()) if ref == "manual_retest" else None,
        "cond_reference": ref, "n": len(g)}
out["dose_response"] = fits
json.dump(out, open(f"{ROOT}/results/noise_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
