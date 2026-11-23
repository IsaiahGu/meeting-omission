"""P09 S2/S3/S5/S7: dose-response over dense white-noise levels and realistic degradations.

Readouts per (model, meeting, condition), all relative to the same model's minutes of the manual transcript:
  d_share    target's embedding-attribution share change (top-3 utterances per sentence, as before)
  d_llm      the same with the batched LLM attribution (attr_batch.py), when available
  d_mention  change in the share of explicit "Speaker X" mentions
  d_recall   change in the target's key-point recall (keypoints.py), when available
Dose = the target's WER in that condition (manual_retest: 0). Per model:
  - linear fit through the origin, d = b * WER, and a hinge, d = b * max(0, WER - tau); compared by
    leave-one-meeting-out prediction error; tau and the WER at a 5% / 10% relative loss with meeting-bootstrap CIs
    (fitted on the white-noise levels only, so the realistic degradations are out-of-sample);
  - realistic degradations against the white-noise curve: mean residual per condition with CI;
    words kept as a second dose (packet loss mostly deletes words);
  - accent: non-native minus native target WER and share change under far-field and small recognisers.
Output: results/dense_rows.csv, results/noise_effects_dense.csv, results/dose_fit.json
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
TASK = "Given a sentence from meeting minutes, retrieve the meeting utterance that supports it"
rng = np.random.default_rng(0)
WHITE = ["snr15", "snr10", "snr5", "snr0", "snr-5", "snr-10"]
REAL = ["plc10", "plc30", "reverb", "babble10", "babble0", "farfield", "whisper-tiny", "whisper-base", "whisper-small"]
DOSE = pd.read_csv(f"{ROOT}/results/dense_dose.csv")
DOSE = pd.concat([DOSE, pd.DataFrame([{"meeting": m, "cond": "manual_retest", "wer": 0.0, "words_kept": 1.0} for m in T])])

# ---- minutes texts: new runs + references
S = []
for f in glob.glob(f"{ROOT}/results/dense_summaries_*.jsonl"):
    model = f.split("dense_summaries_")[1][:-6]
    S += [{**json.loads(l), "model": model} for l in open(f)]
S = [s for s in S if not s.get("error") and s["text"].strip()]
have_manual = {(s["model"], s["meeting"]) for s in S if s["cond"] == "manual"}
for model in {s["model"] for s in S}:
    for fn, cond in ((f"{ROOT}/results/summaries_{model}.jsonl", "manual"), (f"{ROOT}/results/noise_summaries_{model}.jsonl", "manual_retest")):
        if not os.path.exists(fn):
            continue
        for l in open(fn):
            r = json.loads(l)
            if r["cond"] == cond and r["prompt"] == "minutes" and r["text"].strip() and (model, r["meeting"]) not in have_manual:
                S.append({**r, "model": model})
texts = {(s["model"], s["meeting"], s["cond"]): s["text"] for s in S}


def split_sents(t):
    parts = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"[*#]+", " ", t))
    return [p.strip(" -•\t") for p in parts if len(p.strip(" -•\t").split()) >= 4]


keys = list(texts)
sents = [split_sents(texts[k]) for k in keys]
flat = [x for ss in sents for x in ss]
Q = encode("qwen3-emb", flat, "query", qwen_task=TASK, bs=64).astype(np.float32)
utt = {}
for mid in T:
    segs = [s for s in M[mid]["segments"] if len(s["manual"].split()) >= 4]
    utt[mid] = (np.array([s["agent"] for s in segs]), encode("qwen3-emb", [s["manual"] for s in segs], "doc", cache_key=f"p09_utt_{mid}").astype(np.float32))
share, mention = {}, {}
pos = 0
for k, ss in zip(keys, sents):
    mid = k[1]; agents, U = utt[mid]
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
    share[k] = sh; mention[k] = {a: (v / mt if mt else np.nan) for a, v in me.items()}
LLM_SH = {}
for f in glob.glob(f"{ROOT}/results/attr_batch_dense_*.parquet"):
    A = pd.read_parquet(f)
    for r in A.itertuples():
        LLM_SH[(r.model, r.meeting, r.cond, r.agent)] = r.share_llm
# Key-point recall from the main judge (gpt-4o-mini) only; claude-haiku's 10% re-judge measures agreement and is not
# mixed into the readout. The API models' references are their earlier minutes: manual from the natural run
# (kp_natural, same judge, prompt and cache), manual_retest from the earlier noise run (minutes_v1 rows of kp_dense).
KP = json.load(open(f"{ROOT}/data/keypoints.json"))
fresh_models = {m for m, _ in have_manual}
REC = {}
for fn, use in ((f"{ROOT}/results/kp_natural_gpt-4o-mini.jsonl",
                 lambda r: r["prompt"] == "minutes" and r["cond"] == "manual" and r["model"] not in fresh_models),
                (f"{ROOT}/results/kp_dense_gpt-4o-mini.jsonl",
                 lambda r: r["prompt"] == "minutes" or (r["prompt"] == "minutes_v1" and r["cond"] == "manual_retest"
                                                        and r["model"] not in fresh_models))):
    for l in open(fn):
        r = json.loads(l)
        if r["labels"] is None or r["meeting"] not in KP or r["meeting"] not in T or not use(r):
            continue
        mid = r["meeting"]; a = T[mid]
        pts = [x for p, x in zip(KP[mid], r["labels"]) if a in p["agents"]]
        if pts:
            REC[(r["model"], mid, r["cond"])] = np.mean([1.0 if x == "written" else 0.5 if x == "partial" else 0.0 for x in pts])
rows = []
for k in keys:
    model, mid, cond = k
    if cond == "manual" or (model, mid, "manual") not in share:
        continue
    a = T[mid]; sp = M[mid]["speakers"][a]; k0 = (model, mid, "manual")
    rows.append({"model": model, "meeting": mid, "cond": cond, "target": a, "role": sp["role"], "nonnative": int(sp.get("native") != "English"),
                 "share0": share[k0][a], "d_share": share[k][a] - share[k0][a], "d_mention": mention[k][a] - mention[k0][a],
                 "d_llm": LLM_SH.get((model, mid, cond, a), np.nan) - LLM_SH.get((model, mid, "manual", a), np.nan),
                 "llm0": LLM_SH.get((model, mid, "manual", a), np.nan),
                 "d_recall": REC.get((model, mid, cond), np.nan) - REC.get((model, mid, "manual"), np.nan),
                 "recall0": REC.get((model, mid, "manual"), np.nan)})
D = pd.DataFrame(rows).merge(DOSE[["meeting", "cond", "wer", "words_kept"]], on=["meeting", "cond"], how="left")
D.to_csv(f"{ROOT}/results/dense_rows.csv", index=False)


def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    bs = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


eff = []
for (model, cond), g in D.groupby(["model", "cond"]):
    r = {"model": model, "cond": cond, "n": len(g), "wer": float(g.wer.mean()), "words_kept": float(g.words_kept.mean()),
         "share0": float(g.share0.mean())}
    for col in ("d_share", "d_llm", "d_mention", "d_recall"):
        r[col], r[col + "_lo"], r[col + "_hi"] = boot(g[col])
    r["rel_change"] = r["d_share"] / r["share0"]
    eff.append(r)
E = pd.DataFrame(eff)
E.to_csv(f"{ROOT}/results/noise_effects_dense.csv", index=False)


def fit_hinge(w, d, taus=np.arange(0.0, 0.81, 0.01)):
    best = None
    for t in taus:
        x = np.maximum(0, w - t)
        if (x ** 2).sum() == 0:
            continue
        b = (x * d).sum() / (x ** 2).sum()
        sse = ((d - b * x) ** 2).sum()
        if best is None or sse < best[2]:
            best = (t, b, sse)
    return best


def fit_lin(w, d):
    b = (w * d).sum() / (w ** 2).sum()
    return b, ((d - b * w) ** 2).sum()


fits = {"models": {}}
for model, g in D.groupby("model"):
    for readout in ("d_share", "d_llm", "d_recall"):
        G = g[g.cond.isin(WHITE + ["manual_retest", "asr_target"])].dropna(subset=["wer", readout])
        if len(G) < 50:
            continue
        w, d, mids = G.wer.to_numpy(), G[readout].to_numpy(), G.meeting.to_numpy()
        t, b, sse_h = fit_hinge(w, d); bl, sse_l = fit_lin(w, d)
        # leave-one-meeting-out prediction error
        err_h, err_l = [], []
        for m in np.unique(mids):
            tr, te = mids != m, mids == m
            th, bh, _ = fit_hinge(w[tr], d[tr]); b2, _ = fit_lin(w[tr], d[tr])
            err_h += list((d[te] - bh * np.maximum(0, w[te] - th)) ** 2); err_l += list((d[te] - b2 * w[te]) ** 2)
        err_h, err_l = np.array(err_h), np.array(err_l)
        diff = err_l - err_h
        # meeting bootstrap of tau and of the WER at 5% / 10% relative loss
        share0 = float(g[g.cond == "manual_retest"].share0.mean()) if readout == "d_share" else float(np.nanmean(g.llm0 if readout == "d_llm" else g.recall0))
        um = np.unique(mids); idx = {m: np.where(mids == m)[0] for m in um}
        bt, b5, b10 = [], [], []
        for _ in range(500):
            take = np.concatenate([idx[m] for m in rng.choice(um, len(um))])
            th, bh, _ = fit_hinge(w[take], d[take])
            bt.append(th)
            if bh < 0:
                b5.append(th + 0.05 * share0 / -bh); b10.append(th + 0.10 * share0 / -bh)
        pc = lambda x, q: float(np.percentile(x, q)) if len(x) else np.nan
        fits["models"].setdefault(model, {})[readout] = {
            "n": len(G), "tau": float(t), "tau_ci": [pc(bt, 2.5), pc(bt, 97.5)], "hinge_slope_per10": float(b * 0.1),
            "linear_slope_per10": float(bl * 0.1), "loocv_mse_linear": float(err_l.mean()), "loocv_mse_hinge": float(err_h.mean()),
            "loocv_mse_diff_lin_minus_hinge": [float(diff.mean()), *boot(diff)[1:]], "share0": share0,
            "wer_at_5pct_loss": float(t + 0.05 * share0 / -b) if b < 0 else np.nan, "wer_at_5pct_ci": [pc(b5, 2.5), pc(b5, 97.5)],
            "wer_at_10pct_loss": float(t + 0.10 * share0 / -b) if b < 0 else np.nan, "wer_at_10pct_ci": [pc(b10, 2.5), pc(b10, 97.5)]}
        if readout == "d_share":
            # realistic degradations against the white-noise hinge
            res = {}
            for c in REAL:
                h = g[g.cond == c].dropna(subset=["wer", "d_share"])
                if len(h):
                    pred = b * np.maximum(0, h.wer - t)
                    res[c] = {"n": len(h), "wer": float(h.wer.mean()), "words_kept": float(h.words_kept.mean()),
                              "d_share": boot(h.d_share), "residual_vs_white_curve": boot(h.d_share - pred)}
            fits["models"][model]["realistic"] = res
            A = g[g.cond.isin(WHITE + REAL + ["asr_target", "manual_retest"])].dropna(subset=["wer", "words_kept", "d_share"])
            f2 = smf.ols("d_share ~ wer + words_kept", A).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(A.meeting)[0]})
            fits["models"][model]["wer_and_words_kept"] = {k: [float(f2.params[k]), float(f2.pvalues[k])] for k in ("wer", "words_kept")}
            acc = {}
            for c in ["farfield", "whisper-tiny", "whisper-base", "whisper-small", "asr_target"] + WHITE:
                h = g[g.cond == c].dropna(subset=["wer", "d_share"])
                if h.nonnative.nunique() == 2:
                    nn, na = h[h.nonnative == 1], h[h.nonnative == 0]
                    gw = [rng.choice(nn.wer.values, len(nn)).mean() - rng.choice(na.wer.values, len(na)).mean() for _ in range(1000)]
                    gd = [rng.choice(nn.d_share.values, len(nn)).mean() - rng.choice(na.d_share.values, len(na)).mean() for _ in range(1000)]
                    acc[c] = {"wer_gap": [float(nn.wer.mean() - na.wer.mean()), float(np.percentile(gw, 2.5)), float(np.percentile(gw, 97.5))],
                              "d_share_gap": [float(nn.d_share.mean() - na.d_share.mean()), float(np.percentile(gd, 2.5)), float(np.percentile(gd, 97.5))],
                              "n_nonnative": len(nn), "n_native": len(na)}
            fits["models"][model]["accent"] = acc
json.dump(fits, open(f"{ROOT}/results/dose_fit.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(E[["model", "cond", "n", "wer", "words_kept", "d_share", "d_share_lo", "d_share_hi", "rel_change", "d_llm", "d_recall"]].round(4).to_string(index=False))
print(json.dumps({m: {r: {k: v for k, v in x.items() if k in ("tau", "tau_ci", "hinge_slope_per10", "linear_slope_per10", "loocv_mse_diff_lin_minus_hinge", "wer_at_5pct_loss", "wer_at_10pct_loss")}
                      for r, x in f.items() if r.startswith("d_")} for m, f in fits["models"].items()}, indent=1, default=float))
