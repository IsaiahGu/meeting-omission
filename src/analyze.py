"""P09 step 5: who is under-represented, and does ASR error make it worse?

1) Mixed model per (model, prompt, cond): log representation ratio ~ non-native + role + sex
   (+ standardised WER), random intercept per meeting.
2) LLM vs human minutes: paired difference of log ratio per speaker, by native language.
3) ASR effect: delta share (ASR minus manual, same model and prompt) vs speaker WER;
   mediation non-native -> WER -> delta share (bootstrap over meetings).
4) Mitigation: per-speaker-then-merge vs one-pass minutes.
Output: results/analysis.json and printed tables.
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = WORKSPACE_ROOT
import sys
ATTR = sys.argv[1] if len(sys.argv) > 1 else "attribution"  # or attribution_llm (gpt-6-luna attribution)
SUF = "" if ATTR == "attribution" else "_llm"
A = pd.read_parquet(f"{ROOT}/results/{ATTR}.parquet")
W = pd.read_csv(f"{ROOT}/results/speaker_wer.csv")
W["nonnative"] = (W.native != "English").astype(int)
W["wer_z"] = (W.wer - W.wer.mean()) / W.wer.std()
D = A.merge(W[["meeting", "agent", "role", "sex", "nonnative", "wer", "wer_z", "native"]], on=["meeting", "agent"])
D = D[D.share_speech > 0.02].copy()
D["sex"] = D["sex"].fillna("unknown"); D["role"] = D["role"].fillna("unknown"); D["wer_z"] = D["wer_z"].fillna(0.0)
D["logratio"] = np.log(np.clip(D.ratio, 0.02, None))
out = {"n_speakers": int(W.shape[0]), "n_nonnative": int(W.nonnative.sum()),
       "wer_native": float(W[W.nonnative == 0].wer.mean()), "wer_nonnative": float(W[W.nonnative == 1].wer.mean())}
print(out)

res = []
for key, g in D.groupby(["model", "prompt", "cond"]):
    try:
        f = smf.mixedlm("logratio ~ nonnative + C(role) + C(sex) + wer_z", g, groups=g["meeting"]).fit(reml=True)
        for term in ["nonnative", "wer_z", "C(sex)[T.M]", "C(role)[T.PM]", "C(role)[T.ME]", "C(role)[T.UI]"]:
            if term in f.params:
                res.append({"model": key[0], "prompt": key[1], "cond": key[2], "term": term, "coef": f.params[term],
                            "lo": f.conf_int().loc[term, 0], "hi": f.conf_int().loc[term, 1], "p": f.pvalues[term], "n": len(g)})
    except Exception as e:  # noqa: BLE001
        print("fit failed", key, e)
R = pd.DataFrame(res)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
R.to_csv(f"{ROOT}/results/mixed_models{SUF}.csv", index=False)

# 2) LLM vs human
H = D[D.model == "human"][["meeting", "agent", "logratio"]].rename(columns={"logratio": "lr_human"})
cmp_rows = []
for key, g in D[D.model != "human"].groupby(["model", "prompt", "cond"]):
    j = g.merge(H, on=["meeting", "agent"])
    j["diff"] = j.logratio - j.lr_human
    for nn, gg in j.groupby("nonnative"):
        cmp_rows.append({"model": key[0], "prompt": key[1], "cond": key[2], "nonnative": nn,
                         "llm_minus_human_logratio": gg["diff"].mean(), "n": len(gg)})
C = pd.DataFrame(cmp_rows)
print(C.round(3).to_string(index=False))
C.to_csv(f"{ROOT}/results/llm_vs_human{SUF}.csv", index=False)

# 3) ASR effect and mediation
med_rows = []
for key, g in D[D.model != "human"].groupby(["model", "prompt"]):
    p = g.pivot_table(index=["meeting", "agent"], columns="cond", values="share_sum").dropna().reset_index()
    if "asr" not in p or "manual" not in p:
        continue
    p["delta"] = p["asr"] - p["manual"]
    p = p.merge(W[["meeting", "agent", "wer", "wer_z", "nonnative"]], on=["meeting", "agent"]).dropna(subset=["wer_z", "delta"])
    rho = p[["delta", "wer"]].corr(method="spearman").iloc[0, 1]
    fa = smf.ols("wer_z ~ nonnative", p).fit()
    fb = smf.ols("delta ~ wer_z + nonnative", p).fit()
    ind = fa.params["nonnative"] * fb.params["wer_z"]
    rng = np.random.default_rng(0)
    meetings = p.meeting.unique()
    bs = []
    for _ in range(1000):
        pick = rng.choice(meetings, len(meetings))
        q = pd.concat([p[p.meeting == mm] for mm in pick])
        try:
            bs.append(smf.ols("wer_z ~ nonnative", q).fit().params["nonnative"] *
                      smf.ols("delta ~ wer_z + nonnative", q).fit().params["wer_z"])
        except Exception:  # noqa: BLE001
            pass
    med_rows.append({"model": key[0], "prompt": key[1], "spearman_delta_wer": rho,
                     "b_wer_on_delta": fb.params["wer_z"], "p_wer": fb.pvalues["wer_z"],
                     "direct_nonnative": fb.params["nonnative"], "indirect": ind,
                     "indirect_lo": np.percentile(bs, 2.5), "indirect_hi": np.percentile(bs, 97.5), "n": len(p)})
Mdf = pd.DataFrame(med_rows)
print(Mdf.round(4).to_string(index=False))
Mdf.to_csv(f"{ROOT}/results/asr_mediation{SUF}.csv", index=False)

# 4) mitigation: native/non-native gap in log ratio, one-pass vs per-speaker merge
gap = D[D.model != "human"].groupby(["model", "prompt", "cond", "nonnative"]).logratio.mean().unstack("nonnative")
gap["gap_nonnative_minus_native"] = gap[1] - gap[0]
print(gap.round(3).to_string())
gap.reset_index().to_csv(f"{ROOT}/results/mitigation_gap{SUF}.csv", index=False)
json.dump(out, open(f"{ROOT}/results/analysis{SUF}.json", "w"), indent=1)
