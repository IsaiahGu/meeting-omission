"""P09 RQ1 test: are non-native speakers represented less in LLM minutes than in the human summaries?

Per speaker and meeting: d = log representation ratio in the LLM minutes minus that in the AMI human
abstract (same speaker, same meeting). Gap = mean d of non-native minus native speakers.
For each attribution method (embedding top-3; gpt-6-luna speaker attribution), model, prompt and
transcript condition: the gap with a meeting-cluster bootstrap 95% CI, and OLS d ~ nonnative + role
+ sex (+ wer_z) with meeting-clustered SEs; its project-manager coefficient tests whether LLM minutes
amplify the PM over-representation already present in the human abstracts. Also pooled over the four
models (mean d per speaker).
Output: results/gap_vs_human.csv, results/gap_vs_human_pooled.csv
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = WORKSPACE_ROOT
W = pd.read_csv(f"{ROOT}/results/speaker_wer.csv")
W["nonnative"] = (W.native != "English").astype(int)
W["wer_z"] = (W.wer - W.wer.mean()) / W.wer.std()
W["sex"] = W["sex"].fillna("unknown"); W["role"] = W["role"].fillna("unknown")
rng = np.random.default_rng(0)


def boot_gap(g, B=2000):
    """non-native minus native mean of d; meeting-cluster bootstrap via per-meeting sums and counts"""
    s = g.assign(d1=g.d * g.nonnative, n1=g.nonnative, d0=g.d * (1 - g.nonnative), n0=1 - g.nonnative)
    s = s.groupby("meeting")[["d1", "n1", "d0", "n0"]].sum().to_numpy()
    est = s[:, 0].sum() / s[:, 1].sum() - s[:, 2].sum() / s[:, 3].sum()
    M = len(s)
    W = np.zeros((B, M))
    np.add.at(W, (np.repeat(np.arange(B), M), rng.integers(0, M, B * M)), 1)
    T = W @ s
    bs = T[:, 0] / T[:, 1] - T[:, 2] / T[:, 3]
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


rows, pooled = [], []
for attr in ("attribution", "attribution_llm"):
    A = pd.read_parquet(f"{ROOT}/results/{attr}.parquet")
    A = A[A.share_speech > 0.02].copy()
    A["lr"] = np.log(np.clip(A.ratio, 0.02, None))
    H = A[A.model == "human"][["meeting", "agent", "lr"]].rename(columns={"lr": "lr_h"})
    L = A[A.model != "human"].merge(H, on=["meeting", "agent"]).merge(
        W[["meeting", "agent", "nonnative", "wer_z", "sex", "role"]], on=["meeting", "agent"])
    L["d"] = L.lr - L.lr_h
    for (model, prompt, cond), g in L.groupby(["model", "prompt", "cond"]):
        e, lo, hi = boot_gap(g, 1000)
        cl = {"groups": pd.factorize(g.meeting)[0]}
        f1 = smf.ols("d ~ nonnative + C(role) + C(sex)", g).fit(cov_type="cluster", cov_kwds=cl)
        f2 = smf.ols("d ~ nonnative + C(role) + C(sex) + wer_z", g).fit(cov_type="cluster", cov_kwds=cl)
        rows.append({"attribution": attr, "model": model, "prompt": prompt, "cond": cond, "n": len(g),
                     "gap": e, "lo": lo, "hi": hi,
                     "adj_gap": float(f1.params["nonnative"]), "adj_p": float(f1.pvalues["nonnative"]),
                     "adj_gap_wer": float(f2.params["nonnative"]), "adj_p_wer": float(f2.pvalues["nonnative"]),
                     "wer_coef": float(f2.params["wer_z"]), "wer_p": float(f2.pvalues["wer_z"]),
                     "pm_amplification": float(f1.params.get("C(role)[T.PM]", np.nan)),
                     "pm_lo": float(f1.conf_int().loc["C(role)[T.PM]", 0]) if "C(role)[T.PM]" in f1.params else np.nan,
                     "pm_hi": float(f1.conf_int().loc["C(role)[T.PM]", 1]) if "C(role)[T.PM]" in f1.params else np.nan,
                     "pm_p": float(f1.pvalues.get("C(role)[T.PM]", np.nan))})
    for (prompt, cond), g in L.groupby(["prompt", "cond"]):
        P = g.groupby(["meeting", "agent", "nonnative", "wer_z", "sex", "role"]).d.mean().reset_index()
        e, lo, hi = boot_gap(P, 2000)
        f = smf.ols("d ~ nonnative + C(role) + C(sex)", P).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(P.meeting)[0]})
        pooled.append({"attribution": attr, "prompt": prompt, "cond": cond, "n_models": g.model.nunique(), "n": len(P),
                       "gap": e, "lo": lo, "hi": hi, "adj_gap": float(f.params["nonnative"]),
                       "adj_lo": float(f.conf_int().loc["nonnative", 0]), "adj_hi": float(f.conf_int().loc["nonnative", 1]),
                       "adj_p": float(f.pvalues["nonnative"]),
                       "pm_amplification": float(f.params.get("C(role)[T.PM]", np.nan)),
                       "pm_lo": float(f.conf_int().loc["C(role)[T.PM]", 0]), "pm_hi": float(f.conf_int().loc["C(role)[T.PM]", 1]),
                       "pm_p": float(f.pvalues.get("C(role)[T.PM]", np.nan))})
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/gap_vs_human.csv", index=False)
Q = pd.DataFrame(pooled); Q.to_csv(f"{ROOT}/results/gap_vs_human_pooled.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
print(Q.round(3).to_string(index=False))
