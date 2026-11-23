"""P09 S7.4: Holm correction across the four models for each family of same-kind tests.

Families (each = one test per model, the four models together):
  pm_amplification   PM over-representation relative to the human summaries (gap_vs_human.csv, pm_p), per attribution x prompt x transcript
  nonnative_vs_human non-native gap relative to the human summaries, role and sex adjusted (gap_vs_human.csv, adj_p)
  nonnative_mixed    non-native coefficient of the mixed models (mixed_models[_llm].csv), per prompt x transcript
  natural_slope      share change (ASR minus manual) on WER (null_bounds.csv, normal p from slope / SE), per attribution x prompt
  recall_nonnative   non-native coefficient of key-point recall (kp_natural.json, gpt-4o-mini judge), per prompt x transcript
python s7_holm.py   Output: results/s7_holm.csv
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

R = f"{WORKSPACE_ROOT}/results"
MODELS = ["local", "gemma-4-12b", "gpt-4o-mini", "gpt-6-luna"]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


rows = []
G = pd.read_csv(f"{R}/gap_vs_human.csv")
for (a, pr, c), g in G.groupby(["attribution", "prompt", "cond"]):
    g = g[g.model.isin(MODELS)]
    for fam, est, pc in (("pm_amplification", "pm_amplification", "pm_p"), ("nonnative_vs_human", "adj_gap", "adj_p")):
        rows += [{"family": fam, "setting": f"{a}|{pr}|{c}", "model": r.model, "estimate": getattr(r, est), "p": getattr(r, pc)} for r in g.itertuples()]
for fn, a in (("mixed_models.csv", "attribution"), ("mixed_models_llm.csv", "attribution_llm")):
    if os.path.exists(f"{R}/{fn}"):
        X = pd.read_csv(f"{R}/{fn}")
        X = X[(X.term == "nonnative") & X.model.isin(MODELS)]
        rows += [{"family": "nonnative_mixed", "setting": f"{a}|{r.prompt}|{r.cond}", "model": r.model, "estimate": r.coef, "p": r.p} for r in X.itertuples()]
N = pd.read_csv(f"{R}/null_bounds.csv")
N = N[N.model.isin(MODELS)]
rows += [{"family": "natural_slope", "setting": f"{r.attribution}|{r.prompt}", "model": r.model, "estimate": r.slope_per10,
          "p": float(2 * norm.sf(abs(r.slope_per10 / r.se)))} for r in N.itertuples()]
K = json.load(open(f"{R}/kp_natural.json"))["gpt-4o-mini"]
for key, v in K.items():
    model, prompt, cond = key.split("|")
    if model in MODELS and "nonnative" in v:
        rows += [{"family": "recall_nonnative", "setting": f"{prompt}|{cond}", "model": model, "estimate": v["nonnative"][0], "p": v["nonnative"][2]}]
D = pd.DataFrame(rows).dropna(subset=["p"])
D["p_holm"] = np.nan
for (f, s), g in D.groupby(["family", "setting"]):
    D.loc[g.index, "p_holm"] = holm(g.p.values)
D["sig_raw"] = D.p < 0.05; D["sig_holm"] = D.p_holm < 0.05
D.to_csv(f"{R}/s7_holm.csv", index=False)
pd.set_option("display.width", 200)
print(D.groupby("family").agg(tests=("p", "size"), sig_raw=("sig_raw", "sum"), sig_holm=("sig_holm", "sum")).to_string())
print("\nsignificant before Holm, not after:")
print(D[D.sig_raw & ~D.sig_holm][["family", "setting", "model", "estimate", "p", "p_holm"]].round(4).to_string(index=False))
