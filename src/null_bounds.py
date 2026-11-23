"""P09 S1: how tight is the natural-ASR null? Bounds, equivalence and power.

Natural experiment: per speaker and meeting, delta = share in minutes of the Whisper transcript minus share
in minutes of the manual transcript (same model and prompt; embedding attribution, as in the noise
experiment; LLM attribution as a check). Per model and prompt, and pooled over the four models (meeting
random intercept, model fixed effects):
- slope of delta on WER (per 10 WER points) with 90% and 95% CIs (meeting-clustered SEs);
- the indirect effect non-native -> WER -> delta with a 90% bootstrap CI (meetings resampled);
- minimum detectable slope at 80% power (two-sided alpha 0.05): (1.96 + 0.84) x SE;
- the slope implied by the dose-response experiment (S2's six-level dose_fit.json when present, else the
  four-level results/noise_summary.json, kept as dose_slope_v1) and the loss it predicts for the accent WER gap;
  does the natural 90% CI contain it? Also the slope the natural data would show if S2's hinge held for every
  speaker (hinge_implied_slope_per10: the hinge's predicted change per speaker, regressed on WER).
Output: results/null_bounds.json, results/null_bounds.csv
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = WORKSPACE_ROOT
rng = np.random.default_rng(0)
W = pd.read_csv(f"{ROOT}/results/speaker_wer.csv")
W["nonnative"] = (W.native != "English").astype(int)
gap = float(W[W.nonnative == 1].wer.mean() - W[W.nonnative == 0].wer.mean())
dose = json.load(open(f"{ROOT}/results/noise_summary.json"))["dose_response"]
rows, out = [], {"accent_wer_gap": gap}
NAT = {}  # natural rows (embedding attribution, minutes prompt) for the hinge-implied slope
for attr in ("attribution", "attribution_llm"):
    A = pd.read_parquet(f"{ROOT}/results/{attr}.parquet")
    A = A[A.model != "human"]
    P = A.pivot_table(index=["meeting", "agent", "model", "prompt"], columns="cond", values="share_sum").reset_index()
    P = P.dropna(subset=["manual", "asr"]).assign(delta=lambda d: d.asr - d.manual)
    P = P.merge(W[["meeting", "agent", "wer", "nonnative"]], on=["meeting", "agent"])
    P["wer10"] = P.wer * 10  # slope per 10 WER points
    for prompt in ("minutes", "decisions"):
        G = P[P.prompt == prompt]
        for model, g in list(G.groupby("model")) + [("pooled", G)]:
            cl = {"groups": pd.factorize(g.meeting)[0]}
            if attr == "attribution" and prompt == "minutes":
                NAT[model] = g[["model", "wer10"]].copy()
            if model == "pooled":
                f = smf.ols("delta ~ wer10 + C(model)", g).fit(cov_type="cluster", cov_kwds=cl)
                try:
                    mm = smf.mixedlm("delta ~ wer10 + C(model)", g, groups=g.meeting).fit(reml=True)
                    mixed = [float(mm.params.wer10), float(mm.bse.wer10)]
                except Exception:  # noqa: BLE001
                    mixed = [np.nan, np.nan]
            else:
                f = smf.ols("delta ~ wer10", g).fit(cov_type="cluster", cov_kwds=cl)
                mixed = [np.nan, np.nan]
            b, se = float(f.params.wer10), float(f.bse.wer10)
            # indirect effect with a 90% meeting bootstrap
            ms = g.meeting.unique(); bs = []
            idx = {m: np.where(g.meeting.values == m)[0] for m in ms}
            for _ in range(1000):
                take = np.concatenate([idx[m] for m in rng.choice(ms, len(ms))])
                q = g.iloc[take]
                try:
                    a1 = np.polyfit(q.nonnative, q.wer10, 1)[0]
                    X = np.column_stack([np.ones(len(q)), q.wer10, q.nonnative] + ([pd.get_dummies(q.model, drop_first=True).to_numpy(float)] if model == "pooled" else []))
                    b1 = np.linalg.lstsq(X, q.delta.to_numpy(), rcond=None)[0][1]
                    bs.append(a1 * b1)
                except Exception:  # noqa: BLE001
                    pass
            rows.append({"attribution": attr, "prompt": prompt, "model": model, "n": len(g), "slope_per10": b, "se": se,
                         "ci90": [b - 1.645 * se, b + 1.645 * se], "ci95": [b - 1.96 * se, b + 1.96 * se],
                         "mde80_per10": 2.80 * se, "mixed_slope_se": mixed,
                         "indirect": float(np.mean(bs)) if bs else np.nan,
                         "indirect_ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))] if bs else [np.nan, np.nan]})
R = pd.DataFrame(rows)
# dose-response reference (embedding attribution, minutes prompt)
dose_slopes = {k.split("|")[0]: v["slope_per_10pt_wer"] for k, v in dose.items() if k.endswith("|minutes")}
out["dose_slopes_v1_per10"] = dict(dose_slopes)  # earlier four-level dose run
HINGE = {}
if os.path.exists(f"{ROOT}/results/dose_fit.json"):  # S2: six white-noise levels, d_share readout
    df_ = json.load(open(f"{ROOT}/results/dose_fit.json"))
    for m, v in df_.get("models", {}).items():
        if "d_share" in v:
            dose_slopes[m] = v["d_share"]["linear_slope_per10"]
            HINGE[m] = (v["d_share"]["tau"], v["d_share"]["hinge_slope_per10"] * 10)
out["dose_slopes_per10"] = dose_slopes
out["hinge_tau_slope_per1"] = HINGE


def pooled_mean(d):  # mean over the models in the pooled natural data only (not the S11 27B)
    return float(np.mean([d[m] for m in NAT["pooled"].model.unique() if m in d]))


def hinge_implied(m):
    # slope on WER (per 10 points) that the natural data would show if the S2 hinge held for every speaker
    g = NAT.get(m)
    if g is None or any(x not in HINGE for x in g.model.unique()):
        return np.nan
    pred = np.array([HINGE[x][1] * max(0.0, w / 10 - HINGE[x][0]) for x, w in zip(g.model, g.wer10)])
    return float(np.polyfit(g.wer10.to_numpy(float), pred, 1)[0])
cmp = []
for r in R[(R.attribution == "attribution") & (R.prompt == "minutes")].itertuples():
    ds = dose_slopes.get(r.model, pooled_mean(dose_slopes) if r.model == "pooled" else np.nan)
    cmp.append({"model": r.model, "natural_slope_per10": r.slope_per10, "natural_ci90": r.ci90, "dose_slope_per10": ds,
                "natural_ci90_contains_dose_slope": bool(r.ci90[0] <= ds <= r.ci90[1]) if not np.isnan(ds) else None,
                "natural_ci90_excludes_dose_slope_as_too_steep": bool(ds < r.ci90[0]) if not np.isnan(ds) else None,
                "dose_slope_v1_per10": out["dose_slopes_v1_per10"].get(r.model, pooled_mean(out["dose_slopes_v1_per10"]) if r.model == "pooled" else np.nan),
                "hinge_implied_slope_per10": hinge_implied(r.model),
                "natural_ci90_contains_hinge_implied": bool(r.ci90[0] <= hinge_implied(r.model) <= r.ci90[1]),
                "predicted_loss_for_accent_gap": float(ds * gap * 10) if not np.isnan(ds) else np.nan,
                "natural_bound_loss_for_accent_gap_ci90": [float(x * gap * 10) for x in r.ci90], "mde80_per10": r.mde80_per10})
out["comparison"] = cmp
out["table"] = R.to_dict(orient="records")
json.dump(out, open(f"{ROOT}/results/null_bounds.json", "w"), indent=1, default=float)
R.to_csv(f"{ROOT}/results/null_bounds.csv", index=False)
pd.set_option("display.width", 220)
print(R[["attribution", "prompt", "model", "n", "slope_per10", "se", "mde80_per10", "indirect"]].round(4).to_string(index=False))
print(json.dumps(cmp, indent=1, default=float))
