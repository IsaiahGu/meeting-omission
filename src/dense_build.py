"""P09 S2/S3/S7: transcripts in which only the target speaker's lines come from a degraded recognition.

Reads results/degrade_asr_*_s*.jsonl (degrade_asr.py). Condition names: the large-v3 conditions keep
their name (snr15 ... snr-10, plc10, plc30, reverb, babble10, babble0, farfield); the smaller models
on clean audio are whisper-tiny, whisper-base, whisper-small. asr_target = the clean-audio Whisper
large-v3 pass already in data/meetings.json. Everyone except the target keeps the manual transcript.
Per meeting and condition: target WER (Whisper normaliser), words kept (hypothesis words / reference
words), segments kept, and the target's share of all transcript words.
Output: data/dense_transcripts.json {meeting: {target, <cond>: transcript}},
        data/dense_segments.json {meeting: {cond: [[begin, hyp, avg_logprob], ...]}} (target only),
        results/dense_dose.csv
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import re

import jiwer
import numpy as np
import pandas as pd
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

ROOT = WORKSPACE_ROOT
norm = EnglishTextNormalizer(json.load(open(f"{WORKSPACE_ROOT}/models/whisper-large-v3/normalizer.json")))
M = json.load(open(f"{ROOT}/data/meetings.json"))
T = json.load(open(f"{ROOT}/data/noise_targets.json"))


def clean(t):
    t = re.sub(r"[^a-z0-9' ]+", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


hyp = {}  # (meeting, global speaker, begin) -> {cond: (hyp, lp)}
refs = {}
for p in sorted(glob.glob(f"{ROOT}/results/degrade_asr_*_s*.jsonl")):
    tag = p.split("degrade_asr_")[1].rsplit("_s", 1)[0]
    for line in open(p):
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        cond = j["cond"] if tag == "whisper-large-v3" else tag
        k = (j["meeting_id"], j["speaker_id"], round(float(j["begin"]), 2))
        hyp.setdefault(k, {}).setdefault(cond, (j["hyp"], j.get("avg_logprob")))
        refs[k] = j["ref"]
conds = sorted({c for v in hyp.values() for c in v})
print("conditions", conds)
out, segs, dose = {}, {}, []
for mid in sorted(T):
    m = M[mid]; a = T[mid]; g = m["speakers"][a]["global"]
    lines = {c: [] for c in ["asr_target"] + conds}
    seg_out = {c: [] for c in conds}
    rw = {c: [[], []] for c in ["asr_target"] + conds}
    tot_t = {c: 0 for c in lines}; tot_o = 0; man_t = 0; nseg = 0; kept = {c: 0 for c in lines}
    for s in m["segments"]:
        if s["agent"] != a:
            if s["manual"]:
                for c in lines:
                    lines[c].append(f"Speaker {s['agent']}: {s['manual']}")
                tot_o += len(s["manual"].split())
            continue
        nseg += 1; man_t += len(s["manual"].split())
        k = (mid, g, round(float(s["begin"]), 2))
        hz = hyp.get(k, {})
        ref = refs.get(k)
        if s["asr"]:
            lines["asr_target"].append(f"Speaker {a}: {s['asr']}"); tot_t["asr_target"] += len(s["asr"].split()); kept["asr_target"] += 1
        if ref is not None:
            rw["asr_target"][0].append(norm(ref)); rw["asr_target"][1].append(norm(s["asr"]))
        for c in conds:
            h, lp = hz.get(c, ("", None))
            hc = clean(h)
            if c in hz:
                seg_out[c].append([s["begin"], hc, lp])
                rw[c][0].append(norm(ref)); rw[c][1].append(norm(h))
            if hc:
                lines[c].append(f"Speaker {a}: {hc}"); tot_t[c] += len(hc.split()); kept[c] += 1
    out[mid] = {"target": a, **{c: "\n".join(v) for c, v in lines.items()}}
    segs[mid] = seg_out
    for c in lines:
        R, H = rw[c]
        keep = [i for i, x in enumerate(R) if x.strip()]
        w = jiwer.wer([R[i] for i in keep], [H[i] for i in keep]) if keep else np.nan
        hw = sum(len(H[i].split()) for i in keep); rwc = sum(len(R[i].split()) for i in keep)
        dose.append({"meeting": mid, "cond": c, "target": a, "nonnative": int(m["speakers"][a].get("native") != "English"),
                     "wer": w, "words_kept": tot_t[c] / max(man_t, 1), "hyp_ref_words": hw / max(rwc, 1),
                     "segments_kept": kept[c] / max(nseg, 1), "target_word_share": tot_t[c] / max(tot_t[c] + tot_o, 1),
                     "manual_word_share": man_t / max(man_t + tot_o, 1), "coverage": len(R) / max(nseg, 1)})
json.dump(out, open(f"{ROOT}/data/dense_transcripts.json", "w"))
json.dump(segs, open(f"{ROOT}/data/dense_segments.json", "w"))
D = pd.DataFrame(dose)
D.to_csv(f"{ROOT}/results/dense_dose.csv", index=False)
print(D.groupby("cond")[["wer", "words_kept", "segments_kept", "coverage"]].mean().round(3).to_string())
