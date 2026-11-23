"""P09 step 4: attribute every summary sentence to the speakers whose utterances support it.

Alignment: each sentence is matched (Qwen3-Embedding) against the meeting's manual-transcript
utterances with >= 4 words; the top-3 utterances share the sentence's weight in proportion to
their similarity. The same procedure is applied to the human abstractive summaries, whose AMI
summlinks (extractive dialogue act -> abstractive sentence) give gold speakers for validation.
Outputs results/attribution.parquet, results/alignment_validation.json
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from embed import encode  # noqa: E402

ROOT = WORKSPACE_ROOT
M = json.load(open(f"{ROOT}/data/meetings.json"))
TASK = "Given a sentence from meeting minutes, retrieve the meeting utterance that supports it"
TOPK = 3

sums = []
for f in glob.glob(f"{ROOT}/results/summaries_*.jsonl"):
    model = f.split("summaries_")[1][:-6]
    for l in open(f):
        r = json.loads(l)
        sums.append({**r, "model": model})
for mid, m in M.items():
    sums.append({"meeting": mid, "cond": "manual", "prompt": "human", "model": "human",
                 "sentences": [(s["id"], s["text"]) for s in m["human_summary"]]})


def split_sents(t):
    t = re.sub(r"[*#]+", " ", t)
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    return [p.strip(" -•\t") for p in parts if len(p.strip(" -•\t").split()) >= 4]


rows, val = [], []
for mid, m in M.items():
    segs = [s for s in m["segments"] if len(s["manual"].split()) >= 4]
    if not segs:
        continue
    U = encode("qwen3-emb", [s["manual"] for s in segs], "doc", cache_key=f"p09_utt_{mid}")
    agents = np.array([s["agent"] for s in segs])
    words = {}
    for s in m["segments"]:
        words[s["agent"]] = words.get(s["agent"], 0) + len(s["manual"].split())
    tot = sum(words.values())
    for S in [x for x in sums if x["meeting"] == mid]:
        if "sentences" in S:
            sents = S["sentences"]
        else:
            sents = [(None, t) for t in split_sents(S["text"])]
        if not sents:
            continue
        Q = encode("qwen3-emb", [t for _, t in sents], "query", qwen_task=TASK)
        sim = Q.astype(np.float32) @ U.astype(np.float32).T
        share = {a: 0.0 for a in words}
        mention = {a: 0.0 for a in words}
        for i, (sid, t) in enumerate(sents):
            top = np.argsort(-sim[i])[:TOPK]
            w = np.maximum(sim[i][top], 1e-6); w = w / w.sum()
            for j, wj in zip(top, w):
                share[agents[j]] += wj
            ms = set(re.findall(r"Speaker ([A-D])", t))
            for a in ms:
                if a in mention:
                    mention[a] += 1 / len(ms)
            if S["model"] == "human" and sid in m["links"]:
                gold = set(m["links"][sid])
                val.append({"meeting": mid, "top1_in_gold": agents[top[0]] in gold,
                            "weighted_gold": float(sum(wj for j, wj in zip(top, w) if agents[j] in gold)),
                            "pm_baseline": any(m["speakers"][g]["role"] == "PM" for g in gold)})
        n = len(sents)
        mtot = sum(mention.values())
        for a in words:
            rows.append({"meeting": mid, "model": S["model"], "cond": S["cond"], "prompt": S["prompt"], "agent": a,
                         "n_sent": n, "share_sum": share[a] / n, "share_speech": words[a] / tot,
                         "ratio": (share[a] / n) / (words[a] / tot) if words[a] else np.nan,
                         "share_mention": mention[a] / mtot if mtot else np.nan})
df = pd.DataFrame(rows)
df.to_parquet(f"{ROOT}/results/attribution.parquet")
V = pd.DataFrame(val)
out = {"n_linked_sentences": len(V), "top1_speaker_in_gold": V.top1_in_gold.mean(),
       "weighted_gold_share": V.weighted_gold.mean(),
       "always_PM_baseline": V.pm_baseline.mean()}
json.dump(out, open(f"{ROOT}/results/alignment_validation.json", "w"), indent=1)
print(out)
print(df.groupby(["model", "prompt", "cond"]).ratio.describe()[["count", "mean", "std"]])
