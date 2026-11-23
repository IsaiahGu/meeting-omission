"""P09 step 4b: LLM attribution of summary sentences (gpt-6-luna), as a check on the embedding alignment.

For each summary sentence, the 15 utterances most similar to it (Qwen3-Embedding, as in attribute.py)
are shown with their speaker letters; gpt-6-luna names the speaker(s) whose words the sentence
reports. Validated on the human abstractive summaries against the AMI summlinks (gold speakers),
next to the embedding method on the same sentences.
Outputs results/attribution_llm.parquet, results/alignment_validation_llm.json
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
from llm import LLM, parse_json  # noqa: E402

ROOT = WORKSPACE_ROOT
M = json.load(open(f"{ROOT}/data/meetings.json"))
TASK = "Given a sentence from meeting minutes, retrieve the meeting utterance that supports it"
TOPC = 15
P = ("Below are utterances from a meeting (each with its speaker letter) and one sentence from the meeting's "
     "minutes. Which speaker or speakers said the things this sentence reports? Use only the utterances shown.\n\n"
     "Utterances:\n{utts}\n\nMinutes sentence: \"{s}\"\n\nReply with JSON only: {{\"speakers\": [\"A\", ...]}}")


def split_sents(t):
    t = re.sub(r"[*#]+", " ", t)
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    return [p.strip(" -•\t") for p in parts if len(p.strip(" -•\t").split()) >= 4]


sums = []
for f in glob.glob(f"{ROOT}/results/summaries_*.jsonl"):
    model = f.split("summaries_")[1][:-6]
    for l in open(f):
        r = json.loads(l); sums.append({**r, "model": model})
for mid, m in M.items():
    sums.append({"meeting": mid, "cond": "manual", "prompt": "human", "model": "human",
                 "sentences": [(s["id"], s["text"]) for s in m["human_summary"]]})
jobs, meta = [], []
for mid, m in M.items():
    segs = [s for s in m["segments"] if len(s["manual"].split()) >= 4]
    if not segs:
        continue
    U = encode("qwen3-emb", [s["manual"] for s in segs], "doc", cache_key=f"p09_utt_{mid}").astype(np.float32)
    for S in [x for x in sums if x["meeting"] == mid]:
        sents = S["sentences"] if "sentences" in S else [(None, t) for t in split_sents(S["text"])]
        if not sents:
            continue
        Q = encode("qwen3-emb", [t for _, t in sents], "query", qwen_task=TASK).astype(np.float32)
        sim = Q @ U.T
        for i, (sid, t) in enumerate(sents):
            top = sorted(np.argsort(-sim[i])[:TOPC])  # shown in meeting order
            utts = "\n".join(f"Speaker {segs[j]['agent']}: {segs[j]['manual']}" for j in top)
            jobs.append([{"role": "user", "content": P.format(utts=utts, s=t)}])
            meta.append({"meeting": mid, "model": S["model"], "cond": S["cond"], "prompt": S["prompt"], "sid": sid,
                         "n_sent": len(sents), "emb_top1": segs[int(np.argmax(sim[i]))]["agent"]})
print("sentences", len(jobs), flush=True)
llm = LLM("gpt-6-luna", cache_name="p09_attr_luna", concurrency=64)
outs = llm.chat_many(jobs, max_tokens=800)
for m_, o in zip(meta, outs):
    j = parse_json(o[0]["text"]) or {}
    sp = [str(x).strip().upper()[-1:] for x in (j.get("speakers", []) if isinstance(j, dict) else []) if str(x).strip()]
    m_["speakers"] = sorted(set(x for x in sp if x in "ABCD"))
D = pd.DataFrame(meta)
# validation on human summaries
val = []
for r in D[D.model == "human"].itertuples():
    gold = set(M[r.meeting]["links"].get(r.sid, []))
    if gold:
        val.append({"llm_any_in_gold": bool(set(r.speakers) & gold), "llm_exact": set(r.speakers) == gold,
                    "llm_precision": len(set(r.speakers) & gold) / max(1, len(r.speakers)),
                    "emb_top1_in_gold": r.emb_top1 in gold,
                    "pm_baseline": any(M[r.meeting]["speakers"][g]["role"] == "PM" for g in gold)})
V = pd.DataFrame(val)
vres = {"n": len(V), **{k: float(V[k].mean()) for k in V.columns}}
json.dump(vres, open(f"{ROOT}/results/alignment_validation_llm.json", "w"), indent=1)
print(vres, flush=True)
# shares from LLM attribution
rows = []
for (mid, model, cond, prompt), g in D.groupby(["meeting", "model", "cond", "prompt"]):
    words = {}
    for s in M[mid]["segments"]:
        words[s["agent"]] = words.get(s["agent"], 0) + len(s["manual"].split())
    tot = sum(words.values())
    share = {a: 0.0 for a in words}
    for sp in g.speakers:
        for a in sp:
            if a in share:
                share[a] += 1 / len(sp)
    n = len(g)
    for a in words:
        rows.append({"meeting": mid, "model": model, "cond": cond, "prompt": prompt, "agent": a, "n_sent": n,
                     "share_sum": share[a] / n, "share_speech": words[a] / tot,
                     "ratio": (share[a] / n) / (words[a] / tot) if words[a] else np.nan, "share_mention": np.nan})
pd.DataFrame(rows).to_parquet(f"{ROOT}/results/attribution_llm.parquet")
print("done", len(rows))
