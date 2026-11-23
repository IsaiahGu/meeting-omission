"""P09 S5 / S7.2: LLM attribution of minutes sentences, one call per minutes text.

attribute_llm.py asked gpt-6-luna about one sentence at a time (95% top-1 agreement with the AMI summlinks).
To cover every noise condition, all sentences of one minutes text go into one call: for each sentence the
15 most similar utterances (Qwen3-Embedding, as before) are listed with speaker letters, and the judge names
the speaker(s) whose words each sentence reports. Validated first on the human abstractive summaries against
the summlinks (the same gold as before), next to the per-sentence luna result.
python attr_batch.py validate <judge>            -> results/attr_batch_validation_<judge>.json
python attr_batch.py run <judge> dense|noise|mitig -> results/attr_batch_<set>_<judge>.parquet
   (per minutes text and speaker: share of sentences attributed, split equally among named speakers)
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

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from embed import encode  # noqa: E402
from llm import LLM, parse_json  # noqa: E402

ROOT = WORKSPACE_ROOT
M = json.load(open(f"{ROOT}/data/meetings.json"))
TASK = "Given a sentence from meeting minutes, retrieve the meeting utterance that supports it"
TOPC = 15
P = ("Below are numbered sentences from the minutes of a meeting. Under each sentence are utterances from the "
     "meeting (with the speaker's letter) that may support it. For every sentence, name the speaker or speakers who "
     "said the things the sentence reports. Use only the utterances shown under that sentence.\n\n{blocks}\n\n"
     "Reply with JSON only: {{\"1\": [\"A\"], \"2\": [\"B\", \"C\"], ...}} with one entry per sentence.")


def split_sents(t):
    parts = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"[*#]+", " ", t))
    return [p.strip(" -•\t") for p in parts if len(p.strip(" -•\t").split()) >= 4]


_U = {}


def utts(mid):
    if mid not in _U:
        segs = [s for s in M[mid]["segments"] if len(s["manual"].split()) >= 4]
        _U[mid] = (segs, encode("qwen3-emb", [s["manual"] for s in segs], "doc", cache_key=f"p09_utt_{mid}").astype(np.float32))
    return _U[mid]


def build(mid, sents):
    segs, U = utts(mid)
    Q = encode("qwen3-emb", sents, "query", qwen_task=TASK).astype(np.float32)
    sim = Q @ U.T
    blocks = []
    for i, s in enumerate(sents):
        top = sorted(np.argsort(-sim[i])[:TOPC])
        u = "\n".join(f"   Speaker {segs[j]['agent']}: {segs[j]['manual']}" for j in top)
        blocks.append(f"Sentence {i + 1}: \"{s}\"\n{u}")
    return [{"role": "user", "content": P.format(blocks="\n\n".join(blocks))}]


def parse(text, n):
    j = parse_json(text) or {}
    out = []
    for i in range(n):
        v = j.get(str(i + 1), []) if isinstance(j, dict) else []
        v = v if isinstance(v, list) else [v]
        out.append(sorted({str(x).strip().upper()[-1:] for x in v if str(x).strip()} & set("ABCD")))
    return out


def kw(judge):
    return dict(max_tokens=12000) if judge.startswith(("gpt-6", "gpt-5")) else dict(max_tokens=8000)


def validate(judge):
    llm = LLM(judge, cache_name=f"p09_attrb2_{judge}", concurrency=32)
    mids = [m for m in sorted(M) if M[m]["human_summary"]]
    jobs, meta = [], []
    for mid in mids:
        S = M[mid]["human_summary"]
        jobs.append(build(mid, [s["text"] for s in S])); meta.append((mid, S))
    outs = llm.chat_many(jobs, **kw(judge))
    rows = []
    for (mid, S), o in zip(meta, outs):
        sp = parse(o[0]["text"], len(S))
        for s, x in zip(S, sp):
            gold = set(M[mid]["links"].get(s["id"], []))
            if gold:
                rows.append({"any_in_gold": bool(set(x) & gold), "exact": set(x) == gold, "precision": len(set(x) & gold) / max(1, len(x)),
                             "empty": not x})
    V = pd.DataFrame(rows)
    res = {"judge": judge, "n": len(V), **{k: float(V[k].mean()) for k in V.columns},
           "reference_per_sentence_luna": json.load(open(f"{ROOT}/results/alignment_validation_llm.json"))}
    json.dump(res, open(f"{ROOT}/results/attr_batch_validation_{judge}.json", "w"), indent=1)
    print(res)


def docs_of(which):
    pats = {"dense": ["dense_summaries_*.jsonl"], "noise": ["noise_summaries_*.jsonl"], "mitig": ["mitig_summaries_*.jsonl"],
            "natural": ["summaries_*.jsonl"]}[which]
    D = []
    for pat in pats:
        for f in glob.glob(f"{ROOT}/results/{pat}"):
            model = os.path.basename(f).split("summaries_")[1][:-6]
            if model.startswith("asr2"):
                continue
            for l in open(f):
                r = json.loads(l)
                if not r.get("error") and r["text"].strip():
                    D.append((model, r["prompt"], r["cond"], r["meeting"], r["text"]))
    if which == "dense":  # API models reuse their earlier minutes as the references (dense_summarize.py): attribute those too
        fresh = {d[0] for d in D if d[2] == "manual"}
        for model in sorted({d[0] for d in D} - fresh):
            for fn, cond in ((f"{ROOT}/results/summaries_{model}.jsonl", "manual"),
                             (f"{ROOT}/results/noise_summaries_{model}.jsonl", "manual_retest")):
                if os.path.exists(fn):
                    for l in open(fn):
                        r = json.loads(l)
                        if r["cond"] == cond and r["prompt"] == "minutes" and r["text"].strip():
                            D.append((model, "minutes", cond, r["meeting"], r["text"]))
    return D


def run(judge, which):
    llm = LLM(judge, cache_name=f"p09_attrb2_{judge}", concurrency=32)
    D = docs_of(which)
    jobs, meta = [], []
    for model, prompt, cond, mid, text in D:
        ss = split_sents(text)
        if ss:
            jobs.append(build(mid, ss)); meta.append((model, prompt, cond, mid, len(ss)))
    outs = llm.chat_many(jobs, **kw(judge))
    rows = []
    for (model, prompt, cond, mid, n), o in zip(meta, outs):
        sp = parse(o[0]["text"], n) if o[0].get("finish") != "error" else None
        if sp is None:
            continue
        share = {a: 0.0 for a in M[mid]["speakers"]}
        for x in sp:
            for a in x:
                if a in share:
                    share[a] += 1 / len(x) / n
        for a, v in share.items():
            rows.append({"model": model, "prompt": prompt, "cond": cond, "meeting": mid, "agent": a, "share_llm": v, "n_sent": n,
                         "unattributed": sum(not x for x in sp) / n})
    pd.DataFrame(rows).to_parquet(f"{ROOT}/results/attr_batch_{which}_{judge}.parquet")
    print(which, judge, "docs", len(meta), "rows", len(rows))


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(sys.argv[2])
    else:
        run(sys.argv[2], sys.argv[3])
