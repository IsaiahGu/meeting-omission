"""P09 S2/S3/S11: general minutes for every degraded-transcript condition.

python dense_summarize.py <model alias> [cond,cond,...]   (default: every condition in data/dense_transcripts.json)
FRESH_REF=1 (default for local models and new models) also writes minutes of the manual transcript and a
re-ask of it (manual_retest: trailing newline), so the reference and the conditions come from one engine
version; API models from the earlier run reuse results/summaries_<model>.jsonl (manual) and
results/noise_summaries_<model>.jsonl (manual_retest).
Output: results/dense_summaries_<model>.jsonl {meeting, target, cond, prompt, text, error}
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os
import sys

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from llm import LLM  # noqa: E402

ROOT = WORKSPACE_ROOT
P = ("Below is the transcript of a meeting. Write the meeting minutes: a concise summary (about 150-250 words) "
     "of what was discussed, the main points raised and the outcome.\n\nTranscript:\n{t}")
model = sys.argv[1]
M = json.load(open(f"{ROOT}/data/meetings.json"))
TR = os.environ.get("TRANSCRIPTS", "dense")  # dense (S2/S3) or diar (S9)
N = json.load(open(f"{ROOT}/data/{TR}_transcripts.json"))
conds = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else [c for c in next(iter(N.values())) if c != "target"]
old_api = os.path.exists(f"{ROOT}/results/summaries_{model}.jsonl") and model not in ("local", "gemma-4-12b")
fresh = os.environ.get("FRESH_REF", "0" if old_api else "1") == "1"
LOCAL = model in ("local", "gemma-4-12b") or model.startswith(("qwen3.5-", "qwen3.8-", "granite"))
llm = LLM(model, cache_name=f"p09d_{model}", concurrency=None if LOCAL else 16)
MT = 4 if model.startswith(("gpt-6", "gpt-5", "o3", "o4")) else 1
jobs, keys = [], []
for mid in sorted(N):
    if fresh:
        jobs.append([{"role": "user", "content": P.format(t=M[mid]["manual"])}]); keys.append((mid, "manual"))
        jobs.append([{"role": "user", "content": P.format(t=M[mid]["manual"]) + "\n"}]); keys.append((mid, "manual_retest"))
    for c in conds:  # asr_all = every speaker from the clean-audio Whisper pass (the natural experiment)
        jobs.append([{"role": "user", "content": P.format(t=M[mid]["asr"] if c == "asr_all" else N[mid][c])}]); keys.append((mid, c))
kw = {"max_tokens": 700 * MT}
if model.endswith("-think"):
    kw = {"max_tokens": 6000, "thinking": True}
outs = llm.chat_many(jobs, **kw)
res = []
for k, o in zip(keys, outs):
    t = (o[0]["text"] or "").split("</think>")[-1].strip()
    res.append({"meeting": k[0], "target": N[k[0]]["target"], "cond": k[1], "prompt": "minutes", "text": t,
                "error": o[0].get("finish") == "error" or not t})
fn = f"{ROOT}/results/{TR}_summaries_{model}.jsonl"
prev = []
if os.path.exists(fn):  # keep conditions summarised in earlier calls
    have = {(r["meeting"], r["cond"]) for r in res}
    prev = [json.loads(l) for l in open(fn)]
    prev = [r for r in prev if (r["meeting"], r["cond"]) not in have]
with open(fn, "w") as f:
    for r in prev + res:
        f.write(json.dumps(r) + "\n")
print(model, "wrote", len(res), "errors", sum(r["error"] for r in res), "kept from before", len(prev))
