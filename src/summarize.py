"""P09 step 3: LLM meeting minutes for every meeting x transcript condition x prompt x model,
plus the per-speaker mitigation (summarise each speaker's contributions, then merge).

python summarize.py <model alias: local|gemma-4-12b|gpt-4o-mini>
Output: results/summaries_<model>.jsonl {meeting, cond, prompt, text}
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import sys

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from llm import LLM  # noqa: E402

ROOT = WORKSPACE_ROOT
model = sys.argv[1] if len(sys.argv) > 1 else "local"
M = json.load(open(f"{ROOT}/data/meetings.json"))
P = {
    "minutes": "Below is the transcript of a meeting. Write the meeting minutes: a concise summary (about 150-250 words) "
               "of what was discussed, the main points raised and the outcome.\n\nTranscript:\n{t}",
    "decisions": "Below is the transcript of a meeting. List the decisions that were made and the action items "
                 "(who will do what), as bullet points.\n\nTranscript:\n{t}",
}
SPK = ("Below is the transcript of a meeting. Summarise the contributions of Speaker {a} only: the points, proposals "
       "and concerns Speaker {a} raised (2-5 sentences).\n\nTranscript:\n{t}")
MERGE = ("Here are notes on what each participant contributed to a meeting.\n\n{notes}\n\nUsing these notes, write the "
         "meeting minutes: a concise summary (about 150-250 words) of what was discussed, the main points raised and "
         "the outcome.")
llm = LLM(model, cache_name=f"p09_{model}", concurrency=None if model in ("local", "gemma-4-12b") else 16)
MT = 4 if model.startswith("gpt-6") else 1  # reasoning models need room for hidden tokens
jobs, keys = [], []
for mid, m in M.items():
    for cond in ("manual", "asr"):
        for pn, pt in P.items():
            jobs.append([{"role": "user", "content": pt.format(t=m[cond])}]); keys.append((mid, cond, pn))
outs = llm.chat_many(jobs, max_tokens=700 * MT)
res = [{"meeting": k[0], "cond": k[1], "prompt": k[2], "text": o[0]["text"]} for k, o in zip(keys, outs)]
# mitigation: per-speaker notes, then merge
jobs, keys = [], []
for mid, m in M.items():
    for cond in ("manual", "asr"):
        for a in sorted(m["speakers"]):
            jobs.append([{"role": "user", "content": SPK.format(a=a, t=m[cond])}]); keys.append((mid, cond, a))
outs = llm.chat_many(jobs, max_tokens=300 * MT)
notes = {}
for k, o in zip(keys, outs):
    notes.setdefault(k[:2], []).append(f"Speaker {k[2]}: {o[0]['text'].strip()}")
jobs, keys = [], []
for (mid, cond), ns in notes.items():
    jobs.append([{"role": "user", "content": MERGE.format(notes="\n\n".join(ns))}]); keys.append((mid, cond))
outs = llm.chat_many(jobs, max_tokens=700 * MT)
res += [{"meeting": k[0], "cond": k[1], "prompt": "per_speaker_merge", "text": o[0]["text"]} for k, o in zip(keys, outs)]
with open(f"{ROOT}/results/summaries_{model}.jsonl", "w") as f:
    for r in res:
        f.write(json.dumps(r) + "\n")
print("wrote", len(res))
