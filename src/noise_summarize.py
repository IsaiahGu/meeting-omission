"""P09 extension step 2: minutes on transcripts where only the target speaker's lines are degraded.

Conditions per meeting (target speaker from data/noise_targets.json):
  manual_retest  the manual transcript again (a trailing newline defeats the cache): the noise floor of
                 re-summarising the same transcript
  asr_target     target's lines from the clean-audio Whisper pass (natural WER), everyone else manual
  snr<k>         target's lines from Whisper on audio with white noise at k dB SNR, everyone else manual
The "manual" reference itself is the existing minutes in results/summaries_<model>.jsonl.

python noise_summarize.py build            -> data/noise_transcripts.json
python noise_summarize.py <model alias>    -> results/noise_summaries_<model>.jsonl {meeting, target, cond, prompt, text}
   (MERGE=1 also runs the per-speaker-then-merge mitigation on every condition)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import os
import re
import sys

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")

ROOT = WORKSPACE_ROOT
arg = sys.argv[1]
P = ("Below is the transcript of a meeting. Write the meeting minutes: a concise summary (about 150-250 words) "
     "of what was discussed, the main points raised and the outcome.\n\nTranscript:\n{t}")
SPK = ("Below is the transcript of a meeting. Summarise the contributions of Speaker {a} only: the points, proposals "
       "and concerns Speaker {a} raised (2-5 sentences).\n\nTranscript:\n{t}")
MERGE = ("Here are notes on what each participant contributed to a meeting.\n\n{notes}\n\nUsing these notes, write the "
         "meeting minutes: a concise summary (about 150-250 words) of what was discussed, the main points raised and "
         "the outcome.")


def clean(t):
    t = re.sub(r"[^a-z0-9' ]+", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


if arg == "build":
    M = json.load(open(f"{ROOT}/data/meetings.json"))
    T = json.load(open(f"{ROOT}/data/noise_targets.json"))
    noisy = {}
    for p in sorted(glob.glob(f"{ROOT}/results/noise_asr_s*.jsonl")):
        for line in open(p):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            noisy.setdefault((j["meeting_id"], j["speaker_id"], round(float(j["begin"]), 2)), {}).setdefault(j["snr"], j["hyp"])
    snrs = sorted({s for v in noisy.values() for s in v}, reverse=True)
    out, cover = {}, []
    for mid, m in M.items():
        a = T[mid]
        g = m["speakers"][a]["global"]
        conds = {"asr_target": [], **{f"snr{s:g}": [] for s in snrs}}
        hit = tot = 0
        for s in m["segments"]:
            if s["agent"] != a:
                for c in conds:
                    if s["manual"]:
                        conds[c].append(f"Speaker {s['agent']}: {s['manual']}")
                continue
            tot += 1
            nz = noisy.get((mid, g, round(float(s["begin"]), 2)), {})
            hit += bool(nz)
            if s["asr"]:
                conds["asr_target"].append(f"Speaker {a}: {s['asr']}")
            for sn in snrs:
                h = clean(nz.get(sn, ""))
                if h:
                    conds[f"snr{sn:g}"].append(f"Speaker {a}: {h}")
        cover.append(hit / max(tot, 1))
        out[mid] = {"target": a, **{c: "\n".join(v) for c, v in conds.items()}}
    json.dump(out, open(f"{ROOT}/data/noise_transcripts.json", "w"))
    print("meetings", len(out), "snrs", snrs, "mean target-segment coverage", round(sum(cover) / len(cover), 4))
    sys.exit(0)

from llm import LLM  # noqa: E402

model = arg
M = json.load(open(f"{ROOT}/data/meetings.json"))
N = json.load(open(f"{ROOT}/data/noise_transcripts.json"))
llm = LLM(model, cache_name=f"p09_{model}", concurrency=None if model in ("local", "gemma-4-12b") else 16)
MT = 4 if model.startswith("gpt-6") else 1
conds = [c for c in next(iter(N.values())) if c != "target"]
jobs, keys = [], []
for mid in sorted(N):
    jobs.append([{"role": "user", "content": P.format(t=M[mid]["manual"]) + "\n"}]); keys.append((mid, "manual_retest"))
    for c in conds:
        jobs.append([{"role": "user", "content": P.format(t=N[mid][c])}]); keys.append((mid, c))
outs = llm.chat_many(jobs, max_tokens=700 * MT)
res = [{"meeting": k[0], "target": N[k[0]]["target"], "cond": k[1], "prompt": "minutes", "text": o[0]["text"],
        "error": o[0].get("finish") == "error"} for k, o in zip(keys, outs)]
if os.environ.get("MERGE") == "1":
    jobs, keys = [], []
    for mid in sorted(N):
        for c in conds:
            for a in sorted(M[mid]["speakers"]):
                jobs.append([{"role": "user", "content": SPK.format(a=a, t=N[mid][c])}]); keys.append((mid, c, a))
    outs = llm.chat_many(jobs, max_tokens=300 * MT)
    notes = {}
    for k, o in zip(keys, outs):
        notes.setdefault(k[:2], []).append(f"Speaker {k[2]}: {o[0]['text'].strip()}")
    jobs, keys = [], []
    for (mid, c), ns in notes.items():
        jobs.append([{"role": "user", "content": MERGE.format(notes="\n\n".join(ns))}]); keys.append((mid, c))
    outs = llm.chat_many(jobs, max_tokens=700 * MT)
    res += [{"meeting": k[0], "target": N[k[0]]["target"], "cond": k[1], "prompt": "per_speaker_merge", "text": o[0]["text"],
             "error": o[0].get("finish") == "error"} for k, o in zip(keys, outs)]
with open(f"{ROOT}/results/noise_summaries_{model}.jsonl", "w") as f:
    for r in res:
        f.write(json.dumps(r) + "\n")
print("wrote", len(res), "errors", sum(r["error"] for r in res))
