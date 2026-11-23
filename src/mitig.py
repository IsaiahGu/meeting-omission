"""P09 S6: mitigations for a degraded speaker, on four conditions (5 dB and -5 dB white noise, far-field
microphone, Whisper tiny).

M1 flag unclear audio: target segments whose recogniser confidence (mean token log-prob) is below -1.0, the
   usual Whisper threshold, are marked [audio unclear] (kept even when nothing was recognised), and the prompt
   asks to say when a participant is hard to make out instead of leaving them out.
M2 repair, then summarise: one call corrects the marked lines from context (JSON of line number -> text,
   unchanged when it cannot tell), a second call writes the minutes from the repaired transcript.
M3 two recognisers: each target line also shows a second recogniser's output (Qwen3-ASR-1.7B on the same
   degraded audio; Qwen3-ASR-0.6B next to Whisper tiny).
M4 coverage check: the plain minutes (dense_summarize.py) and the transcript go back to the model, which lists
   each participant's main points, adds missing ones and states when someone's speech was unclear.
python mitig.py <model alias> [methods]   Output: results/mitig_summaries_<model>.jsonl
   {meeting, target, cond, prompt (M1-M4), text, calls, error}
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import os
import re
import sys

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from llm import LLM, parse_json  # noqa: E402

ROOT = WORKSPACE_ROOT
CONDS = ["snr5", "snr-5", "farfield", "whisper-tiny"]
SECOND = {"snr5": ("qwen3-asr-1.7b", "snr5"), "snr-5": ("qwen3-asr-1.7b", "snr-5"), "farfield": ("qwen3-asr-1.7b", "farfield"),
          "whisper-tiny": ("qwen3-asr-0.6b", "clean")}
THR = -1.0
P = ("Below is the transcript of a meeting. Write the meeting minutes: a concise summary (about 150-250 words) "
     "of what was discussed, the main points raised and the outcome.{extra}\n\nTranscript:\n{t}")
M1X = (" Some lines are marked [audio unclear] where the speech recognition was unreliable. If a participant's "
       "contributions are hard to make out, say so in the minutes rather than leaving that participant out.")
M3X = (" Some lines also show, in square brackets, what a second speech recogniser heard; use both versions to work "
       "out what was said.")
REPAIR = ("Below is a meeting transcript produced by speech recognition. The numbered lines marked [audio unclear] were "
          "recognised unreliably. Using the context of the whole meeting, correct the recognition errors in each numbered "
          "line: fix words that were obviously misheard, but do not add content that was not said. If you cannot tell "
          "what was said, return the line unchanged.\n\nTranscript:\n{t}\n\nReply with JSON only: "
          "{{\"<line number>\": \"<corrected line text>\", ...}} covering every numbered line.")
CHECK = ("Below are the transcript of a meeting and draft minutes of it. Check the draft against the transcript: for each "
         "participant, list to yourself their main points and check whether each is reflected in the draft; add the "
         "missing ones. If a participant's speech is unclear or garbled in the transcript, say so in the minutes instead "
         "of leaving that participant out. Return only the revised minutes (about 150-300 words).\n\nTranscript:\n{t}\n\n"
         "Draft minutes:\n{d}")


def clean(t):
    t = re.sub(r"[^a-z0-9' ]+", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


M = json.load(open(f"{ROOT}/data/meetings.json"))
T = json.load(open(f"{ROOT}/data/noise_targets.json"))
SEG = json.load(open(f"{ROOT}/data/dense_segments.json"))
second = {}
for p in glob.glob(f"{ROOT}/results/q3asr_*_s*.jsonl"):
    tag = p.split("q3asr_")[1].rsplit("_s", 1)[0]
    for l in open(p):
        try:
            j = json.loads(l)
        except json.JSONDecodeError:
            continue
        second[(tag, j["cond"], j["meeting_id"], round(float(j["begin"]), 2))] = clean(j["hyp"])


def lines_for(mid, cond, method):
    """(lines, index of numbered target lines) for one meeting and condition."""
    a = T[mid]
    hz = {round(float(b), 2): (h, lp) for b, h, lp in SEG[mid].get(cond, [])}
    out, marked = [], []
    for s in M[mid]["segments"]:
        if s["agent"] != a:
            if s["manual"]:
                out.append(f"Speaker {s['agent']}: {s['manual']}")
            continue
        h, lp = hz.get(round(float(s["begin"]), 2), ("", None))
        low = lp is not None and lp < THR
        if method in ("M1", "M2"):
            if low:
                marked.append(len(out))
                out.append(f"Speaker {a}: [audio unclear] {h}".rstrip())
            elif h:
                out.append(f"Speaker {a}: {h}")
        elif method == "M3":
            tag, c2 = SECOND[cond]
            h2 = second.get((tag, c2, mid, round(float(s["begin"]), 2)), "")
            if h or h2:
                out.append(f"Speaker {a}: {h} [second recogniser: {h2}]" if h2 and h2 != h else f"Speaker {a}: {h or h2}")
        else:
            if h:
                out.append(f"Speaker {a}: {h}")
    return out, marked


def main():
    model = sys.argv[1]
    methods = sys.argv[2].split(",") if len(sys.argv) > 2 else ["M1", "M2", "M3", "M4"]
    LOCAL = model in ("local", "gemma-4-12b") or model.startswith(("qwen3.5-", "qwen3.8-", "granite"))
    llm = LLM(model, cache_name=f"p09m_{model}", concurrency=None if LOCAL else 16)
    MT = 4 if model.startswith(("gpt-6", "gpt-5")) else 1
    plain = {}
    for l in open(f"{ROOT}/results/dense_summaries_{model}.jsonl"):
        r = json.loads(l)
        if not r.get("error"):
            plain[(r["meeting"], r["cond"])] = r["text"]
    res = []
    mids = sorted(T)
    for method in methods:
        if method in ("M1", "M3"):
            jobs, keys = [], []
            for mid in mids:
                for c in CONDS:
                    ls, _ = lines_for(mid, c, method)
                    jobs.append([{"role": "user", "content": P.format(extra=M1X if method == "M1" else M3X, t="\n".join(ls))}])
                    keys.append((mid, c))
            outs = llm.chat_many(jobs, max_tokens=700 * MT)
            res += [{"meeting": k[0], "target": T[k[0]], "cond": k[1], "prompt": method, "text": o[0]["text"], "calls": 1,
                     "error": o[0].get("finish") == "error" or not o[0]["text"].strip()} for k, o in zip(keys, outs)]
        elif method == "M2":
            jobs, keys, base = [], [], {}
            for mid in mids:
                for c in CONDS:
                    ls, marked = lines_for(mid, c, "M2")
                    numbered = [f"[{i}] {x}" if i in marked else x for i, x in enumerate(ls)]
                    base[(mid, c)] = (ls, marked)
                    if marked:
                        jobs.append([{"role": "user", "content": REPAIR.format(t="\n".join(numbered))}]); keys.append((mid, c))
            outs = llm.chat_many(jobs, max_tokens=4000 * MT)
            fixed = {}
            for k, o in zip(keys, outs):
                j = parse_json(o[0]["text"]) or {}
                fixed[k] = {int(re.sub(r"\D", "", str(i)) or -1): str(v) for i, v in (j.items() if isinstance(j, dict) else [])}
            jobs2, keys2 = [], []
            for (mid, c), (ls, marked) in base.items():
                fx = fixed.get((mid, c), {})
                a = T[mid]
                new = []
                for i, x in enumerate(ls):
                    if i in marked:
                        t = clean(fx.get(i, x.split("[audio unclear]", 1)[-1]))
                        t = re.sub(r"^speaker [a-d] ", "", t)
                        if t and t != "audio unclear":
                            new.append(f"Speaker {a}: {t}")
                    else:
                        new.append(x)
                jobs2.append([{"role": "user", "content": P.format(extra="", t="\n".join(new))}]); keys2.append((mid, c))
            outs2 = llm.chat_many(jobs2, max_tokens=700 * MT)
            res += [{"meeting": k[0], "target": T[k[0]], "cond": k[1], "prompt": "M2", "text": o[0]["text"], "calls": 2,
                     "n_repaired": len(fixed.get(k, {})), "error": o[0].get("finish") == "error" or not o[0]["text"].strip()}
                    for k, o in zip(keys2, outs2)]
        elif method == "M4":
            jobs, keys = [], []
            for mid in mids:
                for c in CONDS:
                    if (mid, c) not in plain:
                        continue
                    ls, _ = lines_for(mid, c, "M0")
                    jobs.append([{"role": "user", "content": CHECK.format(t="\n".join(ls), d=plain[(mid, c)])}]); keys.append((mid, c))
            outs = llm.chat_many(jobs, max_tokens=900 * MT)
            res += [{"meeting": k[0], "target": T[k[0]], "cond": k[1], "prompt": "M4", "text": o[0]["text"], "calls": 2,
                     "error": o[0].get("finish") == "error" or not o[0]["text"].strip()} for k, o in zip(keys, outs)]
    fn = f"{ROOT}/results/mitig_summaries_{model}.jsonl"
    prev = [json.loads(l) for l in open(fn)] if os.path.exists(fn) else []
    have = {(r["meeting"], r["cond"], r["prompt"]) for r in res}
    prev = [r for r in prev if (r["meeting"], r["cond"], r["prompt"]) not in have]
    with open(fn, "w") as f:
        for r in prev + res:
            f.write(json.dumps(r) + "\n")
    print(model, "wrote", len(res), "errors", sum(r["error"] for r in res))


if __name__ == "__main__":
    main()
