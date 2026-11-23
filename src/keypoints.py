"""P09 S4: key-point recall, a direct omission measure that does not depend on share attribution.

Key point = one sentence of the AMI human abstractive summary (a point the annotators judged worth
reporting), credited to every speaker whose dialogue acts the AMI summlinks connect to it. A speaker's recall
in a set of minutes = share of their key points that the minutes report (written = 1, partial = 0.5; strict:
written only). The judge sees the minutes and the numbered points of that meeting, one call per minutes text.
v1 used each speaker's linked dialogue acts as the point; its control failed (the summlinks tie fragments such
as "be like a copper hinge" to a high-level summary sentence, so "linked" did not mean "reported").
1. build    data/keypoints.json {meeting: [{kid, sid, agents, text}]}, data/kp_control.jsonl
            control (known by construction): 150 points vs their own meeting's summary (verbatim, should be
            written), 150 points vs a paraphrase of that summary by gpt-4o-mini (should be written or partial),
            300 points vs the summary of a meeting of another team and phase whose sentences are all dissimilar
            to the point (max cosine < 0.6, Qwen3-Embedding; should be not_written)
2. judge    python keypoints.py judge <judge model> control|natural|dense|mitig [frac]
3. score    python keypoints.py control <judge>   (gate: >= 0.9 on each class)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from llm import LLM, parse_json  # noqa: E402

ROOT = WORKSPACE_ROOT
JUDGE = ("Below are the minutes of a meeting and a numbered list of points from a reference summary of the same meeting. "
         "For each point, decide whether the minutes report it:\n- written: the minutes clearly report this point "
         "(wording may differ)\n- partial: the minutes report only part of it, or only vaguely\n- not_written: the minutes "
         "do not report it\n\nMinutes:\n\"\"\"\n{minutes}\n\"\"\"\n\nPoints:\n{points}\n\n"
         "Reply with JSON only: {{\"labels\": {{\"1\": \"written|partial|not_written\", ...}}}} with one entry per point.")
PARA = ("Rewrite the following meeting summary in your own words: keep every point and fact, change the wording and the "
        "sentence structure. Return only the rewritten summary.\n\n{t}")


def build():
    M = json.load(open(f"{ROOT}/data/meetings.json"))
    KP = {}
    for mid, m in M.items():
        pts = []
        for s in m["human_summary"]:
            ag = m["links"].get(s["id"], [])
            if ag and len(s["text"].split()) >= 4:
                pts.append({"kid": len(pts) + 1, "sid": s["id"], "agents": sorted(ag), "text": s["text"]})
        if pts:
            KP[mid] = pts
    json.dump(KP, open(f"{ROOT}/data/keypoints.json", "w"))
    print("meetings with key points", len(KP), "points", sum(len(v) for v in KP.values()),
          "speaker-points", sum(len(p["agents"]) for v in KP.values() for p in v))
    from embed import encode
    summ = {mid: [s["text"] for s in m["human_summary"]] for mid, m in M.items() if m["human_summary"]}
    allsent = [(mid, t) for mid, ss in summ.items() for t in ss]
    E = encode("qwen3-emb", [t for _, t in allsent], "doc").astype(np.float32)
    by_mid = defaultdict(list)
    for (mid, _), e in zip(allsent, E):
        by_mid[mid].append(e)
    by_mid = {k: np.vstack(v) for k, v in by_mid.items()}
    rng = random.Random(0)
    pts = [(mid, p) for mid, v in KP.items() for p in v]
    rng.shuffle(pts)
    para_mids = sorted({mid for mid, _ in pts[150:300]})
    paras = LLM("gpt-4o-mini", cache_name="p09_kp_para").chat_many(
        [[{"role": "user", "content": PARA.format(t=" ".join(summ[m]))}] for m in para_mids], max_tokens=1200)
    para = {m: o[0]["text"] for m, o in zip(para_mids, paras)}
    ctl = []
    for mid, p in pts[:150]:
        ctl.append({"cid": f"pos{len(ctl)}", "meeting": mid, "point": p["text"], "minutes": " ".join(summ[mid]), "gold": "written"})
    for mid, p in pts[150:300]:
        ctl.append({"cid": f"para{len(ctl)}", "meeting": mid, "point": p["text"], "minutes": para[mid], "gold": "written_or_partial"})
    emb_pt = encode("qwen3-emb", [p["text"] for _, p in pts[300:600]], "doc").astype(np.float32)
    for (mid, p), e in zip(pts[300:600], emb_pt):
        cands = [o for o in summ if o[:-1] != mid[:-1] and o[-1] != mid[-1]]
        rng.shuffle(cands)
        other = next((o for o in cands if float((by_mid[o] @ e).max()) < 0.6), None)
        if other is None:
            other = min(cands, key=lambda o: float((by_mid[o] @ e).max()))
        ctl.append({"cid": f"neg{len(ctl)}", "meeting": mid, "other": other, "point": p["text"], "minutes": " ".join(summ[other]),
                    "gold": "not_written", "max_cos": float((by_mid[other] @ e).max())})
    with open(f"{ROOT}/data/kp_control.jsonl", "w") as f:
        for c in ctl:
            f.write(json.dumps(c) + "\n")
    print("control", len(ctl))


def prompt(minutes, points):
    return [{"role": "user", "content": JUDGE.format(minutes=minutes.strip(),
                                                      points="\n".join(f"{i + 1}. {p}" for i, p in enumerate(points)))}]


def minutes_docs(which):
    docs = []
    if which == "natural":
        for f in glob.glob(f"{ROOT}/results/summaries_*.jsonl"):
            model = f.split("summaries_")[1][:-6]
            if model.startswith("asr2"):
                continue
            for l in open(f):
                r = json.loads(l)
                if r["prompt"] in ("minutes", "per_speaker_merge") and r["text"].strip():
                    docs.append(((model, r["prompt"], r["cond"], r["meeting"]), r["meeting"], r["text"]))
    else:
        pat = {"dense": "dense_summaries_*.jsonl", "mitig": "mitig_summaries_*.jsonl", "diar": "diar_summaries_*.jsonl"}[which]
        for f in glob.glob(f"{ROOT}/results/{pat}"):
            model = os.path.basename(f).split("_summaries_")[1][:-6]
            for l in open(f):
                r = json.loads(l)
                if not r.get("error") and r["text"].strip():
                    docs.append(((model, r["prompt"], r["cond"], r["meeting"]), r["meeting"], r["text"]))
        if which == "dense":
            for f in glob.glob(f"{ROOT}/results/noise_summaries_*.jsonl"):
                model = f.split("noise_summaries_")[1][:-6]
                for l in open(f):
                    r = json.loads(l)
                    if not r.get("error") and r["text"].strip() and r["prompt"] == "minutes":
                        docs.append(((model, "minutes_v1", r["cond"], r["meeting"]), r["meeting"], r["text"]))
    return docs


def kw(jm):
    return dict(max_tokens=12000) if jm.startswith(("gpt-6", "gpt-5")) else dict(max_tokens=8000)


def judge(jm, which, frac=1.0):
    llm = LLM(jm, cache_name=f"p09_kp2_{jm}", concurrency=32)
    out = f"{ROOT}/results/kp_{which}_{jm}.jsonl"
    if which == "control":
        C = [json.loads(l) for l in open(f"{ROOT}/data/kp_control.jsonl")]
        groups = defaultdict(list)
        for c in C:
            groups[c["minutes"]].append(c)
        keys = list(groups)
        outs = llm.chat_many([prompt(k, [c["point"] for c in groups[k]]) for k in keys], **kw(jm))
        with open(out, "w") as f:
            for k, o in zip(keys, outs):
                lab = (parse_json(o[0]["text"]) or {}).get("labels", {}) if o[0].get("finish") != "error" else {}
                for i, c in enumerate(groups[k]):
                    f.write(json.dumps({"cid": c["cid"], "gold": c["gold"], "label": (lab or {}).get(str(i + 1))}) + "\n")
        return
    KP = json.load(open(f"{ROOT}/data/keypoints.json"))
    docs = [d for d in minutes_docs(which) if d[1] in KP]
    if frac < 1:
        docs = [d for d in docs if random.Random(str(d[0])).random() < frac]
    outs = llm.chat_many([prompt(t, [p["text"] for p in KP[mid]]) for _, mid, t in docs], **kw(jm))
    with open(out, "w") as f:
        for (key, mid, _), o in zip(docs, outs):
            lab = (parse_json(o[0]["text"]) or {}).get("labels", {}) if o[0].get("finish") != "error" else None
            f.write(json.dumps({"model": key[0], "prompt": key[1], "cond": key[2], "meeting": mid,
                                "labels": [(lab or {}).get(str(p["kid"])) for p in KP[mid]] if isinstance(lab, dict) else None}) + "\n")
    print(jm, which, "docs", len(docs))


def control_score(jm):
    R = [json.loads(l) for l in open(f"{ROOT}/results/kp_control_{jm}.jsonl")]
    cls = {"verbatim": [r for r in R if r["gold"] == "written"], "paraphrase": [r for r in R if r["gold"] == "written_or_partial"],
           "negative": [r for r in R if r["gold"] == "not_written"]}
    res = {"judge": jm, **{f"n_{k}": len(v) for k, v in cls.items()},
           "verbatim_acc_written": sum(r["label"] == "written" for r in cls["verbatim"]) / max(1, len(cls["verbatim"])),
           "paraphrase_acc_written_or_partial": sum(r["label"] in ("written", "partial") for r in cls["paraphrase"]) / max(1, len(cls["paraphrase"])),
           "paraphrase_acc_written": sum(r["label"] == "written" for r in cls["paraphrase"]) / max(1, len(cls["paraphrase"])),
           "negative_acc": sum(r["label"] == "not_written" for r in cls["negative"]) / max(1, len(cls["negative"])),
           "unparsed": sum(r["label"] is None for r in R)}
    res["pass"] = min(res["verbatim_acc_written"], res["paraphrase_acc_written_or_partial"], res["negative_acc"]) >= 0.9
    json.dump(res, open(f"{ROOT}/results/kp_control_score_{jm}.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build":
        build()
    elif c == "judge":
        judge(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
    elif c == "control":
        control_score(sys.argv[2])
