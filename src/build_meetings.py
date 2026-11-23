"""P09 step 2: meetings, speakers, transcripts (manual vs ASR, identical segmentation), WER.

Both transcript conditions are lower-cased and stripped of punctuation, so they differ only in
the words. Speakers appear as neutral letters (Speaker A-D); roles are not shown to the model.
Outputs data/meetings.json  {meeting: {speakers: {...}, manual: str, asr: str, human_summary: [...],
                              links: {sentence_id: [agents]}}}
         results/speaker_wer.csv
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import jiwer
import pandas as pd
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

ROOT = WORKSPACE_ROOT
A = f"{ROOT}/data/ami_manual"
NS = "{http://nite.sourceforge.net/}"
norm = EnglishTextNormalizer(json.load(open(f"{WORKSPACE_ROOT}/models/whisper-large-v3/normalizer.json")))

# ---- metadata
parts = {}
for p in ET.parse(f"{A}/corpusResources/participants.xml").getroot():
    if p.tag.endswith("participant"):
        parts[p.get(f"{NS}id")] = {"sex": p.get("sex"), "age": float(p.get("age_at_collection") or "nan"),
                                    "native": p.get("native_language")}
meet = {}
for m in ET.parse(f"{A}/corpusResources/meetings.xml").getroot():
    if not m.tag.endswith("meeting"):
        continue
    sp = {}
    for s in m:
        if s.tag.endswith("speaker"):
            g = s.get("global_name")
            sp[s.get("nxt_agent")] = {"global": g, "role": s.get("role"), **parts.get(g, {})}
    meet[m.get("observation")] = sp

# ---- human abstractive summaries + summlinks (extractive DA -> abstractive sentence)
def read_abs(mid):
    r = ET.parse(f"{A}/abstractive/{mid}.abssumm.xml").getroot()
    out = []
    for sec in r:
        tag = sec.tag.split("}")[-1]
        for s in sec.iter():
            if s.tag.endswith("sentence") and (s.text or "").strip():
                out.append({"id": s.get(f"{NS}id"), "section": tag, "text": s.text.strip()})
    return out


def read_links(mid):
    p = f"{A}/extractive/{mid}.summlink.xml"
    links = defaultdict(set)
    if not os.path.exists(p):
        return {}
    for sl in ET.parse(p).getroot().iter():
        if not sl.tag.endswith("summlink"):
            continue
        ext = abs_ = None
        for ptr in sl:
            href = ptr.get("href", "")
            if ptr.get("role") == "extractive":
                mm = re.match(rf"{mid}\.([A-Z])\.dialog-act", href)
                ext = mm.group(1) if mm else None
            elif ptr.get("role") == "abstractive":
                mm = re.search(r"id\(([^)]+)\)", href)
                abs_ = mm.group(1) if mm else None
        if ext and abs_:
            links[abs_].add(ext)
    return {k: sorted(v) for k, v in links.items()}


segs = pd.read_json(f"{ROOT}/results/asr_segments.jsonl", lines=True)
segs = segs.drop_duplicates("audio_id").sort_values(["meeting_id", "begin"])


def clean(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


meetings, wer_rows = {}, []
for mid, g in segs.groupby("meeting_id"):
    if mid not in meet or not os.path.exists(f"{A}/abstractive/{mid}.abssumm.xml"):
        continue
    g2a = {v["global"]: k for k, v in meet[mid].items()}
    g = g[g.speaker_id.isin(g2a)]
    lines_m, lines_a, segrows = [], [], []
    for r in g.itertuples():
        ag = g2a[r.speaker_id]
        rm, ra = clean(r.ref), clean(r.hyp)
        if rm:
            lines_m.append(f"Speaker {ag}: {rm}")
        if ra:
            lines_a.append(f"Speaker {ag}: {ra}")
        segrows.append({"agent": ag, "begin": r.begin, "manual": rm, "asr": ra})
    # per-speaker WER on Whisper-normalised text
    for ag, sg in g.groupby(g.speaker_id.map(g2a)):
        refs = [norm(x) for x in sg.ref]
        hyps = [norm(x) for x in sg.hyp]
        keep = [i for i, x in enumerate(refs) if x.strip()]
        refs = [refs[i] for i in keep]; hyps = [hyps[i] for i in keep]
        w = jiwer.process_words(refs, hyps)
        n_ref = sum(len(x.split()) for x in refs)
        wer_rows.append({"meeting": mid, "agent": ag, **meet[mid][ag], "wer": w.wer, "ref_words": n_ref,
                         "subs": w.substitutions, "dels": w.deletions, "ins": w.insertions})
    meetings[mid] = {"speakers": meet[mid], "manual": "\n".join(lines_m), "asr": "\n".join(lines_a),
                     "segments": segrows, "human_summary": read_abs(mid), "links": read_links(mid)}
json.dump(meetings, open(f"{ROOT}/data/meetings.json", "w"))
W = pd.DataFrame(wer_rows)
W.to_csv(f"{ROOT}/results/speaker_wer.csv", index=False)
print("meetings", len(meetings), "speakers", len(W))
print(W.groupby(W.native.eq("English")).wer.describe())
print("words per meeting (manual):", pd.Series([len(m["manual"].split()) for m in meetings.values()]).describe().round(0).to_dict())
