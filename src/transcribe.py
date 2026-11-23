"""Transcribe every AMI IHM segment of the summarised meetings with Whisper large-v3.

Same segmentation as the manual transcript (oracle segments + speaker labels), so the
ASR condition differs from the manual condition only in the words.
Output: results/asr_segments.jsonl  (audio_id, meeting_id, speaker_id, begin, end, ref, hyp)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = WORKSPACE_ROOT
OUT = f"{ROOT}/results/asr_segments.jsonl"
MODEL = f"{WORKSPACE_ROOT}/models/whisper-large-v3"
BATCH = int(os.environ.get("BATCH", 32))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.30)))

summ_meetings = {os.path.basename(p).split(".")[0] for p in glob.glob(f"{ROOT}/data/ami_manual/abstractive/*.abssumm.xml")}
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        done.add(json.loads(line)["audio_id"])
print("summarised meetings:", len(summ_meetings), "already done:", len(done), flush=True)

proc = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()

# collect metadata first (cheap), audio later per file
files = sorted(glob.glob(f"{ROOT}/data/ami_hf/ihm/*.parquet"))
fout = open(OUT, "a")
t0 = time.time()
n_total = 0
for fi, f in enumerate(files):
    pf = pq.ParquetFile(f)
    for rg in range(pf.num_row_groups):
        tab = pf.read_row_group(rg).to_pylist()
        rows = [r for r in tab if r["meeting_id"] in summ_meetings and r["audio_id"] not in done]
        if not rows:
            continue
        audios = []
        for r in rows:
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            assert sr == 16000, sr
            audios.append(a[: 30 * 16000])
        order = np.argsort([len(a) for a in audios])
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            feats = proc.feature_extractor([audios[i] for i in idx], sampling_rate=16000, return_tensors="pt", device="cuda").input_features
            with torch.inference_mode():
                ids = model.generate(feats.cuda().half(), language="en", task="transcribe",
                                     max_new_tokens=160, num_beams=1, do_sample=False)
            hyps = proc.batch_decode(ids, skip_special_tokens=True)
            for i, h in zip(idx, hyps):
                r = rows[i]
                fout.write(json.dumps({"audio_id": r["audio_id"], "meeting_id": r["meeting_id"],
                                       "speaker_id": r["speaker_id"], "begin": r["begin_time"],
                                       "end": r["end_time"], "ref": r["text"], "hyp": h.strip()}) + "\n")
            n_total += len(idx)
        fout.flush()
    print(f"[{time.time()-t0:.0f}s] file {fi+1}/{len(files)} segments so far {n_total}", flush=True)
fout.close()
print("DONE", n_total, flush=True)
