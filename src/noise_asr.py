"""P09 extension: controlled transcription noise on one speaker per meeting.

The natural experiment (manual vs Whisper transcript) only spans speaker WERs of about 14% (native)
vs 21% (non-native), and WER there is a property of the speaker. Here the dose is set by design:
for each summarised meeting one target speaker is drawn at random (seed 0); white Gaussian noise is
added to that speaker's headset audio at fixed SNRs and the segments are re-transcribed with Whisper
large-v3 (greedy as in transcribe.py; output capped at 6 tokens per second of audio + 10, which stops
hallucination loops on noisy audio). The other speakers keep the manual transcript, so only the
target's words change between conditions.

python noise_asr.py pilot                 WER of ~300 target segments at SNR 5/0/-5/-10 dB
python noise_asr.py run 0,-5,-10          all target segments (SHARD/NSHARD env split meetings)
Output: data/noise_targets.json, results/noise_asr_s<k>.jsonl
        {audio_id, meeting_id, speaker_id, begin, snr, ref, hyp}
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import io
import json
import os
import sys
import time
import zlib

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

ROOT = WORKSPACE_ROOT
MODEL = f"{WORKSPACE_ROOT}/models/whisper-large-v3"
mode = sys.argv[1]
SHARD, NSHARD = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
BATCH = int(os.environ.get("BATCH", 32))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.25)))
M = json.load(open(f"{ROOT}/data/meetings.json"))
rng = np.random.default_rng(0)
targets = {mid: str(rng.choice(sorted(M[mid]["speakers"]))) for mid in sorted(M)}
if SHARD == 0:
    json.dump(targets, open(f"{ROOT}/data/noise_targets.json", "w"))
MF = os.environ.get("MEETINGS_FILE")  # explicit meeting list (used to rebalance work between servers)
mids = json.load(open(MF)) if MF else [m for i, m in enumerate(sorted(M)) if i % NSHARD == SHARD]
tgt = {(m, M[m]["speakers"][targets[m]]["global"]) for m in mids}
norm = EnglishTextNormalizer(json.load(open(f"{MODEL}/normalizer.json")))
proc = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()


def add_noise(a, snr_db, seed):
    p = float(np.mean(a ** 2)) + 1e-10
    n = np.random.default_rng(seed).normal(0.0, np.sqrt(p / 10 ** (snr_db / 10)), a.shape)
    return (a + n).astype(np.float32)


def _gen(auds):
    # cap output at 6 tokens per second of audio (+10): fast speech needs < 5.5; longer output on noisy
    # audio is hallucination and would dominate decoding time
    mx = min(160, int(6 * max(len(a) for a in auds) / 16000) + 10)
    feats = proc.feature_extractor(auds, sampling_rate=16000, return_tensors="pt", device="cuda").input_features
    with torch.inference_mode():
        ids = model.generate(feats.cuda().half(), language="en", task="transcribe",
                             max_new_tokens=mx, num_beams=1, do_sample=False)
    return [h.strip() for h in proc.batch_decode(ids, skip_special_tokens=True)]


def transcribe(auds):
    try:
        return _gen(auds)
    except torch.cuda.OutOfMemoryError:
        pass
    torch.cuda.empty_cache()
    if len(auds) == 1:
        raise RuntimeError("out of GPU memory on a single segment")
    h = len(auds) // 2
    return transcribe(auds[:h]) + transcribe(auds[h:])


def row_groups():
    for f in sorted(glob.glob(f"{ROOT}/data/ami_hf/ihm/*.parquet")):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            meta = pf.read_row_group(rg, columns=["meeting_id", "speaker_id"]).to_pydict()
            if not any(k in tgt for k in zip(meta["meeting_id"], meta["speaker_id"])):
                continue
            yield [r for r in pf.read_row_group(rg).to_pylist() if (r["meeting_id"], r["speaker_id"]) in tgt]


def load_audio(r):
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    return (a.mean(1) if a.ndim > 1 else a)[: 30 * 16000]


def wer(refs, hyps):
    keep = [i for i, x in enumerate(refs) if norm(x).strip()]
    return jiwer.wer([norm(refs[i]) for i in keep], [norm(hyps[i]) for i in keep])


if mode == "pilot":
    rows = []
    for rs in row_groups():
        rows += [r for r in rs if len(r["text"].split()) >= 3]
        if len(rows) >= 2000:
            break
    rows = [rows[i] for i in sorted(np.random.default_rng(1).choice(len(rows), 300, replace=False))]
    auds = [load_audio(r) for r in rows]
    refs = [r["text"] for r in rows]
    clean = sum((transcribe(auds[i:i + BATCH]) for i in range(0, len(auds), BATCH)), [])
    print("clean WER", round(wer(refs, clean), 3), flush=True)
    for snr in (5, 0, -5, -10):
        na = [add_noise(a, snr, zlib.crc32(r["audio_id"].encode())) for a, r in zip(auds, rows)]
        hy = sum((transcribe(na[i:i + BATCH]) for i in range(0, len(na), BATCH)), [])
        print(f"SNR {snr} dB WER {wer(refs, hy):.3f}  e.g. {refs[5][:60]!r} -> {hy[5][:60]!r}", flush=True)
    sys.exit(0)

SNRS = [float(x) for x in sys.argv[2].split(",")]
OUT = f"{ROOT}/results/noise_asr_s{SHARD}.jsonl"
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            j = json.loads(line); done.add((j["audio_id"], j["snr"]))
        except json.JSONDecodeError:
            pass
fout = open(OUT, "a")
t0, n = time.time(), 0
for rows in row_groups():
    auds = [load_audio(r) for r in rows]
    order = np.argsort([len(a) for a in auds])
    for snr in SNRS:
        todo = [i for i in order if (rows[i]["audio_id"], snr) not in done]
        for s in range(0, len(todo), BATCH):
            idx = todo[s:s + BATCH]
            hyps = transcribe([add_noise(auds[i], snr, zlib.crc32(rows[i]["audio_id"].encode())) for i in idx])
            for i, h in zip(idx, hyps):
                r = rows[i]
                fout.write(json.dumps({"audio_id": r["audio_id"], "meeting_id": r["meeting_id"], "speaker_id": r["speaker_id"],
                                       "begin": r["begin_time"], "snr": snr, "ref": r["text"], "hyp": h}) + "\n")
            n += len(idx)
    fout.flush()
    print(f"[{time.time()-t0:.0f}s] shard {SHARD} transcribed {n}", flush=True)
print("DONE", n, flush=True)
