"""P09 S2/S3: re-transcribe the target speaker under denser white-noise doses and realistic degradations.

Same design as noise_asr.py: one random target speaker per meeting (data/noise_targets.json), only the
target's segments are degraded, everyone else keeps the manual transcript. Every hypothesis carries the
recogniser's mean token log-probability (segment confidence, used by mitigation M1).
Conditions (all on the target's segments):
  snr15 snr10 snr5 snr0 snr-5 snr-10   white Gaussian noise scaled to the segment energy (seed crc32(audio_id))
  plc10 plc30        bursty packet loss: 20 ms frames, Gilbert-Elliott loss with mean burst 3 frames, lost frames zeroed
  reverb             convolution with a real room impulse response (OpenSLR 28 real_rirs_isotropic_noises)
  babble10 babble0   multi-talker babble: 4 headset segments of other meetings summed, at 10 / 0 dB SNR
  farfield           the same segment from the AMI single distant microphone (sdm config, aligned by
                     meeting, speaker and start time)
  tiny base small    clean headset audio, smaller Whisper models (no change to the audio)
Recogniser: Whisper large-v3 greedy (except tiny/base/small), output capped at 6 tokens per second + 10.
python degrade_asr.py <comma-separated conditions> [whisper model dir]   (SHARD/NSHARD split meetings)
Output: results/degrade_asr_<tag>_s<k>.jsonl {audio_id, meeting_id, speaker_id, begin, cond, ref, hyp, avg_logprob, n_tok}
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

os.environ.setdefault("OMP_NUM_THREADS", "3")  # the container has a ~27-core quota shared with vLLM
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = WORKSPACE_ROOT
CONDS = sys.argv[1].split(",")
MODEL = sys.argv[2] if len(sys.argv) > 2 else f"{WORKSPACE_ROOT}/models/whisper-large-v3"
TAG = os.environ.get("TAG", os.path.basename(MODEL))
SHARD, NSHARD = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
BATCH = int(os.environ.get("BATCH", 32))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.3)))
torch.set_num_threads(3)
M = json.load(open(f"{ROOT}/data/meetings.json"))
T = json.load(open(f"{ROOT}/data/noise_targets.json"))
MF = os.environ.get("MEETINGS_FILE")  # explicit meeting list: lets work be re-split between GPUs mid-run
mids = json.load(open(MF)) if MF else [m for i, m in enumerate(sorted(T)) if i % NSHARD == SHARD]
tgt = {(m, M[m]["speakers"][T[m]]["global"]) for m in mids}
proc = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
EOS = proc.tokenizer.eos_token_id


def load_audio(r):
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    return (a.mean(1) if a.ndim > 1 else a)[: 30 * 16000]


def white(a, snr_db, seed):
    p = float(np.mean(a ** 2)) + 1e-10
    return (a + np.random.default_rng(seed).normal(0.0, np.sqrt(p / 10 ** (snr_db / 10)), a.shape)).astype(np.float32)


def plc(a, rate, seed, frame=320, burst=3.0):
    r = 1.0 / burst
    p = r * rate / (1 - rate)
    rng = np.random.default_rng(seed)
    out, bad = a.copy(), rng.random() < rate
    for s in range(0, len(a), frame):
        bad = (rng.random() >= r) if bad else (rng.random() < p)
        if bad:
            out[s:s + frame] = 0.0
    return out


_rirs = None


def reverb(a, seed):
    global _rirs
    if _rirs is None:
        fs = sorted(glob.glob(f"{WORKSPACE_ROOT}/data/rirs/RIRS_NOISES/real_rirs_isotropic_noises/*rir*.wav"))
        _rirs = []
        for f in fs:
            x, sr = sf.read(f, dtype="float32")
            x = x[:, 0] if x.ndim > 1 else x
            if sr == 16000 and len(x) > 800:
                _rirs.append(x / (np.abs(x).max() + 1e-9))
        print("RIRs", len(_rirs), flush=True)
    from scipy.signal import fftconvolve
    h = _rirs[np.random.default_rng(seed).integers(len(_rirs))]
    y = fftconvolve(a, h)[: len(a)]
    return (y * np.sqrt((np.mean(a ** 2) + 1e-10) / (np.mean(y ** 2) + 1e-10))).astype(np.float32)


_babble = None


def babble(a, snr_db, seed, mid):
    global _babble
    if _babble is None:  # pool of headset segments >= 2 s from all meetings, loaded once
        _babble = []
        rng = np.random.default_rng(123)
        files = sorted(glob.glob(f"{ROOT}/data/ami_hf/ihm/train-*.parquet"))
        for f in rng.choice(files, min(6, len(files)), replace=False):
            pf = pq.ParquetFile(f)
            for rg in rng.choice(pf.num_row_groups, min(3, pf.num_row_groups), replace=False):
                for r in pf.read_row_group(int(rg)).to_pylist():
                    if r["end_time"] - r["begin_time"] >= 2.0:
                        _babble.append((r["meeting_id"], load_audio(r)))
        print("babble pool", len(_babble), flush=True)
    rng = np.random.default_rng(seed)
    pool = [x for m, x in _babble if m != mid]
    n = np.zeros(len(a), np.float32)
    for k in rng.choice(len(pool), 4, replace=False):
        x = pool[k]
        x = np.tile(x, int(np.ceil(len(a) / len(x))) + 1)
        off = rng.integers(0, len(x) - len(a) + 1)
        x = x[off:off + len(a)]
        n += x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
    pa, pn = float(np.mean(a ** 2)) + 1e-10, float(np.mean(n ** 2)) + 1e-10
    return (a + n * np.sqrt(pa / pn / 10 ** (snr_db / 10))).astype(np.float32)


def _gen(auds):
    mx = min(160, int(6 * max(len(a) for a in auds) / 16000) + 10)
    feats = proc.feature_extractor(auds, sampling_rate=16000, return_tensors="pt", device="cuda").input_features
    with torch.inference_mode():
        out = model.generate(feats.cuda().half(), language="en", task="transcribe", max_new_tokens=mx, num_beams=1,
                             do_sample=False, return_dict_in_generate=True, output_scores=True)
        ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float()
    gen = out.sequences[:, -ts.shape[1]:]
    mask = (gen < EOS) & torch.isfinite(ts)  # text tokens only (special tokens start at <|endoftext|>)
    n = mask.sum(1)
    lp = torch.where(mask, ts, torch.zeros_like(ts)).sum(1) / n.clamp(min=1)
    txt = [h.strip() for h in proc.batch_decode(out.sequences, skip_special_tokens=True)]
    return list(zip(txt, lp.cpu().tolist(), n.cpu().tolist()))


def transcribe(auds):
    try:
        return _gen(auds)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    if len(auds) == 1:
        raise RuntimeError("out of GPU memory on a single segment")
    h = len(auds) // 2
    return transcribe(auds[:h]) + transcribe(auds[h:])


def rows_of(cfg, n=600):
    """Target rows of an AMI config (ihm or sdm), in chunks of about n rows (fuller GPU batches)."""
    buf = []
    for f in sorted(glob.glob(f"{ROOT}/data/ami_hf/{cfg}/*.parquet")):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            meta = pf.read_row_group(rg, columns=["meeting_id", "speaker_id"]).to_pydict()
            if any(k in tgt for k in zip(meta["meeting_id"], meta["speaker_id"])):
                buf += [r for r in pf.read_row_group(rg).to_pylist() if (r["meeting_id"], r["speaker_id"]) in tgt]
                if len(buf) >= n:
                    yield buf
                    buf = []
    if buf:
        yield buf


def degrade(cond, a, r):
    seed = zlib.crc32(r["audio_id"].encode())
    if cond.startswith("snr"):
        return white(a, float(cond[3:]), seed)
    if cond.startswith("plc"):
        return plc(a, int(cond[3:]) / 100, seed)
    if cond == "reverb":
        return reverb(a, seed)
    if cond.startswith("babble"):
        return babble(a, float(cond[6:]), seed, r["meeting_id"])
    return a  # clean (tiny / base / small) and farfield (already the distant-microphone audio)


OUT = f"{ROOT}/results/degrade_asr_{TAG}_s{os.environ.get('OUTSHARD', SHARD)}.jsonl"
done = set()
for p in glob.glob(f"{ROOT}/results/degrade_asr_{TAG}_s*.jsonl"):  # every shard file, so work can be re-split
    for line in open(p):
        try:
            j = json.loads(line); done.add((j["audio_id"], j["cond"]))
        except json.JSONDecodeError:
            pass
fout = open(OUT, "a")
t0, n = time.time(), 0
groups = []
if any(c != "farfield" for c in CONDS):
    groups.append(("ihm", [c for c in CONDS if c != "farfield"]))
if "farfield" in CONDS:
    groups.append(("sdm", ["farfield"]))
for cfg, conds in groups:
    for rows in rows_of(cfg):
        auds = [load_audio(r) for r in rows]
        order = np.argsort([len(a) for a in auds])
        for cond in conds:
            # conditions join on (meeting_id, speaker_id, begin); audio_id is the source recording's id
            key = [r["audio_id"] for r in rows]
            todo = [i for i in order if (key[i], cond) not in done]
            for s in range(0, len(todo), BATCH):
                idx = todo[s:s + BATCH]
                hyps = transcribe([degrade(cond, auds[i], rows[i]) for i in idx])
                for i, (h, lp, nt) in zip(idx, hyps):
                    r = rows[i]
                    fout.write(json.dumps({"audio_id": key[i], "meeting_id": r["meeting_id"], "speaker_id": r["speaker_id"],
                                           "begin": round(float(r["begin_time"]), 2), "cond": cond, "ref": r["text"], "hyp": h,
                                           "avg_logprob": lp, "n_tok": nt}) + "\n")
                n += len(idx)
        fout.flush()
        print(f"[{time.time() - t0:.0f}s] {TAG} shard {SHARD} {cfg} transcribed {n}", flush=True)
print("DONE", n, flush=True)
