# How Bad Must the Audio Be?
**A Speaker-Level Tolerance Test for LLM Meeting Minutes**

The experiment degrades one speaker's audio per AMI meeting, transcribes that speaker again and holds everyone else's transcript fixed. It compares the resulting minutes to minutes generated from the manual transcript.

## Inspect the paper results

[Dose-response estimates](reference/noise_effects_dense.csv) give target word error rate (`wer`), absolute change in attributed share (`d_share`), bootstrap endpoints and relative change (`rel_change`) for each model and condition. [Target assignments](reference/noise_targets.json) identify the selected speaker in each of 142 meetings.

## Prepare inputs

Use Linux, Python 3.10 or newer, CUDA-compatible PyTorch and sufficient GPU memory for Whisper large-v3. The key-point control builder calls gpt-4o-mini for paraphrases. Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` before that step. Run from the repository root:

```bash
python -m pip install -r requirements.txt
mkdir -p data/ami_manual results logs models
curl -fL https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip -o data/ami_manual.zip
unzip -q data/ami_manual.zip -d data/ami_manual
hf download --repo-type dataset edinburghcstr/ami --include 'ihm/*' 'sdm/*' --local-dir data/ami_hf
hf download openai/whisper-large-v3 --local-dir models/whisper-large-v3
hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/qwen3-emb-0.6b
cp reference/noise_targets.json data/
python src/transcribe.py
python src/build_meetings.py
python src/keypoints.py build
```

The meeting builder expects manual annotations under `data/ami_manual/abstractive/` and the individual-headset and single-distant-microphone parquet shards under `data/ami_hf/ihm/` and `data/ami_hf/sdm/`.

## Run the dose experiment

```bash
python src/degrade_asr.py snr15,snr10,snr5,snr0,snr-5,snr-10,plc10,plc30,babble10,babble0,farfield
python src/dense_build.py
FRESH_REF=1 python src/dense_summarize.py local
python src/dense_analysis.py
```

Serve a local OpenAI-compatible model named `qwen3.5-9b` at `LOCAL_LLM_URL` (default http://localhost:8000/v1). `FRESH_REF=1` generates paired manual reference minutes with the same model. Remote models use `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

For all 15 degradation conditions, also run `degrade_asr.py reverb` with room impulse responses under `data/rirs/RIRS_NOISES/real_rirs_isotropic_noises/`. Run the following before `dense_build.py` to evaluate the three smaller recognizers:

```bash
for size in tiny base small; do
  hf download openai/whisper-$size --local-dir models/whisper-$size
  TAG=whisper-$size python src/degrade_asr.py clean models/whisper-$size
done
```

`dense_analysis.py` writes `results/noise_effects_dense.csv` and `results/dose_fit.json`. Repeat the summarization stage for each model before fitting the cross-condition results. `attr_batch.py` and `keypoints.py` provide the LLM attribution and key-point judges. `mitig.py` and `mitig_analysis.py` implement and score the recovery prompts.


