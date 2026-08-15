# Hebrew TTS with S2-Pro — inference and LoRA fine-tuning

![Fish Audio S2-Pro Hebrew](https://cdn-uploads.huggingface.co/production/uploads/63453ab89ad67b3d069effdf/Kq3sPWLcMbdmMHmqZWqPs.png)

Released weights: [`notmax123/Fish-Audio-S2-Pro-He`](https://huggingface.co/notmax123/Fish-Audio-S2-Pro-He)
— a 67M-parameter LoRA adapter plus 26 atomic Hebrew-IPA tokens, on top of the
S2-Pro base weights (pulled separately, ~11GB).

Hebrew is driven by **IPA**, not nikud, and voice cloning works as it does in the
base model. Everything here also applies to adapting S2-Pro to other languages —
see [The embedding-scale bug](#the-embedding-scale-bug), which is not
Hebrew-specific.

---

## 1. Inference

```bash
bash tools/hebrew/setup_hebrew.sh     # base weights + adapter + IPA checkpoint
pip install renikud-plus              # Hebrew grapheme-to-phoneme

python tools/hebrew/infer_hebrew.py \
    --text "שלום, מה שלומך היום?" \
    --lora-checkpoint checkpoints/hebrew/hebrew_lora_step2200.safetensors \
    --output out.wav
```

Input is plain unvocalized Hebrew. `infer_hebrew.py` runs it through RenikudPlus
G2P, maps the IPA to the atomic `<ipa_*>` tokens, splits long text on sentence
boundaries, and concatenates the chunks.

**Voice cloning** — add a reference clip and its transcript:

```bash
python tools/hebrew/infer_hebrew.py \
    --text "..." --ref-audio my_voice.wav --ref-text "טקסט הייחוס" \
    --output out.wav
```

`--ref-text` defaults to a sibling `.lab` file. Reference audio of 5–15s at any
sample rate works; the codec resamples.

Useful flags:

| Flag | Effect |
|---|---|
| `--lora-scale 0.5` | Scale the adapter delta; `0.0` is the pure base model |
| `--ipa` | Input is already IPA — skip G2P |
| `--max-chars 200` | Chunk size, in IPA characters |
| `--temperature / --top-p / --top-k` | Sampling; defaults 0.7 / 0.7 / 30 |
| `--lora-config` | Must match how the adapter was trained (default `r_32_alpha_16_core`) |

The adapter also loads from the Lightning `.ckpt`, if you prefer that file.

---

## 2. Fine-tuning on your own data

```bash
# One directory per speaker: <root>/<speaker>/<utt>.wav + <utt>.lab (Hebrew text)
AUDIO_ROOT=my_audio tools/hebrew/run_hebrew_pipeline.sh
```

That runs all six steps. Individually:

```bash
tools/hebrew/run_hebrew_pipeline.sh download   # S2-Pro weights
tools/hebrew/run_hebrew_pipeline.sh ipa        # build checkpoints/s2-pro-he-ipa
AUDIO_ROOT=my_audio \
tools/hebrew/run_hebrew_pipeline.sh prepare    # G2P + dataset layout
tools/hebrew/run_hebrew_pipeline.sh extract    # VQ tokens (all GPUs, resumable)
tools/hebrew/run_hebrew_pipeline.sh pack       # protobuf shards
tools/hebrew/run_hebrew_pipeline.sh train      # LoRA
```

Every step is idempotent — rerun after an interruption and it picks up where it
stopped. Training auto-resumes from the latest checkpoint in
`results/hebrew_lora/checkpoints/`.

**Data prep entry points.** Pick the one matching your source:

| Script | Input |
|---|---|
| `prepare_from_wavs.py` | `<root>/<speaker>/*.wav` + sibling `.lab` — **use this for your own data** |
| `prepare_hebrew_data.py` | Qwen3-TTS JSONL manifests (`audio`, `text`, `orig_text`, `speaker`, `duration_sec`) |
| `prepare_from_csv.py` | AE_training_data `*_slow_filtered.csv`, with a `--wer-max` filter |

All three write the same layout — audio **symlinked** (the source corpus is never
touched, and `extract_vq.py` writes `.npy` next to the symlink) and a `.lab`
holding IPA. The conversion from raw IPA to atomic `<ipa_*>` tokens happens in
the dataset at training time, driven by `ipa_token_map` in the config, so the
`.lab` files stay human-readable.

Switching text representation only needs `prepare` + `pack` rerun — VQ codes
don't depend on text, so `extract` skips everything it has already done.

**Monitoring.** Loss is a poor checkpoint selector here; listen instead.
`SampleAudioCallback` synthesizes fixed prompts every 200 optimizer steps into
`results/hebrew_lora/samples/step_*/` and TensorBoard:

```bash
tensorboard --logdir results/hebrew_lora/tensorboard
```

Override prompts and cadence without editing the config:

```bash
python fish_speech/train.py --config-name text2semantic_hebrew_lora \
    callbacks.sample_audio.every_n_train_steps=250 \
    callbacks.sample_audio.prompt_audio=refs/spk.wav \
    callbacks.sample_audio.prompt_text="..."      # must match the audio, in IPA
```

**VRAM.** The config is sized for 2× 32GB (DDP, `batch_size=3`,
`max_length=2048`, `accumulate_grad_batches=4` → effective batch 24). If you OOM:

```bash
python fish_speech/train.py --config-name text2semantic_hebrew_lora \
    data.batch_size=1 trainer.accumulate_grad_batches=12
```

**Merging** into a single standalone checkpoint:

```bash
python tools/llama/merge_lora.py \
    --lora-config r_32_alpha_16_core \
    --base-weight checkpoints/s2-pro-he-ipa \
    --lora-weight results/hebrew_lora/checkpoints/step_000002200.ckpt \
    --output checkpoints/s2-pro-hebrew
```

Note `--base-weight` is the **IPA** checkpoint, not plain `s2-pro` — the merged
model needs the extended tokenizer, the trained `ipa_embeddings` table and the
`ipa_token_map.json` that `merge_lora.py` copies across.

---

## The embedding-scale bug

**This is the reason earlier fine-tunes collapsed, and it is not Hebrew-specific.**

S2-Pro sets `scale_codebook_embeddings=True`. At inference, `forward_generate()`
divides semantic-position embeddings by `sqrt(num_codebooks + 1)` = 3.317. The
training path in `embed()` did **not**. Every fine-tune therefore learned against
embeddings 3.3× larger than the ones it would see at generation time.

The symptom is distinctive: teacher-forced CE looks healthy and keeps improving
while free-running generation produces the first word and then fades to silence.
That matches unresolved upstream reports —
[#1136](https://github.com/fishaudio/fish-speech/issues/1136) (Japanese gibberish),
[#682](https://github.com/fishaudio/fish-speech/issues/682) (Hindi noise),
[#814](https://github.com/fishaudio/fish-speech/issues/814).

Five Hebrew runs collapsed this way before it was found. After the fix in
`llama.py` (train and inference embeddings verified bit-identical):

| | sample RMS | energy decay over the utterance |
|---|---|---|
| before | 0.008 – 0.022 | 0.07× |
| after | 0.171 – 0.205 | 1.02× |
| base model | 0.181 | — |

---

## Design decisions

- **Base model: `fishaudio/s2-pro`.** `openaudio-s1-mini` ships only a
  `tokenizer.tiktoken` the AutoTokenizer-based `FishTokenizer` cannot load, and
  S1 never supported Hebrew. S2-Pro has Hebrew in its pretraining set and ships
  an HF-format tokenizer. Upstream warns against fine-tuning RL-aligned models —
  mitigate by preferring the earliest checkpoint that sounds good.
- **Text: IPA, not nikud.** IPA removes the nikud frontend from the inference
  path entirely — one phonemizer, used at both train and inference time, so the
  two cannot disagree. It also recovers speakers whose corpora are phonemized at
  source and have no Hebrew-script transcript at all. Nikud is still supported
  (`--text-repr nikud`). Whichever you pick, training data, sampling prompts and
  inference text must all use it.
- **Atomic IPA tokens.** S2-Pro's BPE splits IPA into pieces that collide with
  English orthography — Hebrew `י`, phonemized `j`, was read as the English
  letter *jay*. Each of the 26 Hebrew IPA symbols therefore gets a dedicated
  input-only token (`<ipa_j>`, `<ipa_u0283>`, …) in a separate trainable
  `nn.Embedding`, initialized to the mean of the symbol's original BPE-piece
  embeddings. The output vocabulary is untouched: these tokens are read, never
  predicted. `build_ipa_checkpoint.py` extends the tokenizer 155,774 → 155,800
  and precomputes the table.
- **Training prompts match inference exactly.** `InferenceMatchedIterableDataset`
  builds the same `generate_long()` prompt the inference path builds, with a
  same-speaker voice-cloning reference 80% of the time (`ref_prob`). Training on
  a different prompt shape than you generate with is its own quiet source of
  degradation.
- **LoRA `r_32_alpha_16_core`** — r=32, α=16, dropout 0.05, `attention` + `mlp`,
  66.9M trainable params: 60.2M slow transformer, 6.7M fast, 0.07M IPA
  embeddings. Frozen are the direct interfaces to codebook space —
  `fast_embeddings`, `fast_output`, and the tied slow embeddings/output — which
  is what keeps timbre and cloning close to base.

  The upstream default `r_8_alpha_16` is *wrong for S2-Pro*: it targets
  `embeddings`, but S2-Pro ties word embeddings to the output head
  (`F.linear(x, embeddings.weight)`), and loralib only applies the delta in the
  lookup path during training while merging it into the weight at eval — so the
  output head would differ between training and inference. Its `output` target
  is a silent no-op when tied.
- **`semantic_loss_weight: 0.3`** down-weights the residual-codebook loss
  (Qwen3-TTS uses 0.3 for its equivalent sub-talker term). At 1.0 it dominated
  the gradient (semantic ~4.4 vs base ~2.8) and pulled training away from the
  text→semantic mapping that language adaptation actually needs.
- **U+05AF (masora circle) is stripped.** renikud marks silent letters with it;
  it is not standard nikud and the base model never saw it. `--keep-masora`
  keeps it — then you must emit it at inference time too.
- **VQ codes from other codecs are not reusable.** Everything is re-encoded with
  the S2-Pro DAC codec.
- **Gradient-checkpointing fix (`llama.py`).** `use_reentrant=True` silently
  dropped all slow-transformer LoRA gradients when embeddings are frozen (the
  checkpointed block's input doesn't require grad). Changed to
  `use_reentrant=False`; verified 202/202 expected params receive gradients.

---

## Known limitations of the released adapter

- **Undertrained, stopped by hand** at 2,200 optimizer steps ≈ 53k utterances,
  under 20% of one epoch. Val loss was still improving monotonically at every
  checkpoint (2.934 → 2.844 → 2.820 → 2.807).
- **α/r = 0.5 is a leftover workaround.** It was chosen empirically because α=64
  destroyed generation — which turned out to be the embedding-scale bug, not the
  LoRA strength. Standard α = 2r was never re-tried post-fix and may be better.
- **Emotion tags (`[whisper]`, `[excited]`, …) do not work** — and this is not a
  fine-tuning regression. Measured on the *base* model in *English*: plain /
  whisper / shouting gave RMS 0.0655 / 0.0652 / 0.0689, i.e. no response. The
  released S2-Pro weights lack the tag alignment.
  `generate_tagged_data.py` implements a self-distillation attempt at teaching
  them; it did not work and is kept only as tooling.
- **Pitch is not cloned.** Timbre transfers well; F0 does not (base mean |err|
  23 Hz, LoRA 20 Hz). The adapter homogenizes pitch somewhat — spread across
  speakers drops 66 Hz → 26 Hz. Putting LoRA on the fast transformer is the
  likely cause; freezing `fast_layers` entirely is the obvious next experiment.

## Files

| Path | Purpose |
|---|---|
| `setup_hebrew.sh` | One-command setup for inference |
| `run_hebrew_pipeline.sh` | Six-step fine-tuning pipeline |
| `build_ipa_checkpoint.py` | Extends S2-Pro with the 26 atomic IPA tokens |
| `prepare_from_wavs.py` | Dataset from your own audio + transcripts |
| `prepare_hebrew_data.py` | Dataset from Qwen3-TTS JSONL manifests |
| `prepare_from_csv.py` | Dataset from AE_training_data CSVs |
| `infer_hebrew.py` | Hebrew → G2P → IPA → audio, with optional cloning |
| `generate_tagged_data.py` | Emotion-tag self-distillation (did not work) |

Config: `fish_speech/configs/text2semantic_hebrew_lora.yaml`,
LoRA: `fish_speech/configs/lora/r_32_alpha_16_core.yaml`,
IPA tokens: `fish_speech/text/ipa_tokens.py`.
