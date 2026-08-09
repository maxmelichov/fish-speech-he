# Hebrew LoRA training

LoRA fine-tuning of **S2-Pro** on the Hebrew data prepared in the Qwen3-TTS repo
(`/mnt/windows_nvme/Qwen3-TTS`): 132,169 utterances / ~165h across 12 speakers,
using the IPA transcriptions.

Switching the text representation only requires rerunning `prepare` and `pack`
— the VQ codes in the `.npy` files do not depend on the text, so `extract` only
processes files it has not seen before.

## Design decisions

- **Base model: `fishaudio/s2-pro`.** The older `openaudio-s1-mini` ships only a
  `tokenizer.tiktoken` file the current `FishTokenizer` (AutoTokenizer-based)
  cannot load, and S1 never supported Hebrew. S2-Pro has Hebrew in its
  pretraining set and ships an HF-format tokenizer. Note the upstream docs warn
  against fine-tuning RL-aligned models — mitigate by preferring the earliest
  checkpoint that sounds good.
- **Text: IPA** (`text` from the manifests — the same representation used for
  Qwen3-TTS), selected with `--text-repr ipa` (the default). Nikud
  (`--text-repr nikud`, the manifests' `orig_text`) is also supported and was
  the original choice here, on the theory that S2-Pro's BPE saw Hebrew script
  in pretraining while IPA is out of distribution.

  IPA won for two reasons. It removes the nikud frontend from the inference
  path entirely — whatever phonemizer produced the manifests is the only
  frontend, so training and inference cannot disagree. And it recovers the
  `female1` / `male1` speakers, which have **no** `orig_text` at all
  (Knesset-derived, phonemized at source): 131,569 train rows / ~165h /
  12 speakers with IPA, versus 107,669 / ~135h / 10 speakers with nikud.

  **Whatever you pick, the sampling prompts in the training config and any
  inference-time text must use the same representation.**
- **U+05AF (masora circle) is stripped** from the text by default. renikud uses
  it to mark silent letters; it is not standard nikud and the base model never
  saw it. Pass `--keep-masora` to `prepare_hebrew_data.py` to keep it (then you
  must also emit it at inference time).
- **Audio is symlinked, not copied.** `extract_vq.py` writes each `.npy` next
  to the symlink (inside `data/hebrew/`), so the source dataset is untouched.
- **VQ codes from Qwen3-TTS (`audio_codes`) are not reusable** — different
  codec. Everything is re-encoded with the S2-Pro DAC codec.
- **LoRA recipe: `r_32_alpha_64_slow`** (r=32, α=64, dropout 0.05, ~60M
  trainable params = 1.30% of the model), targeting **only the slow
  transformer's** attention and MLP (`slow_attention` / `slow_mlp` — target
  names added in `lora.py` for this; unprefixed names hit both stacks). The
  fast (residual codebook / acoustic) transformer is left entirely frozen: it
  carries timbre and voice-cloning behaviour, and adapting it on a 12-speaker
  corpus trades away the base model's audio quality and cloning generality for
  nothing that Hebrew pronunciation needs. The upstream default `r_8_alpha_16`
  is additionally wrong for S2-Pro: it targets `embeddings`, but S2-Pro ties
  word embeddings to the output head (`F.linear(x, embeddings.weight)`), and
  loralib only applies the LoRA delta in the lookup path during training while
  merging it into the weight on eval — so the output head would differ between
  training and inference. Its `output` target is a silent no-op when tied.
- **Gradient-checkpointing fix (`llama.py`)**: `use_reentrant=True` silently
  dropped all slow-transformer LoRA gradients when embeddings are frozen (the
  checkpointed block's input doesn't require grad). Changed to
  `use_reentrant=False`; verified 202/202 expected params receive gradients.

## Run

```bash
tools/hebrew/run_hebrew_pipeline.sh            # all steps
tools/hebrew/run_hebrew_pipeline.sh extract    # or one step:
                                               # download|prepare|extract|pack|train
```

Steps: download weights → build `data/hebrew/{train,eval}/<spk>/*.{wav,lab}` →
extract VQ tokens (sharded over both GPUs, resumable) → pack protobufs →
train with `fish_speech/configs/text2semantic_hebrew_lora.yaml`.

Training auto-resumes from the latest checkpoint in
`results/hebrew_lora/checkpoints/`. Checkpoints contain **only LoRA weights**
(small). Monitor with:

```bash
tensorboard --logdir results/hebrew_lora/tensorboard
```

## Audio sampling during training

`SampleAudioCallback` (`fish_speech/callbacks/sample_audio.py`) synthesizes a
fixed set of Hebrew prompts every `callbacks.sample_audio.every_n_train_steps`
optimizer steps, logging to TensorBoard (`samples/`) and to
`results/hebrew_lora/samples/step_*/`. Listening to these is the only reliable
way to pick a checkpoint — loss keeps dropping well past the point where the
base model's prosody starts degrading.

```bash
# change prompts / cadence without editing the config
python fish_speech/train.py --config-name text2semantic_hebrew_lora \
    callbacks.sample_audio.every_n_train_steps=250 \
    callbacks.sample_audio.prompt_audio=/mnt/windows_nvme/Qwen3-TTS/data/refs/hebrew/male2.wav \
    callbacks.sample_audio.prompt_text="..."          # must match the audio
python fish_speech/train.py --config-name text2semantic_hebrew_lora \
    callbacks.sample_audio.enabled=false              # disable entirely
```

Two details it has to get right, both verified by test:

- It **tears down the KV caches** that `generate()` allocates. The training
  forward pass calls attention blocks without `input_pos`, so leaving a cache
  attached crashes the next training step.
- It puts the model in eval *semantics* (dropout off) **without** letting
  `loralib` merge the LoRA delta into the base weights, because that
  merge/unmerge round trip is lossy in `bf16-true`. Pass `merge_lora=true` for
  the conventional `eval()`/`train()` behaviour.

Sampling runs on rank 0 only; the other rank blocks at the next allreduce until
it finishes, so keep `max_new_tokens` well inside the NCCL timeout. Note also
that `model.max_length` (4096) is deliberately larger than the training
sequence length (2048): `generate_long` rejects prompts longer than
`max_seq_len - 2048`, so sampling needs the headroom.

## VRAM tuning (2x RTX 5090, 32GB)

The config runs DDP on both GPUs, `batch_size=2`, `max_length=2048`,
`accumulate_grad_batches=4` (effective batch 16). If you OOM:

```bash
python fish_speech/train.py --config-name text2semantic_hebrew_lora \
    data.batch_size=1 trainer.accumulate_grad_batches=8
```

Other useful overrides:

- `lora@model.model.lora_config=r_32_alpha_16_fast` — LoRA on the fast
  (acoustic) transformer only; useful for timbre-only tweaks, not language
  adaptation. Note it targets `fast_embeddings`, so it doesn't depend on the
  embeddings-tying caveat above.
- `trainer.max_steps=...`, `model.optimizer.lr=...`

## After training: merge and test

```bash
python tools/llama/merge_lora.py \
    --lora-config r_32_alpha_64_slow \
    --base-weight checkpoints/s2-pro \
    --lora-weight results/hebrew_lora/checkpoints/step_000000500.ckpt \
    --output checkpoints/s2-pro-hebrew-lora

# Quick smoke test (WebUI) — prefer the earliest checkpoint that sounds good
python tools/run_webui.py \
    --llama-checkpoint-path checkpoints/s2-pro-hebrew-lora \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth
```

Reference voices for cloning live under `/mnt/windows_nvme/Qwen3-TTS/data/refs/hebrew/`.
