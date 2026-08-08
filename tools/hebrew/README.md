# Hebrew LoRA training

LoRA fine-tuning of **S2-Pro** on the Hebrew data prepared in the Qwen3-TTS repo
(`/mnt/windows_nvme/Qwen3-TTS`): ~131k utterances / ~183h across 12 speakers.

## Design decisions

- **Base model: `fishaudio/s2-pro`.** The older `openaudio-s1-mini` ships only a
  `tokenizer.tiktoken` file the current `FishTokenizer` (AutoTokenizer-based)
  cannot load, and S1 never supported Hebrew. S2-Pro has Hebrew in its
  pretraining set and ships an HF-format tokenizer. Note the upstream docs warn
  against fine-tuning RL-aligned models — mitigate by preferring the earliest
  checkpoint that sounds good.
- **Text: Hebrew script with nikud** (`orig_text` from the manifests), not the
  IPA used for Qwen3-TTS. S2-Pro's BPE saw Hebrew script in pretraining; IPA
  would be out-of-distribution. Nikud provides the same pronunciation
  disambiguation IPA did. At inference time, run text through the same
  nikud frontend (renikud) used to build the training data.
- **U+05AF (masora circle) is stripped** from the text by default. renikud uses
  it to mark silent letters; it is not standard nikud and the base model never
  saw it. Pass `--keep-masora` to `prepare_hebrew_data.py` to keep it (then you
  must also emit it at inference time).
- **Audio is symlinked, not copied.** `extract_vq.py` writes each `.npy` next
  to the symlink (inside `data/hebrew/`), so the source dataset is untouched.
- **VQ codes from Qwen3-TTS (`audio_codes`) are not reusable** — different
  codec. Everything is re-encoded with the S2-Pro DAC codec.

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

## VRAM tuning (2x RTX 5090, 32GB)

The config runs DDP on both GPUs, `batch_size=2`, `max_length=2048`,
`accumulate_grad_batches=4` (effective batch 16). If you OOM:

```bash
python fish_speech/train.py --config-name text2semantic_hebrew_lora \
    data.batch_size=1 trainer.accumulate_grad_batches=8
```

Other useful overrides:

- `lora@model.model.lora_config=r_32_alpha_16_fast` — higher-rank LoRA on the
  fast (acoustic) transformer only; the default `r_8_alpha_16` adapts both slow
  and fast transformers, which is what a new-language adaptation needs.
- `trainer.max_steps=...`, `model.optimizer.lr=...`

## After training: merge and test

```bash
python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/s2-pro \
    --lora-weight results/hebrew_lora/checkpoints/step_000000500.ckpt \
    --output checkpoints/s2-pro-hebrew-lora

# Quick smoke test (WebUI) — prefer the earliest checkpoint that sounds good
python tools/run_webui.py \
    --llama-checkpoint-path checkpoints/s2-pro-hebrew-lora \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth
```

Reference voices for cloning live under `/mnt/windows_nvme/Qwen3-TTS/data/refs/hebrew/`.
