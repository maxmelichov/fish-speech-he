# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fish Speech / Fish Audio S2 Pro — a multilingual TTS system. Two models work together at inference time:

1. **text2semantic** (`fish_speech/models/text2semantic/`) — a Dual-AR transformer. A 4B "slow" AR transformer predicts the primary semantic codebook along the time axis; a 400M "fast" AR head (`fast_*` config fields, `DualARTransformer`) predicts the remaining 9 residual codebooks at each timestep. `NaiveTransformer` is the non-Dual-AR variant.
2. **DAC codec** (`fish_speech/models/dac/`) — an RVQ audio codec (10 codebooks, ~21 Hz) that encodes reference audio into VQ tokens and decodes generated semantic tokens back to a waveform.

Checkpoints are expected under `checkpoints/s2-pro/` (weights + `tokenizer.tiktoken`) with the codec at `checkpoints/s2-pro/codec.pth`. Download with `hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro`.

## Setup

```bash
apt install portaudio19-dev libsox-dev ffmpeg      # system deps

uv sync --python 3.12 --extra cu129                # or --extra cpu / cu126 / cu128
# or: pip install -e .[cu129]
```

The `cpu`/`cu126`/`cu128`/`cu129` extras are mutually exclusive (declared as uv conflicts) and pull torch from different PyTorch indexes. Python >= 3.10; docs and CI use 3.12.

## Common commands

```bash
# Gradio WebUI (default entrypoint, also what docker entrypoint.sh runs)
python tools/run_webui.py [--compile] [--half] [--device cpu]

# HTTP API server (kui/uvicorn)
python tools/api_server.py --listen 0.0.0.0:8080 \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth [--compile] [--api-key ...]
python tools/api_client.py --url http://127.0.0.1:8080/v1/tts --text "..." --output out

# Awesome WebUI (React/Vite frontend, served by the API server at /ui)
cd awesome_webui && npm install && npm run build   # npm run dev / npm run lint

# 3-step CLI inference
python fish_speech/models/dac/inference.py -i ref.wav --checkpoint-path checkpoints/s2-pro/codec.pth   # -> fake.npy
python fish_speech/models/text2semantic/inference.py --text "..." --prompt-text "..." --prompt-tokens fake.npy [--compile]  # -> codes_0.npy
python fish_speech/models/dac/inference.py -i codes_0.npy                                              # -> fake.wav

# LoRA fine-tuning pipeline (dataset of .lab + audio pairs under data/)
python tools/vqgan/extract_vq.py data --config-name modded_dac_vq --checkpoint-path checkpoints/s2-pro/codec.pth
python tools/llama/build_dataset.py --input data --output data/protos --text-extension .lab --num-workers 16
python fish_speech/train.py --config-name text2semantic_finetune project=$project +lora@model.model.lora_config=r_8_alpha_16
python tools/llama/merge_lora.py --lora-config r_8_alpha_16 --base-weight checkpoints/s2-pro \
  --lora-weight results/$project/checkpoints/step_000000010.ckpt --output checkpoints/s2-pro-lora/

# Docker
docker compose --profile webui up          # or --profile server; BACKEND=cpu, COMPILE=1
docker compose -f compose.rocm.yml --profile webui up --build   # AMD ROCm
```

There is **no test suite** in this repo. Lint/format is enforced by pre-commit only: `isort --profile=black`, `black`, plus whitespace/YAML/JSON hooks. Run `pre-commit run --all-files` before committing.

## Architecture notes

**Entrypoints all converge on `TTSInferenceEngine`.** `tools/run_webui.py` and `tools/server/model_manager.py` both load the llama model via `launch_thread_safe_queue()` and the codec via `fish_speech/models/dac/inference.py:load_model()`, then wrap them in `fish_speech/inference_engine/TTSInferenceEngine`. If you change the inference contract, both callers must be updated.

**The llama model runs in its own thread behind a queue.** `launch_thread_safe_queue()` (`models/text2semantic/inference.py`) spawns a worker holding the compiled model; callers push `GenerateRequest` objects and read `WrappedGenerateResponse` off a per-request response queue. This exists because `torch.compile` state and the KV cache are not safe to share across threads. Code inside `decode_one_token_*` must stay compile-friendly — no data-dependent Python branching (use tensor ops like `torch.where`); see the comment at `inference.py:137`.

**`ContentSequence` (`fish_speech/content_sequence.py`) is the prompt format.** Prompts are built from `TextPart` / `VQPart` / `AudioPart` and encoded into interleaved token+codebook tensors, e.g. `<|interleave|><|speaker:1|> TEXT AUDIO <|im_end|>`. Special tokens (`<|semantic:i|>`, modality tokens, speaker tokens) are defined in `fish_speech/tokenizer.py` and must stay in sync with the checkpoint's tokenizer. This is the shared representation for both training datasets and inference.

**Reference audio / voice cloning.** `ReferenceLoader` resolves either `reference_id` (a folder under `references/<id>/` containing audio + matching `.lab` text) or inline `references` (hashed audio bytes), and caches encoded VQ tokens by id and by sha256 hash. `use_memory_cache` on the request controls reuse.

**Checkpoint loading is format-tolerant.** `BaseModelArgs.from_pretrained` detects Fish-Qwen3-Omni-style configs and remaps them (`_from_fish_qwen3_omni`, `_remap_fish_qwen3_omni_keys` maps audio-decoder keys onto `fast_*` params). The codec is instantiated through Hydra from `fish_speech/configs/modded_dac_vq.yaml`, and `load_model` strips `generator.` prefixes and loads with `strict=False`.

**Training is Lightning + Hydra.** `fish_speech/train.py` composes `fish_speech/configs/*.yaml` (`base.yaml` + `text2semantic_finetune.yaml`, LoRA configs under `configs/lora/`). Datasets are protobuf-packed semantic streams (`fish_speech/datasets/semantic.py`, protos in `fish_speech/datasets/protos/`); only the text2semantic model is fine-tuned, never the codec.

**API surface** lives in `tools/server/views.py`: `/v1/tts`, `/v1/vqgan/encode`, `/v1/vqgan/decode`, `/v1/references/{add,list,update,delete}`, `/v1/health`, and `/ui` (serves `awesome_webui/dist/index.html` if built). Request/response models are pydantic schemas in `fish_speech/utils/schema.py`; requests are msgpack-encoded (`MsgPackRequest` in `tools/server/api_utils.py`).

**Device handling** is duplicated across entrypoints: each checks mps → xpu → cuda → cpu and picks `torch.half` vs `torch.bfloat16` from `--half`. `--compile` gives a large speedup but is unsupported on Windows/macOS without manually installing Triton.

## Conventions

- Scripts under `tools/` and `fish_speech/models/*/inference.py` call `pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)` before importing project modules — keep that ordering when adding new entrypoints (isort must not reorder those imports above it).
- Logging uses `loguru`; CLI arg parsing is a mix of `click` (model inference scripts) and `argparse` (tools).
- Gradio WebUI strings go through `fish_speech/i18n/` with locale JSON files in `fish_speech/i18n/locale/`; `fish_speech/i18n/scan.py` extracts new keys.
- Docs are MkDocs (`mkdocs.yml`) with per-language directories `docs/{en,zh,ja,ko,pt,es,ar}/`. Changes to install/inference/finetune instructions should be mirrored in the English docs at minimum.
