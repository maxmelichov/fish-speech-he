#!/bin/bash
# Hebrew LoRA pipeline for fish-speech (S2-Pro base).
# Each step is idempotent/resumable; rerun the script after interruptions.
set -euo pipefail
cd "$(dirname "$0")/../.."

QWEN_TTS_DIR=${QWEN_TTS_DIR:-/mnt/windows_nvme/Qwen3-TTS}
CKPT=${CKPT:-checkpoints/s2-pro}
DATA=${DATA:-data/hebrew}

step=${1:-all}

if [[ "$step" == "download" || "$step" == "all" ]]; then
    echo "==> [1/5] Downloading S2-Pro weights"
    hf download fishaudio/s2-pro --local-dir "$CKPT"
fi

if [[ "$step" == "prepare" || "$step" == "all" ]]; then
    echo "==> [2/5] Preparing dataset (symlinks + .lab files)"
    python tools/hebrew/prepare_hebrew_data.py \
        --train-manifest "$QWEN_TTS_DIR/data/manifests/hebrew_train.jsonl" \
        --eval-manifest "$QWEN_TTS_DIR/data/manifests/hebrew_eval.jsonl" \
        --output "$DATA"
fi

if [[ "$step" == "extract" || "$step" == "all" ]]; then
    echo "==> [3/5] Extracting VQ tokens (uses all GPUs; resumable, skips existing .npy)"
    python tools/vqgan/extract_vq.py "$DATA" \
        --num-workers "$(nvidia-smi -L | wc -l)" \
        --batch-size 32 \
        --config-name modded_dac_vq \
        --checkpoint-path "$CKPT/codec.pth"
fi

if [[ "$step" == "pack" || "$step" == "all" ]]; then
    echo "==> [4/5] Packing protobuf datasets"
    python tools/llama/build_dataset.py \
        --input "$DATA/train" --output "$DATA/protos/train" \
        --text-extension .lab --num-workers 16
    python tools/llama/build_dataset.py \
        --input "$DATA/eval" --output "$DATA/protos/eval" \
        --text-extension .lab --num-workers 16
fi

if [[ "$step" == "train" || "$step" == "all" ]]; then
    echo "==> [5/5] LoRA training (auto-resumes from results/hebrew_lora/checkpoints)"
    python fish_speech/train.py --config-name text2semantic_hebrew_lora
fi
