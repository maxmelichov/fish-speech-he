#!/bin/bash
# Hebrew LoRA pipeline for fish-speech (S2-Pro base).
# Each step is idempotent/resumable; rerun the script after interruptions.
set -euo pipefail
cd "$(dirname "$0")/../.."

QWEN_TTS_DIR=${QWEN_TTS_DIR:-/mnt/windows_nvme/Qwen3-TTS}
CKPT=${CKPT:-checkpoints/s2-pro}          # base S2-Pro weights
IPA_CKPT=${IPA_CKPT:-checkpoints/s2-pro-he-ipa}  # base + 26 atomic IPA tokens
DATA=${DATA:-data/hebrew}

step=${1:-all}

if [[ "$step" == "download" || "$step" == "all" ]]; then
    echo "==> [1/6] Downloading S2-Pro weights"
    hf download fishaudio/s2-pro --local-dir "$CKPT"
fi

if [[ "$step" == "ipa" || "$step" == "all" ]]; then
    echo "==> [2/6] Building the IPA-extended checkpoint ($IPA_CKPT)"
    # Symlinks the S2-Pro weights and layers on the extended tokenizer
    # (155,774 -> 155,800), the IPA token map and the initial IPA embeddings.
    python tools/hebrew/build_ipa_checkpoint.py --base "$CKPT" --output "$IPA_CKPT"
fi

if [[ "$step" == "prepare" || "$step" == "all" ]]; then
    echo "==> [3/6] Preparing dataset (symlinks + .lab files)"
    if [[ -n "${AUDIO_ROOT:-}" ]]; then
        # Bring-your-own data: AUDIO_ROOT/<speaker>/*.wav + sibling .lab
        python tools/hebrew/prepare_from_wavs.py \
            --root "$AUDIO_ROOT" --output "$DATA"
    else
        python tools/hebrew/prepare_hebrew_data.py \
            --train-manifest "$QWEN_TTS_DIR/data/manifests/hebrew_train.jsonl" \
            --eval-manifest "$QWEN_TTS_DIR/data/manifests/hebrew_eval.jsonl" \
            --output "$DATA" \
            --text-repr "${TEXT_REPR:-ipa}"
    fi
fi

if [[ "$step" == "extract" || "$step" == "all" ]]; then
    echo "==> [4/6] Extracting VQ tokens (uses all GPUs; resumable, skips existing .npy)"
    # batch-size 8: the DAC encoder OOMs on a 32GB card at 32 with 20s clips
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python tools/vqgan/extract_vq.py "$DATA" \
        --num-workers "$(nvidia-smi -L | wc -l)" \
        --batch-size 8 \
        --config-name modded_dac_vq \
        --checkpoint-path "$CKPT/codec.pth"
fi

if [[ "$step" == "pack" || "$step" == "all" ]]; then
    echo "==> [5/6] Packing protobuf datasets"
    python tools/llama/build_dataset.py \
        --input "$DATA/train" --output "$DATA/protos/train" \
        --text-extension .lab --num-workers 16
    python tools/llama/build_dataset.py \
        --input "$DATA/eval" --output "$DATA/protos/eval" \
        --text-extension .lab --num-workers 16
fi

if [[ "$step" == "train" || "$step" == "all" ]]; then
    echo "==> [6/6] LoRA training (auto-resumes from results/hebrew_lora/checkpoints)"
    python fish_speech/train.py --config-name text2semantic_hebrew_lora
fi
