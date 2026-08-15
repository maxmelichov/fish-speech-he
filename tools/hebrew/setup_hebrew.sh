#!/bin/bash
# One-command setup for Hebrew inference: base weights + IPA checkpoint + adapter.
#
#   bash tools/hebrew/setup_hebrew.sh
#   python tools/hebrew/infer_hebrew.py --text "שלום עולם" \
#       --lora-checkpoint checkpoints/hebrew/hebrew_lora_step2200.safetensors \
#       --output out.wav
#
# Fine-tuning needs the dataset steps too — see tools/hebrew/run_hebrew_pipeline.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=${CKPT:-checkpoints/s2-pro}
IPA_CKPT=${IPA_CKPT:-checkpoints/s2-pro-he-ipa}
HE_CKPT=${HE_CKPT:-checkpoints/hebrew}
HE_REPO=${HE_REPO:-notmax123/Fish-Audio-S2-Pro-He}

command -v hf >/dev/null || { echo "hf CLI not found: pip install huggingface_hub[cli]" >&2; exit 1; }

echo "==> [1/3] S2-Pro base weights -> $CKPT (~11GB)"
hf download fishaudio/s2-pro --local-dir "$CKPT"

echo "==> [2/3] Hebrew adapter -> $HE_CKPT (~134MB)"
hf download "$HE_REPO" --local-dir "$HE_CKPT" \
    --include "*.safetensors" "*.json" "*.pt" "tokenizer/*"

echo "==> [3/3] Building the IPA-extended checkpoint -> $IPA_CKPT"
python tools/hebrew/build_ipa_checkpoint.py --base "$CKPT" --output "$IPA_CKPT"

cat <<EOF

Ready. Hebrew G2P needs one more package:

    pip install renikud-plus

Then:

    python tools/hebrew/infer_hebrew.py \\
        --text "שלום, מה שלומך היום?" \\
        --lora-checkpoint $HE_CKPT/hebrew_lora_step2200.safetensors \\
        --output out.wav
EOF
