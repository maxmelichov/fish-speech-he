"""Synthesize Hebrew from plain (unvocalized) text with a trained Hebrew LoRA.

Text is phonemized to IPA with RenikudPlus (`pip install renikud-plus`), the
same representation the LoRA was trained on, then synthesized through S2-Pro
with an optional voice-cloning reference.

    python tools/hebrew/infer_hebrew.py \
        --text "שלום, מה שלומך היום?" \
        --lora-checkpoint results/hebrew_lora/checkpoints/step_000001000.ckpt \
        --ref-audio data/hebrew/eval/male3/utt_1270581.wav \
        --output out.wav

Long inputs are split on sentence boundaries and synthesized chunk by chunk
(each chunk conditioned on the same reference), then concatenated — generation
stays well inside the model's context and quality does not decay with length.
"""

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from fish_speech.models.text2semantic.inference import (
    decode_one_token_ar,
    decode_to_audio,
    encode_audio,
    generate_long,
    load_codec_model,
)
from fish_speech.models.text2semantic.llama import BaseTransformer

# Sentence-ish boundaries; keeps the delimiter with the preceding chunk.
SENTENCE_RE = re.compile(r"(?<=[.!?:;])\s+")


def split_sentences(text: str, max_chars: int) -> list[str]:
    chunks, current = [], ""
    for piece in SENTENCE_RE.split(text.strip()):
        if not piece:
            continue
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)

    # Hard-wrap anything still oversized (a single very long sentence).
    out = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            cut = chunk.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if chunk:
            out.append(chunk)
    return out


def load_lora_checkpoint(model, checkpoint_path: Path) -> None:
    """Load a LoRA adapter, either a Lightning .ckpt or a .safetensors export."""
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_path))
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
    state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    lora_keys = [k for k in state if "lora" in k]
    if not lora_keys:
        raise ValueError(f"No LoRA weights found in {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    logger.info(f"Loaded {len(lora_keys)} LoRA tensors from {checkpoint_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--text", type=str, default=None, help="Hebrew text (unvocalized)")
    p.add_argument("--text-file", type=Path, default=None)
    p.add_argument("--ipa", action="store_true", help="Input is already IPA; skip G2P")
    p.add_argument("--output", type=Path, default=Path("hebrew_out.wav"))
    p.add_argument(
        "--base-checkpoint", type=Path, default=Path("checkpoints/s2-pro-he-ipa")
    )
    p.add_argument("--codec-checkpoint", type=Path, default=None)
    p.add_argument("--lora-checkpoint", type=Path, default=None)
    # The released adapter (notmax123/Fish-Audio-S2-Pro-He) was trained with
    # r_32_alpha_16_core; a mismatched config silently rescales the delta.
    p.add_argument("--lora-config", type=str, default="r_32_alpha_16_core")
    p.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scale the LoRA delta (0 = pure base model). Mid-training "
        "checkpoints often sound best well below 1.0 — sweep by ear.",
    )
    p.add_argument("--ref-audio", type=Path, default=None)
    p.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of --ref-audio; defaults to its sibling .lab file",
    )
    p.add_argument("--max-chars", type=int, default=200, help="Chunk size in IPA chars")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gap-ms", type=int, default=120, help="Silence between chunks")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if not args.text and not args.text_file:
        p.error("one of --text or --text-file is required")
    text = args.text if args.text else args.text_file.read_text(encoding="utf-8")

    codec_path = args.codec_checkpoint or args.base_checkpoint / "codec.pth"
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    precision = torch.bfloat16 if device.type != "cpu" else torch.float32

    # 1. Hebrew -> IPA
    if args.ipa:
        ipa = text.strip()
    else:
        from renikud_onnx import G2P

        g2p = G2P()
        ipa = g2p.phonemize(text.strip())
        logger.info(f"G2P: {text.strip()[:60]}... -> {ipa[:60]}...")

    # 1b. Atomic IPA tokens, when the checkpoint carries a map
    map_path = args.base_checkpoint / "ipa_token_map.json"
    if map_path.exists():
        from fish_speech.text.ipa_tokens import convert_ipa, load_token_map

        ipa = convert_ipa(ipa, load_token_map(map_path))
        logger.info("Converted to atomic IPA tokens")

    chunks = split_sentences(ipa, args.max_chars)
    logger.info(f"Split into {len(chunks)} chunk(s)")

    # 2. Models
    lora_config = None
    if args.lora_checkpoint is not None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg_path = Path("fish_speech/configs/lora") / f"{args.lora_config}.yaml"
        if not cfg_path.exists():
            p.error(f"LoRA config not found: {cfg_path}")
        lora_config = instantiate(OmegaConf.load(cfg_path))
        logger.info(f"LoRA config: {args.lora_config} ({cfg_path})")

    model = BaseTransformer.from_pretrained(
        str(args.base_checkpoint),
        load_weights=True,
        max_length=args.max_length,
        lora_config=lora_config,
    )
    if args.lora_checkpoint is not None:
        load_lora_checkpoint(model, args.lora_checkpoint)
        if args.lora_scale != 1.0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "lora_B" in name:
                        param.mul_(args.lora_scale)
            logger.info(f"LoRA delta scaled to {args.lora_scale}")
    # eval() merges the (scaled) LoRA delta into the base weights (loralib)
    model = model.to(device=device, dtype=precision).eval()

    codec = load_codec_model(str(codec_path), device=device, precision=precision)

    # 3. Reference (voice cloning)
    prompt_tokens = prompt_text = None
    if args.ref_audio is not None:
        ref_text = args.ref_text
        if ref_text is None:
            lab = args.ref_audio.with_suffix(".lab")
            if not lab.exists():
                p.error(f"--ref-text not given and {lab} does not exist")
            ref_text = lab.read_text(encoding="utf-8").strip()
        if map_path.exists():
            from fish_speech.text.ipa_tokens import convert_ipa, load_token_map

            ref_text = convert_ipa(ref_text, load_token_map(map_path))
        prompt_tokens = [encode_audio(str(args.ref_audio), codec, device).cpu()]
        prompt_text = [ref_text]
        logger.info(f"Reference: {args.ref_audio.name} ({ref_text[:50]}...)")

    # 4. Synthesize chunk by chunk
    sample_rate = codec.sample_rate
    gap = np.zeros(int(sample_rate * args.gap_ms / 1000), dtype=np.float32)
    waves = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"[{i}/{len(chunks)}] {chunk[:70]}")
        segments = [
            r.codes
            for r in generate_long(
                model=model,
                device=device,
                decode_one_token=decode_one_token_ar,
                text=chunk,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                compile=False,
                chunk_length=args.max_chars,
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
            )
            if r.action == "sample"
        ]
        if not segments:
            logger.warning(f"chunk {i} produced no codes, skipping")
            continue
        codes = torch.cat(segments, dim=1).to(device)
        wav = decode_to_audio(codes, codec).float().cpu().numpy().copy()
        waves.append(wav)
        if i < len(chunks):
            waves.append(gap)

    if not waves:
        raise RuntimeError("nothing was generated")

    audio = np.concatenate(waves)
    peak = float(np.abs(audio).max())
    if peak > 1.0:
        audio = audio / peak

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, sample_rate)
    logger.info(
        f"Wrote {args.output} — {len(audio) / sample_rate:.2f}s, "
        f"rms={np.sqrt((audio.astype(np.float64) ** 2).mean()):.4f}, peak={peak:.3f}"
    )


if __name__ == "__main__":
    main()
