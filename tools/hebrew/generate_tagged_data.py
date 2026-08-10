"""Self-distill emotion-tag behaviour into the Hebrew IPA training set.

Measured problem: S2-Pro's [whisper]/[excited]/... tags are RL-trained against
natural-language text and are IGNORED when the surrounding text is IPA
(verified: [whisper] vs plain IPA produced identical rms). The fix is training
rows that pair tag+IPA text with tag-obeying audio.

Recipe (per row):
  1. take a Hebrew-script sentence (manifest orig_text) + a random tag
  2. BASE model generates audio for "[tag] <hebrew script>" (tags work there),
     voice-cloned to one of our speakers -> semantic codes, no codec roundtrip
  3. training row = codes + "[tag] <converted IPA of the same sentence>"

Rows land in data/hebrew/train/tagged_<speaker>/ as .npy + .lab, so
build_dataset packs them like any other speaker group.

Run AFTER extract_vq (needs a free GPU):
  CUDA_VISIBLE_DEVICES=0 python tools/hebrew/generate_tagged_data.py --num 600 --shard 0/2
  CUDA_VISIBLE_DEVICES=1 python tools/hebrew/generate_tagged_data.py --num 600 --shard 1/2
"""

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

TAGS = [
    "[whisper]",
    "[excited]",
    "[sad]",
    "[angry]",
    "[laughing]",
    "[surprised]",
    "[shouting]",
    "[low voice]",
    "[professional broadcast tone]",
    "[sigh]",
    "[pause]",
    "[short pause]",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/windows_nvme/Qwen3-TTS/data/manifests/hebrew_train.jsonl"),
    )
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/s2-pro-he-ipa"))
    p.add_argument("--output", type=Path, default=Path("data/hebrew/train"))
    p.add_argument("--num", type=int, default=600, help="rows for this shard")
    p.add_argument("--shard", default="0/1", help="i/n round-robin shard")
    p.add_argument("--min-chars", type=int, default=25)
    p.add_argument("--max-chars", type=int, default=90)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    rng = random.Random(args.seed)  # same seed both shards -> same row plan

    from fish_speech.models.text2semantic.inference import (
        decode_one_token_ar,
        encode_audio,
        generate_long,
        load_codec_model,
    )
    from fish_speech.models.text2semantic.llama import BaseTransformer
    from fish_speech.text.ipa_tokens import convert_ipa, load_token_map

    token_map = load_token_map(args.checkpoint / "ipa_token_map.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Rows: (script_text, ipa_text) pairs of sane length
    pool = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            script, ipa, spk = r.get("orig_text"), r.get("text"), r["speaker"]
            if not script or not ipa:
                continue
            if not (args.min_chars <= len(script) <= args.max_chars):
                continue
            pool.append((script.strip(), ipa.strip(), spk))
    rng.shuffle(pool)

    # One reference per speaker (fixed, from eval)
    refs = {}
    for spk_dir in sorted(Path("data/hebrew/eval").iterdir()):
        for wav in sorted(spk_dir.glob("*.wav")):
            if wav.with_suffix(".lab").exists():
                refs[spk_dir.name] = wav
                break

    # Base model (tags only work without the LoRA)
    model = (
        BaseTransformer.from_pretrained(
            str(args.checkpoint), load_weights=True, max_length=4096
        )
        .to(device, dtype=dtype)
        .eval()
    )
    codec = load_codec_model(
        str(args.checkpoint / "codec.pth"), device=device, precision=dtype
    )
    ref_cache = {
        s: (
            encode_audio(str(w), codec, device).cpu(),
            w.with_suffix(".lab").read_text(encoding="utf-8").strip(),
        )
        for s, w in refs.items()
    }

    made = skipped = 0
    plan = [row for i, row in enumerate(pool) if i % shard_n == shard_i][: args.num]
    logger.info(f"shard {args.shard}: generating {len(plan)} tagged rows")

    for i, (script, ipa, spk) in enumerate(plan):
        if spk not in ref_cache:
            spk = rng.choice(list(ref_cache))
        tag = rng.choice(TAGS)
        out_dir = args.output / f"tagged_{spk}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"tag{shard_i}_{i:05d}"
        if (out_dir / f"{stem}.npy").exists():
            continue

        rtok, rtxt = ref_cache[spk]
        try:
            segs = [
                r.codes
                for r in generate_long(
                    model=model,
                    device=device,
                    decode_one_token=decode_one_token_ar,
                    text=f"{tag} {script}",
                    max_new_tokens=400,
                    temperature=0.8,
                    top_p=0.8,
                    top_k=30,
                    compile=False,
                    chunk_length=400,
                    prompt_text=[rtxt],
                    prompt_tokens=[rtok],
                )
                if r.action == "sample"
            ]
        except Exception as e:
            logger.warning(f"{stem}: {e}")
            skipped += 1
            continue
        if not segs:
            skipped += 1
            continue
        codes = torch.cat(segs, dim=1).cpu().numpy()
        if not (20 <= codes.shape[1] <= 500):  # ~1-23s sanity window
            skipped += 1
            continue

        np.save(out_dir / f"{stem}.npy", codes)
        lab = f"{tag} {convert_ipa(ipa, token_map)}"
        (out_dir / f"{stem}.lab").write_text(lab, encoding="utf-8")
        made += 1
        if made % 50 == 0:
            logger.info(f"{made}/{len(plan)} rows")

    logger.info(f"DONE shard {args.shard}: {made} rows, {skipped} skipped")


if __name__ == "__main__":
    main()
