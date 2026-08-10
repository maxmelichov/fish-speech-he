"""Create checkpoints/s2-pro-he-ipa: S2-Pro plus atomic Hebrew-IPA tokens.

Mirrors the Qwen3-TTS `create_ipa_token_model.py` approach for fish-speech:
weights are symlinked from the base checkpoint (nothing is duplicated), the
tokenizer gains one dedicated token per IPA symbol, and the new tokens'
embeddings are initialised from the mean of their original BPE-piece
embeddings and stored in ipa_embeddings.pt (loaded by
BaseTransformer.from_pretrained).
"""

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from fish_speech.text.ipa_tokens import (
    HEBREW_IPA_SYMBOLS,
    build_token_map,
    save_token_map,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", type=Path, default=Path("checkpoints/s2-pro"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/s2-pro-he-ipa"))
    args = p.parse_args()

    out = args.output
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1. Symlink the heavy files
    for f in sorted(args.base.iterdir()):
        if f.suffix in (".safetensors", ".pth") or f.name.endswith(".index.json"):
            (out / f.name).symlink_to(f.resolve())
    print(f"symlinked weights from {args.base}")

    # 2. Extended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    base_len = len(tokenizer)
    token_map = build_token_map(HEBREW_IPA_SYMBOLS)
    added = tokenizer.add_tokens(list(token_map.values()), special_tokens=False)
    assert added == len(token_map), (added, len(token_map))
    tokenizer.save_pretrained(str(out))
    save_token_map(token_map, out / "ipa_token_map.json")
    print(f"tokenizer: {base_len} -> {len(tokenizer)} (+{added} IPA tokens)")

    # sanity: each new token encodes to exactly one id, at/after base_len
    ids = []
    for sym, tok in token_map.items():
        enc = tokenizer.encode(tok, add_special_tokens=False)
        assert len(enc) == 1, (sym, tok, enc)
        assert enc[0] >= base_len
        ids.append(enc[0])
        # raw symbol must still encode to its ORIGINAL pieces (all < base_len)
        raw = tokenizer.encode(sym, add_special_tokens=False)
        assert all(i < base_len for i in raw), (sym, raw)
    ipa_token_start = min(ids)
    assert max(ids) - ipa_token_start == len(ids) - 1, "ids not contiguous"

    # 3. Config with the extension fields (top-level; _from_fish_qwen3_omni
    #    reads them from the root of the dict)
    cfg = json.loads((args.base / "config.json").read_text())
    cfg["num_ipa_tokens"] = len(token_map)
    cfg["ipa_token_start"] = ipa_token_start
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"config: num_ipa_tokens={len(token_map)}, ipa_token_start={ipa_token_start}")

    # 4. Precompute the embedding init (avoids a DDP-rank race on first load)
    from fish_speech.models.text2semantic.llama import BaseTransformer

    model = BaseTransformer.from_pretrained(str(out), load_weights=True)
    assert (out / "ipa_embeddings.pt").exists(), "init file was not written"
    w = torch.load(out / "ipa_embeddings.pt", weights_only=True)
    print(
        f"ipa_embeddings.pt: {tuple(w.shape)}, norms "
        f"{w.norm(dim=1).min():.3f}-{w.norm(dim=1).max():.3f}"
    )

    # codec convenience symlink
    codec = args.base / "codec.pth"
    if codec.exists() and not (out / "codec.pth").exists():
        (out / "codec.pth").symlink_to(codec.resolve())

    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
