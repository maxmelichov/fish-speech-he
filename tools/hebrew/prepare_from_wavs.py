"""Build a Hebrew fish-speech dataset from your own audio + transcripts.

This is the bring-your-own-data entry point. The other two prepare scripts are
tied to specific corpora (`prepare_hebrew_data.py` reads Qwen3-TTS JSONL
manifests, `prepare_from_csv.py` reads the AE_training_data CSVs); this one only
needs a folder of audio with sibling transcripts.

Input layout — one directory per speaker:

    <root>/<speaker>/<utt>.wav      (or .mp3 / .flac / .ogg / .m4a)
    <root>/<speaker>/<utt>.lab      Hebrew text, with or without nikud

Output layout — audio symlinked, transcripts phonemized to IPA:

    <output>/train/<speaker>/<utt>.wav  -> original audio
    <output>/train/<speaker>/<utt>.lab  -> IPA
    <output>/eval/<speaker>/...

Transcripts are phonemized with RenikudPlus (`pip install renikud-plus`), the
same frontend `infer_hebrew.py` uses — so training and inference cannot
disagree about pronunciation. Transcripts that are already IPA are passed
through untouched (detected per-line). The IPA is left as plain characters;
conversion to the atomic `<ipa_*>` tokens happens in the dataset at training
time, driven by `ipa_token_map` in the training config.

    python tools/hebrew/prepare_from_wavs.py --root my_audio --output data/hebrew

Then continue with `run_hebrew_pipeline.sh extract` and `pack`.
"""

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import random
import re
from collections import Counter
from pathlib import Path

from loguru import logger

from fish_speech.text.ipa_tokens import looks_like_ipa

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")

# U+05AF HEBREW MARK MASORA CIRCLE — renikud marks silent letters with it. Not
# standard nikud, and the base model never saw it.
MASORA_CIRCLE = "֯"


def clean(text: str) -> str:
    text = text.replace(MASORA_CIRCLE, "")
    # build_dataset.py blanks {...} and <...> spans; drop the brackets so no
    # text is silently lost downstream.
    text = re.sub(r"[{}<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def audio_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True, help="<root>/<speaker>/*.wav")
    p.add_argument("--output", type=Path, default=Path("data/hebrew"))
    p.add_argument("--eval-per-speaker", type=int, default=20)
    p.add_argument("--min-sec", type=float, default=1.0)
    p.add_argument("--max-sec", type=float, default=20.0)
    p.add_argument(
        "--no-duration-check",
        action="store_true",
        help="Skip reading audio headers (faster; keeps every clip)",
    )
    p.add_argument(
        "--ipa",
        action="store_true",
        help="Transcripts are already IPA everywhere; skip G2P entirely",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    speakers = sorted(d for d in args.root.iterdir() if d.is_dir())
    if not speakers:
        p.error(f"no speaker directories under {args.root}")

    g2p = None
    if not args.ipa:
        try:
            from renikud_onnx import G2P
        except ImportError:
            p.error("RenikudPlus is required: pip install renikud-plus (or use --ipa)")
        g2p = G2P()

    rng = random.Random(args.seed)
    stats = Counter()

    for spk_dir in speakers:
        rows = []
        for audio in sorted(spk_dir.iterdir()):
            if audio.suffix.lower() not in AUDIO_EXTS:
                continue
            lab = audio.with_suffix(".lab")
            if not lab.exists():
                stats["no_transcript"] += 1
                continue
            text = clean(lab.read_text(encoding="utf-8"))
            if not text:
                stats["empty_transcript"] += 1
                continue
            if not args.no_duration_check:
                dur = audio_duration(audio)
                if dur is None:
                    stats["unreadable_audio"] += 1
                    continue
                if not (args.min_sec <= dur <= args.max_sec):
                    stats["out_of_duration_range"] += 1
                    continue
            rows.append((audio, text))

        if not rows:
            logger.warning(f"{spk_dir.name}: nothing usable, skipping")
            continue

        rng.shuffle(rows)
        n_eval = min(args.eval_per_speaker, max(0, len(rows) - 1))
        splits = [("eval", rows[:n_eval]), ("train", rows[n_eval:])]

        for split, split_rows in splits:
            if not split_rows:
                continue
            out_dir = args.output / split / spk_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for audio, text in split_rows:
                ipa = (
                    text if (args.ipa or looks_like_ipa(text)) else g2p.phonemize(text)
                )
                dst = out_dir / f"{audio.stem}{audio.suffix.lower()}"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(audio.resolve())
                dst.with_suffix(".lab").write_text(clean(ipa), encoding="utf-8")
                stats[split] += 1

        logger.info(f"{spk_dir.name}: {len(rows) - n_eval} train / {n_eval} eval")

    logger.info(f"Done: {dict(stats)}")
    if stats["train"] == 0:
        raise SystemExit("no training rows were written — check --root layout")
    logger.info(
        f"Next: tools/hebrew/run_hebrew_pipeline.sh extract && "
        f"tools/hebrew/run_hebrew_pipeline.sh pack"
    )


if __name__ == "__main__":
    main()
