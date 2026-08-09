"""Build the fish-speech dataset straight from the slow_44K filtered CSVs.

The Qwen3-TTS manifests capped each speaker at 12k rows; the source CSVs hold
5x more. This bypasses that cap and lets the WER threshold be chosen here.

    python tools/hebrew/prepare_from_csv.py --wer-max 0.1 --output data/hebrew

CSV columns: filename, original_phonemes, whisper_phonemes, wer_score,
passed_filter. Audio lives at <root>/data/<speaker>_slow/<filename>.
Existing symlinks and .npy files are left alone, so re-running only adds work
for genuinely new utterances (extract_vq skips anything already encoded).
"""

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

MASORA_CIRCLE = "֯"


def clean(text: str) -> str:
    text = text.replace(MASORA_CIRCLE, "")
    text = re.sub(r"[{}<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("/home/maxm/AE_training_data_all/slow_44K"))
    p.add_argument("--output", type=Path, default=Path("data/hebrew"))
    p.add_argument("--wer-max", type=float, default=0.1)
    p.add_argument("--max-per-speaker", type=int, default=0, help="0 = uncapped")
    p.add_argument("--min-sec", type=float, default=1.0)
    p.add_argument("--max-sec", type=float, default=20.0)
    p.add_argument("--eval-per-speaker", type=int, default=50)
    p.add_argument("--text-field", default="original_phonemes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-duration", action="store_true",
                   help="stat every wav for duration (slow; the CSVs are already length-filtered)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    stats = Counter()
    per_speaker = Counter()
    seconds = Counter()

    csv_files = sorted(args.root.glob("*_slow_filtered.csv"))
    if not csv_files:
        print(f"ERROR: no *_slow_filtered.csv under {args.root}", file=sys.stderr)
        sys.exit(1)

    for csv_path in csv_files:
        speaker = csv_path.name.replace("_slow_filtered.csv", "")
        audio_dir = args.root / "data" / f"{speaker}_slow"
        if not audio_dir.is_dir():
            print(f"  skip {speaker}: {audio_dir} missing")
            continue

        rows = []
        with open(csv_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                stats["rows"] += 1
                if str(row.get("passed_filter", "")).strip().lower() != "true":
                    stats["skip_filter"] += 1
                    continue
                try:
                    if float(row["wer_score"]) > args.wer_max:
                        stats["skip_wer"] += 1
                        continue
                except (KeyError, ValueError):
                    stats["skip_wer_parse"] += 1
                    continue
                text = clean(row.get(args.text_field) or "")
                if not text:
                    stats["skip_empty"] += 1
                    continue
                src = audio_dir / row["filename"]
                if not src.exists():
                    stats["skip_missing"] += 1
                    continue
                rows.append((src, text))

        rng.shuffle(rows)
        if args.max_per_speaker > 0:
            rows = rows[: args.max_per_speaker]

        eval_rows = rows[: args.eval_per_speaker]
        train_rows = rows[args.eval_per_speaker :]

        for split, split_rows in (("eval", eval_rows), ("train", train_rows)):
            out_dir = args.output / split / speaker
            out_dir.mkdir(parents=True, exist_ok=True)
            for src, text in split_rows:
                if args.check_duration:
                    try:
                        dur = sf.info(str(src)).duration
                    except Exception:
                        stats["skip_unreadable"] += 1
                        continue
                    if not (args.min_sec <= dur <= args.max_sec):
                        stats["skip_duration"] += 1
                        continue
                    seconds[speaker] += dur

                dst = out_dir / src.name
                if not dst.is_symlink():
                    dst.symlink_to(src)
                    stats[f"new_{split}"] += 1
                dst.with_suffix(".lab").write_text(text, encoding="utf-8")
                stats[f"written_{split}"] += 1
                per_speaker[speaker] += 1

        print(f"  {speaker:16s} kept {len(rows):7d}")

    print(f"\nper speaker: {dict(per_speaker)}")
    print(f"stats: {dict(stats)}")
    if seconds:
        print(f"hours: {sum(seconds.values())/3600:.1f}")


if __name__ == "__main__":
    main()
