"""Convert Qwen3-TTS-style Hebrew JSONL manifests into the fish-speech dataset layout.

Input rows (one JSON object per line):
    {"audio": "/abs/path/utt.wav", "orig_text": "<Hebrew with nikud>",
     "speaker": "female1_hebrew", "duration_sec": 2.74, ...}

Output layout (audio is symlinked, not copied):
    <output>/<split>/<speaker>/<utt>.wav  -> original audio
    <output>/<split>/<speaker>/<utt>.lab  -> cleaned Hebrew text

After this, run tools/vqgan/extract_vq.py and tools/llama/build_dataset.py on
<output>/<split> (see tools/hebrew/README.md).
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

# U+05AF HEBREW MARK MASORA CIRCLE — used by renikud to mark silent (orthographic)
# letters. Not standard nikud; the base model never saw it, so strip by default.
MASORA_CIRCLE = "֯"


def clean_hebrew_text(text: str, keep_masora: bool = False) -> str:
    if not keep_masora:
        text = text.replace(MASORA_CIRCLE, "")
    # build_dataset.py replaces {...} and <...> spans with spaces; drop the
    # brackets here so no text is lost if they ever appear.
    text = re.sub(r"[{}<>]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_manifest(
    manifest: Path,
    out_dir: Path,
    keep_masora: bool,
    min_sec: float,
    max_sec: float,
) -> Counter:
    stats = Counter()
    seen_names: set[Path] = set()

    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stats["rows"] += 1

            duration = row.get("duration_sec")
            if duration is not None and not (min_sec <= duration <= max_sec):
                stats["skipped_duration"] += 1
                continue

            text = clean_hebrew_text(row["orig_text"], keep_masora=keep_masora)
            if not text:
                stats["skipped_empty_text"] += 1
                continue

            audio = Path(row["audio"])
            if not audio.exists():
                stats["skipped_missing_audio"] += 1
                continue

            speaker_dir = out_dir / row["speaker"]
            speaker_dir.mkdir(parents=True, exist_ok=True)

            target = speaker_dir / audio.name
            if target in seen_names or (
                target.is_symlink() and target.resolve() != audio.resolve()
            ):
                # Same basename from a different source dir: disambiguate.
                digest = hashlib.sha1(str(audio).encode()).hexdigest()[:8]
                target = speaker_dir / f"{audio.stem}_{digest}{audio.suffix}"
            seen_names.add(target)

            if not target.is_symlink():
                target.symlink_to(audio)
            target.with_suffix(".lab").write_text(text, encoding="utf-8")

            stats["written"] += 1
            if duration is not None:
                stats["seconds"] += int(duration)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/hebrew"))
    parser.add_argument(
        "--keep-masora",
        action="store_true",
        help="Keep U+05AF silent-letter marks instead of stripping them",
    )
    parser.add_argument("--min-sec", type=float, default=1.0)
    parser.add_argument("--max-sec", type=float, default=20.0)
    args = parser.parse_args()

    splits = [("train", args.train_manifest)]
    if args.eval_manifest is not None:
        splits.append(("eval", args.eval_manifest))

    failed = False
    for split, manifest in splits:
        if not manifest.exists():
            print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
            failed = True
            continue
        out_dir = args.output / split
        stats = process_manifest(
            manifest, out_dir, args.keep_masora, args.min_sec, args.max_sec
        )
        hours = stats.pop("seconds", 0) / 3600
        print(f"[{split}] {dict(stats)} (~{hours:.1f}h) -> {out_dir}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
