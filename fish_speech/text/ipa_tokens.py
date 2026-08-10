"""Atomic IPA tokens: one dedicated tokenizer token per IPA symbol.

Why: the IPA transcriptions use ASCII letters ("j", "w", "a"...) whose BPE
tokens carry strong English pronunciation priors — the base model reads IPA
"j" (Hebrew י, the /j/ glide) like English "j" (/dʒ/). Mapping every IPA
symbol to a dedicated token (`<ipa_j>`, `<ipa_u0283>`, ...) severs that
association completely. This mirrors the Qwen3-TTS Hebrew setup
(finetuning/create_ipa_token_model.py in that repo): atomic tokens, embeddings
initialised from the mean of the symbol's original BPE-piece embeddings, and
trained during fine-tuning.

Text inside [emotion tags] and any unmapped character (punctuation, spaces,
digits) is left untouched.
"""

import json
import re
import unicodedata
from pathlib import Path

TAG_RE = re.compile(r"(\[[^\]]+\])")

# Every letter/mark symbol appearing in the Hebrew IPA corpus (punctuation and
# spaces deliberately excluded — the base model's prosody handling of those is
# worth keeping).
HEBREW_IPA_SYMBOLS = [
    "a",
    "b",
    "d",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "s",
    "t",
    "u",
    "v",
    "w",
    "z",
    "ɡ",
    "ʁ",
    "ʃ",
    "ʔ",
    "ˈ",
    "χ",
]


def token_for_symbol(symbol: str) -> str:
    if ord(symbol) > 127 or unicodedata.category(symbol).startswith("M"):
        return f"<ipa_u{ord(symbol):04X}>"
    return f"<ipa_{symbol}>"


def build_token_map(symbols=HEBREW_IPA_SYMBOLS) -> dict[str, str]:
    mapping = {s: token_for_symbol(s) for s in symbols}
    assert len(set(mapping.values())) == len(mapping), "token names collide"
    return mapping


def save_token_map(mapping: dict[str, str], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_token_map(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def convert_ipa(text: str, token_map: dict[str, str]) -> str:
    """Per-character mapping of IPA symbols to atomic tokens, skipping [tags]."""
    out = []
    for part in TAG_RE.split(text):
        if TAG_RE.fullmatch(part):
            out.append(part)
        else:
            out.append("".join(token_map.get(ch, ch) for ch in part))
    return "".join(out)


def looks_like_ipa(text: str) -> bool:
    """Heuristic for mixed pipelines (e.g. the English regression probe):
    IPA text virtually always carries a stress mark or one of the non-ASCII
    Hebrew-IPA consonants."""
    return any(c in text for c in "ˈʔʁʃχɡ")
