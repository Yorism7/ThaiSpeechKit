import re
from typing import List

_DEF_MAX_CHARS = 200


def normalize_text(txt: str) -> str:
    # Collapse whitespace, remove control chars
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def split_text_th(text: str, max_chars: int = _DEF_MAX_CHARS) -> List[str]:
    text = normalize_text(text)
    # Split by Thai/Latin punctuation to help prosody
    parts = re.split(r"([.!?…]|[ๆฯ]|[\u0E2F\u0E46])", text)
    merged = ["".join(parts[i : i + 2]).strip() for i in range(0, len(parts), 2)]
    chunks, buf = [], ""
    for seg in merged:
        if not seg:
            continue
        if len(buf) + 1 + len(seg) <= max_chars:
            buf = (buf + " " + seg).strip() if buf else seg
        else:
            if buf:
                chunks.append(buf.strip())
            buf = seg
    if buf:
        chunks.append(buf.strip())

    # Merge too-short trailing segments into previous
    final: List[str] = []
    for c in chunks:
        if final and len(c) < 40:
            final[-1] = (final[-1] + " " + c).strip()
        else:
            final.append(c)
    return [c for c in final if c]

