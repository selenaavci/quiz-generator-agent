from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Optional


_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LIST_HEAD_SPLIT_RE = re.compile(
    r"(?m)(?<=\S)\n(?=(?:[-•*]\s+|\d+[\).]\s+|[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü0-9 ,/%()\-]{4,}:))"
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _signature(text: str) -> str:
    norm = _normalize_ws(text).lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def split_into_sentences(text: str) -> List[str]:
    t = _nfc(text).replace("\n", " ").strip()
    return [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]


def _looks_like_table_or_noise(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True

    if "|" in t and len(t.split("|")) >= 3:
        return True
    if "\t" in t:
        return True

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 3:
        colonish = sum(1 for ln in lines if ":" in ln and len(ln.split()) <= 8)
        if colonish >= 2:
            return True

    words = t.split()
    if not words:
        return True

    digit_ratio = sum(ch.isdigit() for ch in t) / max(len(t), 1)
    if digit_ratio > 0.35:
        return True

    upper_tokens = sum(1 for w in words if len(w) >= 3 and w.isupper())
    if upper_tokens / max(len(words), 1) > 0.60:
        return True

    return False


def split_paragraphs(text: str) -> List[str]:
    t = _nfc(text).replace("\r\n", "\n").replace("\r", "\n")
    t = _LIST_HEAD_SPLIT_RE.sub("\n\n", t)

    raw = re.split(r"\n\s*\n+", t)
    cleaned = [_normalize_ws(p) for p in raw if p.strip()]

    if len(cleaned) <= 1:
        sents = split_into_sentences(t)
        if len(sents) > 6:
            grouped: List[str] = []
            bucket: List[str] = []
            for s in sents:
                bucket.append(s)
                if len(bucket) >= 4 or sum(len(x.split()) for x in bucket) >= 90:
                    grouped.append(" ".join(bucket).strip())
                    bucket = []
            if bucket:
                grouped.append(" ".join(bucket).strip())
            cleaned = grouped or cleaned

    return [p for p in cleaned if not _looks_like_table_or_noise(p)]


def merge_small_paragraphs(paragraphs: List[str], min_words: int = 40) -> List[str]:
    merged: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            merged.append(" ".join(buffer).strip())
            buffer = []

    for para in paragraphs:
        if len(para.split()) < min_words:
            buffer.append(para)
        else:
            flush()
            merged.append(para)

    flush()
    return merged


def chunk_with_overlap(
    paragraph: str,
    max_words: int = 220,
    overlap_words: int = 20,
) -> List[str]:
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]

    if overlap_words < 0 or overlap_words >= max_words:
        overlap_words = max(0, max_words // 5)

    chunks: List[str] = []
    n = len(words)
    start = 0
    while start < n:
        end = min(start + max_words, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start = max(0, end - overlap_words)
    return chunks


def _dedupe(chunks: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for ch in chunks:
        sig = _signature(ch)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(ch)
    return out


_LOW_WORD_KEYWORDS = (
    "denir", "olarak", "tanımlan", "ifade eder",
    "şudur", "amacı", "özelliği", "sağlar",
)


def extract_context_chunks(
    text: str,
    max_words: int = 170,
    min_words: int = 25,
    overlap_words: int = 20,
    metrics: Optional[Dict[str, Any]] = None,
) -> List[str]:
    paragraphs = split_paragraphs(text)
    if metrics is not None:
        metrics["preprocessing_total_paragraphs"] = len(paragraphs)

    paragraphs = merge_small_paragraphs(paragraphs, min_words=min_words)
    if metrics is not None:
        metrics["preprocessing_merged_paragraphs"] = len(paragraphs)

    all_chunks: List[str] = []
    for para in paragraphs:
        if _looks_like_table_or_noise(para):
            continue
        all_chunks.extend(chunk_with_overlap(para, max_words, overlap_words))

    filtered: List[str] = []
    for chunk in all_chunks:
        c = _normalize_ws(chunk)
        if not c or _looks_like_table_or_noise(c):
            continue

        wc = len(c.split())
        if wc >= 12:
            filtered.append(c)
            continue

        if wc >= 8 and any(kw in c.lower() for kw in _LOW_WORD_KEYWORDS):
            filtered.append(c)

    result = _dedupe(filtered or all_chunks)
    if metrics is not None:
        metrics["preprocessing_selected_paragraphs"] = len(result)
    return result


def _is_good_fill_sentence(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s:
        return False

    wc = len(s.split())
    if wc < 8 or wc > 28:
        return False
    if "_______" in s or s.endswith("?"):
        return False
    if s.count(";") >= 2 or s.count(",") >= 6:
        return False

    low = s.lower()
    if any(x in low for x in ("örneğin", "ornegin", "örnek olarak", "example")):
        return False

    has_proper_or_quoted = bool(
        re.search(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b", s)
        or re.search(r"“.+?”|\".+?\"|\(.+?\)", s)
    )
    has_definition_hint = any(
        x in low for x in (
            " olarak ", " denir", " ifade eder", " ölçer",
            " amacı", " sağlar", " saglar",
        )
    )
    return has_proper_or_quoted or has_definition_hint


def pick_fill_sentence(chunk: str) -> Optional[str]:
    candidates = [s for s in split_into_sentences(chunk) if _is_good_fill_sentence(s)]
    if not candidates:
        return None

    priority = (" olarak ", " denir", " ifade eder", " ölçer", " amacı", " sağlar", " saglar")
    for s in candidates:
        low = s.lower()
        if any(m in low for m in priority):
            return s
    return candidates[0]
