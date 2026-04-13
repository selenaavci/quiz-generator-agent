from __future__ import annotations

import re
import string
from typing import Any, Dict, List

from metrics import inc


_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”’‘…•–—")
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    t = s.lower().translate(_PUNCT_TABLE)
    return _WS_RE.sub(" ", t).strip()


def text_signature(s: str) -> str:
    import hashlib
    return hashlib.md5(normalize_text(s).encode("utf-8")).hexdigest()


def too_similar(candidate: str, seen_norms: List[str], threshold: float = 0.92) -> bool:
    a = set(normalize_text(candidate).split())
    if not a:
        return False
    for old in seen_norms[-30:]:
        b = set(old.split())
        if not b:
            continue
        jacc = len(a & b) / max(len(a | b), 1)
        if jacc >= threshold:
            return True
    return False


_TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9\-]{3,}")
_GROUND_STOPWORDS = {
    "ve", "veya", "ile", "bir", "bu", "şu", "o", "için", "olarak", "gibi",
    "de", "da", "mi", "mı", "mu", "mü",
    "the", "is", "are", "of", "in", "and", "to",
}


def _tokens(text: str) -> List[str]:
    return [
        w for w in _TOKEN_RE.findall((text or "").lower())
        if w not in _GROUND_STOPWORDS
    ]


def is_grounded_to_context(question: str, context: str, min_ratio: float = 0.18) -> bool:
    q_tokens = _tokens(question)
    c_tokens = set(_tokens(context))
    if not q_tokens:
        return False
    if not c_tokens:
        return True
    overlap = sum(1 for w in q_tokens if w in c_tokens)
    return (overlap / len(q_tokens)) >= min_ratio


_ENGLISH_MARKERS = (
    " the ", " is ", " are ", " of ", " in ", " and ",
    " which ", " what ", " how ", " when ",
)


def is_probably_english(text: str) -> bool:
    if not text:
        return False
    low = f" {text.lower()} "
    return any(m in low for m in _ENGLISH_MARKERS)


_ABSOLUTE_PAT = re.compile(
    r"\b(her zaman|asla|kesinlikle|mutlaka|tamamen|daima|hiçbir zaman|istisnasız)\b",
    re.IGNORECASE,
)
_NEG_PAT = re.compile(
    r"\b(değil|değildir|olmaz|içermez|yapılamaz|yasaktır|mümkün değildir|zorunlu değildir)\b",
    re.IGNORECASE,
)


def has_absolute_language(text: str) -> bool:
    return bool(_ABSOLUTE_PAT.search(text or ""))


def absolute_supported_by_context(question: str, context: str) -> bool:
    matches = _ABSOLUTE_PAT.findall((question or "").lower())
    if not matches:
        return True
    c = (context or "").lower()
    return any(m.lower() in c for m in matches)


def is_negative_sentence(text: str) -> bool:
    return bool(_NEG_PAT.search(text or ""))


_DEF_PATTERNS = [
    r"\bdenir\b", r"\bolarak\b", r"\bifade eder\b",
    r"\btanımlan(ır|ir)\b", r"\bşudur\b", r"\bis\b", r"\bare\b",
]
_EXCEPTION_PATTERNS = [
    r"\bdeğildir\b", r"\byanlıştır\b", r"\bistisna\b",
    r"\bsadece\b", r"\bharic\b", r"\bdışında\b",
]
_LISTY_HINTS = [r":\s*$", r"\b1\)|\b2\)|\b3\)", r"•", r"-\s"]


def score_paragraph_for_type(paragraph: str, qtype: str) -> float:
    if not paragraph:
        return -1e9

    p = " ".join(paragraph.strip().split())
    low = p.lower()
    wc = len(p.split())

    if wc < 18:
        return -8.0

    score = 0.0
    if wc < 28:
        score += -3.5 if qtype == "mcq" else 1.0
    if 35 <= wc <= 120:
        score += 2.0
    elif 121 <= wc <= 180:
        score += 1.0
    elif wc > 220:
        score -= 1.5

    def_hits = sum(1 for pat in _DEF_PATTERNS if re.search(pat, low))
    exc_hits = sum(1 for pat in _EXCEPTION_PATTERNS if re.search(pat, low))
    listy = any(re.search(pat, p) for pat in _LISTY_HINTS)

    if qtype == "mcq":
        score += 0.8 * def_hits + 1.0 * exc_hits
        if listy:
            score += 1.0
        if any(x in low for x in (
            "amacı", "sonucu", "koşul", "şart", "neden", "hangi durumda", "gereklidir",
        )):
            score += 2.0

    elif qtype == "tf":
        score += 1.2 * def_hits + 1.8 * exc_hits
        if any(x in low for x in (
            "değildir", "yanlıştır", "yalnızca", "ancak", "hariç", "dışında",
        )):
            score += 2.0

    elif qtype == "fill":
        score += 2.0 * def_hits
        if listy:
            score -= 2.0
        if any(x in low for x in (
            "denir", "olarak", "ifade eder", "tanımlanır", "amacı", "sağlar",
        )):
            score += 2.5

    return score


_BANNED_MCQ_PHRASES = (
    "hepsi", "yukarıdakilerin hepsi", "hiçbiri",
    "all of the above", "none of the above",
)


def options_too_similar(options: Dict[str, str], threshold: float = 0.80) -> bool:
    norms: Dict[str, set] = {}
    for k, v in options.items():
        tokens = set(normalize_text(str(v)).split())
        norms[k] = {t for t in tokens if len(t) >= 3}

    keys = list(norms.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = norms[keys[i]], norms[keys[j]]
            if not a or not b:
                continue
            if len(a & b) / max(len(a | b), 1) >= threshold:
                return True
    return False


def mcq_is_valid(mcq: Dict[str, Any]) -> bool:
    if not isinstance(mcq, dict):
        return False

    question = str(mcq.get("question", "")).strip()
    options = mcq.get("options")
    correct = mcq.get("correct")

    if not question or not isinstance(options, dict):
        return False
    if correct not in {"A", "B", "C", "D"}:
        return False

    values: List[str] = []
    for key in ("A", "B", "C", "D"):
        v = str(options.get(key, "")).strip()
        if not v:
            return False
        values.append(normalize_text(v))

    if len(set(values)) < 4:
        return False
    if options_too_similar(options, threshold=0.80):
        return False
    if any(any(b in v for b in _BANNED_MCQ_PHRASES) for v in values):
        return False

    return True


def mcq_verify_is_blocking(verify: Dict[str, Any]) -> bool:
    if not isinstance(verify, dict):
        return False
    if verify.get("pass") is True:
        return False

    fix = str(((verify.get("suggestion") or {}).get("fix") or "")).strip().lower()
    if fix in {"rewrite_question", "regenerate_question"}:
        return True

    reason = str(verify.get("reason") or verify.get("error") or "").lower()
    hard_markers = (
        "incorrect", "wrong", "not in context", "halluc", "contradict",
        "yanlış", "bağlam dışı", "uyuşm",
    )
    return any(m in reason for m in hard_markers)


_BLANK_RE = re.compile(r"_{4,}")
_WORD_RE = re.compile(r"^[\wçğıöşüÇĞİÖŞÜ\-]+$", re.UNICODE)

_TR_STOPWORDS = {
    "ve", "veya", "ile", "ya", "da", "de", "ki",
    "mi", "mı", "mu", "mü",
    "bu", "şu", "o", "bir", "biri", "olarak", "için", "gibi", "daha",
    "en", "çok", "az", "her", "tüm", "bazı", "şekilde", "kadar",
    "ancak", "fakat", "ama", "çünkü", "dolayı", "sonra", "önce",
}
_GENERIC_ABSTRACT = {
    "şey", "durum", "süreç", "yöntem", "bilgi", "veri", "sistem", "uygulama",
    "konu", "işlem", "amaç", "kural", "madde", "husus", "unsur", "kapsam",
    "örnek", "genel", "temel", "ilke", "politika", "prosedür",
}


def normalize_blank(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return q
    return _WS_RE.sub(" ", _BLANK_RE.sub("_____", q))


def is_generic_fill_answer(answer: str, context: str = "") -> bool:
    a = (answer or "").strip().lower()
    if not a or len(a) <= 2:
        return True
    if a in _TR_STOPWORDS or a in _GENERIC_ABSTRACT:
        return True
    if " " not in a and not _WORD_RE.match(a):
        return True
    if context and len(a) <= 4:
        freq = len(re.findall(rf"\b{re.escape(a)}\b", context.lower()))
        if freq >= 3:
            return True
    return False


def salvage_fill(parsed: Dict[str, Any], source_sentence: str, metrics: Dict = None) -> Dict[str, Any]:
    q = normalize_blank(str(parsed.get("question", "") or ""))
    a = str(parsed.get("answer", "") or "").strip()
    if not q or not a:
        return parsed

    if "_____" not in q and source_sentence and a.lower() in source_sentence.lower():
        pattern = re.compile(re.escape(a), re.IGNORECASE)
        replaced, count = pattern.subn("_____", source_sentence, count=1)
        if count > 0:
            parsed["question"] = normalize_blank(replaced)
            inc(metrics, "salvage_triggered_count")
            return parsed

    parsed["question"] = q
    return parsed


def is_good_fill(parsed: Dict[str, Any], metrics: Dict = None, context: str = "") -> bool:
    if not isinstance(parsed, dict):
        return False

    q = normalize_blank(str(parsed.get("question", "") or ""))
    a = str(parsed.get("answer", "") or "").strip()

    if is_generic_fill_answer(a, context=context):
        inc(metrics, "fill_generic_answer_rejected")
        return False
    if not q or not a:
        return False

    blank_count = q.count("_____")
    if blank_count < 1 or blank_count > 2:
        return False
    if len(a.split()) > 6:
        return False

    q_without_blank = q.replace("_____", " ")
    if re.search(rf"\b{re.escape(a.lower())}\b", q_without_blank.lower()):
        return False

    return True


def source_reuse_cap(total_sources: int, requested_questions: int) -> int:
    if total_sources >= max(requested_questions * 2, 16):
        return 1
    if total_sources >= 10:
        return 2
    return 3
