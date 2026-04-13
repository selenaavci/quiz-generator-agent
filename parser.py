from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_FALLBACK_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clean(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = _FENCE_RE.sub("", t)
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    return _TRAILING_COMMA_RE.sub(r"\1", t).strip()


def _balanced_object(text: str) -> Optional[str]:
    depth = 0
    start = -1
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _try_literal_dict(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return {str(k): v for k, v in obj.items()}
    except Exception:
        return None
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    cleaned = _clean(text)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    obj = _try_literal_dict(cleaned)
    if obj is not None:
        return obj

    chunk = _balanced_object(cleaned)
    if chunk is None:
        m = _FALLBACK_JSON_RE.search(cleaned)
        chunk = m.group(0) if m else None
    if chunk is None:
        return None

    chunk = _TRAILING_COMMA_RE.sub(r"\1", chunk)
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return _try_literal_dict(chunk)


def _find_label(text: str, labels: List[str]) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    for label in labels:
        m = re.search(rf"(?im)^\s*{label}\s*[:\-]\s*(.+)$", t)
        if m:
            return m.group(1).strip().strip('"')
    return ""


def _find_list(text: str, labels: List[str]) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    for label in labels:
        m = re.search(rf"(?ims)^\s*{label}\s*[:\-]\s*(.+)$", t)
        if not m:
            continue
        blob = m.group(1).strip()
        quoted = re.findall(r'"([^\"]+)"', blob)
        if quoted:
            return [x.strip() for x in quoted if x.strip()]
        parts = re.split(r"[;,\n]+", blob)
        return [p.strip(' -•\"') for p in parts if p.strip(' -•\"')]
    return []


def parse_mcq(text: str) -> Dict[str, Any]:
    obj = extract_json_object(text)
    if obj and "question" in obj and "options" in obj and "correct" in obj:
        obj.setdefault("type", "mcq")
        return obj

    if not text:
        raise ValueError("MCQ parse edilemedi: boş çıktı")

    question = _find_label(text, ["Soru", "Question"])
    if not question:
        m = re.search(r"Soru\s*[:\-]\s*(.+)", text)
        question = m.group(1).strip() if m else ""

    def opt(letter: str) -> Optional[str]:
        m = re.search(rf"^{letter}\s*[\)\.\-\:]\s*(.+)$", text, re.MULTILINE)
        return m.group(1).strip() if m else None

    options = {k: opt(k) for k in ("A", "B", "C", "D")}
    if not all(options.values()):
        raise ValueError(f"MCQ parse edilemedi: şıklar eksik | {text[:200]}")

    m = re.search(r"(?:Doğru|Doğru\s*cevap)\s*[:\-]\s*([A-D])", text, re.IGNORECASE)
    correct = m.group(1).strip().upper() if m else ""

    if not question or not correct:
        raise ValueError(f"MCQ parse edilemedi: soru/doğru yok | {text[:200]}")

    m = re.search(r"(?:Açıklama|Gerekçe)\s*[:\-]\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else ""

    return {
        "type": "mcq",
        "question": question,
        "options": options,
        "correct": correct,
        "explanation": explanation,
    }


def parse_true_false(text: str) -> Dict[str, Any]:
    obj = extract_json_object(text)
    if obj and "question" in obj and "answer" in obj:
        obj.setdefault("type", "true_false")
        return obj

    if not text:
        raise ValueError("TF parse edilemedi: boş çıktı")

    q = re.search(r"Soru:\s*(.+)", text)
    ans = re.search(
        r"Cevap:\s*(Doğru|Yanlış|Dogru|Yanlis|True|False)",
        text,
        re.IGNORECASE,
    )
    if not q or not ans:
        raise ValueError(f"TF parse edilemedi: format uyumsuz | {text[:200]}")

    raw_ans = ans.group(1).strip().lower()
    answer = "Doğru" if raw_ans in ("doğru", "dogru", "true") else "Yanlış"

    m = re.search(r"(?:Açıklama|Aciklama):\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else ""

    return {
        "type": "true_false",
        "question": q.group(1).strip(),
        "answer": answer,
        "explanation": explanation,
    }


def parse_fill(text: str) -> Dict[str, Any]:
    obj = extract_json_object(text)
    if obj and "question" in obj and "answer" in obj:
        obj.setdefault("type", "fill")
        return obj

    if not text:
        raise ValueError("Fill parse edilemedi: boş çıktı")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def pick(prefix: str) -> str:
        for ln in lines:
            if ln.lower().startswith(prefix.lower()):
                return ln.split(":", 1)[1].strip() if ":" in ln else ln[len(prefix):].strip()
        return ""

    question = pick("Soru")
    answer = pick("Cevap")
    if not question or not answer:
        raise ValueError(f"Fill parse edilemedi: format uyumsuz | {text[:200]}")

    explanation = ""
    for i, ln in enumerate(lines):
        head = ln.lower()
        if head.startswith("açıklama") or head.startswith("aciklama"):
            first = ln.split(":", 1)[1].strip() if ":" in ln else ""
            rest: List[str] = []
            for j in range(i + 1, len(lines)):
                if lines[j].split(":", 1)[0].lower() in ("soru", "cevap", "açıklama", "aciklama"):
                    break
                rest.append(lines[j])
            explanation = " ".join([first] + rest).strip()
            break

    return {
        "type": "fill",
        "question": question,
        "answer": answer,
        "explanation": explanation,
    }


def parse_mcq_stage1(raw: str) -> Dict[str, Any]:
    obj = extract_json_object(raw) or {}
    if obj.get("question") and obj.get("correct_answer"):
        return obj

    q = _find_label(raw, ["question", "soru"])
    a = _find_label(raw, [
        "correct_answer", "correct answer",
        "doğru cevap", "dogru cevap", "cevap",
    ])
    r = _find_label(raw, [
        "rationale", "explanation",
        "gerekçe", "gerekce", "açıklama", "aciklama",
    ])
    t = _find_label(raw, ["answer_type", "answer type", "cevap_tipi", "cevap tipi"]) or "definition"

    out = {"question": q, "correct_answer": a, "rationale": r, "answer_type": t}
    return {k: v for k, v in out.items() if v}


def parse_mcq_stage2(raw: str) -> Dict[str, Any]:
    obj = extract_json_object(raw) or {}
    ds = obj.get("distractors") if isinstance(obj, dict) else None
    if isinstance(ds, list) and len(ds) >= 3:
        obj["distractors"] = [str(x).strip() for x in ds if str(x).strip()][:3]
        return obj

    distractors = _find_list(raw, [
        "distractors", "seçenekler", "secenekler",
        "yanlış seçenekler", "yanlis secenekler",
    ])
    if len(distractors) < 3:
        line_items: List[str] = []
        for ln in (raw or "").splitlines():
            m = re.match(r"^\s*(?:[-•*]|[A-C1-3][\)\.:\-])\s*(.+?)\s*$", ln.strip())
            if m:
                line_items.append(m.group(1).strip())
        distractors = distractors or line_items

    if len(distractors) >= 3:
        return {"distractors": distractors[:3]}
    return {}


def parse_mcq_stage3(raw: str) -> Dict[str, Any]:
    obj = extract_json_object(raw) or {}
    if "pass" in obj:
        return obj

    txt = (raw or "").lower()
    passed: Optional[bool] = None
    if re.search(r"\bpass\s*[:\-]\s*true\b", txt) or "tek doğru" in txt or "uygun" in txt:
        passed = True
    elif re.search(r"\bpass\s*[:\-]\s*false\b", txt) or "birden fazla doğru" in txt or "belirsiz" in txt:
        passed = False
    if passed is None:
        return {}

    return {
        "pass": passed,
        "issues": _find_list(raw, ["issues", "sorunlar"]),
        "suggestion": {
            "fix": _find_label(raw, ["fix", "düzeltme", "duzeltme"]) or "none",
            "notes": _find_label(raw, ["notes", "not", "öneri", "oneri"]),
        },
    }
