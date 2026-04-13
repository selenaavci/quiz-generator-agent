from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from chunker import pick_fill_sentence, split_into_sentences
from difficulty import difficulty_balanced, difficulty_for
from llm_client import LLMClient
from metrics import MetricsDict, inc, new_metrics
from parser import (
    parse_fill,
    parse_mcq,
    parse_mcq_stage1,
    parse_mcq_stage2,
    parse_mcq_stage3,
    parse_true_false,
)
from prompts import (
    prompt_fill,
    prompt_mcq,
    prompt_mcq_stage1_core,
    prompt_mcq_stage2_distractors,
    prompt_mcq_stage3_verify,
    prompt_true_false,
)
from quality import (
    has_absolute_language,
    absolute_supported_by_context,
    is_good_fill,
    is_grounded_to_context,
    is_negative_sentence,
    is_probably_english,
    mcq_is_valid,
    mcq_verify_is_blocking,
    normalize_blank,
    normalize_text,
    salvage_fill,
    score_paragraph_for_type,
    text_signature,
    too_similar,
)


log = logging.getLogger(__name__)


def build_type_plan(mcq: int, tf: int, fill: int) -> List[str]:
    if mcq < 0 or tf < 0 or fill < 0:
        raise ValueError("Soru sayıları negatif olamaz.")

    counts = {"mcq": mcq, "tf": tf, "fill": fill}
    if not sum(counts.values()):
        return []

    priority = {"mcq": 3, "tf": 2, "fill": 1}
    order: List[str] = []
    while sum(counts.values()) > 0:
        best = max(counts.keys(), key=lambda k: (counts[k], priority[k]))
        if counts[best] > 0:
            order.append(best)
            counts[best] -= 1
            continue
        for k in ("mcq", "tf", "fill"):
            if counts[k] > 0:
                order.append(k)
                counts[k] -= 1
                break
    return order


def pick_paragraph(
    paragraphs: List[str],
    qtype: str,
    cursor: int,
    source_use_count: Dict[str, int],
    recent_sources: List[str],
    cooldown_k: int = 3,
) -> Tuple[str, int]:
    n = len(paragraphs)
    if n == 0:
        return "", cursor

    recent_tail = recent_sources[-cooldown_k:] if recent_sources else []
    candidates: List[Tuple[float, int, str]] = []

    for k in range(n):
        idx = (cursor + k) % n
        para = paragraphs[idx]
        base = score_paragraph_for_type(para, qtype)
        sig = text_signature(para)

        uses = int(source_use_count.get(sig, 0))
        reuse_penalty = 3.0 * uses
        recent_penalty = 2.5 if sig in recent_tail else 0.0
        unused_bonus = 4.0 if uses == 0 else 0.0

        candidates.append((base + unused_bonus - reuse_penalty - recent_penalty, idx, para))

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_idx, best_para = candidates[0]
    return best_para, (best_idx + 1) % n


async def _llm_generate(
    client: LLMClient,
    metrics: MetricsDict,
    *,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    inc(metrics, "llm_call_count")
    return await client.generate(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _assemble_mcq(question: str, correct: str, distractors: List[str]) -> Dict[str, Any]:
    letters = ["A", "B", "C", "D"]
    shuffled = [correct.strip()] + [d.strip() for d in distractors]
    random.shuffle(shuffled)
    correct_pos = shuffled.index(correct.strip())

    return {
        "type": "mcq",
        "question": question.strip(),
        "options": {letters[i]: shuffled[i] for i in range(4)},
        "correct": letters[correct_pos],
    }


async def _generate_mcq_multistage(
    client: LLMClient,
    paragraph: str,
    difficulty: int,
    metrics: MetricsDict,
) -> Dict[str, Any]:
    inc(metrics, "mcq_total")

    raw1 = await _llm_generate(
        client,
        metrics,
        messages=[{"role": "user", "content": prompt_mcq_stage1_core(paragraph, difficulty)}],
        temperature=0.2,
        max_tokens=500,
    )
    core = parse_mcq_stage1(raw1)
    question = str(core.get("question", "")).strip()
    correct_answer = str(core.get("correct_answer", "")).strip()
    rationale = str(core.get("rationale", "")).strip()
    answer_type = str(core.get("answer_type", "definition")).strip()

    if not question or not correct_answer:
        raise ValueError("MCQ stage1 başarısız: question/correct_answer boş")

    last_verify: Optional[Dict[str, Any]] = None

    for attempt in range(1, 5):
        if attempt > 1:
            inc(metrics, "mcq_regen_distractors")

        raw2 = await _llm_generate(
            client,
            metrics,
            messages=[{"role": "user", "content": prompt_mcq_stage2_distractors(
                correct_answer=correct_answer,
                answer_type=answer_type,
                rationale=rationale,
                context=paragraph,
            )}],
            temperature=0.6,
            max_tokens=300,
        )
        d2 = parse_mcq_stage2(raw2)
        distractors = [str(x).strip() for x in (d2.get("distractors") or []) if str(x).strip()]
        if len(distractors) < 3:
            continue
        distractors = distractors[:3]

        mcq = _assemble_mcq(question, correct_answer, distractors)

        raw3 = await _llm_generate(
            client,
            metrics,
            messages=[{"role": "user", "content": prompt_mcq_stage3_verify(
                question=mcq["question"],
                options=mcq["options"],
                correct_letter=mcq["correct"],
                context=paragraph,
            )}],
            temperature=0.2,
            max_tokens=300,
        )
        verify = parse_mcq_stage3(raw3)
        last_verify = verify

        if isinstance(verify, dict) and verify.get("pass") is False:
            inc(metrics, "mcq_verify_fail")

        fix = ""
        if isinstance(verify, dict):
            fix = ((verify.get("suggestion") or {}).get("fix") or "").strip()

        if fix == "rewrite_question":
            inc(metrics, "mcq_rewrite_question_suggested")
            break

        if mcq_verify_is_blocking(verify):
            continue

        if mcq_is_valid(mcq):
            inc(metrics, "mcq_multistage_success")
            mcq["explanation"] = rationale
            mcq["mcq_answer_type"] = answer_type
            mcq["difficulty"] = int(difficulty)
            return mcq

        inc(metrics, "mcq_option_guard_fail")

    raise ValueError(f"MCQ multistage başarısız: last_verify={last_verify}")


async def _generate_mcq_legacy(
    client: LLMClient,
    paragraph: str,
    difficulty: int,
    metrics: MetricsDict,
) -> Dict[str, Any]:
    inc(metrics, "mcq_fallback_legacy")

    prompt = prompt_mcq(paragraph, difficulty=difficulty)

    raw = await _llm_generate(
        client, metrics,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=520,
    )
    try:
        out = parse_mcq(raw)
    except Exception:
        raw = await _llm_generate(
            client, metrics,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=620,
        )
        out = parse_mcq(raw)

    q_text = str(out.get("question", "")).strip()
    if is_probably_english(q_text):
        inc(metrics, "language_guard_triggered")
        raise ValueError("MCQ legacy output English")
    if not is_grounded_to_context(q_text, paragraph, min_ratio=0.10):
        raise ValueError("MCQ legacy grounding guard failed")

    opts = out.get("options")
    correct_key = str(out.get("correct", "")).strip().upper()
    if isinstance(opts, dict) and correct_key in opts:
        correct_text = str(opts[correct_key]).strip()
        distractor_texts = [str(opts[k]).strip() for k in sorted(opts.keys()) if k != correct_key]
        if len(distractor_texts) == 3 and correct_text:
            out = _assemble_mcq(q_text, correct_text, distractor_texts)
            out.setdefault("explanation", "")

    out["difficulty"] = int(difficulty)
    return out


async def generate_mcq(
    client: LLMClient,
    paragraph: str,
    difficulty: int,
    metrics: MetricsDict,
) -> Dict[str, Any]:
    try:
        mcq = await _generate_mcq_multistage(client, paragraph, difficulty, metrics)
        if not is_grounded_to_context(str(mcq.get("question", "")), paragraph, min_ratio=0.12):
            raise ValueError("MCQ grounding guard failed")
        return mcq
    except Exception:
        inc(metrics, "mcq_multistage_fail")
        return await _generate_mcq_legacy(client, paragraph, difficulty, metrics)


def _tf_target_for_index(idx: int) -> str:
    return "Yanlış" if (idx % 2 == 0) else "Doğru"


def _tf_answer_matches(target: str, parsed: Dict[str, Any]) -> bool:
    ans = str(parsed.get("answer", "")).strip().lower()
    t = target.strip().lower()
    if t in ("doğru", "dogru"):
        return ans in ("doğru", "dogru", "true")
    if t in ("yanlış", "yanlis"):
        return ans in ("yanlış", "yanlis", "false")
    return True


async def generate_tf(
    client: LLMClient,
    paragraph: str,
    difficulty: int,
    tf_index: int,
    metrics: MetricsDict,
) -> Dict[str, Any]:
    target = _tf_target_for_index(tf_index)
    last_err: Optional[Exception] = None
    last_preview: Optional[str] = None

    for _ in range(4):
        try:
            r = random.random()
            if r < 0.25:
                style_hint = (
                    "\n Stil Notu: Eğer anlamlıysa olumsuz (negatif) yapıda bir ifade "
                    "kurabilirsin (değildir/olmaz/içermez/yapılamaz.)\n"
                )
            elif r < 0.50:
                style_hint = (
                    "\n Stil Notu: Eğer anlamlıysa olumlu yapıda bir ifade kurabilirsin "
                    "(olumsuzluk kullanmadan).\n"
                )
            else:
                style_hint = ""

            full_prompt = (
                prompt_true_false(paragraph, difficulty=difficulty)
                + style_hint
                + f"\n\nEk Kural: Üreteceğin ifadenin cevabı mutlaka '{target}' olmalı. "
                "Cevap formatını bozma."
            )

            raw = await _llm_generate(
                client, metrics,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.25, max_tokens=260,
            )
            last_preview = (raw or "")[:400]
            parsed = parse_true_false(raw)
            q_text = (parsed.get("question") or "").strip()

            if has_absolute_language(q_text) and not absolute_supported_by_context(q_text, paragraph):
                inc(metrics, "tf_absolute_guard_triggered")
                last_err = ValueError("TF absolute guard: mutlak ifade context tarafından desteklenmiyor")
                continue

            if _tf_answer_matches(target, parsed):
                parsed["tf_target"] = target
                parsed["difficulty"] = int(difficulty)
                inc(
                    metrics,
                    "tf_negative_count" if is_negative_sentence(parsed.get("question", "")) else "tf_positive_count",
                )
                return parsed

            last_err = ValueError(f"TF target mismatch (target={target})")
            inc(metrics, "tf_target_mismatch")

        except Exception as e:
            last_err = e

    raise ValueError(f"TF üretimi başarısız: {last_err} | raw={last_preview}")


def _fill_candidate_sentences(paragraph: str, k: int = 3) -> List[str]:
    p = " ".join((paragraph or "").strip().split())
    if not p:
        return []

    out: List[str] = []
    top = pick_fill_sentence(paragraph)
    if top:
        out.append(top.strip())

    sents = [s.strip() for s in split_into_sentences(p) if s.strip()]
    if not sents:
        return out or [p]

    def score(s: str) -> int:
        words = s.split()
        wc = len(words)
        if wc < 8 or wc > 32:
            return -999
        low = s.lower()
        bonus = 0
        if any(x in low for x in (" olarak ", " denir", " ifade eder")):
            bonus += 6
        if "dır" in low or "dir" in low:
            bonus += 6
        if "," in s:
            bonus += 2
        digit_ratio = sum(ch.isdigit() for ch in s) / max(len(s), 1)
        if digit_ratio > 0.05:
            bonus -= 5
        mid_bonus = 10 - abs(wc - 18)
        return bonus + mid_bonus

    for s in sorted(sents, key=score, reverse=True):
        if s and s not in out:
            out.append(s)
        if len(out) >= k:
            break

    return out[:k] if out else [p]


async def generate_fill(
    client: LLMClient,
    paragraph: str,
    difficulty_setting: Union[int, str],
    question_idx: int,
    metrics: MetricsDict,
) -> Dict[str, Any]:
    candidates = _fill_candidate_sentences(paragraph, k=3)
    last_err: Optional[Exception] = None
    last_preview: Optional[str] = None

    for attempt, sentence in enumerate(candidates, start=1):
        try:
            d = difficulty_for(difficulty_setting, question_idx * 10 + attempt)

            raw = await _llm_generate(
                client, metrics,
                messages=[{"role": "user", "content": prompt_fill(sentence, difficulty=d)}],
                temperature=0.2, max_tokens=360,
            )
            last_preview = (raw or "")[:400]

            parsed = parse_fill(raw)
            parsed = salvage_fill(parsed, source_sentence=sentence, metrics=metrics)
            parsed["question"] = normalize_blank(parsed.get("question", ""))

            if is_good_fill(parsed, metrics=metrics, context=paragraph):
                parsed["difficulty"] = int(d)
                return parsed

            inc(metrics, "fill_quality_fail")
            last_err = ValueError("Fill kalite kontrolünden geçemedi")

        except Exception as e:
            last_err = e

    raise ValueError(f"Fill üretimi başarısız: {last_err} | raw={last_preview}")


async def generate_one_question(
    *,
    client: LLMClient,
    qtype: str,
    paragraph: str,
    difficulty_setting: Union[int, str],
    question_idx: int,
    tf_index: Optional[int],
    metrics: MetricsDict,
) -> Dict[str, Any]:
    d = difficulty_balanced(difficulty_setting, question_idx, metrics)

    if qtype == "mcq":
        return await generate_mcq(client, paragraph, d, metrics)

    if qtype == "tf":
        inc(metrics, "tf_total")
        try:
            out = await generate_tf(
                client=client,
                paragraph=paragraph,
                difficulty=d,
                tf_index=(tf_index if tf_index is not None else question_idx),
                metrics=metrics,
            )
            if not is_grounded_to_context(str(out.get("question", "")), paragraph, min_ratio=0.10):
                raise ValueError("TF grounding guard failed")
            inc(metrics, "tf_success")
            return out
        except Exception:
            inc(metrics, "tf_fail")
            raise

    if qtype == "fill":
        inc(metrics, "fill_total")
        try:
            out = await generate_fill(
                client=client,
                paragraph=paragraph,
                difficulty_setting=d,
                question_idx=question_idx,
                metrics=metrics,
            )
            if not is_grounded_to_context(str(out.get("question", "")), paragraph, min_ratio=0.10):
                raise ValueError("Fill grounding guard failed")
            inc(metrics, "fill_success")
            return out
        except Exception:
            inc(metrics, "fill_fail")
            raise

    raise ValueError(f"Unknown qtype: {qtype}")


@dataclass
class _QuizSession:
    paragraphs: List[str]
    type_plan: List[str]
    metrics: MetricsDict

    seen_question_sigs: set = field(default_factory=set)
    seen_question_norms: List[str] = field(default_factory=list)
    seen_source_sigs: set = field(default_factory=set)
    source_use_count: Dict[str, int] = field(default_factory=dict)
    recent_sources: List[str] = field(default_factory=list)
    all_used_sources: List[str] = field(default_factory=list)

    cursor: int = 0
    tf_counter: int = 0

    @property
    def scarce(self) -> bool:
        total = len(self.type_plan)
        return len(self.paragraphs) < total * 2

    @property
    def max_per_source(self) -> int:
        return 2 if len(self.paragraphs) >= 18 else 3

    @property
    def cooldown_k(self) -> int:
        return 3 if len(self.paragraphs) >= 12 else 2

    @property
    def sim_threshold(self) -> float:
        return 0.94 if len(self.paragraphs) >= 12 else 0.96


def _finalize_coverage(session: _QuizSession) -> None:
    m = session.metrics
    total_sources = len(session.paragraphs)
    unique_used = len(set(session.all_used_sources))

    m["coverage_total_sources"] = total_sources
    m["coverage_unique_sources_used"] = unique_used
    m["coverage_ratio"] = round(unique_used / total_sources if total_sources else 0.0, 4)

    total_q = max(len(session.all_used_sources), 1)
    m["coverage_per_question"] = round(unique_used / total_q, 4)
    m["source_reuse_rate"] = round(1.0 - (unique_used / total_q), 4)

    if session.all_used_sources:
        top = sorted(session.source_use_count.items(), key=lambda x: x[1], reverse=True)[:5]
        m["coverage_top_reused"] = top
        m["coverage_avg_reuse"] = round(
            sum(session.source_use_count.values()) / max(unique_used, 1), 4
        )
    else:
        m["coverage_top_reused"] = []
        m["coverage_avg_reuse"] = 0.0


async def generate_quiz(
    paragraphs: List[str],
    mcq_count: int,
    tf_count: int,
    fill_count: int,
    *,
    difficulty: Union[int, str] = "Orta",
    metrics: Optional[MetricsDict] = None,
) -> List[Dict[str, Any]]:
    if not paragraphs:
        return [{"type": "error", "error": "Paragraph list is empty"}]

    if metrics is None:
        metrics = new_metrics()

    session = _QuizSession(
        paragraphs=paragraphs,
        type_plan=build_type_plan(mcq_count, tf_count, fill_count),
        metrics=metrics,
    )
    quiz: List[Dict[str, Any]] = []

    async with LLMClient() as client:
        for i, qtype in enumerate(session.type_plan, start=1):
            try:
                q = await _try_one_slot(session, client, qtype, i, difficulty)
                if q is not None:
                    quiz.append(q)
                else:
                    inc(metrics, "skipped_questions")
            except Exception as e:
                log.warning("Soru %d (%s) üretilemedi: %s", i, qtype, e)
                inc(metrics, "skipped_questions")

    _finalize_coverage(session)
    return quiz


async def _try_one_slot(
    session: _QuizSession,
    client: LLMClient,
    qtype: str,
    slot_idx: int,
    difficulty: Union[int, str],
    max_tries: int = 10,
) -> Optional[Dict[str, Any]]:
    m = session.metrics
    last_err: Optional[Exception] = None

    for tries in range(1, max_tries + 1):
        if tries > 1:
            inc(m, "question_generation_retry_count")

        paragraph, session.cursor = pick_paragraph(
            session.paragraphs, qtype, session.cursor,
            source_use_count=session.source_use_count,
            recent_sources=session.recent_sources,
            cooldown_k=session.cooldown_k,
        )
        if not paragraph:
            last_err = ValueError("Empty paragraph")
            continue

        src_sig = text_signature(paragraph)
        src_preview = (paragraph[:200] + "...") if paragraph else ""

        cooldown_limit = 2 if session.scarce else 3
        if tries <= cooldown_limit and src_sig in session.recent_sources[-session.cooldown_k:]:
            inc(m, "skip_recent_source")
            continue

        reuse_limit = 4 if session.scarce else 6
        uses = int(session.source_use_count.get(src_sig, 0))
        if uses >= session.max_per_source and tries <= reuse_limit:
            inc(m, "skip_max_per_source")
            continue

        seen_limit = 2 if session.scarce else 3
        if qtype in ("mcq", "fill") and tries <= seen_limit and src_sig in session.seen_source_sigs:
            continue

        tf_local_index: Optional[int] = None
        if qtype == "tf":
            session.tf_counter += 1
            tf_local_index = session.tf_counter

        try:
            q = await generate_one_question(
                client=client,
                qtype=qtype,
                paragraph=paragraph,
                difficulty_setting=difficulty,
                question_idx=slot_idx * 10 + tries,
                tf_index=tf_local_index,
                metrics=m,
            )
        except Exception as e:
            last_err = e
            continue

        q_text = str(q.get("question", "")).strip()
        if not q_text:
            last_err = ValueError("Üretilen soru metni boş")
            continue

        q_norm = normalize_text(q_text)
        q_sig = text_signature(q_norm)

        if q_sig in session.seen_question_sigs:
            continue
        if session.seen_question_norms and too_similar(
            q_text, session.seen_question_norms, threshold=session.sim_threshold
        ):
            inc(m, "skip_too_similar")
            continue

        q["source"] = src_preview
        try:
            d_int = int(q.get("difficulty"))
        except Exception:
            d_int = None
        if d_int in (1, 2, 3, 4, 5):
            inc(m, f"difficulty_count_{d_int}")

        session.seen_question_sigs.add(q_sig)
        session.seen_question_norms.append(q_norm)
        session.source_use_count[src_sig] = uses + 1
        session.recent_sources.append(src_sig)
        session.all_used_sources.append(src_sig)
        if qtype != "tf":
            session.seen_source_sigs.add(src_sig)

        return q

    log.warning("Slot %d (%s) %d denemede tamamlanamadı: %s", slot_idx, qtype, max_tries, last_err)
    return None


async def generate_quiz_with_metrics(
    paragraphs: List[str],
    mcq_count: int,
    tf_count: int,
    fill_count: int,
    *,
    difficulty: Union[int, str] = "Orta",
    preprocessing_metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], MetricsDict]:
    metrics = new_metrics()
    metrics["preprocessing_selected_paragraphs"] = len(paragraphs or [])

    if isinstance(preprocessing_metrics, dict):
        for k in (
            "preprocessing_total_paragraphs",
            "preprocessing_merged_paragraphs",
            "preprocessing_selected_paragraphs",
        ):
            if k in preprocessing_metrics:
                metrics[k] = int(preprocessing_metrics.get(k, 0))
    elif paragraphs:
        metrics["preprocessing_total_paragraphs"] = len(paragraphs)
        metrics["preprocessing_merged_paragraphs"] = len(paragraphs)

    try:
        quiz = await generate_quiz(
            paragraphs=paragraphs,
            mcq_count=mcq_count,
            tf_count=tf_count,
            fill_count=fill_count,
            difficulty=difficulty,
            metrics=metrics,
        )
        return quiz, metrics
    except Exception as e:
        metrics["fatal_error"] = str(e)
        return [{"type": "error", "error": f"Quiz generation crashed: {e}"}], metrics
