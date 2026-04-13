from __future__ import annotations

from typing import Any, Dict


MetricsDict = Dict[str, Any]


def new_metrics() -> MetricsDict:
    return {
        "preprocessing_total_paragraphs": 0,
        "preprocessing_merged_paragraphs": 0,
        "preprocessing_selected_paragraphs": 0,

        "llm_call_count": 0,
        "question_generation_retry_count": 0,
        "salvage_triggered_count": 0,
        "language_guard_triggered": 0,

        "difficulty_count_1": 0,
        "difficulty_count_2": 0,
        "difficulty_count_3": 0,
        "difficulty_count_4": 0,
        "difficulty_count_5": 0,

        "mcq_total": 0,
        "mcq_multistage_success": 0,
        "mcq_multistage_fail": 0,
        "mcq_fallback_legacy": 0,
        "mcq_regen_distractors": 0,
        "mcq_verify_fail": 0,
        "mcq_option_guard_fail": 0,
        "mcq_rewrite_question_suggested": 0,

        "tf_total": 0,
        "tf_success": 0,
        "tf_fail": 0,
        "tf_target_mismatch": 0,
        "tf_absolute_guard_triggered": 0,
        "tf_negative_count": 0,
        "tf_positive_count": 0,

        "fill_total": 0,
        "fill_success": 0,
        "fill_fail": 0,
        "fill_quality_fail": 0,
        "fill_generic_answer_rejected": 0,

        "coverage_total_sources": 0,
        "coverage_unique_sources_used": 0,
        "coverage_ratio": 0.0,
        "coverage_per_question": 0.0,
        "source_reuse_rate": 0.0,
        "coverage_top_reused": [],
        "coverage_avg_reuse": 0.0,

        "skip_recent_source": 0,
        "skip_max_per_source": 0,
        "skip_too_similar": 0,
        "skipped_questions": 0,
    }


def inc(metrics: MetricsDict, key: str, n: int = 1) -> None:
    if metrics is None:
        return
    metrics[key] = int(metrics.get(key, 0)) + n
