from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


AVERAGE_CUE_SECONDS = 3.4
MAX_ESTIMATED_SEGMENTS = 200_000


def _bounded_int(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _calls(item_count: int, batch_size: int) -> int:
    return math.ceil(item_count / batch_size) if item_count > 0 else 0


def _model_seconds_per_call(model: str) -> tuple[float, float]:
    normalized = str(model or "").casefold()
    if "122b" in normalized:
        return 35.0, 150.0
    if "30b" in normalized or "32b" in normalized:
        return 15.0, 75.0
    if "14b" in normalized:
        return 10.0, 60.0
    if any(token in normalized for token in ("0.6b", "1.5b", "3b", "7b", "8b")):
        return 5.0, 35.0
    return 15.0, 90.0


def estimate_run_workload(
    *,
    total_segments: Any = None,
    completed_segments: Any = 0,
    video_duration_seconds: Any = None,
    batch_size: Any = 5,
    autonomous_review_rounds: Any = 0,
    llm_model: str = "",
    run_mode: str = "resume",
    operations: Iterable[str] | None = None,
    source_cache_reused: bool = False,
    artifact_cache: Mapping[str, Any] | None = None,
    checkpoint_compatible: bool | None = None,
) -> dict[str, Any]:
    exact_total = _bounded_int(total_segments, 0, 0, MAX_ESTIMATED_SEGMENTS)
    duration = _positive_float(video_duration_seconds)
    if exact_total > 0:
        segment_count = exact_total
        segment_basis = "source"
    elif duration is not None:
        segment_count = _bounded_int(
            round(duration / AVERAGE_CUE_SECONDS),
            12,
            1,
            MAX_ESTIMATED_SEGMENTS,
        )
        segment_basis = "duration"
    else:
        return {
            "available": False,
            "segment_count": None,
            "segment_basis": "unknown",
            "reason": "subtitle_count_unavailable",
        }

    effective_batch = _bounded_int(batch_size, 5, 1, 100)
    rounds = _bounded_int(autonomous_review_rounds, 0, 0, 3)
    completed = _bounded_int(completed_segments, 0, 0, segment_count)
    mode = str(run_mode or "resume").strip().casefold()
    can_resume = mode == "resume" and checkpoint_compatible is not False
    primary_items = max(0, segment_count - completed) if can_resume else segment_count
    review_batch = max(8, effective_batch * 2)

    cache = dict(artifact_cache or {})
    stable_resume = can_resume and primary_items == 0
    style_cached = stable_resume and bool(cache.get("style_guide_exists"))
    terminology_cached = stable_resume and bool(cache.get("terminology_review_exists"))
    final_cached = stable_resume and bool(cache.get("final_qa_exists"))
    cached_rounds = (
        _bounded_int(cache.get("autonomous_rounds_completed"), 0, 0, 3)
        if stable_resume
        else 0
    )

    primary_calls = _calls(primary_items, effective_batch)
    style_calls = 0 if segment_count < 12 or style_cached else 1
    terminology_calls = 0 if terminology_cached else 1
    final_calls = 0
    if segment_count >= 12 and not final_cached:
        final_calls = _calls(segment_count, review_batch)
    autonomous_calls = 0
    if segment_count >= 12 and rounds > 0:
        autonomous_calls = max(0, rounds - cached_rounds) * _calls(
            segment_count,
            review_batch,
        )

    breakdown = {
        "style": style_calls,
        "primary": primary_calls,
        "terminology": terminology_calls,
        "final_review": final_calls,
        "autonomous_review": autonomous_calls,
    }
    total_calls = sum(breakdown.values())

    low_seconds, high_seconds = _model_seconds_per_call(llm_model)
    min_minutes = total_calls * low_seconds / 60.0
    max_minutes = total_calls * high_seconds / 60.0

    operation_set = {str(item) for item in (operations or [])}
    source_min_minutes = 0.0
    source_max_minutes = 0.0
    source_must_run = mode == "restart" or not source_cache_reused
    if source_must_run and duration is not None:
        if "transcribe_audio" in operation_set:
            source_min_minutes += duration * 0.06 / 60.0
            source_max_minutes += duration * 0.35 / 60.0
        if "extract_bitmap_subtitles" in operation_set:
            source_min_minutes += duration * 0.04 / 60.0
            source_max_minutes += duration * 0.25 / 60.0
        if "synchronize_existing_tracks" in operation_set:
            source_min_minutes += duration * 0.03 / 60.0
            source_max_minutes += duration * 0.15 / 60.0

    finishing_min_minutes = 2.0
    finishing_max_minutes = 8.0
    min_minutes += source_min_minutes + finishing_min_minutes
    max_minutes += source_max_minutes + finishing_max_minutes
    min_minutes = max(1, math.ceil(min_minutes))
    max_minutes = max(min_minutes, math.ceil(max_minutes))

    notes = ["network_retries_excluded", "risk_repair_calls_excluded"]
    if segment_basis == "duration":
        notes.append("segment_count_estimated_from_duration")

    return {
        "available": True,
        "segment_count": segment_count,
        "segment_basis": segment_basis,
        "completed_segments": completed,
        "primary_items": primary_items,
        "batch_size": effective_batch,
        "review_batch_size": review_batch,
        "autonomous_review_rounds": rounds,
        "calls": breakdown,
        "total_calls": total_calls,
        "min_minutes": min_minutes,
        "max_minutes": max_minutes,
        "source_min_minutes": math.ceil(source_min_minutes),
        "source_max_minutes": math.ceil(source_max_minutes),
        "notes": notes,
    }
