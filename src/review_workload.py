from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

CHECK_DEFAULT_SEVERITY = {
    "pixel_width": "low",
    "readability": "low",
    "timing": "high",
    "completeness": "high",
    "model_display_guard": "medium",
    "terminology": "medium",
    "subtitle_sync": "high",
    "source_completeness": "high",
    "recognition_confidence": "medium",
    "movie_style": "medium",
    "independent_final_qa": "high",
    "delivery_render": "high",
}

ISSUE_SEVERITY = {
    "invalid_duration": "critical",
    "timeline_overlap": "critical",
    "empty_visible_event": "critical",
    "missing_chinese": "high",
    "chinese_target_has_no_cjk": "high",
    "english_track_contains_cjk": "high",
    "source_language_changed": "high",
    "invalid_language_or_terminology": "high",
    "high_asr_compression_ratio": "high",
    "high_no_speech_probability": "high",
    "over_max_duration": "medium",
    "under_min_duration": "medium",
    "low_ocr_confidence": "medium",
    "low_asr_log_probability": "medium",
    "low_word_confidence": "medium",
    "model_display_guard": "medium",
    "chinese_cps": "low",
    "english_cps": "low",
    "east_asian_source_cps": "low",
    "pixel_overflow": "low",
}

STABLE_GLOBAL_ROOT_CHECKS = {
    "subtitle_sync",
    "source_completeness",
    "movie_style",
    "independent_final_qa",
    "delivery_render",
}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def issue_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        values = value
    else:
        values = str(value or "").split(",")
    return [str(item).strip() for item in values if str(item).strip()]


def canonical_review_issue(item: Mapping[str, Any]) -> str:
    check = str(item.get("check") or "unknown")
    tokens = issue_tokens(item.get("issue"))
    if check == "recognition_confidence":
        return "recognition_confidence"
    if check == "pixel_width" or any(token.endswith("_pixel_overflow") for token in tokens):
        return "pixel_overflow"
    if check == "model_display_guard":
        return "model_display_guard"
    if not tokens:
        return check
    if check in STABLE_GLOBAL_ROOT_CHECKS and not any(
        token in ISSUE_SEVERITY for token in tokens
    ):
        return check
    return max(
        tokens,
        key=lambda token: (
            SEVERITY_RANK.get(ISSUE_SEVERITY.get(token, "low"), 0),
            -tokens.index(token),
        ),
    )


def _metric_risk(item: Mapping[str, Any]) -> float:
    details = item.get("details")
    if not isinstance(details, Mapping):
        return 0.0
    cps = _finite_float(details.get("cps"))
    target = _finite_float(details.get("target"))
    if cps is not None and target is not None and target > 0:
        return max(0.0, cps / target - 1.0)
    ratio = _finite_float(details.get("ratio"))
    if ratio is not None:
        return max(0.0, ratio - 1.0)
    metrics = details.get("metrics")
    if isinstance(metrics, Mapping):
        ocr = _finite_float(metrics.get("ocr_confidence"))
        avg_logprob = _finite_float(metrics.get("asr_avg_logprob"))
        compression = _finite_float(metrics.get("asr_compression_ratio"))
        no_speech = _finite_float(metrics.get("asr_no_speech_prob"))
        low_word_ratio = _finite_float(metrics.get("low_word_confidence_ratio"))
        risks = [
            max(0.0, 0.75 - ocr) if ocr is not None else 0.0,
            max(0.0, -1.0 - avg_logprob) if avg_logprob is not None else 0.0,
            max(0.0, compression - 2.4) if compression is not None else 0.0,
            max(0.0, no_speech - 0.6) if no_speech is not None else 0.0,
            max(0.0, low_word_ratio - 0.25) if low_word_ratio is not None else 0.0,
        ]
        return max(risks)
    return 0.0


def review_item_severity(item: Mapping[str, Any]) -> tuple[str, float]:
    check = str(item.get("check") or "")
    tokens = issue_tokens(item.get("issue"))
    issue_severities = [ISSUE_SEVERITY[token] for token in tokens if token in ISSUE_SEVERITY]
    known_issue = bool(issue_severities)
    if str(item.get("status") or "") == "fail" and not known_issue:
        return "critical", 399.0
    severity = max(
        issue_severities,
        key=lambda value: SEVERITY_RANK[value],
        default=CHECK_DEFAULT_SEVERITY.get(check, "medium"),
    )
    risk = _metric_risk(item)
    if severity == "low" and risk >= 0.5:
        severity = "high"
    elif severity == "low" and risk >= 0.2:
        severity = "medium"
    elif severity == "medium" and risk >= 0.5:
        severity = "high"
    score = SEVERITY_RANK[severity] * 100.0 + min(99.0, risk * 100.0)
    return severity, round(score, 3)


def _item_start(item: Mapping[str, Any]) -> float:
    segment = item.get("segment")
    if isinstance(segment, Mapping):
        value = _finite_float(segment.get("start"))
        if value is not None:
            return value
    return _finite_float(item.get("start")) or 0.0


def build_review_workload(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    collapsed_by_group: dict[str, int] = {}
    input_count = 0
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            continue
        input_count += 1
        item = dict(raw_item)
        root_issue = canonical_review_issue(item)
        check = str(item.get("check") or "unknown")
        group_key = f"{check}:{root_issue}"
        severity, score = review_item_severity(item)
        item.update(
            {
                "root_issue": root_issue,
                "group_key": group_key,
                "severity": severity,
                "severity_score": score,
            }
        )
        segment_id = item.get("segment_id")
        identity = str(segment_id) if segment_id is not None else f"global:{item.get('issue') or check}"
        dedupe_key = (group_key, identity)
        previous = deduplicated.get(dedupe_key)
        if previous is None or float(previous.get("severity_score") or 0.0) < score:
            deduplicated[dedupe_key] = item
        if previous is not None:
            collapsed_by_group[group_key] = collapsed_by_group.get(group_key, 0) + 1

    annotated = sorted(
        deduplicated.values(),
        key=lambda item: (
            -float(item.get("severity_score") or 0.0),
            _item_start(item),
            str(item.get("group_key") or ""),
        ),
    )
    groups_by_key: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(annotated):
        key = str(item["group_key"])
        start = _item_start(item)
        group = groups_by_key.setdefault(
            key,
            {
                "key": key,
                "check": str(item.get("check") or "unknown"),
                "issue": str(item.get("root_issue") or item.get("issue") or ""),
                "severity": str(item.get("severity") or "medium"),
                "severity_score": float(item.get("severity_score") or 0.0),
                "sample_count": 0,
                "collapsed_sample_count": collapsed_by_group.get(key, 0),
                "first_start": start,
                "last_start": start,
                "representative_index": index,
                "item_indices": [],
            },
        )
        group["sample_count"] += 1
        group["first_start"] = min(float(group["first_start"]), start)
        group["last_start"] = max(float(group["last_start"]), start)
        group["item_indices"].append(index)
        if float(item.get("severity_score") or 0.0) > float(group["severity_score"]):
            group["severity"] = str(item.get("severity") or "medium")
            group["severity_score"] = float(item.get("severity_score") or 0.0)
            group["representative_index"] = index

    groups = sorted(
        groups_by_key.values(),
        key=lambda group: (
            -float(group.get("severity_score") or 0.0),
            -int(group.get("sample_count") or 0),
            str(group.get("key") or ""),
        ),
    )
    return {
        "items": annotated,
        "groups": groups,
        "input_sample_count": input_count,
        "actionable_sample_count": len(annotated),
        "collapsed_sample_count": input_count - len(annotated),
    }
