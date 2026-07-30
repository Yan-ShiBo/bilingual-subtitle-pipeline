from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ass_styles import ResolvedAssStyle, resolve_ass_style


QUALITY_REPORT_VERSION = 2
MAX_REPORT_SAMPLES = 20
MIN_REVIEW_DURATION_SECONDS = 5 / 6
ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")


class FontAdvanceEstimator:
    def __init__(self, font_path: Path | None = None) -> None:
        self.font_path = font_path
        self.method = "unicode_estimate"
        self.units_per_em = 1.0
        self.cmap: dict[int, str] = {}
        self.advances: dict[str, int] = {}
        if font_path is not None:
            self._load_font(font_path)

    def _load_font(self, font_path: Path) -> None:
        try:
            from fontTools.ttLib import TTCollection, TTFont
        except ImportError as exc:
            raise RuntimeError("fonttools is required to measure the selected subtitle font") from exc

        collection = None
        font = None
        try:
            if font_path.suffix.lower() == ".ttc":
                collection = TTCollection(str(font_path), lazy=False)
                if not collection.fonts:
                    raise ValueError(f"Font collection contains no fonts: {font_path}")
                font = collection.fonts[0]
            else:
                font = TTFont(str(font_path), lazy=False)
            self.units_per_em = float(font["head"].unitsPerEm)
            self.cmap = dict(font.getBestCmap() or {})
            self.advances = {
                glyph_name: int(metrics[0])
                for glyph_name, metrics in font["hmtx"].metrics.items()
            }
            self.method = "font_metrics"
        finally:
            if collection is not None:
                collection.close()
            elif font is not None:
                font.close()

    @staticmethod
    def _fallback_em_width(character: str) -> float:
        if character.isspace():
            return 0.32
        east_asian_width = unicodedata.east_asian_width(character)
        if east_asian_width in {"W", "F"}:
            return 1.0
        if character in "ilI.,'`:;!|":
            return 0.30
        if character in "MW@%&#":
            return 0.90
        if character in "()[]{}<>/\\":
            return 0.42
        if character.isupper():
            return 0.66
        if character.islower():
            return 0.54
        if character.isdigit():
            return 0.58
        category = unicodedata.category(character)
        if category.startswith(("P", "S")):
            return 0.55
        return 0.62

    def em_width(self, text: str) -> float:
        visible_text = ASS_OVERRIDE_RE.sub("", str(text or "")).replace("\\N", "\n")
        width = 0.0
        widest_line = 0.0
        for character in visible_text:
            if character == "\n":
                widest_line = max(widest_line, width)
                width = 0.0
                continue
            glyph_name = self.cmap.get(ord(character))
            advance = self.advances.get(glyph_name or "")
            if advance is not None and self.units_per_em > 0:
                width += advance / self.units_per_em
            else:
                width += self._fallback_em_width(character)
        return max(widest_line, width)

    def pixel_width(self, text: str, font_size: int | float) -> float:
        return self.em_width(text) * float(font_size)


@dataclass(frozen=True)
class SubtitleLayoutPolicy:
    style: ResolvedAssStyle
    estimator: FontAdvanceEstimator
    font_file: str = ""

    def font_size(self, track: str, *, bilingual: bool = True) -> int:
        if track == "zh":
            return self.style.chinese_size
        if track == "en" and bilingual:
            return self.style.source_size
        return self.style.source_only_size

    def pixel_width(self, text: str, track: str, *, bilingual: bool = True) -> float:
        return self.estimator.pixel_width(
            text,
            self.font_size(track, bilingual=bilingual),
        )

    def required_chunks(self, text: str, track: str, *, bilingual: bool = True) -> int:
        if not str(text or "").strip():
            return 1
        width = self.pixel_width(text, track, bilingual=bilingual)
        return max(1, math.ceil(width / self.style.safe_line_width))

    def fits(self, text: str, track: str, *, bilingual: bool = True) -> bool:
        return self.pixel_width(text, track, bilingual=bilingual) <= self.style.safe_line_width + 0.01

    def report_dict(self) -> dict[str, Any]:
        return {
            "play_resolution": [self.style.play_res_x, self.style.play_res_y],
            "profile": self.style.profile_name,
            "font_name": self.style.font_name,
            "font_file": self.font_file,
            "font_scale_percent": round(self.style.scale * 100),
            "measurement_method": self.estimator.method,
            "safe_line_width_px": round(self.style.safe_line_width, 2),
            "margin_horizontal_px": self.style.margin_horizontal,
            "font_sizes": {
                "chinese": self.style.chinese_size,
                "english_bilingual": self.style.source_size,
                "source_only": self.style.source_only_size,
            },
        }


def build_layout_policy(
    play_resolution: tuple[int, int],
    *,
    profile_name: str,
    font_name: str,
    font_scale: int | float,
    font_path: Path | None = None,
) -> SubtitleLayoutPolicy:
    style = resolve_ass_style(
        *play_resolution,
        profile_name=profile_name,
        font_name=font_name,
        font_scale=font_scale,
    )
    return SubtitleLayoutPolicy(
        style=style,
        estimator=FontAdvanceEstimator(font_path),
        font_file=str(font_path) if font_path is not None else "",
    )


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _track_texts(segment: dict[str, Any]) -> tuple[str, str]:
    chinese = _normalized_text(segment.get("zh"))
    english = _normalized_text(segment.get("en"))
    if not chinese and not english:
        text = _normalized_text(segment.get("text"))
        if any("\u3400" <= character <= "\u9fff" for character in text):
            chinese = text
        else:
            english = text
    return chinese, english


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in text
    )


def _contains_east_asian(text: str) -> bool:
    return any(
        "\u3040" <= character <= "\u30ff"
        or "\u3400" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        or "\uac00" <= character <= "\ud7af"
        for character in text
    )


def _source_track_allows_east_asian(segment: dict[str, Any]) -> bool:
    language = str(segment.get("source_language") or "").strip().casefold()
    if "english" in language or language in {"en", "eng"}:
        return False
    return (
        language in {"ja", "jp", "jpn", "japanese", "ko", "kor", "korean", "zh", "zho", "chi"}
        or "japanese" in language
        or "korean" in language
        or "chinese" in language
    )


def _sample(
    samples: list[dict[str, Any]],
    *,
    segment: dict[str, Any],
    **values: Any,
) -> None:
    if len(samples) >= MAX_REPORT_SAMPLES:
        return
    samples.append(
        {
            "id": segment.get("id"),
            **(
                {"checkpoint_segment_id": segment.get("checkpoint_segment_id")}
                if segment.get("checkpoint_segment_id") is not None
                else {}
            ),
            "start": round(float(segment.get("start") or 0.0), 3),
            "end": round(float(segment.get("end") or 0.0), 3),
            **values,
        }
    )


def _terminology_inconsistencies(
    segments: Iterable[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    targets_by_source: dict[str, set[str]] = {}
    display_by_source: dict[str, str] = {}
    for segment in segments:
        for entry in segment.get("terminology") or []:
            if not isinstance(entry, dict):
                continue
            source = _normalized_text(entry.get("source"))
            target = _normalized_text(entry.get("target"))
            if not source or not target:
                continue
            key = source.casefold()
            targets_by_source.setdefault(key, set()).add(target)
            display_by_source.setdefault(key, source)
    samples = [
        {
            "source": display_by_source[key],
            "targets": sorted(targets),
        }
        for key, targets in targets_by_source.items()
        if len(targets) > 1
    ]
    return len(samples), samples[:MAX_REPORT_SAMPLES]


def _uses_existing_subtitle_timing(
    segments: Iterable[dict[str, Any]],
    source_kind: str,
) -> bool:
    normalized_source = source_kind.casefold()
    if any(
        marker in normalized_source
        for marker in ("sidecar", "embedded", "existing subtitle")
    ):
        return True
    return any(
        str(segment.get("timing_origin") or "").casefold() == "existing_subtitle"
        for segment in segments
    )


def _timeline_union_duration(
    segments: Iterable[dict[str, Any]],
    video_duration_seconds: float | None,
) -> float:
    intervals: list[tuple[float, float]] = []
    upper_bound = (
        float(video_duration_seconds)
        if video_duration_seconds is not None and video_duration_seconds > 0
        else None
    )
    for segment in segments:
        try:
            start = max(0.0, float(segment.get("start") or 0.0))
            end = float(segment.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        if upper_bound is not None:
            start = min(start, upper_bound)
            end = min(end, upper_bound)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _sync_quality_check(
    sync_report: dict[str, Any] | None,
    *,
    existing_timing_source: bool,
) -> dict[str, Any]:
    if not existing_timing_source:
        return {
            "status": "not_applicable",
            "review_items": 0,
            "mode": None,
            "result": None,
            "applied": False,
            "reasons": [],
        }
    if not isinstance(sync_report, dict):
        return {
            "status": "review",
            "review_items": 1,
            "mode": None,
            "result": "missing_report",
            "applied": False,
            "reasons": [
                "No subtitle synchronization report was available for the existing subtitle source."
            ],
        }

    mode = str(sync_report.get("mode") or "").casefold()
    result = str(sync_report.get("status") or "unknown").casefold()
    applied = bool(sync_report.get("applied"))
    reasons: list[str] = []

    def collect_reasons(report: dict[str, Any]) -> None:
        for value in report.get("quality_reasons") or []:
            normalized = str(value).strip()
            if normalized and normalized not in reasons:
                reasons.append(normalized)

    collect_reasons(sync_report)
    tracks = sync_report.get("tracks")
    if isinstance(tracks, dict):
        for track_report in tracks.values():
            if isinstance(track_report, dict):
                collect_reasons(track_report)

    review_results = {
        "disabled",
        "detected",
        "detected_with_warnings",
        "error",
        "failed",
        "pending",
        "rejected",
        "rejected_low_quality",
        "rejected_partial",
        "unavailable",
        "unknown",
    }
    needs_review = (
        sync_report.get("pipeline_quality_passed") is False
        or result in review_results
        or (mode == "detect" and applied is False and result not in {"already_aligned"})
    )
    if needs_review and not reasons:
        if result == "disabled" or mode == "off":
            reasons.append("Subtitle synchronization was disabled for an existing subtitle source.")
        elif result.startswith("detected"):
            reasons.append("A timing shift was detected but was not automatically applied.")
        else:
            reasons.append(f"Subtitle synchronization result requires review: {result}.")

    return {
        "status": "review" if needs_review else "pass",
        "review_items": int(needs_review),
        "mode": mode or None,
        "result": result,
        "applied": applied,
        "reasons": reasons,
    }


def _source_completeness_check(
    source_segments: list[dict[str, Any]],
    *,
    existing_timing_source: bool,
    video_duration_seconds: float | None,
) -> dict[str, Any]:
    duration = (
        float(video_duration_seconds)
        if video_duration_seconds is not None and video_duration_seconds > 0
        else None
    )
    covered_seconds = _timeline_union_duration(source_segments, duration)
    coverage_ratio = covered_seconds / duration if duration else None
    event_rate = len(source_segments) * 60.0 / duration if duration else None
    assessable = existing_timing_source and duration is not None and duration >= 600.0
    likely_sparse = bool(
        assessable
        and event_rate is not None
        and coverage_ratio is not None
        and (
            (event_rate < 0.5 and coverage_ratio < 0.03)
            or (len(source_segments) < 10 and coverage_ratio < 0.01)
        )
    )
    reasons: list[str] = []
    if likely_sparse:
        reasons.append(
            "The selected existing subtitle track is unusually sparse for the video duration; "
            "it may be a signs/forced-only track."
        )
    return {
        "status": "review" if likely_sparse else ("pass" if assessable else "not_applicable"),
        "review_items": int(likely_sparse),
        "assessed": assessable,
        "likely_sparse": likely_sparse,
        "video_duration_seconds": round(duration, 3) if duration is not None else None,
        "covered_seconds": round(covered_seconds, 3),
        "timeline_coverage_ratio": (
            round(coverage_ratio, 6) if coverage_ratio is not None else None
        ),
        "events_per_minute": round(event_rate, 3) if event_rate is not None else None,
        "reasons": reasons,
    }


def _recognition_confidence_check(
    source_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    low_confidence_count = 0
    assessed_count = 0
    for segment in source_segments:
        issues: list[str] = []
        metrics: dict[str, float] = {}

        def numeric(key: str) -> float | None:
            value = segment.get(key)
            try:
                normalized = float(value)
            except (TypeError, ValueError):
                return None
            return normalized if math.isfinite(normalized) else None

        ocr_confidence = numeric("ocr_confidence")
        avg_logprob = numeric("asr_avg_logprob")
        compression_ratio = numeric("asr_compression_ratio")
        no_speech_prob = numeric("asr_no_speech_prob")
        word_confidence = numeric("asr_word_confidence")
        words = segment.get("words")
        word_probabilities: list[float] = []
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, dict):
                    continue
                try:
                    probability = float(word.get("probability"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(probability) and 0 <= probability <= 1:
                    word_probabilities.append(probability)

        if any(
            value is not None
            for value in (
                ocr_confidence,
                avg_logprob,
                compression_ratio,
                no_speech_prob,
                word_confidence,
            )
        ) or word_probabilities:
            assessed_count += 1
        if ocr_confidence is not None:
            metrics["ocr_confidence"] = round(ocr_confidence, 4)
            if ocr_confidence < 0.75:
                issues.append("low_ocr_confidence")
        if avg_logprob is not None:
            metrics["asr_avg_logprob"] = round(avg_logprob, 4)
            if avg_logprob < -1.0:
                issues.append("low_asr_log_probability")
        if compression_ratio is not None:
            metrics["asr_compression_ratio"] = round(compression_ratio, 4)
            if compression_ratio > 2.4:
                issues.append("high_asr_compression_ratio")
        if no_speech_prob is not None:
            metrics["asr_no_speech_prob"] = round(no_speech_prob, 4)
            if no_speech_prob > 0.6:
                issues.append("high_no_speech_probability")
        if word_confidence is not None:
            metrics["asr_word_confidence"] = round(word_confidence, 4)
        if word_probabilities:
            low_word_ratio = sum(value < 0.35 for value in word_probabilities) / len(
                word_probabilities
            )
            metrics["low_word_confidence_ratio"] = round(low_word_ratio, 4)
            if low_word_ratio >= 0.25:
                issues.append("low_word_confidence")
        elif word_confidence is not None and word_confidence < 0.55:
            issues.append("low_word_confidence")

        if issues:
            low_confidence_count += 1
            if len(samples) < MAX_REPORT_SAMPLES:
                samples.append(
                    {
                        "id": segment.get("id"),
                        "start": round(float(segment.get("start") or 0.0), 3),
                        "end": round(float(segment.get("end") or 0.0), 3),
                        "issues": issues,
                        "metrics": metrics,
                        "text": _normalized_text(
                            segment.get("text") or segment.get("en") or segment.get("zh")
                        ),
                    }
                )
    return {
        "status": "review" if low_confidence_count else (
            "pass" if assessed_count else "not_applicable"
        ),
        "assessed_count": assessed_count,
        "low_confidence_count": low_confidence_count,
        "thresholds": {
            "minimum_ocr_confidence": 0.75,
            "minimum_asr_avg_logprob": -1.0,
            "maximum_asr_compression_ratio": 2.4,
            "maximum_asr_no_speech_probability": 0.6,
            "minimum_asr_word_probability": 0.35,
            "maximum_low_word_ratio": 0.25,
        },
        "samples": samples,
    }


def _llm_artifact_check(
    artifact: dict[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {
            "status": "not_applicable",
            "review_items": 0,
            "result": "missing",
            "reasons": [],
            "samples": [],
        }
    result = str(artifact.get("status") or "unknown").casefold()
    if result in {"skipped_short", "not_applicable"}:
        return {
            "status": "not_applicable",
            "review_items": 0,
            "result": result,
            "reasons": [str(artifact.get("reason") or "")],
            "samples": [],
        }
    needs_review = result not in {"pass", "ready"}
    reasons: list[str] = []
    samples: list[dict[str, Any]] = []
    if artifact.get("error"):
        reasons.append(str(artifact["error"]))
    if artifact.get("reason"):
        reasons.append(str(artifact["reason"]))
    rejected_count = int(artifact.get("rejected_count") or 0)
    if rejected_count:
        reasons.append(f"{rejected_count} subtitle events were rejected by validation.")
        needs_review = True
    for item in artifact.get("rejected_items") or []:
        if not isinstance(item, dict) or len(samples) >= MAX_REPORT_SAMPLES:
            continue
        samples.append(
            {
                "id": item.get("id"),
                "start": item.get("start"),
                "end": item.get("end"),
                "issue": item.get("reason") or f"{label} rejected this event.",
            }
        )
    if artifact.get("sample_event_id") is not None and needs_review:
        samples.append(
            {
                "id": artifact.get("sample_event_id"),
                **(
                    {"checkpoint_segment_id": artifact.get("checkpoint_segment_id")}
                    if artifact.get("checkpoint_segment_id") is not None
                    else {}
                ),
                "start": artifact.get("sample_time_seconds"),
                "issue": artifact.get("reason") or f"{label} failed.",
            }
        )
    if needs_review and not reasons:
        reasons.append(f"{label} result requires review: {result}.")
    return {
        "status": "review" if needs_review else "pass",
        "review_items": max(int(needs_review), len(samples)),
        "result": result,
        "reviewed_count": int(artifact.get("reviewed_count") or 0),
        "changed_count": int(artifact.get("changed_count") or 0),
        "rejected_count": rejected_count,
        "reasons": reasons,
        "samples": samples[:MAX_REPORT_SAMPLES],
    }


def build_quality_report(
    source_segments: list[dict[str, Any]],
    processed_segments: list[dict[str, Any]],
    output_segments: list[dict[str, Any]],
    layout: SubtitleLayoutPolicy,
    *,
    max_duration: float,
    target_chinese_cps: float,
    target_english_cps: float,
    artifacts: dict[str, str] | None = None,
    sync_report: dict[str, Any] | None = None,
    source_kind: str = "",
    video_duration_seconds: float | None = None,
    min_duration_seconds: float = MIN_REVIEW_DURATION_SECONDS,
    style_guide: dict[str, Any] | None = None,
    final_qa_report: dict[str, Any] | None = None,
    render_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pixel_samples: list[dict[str, Any]] = []
    cps_samples: list[dict[str, Any]] = []
    timing_samples: list[dict[str, Any]] = []
    completeness_samples: list[dict[str, Any]] = []
    pixel_overflow_count = 0
    chinese_over_cps = 0
    english_over_cps = 0
    compact_source_over_cps = 0
    over_duration_count = 0
    under_min_duration_count = 0
    invalid_duration_count = 0
    overlap_count = 0
    intentional_overlap_count = 0
    missing_chinese_count = 0
    invalid_chinese_target_count = 0
    english_contains_cjk_count = 0
    empty_event_count = 0
    max_width_ratio = 0.0

    previous_end: float | None = None
    previous_segment: dict[str, Any] | None = None
    for segment in output_segments:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or 0.0)
        duration = end - start
        chinese, english = _track_texts(segment)
        bilingual = bool(chinese and english)

        if duration <= 0:
            invalid_duration_count += 1
            _sample(timing_samples, segment=segment, issue="invalid_duration")
        elif duration < min_duration_seconds - 0.001:
            under_min_duration_count += 1
            _sample(
                timing_samples,
                segment=segment,
                issue="under_min_duration",
                duration=round(duration, 3),
            )
        if previous_end is not None and start < previous_end - 0.001:
            previous_text = _track_texts(previous_segment or {})
            current_text = _track_texts(segment)
            intentional_overlap = (
                bool((previous_segment or {}).get("preserve_distinct_overlap"))
                and bool(segment.get("preserve_distinct_overlap"))
                and previous_text != current_text
            )
            if intentional_overlap:
                intentional_overlap_count += 1
            else:
                overlap_count += 1
                _sample(timing_samples, segment=segment, issue="timeline_overlap")
        if previous_end is None or end >= previous_end:
            previous_end = end
            previous_segment = segment
        if duration > max_duration + 0.01:
            over_duration_count += 1
            _sample(
                timing_samples,
                segment=segment,
                issue="over_max_duration",
                duration=round(duration, 3),
            )

        if not chinese and not english:
            empty_event_count += 1
            _sample(completeness_samples, segment=segment, issue="empty_visible_event")
        elif not chinese:
            missing_chinese_count += 1
            _sample(completeness_samples, segment=segment, issue="missing_chinese")
        elif not _contains_cjk(chinese) and any(character.isalpha() for character in chinese):
            invalid_chinese_target_count += 1
            _sample(
                completeness_samples,
                segment=segment,
                issue="chinese_target_has_no_cjk",
                text=chinese,
            )
        if (
            english
            and _contains_east_asian(english)
            and not _source_track_allows_east_asian(segment)
        ):
            english_contains_cjk_count += 1
            _sample(
                completeness_samples,
                segment=segment,
                issue="english_track_contains_cjk",
                text=english,
            )

        for track, text in (("zh", chinese), ("en", english)):
            if not text:
                continue
            width = layout.pixel_width(text, track, bilingual=bilingual)
            ratio = width / layout.style.safe_line_width
            max_width_ratio = max(max_width_ratio, ratio)
            if ratio > 1.0 + 1e-6:
                pixel_overflow_count += 1
                _sample(
                    pixel_samples,
                    segment=segment,
                    issue=f"{track}_pixel_overflow",
                    text=text,
                    width_px=round(width, 2),
                    safe_width_px=round(layout.style.safe_line_width, 2),
                    ratio=round(ratio, 3),
                )

        readable_duration = max(0.01, duration)
        if chinese:
            chinese_count = len(re.sub(r"\s+", "", chinese))
            chinese_cps = chinese_count / readable_duration
            if chinese_cps > target_chinese_cps + 1e-9:
                chinese_over_cps += 1
                _sample(
                    cps_samples,
                    segment=segment,
                    issue="chinese_cps",
                    cps=round(chinese_cps, 2),
                    target=target_chinese_cps,
                )
        if english:
            source_cps = len(english) / readable_duration
            if _contains_east_asian(english):
                if source_cps > target_chinese_cps + 1e-9:
                    compact_source_over_cps += 1
                    _sample(
                        cps_samples,
                        segment=segment,
                        issue="east_asian_source_cps",
                        cps=round(source_cps, 2),
                        target=target_chinese_cps,
                    )
            elif source_cps > target_english_cps + 1e-9:
                english_over_cps += 1
                _sample(
                    cps_samples,
                    segment=segment,
                    issue="english_cps",
                    cps=round(source_cps, 2),
                    target=target_english_cps,
                )

    guard_overrides = [
        segment
        for segment in processed_segments
        if segment.get("display_guard") == "forced_visible"
    ]
    accepted_model_hides = [
        segment
        for segment in processed_segments
        if segment.get("display_guard") == "accepted_hidden"
    ]
    guard_samples = [
        {
            "id": segment.get("id"),
            "start": round(float(segment.get("start") or 0.0), 3),
            "reason": segment.get("display_guard_reason") or "",
            "text": _normalized_text(segment.get("text") or segment.get("en") or segment.get("zh")),
        }
        for segment in guard_overrides[:MAX_REPORT_SAMPLES]
    ]
    terminology_count, terminology_samples = _terminology_inconsistencies(processed_segments)
    existing_timing_source = _uses_existing_subtitle_timing(source_segments, source_kind)
    subtitle_sync = _sync_quality_check(
        sync_report,
        existing_timing_source=existing_timing_source,
    )
    source_completeness = _source_completeness_check(
        source_segments,
        existing_timing_source=existing_timing_source,
        video_duration_seconds=video_duration_seconds,
    )
    recognition_confidence = _recognition_confidence_check(source_segments)
    movie_style = _llm_artifact_check(
        style_guide,
        label="Movie-wide style analysis",
    )
    independent_final_qa = _llm_artifact_check(
        final_qa_report,
        label="Independent final QA",
    )
    delivery_render = _llm_artifact_check(
        render_validation,
        label="ASS delivery render validation",
    )
    llm_review_items = int(movie_style["review_items"]) + int(
        independent_final_qa["review_items"]
    )

    fatal_count = invalid_duration_count + overlap_count + empty_event_count
    review_count = (
        pixel_overflow_count
        + chinese_over_cps
        + english_over_cps
        + compact_source_over_cps
        + over_duration_count
        + under_min_duration_count
        + missing_chinese_count
        + invalid_chinese_target_count
        + english_contains_cjk_count
        + len(guard_overrides)
        + terminology_count
        + int(subtitle_sync["review_items"])
        + int(source_completeness["review_items"])
        + int(recognition_confidence["low_confidence_count"])
        + llm_review_items
        + int(delivery_render["review_items"])
    )
    status = "fail" if fatal_count else ("review" if review_count else "pass")
    return {
        "version": QUALITY_REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "source_events": len(source_segments),
            "processed_events": len(processed_segments),
            "output_events": len(output_segments),
            "fatal_issues": fatal_count,
            "review_items": review_count,
            "pixel_overflow_events": pixel_overflow_count,
            "model_hide_overrides": len(guard_overrides),
            "under_min_duration_events": under_min_duration_count,
            "sync_review_items": subtitle_sync["review_items"],
            "source_review_items": source_completeness["review_items"],
            "low_confidence_events": recognition_confidence["low_confidence_count"],
            "llm_review_items": llm_review_items,
            "render_review_items": delivery_render["review_items"],
        },
        "layout": layout.report_dict(),
        "thresholds": {
            "max_duration_seconds": max_duration,
            "min_duration_seconds": min_duration_seconds,
            "target_chinese_cps": target_chinese_cps,
            "target_english_cps": target_english_cps,
        },
        "checks": {
            "pixel_width": {
                "status": "review" if pixel_overflow_count else "pass",
                "overflow_count": pixel_overflow_count,
                "max_width_ratio": round(max_width_ratio, 3),
                "samples": pixel_samples,
            },
            "readability": {
                "status": (
                    "review"
                    if chinese_over_cps
                    or english_over_cps
                    or compact_source_over_cps
                    else "pass"
                ),
                "chinese_over_target": chinese_over_cps,
                "english_over_target": english_over_cps,
                "east_asian_source_over_target": compact_source_over_cps,
                "samples": cps_samples,
            },
            "timing": {
                "status": "fail" if invalid_duration_count or overlap_count else (
                    "review" if over_duration_count or under_min_duration_count else "pass"
                ),
                "invalid_duration_count": invalid_duration_count,
                "overlap_count": overlap_count,
                "intentional_overlap_count": intentional_overlap_count,
                "over_max_duration_count": over_duration_count,
                "under_min_duration_count": under_min_duration_count,
                "samples": timing_samples,
            },
            "completeness": {
                "status": "fail" if empty_event_count else (
                    "review"
                    if missing_chinese_count
                    or invalid_chinese_target_count
                    or english_contains_cjk_count
                    else "pass"
                ),
                "empty_event_count": empty_event_count,
                "missing_chinese_count": missing_chinese_count,
                "invalid_chinese_target_count": invalid_chinese_target_count,
                "english_contains_cjk_count": english_contains_cjk_count,
                "samples": completeness_samples,
            },
            "model_display_guard": {
                "status": "review" if guard_overrides else "pass",
                "forced_visible_count": len(guard_overrides),
                "accepted_hidden_count": len(accepted_model_hides),
                "samples": guard_samples,
            },
            "terminology": {
                "status": "review" if terminology_count else "pass",
                "inconsistency_count": terminology_count,
                "samples": terminology_samples,
            },
            "subtitle_sync": subtitle_sync,
            "source_completeness": source_completeness,
            "recognition_confidence": recognition_confidence,
            "movie_style": movie_style,
            "independent_final_qa": independent_final_qa,
            "delivery_render": delivery_render,
        },
        "artifacts": artifacts or {},
    }


def write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
