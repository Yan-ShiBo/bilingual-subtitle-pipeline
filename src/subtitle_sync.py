from __future__ import annotations

import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any


Segment = dict[str, Any]
SUBTITLE_SYNC_MODES = ("auto", "detect", "off")
SYNC_POLICY_VERSION = 5
MAX_OFFSET_SECONDS = 120.0
SPLIT_PENALTY = 20.0
MAX_PIECEWISE_JUMPS = 6
MIN_MEANINGFUL_SHIFT_SECONDS = 0.08


def _seconds_to_srt(value: float) -> str:
    milliseconds = max(0, int(round(float(value) * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _srt_to_seconds(value: str) -> float:
    hours, minutes, remainder = value.strip().split(":")
    seconds, milliseconds = remainder.replace(".", ",").split(",", 1)
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds[:3].ljust(3, "0")) / 1000
    )


def _write_timing_srt(path: Path, segments: list[Segment]) -> None:
    rows: list[str] = []
    for index, segment in enumerate(segments):
        rows.extend(
            [
                str(index + 1),
                f"{_seconds_to_srt(segment['start'])} --> {_seconds_to_srt(segment['end'])}",
                f"SUBTITLE_SYNC_EVENT_{index:06d}",
                "",
            ]
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def _read_timing_srt(path: Path) -> list[tuple[float, float]]:
    import re

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    ordered: list[tuple[float, float]] = []
    indexed: dict[int, tuple[float, float]] = {}
    for block in blocks:
        timing = re.search(
            r"(\d+:\d\d:\d\d[,.]\d{1,3})\s*-->\s*(\d+:\d\d:\d\d[,.]\d{1,3})",
            block,
        )
        if not timing:
            continue
        value = (_srt_to_seconds(timing.group(1)), _srt_to_seconds(timing.group(2)))
        ordered.append(value)
        marker = re.search(r"SUBTITLE_SYNC_EVENT_(\d{6})", block)
        if marker:
            indexed[int(marker.group(1))] = value
    if indexed and len(indexed) == len(ordered) and set(indexed) == set(range(len(indexed))):
        return [indexed[index] for index in range(len(indexed))]
    return ordered


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _run_ffsubsync(
    video_path: Path,
    input_srt: Path,
    output_srt: Path,
    *,
    audio_stream: int | None,
    ffmpeg_path: str | None,
    piecewise: bool = True,
) -> dict[str, Any]:
    try:
        from ffsubsync.ffsubsync import make_parser, run
    except ImportError as exc:
        raise RuntimeError(
            "ffsubsync is not installed. Install the locked project dependencies first."
        ) from exc

    arguments = [
        str(video_path),
        "-i",
        str(input_srt),
        "-o",
        str(output_srt),
        "--vad",
        "webrtc",
        "--skip-sync-on-low-quality",
        "--min-score",
        "0",
        "--quality-max-offset-seconds",
        "45",
        "--max-framerate-deviation",
        "0.08",
        "--max-offset-seconds",
        str(MAX_OFFSET_SECONDS),
        "--max-subtitle-seconds",
        "120",
    ]
    if piecewise:
        arguments.extend(["--split-penalty", str(SPLIT_PENALTY)])
    if audio_stream is not None:
        arguments.extend(["--reference-stream", f"0:{int(audio_stream)}"])
    if ffmpeg_path:
        arguments.extend(["--ffmpeg-path", str(ffmpeg_path)])

    parser = make_parser()
    parsed = parser.parse_args(arguments)
    result = run(parsed)
    if int(result.get("retval") or 0) != 0:
        raise RuntimeError(f"ffsubsync exited with code {result.get('retval')}")
    return result


def _execute_alignment(
    video_path: Path,
    segments: list[Segment],
    *,
    audio_stream: int | None,
    ffmpeg_path: str | None,
    piecewise: bool,
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    with tempfile.TemporaryDirectory(prefix="subtitle-sync-") as temp_dir:
        temporary_root = Path(temp_dir)
        input_srt = temporary_root / "source.srt"
        output_srt = temporary_root / "synced.srt"
        _write_timing_srt(input_srt, segments)
        result = _run_ffsubsync(
            video_path,
            input_srt,
            output_srt,
            audio_stream=audio_stream,
            ffmpeg_path=ffmpeg_path,
            piecewise=piecewise,
        )
        if not output_srt.exists():
            raise RuntimeError("ffsubsync did not produce an output subtitle")
        return result, _read_timing_srt(output_srt)


def _candidate_metrics(
    segments: list[Segment],
    synchronized_times: list[tuple[float, float]],
) -> dict[str, Any]:
    start_shifts = [
        synchronized_times[index][0] - float(segment["start"])
        for index, segment in enumerate(segments)
    ]
    end_shifts = [
        synchronized_times[index][1] - float(segment["end"])
        for index, segment in enumerate(segments)
    ]
    absolute_shifts = [abs(value) for value in start_shifts + end_shifts]
    duration_changes = [
        (synchronized_times[index][1] - synchronized_times[index][0])
        - (float(segment["end"]) - float(segment["start"]))
        for index, segment in enumerate(segments)
    ]
    piecewise_jumps = [
        index
        for index in range(1, len(start_shifts))
        if abs(start_shifts[index] - start_shifts[index - 1]) >= 2.0
    ]
    clipped_offsets = sum(
        abs(value) >= MAX_OFFSET_SECONDS - 0.5
        for value in start_shifts
    )
    out_of_order = sum(
        synchronized_times[index][0] < synchronized_times[index - 1][0]
        for index in range(1, len(synchronized_times))
    )
    quality_reasons: list[str] = []
    if clipped_offsets:
        quality_reasons.append(
            f"{clipped_offsets} cue offsets reached the search boundary"
        )
    if out_of_order:
        quality_reasons.append(
            f"{out_of_order} aligned cues became out of order"
        )
    if len(piecewise_jumps) > MAX_PIECEWISE_JUMPS:
        quality_reasons.append(
            f"{len(piecewise_jumps)} piecewise jumps exceed the limit of {MAX_PIECEWISE_JUMPS}"
        )
    max_duration_change = max((abs(value) for value in duration_changes), default=0.0)
    if max_duration_change > 2.0:
        quality_reasons.append(
            f"subtitle duration changed by up to {max_duration_change:.3f} seconds"
        )
    return {
        "changed": any(value >= MIN_MEANINGFUL_SHIFT_SECONDS for value in absolute_shifts),
        "median_start_shift_seconds": round(
            float(statistics.median(start_shifts) if start_shifts else 0.0),
            4,
        ),
        "p95_absolute_shift_seconds": round(_percentile(absolute_shifts, 0.95), 4),
        "max_absolute_shift_seconds": round(max(absolute_shifts, default=0.0), 4),
        "piecewise_jump_count": len(piecewise_jumps),
        "search_boundary_cue_count": clipped_offsets,
        "out_of_order_cue_count": out_of_order,
        "max_duration_change_seconds": round(max_duration_change, 4),
        "quality_reasons": quality_reasons,
    }


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def synchronize_subtitle_segments(
    video_path: Path,
    segments: list[Segment],
    *,
    mode: str = "auto",
    report_path: Path | None = None,
    audio_stream: int | None = None,
    ffmpeg_path: str | None = None,
) -> tuple[list[Segment], dict[str, Any]]:
    if mode not in SUBTITLE_SYNC_MODES:
        raise ValueError(f"Unsupported subtitle sync mode: {mode}")

    report: dict[str, Any] = {
        "version": SYNC_POLICY_VERSION,
        "mode": mode,
        "backend": "ffsubsync",
        "video": str(video_path),
        "audio_stream": audio_stream,
        "event_count": len(segments),
        "status": "disabled" if mode == "off" else "pending",
        "applied": False,
    }
    if mode == "off" or not segments:
        _write_report(report_path, report)
        return [dict(segment) for segment in segments], report

    for index, segment in enumerate(segments):
        if float(segment["end"]) <= float(segment["start"]):
            raise ValueError(f"Subtitle event {index} has invalid timing")

    try:
        result, synchronized_times = _execute_alignment(
            video_path,
            segments,
            audio_stream=audio_stream,
            ffmpeg_path=ffmpeg_path,
            piecewise=True,
        )
    except Exception as exc:
        report.update({"status": "unavailable", "error": str(exc)})
        _write_report(report_path, report)
        return [dict(segment) for segment in segments], report

    if len(synchronized_times) != len(segments):
        report.update(
            {
                "status": "rejected",
                "error": (
                    "Aligned subtitle event count changed "
                    f"from {len(segments)} to {len(synchronized_times)}"
                ),
            }
        )
        _write_report(report_path, report)
        return [dict(segment) for segment in segments], report

    metrics = _candidate_metrics(segments, synchronized_times)
    backend_successful = bool(result.get("sync_was_successful"))
    piecewise_rejected = backend_successful and bool(metrics["quality_reasons"])
    used_global_fallback = False
    if piecewise_rejected:
        piecewise_summary = {
            **metrics,
            "offset_seconds": (
                float(result["offset_seconds"])
                if result.get("offset_seconds") is not None
                else None
            ),
            "framerate_scale_factor": (
                float(result["framerate_scale_factor"])
                if result.get("framerate_scale_factor") is not None
                else None
            ),
        }
        try:
            fallback_result, fallback_times = _execute_alignment(
                video_path,
                segments,
                audio_stream=audio_stream,
                ffmpeg_path=ffmpeg_path,
                piecewise=False,
            )
            if len(fallback_times) != len(segments):
                raise RuntimeError(
                    "Global fallback changed the subtitle event count "
                    f"from {len(segments)} to {len(fallback_times)}"
                )
            fallback_metrics = _candidate_metrics(segments, fallback_times)
            fallback_successful = (
                bool(fallback_result.get("sync_was_successful"))
                and not fallback_metrics["quality_reasons"]
            )
        except Exception as exc:
            fallback_successful = False
            report["global_fallback_error"] = str(exc)
        if fallback_successful:
            report["piecewise_candidate"] = piecewise_summary
            result = fallback_result
            synchronized_times = fallback_times
            metrics = fallback_metrics
            backend_successful = True
            used_global_fallback = True

    successful = backend_successful and not metrics["quality_reasons"]
    report.update(
        {
            "status": (
                "aligned_global_fallback"
                if successful and used_global_fallback
                else ("aligned" if successful else "rejected_low_quality")
            ),
            "sync_was_successful": backend_successful,
            "pipeline_quality_passed": successful,
            **metrics,
            "offset_seconds": (
                float(result["offset_seconds"])
                if result.get("offset_seconds") is not None
                else None
            ),
            "framerate_scale_factor": (
                float(result["framerate_scale_factor"])
                if result.get("framerate_scale_factor") is not None
                else None
            ),
            "used_global_fallback": used_global_fallback,
        }
    )

    should_apply = mode == "auto" and successful and bool(metrics["changed"])
    report["applied"] = should_apply
    if not should_apply:
        _write_report(report_path, report)
        return [dict(segment) for segment in segments], report

    output: list[Segment] = []
    for segment, (start, end) in zip(segments, synchronized_times):
        item = dict(segment)
        if end <= start:
            report.update(
                {
                    "status": "rejected",
                    "applied": False,
                    "error": "ffsubsync produced a non-positive subtitle duration",
                }
            )
            _write_report(report_path, report)
            return [dict(source) for source in segments], report
        if abs(start - float(segment["start"])) >= 0.001 or abs(end - float(segment["end"])) >= 0.001:
            item["pre_sync_start"] = float(segment["start"])
            item["pre_sync_end"] = float(segment["end"])
        item["start"] = start
        item["end"] = end
        item["subtitle_sync_policy_version"] = SYNC_POLICY_VERSION
        item["subtitle_sync_backend"] = "ffsubsync"
        output.append(item)

    _write_report(report_path, report)
    return output, report
