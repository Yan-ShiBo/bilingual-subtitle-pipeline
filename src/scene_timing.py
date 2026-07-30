from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


SCENE_TIMING_VERSION = 1
SCENE_TIME_RE = re.compile(r"\bpts_time:([0-9]+(?:\.[0-9]+)?)")


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_scene_times(output: str) -> list[float]:
    values = {
        round(float(match.group(1)), 6)
        for match in SCENE_TIME_RE.finditer(output or "")
    }
    return sorted(value for value in values if math.isfinite(value) and value > 0)


def detect_scene_cuts(
    video_path: Path,
    cache_path: Path,
    *,
    ffmpeg_path: str | None = None,
    threshold: float = 0.35,
    sample_fps: int = 6,
) -> list[float]:
    if not video_path.exists() or video_path.stat().st_size < 1_048_576:
        return []
    config = {
        "threshold": threshold,
        "sample_fps": sample_fps,
        "analysis_width": 320,
    }
    identity = _file_identity(video_path)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("version") == SCENE_TIMING_VERSION
                and cached.get("video") == identity
                and cached.get("config") == config
                and isinstance(cached.get("cuts"), list)
            ):
                return [
                    float(value)
                    for value in cached["cuts"]
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                ]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    scene_filter = (
        f"fps={sample_fps},scale=320:-2:flags=fast_bilinear,"
        f"select='gt(scene,{threshold})',showinfo"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vf",
        scene_filter,
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    cuts = parse_scene_times(result.stderr)
    _write_json_atomic(
        cache_path,
        {
            "version": SCENE_TIMING_VERSION,
            "video": identity,
            "config": config,
            "cuts": cuts,
        },
    )
    return cuts


def _word_bounds(segment: dict[str, Any]) -> tuple[float | None, float | None]:
    starts: list[float] = []
    ends: list[float] = []
    for word in segment.get("words") or []:
        if not isinstance(word, dict):
            continue
        try:
            start = float(word.get("start"))
            end = float(word.get("end"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            starts.append(start)
            ends.append(end)
    return (min(starts), max(ends)) if starts and ends else (None, None)


def align_subtitles_to_scene_cuts(
    segments: list[dict[str, Any]],
    scene_cuts: list[float],
    *,
    max_shift_seconds: float = 0.25,
    min_duration_seconds: float = 5 / 6,
    min_gap_seconds: float = 2 / 24,
    max_duration_seconds: float = 5.5,
) -> list[dict[str, Any]]:
    normalized_cuts: set[float] = set()
    for value in scene_cuts:
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(normalized) and normalized > 0:
            normalized_cuts.add(normalized)
    cuts = sorted(normalized_cuts)
    if not cuts:
        return [dict(segment) for segment in segments]

    ordered = [dict(segment) for segment in segments]
    for index, segment in enumerate(ordered):
        start = float(segment["start"])
        end = float(segment["end"])
        word_start, word_end = _word_bounds(segment)
        snaps: list[str] = []
        previous_end = (
            float(ordered[index - 1]["end"]) if index > 0 else None
        )
        next_start = (
            float(ordered[index + 1]["start"]) if index + 1 < len(ordered) else None
        )

        earlier_start_cuts = [
            cut for cut in cuts if 0 <= start - cut <= max_shift_seconds
        ]
        if earlier_start_cuts:
            candidate = max(earlier_start_cuts)
            minimum_start = (
                previous_end + min_gap_seconds if previous_end is not None else 0.0
            )
            if candidate >= minimum_start:
                start = candidate
                snaps.append("start_to_previous_cut")

        nearby_end_cuts = [
            cut for cut in cuts if abs(cut - end) <= max_shift_seconds
        ]
        if nearby_end_cuts:
            candidate = min(nearby_end_cuts, key=lambda cut: abs(cut - end))
            minimum_end = max(
                start + min_duration_seconds,
                word_end if word_end is not None else start,
            )
            maximum_end = start + max_duration_seconds
            if next_start is not None:
                maximum_end = min(maximum_end, next_start - min_gap_seconds)
            if minimum_end <= candidate <= maximum_end:
                end = candidate
                snaps.append(
                    "end_to_next_cut" if candidate >= float(segment["end"])
                    else "end_to_previous_cut"
                )

        if word_start is not None:
            start = min(start, word_start)
        if word_end is not None:
            end = max(end, word_end)
        if end <= start:
            continue
        segment["start"] = start
        segment["end"] = end
        if snaps:
            segment["scene_timing_adjustments"] = snaps
    return ordered
