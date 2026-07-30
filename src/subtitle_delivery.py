from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image


ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def plain_subtitle_text(value: Any) -> str:
    text = ASS_OVERRIDE_RE.sub("", str(value or ""))
    text = text.replace("\\N", "\n").replace("\\n", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def segment_track_text(segment: dict[str, Any], track: str) -> str:
    source = plain_subtitle_text(segment.get("en") or segment.get("text"))
    chinese = plain_subtitle_text(segment.get("zh"))
    if track == "en":
        return source
    if track == "zh":
        return chinese
    if track == "bilingual":
        return "\n".join(value for value in (chinese, source) if value)
    raise ValueError(f"Unsupported subtitle track: {track}")


def write_srt_track(
    segments: list[dict[str, Any]],
    path: Path,
    track: str,
) -> None:
    blocks: list[str] = []
    for segment in segments:
        if segment.get("display", True) is False:
            continue
        text = segment_track_text(segment, track)
        if not text:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        blocks.append(
            "\n".join(
                [
                    str(len(blocks) + 1),
                    f"{srt_timestamp(start)} --> {srt_timestamp(end)}",
                    text,
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    temporary.replace(path)


def write_delivery_srts(
    segments: list[dict[str, Any]],
    output_dir: Path,
    movie_name: str,
) -> dict[str, str]:
    paths = {
        "english_srt": output_dir / f"{movie_name}.en.srt",
        "chinese_srt": output_dir / f"{movie_name}.zh.srt",
        "bilingual_srt": output_dir / f"{movie_name}.bilingual.srt",
    }
    write_srt_track(segments, paths["english_srt"], "en")
    write_srt_track(segments, paths["chinese_srt"], "zh")
    write_srt_track(segments, paths["bilingual_srt"], "bilingual")
    return {key: str(path) for key, path in paths.items()}


def _filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def validate_ass_rendering(
    ass_path: Path,
    segments: list[dict[str, Any]],
    preview_path: Path,
    *,
    play_resolution: tuple[int, int],
    font_directory: Path | None = None,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    visible = [
        segment
        for segment in segments
        if segment.get("display", True) is not False
        and segment_track_text(segment, "bilingual")
        and float(segment.get("end") or 0) > float(segment.get("start") or 0)
    ]
    if not visible:
        return {
            "status": "not_applicable",
            "reason": "No visible subtitle event was available for rendering.",
        }
    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        return {
            "status": "unavailable",
            "reason": "ffmpeg was not found.",
        }

    sample = visible[0]
    midpoint = (
        float(sample["start"]) + float(sample["end"])
    ) / 2
    width = max(320, min(3840, int(play_resolution[0])))
    height = max(180, min(2160, int(play_resolution[1])))
    ass_filter = f"ass=filename='{_filter_path(ass_path)}'"
    if font_directory is not None:
        ass_filter += f":fontsdir='{_filter_path(font_directory)}'"
    video_filter = f"setpts=PTS+{midpoint:.6f}/TB,{ass_filter}"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x101216:s={width}x{height}:r=1:d=1",
        "-vf",
        video_filter,
        "-frames:v",
        "1",
        "-y",
        str(preview_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        with Image.open(preview_path) as image:
            rgb = image.convert("RGB")
            background = (16, 18, 22)
            changed_pixels = sum(
                1
                for red, green, blue in rgb.getdata()
                if max(
                    abs(red - background[0]),
                    abs(green - background[1]),
                    abs(blue - background[2]),
                )
                > 28
            )
            image_size = list(rgb.size)
        if changed_pixels < 32:
            return {
                "status": "failed",
                "reason": "The rendered frame did not contain visible subtitle pixels.",
                "sample_event_id": sample.get("id"),
                "checkpoint_segment_id": sample.get("checkpoint_segment_id"),
                "sample_time_seconds": round(midpoint, 3),
                "changed_pixels": changed_pixels,
                "image_size": image_size,
                "preview_path": str(preview_path),
            }
        return {
            "status": "pass",
            "sample_event_id": sample.get("id"),
            "checkpoint_segment_id": sample.get("checkpoint_segment_id"),
            "sample_time_seconds": round(midpoint, 3),
            "changed_pixels": changed_pixels,
            "image_size": image_size,
            "preview_path": str(preview_path),
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "sample_event_id": sample.get("id"),
            "checkpoint_segment_id": sample.get("checkpoint_segment_id"),
            "sample_time_seconds": round(midpoint, 3),
            "preview_path": str(preview_path),
        }
