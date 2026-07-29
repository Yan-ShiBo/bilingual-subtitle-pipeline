from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PLAY_RES_Y = 1080
DEFAULT_FONT_NAME = "Arial"
STYLE_PROFILE_NAMES = ("adaptive", "mobile", "compact")


@dataclass(frozen=True)
class AssStyleProfile:
    chinese_size: int
    source_size: int
    source_only_size: int
    margin_vertical: int
    margin_horizontal_ratio: float
    outline: float
    shadow: float


STYLE_PROFILES = {
    "adaptive": AssStyleProfile(
        chinese_size=52,
        source_size=38,
        source_only_size=46,
        margin_vertical=62,
        margin_horizontal_ratio=0.05,
        outline=2.4,
        shadow=0.8,
    ),
    "mobile": AssStyleProfile(
        chinese_size=60,
        source_size=43,
        source_only_size=52,
        margin_vertical=72,
        margin_horizontal_ratio=0.06,
        outline=2.8,
        shadow=0.9,
    ),
    "compact": AssStyleProfile(
        chinese_size=46,
        source_size=34,
        source_only_size=42,
        margin_vertical=54,
        margin_horizontal_ratio=0.045,
        outline=2.2,
        shadow=0.7,
    ),
}


def parse_ratio(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"0:1", "N/A"}:
        return None
    if ":" in text:
        numerator_text, denominator_text = text.split(":", 1)
        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError:
            return None
        if denominator <= 0:
            return None
        ratio = numerator / denominator
    else:
        try:
            ratio = float(text)
        except ValueError:
            return None
    return ratio if math.isfinite(ratio) and ratio > 0 else None


def calculate_play_resolution(
    width: int,
    height: int,
    sample_aspect_ratio: float = 1.0,
    rotation: int = 0,
    play_res_y: int = DEFAULT_PLAY_RES_Y,
    display_aspect_ratio: float | None = None,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 1920, play_res_y
    aspect_ratio = display_aspect_ratio or (width * sample_aspect_ratio / height)
    if abs(rotation) % 180 == 90:
        aspect_ratio = 1.0 / aspect_ratio
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        aspect_ratio = 16 / 9
    play_res_x = int(round(play_res_y * aspect_ratio))
    play_res_x += play_res_x % 2
    return max(320, min(4320, play_res_x)), play_res_y


def find_ffprobe() -> str | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        candidate = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix else "ffprobe")
        if candidate.exists():
            return str(candidate)
    return None


def stream_rotation(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list") or []:
        try:
            return int(round(float(side_data.get("rotation", 0))))
        except (TypeError, ValueError):
            continue
    try:
        return int(round(float((stream.get("tags") or {}).get("rotate", 0))))
    except (TypeError, ValueError):
        return 0


def probe_video_play_resolution(
    video_path: Path,
    ffprobe_path: str | None = None,
    play_res_y: int = DEFAULT_PLAY_RES_Y,
) -> tuple[int, int]:
    ffprobe = ffprobe_path or find_ffprobe()
    if not ffprobe:
        return 1920, play_res_y
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,sample_aspect_ratio,display_aspect_ratio:"
        "stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [])[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        sample_aspect_ratio = parse_ratio(stream.get("sample_aspect_ratio")) or 1.0
        display_aspect_ratio = parse_ratio(stream.get("display_aspect_ratio"))
        return calculate_play_resolution(
            width,
            height,
            sample_aspect_ratio=sample_aspect_ratio,
            rotation=stream_rotation(stream),
            play_res_y=play_res_y,
            display_aspect_ratio=display_aspect_ratio,
        )
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, IndexError, json.JSONDecodeError):
        return 1920, play_res_y


def sanitize_font_name(value: str | None) -> str:
    name = re.sub(r"[\r\n,]+", " ", str(value or "")).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:120] or DEFAULT_FONT_NAME


def normalized_style_options(
    profile_name: str = "adaptive",
    font_name: str | None = None,
    font_scale: int | float = 100,
) -> tuple[str, str, float]:
    profile = profile_name if profile_name in STYLE_PROFILES else "adaptive"
    try:
        scale = float(font_scale) / 100.0
    except (TypeError, ValueError):
        scale = 1.0
    return profile, sanitize_font_name(font_name), min(1.6, max(0.7, scale))


def ass_style_header(
    play_res_x: int = 1920,
    play_res_y: int = DEFAULT_PLAY_RES_Y,
    profile_name: str = "adaptive",
    font_name: str | None = None,
    font_scale: int | float = 100,
) -> str:
    profile_name, font_name, scale = normalized_style_options(profile_name, font_name, font_scale)
    profile = STYLE_PROFILES[profile_name]
    margin_horizontal = max(20, int(round(play_res_x * profile.margin_horizontal_ratio)))
    margin_vertical = max(20, int(round(profile.margin_vertical * min(scale, 1.3))))
    chinese_size = max(16, int(round(profile.chinese_size * scale)))
    source_size = max(14, int(round(profile.source_size * scale)))
    source_only_size = max(16, int(round(profile.source_only_size * scale)))
    outline = profile.outline * scale
    shadow = profile.shadow * scale
    format_line = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding"
    )

    def style_line(name: str, size: int, primary: str) -> str:
        return (
            f"Style: {name},{font_name},{size},{primary},&H000000FF,&H00000000,"
            f"&H64000000,0,0,0,0,100,100,0,0,1,{outline:.2f},{shadow:.2f},2,"
            f"{margin_horizontal},{margin_horizontal},{margin_vertical},1"
        )

    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "Collisions: Normal",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {play_res_x}",
            f"PlayResY: {play_res_y}",
            "",
            "[V4+ Styles]",
            format_line,
            style_line("Default", chinese_size, "&H00FFFFFF"),
            style_line("Chinese", chinese_size, "&H00FFFFFF"),
            style_line("Source", source_size, "&H00E6E6E6"),
            style_line("SourceOnly", source_only_size, "&H00FFFFFF"),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            "",
        ]
    )
