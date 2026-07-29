from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


SUPPORTED_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
FONT_DELIVERY_VERSION = 1


def validate_font_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Font file not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
        raise ValueError("Font file must use .ttf, .otf, or .ttc")
    return resolved


def _preferred_name(name_table: Any) -> str:
    for name_id in (16, 1):
        values: list[str] = []
        for record in name_table.names:
            if record.nameID != name_id:
                continue
            try:
                value = record.toUnicode().strip()
            except Exception:
                continue
            if value and value not in values:
                values.append(value)
        if values:
            return values[0]
    return ""


def font_family_name(path: Path) -> str:
    path = validate_font_path(path)
    try:
        from fontTools.ttLib import TTCollection, TTFont
    except ImportError as exc:
        raise RuntimeError("fonttools is required to read the font family name") from exc

    if path.suffix.lower() == ".ttc":
        collection = TTCollection(str(path), lazy=True)
        try:
            if not collection.fonts:
                raise ValueError(f"Font collection contains no fonts: {path}")
            family = _preferred_name(collection.fonts[0]["name"])
        finally:
            collection.close()
    else:
        font = TTFont(str(path), lazy=True)
        try:
            family = _preferred_name(font["name"])
        finally:
            font.close()
    if not family:
        raise ValueError(f"Could not read the internal font family name from {path}")
    return family


def _find_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    command = shutil.which("ffmpeg")
    if command:
        return command
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg is required to package subtitle fonts") from exc


def _font_mime_type(path: Path) -> str:
    if path.suffix.lower() == ".otf":
        return "application/vnd.ms-opentype"
    return "application/x-truetype-font"


def package_ass_with_font(
    ass_path: Path,
    font_path: Path,
    *,
    font_family: str,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    font_path = validate_font_path(font_path)
    if not ass_path.is_file():
        raise FileNotFoundError(f"ASS subtitle file not found: {ass_path}")

    fonts_dir = ass_path.parent / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    copied_font = fonts_dir / font_path.name
    if copied_font.resolve() != font_path.resolve():
        shutil.copy2(font_path, copied_font)

    bundle_path = ass_path.with_suffix(".mks")
    temporary_bundle = bundle_path.with_name(f"{bundle_path.stem}.tmp.mks")
    command = [
        _find_ffmpeg(ffmpeg_path),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(ass_path),
        "-map",
        "0:s:0",
        "-c:s",
        "copy",
        "-metadata:s:s:0",
        "language=zho",
        "-attach",
        str(copied_font),
        "-metadata:s:t:0",
        f"mimetype={_font_mime_type(copied_font)}",
        "-metadata:s:t:0",
        f"filename={copied_font.name}",
        "-f",
        "matroska",
        str(temporary_bundle),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        temporary_bundle.unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg could not package the subtitle font: "
            + (completed.stderr.strip() or f"exit code {completed.returncode}")
        )
    if not temporary_bundle.is_file() or temporary_bundle.stat().st_size <= 0:
        temporary_bundle.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg reported success but did not create the subtitle bundle")
    temporary_bundle.replace(bundle_path)

    manifest = {
        "version": FONT_DELIVERY_VERSION,
        "ass_path": str(ass_path),
        "font_family": font_family,
        "font_file": str(copied_font),
        "matroska_subtitle_bundle": str(bundle_path),
        "mobile_scaling": "ASS PlayRes and style profile remain resolution-relative.",
    }
    manifest_path = ass_path.with_name(f"{ass_path.stem}.font-delivery.json")
    temporary_manifest = manifest_path.with_name(f"{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest
