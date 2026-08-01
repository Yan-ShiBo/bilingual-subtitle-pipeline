from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

from process_lock import FileMutex
from subtitle_pipeline import find_ffmpeg


EVIDENCE_VERSION = 1
CONTEXT_SECONDS = 4.0
MAX_CLIP_SECONDS = 14.0


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_ffmpeg(command: list[str], expected: Path) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=120,
    )
    if result.returncode != 0 or not expected.is_file() or expected.stat().st_size == 0:
        detail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg could not generate review evidence: {detail}")


def _segment_summary(segment: dict[str, Any], index: int, target_index: int) -> dict[str, Any]:
    source = segment.get("en")
    if source in (None, ""):
        source = segment.get("text") or ""
    return {
        "id": segment.get("id"),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "en": str(source or ""),
        "zh": str(segment.get("zh") or ""),
        "display": segment.get("display", True) is not False,
        "is_current": index == target_index,
    }


def _target_index(
    checkpoint_items: list[dict[str, Any]],
    segment_id: Any,
    expected_start: Any = None,
) -> int:
    candidates = [
        index
        for index, item in enumerate(checkpoint_items)
        if str(item.get("id")) == str(segment_id)
    ]
    if expected_start is not None:
        try:
            expected = float(expected_start)
            candidates = [
                index
                for index in candidates
                if abs(float(checkpoint_items[index].get("start")) - expected) <= 0.001
            ]
        except (TypeError, ValueError):
            pass
    if not candidates:
        raise KeyError(f"Checkpoint segment not found: {segment_id}")
    return candidates[0]


def _cache_key(
    video_path: Path,
    segment: dict[str, Any],
    audio_stream: int | None,
) -> str:
    stat = video_path.stat()
    payload = {
        "version": EVIDENCE_VERSION,
        "video": str(video_path.resolve()),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "segment_id": segment.get("id"),
        "start": float(segment.get("start") or 0),
        "end": float(segment.get("end") or 0),
        "audio_stream": audio_stream,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def generate_review_evidence(
    video_path: Path,
    checkpoint_items: list[dict[str, Any]],
    segment_id: Any,
    evidence_root: Path,
    *,
    expected_start: Any = None,
    audio_stream: int | None = None,
    neighbor_count: int = 2,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    index = _target_index(checkpoint_items, segment_id, expected_start)
    segment = checkpoint_items[index]
    try:
        event_start = max(0.0, float(segment.get("start")))
        event_end = max(event_start + 0.05, float(segment.get("end")))
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint segment has invalid timing") from exc

    clip_start = max(0.0, event_start - CONTEXT_SECONDS)
    clip_end = event_end + CONTEXT_SECONDS
    if clip_end - clip_start > MAX_CLIP_SECONDS:
        midpoint = (event_start + event_end) / 2
        clip_start = max(0.0, midpoint - MAX_CLIP_SECONDS / 2)
        clip_end = clip_start + MAX_CLIP_SECONDS
    clip_duration = max(0.5, clip_end - clip_start)
    frame_time = (event_start + event_end) / 2

    cache_key = _cache_key(video_path, segment, audio_stream)
    evidence_dir = evidence_root / cache_key
    frame_path = evidence_dir / "frame.jpg"
    audio_path = evidence_dir / "context.wav"
    waveform_path = evidence_dir / "waveform.png"
    manifest_path = evidence_dir / "manifest.json"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    with FileMutex(evidence_dir / "generate.lock", timeout=130):
        if not (frame_path.is_file() and audio_path.is_file() and waveform_path.is_file()):
            ffmpeg = ffmpeg_path or find_ffmpeg(None)
            frame_temporary = evidence_dir / ".frame.tmp.jpg"
            audio_temporary = evidence_dir / ".context.tmp.wav"
            waveform_temporary = evidence_dir / ".waveform.tmp.png"
            for temporary in (frame_temporary, audio_temporary, waveform_temporary):
                temporary.unlink(missing_ok=True)
            try:
                _run_ffmpeg(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{frame_time:.3f}",
                        "-i",
                        str(video_path),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale='min(1280,iw)':-2",
                        "-q:v",
                        "3",
                        str(frame_temporary),
                    ],
                    frame_temporary,
                )
                audio_command = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{clip_start:.3f}",
                    "-t",
                    f"{clip_duration:.3f}",
                    "-i",
                    str(video_path),
                ]
                audio_command.extend(
                    ["-map", f"0:{audio_stream}"]
                    if audio_stream is not None
                    else ["-map", "0:a:0"]
                )
                audio_command.extend(
                    ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_temporary)]
                )
                _run_ffmpeg(audio_command, audio_temporary)
                _run_ffmpeg(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(audio_temporary),
                        "-filter_complex",
                        "showwavespic=s=1200x180:colors=2f6feb",
                        "-frames:v",
                        "1",
                        str(waveform_temporary),
                    ],
                    waveform_temporary,
                )
                frame_temporary.replace(frame_path)
                audio_temporary.replace(audio_path)
                waveform_temporary.replace(waveform_path)
            finally:
                for temporary in (frame_temporary, audio_temporary, waveform_temporary):
                    temporary.unlink(missing_ok=True)

        start_index = max(0, index - max(0, int(neighbor_count)))
        end_index = min(len(checkpoint_items), index + max(0, int(neighbor_count)) + 1)
        neighbors = [
            _segment_summary(checkpoint_items[item_index], item_index, index)
            for item_index in range(start_index, end_index)
        ]
        manifest = {
            "version": EVIDENCE_VERSION,
            "cache_key": cache_key,
            "video_path": str(video_path),
            "segment_id": segment.get("id"),
            "event_start": event_start,
            "event_end": event_end,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "event_offset": event_start - clip_start,
            "audio_stream": audio_stream,
            "frame_path": str(frame_path),
            "audio_path": str(audio_path),
            "waveform_path": str(waveform_path),
            "neighbors": neighbors,
        }
        _atomic_write_json(manifest_path, manifest)
    return manifest
