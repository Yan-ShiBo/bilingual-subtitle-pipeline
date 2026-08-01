import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ass_styles import STYLE_PROFILE_NAMES
from frontend_settings import settings_store
from output_paths import resolve_output_root
from pipeline_policy import PROCESSING_POLICY_VERSION, TERMINOLOGY_POLICY_VERSION
from remote_bridge_manager import bridge_status, ensure_remote_bridge
from review_evidence import generate_review_evidence
from subtitle_queue import discover_episode_files, ensure_queue_worker, queue_store
from subtitle_sources import (
    SubtitleAsset,
    asset_from_dict,
    build_audio_asset,
    build_embedded_asset,
    build_processing_plan,
    build_sidecar_asset,
)
from subtitle_sync import SUBTITLE_SYNC_MODES

try:
    import psutil
except ImportError:
    psutil = None

try:
    from ssh_tunnel import tunnel_manager
except ImportError:
    tunnel_manager = None


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"
LEGACY_DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / ("1 " + "\u5b57\u5e55")
CONFIGURED_OUTPUT_ROOT = Path(os.environ["SUBTITLE_OUTPUT_ROOT"]).expanduser() if os.environ.get("SUBTITLE_OUTPUT_ROOT") else None
DEFAULT_OUTPUT_ROOT = CONFIGURED_OUTPUT_ROOT or LEGACY_DEFAULT_OUTPUT_ROOT
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m2ts", ".ts", ".mov", ".wmv"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}
RUNS: Dict[int, Dict[str, Any]] = {}
RUNS_LOCK = threading.Lock()
NAME_CACHE: Dict[str, Dict[str, str]] = {}
USER_STOP_MARKER = "Frontend task stopped by user"
REVIEW_MEDIA: Dict[str, Dict[str, Any]] = {}
REVIEW_MEDIA_LOCK = threading.Lock()
REVIEW_MEDIA_LIMIT = 600


def frontend_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(APP_DIR.glob("*.py"), key=lambda item: item.name.lower()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


FRONTEND_SOURCE_SHA256 = frontend_source_fingerprint()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_name(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r"\[[^\]]+\]", " ", name)
    name = re.sub(r"\(([^)]*)\)", r" \1 ", name)
    name = name.replace(".", " ").replace("_", " ")
    name = strip_release_tags(name)
    name = normalize_space(name)
    return name or "Unknown"


def strip_release_tags(name: str) -> str:
    value = re.sub(r"\[[^\]]+\]", " ", name or "")
    value = re.sub(r"\(([^)]*)\)", r" \1 ", value)
    value = value.replace(".", " ").replace("_", " ")
    language_tag = (
        r"(?:eng(?:lish)?|fre|fra|french|ger|deu|german|ita|italian|por|pt|"
        r"spa|es|spanish|cze|ces|hun|pol|rus|tha|tur|jpn|japanese|kor|"
        r"korean|chi|zho|chs|cht)"
    )
    value = re.sub(
        rf"\b{language_tag}(?:[\s-]+{language_tag}){{1,}}\b.*$",
        " ",
        value,
        flags=re.I,
    )
    tag_pattern = (
        r"\b(2160p|1080p|720p|480p|uhd|bluray|blu-ray|bdrip|remux|web[- ]?dl|webrip|"
        r"hdr10\+?|hdr|dv|dovi|dolby|vision|hevc|h265|x265|avc|h264|x264|"
        r"truehd|atmos|dts(?:-hd)?|ma|aac|flac|ddp?|eac3|ac3)\b"
    )
    value = re.sub(tag_pattern + r".*$", " ", value, flags=re.I)
    value = re.sub(tag_pattern, " ", value, flags=re.I)
    value = re.sub(r"\b\d[ .]?\d\b.*$", " ", value)
    value = re.sub(r"\s+-\s*[A-Za-z0-9]+$", " ", value)
    return normalize_space(value)


def clean_output_name(name: str, fallback: str) -> str:
    value = strip_release_tags(str(name or ""))
    value = re.sub(r'[<>:"/\\|?*]+', " ", value)
    value = normalize_space(value).strip(". ")
    return value or fallback


def strip_llm_noise(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def parse_json_response(text: str) -> Any:
    cleaned = strip_llm_noise(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_candidates = [pos for pos in (cleaned.find("["), cleaned.find("{")) if pos >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def remote_ollama_base_url(*, wait_for_connection: bool = True) -> str:
    configured = str(os.environ.get("SUBTITLE_REMOTE_OLLAMA_URL") or "").rstrip("/")
    if configured:
        return configured
    try:
        status = ensure_remote_bridge(
            wait_for_connection=wait_for_connection,
            timeout=25 if wait_for_connection else 3,
        )
    except Exception as exc:
        raise RuntimeError(
            "远程 Ollama 桥接尚未就绪；后台会持续重连，请检查 SSH 配置后重试。"
        ) from exc
    base_url = str(status.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("远程 Ollama 桥接没有发布本地访问地址。")
    return base_url


def remote_settings_support_restart(remote: Dict[str, Any]) -> bool:
    auth_method = str(remote.get("auth_method") or "key").strip().casefold()
    if auth_method != "password":
        return bool(str(remote.get("host") or "").strip())
    return bool(
        str(remote.get("host") or "").strip()
        and remote.get("remember_password")
        and str(remote.get("password") or "")
    )


def persistent_remote_status(*, start_if_configured: bool = True) -> Dict[str, Any]:
    status = bridge_status()
    if not status.get("running") and start_if_configured:
        loaded = settings_store.load()
        remote = loaded.get("remote") if isinstance(loaded, dict) else {}
        if isinstance(remote, dict) and remote_settings_support_restart(remote):
            try:
                status = ensure_remote_bridge(
                    wait_for_connection=False,
                    timeout=3,
                )
            except Exception as exc:
                status = {
                    **status,
                    "connected": False,
                    "status": "reconnecting",
                    "error": str(exc),
                }
    if status.get("running"):
        status["persistent_reconnect"] = True
        return status

    if tunnel_manager:
        frontend_status = tunnel_manager.status()
        if frontend_status.get("connected"):
            frontend_status.update(
                {
                    "running": True,
                    "status": "connected",
                    "models": tunnel_manager.fetch_models(),
                    "persistent_reconnect": False,
                    "connection_owner": "frontend",
                }
            )
            return frontend_status
    status["persistent_reconnect"] = False
    return status


def call_ollama_json(prompt: str, system_prompt: str, model: str = "qwen3:14b", timeout: int = 45) -> Dict[str, Any]:
    base_url = "http://127.0.0.1:11434"
    if model.startswith("remote:"):
        parts = model.split(":", 2)
        if len(parts) != 3:
            raise ValueError("远程模型标识无效，请重新连接远程服务器。")
        base_url = remote_ollama_base_url()
        model = parts[2]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
        "options": {"temperature": 0},
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = (body.get("message") or {}).get("content") or ""
    data = parse_json_response(content)
    if not isinstance(data, dict):
        raise ValueError("Ollama did not return a JSON object")
    return data


def infer_single_file_names(
    input_path: Path,
    video_path: Path,
    fallback: Dict[str, str],
    llm_model: str = "qwen3:14b",
) -> Dict[str, str]:
    try:
        cache_key = f"{video_path.resolve()}::{video_path.stat().st_mtime_ns}::{llm_model}"
    except OSError:
        cache_key = f"{video_path.resolve()}::{llm_model}"
    if cache_key in NAME_CACHE:
        return NAME_CACHE[cache_key]

    system_prompt = "You extract clean movie and episode names from release filenames."
    prompt = f"""
Extract clean movie or episode names from the selected path and the video filename.

Selected name: {input_path.name}
Filename: {video_path.name}
Parent folder of video: {video_path.parent.name}

Return ONLY a JSON object:
{{
  "series_name": "...",
  "movie_name": "...",
  "kind": "movie" or "episode"
}}

Rules:
1. Remove all release-specific noise: tags, codecs, resolution (2160p, 1080p, UHD, etc.), source (Remux, BluRay, Web-dl, etc.), HDR/DV/HDR10, audio channels, release groups, and subtitles info.
2. If Chinese title text is present in the Selected Name, Parent Folder or Filename, PRESERVE it. Combine the Chinese title and English title if both exist (e.g., "某种物质 The Substance").
3. For standalone movies:
   - Set "series_name" to the clean movie title (with the year, e.g. "某种物质 The Substance 2024" or "The Substance 2024").
   - Set "movie_name" to the same clean movie title.
4. For TV episodes:
   - Set "series_name" to the clean show name (e.g. "Game of Thrones").
   - Set "movie_name" to the episode identifier/name (e.g. "S01E01" or "Episode 1").
"""
    try:
        data = call_ollama_json(prompt, system_prompt, model=llm_model)
        movie_name = clean_output_name(data.get("movie_name"), fallback["movie_name"])
        series_name = clean_output_name(data.get("series_name"), fallback["series_name"])
        if str(data.get("kind") or "").lower() == "movie":
            series_name = movie_name
        result = {
            "series_name": series_name,
            "movie_name": movie_name,
            "name_source": f"ollama:{llm_model}",
        }
    except Exception as exc:
        result = {
            "series_name": fallback["series_name"],
            "movie_name": fallback["movie_name"],
            "name_source": f"heuristic ({exc})",
        }

    NAME_CACHE[cache_key] = result
    return result


def safe_file_part(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return value or "subtitle"


def frontend_log_paths(
    series_name: str,
    movie_name: str,
    scope: Path | str | None = None,
) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{safe_file_part(series_name)}__{safe_file_part(movie_name)}"
    if scope is not None:
        scope_key = str(Path(scope).expanduser().resolve(strict=False)).replace("\\", "/").casefold()
        scope_digest = hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:12]
        base = f"{base}.{scope_digest}"
    return LOG_DIR / f"{base}.stdout.log", LOG_DIR / f"{base}.stderr.log"


def find_main_video_in_folder(folder_path: Path) -> Path:
    stream_dir = folder_path / "BDMV" / "STREAM"
    search_roots = [stream_dir] if stream_dir.exists() else [folder_path]
    candidates: List[Path] = []

    for root in search_roots:
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                candidates.append(candidate)

    if not candidates and stream_dir.exists():
        for candidate in folder_path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No supported video file found in {folder_path}")

    return max(candidates, key=lambda path: path.stat().st_size)


def resolve_video_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return find_main_video_in_folder(input_path)
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return input_path
    raise FileNotFoundError(f"Unsupported video input: {input_path}")


def sidecar_match_score(video_path: Path, subtitle_path: Path) -> int:
    video_stem = video_path.stem.casefold()
    subtitle_stem = subtitle_path.stem.casefold()
    if subtitle_stem == video_stem:
        return 100
    if video_stem in subtitle_stem or subtitle_stem in video_stem:
        return 50
    return 0


def list_sidecar_subtitles(selected_path: Path, video_path: Path) -> List[Dict[str, Any]]:
    roots = [video_path.parent]
    if selected_path.is_dir() and selected_path not in roots:
        roots.append(selected_path)
    paths: Dict[Path, int] = {}
    for root in roots:
        for ext in SUBTITLE_EXTENSIONS:
            direct = video_path.with_suffix(ext)
            if direct.exists():
                paths[direct] = max(paths.get(direct, 0), 100)
            for candidate in root.glob(f"*{ext}"):
                match_score = sidecar_match_score(video_path, candidate)
                score = match_score
                lower = candidate.stem.lower()
                if any(token in lower for token in ("en", "eng", "english")):
                    score += 10
                paths[candidate] = max(paths.get(candidate, 0), score)
    has_matched_candidate = any(sidecar_match_score(video_path, path) > 0 for path in paths)
    allow_single_fallback = len(paths) == 1 and not has_matched_candidate
    items: List[Dict[str, Any]] = []
    for path, score in sorted(paths.items(), key=lambda item: (item[1], item[0].name.lower()), reverse=True):
        likely_match = sidecar_match_score(video_path, path) > 0 or allow_single_fallback
        asset = build_sidecar_asset(
            video_path,
            path,
            likely_match=likely_match,
            score=score,
        )
        items.append(
            {
            "path": str(path),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_kb": round(path.stat().st_size / 1024, 1),
            "score": score,
                "likely_match": likely_match,
                "language": asset.language,
                "script": asset.script,
                "role": asset.role,
                "representation": asset.representation,
                "text_authority": asset.text_authority,
                "timing_authority": asset.timing_authority,
                "asset": asset.to_dict(),
            }
        )
    return items


def list_embedded_subtitles(video_path: Path) -> List[Dict[str, Any]]:
    try:
        from subtitle_pipeline import find_ffmpeg, probe_streams, stream_score

        ffmpeg = find_ffmpeg(None)
        streams = [stream for stream in probe_streams(video_path, ffmpeg) if stream.is_subtitle]
        items: List[Dict[str, Any]] = []
        for stream in streams:
            asset = build_embedded_asset(video_path, stream)
            items.append(
                {
                "index": stream.index,
                "label": f"0:{stream.index} {stream.lang or '-'} {stream.codec} {stream.title}".strip(),
                "language": stream.lang,
                "codec": stream.codec,
                "title": stream.title,
                "is_image": asset.representation == "bitmap_ocr",
                "is_text": asset.representation == "authored_text",
                "score": stream_score(stream),
                    "disposition": stream.disposition or {},
                    "script": asset.script,
                    "role": asset.role,
                    "representation": asset.representation,
                    "text_authority": asset.text_authority,
                    "timing_authority": asset.timing_authority,
                    "supported": asset.supported,
                    "asset": asset.to_dict(),
                }
            )
        return items
    except Exception as exc:
        return [{"error": str(exc)}]


def list_embedded_audio(video_path: Path) -> List[Dict[str, Any]]:
    try:
        from subtitle_pipeline import find_ffmpeg, probe_streams

        ffmpeg = find_ffmpeg(None)
        streams = [stream for stream in probe_streams(video_path, ffmpeg) if stream.is_audio]
        items: List[Dict[str, Any]] = []
        for stream in streams:
            asset = build_audio_asset(video_path, stream)
            items.append(
                {
                "index": stream.index,
                "label": f"0:{stream.index} {stream.lang or '-'} {stream.codec} {stream.title}".strip(),
                "language": stream.lang,
                "codec": stream.codec,
                "title": stream.title,
                    "role": asset.role,
                    "disposition": stream.disposition or {},
                    "asset": asset.to_dict(),
                }
            )
        return items
    except Exception as exc:
        return [{"error": str(exc)}]


def _normalized_selected_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        path = Path(raw).expanduser().resolve()
    except OSError:
        path = Path(raw).expanduser().absolute()
    return os.path.normcase(str(path))


def _payload_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() not in {"", "0", "false", "no", "off"}


def select_assets_for_request(
    asset_items: List[tuple[Dict[str, Any], SubtitleAsset]],
    payload: Dict[str, Any],
) -> List[SubtitleAsset]:
    source_mode = str(payload.get("source") or "auto").strip().casefold()
    if source_mode == "srt":
        source_mode = "sidecar"
    subtitle_sync = str(payload.get("subtitle_sync") or "auto").strip().casefold()
    merge_existing = _payload_bool(
        payload.get("merge_existing_subtitles"),
        default=True,
    )
    sidecar_selections = {
        key: _normalized_selected_path(payload.get(key))
        for key in (
            "subtitle_file",
            "chinese_subtitle_file",
            "english_subtitle_file",
        )
    }
    stream_selections: dict[str, int | None] = {}
    for key in (
        "subtitle_stream",
        "chinese_subtitle_stream",
        "english_subtitle_stream",
    ):
        value = payload.get(key)
        if value in (None, ""):
            stream_selections[key] = None
            continue
        try:
            stream_selections[key] = int(value)
        except (TypeError, ValueError):
            stream_selections[key] = None
    explicit_audio_stream: int | None = None
    if payload.get("audio_stream") not in (None, ""):
        try:
            explicit_audio_stream = int(payload["audio_stream"])
        except (TypeError, ValueError):
            explicit_audio_stream = None

    selected: List[SubtitleAsset] = []
    for item, asset in asset_items:
        if asset.origin == "sidecar":
            if source_mode not in {"auto", "sidecar"}:
                continue
            asset_path = _normalized_selected_path(asset.path)
            if merge_existing:
                chinese_path = sidecar_selections["chinese_subtitle_file"]
                source_path = sidecar_selections["english_subtitle_file"]
                if asset_path in {chinese_path, source_path} - {""}:
                    pass
                elif asset.language == "zh" and chinese_path:
                    continue
                elif asset.language != "zh" and source_path:
                    continue
                elif not bool(item.get("likely_match")):
                    continue
            else:
                selected_path = sidecar_selections["subtitle_file"]
                if selected_path and asset_path != selected_path:
                    continue
                if not selected_path and not bool(item.get("likely_match")):
                    continue
        elif asset.origin == "embedded":
            if source_mode not in {"auto", "embedded"}:
                continue
            if merge_existing:
                chinese_stream = stream_selections["chinese_subtitle_stream"]
                source_stream = stream_selections["english_subtitle_stream"]
                if asset.stream_index in {
                    chinese_stream,
                    source_stream,
                } - {None}:
                    pass
                elif asset.language == "zh" and chinese_stream is not None:
                    continue
                elif asset.language != "zh" and source_stream is not None:
                    continue
            else:
                selected_stream = stream_selections["subtitle_stream"]
                if (
                    selected_stream is not None
                    and asset.stream_index != selected_stream
                ):
                    continue
        elif asset.origin == "audio":
            uses_audio = source_mode in {"auto", "audio"} or (
                source_mode in {"sidecar", "embedded"}
                and subtitle_sync != "off"
            )
            if not uses_audio:
                continue
            if (
                explicit_audio_stream is not None
                and asset.stream_index != explicit_audio_stream
            ):
                continue
        else:
            continue
        selected.append(asset)
    return selected


def default_names(input_path: Path, video_path: Path, llm_model: str = "qwen3:14b") -> Dict[str, str]:
    fallback_movie = clean_name(video_path.name)
    if input_path.is_dir():
        fallback_series = clean_name(input_path.name)
    else:
        fallback_series = fallback_movie
    fallback = {"series_name": fallback_series, "movie_name": fallback_movie}
    return infer_single_file_names(input_path, video_path, fallback, llm_model)


def output_dir(output_root: Path, series_name: str, movie_name: str) -> Path:
    return output_root / series_name / movie_name


def resolve_request_output_root(
    payload: Dict[str, Any],
    video: Path,
    series_name: str,
    movie_name: str,
) -> tuple[Path, str]:
    raw_requested = str(payload.get("output_root") or "").strip()
    if not raw_requested and CONFIGURED_OUTPUT_ROOT is not None:
        return CONFIGURED_OUTPUT_ROOT, "configured"
    requested = Path(raw_requested).expanduser() if raw_requested else None
    return resolve_output_root(
        video,
        series_name,
        movie_name,
        requested=requested,
        legacy_default=LEGACY_DEFAULT_OUTPUT_ROOT,
    )


def run_state_path(out_dir: Path, movie_name: str) -> Path:
    return out_dir / f".{safe_file_part(movie_name)}.subtitle-run.json"


def write_run_state(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def remove_run_state(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def process_create_time(pid: int) -> Optional[float]:
    if psutil is None or pid <= 0:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, OSError):
        return None


def process_matches_run(info: Dict[str, Any]) -> bool:
    process = info.get("process")
    if process is not None:
        return process.poll() is None

    pid = int(info.get("pid") or 0)
    if pid <= 0 or psutil is None:
        return False
    try:
        current = psutil.Process(pid)
        expected_create_time = info.get("process_create_time")
        if expected_create_time is not None and abs(current.create_time() - float(expected_create_time)) > 1.0:
            return False
        command = [str(part) for part in current.cmdline()]
    except (psutil.Error, OSError, ValueError):
        return False

    expected_script = str((APP_DIR / "audio_to_subtitle.py").resolve()).casefold()
    normalized_command = [str(Path(part).resolve()).casefold() if part.lower().endswith(".py") else part.casefold() for part in command]
    if expected_script not in normalized_command:
        return False

    expected_series = str(info.get("series_name") or "")
    expected_movie = str(info.get("movie_name") or "")
    expected_output_root = str(info.get("output_root") or "")
    expected_state_path = str(info.get("state_path") or "")
    expected_options = (
        ("--series-name", expected_series),
        ("--movie-name", expected_movie),
        ("--output-root", expected_output_root),
        ("--run-state-file", expected_state_path),
    )
    for option, expected in expected_options:
        if not expected:
            return False
        try:
            option_index = command.index(option)
        except ValueError:
            return False
        if option_index + 1 >= len(command) or command[option_index + 1] != expected:
            return False
    return True


def load_active_run(out_dir: Path, movie_name: str) -> Optional[Dict[str, Any]]:
    state_path = run_state_path(out_dir, movie_name)
    if not state_path.exists():
        return None
    try:
        info = read_json(state_path)
        if not isinstance(info, dict):
            raise ValueError("run state is not an object")
        info["state_path"] = str(state_path)
        if process_matches_run(info):
            return info
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    remove_run_state(state_path)
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def register_review_media(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Review media not found: {resolved}")
    token = secrets.token_urlsafe(24)
    with REVIEW_MEDIA_LOCK:
        if len(REVIEW_MEDIA) >= REVIEW_MEDIA_LIMIT:
            oldest = sorted(
                REVIEW_MEDIA,
                key=lambda key: float(REVIEW_MEDIA[key].get("created_at") or 0),
            )[: max(1, REVIEW_MEDIA_LIMIT // 5)]
            for old_token in oldest:
                REVIEW_MEDIA.pop(old_token, None)
        REVIEW_MEDIA[token] = {
            "path": str(resolved),
            "created_at": time.time(),
        }
    return f"/api/review/media/{token}"


def review_media_path(token: str) -> Path | None:
    with REVIEW_MEDIA_LOCK:
        item = REVIEW_MEDIA.get(str(token or ""))
    if not item:
        return None
    path = Path(str(item.get("path") or ""))
    return path if path.is_file() else None


def build_review_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    selected = Path(str(payload.get("path") or "")).expanduser()
    video = resolve_video_path(selected)
    series_name = str(payload.get("series_name") or "").strip()
    movie_name = str(payload.get("movie_name") or "").strip()
    if not series_name or not movie_name:
        names = default_names(
            selected,
            video,
            str(payload.get("llm_model") or "qwen3:14b"),
        )
        series_name = series_name or names["series_name"]
        movie_name = movie_name or names["movie_name"]
    output_root, _source = resolve_request_output_root(
        payload,
        video,
        series_name,
        movie_name,
    )
    out_dir = output_dir(output_root, series_name, movie_name)
    checkpoint_path = out_dir / f"{movie_name}.segments.checkpoint.json"
    if not checkpoint_path.is_file():
        raise RequestValidationError("没有可用于视听复核的 checkpoint", status=404)
    checkpoint_items = read_json(checkpoint_path)
    if not isinstance(checkpoint_items, list):
        raise RequestValidationError("checkpoint 格式无效")
    audio_stream_raw = payload.get("audio_stream")
    audio_stream = None
    if audio_stream_raw not in (None, ""):
        try:
            audio_stream = int(audio_stream_raw)
        except (TypeError, ValueError) as exc:
            raise RequestValidationError("音轨编号无效") from exc

    evidence = generate_review_evidence(
        video,
        [item for item in checkpoint_items if isinstance(item, dict)],
        payload.get("id"),
        out_dir / ".review-evidence",
        expected_start=payload.get("expected_start"),
        audio_stream=audio_stream,
    )
    evidence["frame_url"] = register_review_media(Path(evidence["frame_path"]))
    evidence["audio_url"] = register_review_media(Path(evidence["audio_path"]))
    evidence["waveform_url"] = register_review_media(Path(evidence["waveform_path"]))
    for key in ("frame_path", "audio_path", "waveform_path"):
        evidence.pop(key, None)
    return evidence


def flatten_quality_review_items(
    report: Dict[str, Any],
    checkpoint_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    segments_by_id = {
        str(item.get("id")): summarize_segment(item)
        for item in checkpoint_items
        if isinstance(item, dict) and item.get("id") is not None
    }
    review_items: List[Dict[str, Any]] = []
    checks = report.get("checks")
    if not isinstance(checks, dict):
        return review_items

    for check_name, check in checks.items():
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "")
        if status not in {"review", "fail"}:
            continue
        samples = check.get("samples")
        sample_items = samples if isinstance(samples, list) else []
        if not sample_items:
            reasons = check.get("reasons")
            normalized_reasons = (
                [str(reason) for reason in reasons if str(reason).strip()]
                if isinstance(reasons, list)
                else []
            )
            review_items.append(
                {
                    "check": str(check_name),
                    "status": status,
                    "issue": normalized_reasons[0] if normalized_reasons else str(check_name),
                    "reasons": normalized_reasons,
                    "segment": None,
                    "details": {
                        key: value
                        for key, value in check.items()
                        if key not in {"samples", "reasons"}
                    },
                }
            )
            continue
        for sample in sample_items:
            if not isinstance(sample, dict):
                continue
            segment_id = sample.get("checkpoint_segment_id", sample.get("id"))
            segment = segments_by_id.get(str(segment_id))
            if segment and segment.get("manual_reviewed_at"):
                continue
            issues = sample.get("issues")
            issue = sample.get("issue")
            if not issue and isinstance(issues, list):
                issue = ", ".join(str(value) for value in issues)
            review_items.append(
                {
                    "check": str(check_name),
                    "status": status,
                    "issue": str(issue or check_name),
                    "segment_id": segment_id,
                    "start": sample.get("start"),
                    "end": sample.get("end"),
                    "segment": segment,
                    "details": sample,
                }
            )
    return review_items[:200]


def checkpoint_info(
    output_root: Path,
    series_name: str,
    movie_name: str,
    llm_model: str = "",
) -> Dict[str, Any]:
    out_dir = output_dir(output_root, series_name, movie_name)
    checkpoint = out_dir / f"{movie_name}.segments.checkpoint.json"
    source = out_dir / f"{movie_name}.segments.source.json"
    processing_plan = out_dir / f"{movie_name}.processing-plan.json"
    quality_report = out_dir / f"{movie_name}.quality-report.json"
    info: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "checkpoint_path": str(checkpoint),
        "source_segments_path": str(source),
        "processing_plan_path": str(processing_plan),
        "quality_report_path": str(quality_report),
        "checkpoint_exists": checkpoint.exists(),
        "source_segments_exists": source.exists(),
        "processing_plan_exists": processing_plan.exists(),
        "quality_report_exists": quality_report.exists(),
        "completed_count": 0,
        "total_count": None,
        "last_item": None,
        "preview": [],
        "checkpoint_compatible": None,
        "quality_review_items": [],
        "quality_review_pending_count": 0,
        "manual_reviewed_count": 0,
    }
    checkpoint_items: List[Dict[str, Any]] = []

    if source.exists():
        try:
            source_items = read_json(source)
            if isinstance(source_items, list):
                info["total_count"] = len(source_items)
        except Exception as exc:
            info["source_error"] = str(exc)

    if processing_plan.exists():
        try:
            plan = read_json(processing_plan)
            if isinstance(plan, dict):
                info["processing_plan"] = plan
        except Exception as exc:
            info["processing_plan_error"] = str(exc)

    if checkpoint.exists():
        try:
            items = read_json(checkpoint)
            if isinstance(items, list):
                checkpoint_items = [item for item in items if isinstance(item, dict)]
                info["completed_count"] = len(items)
                info["manual_reviewed_count"] = sum(
                    bool(item.get("manual_reviewed_at"))
                    for item in checkpoint_items
                )
                if items:
                    checkpoint_model = str(items[0].get("llm_model") or "")
                    checkpoint_policy = int(items[0].get("processing_policy_version") or 0)
                    checkpoint_terminology_policy = int(
                        items[0].get("terminology_policy_version") or 0
                    )
                    info["checkpoint_model"] = checkpoint_model
                    info["checkpoint_policy_version"] = checkpoint_policy
                    info["checkpoint_terminology_policy_version"] = checkpoint_terminology_policy
                    info["current_policy_version"] = PROCESSING_POLICY_VERSION
                    info["current_terminology_policy_version"] = TERMINOLOGY_POLICY_VERSION
                    info["checkpoint_compatible"] = (
                        checkpoint_policy == PROCESSING_POLICY_VERSION
                        and checkpoint_terminology_policy == TERMINOLOGY_POLICY_VERSION
                        and (not llm_model or checkpoint_model == llm_model)
                    )
                visible_items = [item for item in items if item.get("display", True) is not False]
                info["visible_count"] = len(visible_items)
                if visible_items:
                    info["last_item"] = summarize_segment(visible_items[-1])
                    info["preview"] = [summarize_segment(item) for item in visible_items[-10:]]
        except Exception as exc:
            info["checkpoint_error"] = str(exc)

    if quality_report.exists():
        try:
            report = read_json(quality_report)
            if not isinstance(report, dict):
                raise ValueError("quality report is not an object")
            info["quality_status"] = str(report.get("status") or "")
            info["quality_summary"] = report.get("summary") or {}
            info["quality_checks"] = report.get("checks") or {}
            info["quality_review_items"] = flatten_quality_review_items(
                report,
                checkpoint_items,
            )
            info["quality_review_pending_count"] = len(
                info["quality_review_items"]
            )
            info["manual_review_pending"] = bool(report.get("manual_review_pending"))
        except Exception as exc:
            info["quality_report_error"] = str(exc)

    return info


def summarize_segment(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "start": item.get("start"),
        "end": item.get("end"),
        "text": item.get("text"),
        "en": item.get("en"),
        "zh": item.get("zh"),
        "source_language": item.get("source_language"),
        "processing_mode": item.get("processing_mode"),
        "display": item.get("display", True),
        "manual_reviewed_at": item.get("manual_reviewed_at"),
    }


def update_checkpoint_segment(payload: Dict[str, Any]) -> Dict[str, Any]:
    output_root = Path(str(payload.get("output_root") or DEFAULT_OUTPUT_ROOT)).expanduser()
    series_name = str(payload.get("series_name") or "").strip()
    movie_name = str(payload.get("movie_name") or "").strip()
    if not series_name or not movie_name:
        raise RequestValidationError("系列名和片名不能为空")

    with RUNS_LOCK:
        active_runs = list(RUNS.values())
    if any(
        info.get("series_name") == series_name
        and info.get("movie_name") == movie_name
        and process_matches_run(info)
        for info in active_runs
    ):
        raise RequestValidationError("任务运行中，不能同时修改 checkpoint", status=409)

    out_dir = output_dir(output_root, series_name, movie_name)
    checkpoint_path = out_dir / f"{movie_name}.segments.checkpoint.json"
    if not checkpoint_path.exists():
        raise RequestValidationError("没有可编辑的 checkpoint", status=404)
    items = read_json(checkpoint_path)
    if not isinstance(items, list):
        raise RequestValidationError("checkpoint 格式无效")

    target_id = payload.get("id")
    expected_start = payload.get("expected_start")
    target: Dict[str, Any] | None = None
    for item in items:
        if not isinstance(item, dict) or str(item.get("id")) != str(target_id):
            continue
        if expected_start is not None:
            try:
                if abs(float(item.get("start")) - float(expected_start)) > 0.001:
                    continue
            except (TypeError, ValueError):
                continue
        target = item
        break
    if target is None:
        raise RequestValidationError(
            "该字幕已被其他处理更新，请刷新后重试",
            status=409,
        )

    try:
        start = float(payload.get("start"))
        end = float(payload.get("end"))
    except (TypeError, ValueError) as exc:
        raise RequestValidationError("开始和结束时间必须是数字") from exc
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise RequestValidationError("字幕时间范围无效")

    en = str(payload.get("en") or "").strip()
    zh = str(payload.get("zh") or "").strip()
    if not en and not zh:
        raise RequestValidationError("英文和中文不能同时为空")
    if len(en) > 1000 or len(zh) > 1000:
        raise RequestValidationError("单条字幕内容过长")

    target.setdefault("manual_source_start", float(target["start"]))
    target.setdefault("manual_source_end", float(target["end"]))
    target.setdefault("manual_source_text", str(target.get("text") or ""))
    target.update(
        {
            "start": start,
            "end": end,
            "en": en,
            "zh": zh,
            "display": bool(payload.get("display", True)),
            "manual_reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json_atomic(checkpoint_path, items)

    quality_report_path = out_dir / f"{movie_name}.quality-report.json"
    if quality_report_path.exists():
        try:
            report = read_json(quality_report_path)
            if isinstance(report, dict):
                report["status"] = "review"
                report["manual_review_pending"] = True
                report["manual_review_checkpoint_path"] = str(checkpoint_path)
                write_json_atomic(quality_report_path, report)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    return {
        "updated": True,
        "segment": summarize_segment(target),
        "checkpoint_path": str(checkpoint_path),
        "regeneration_required": True,
        "message": "已保存人工复核；请运行“继续任务”重新生成 ASS 和质检报告。",
    }


def analyze_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    selected = Path(payload.get("path", "")).expanduser()
    if not selected.exists():
        raise FileNotFoundError(f"Path not found: {selected}")

    video = resolve_video_path(selected)
    llm_model = payload.get("llm_model") or "qwen3:14b"
    names = default_names(selected, video, llm_model)
    series_name = names["series_name"]
    movie_name = names["movie_name"]
    output_root, output_root_source = resolve_request_output_root(
        payload,
        video,
        series_name,
        movie_name,
    )

    info = checkpoint_info(output_root, series_name, movie_name, llm_model)
    sidecars = list_sidecar_subtitles(selected, video)
    embedded = list_embedded_subtitles(video)
    embedded_audio = list_embedded_audio(video)
    embedded_tracks = [item for item in embedded if "error" not in item]
    audio_tracks = [item for item in embedded_audio if "error" not in item]
    asset_items = [
        (item, asset)
        for item in [*sidecars, *embedded_tracks, *audio_tracks]
        if isinstance(item.get("asset"), dict)
        for asset in [asset_from_dict(item["asset"])]
        if asset is not None
    ]
    assets = [asset for _, asset in asset_items]
    eligible_assets = select_assets_for_request(asset_items, payload)
    source_mode = str(payload.get("source") or "auto").strip().casefold()
    processing_plan = build_processing_plan(
        eligible_assets,
        preferred_source_language=str(payload.get("source_language") or "auto"),
        merge_existing=_payload_bool(
            payload.get("merge_existing_subtitles"),
            default=True,
        ),
        synchronize_existing=str(payload.get("subtitle_sync") or "auto") != "off",
        allow_audio_source=source_mode in {"auto", "audio"},
    )
    auto_source = str(processing_plan.get("recommended_source") or "audio")
    info.update(
        {
            "selected_path": str(selected),
            "selected_is_folder": selected.is_dir(),
            "video_path": str(video),
            "video_size_gb": round(video.stat().st_size / 1024 / 1024 / 1024, 2),
            "output_root": str(output_root),
            "output_root_source": output_root_source,
            "series_name": series_name,
            "movie_name": movie_name,
            "default_series_name": names["series_name"],
            "default_movie_name": names["movie_name"],
            "name_source": names.get("name_source", "heuristic"),
            "has_sidecar_subtitles": bool(sidecars),
            "has_embedded_subtitles": bool(embedded_tracks),
            "has_embedded_audio": bool(audio_tracks),
            "sidecar_subtitles": sidecars,
            "embedded_subtitles": embedded,
            "embedded_audio": embedded_audio,
            "auto_source": auto_source,
            "source_assets": [asset.to_dict() for asset in assets],
            "processing_plan": processing_plan,
            "analysis_config_fingerprint": str(
                payload.get("analysis_config_fingerprint") or ""
            ),
        }
    )
    return info


def choose_file() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[
            ("Video files", "*.mkv *.mp4 *.avi *.m2ts *.ts *.mov *.wmv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def choose_folder() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title="选择视频或蓝光文件夹")
    root.destroy()
    return path


def choose_font() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="选择字幕字体",
        filetypes=[
            ("Font files", "*.ttf *.otf *.ttc"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def remove_translation_outputs(out_dir: Path, movie_name: str) -> None:
    names = [
        f"{movie_name}.segments.checkpoint.json",
        f"{movie_name}.terminology.json",
        f"{movie_name}.final-qa.json",
        f"{movie_name}.autonomous-review.json",
        f"{movie_name}.autonomous-review.round-1.json",
        f"{movie_name}.autonomous-review.round-2.json",
        f"{movie_name}.autonomous-review.round-3.json",
        f"{movie_name}.en.ass",
        f"{movie_name}.zh.ass",
        f"{movie_name}.bilingual.ass",
        f"{movie_name}.bilingual.mks",
        f"{movie_name}.bilingual.font-delivery.json",
        f"{movie_name}.quality-report.json",
    ]
    for name in names:
        path = out_dir / name
        if path.exists():
            path.unlink()
    remove_run_state(run_state_path(out_dir, movie_name))


def remove_resume_files(out_dir: Path, movie_name: str) -> None:
    remove_translation_outputs(out_dir, movie_name)
    source_segments_path = out_dir / f"{movie_name}.segments.source.json"
    if source_segments_path.exists():
        source_segments_path.unlink()
    sync_report_path = out_dir / f"{movie_name}.subtitle-sync.report.json"
    if sync_report_path.exists():
        sync_report_path.unlink()

    import shutil

    embedded_dir = out_dir / "embedded"
    if embedded_dir.exists():
        shutil.rmtree(embedded_dir, ignore_errors=True)

    if out_dir.exists():
        prefix = f".{movie_name}.subtitle-audio."
        for temp_audio in out_dir.iterdir():
            if not temp_audio.is_file() or not temp_audio.name.startswith(prefix) or temp_audio.suffix.lower() != ".wav":
                continue
            try:
                temp_audio.unlink()
            except OSError:
                pass


def resolve_run_mode(payload: Dict[str, Any]) -> str:
    run_mode = str(payload.get("run_mode") or "").strip().lower()
    if run_mode in {"resume", "reprocess", "restart"}:
        return run_mode
    return "restart" if bool(payload.get("restart")) else "resume"


def start_processing(payload: Dict[str, Any]) -> Dict[str, Any]:
    selected = Path(payload["path"]).expanduser()
    series_name = payload.get("series_name") or ""
    movie_name = payload.get("movie_name") or ""
    llm_model = payload.get("llm_model") or "qwen3:14b"
    remote_ollama_url = ""
    if llm_model.startswith("remote:"):
        if len(llm_model.split(":", 2)) != 3:
            raise ValueError("远程模型标识无效，请重新连接远程服务器。")
        remote_ollama_url = remote_ollama_base_url()
    video = resolve_video_path(selected)
    if not series_name or not movie_name:
        names = default_names(selected, video, llm_model)
        series_name = series_name or names["series_name"]
        movie_name = movie_name or names["movie_name"]
    output_root, output_root_source = resolve_request_output_root(
        payload,
        video,
        series_name,
        movie_name,
    )
    run_mode = resolve_run_mode(payload)
    source_mode = payload.get("source") or "auto"
    sidecar_path = payload.get("subtitle_file")
    merge_existing_subtitles = bool(payload.get("merge_existing_subtitles", True))
    chinese_sidecar_path = payload.get("chinese_subtitle_file")
    english_sidecar_path = payload.get("english_subtitle_file")
    chinese_subtitle_text_authority = str(
        payload.get("chinese_subtitle_text_authority") or "authored"
    ).strip().lower()
    english_subtitle_text_authority = str(
        payload.get("english_subtitle_text_authority") or "authored"
    ).strip().lower()
    if chinese_subtitle_text_authority not in {"authored", "ocr"}:
        raise ValueError("Chinese subtitle text authority must be authored or ocr.")
    if english_subtitle_text_authority not in {"authored", "ocr"}:
        raise ValueError("English subtitle text authority must be authored or ocr.")
    subtitle_stream = payload.get("subtitle_stream")
    chinese_subtitle_stream = payload.get("chinese_subtitle_stream")
    english_subtitle_stream = payload.get("english_subtitle_stream")
    audio_stream = payload.get("audio_stream")
    source_language = payload.get("source_language") or "auto"
    asr_language = payload.get("asr_language") or "source"
    subtitle_ocr_lang = payload.get("subtitle_ocr_lang") or "auto"
    batch_size = int(payload.get("batch_size") or 5)
    context_lines = int(payload.get("context_lines") or 30)
    autonomous_review_rounds = int(payload.get("autonomous_review_rounds") or 0)
    if not 0 <= autonomous_review_rounds <= 3:
        raise ValueError("Autonomous review rounds must be between 0 and 3.")
    max_words = int(payload.get("max_words") or 12)
    max_chars = int(payload.get("max_chars") or 42)
    max_duration = float(payload.get("max_duration") or 5.5)
    subtitle_style_profile = str(payload.get("subtitle_style_profile") or "adaptive")
    if subtitle_style_profile not in STYLE_PROFILE_NAMES:
        subtitle_style_profile = "adaptive"
    subtitle_font_name = str(payload.get("subtitle_font_name") or "").strip()
    subtitle_font_scale = max(70, min(160, int(payload.get("subtitle_font_scale") or 100)))
    subtitle_font_file = str(payload.get("subtitle_font_file") or "").strip()
    subtitle_sync = str(payload.get("subtitle_sync") or "auto")
    if subtitle_sync not in SUBTITLE_SYNC_MODES:
        subtitle_sync = "auto"

    out_dir = output_dir(output_root, series_name, movie_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_state_path(out_dir, movie_name)

    log_path, err_path = frontend_log_paths(series_name, movie_name, out_dir)

    args = [
        sys.executable,
        str(APP_DIR / "audio_to_subtitle.py"),
        "--video",
        str(selected),
        "--source",
        source_mode,
        "--output-root",
        str(output_root),
        "--series-name",
        series_name,
        "--movie-name",
        movie_name,
        "--source-language",
        source_language,
        "--asr-language",
        asr_language,
        "--subtitle-ocr-lang",
        subtitle_ocr_lang,
        "--llm-model",
        llm_model,
        "--batch-size",
        str(batch_size),
        "--context-lines",
        str(context_lines),
        "--autonomous-review-rounds",
        str(autonomous_review_rounds),
        "--max-words",
        str(max_words),
        "--max-chars",
        str(max_chars),
        "--max-duration",
        str(max_duration),
        "--subtitle-style-profile",
        subtitle_style_profile,
        "--subtitle-font-scale",
        str(subtitle_font_scale),
        "--subtitle-sync",
        subtitle_sync,
        "--merge-existing-subtitles",
        "yes" if merge_existing_subtitles else "no",
        "--chinese-subtitle-text-authority",
        chinese_subtitle_text_authority,
        "--english-subtitle-text-authority",
        english_subtitle_text_authority,
        "--run-state-file",
        str(state_path),
    ]
    if subtitle_font_name:
        args.extend(["--subtitle-font-name", subtitle_font_name])
    if subtitle_font_file:
        args.extend(["--subtitle-font-file", subtitle_font_file])
    if sidecar_path:
        args.extend(["--subtitle-file", sidecar_path])
    if chinese_sidecar_path:
        args.extend(["--chinese-subtitle-file", chinese_sidecar_path])
    if english_sidecar_path:
        args.extend(["--english-subtitle-file", english_sidecar_path])
    if subtitle_stream not in (None, ""):
        args.extend(["--subtitle-stream", str(subtitle_stream)])
    if chinese_subtitle_stream not in (None, ""):
        args.extend(["--chinese-subtitle-stream", str(chinese_subtitle_stream)])
    if english_subtitle_stream not in (None, ""):
        args.extend(["--english-subtitle-stream", str(english_subtitle_stream)])
    if audio_stream not in (None, ""):
        args.extend(["--audio-stream", str(audio_stream)])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if remote_ollama_url:
        env["SUBTITLE_REMOTE_OLLAMA_URL"] = remote_ollama_url
    with RUNS_LOCK:
        persisted_run = load_active_run(out_dir, movie_name)
        if persisted_run is not None:
            pid = int(persisted_run["pid"])
            RUNS[pid] = persisted_run
            raise RuntimeError(f"任务已在运行，PID {pid}。请先等待完成或终止现有任务。")

        for pid, info in list(RUNS.items()):
            if not process_matches_run(info):
                RUNS.pop(pid, None)
                remove_run_state(Path(info["state_path"]) if info.get("state_path") else None)
                continue
            if info.get("series_name") == series_name and info.get("movie_name") == movie_name:
                raise RuntimeError(f"任务已在运行，PID {pid}。请先等待完成或终止现有任务。")

        if run_mode == "reprocess":
            remove_translation_outputs(out_dir, movie_name)
        elif run_mode == "restart":
            remove_resume_files(out_dir, movie_name)
        for path in (log_path, err_path):
            if path.exists():
                path.unlink()

        stdout = log_path.open("w", encoding="utf-8")
        stderr = err_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(args, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, env=env)
        finally:
            stdout.close()
            stderr.close()
        run_info = {
            "version": 1,
            "pid": process.pid,
            "process_create_time": process_create_time(process.pid),
            "started_at": time.time(),
            "input_path": str(selected),
            "output_root": str(output_root),
            "output_dir": str(out_dir),
            "stdout_log": str(log_path),
            "stderr_log": str(err_path),
            "state_path": str(state_path),
            "command": args,
            "process": process,
            "series_name": series_name,
            "movie_name": movie_name,
            "run_mode": run_mode,
        }
        RUNS[process.pid] = run_info
        persisted_info = {key: value for key, value in run_info.items() if key != "process"}
        write_run_state(state_path, persisted_info)

    return {
        "pid": process.pid,
        "stdout_log": str(log_path),
        "stderr_log": str(err_path),
        "output_root": str(output_root),
        "output_root_source": output_root_source,
        "output_dir": str(out_dir),
        "series_name": series_name,
        "movie_name": movie_name,
        "run_mode": run_mode,
        "command": " ".join(args),
    }


def stop_process_tree(pid: int, recovered_info: Optional[Dict[str, Any]] = None) -> bool:
    if pid <= 0:
        return False
    with RUNS_LOCK:
        info = RUNS.get(pid) or recovered_info
    if not info or not process_matches_run(info):
        remove_run_state(Path(info["state_path"]) if info and info.get("state_path") else None)
        return False
    process = info.get("process")

    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        with RUNS_LOCK:
            RUNS.pop(pid, None)
        remove_run_state(Path(info["state_path"]) if info.get("state_path") else None)
        return result.returncode == 0

    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    elif psutil is not None:
        recovered_process = psutil.Process(pid)
        recovered_process.terminate()
        try:
            recovered_process.wait(timeout=10)
        except psutil.TimeoutExpired:
            recovered_process.kill()
            recovered_process.wait(timeout=10)
    with RUNS_LOCK:
        RUNS.pop(pid, None)
    remove_run_state(Path(info["state_path"]) if info.get("state_path") else None)
    return True


def stop_processing(payload: Dict[str, Any]) -> Dict[str, Any]:
    pid = int(payload.get("pid") or 0)
    output_root = Path(payload.get("output_root") or DEFAULT_OUTPUT_ROOT)
    series_name = payload["series_name"]
    movie_name = payload["movie_name"]
    recovered_info = load_active_run(output_dir(output_root, series_name, movie_name), movie_name)
    if pid == 0 and recovered_info is not None:
        pid = int(recovered_info["pid"])
    if recovered_info is not None and int(recovered_info["pid"]) != pid:
        recovered_info = None
    stopped = stop_process_tree(pid, recovered_info=recovered_info)
    if stopped:
        log_path = (
            Path(recovered_info["stdout_log"])
            if recovered_info and recovered_info.get("stdout_log")
            else frontend_log_paths(
                series_name,
                movie_name,
                output_dir(output_root, series_name, movie_name),
            )[0]
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{USER_STOP_MARKER}\n")
    status = run_status(payload)
    status.update(
        {
            "pid": pid,
            "running": False if stopped else status.get("running", False),
            "stopped": stopped,
            "outcome": "stopped" if stopped else status.get("outcome", "idle"),
            "stage": status.get("stage", "已终止") if stopped else status.get("stage", "未运行"),
            "error_summary": "",
            "message": "任务已终止" if stopped else "没有找到正在运行的前端任务",
        }
    )
    return status


def tail_text(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def active_stage_info(stdout_text: str) -> tuple[str, int]:
    lines = stdout_text.strip().split("\n")
    for line in reversed(lines):
        if "Autonomous overnight review:" in line:
            return "大模型自动复核中", 2
        if "Generating bilingual subtitles" in line:
            return "合并中英文字幕中", 2
        if "Starting sentence-level LLM" in line or "Processing segments" in line or "Starting LLM proofreading" in line or "Proofreading segments" in line:
            return "LLM翻译校对中", 2
        if "Transcribing audio" in line or "Loading faster-whisper" in line:
            return "语音识别中", 1
        if "Extracting audio from" in line:
            return "抽取音频中", 0
        if "Running OCR" in line or "OCR source images:" in line:
            return "OCR图像识别中", 1
        if "Extracting PGS image" in line or "Extracting PGS stream" in line or "Rendering PGS images" in line or "Extracting subtitles from" in line or "Using embedded subtitle stream" in line:
            return "抽取字幕中", 0
        if "Using sidecar" in line:
            return "读取字幕文件中", 0
    return "启动中", 0


def summarize_error(stderr_text: str, max_chars: int = 280) -> str:
    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    if not lines:
        return ""
    summary = next(
        (
            line
            for line in reversed(lines)
            if re.search(r"(?:Error|Exception|Failure|Timeout)(?::|\b)", line)
        ),
        lines[-1],
    )
    if summary.startswith("During handling of the above exception"):
        summary = "处理字幕时发生未知错误"
    if len(summary) > max_chars:
        summary = f"{summary[: max_chars - 3]}..."
    return summary


def infer_stage_info(
    stdout_text: str,
    stderr_text: str,
    running: bool,
    exit_code: int | None = None,
) -> tuple[str, int, str]:
    active_text, active_index = active_stage_info(stdout_text)
    if running:
        return active_text, active_index, "running"
    if "Success! Subtitles saved" in stdout_text:
        return "完成", 3, "success"
    if USER_STOP_MARKER in stdout_text:
        stopped_labels = {
            0: "提取源已终止",
            1: "内容识别已终止",
            2: "翻译与校对已终止",
        }
        return stopped_labels.get(active_index, "已终止"), active_index, "stopped"
    failure_markers = ("Traceback (most recent call last)", "RuntimeError:", "Error:", "FAILED")
    failed = bool(stderr_text.strip()) or any(marker in stdout_text for marker in failure_markers)
    failed = failed or (exit_code is not None and exit_code != 0)
    if failed:
        failed_labels = {
            0: "提取源失败",
            1: "内容识别失败",
            2: "翻译与校对失败",
        }
        return failed_labels.get(active_index, "运行失败"), active_index, "failed"
    if stdout_text.strip():
        interrupted_labels = {
            0: "提取源已中断",
            1: "内容识别已中断",
            2: "翻译与校对已中断",
        }
        return interrupted_labels.get(active_index, "运行已中断"), active_index, "interrupted"
    return "未运行", -1, "idle"


def run_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    pid = int(payload.get("pid") or 0)
    output_root = Path(payload.get("output_root") or DEFAULT_OUTPUT_ROOT)
    series_name = payload["series_name"]
    movie_name = payload["movie_name"]
    out_dir = output_dir(output_root, series_name, movie_name)
    with RUNS_LOCK:
        run_items = list(RUNS.items())
    if pid == 0:
        for p_pid, info in run_items:
            if info["series_name"] == series_name and info["movie_name"] == movie_name:
                if process_matches_run(info):
                    pid = p_pid
                    break

    info_proc = dict(run_items).get(pid)
    if info_proc is None:
        recovered = load_active_run(out_dir, movie_name)
        if recovered is not None and (pid == 0 or int(recovered["pid"]) == pid):
            pid = int(recovered["pid"])
            info_proc = recovered
            with RUNS_LOCK:
                RUNS[pid] = recovered
    exit_code: int | None = None
    if info_proc and info_proc.get("process") is not None:
        try:
            exit_code = info_proc["process"].poll()
        except (AttributeError, OSError):
            exit_code = None
    running = process_matches_run(info_proc) if info_proc else False
    if not running and info_proc:
        with RUNS_LOCK:
            RUNS.pop(pid, None)
        remove_run_state(Path(info_proc["state_path"]) if info_proc.get("state_path") else None)
    if info_proc and info_proc.get("stdout_log") and info_proc.get("stderr_log"):
        log_path = Path(info_proc["stdout_log"])
        err_path = Path(info_proc["stderr_log"])
    else:
        log_path, err_path = frontend_log_paths(series_name, movie_name, out_dir)

    info = checkpoint_info(
        output_root,
        series_name,
        movie_name,
        str(payload.get("llm_model") or ""),
    )
    stdout_tail = tail_text(log_path)
    stderr_tail = tail_text(err_path)
    stage_text, stage_index, outcome = infer_stage_info(
        stdout_tail,
        stderr_tail,
        running,
        exit_code=exit_code,
    )
    info.update(
        {
            "pid": pid,
            "running": running,
            "stage": stage_text,
            "stage_index": stage_index,
            "outcome": outcome,
            "exit_code": exit_code,
            "error_summary": summarize_error(stderr_tail) if outcome == "failed" else "",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return info


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def html_page() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>字幕识别翻译控制台</title>
  <style>
    :root { color-scheme: light; font-family: "Microsoft YaHei", system-ui, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #20242a; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 18px; font-weight: 650; }
    h2 { font-size: 17px; margin: 0 0 12px; }
    section { background: white; border: 1px solid #dde1e7; border-radius: 8px; padding: 18px; margin-bottom: 16px; }
    label { display: block; font-size: 13px; color: #4f5a66; margin-bottom: 6px; }
    input, select, textarea { width: 100%; box-sizing: border-box; border: 1px solid #c8ced8; border-radius: 6px; padding: 0 10px; background: white; color: #1f242b; font: inherit; }
    input, select { height: 38px; }
    textarea { min-height: 88px; padding: 9px 10px; resize: vertical; line-height: 1.5; }
    button { height: 38px; border: 1px solid #1f6feb; background: #1f6feb; color: white; border-radius: 6px; padding: 0 14px; cursor: pointer; }
    button.secondary { background: white; color: #1f6feb; }
    button.danger { background: #b42318; border-color: #b42318; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    [hidden] { display: none !important; }
    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px; align-items: end; }
    .workflow-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px; align-items: end; }
    .workflow-group { grid-column: span 12; border-top: 1px solid #e5e7eb; padding-top: 14px; margin-top: 2px; }
    .workflow-group h3 { font-size: 14px; margin: 0 0 12px; color: #303842; }
    .source-plan-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
    .source-plan-header h3 { margin-bottom: 8px; }
    .source-plan-route { color: #1f6feb; font-size: 13px; font-weight: 600; }
    .source-plan-list { border-top: 1px solid #eef0f3; }
    .source-plan-row { display: grid; grid-template-columns: 132px minmax(0, 1fr); gap: 12px; padding: 9px 0; border-bottom: 1px solid #eef0f3; }
    .source-plan-key { color: #66717f; font-size: 13px; }
    .source-plan-value { min-width: 0; overflow-wrap: anywhere; font-size: 13px; }
    .source-plan-warning { color: #854d0e; margin-top: 9px; font-size: 13px; }
    .checkbox-control { display: flex; align-items: center; gap: 8px; min-height: 38px; margin: 0; color: #20242a; }
    .checkbox-control input { width: auto; height: auto; }
    .field-actions { display: flex; gap: 8px; align-items: end; }
    .field-actions select, .field-actions input { min-width: 0; }
    .field-actions button { flex: 0 0 auto; white-space: nowrap; }
    .field-note { margin-top: 6px; color: #66717f; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
    .field-note.warning { color: #854d0e; }
    .field-note.success { color: #1a6e35; }
    .actions-row { grid-column: span 12; display: flex; justify-content: flex-end; gap: 8px; }
    details.advanced { grid-column: span 12; border-top: 1px solid #e5e7eb; padding-top: 12px; }
    details.advanced summary { width: fit-content; color: #1f6feb; font-size: 13px; cursor: pointer; user-select: none; }
    details.advanced .workflow-grid { margin-top: 12px; }
    .span-2 { grid-column: span 2; }
    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-6 { grid-column: span 6; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .stat { border-right: 1px solid #e5e7eb; padding: 8px 12px; min-height: 52px; }
    .stat:nth-child(4n) { border-right: 0; }
    .stat b { display: block; font-size: 20px; margin-top: 4px; }
    .muted { color: #66717f; font-size: 13px; }
    #paths { overflow-wrap: anywhere; }
    pre { white-space: pre-wrap; word-break: break-word; background: #111827; color: #e5e7eb; border-radius: 8px; padding: 12px; max-height: 320px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px; vertical-align: top; }
    th { color: #4f5a66; font-weight: 600; background: #fafbfc; }
    .inline { display: flex; gap: 8px; align-items: center; }
    .radio { display: flex; flex-wrap: wrap; gap: 8px 16px; min-height: 38px; align-items: center; }
    .radio label { margin: 0; color: #20242a; }
    .radio input { width: auto; height: auto; margin-right: 6px; }
    .stage-container { display: flex; align-items: center; justify-content: space-between; padding: 16px 24px; background: white; border: 1px solid #dde1e7; border-radius: 8px; margin-bottom: 16px; }
    .stage-step { display: flex; align-items: center; gap: 8px; font-weight: 600; color: #4f5a66; }
    .stage-step.completed { color: #1f242b; }
    .stage-step.active { color: #1f242b; }
    .stage-step.failed { color: #b42318; }
    .stage-step.interrupted { color: #854d0e; }
    .dot { width: 14px; height: 14px; border-radius: 50%; background: #cfd6e4; transition: all 0.3s; }
    .stage-step.completed .dot { background: #2da44e; }
    .stage-step.active .dot { background: #bf8700; box-shadow: 0 0 6px rgba(191,135,0,0.6); }
    .stage-step.failed .dot { background: #da3633; }
    .stage-step.interrupted .dot { background: #bf8700; }
    .stage-step.pending .dot { background: #cfd6e4; }
    .stage-line { flex: 1; height: 2px; background: #dde1e7; margin: 0 16px; }
    .status-message { margin: 14px 0 0; padding: 10px 12px; border-left: 4px solid #bf8700; background: #fff8c5; color: #633c01; line-height: 1.55; overflow-wrap: anywhere; }
    .status-message.error { border-left-color: #da3633; background: #ffebe9; color: #82071e; }
    .run-preflight { margin-top: 14px; padding: 14px 0; border-top: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; }
    .run-preflight-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
    .run-preflight-header h3 { margin: 0; font-size: 14px; }
    .preflight-state { color: #854d0e; font-size: 13px; font-weight: 600; }
    .preflight-state.ready { color: #1a6e35; }
    .run-summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .run-summary-item { min-width: 0; }
    .run-summary-item.output { grid-column: 1 / -1; }
    .run-summary-item span { display: block; color: #66717f; font-size: 12px; margin-bottom: 3px; }
    .run-summary-item b { display: block; font-size: 13px; font-weight: 600; overflow-wrap: anywhere; }
    .preflight-message { margin: 10px 0 0; color: #854d0e; font-size: 13px; line-height: 1.45; }
    .preflight-message.ready { color: #4f5a66; }
    .quality-toolbar { display: flex; align-items: end; gap: 12px; margin-bottom: 10px; }
    .quality-toolbar > div { width: min(320px, 100%); }
    .quality-toolbar .muted { margin-left: auto; padding-bottom: 10px; }
    .table-scroll { width: 100%; overflow-x: auto; }
    .quality-status { display: inline-block; min-width: 52px; font-weight: 600; }
    .quality-status.fail { color: #b42318; }
    .quality-status.review { color: #854d0e; }
    .compact-button { height: 30px; padding: 0 10px; white-space: nowrap; }
    .section-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
    .section-header h2 { margin: 0; }
    .queue-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .queue-toolbar .muted { margin-left: auto; }
    .queue-discovery { margin-top: 14px; padding-top: 14px; border-top: 1px solid #e5e7eb; }
    .queue-discovery-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
    .episode-list { max-height: 230px; overflow: auto; border-top: 1px solid #e5e7eb; }
    .episode-row { display: grid; grid-template-columns: 28px minmax(0, 1fr) 88px; gap: 8px; align-items: center; min-height: 38px; border-bottom: 1px solid #eef0f3; }
    .episode-row input { width: auto; height: auto; margin: 0 0 0 7px; }
    .episode-row span { min-width: 0; overflow-wrap: anywhere; }
    .episode-size { color: #66717f; text-align: right; padding-right: 8px; }
    .queue-table td:first-child { width: 32px; color: #66717f; }
    .queue-name { max-width: 360px; overflow-wrap: anywhere; }
    .queue-status { font-weight: 600; white-space: nowrap; }
    .queue-status.running { color: #854d0e; }
    .queue-status.success { color: #1a6e35; }
    .queue-status.failed { color: #b42318; }
    .queue-actions { display: flex; flex-wrap: wrap; gap: 6px; }
    progress { width: 150px; height: 12px; accent-color: #1f6feb; }
    .review-media { min-width: 0; }
    .review-frame { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #111827; border-radius: 6px; }
    .review-waveform { display: block; width: 100%; aspect-ratio: 20 / 3; object-fit: contain; margin-top: 10px; border: 1px solid #dde1e7; border-radius: 6px; }
    .review-audio { width: 100%; margin-top: 10px; }
    .review-context { margin-top: 14px; padding-top: 12px; border-top: 1px solid #e5e7eb; }
    .review-context h3 { font-size: 14px; margin: 0 0 8px; }
    .review-context tr.current { background: #fff8c5; }
    .review-context td:first-child { white-space: nowrap; }
    .review-evidence-status { min-height: 20px; margin: 0 0 8px; }
    @media (max-width: 800px) {
      main { padding: 14px; }
      .grid { grid-template-columns: 1fr; }
      .workflow-grid { grid-template-columns: 1fr; }
      .span-2, .span-3, .span-4, .span-6, .span-8, .span-12 { grid-column: span 1; }
      .workflow-group, details.advanced, .actions-row { grid-column: span 1; }
      .actions-row { justify-content: stretch; }
      .actions-row button { flex: 1; }
      .stats { grid-template-columns: 1fr 1fr; }
      .stat:nth-child(4n) { border-right: 1px solid #e5e7eb; }
      .stat:nth-child(2n) { border-right: 0; }
      .inline { flex-wrap: wrap; }
      .inline .radio { flex-basis: 100%; }
      .stage-container { overflow-x: auto; padding: 14px; gap: 6px; }
      .stage-step { flex: 0 0 auto; white-space: nowrap; }
      .stage-line { min-width: 24px; margin: 0 6px; }
      .quality-toolbar { align-items: stretch; flex-direction: column; }
      .quality-toolbar > div { width: 100%; }
      .quality-toolbar .muted { margin-left: 0; padding-bottom: 0; }
      .source-plan-header { align-items: flex-start; flex-direction: column; gap: 2px; }
      .source-plan-row { grid-template-columns: 1fr; gap: 3px; }
      .run-summary-grid { grid-template-columns: 1fr 1fr; }
      .run-preflight-header { align-items: flex-start; flex-direction: column; gap: 3px; }
      .queue-toolbar .muted { flex-basis: 100%; margin-left: 0; }
    }
    .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); align-items: center; justify-content: center; z-index: 1000; }
    .modal { background: white; padding: 24px; border-radius: 8px; width: 100%; max-width: 400px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .modal.review-modal { box-sizing: border-box; max-width: 1080px; max-height: calc(100vh - 32px); overflow-y: auto; }
    .modal h2 { margin-top: 0; }
    .modal .actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 20px; }
    .review-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .review-grid .full { grid-column: 1 / -1; }
    .review-workbench { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(320px, .8fr); gap: 20px; align-items: start; }
    @media (max-width: 640px) {
      .modal-overlay { align-items: stretch; }
      .modal { max-width: none; margin: 12px; overflow-y: auto; }
      .review-grid { grid-template-columns: 1fr; }
      .review-grid .full { grid-column: 1; }
      .review-workbench { grid-template-columns: 1fr; }
      .review-context .table-scroll { overflow: visible; }
      .review-context table,
      .review-context tbody { display: block; width: 100%; }
      .review-context thead { display: none; }
      .review-context tr { display: grid; gap: 7px; padding: 10px; border-bottom: 1px solid #e5e7eb; }
      .review-context tr.current { border-radius: 6px; }
      .review-context td { display: grid; grid-template-columns: 52px minmax(0, 1fr); gap: 8px; padding: 0; white-space: normal; overflow-wrap: anywhere; }
      .review-context td::before { content: attr(data-label); color: #66717f; font-weight: 600; }
      .review-context td[colspan] { display: block; }
      .review-context td[colspan]::before { content: none; }
      .run-summary-grid { grid-template-columns: 1fr; }
      .episode-row { grid-template-columns: 28px minmax(0, 1fr) 64px; }
      .modal .actions { flex-wrap: wrap; }
      .modal .actions button { flex: 1; }
    }
  </style>
</head>
<body>
<div class="modal-overlay" id="remoteModal">
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="remoteModalTitle">
    <h2 id="remoteModalTitle">远程服务器设置</h2>
    <label for="remoteName">连接名称</label><input id="remoteName" value="AI Server" style="margin-bottom:12px">
    <label for="remoteHost">服务器</label><input id="remoteHost" value="10.12.96.203" placeholder="10.12.96.203 或 ai-server" style="margin-bottom:12px">
    <label for="remotePort">SSH 端口</label><input id="remotePort" type="number" value="22" style="margin-bottom:12px">
    <label for="remoteUser">用户名</label><input id="remoteUser" value="csynth" placeholder="留空读取 SSH config" style="margin-bottom:12px">
    <label for="remoteAuthMethod">认证方式</label>
    <select id="remoteAuthMethod" onchange="updateRemoteAuthVisibility()" style="margin-bottom:12px">
      <option value="key" selected>SSH 密钥</option>
      <option value="password">密码</option>
    </select>
    <div id="remoteKeyOptions">
      <label for="remoteKeyPath">密钥文件</label>
      <input id="remoteKeyPath" placeholder="自动读取 ~/.ssh/config" autocomplete="off" style="margin-bottom:12px">
    </div>
    <div id="remotePasswordOptions" hidden>
      <label for="remotePass">密码</label>
      <input id="remotePass" type="password" autocomplete="current-password" style="margin-bottom:12px">
      <label style="display:flex;align-items:center;gap:8px;margin-bottom:12px"><input id="remoteRememberPassword" type="checkbox">使用 Windows 凭据保护保存密码</label>
    </div>
    <div id="remoteError" role="status" aria-live="polite" style="color:#b42318; font-size:13px; margin-bottom:8px;"></div>
    <div class="actions">
      <button class="secondary" onclick="closeRemoteModal()">取消</button>
      <button onclick="connectRemoteServer()" id="remoteConnectBtn">连接</button>
    </div>
  </div>
</div>
<div class="modal-overlay" id="reviewModal">
  <div class="modal review-modal" role="dialog" aria-modal="true" aria-labelledby="reviewModalTitle">
    <h2 id="reviewModalTitle">视听复核工作台</h2>
    <p class="muted" id="reviewIssue"></p>
    <div class="review-workbench">
      <div class="review-media">
        <p class="muted review-evidence-status" id="reviewEvidenceStatus">正在生成视听证据…</p>
        <img class="review-frame" id="reviewFrame" alt="当前字幕时间点的视频帧" hidden>
        <img class="review-waveform" id="reviewWaveform" alt="当前字幕附近的音频波形" hidden>
        <audio class="review-audio" id="reviewAudio" controls preload="metadata" hidden></audio>
        <div class="review-context">
          <h3>相邻字幕</h3>
          <div class="table-scroll">
            <table>
              <thead><tr><th>时间</th><th>源文 / 英文</th><th>中文</th></tr></thead>
              <tbody id="reviewNeighborRows"><tr><td colspan="3" class="muted">正在读取…</td></tr></tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="review-grid">
        <div>
          <label for="reviewStart">开始时间（秒）</label>
          <input id="reviewStart" type="number" min="0" step="0.001">
        </div>
        <div>
          <label for="reviewEnd">结束时间（秒）</label>
          <input id="reviewEnd" type="number" min="0" step="0.001">
        </div>
        <div class="full">
          <label for="reviewEnglish">源文 / 英文</label>
          <textarea id="reviewEnglish"></textarea>
        </div>
        <div class="full">
          <label for="reviewChinese">中文</label>
          <textarea id="reviewChinese"></textarea>
        </div>
        <div class="full">
          <label class="checkbox-control"><input id="reviewDisplay" type="checkbox" checked>在成片中显示</label>
        </div>
      </div>
    </div>
    <p class="status-message error" id="reviewError" hidden></p>
    <div class="actions">
      <button class="secondary" onclick="openAdjacentReview(-1)">上一条</button>
      <button class="secondary" onclick="openAdjacentReview(1)">下一条</button>
      <button class="secondary" onclick="closeReviewModal()">关闭</button>
      <button onclick="saveCheckpointReview()" id="reviewSaveBtn">保存复核</button>
    </div>
  </div>
</div>
<main>
  <h1>字幕识别翻译控制台</h1>

  <section aria-labelledby="inputHeading">
    <h2 id="inputHeading">输入</h2>
    <div class="grid">
      <div class="span-8">
        <label for="path">视频文件或蓝光文件夹</label>
        <input id="path" placeholder="例如 E:\4杜比HDR\电影\解除好友2：暗网的磁力 ...">
      </div>
      <div class="span-2"><button class="secondary" onclick="selectFile()">选择视频</button></div>
      <div class="span-2"><button class="secondary" onclick="selectFolder()">选择文件夹</button></div>
      <div class="span-4">
        <label for="outputRoot">输出根目录</label>
        <input id="outputRoot" placeholder="自动查找已有输出，或使用视频附近的 1 字幕目录">
      </div>
      <div class="span-4">
        <label for="seriesName">系列名</label>
        <input id="seriesName">
      </div>
      <div class="span-4">
        <label for="movieName">片名/集名</label>
        <input id="movieName">
      </div>
      <div class="span-6">
        <label for="source">字幕来源</label>
        <select id="source">
          <option value="auto" selected>自动：已有字幕 → 内封字幕 → 音频识别</option>
          <option value="sidecar">手动：已有字幕文件</option>
          <option value="embedded">手动：视频内封字幕</option>
          <option value="audio">手动：Whisper 音频识别</option>
        </select>
      </div>
      <div class="span-3">
        <label for="source_language">源字幕语言</label>
        <select id="source_language">
          <option value="auto" selected>自动</option>
          <option value="en">English</option>
          <option value="ja">Japanese</option>
          <option value="ko">Korean</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="es">Spanish</option>
          <option value="zh">中文</option>
        </select>
      </div>
      <div class="span-3">
        <label for="llmModel">校对与翻译模型</label>
        <div class="field-actions">
          <select id="llmModel">
            <option value="qwen3:14b">[Local] qwen3:14b</option>
          </select>
          <button class="secondary" id="remoteBtn" onclick="openRemoteModal()" title="连接远程 Ollama 服务器">连接</button>
        </div>
        <div class="field-note" id="modelConnectionState" role="status" aria-live="polite">本地模型会使用本机计算资源</div>
      </div>

      <div class="workflow-group" id="sourcePlan" hidden>
        <div class="source-plan-header">
          <h3>推荐处理方案</h3>
          <span class="source-plan-route" id="sourcePlanRoute"></span>
        </div>
        <div class="source-plan-list" id="sourcePlanRows"></div>
        <div class="source-plan-warning" id="sourcePlanWarning" hidden></div>
      </div>

      <div class="workflow-group" id="existingSubtitleOptions" hidden>
        <h3>已有字幕</h3>
        <div class="workflow-grid">
          <div class="span-4">
            <label>中英字幕合并</label>
            <label class="checkbox-control"><input id="mergeExistingSubtitles" type="checkbox" checked>合并中文字幕和英文字幕</label>
          </div>
          <div class="span-4">
            <label for="subtitleSync">字幕时间同步</label>
            <select id="subtitleSync">
              <option value="auto" selected>自动校正</option>
              <option value="detect">仅检测并生成报告</option>
              <option value="off">关闭</option>
            </select>
          </div>
          <div class="workflow-grid span-12" id="sidecarOptions" hidden>
            <div class="span-4" id="sidecarPathGroup">
              <label for="sidecarPath">已有字幕文件</label>
              <select id="sidecarPath"><option value="">自动选择</option></select>
            </div>
            <div class="span-4" id="chineseSidecarPathGroup">
              <label for="chineseSidecarPath">中文字幕文件</label>
              <select id="chineseSidecarPath"><option value="">自动选择</option></select>
            </div>
            <div class="span-4" id="englishSidecarPathGroup">
              <label for="englishSidecarPath">英文字幕文件</label>
              <select id="englishSidecarPath"><option value="">可选：自动选择</option></select>
            </div>
          </div>
          <div class="workflow-grid span-12" id="embeddedOptions" hidden>
            <div class="span-4" id="subtitleStreamGroup">
              <label for="subtitleStream">视频内封字幕轨</label>
              <select id="subtitleStream"><option value="">自动选择</option></select>
            </div>
            <div class="span-4" id="chineseSubtitleStreamGroup">
              <label for="chineseSubtitleStream">内封中文字幕轨</label>
              <select id="chineseSubtitleStream"><option value="">自动选择</option></select>
            </div>
            <div class="span-4" id="englishSubtitleStreamGroup">
              <label for="englishSubtitleStream">内封英文字幕轨</label>
              <select id="englishSubtitleStream"><option value="">可选：自动选择</option></select>
            </div>
            <div class="span-4" id="subtitleOcrLangGroup">
              <label for="subtitleOcrLang">图像字幕 OCR 语言</label>
              <select id="subtitleOcrLang">
                <option value="auto" selected>自动</option>
                <option value="en">English</option>
                <option value="ch">简体中文</option>
                <option value="chinese_cht">繁体中文</option>
                <option value="japan">Japanese</option>
                <option value="korean">Korean</option>
                <option value="fr">French</option>
                <option value="german">German</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div class="workflow-group" id="audioRecognitionOptions" hidden>
        <h3 id="audioRecognitionHeading">音频识别</h3>
        <div class="workflow-grid">
          <div class="span-4" id="audioStreamGroup">
            <label for="audioStream" id="audioStreamLabel">用于语音识别的音轨</label>
            <select id="audioStream"><option value="">自动选择</option></select>
          </div>
          <div class="span-4" id="asrLanguageGroup">
            <label for="asrLanguage">音频识别语言</label>
            <select id="asrLanguage">
              <option value="source" selected>跟随源语言</option>
              <option value="auto">自动检测</option>
              <option value="en">English</option>
              <option value="ja">Japanese</option>
              <option value="ko">Korean</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="es">Spanish</option>
              <option value="zh">中文</option>
            </select>
          </div>
        </div>
      </div>

      <details class="advanced" id="advancedSettings">
        <summary>高级设置</summary>
        <div class="workflow-grid">
          <div class="span-2">
            <label for="batchSize">每组字幕</label>
            <input id="batchSize" type="number" value="5" min="1" max="20">
          </div>
          <div class="span-2">
            <label for="contextLines">前后参考单元</label>
            <input id="contextLines" type="number" value="30" min="0" max="100">
          </div>
          <div class="span-2">
            <label for="maxWords">最大词数</label>
            <input id="maxWords" type="number" value="12" min="4" max="40">
          </div>
          <div class="span-2">
            <label for="maxChars">英文每条最大字符</label>
            <input id="maxChars" type="number" value="42" min="20" max="42">
          </div>
          <div class="span-2">
            <label for="maxDuration">最长秒数</label>
            <input id="maxDuration" type="number" value="5.5" min="1" max="20" step="0.5">
          </div>
          <div class="span-4">
            <label class="checkbox-control">
              <input id="autonomousReview" type="checkbox">
              夜间无人值守复核（2 轮）
            </label>
          </div>
        </div>
      </details>
      <details class="advanced" id="subtitleAppearance">
        <summary>字幕外观</summary>
        <div class="workflow-grid">
          <div class="span-4">
            <label for="subtitleStyleProfile">显示方案</label>
            <select id="subtitleStyleProfile">
              <option value="adaptive" selected>自适应画面</option>
              <option value="mobile">手机大字</option>
              <option value="compact">紧凑</option>
            </select>
          </div>
          <div class="span-4">
            <label for="subtitleFontName">字体族</label>
            <input
              id="subtitleFontName"
              placeholder="Arial / Noto Sans CJK SC"
              title="填写字体内部的 family name；外置 ASS 不会自动携带字体文件"
            >
          </div>
          <div class="span-4">
            <label for="subtitleFontScale">字号比例</label>
            <input id="subtitleFontScale" type="number" value="100" min="70" max="160" step="5">
          </div>
          <div class="span-8">
            <label for="subtitleFontFile">字体文件</label>
            <div class="field-actions">
              <input id="subtitleFontFile" placeholder="可选：TTF / OTF / TTC">
              <button class="secondary" onclick="selectFont()">选择字体</button>
            </div>
          </div>
        </div>
      </details>
      <div class="actions-row"><button id="analyzeBtn" onclick="analyze()">分析视频</button></div>
    </div>
  </section>

  <section id="queueSection" aria-labelledby="queueHeading">
    <div class="section-header">
      <h2 id="queueHeading">系列批量队列</h2>
      <span class="muted" id="queueSummary">尚未加入剧集</span>
    </div>
    <div class="queue-toolbar">
      <button class="secondary" id="queueScanBtn" onclick="scanSeriesQueue()">扫描同目录剧集</button>
      <button id="queueResumeBtn" onclick="queueAction('resume')">开始队列</button>
      <button class="secondary" id="queuePauseBtn" onclick="queueAction('pause')">暂停领取</button>
      <button class="secondary" onclick="refreshQueue()">刷新</button>
      <span class="muted" id="queueWorkerState">worker 未启动</span>
    </div>
    <div class="queue-discovery" id="queueDiscovery" hidden>
      <div class="queue-discovery-header">
        <label class="checkbox-control"><input id="queueSelectAll" type="checkbox" onchange="toggleAllEpisodes(this.checked)">全选可加入剧集</label>
        <button class="secondary" id="queueAddBtn" onclick="addSelectedEpisodes()">加入队列</button>
      </div>
      <div class="episode-list" id="episodeList"></div>
    </div>
    <div class="table-scroll" id="queueTableWrap" hidden>
      <table class="queue-table">
        <thead><tr><th>#</th><th>剧集</th><th>状态</th><th>进度</th><th>模型</th><th>操作</th></tr></thead>
        <tbody id="queueRows"></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>状态</h2>
    <div class="stats">
      <div class="stat"><span class="muted">主视频</span><b id="videoSize">-</b></div>
      <div class="stat"><span class="muted">总字幕单元</span><b id="totalCount">未知</b></div>
      <div class="stat"><span class="muted">已完成</span><b id="completedCount">0</b></div>
      <div class="stat"><span class="muted">进程</span><b id="runningState">未运行</b></div>
      <div class="stat"><span class="muted">已有字幕</span><b id="sidecarState">未知</b></div>
      <div class="stat"><span class="muted">内封字幕</span><b id="embeddedState">未知</b></div>
      <div class="stat"><span class="muted">自动来源</span><b id="autoSourceState">-</b></div>
      <div class="stat"><span class="muted">成片质检</span><b id="qualityState">未生成</b></div>
    </div>
    <p class="muted" id="paths"></p>
    <p class="status-message" id="runMessage" role="alert" aria-live="polite" hidden></p>
    <div class="run-preflight" id="runPreflight">
      <div class="run-preflight-header">
        <h3>运行前确认</h3>
        <span class="preflight-state" id="preflightState">等待分析</span>
      </div>
      <div class="run-summary-grid">
        <div class="run-summary-item"><span>字幕来源</span><b id="runSummarySource">等待分析</b></div>
        <div class="run-summary-item"><span>模型位置</span><b id="runSummaryModel">未选择</b></div>
        <div class="run-summary-item"><span>复核策略</span><b id="runSummaryReview">标准终审</b></div>
        <div class="run-summary-item output"><span>输出位置</span><b id="runSummaryOutput">等待分析</b></div>
      </div>
      <p class="preflight-message" id="preflightMessage">请先分析视频，确认字幕轨、同步音轨和处理方案。</p>
    </div>
    <div class="inline">
      <div class="radio" id="runModeGroup" hidden>
        <label><input type="radio" name="runMode" value="resume" checked>继续任务</label>
        <label><input type="radio" name="runMode" value="reprocess">重新校对（保留识别）</label>
        <label><input type="radio" name="runMode" value="restart">完全重建</label>
      </div>
      <span class="muted" id="newTaskMode">新任务</span>
      <button id="runBtn" onclick="startRun()" disabled>运行程序</button>
      <button id="stopBtn" class="danger" onclick="stopRun()" disabled>终止运行</button>
      <button class="secondary" onclick="refreshStatus()">刷新状态</button>
    </div>
  </section>

  <div class="stage-container" id="stageContainer">
    <div class="stage-step" id="step-0"><span class="dot"></span><span class="step-text">提取源</span></div>
    <div class="stage-line"></div>
    <div class="stage-step" id="step-1"><span class="dot"></span><span class="step-text">内容识别</span></div>
    <div class="stage-line"></div>
    <div class="stage-step" id="step-2"><span class="dot"></span><span class="step-text">翻译与校对</span></div>
    <div class="stage-line"></div>
    <div class="stage-step" id="step-3"><span class="dot"></span><span class="step-text">完成</span></div>
  </div>

  <section id="qualityReview" hidden>
    <h2>成片复核</h2>
    <div class="quality-toolbar">
      <div>
        <label for="qualityFilter">问题类型</label>
        <select id="qualityFilter" onchange="renderQualityReviewRows()">
          <option value="">全部问题</option>
        </select>
      </div>
      <span class="muted" id="qualityReviewCount"></span>
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr><th>状态</th><th>检查</th><th>时间</th><th>问题</th><th>字幕</th><th>操作</th></tr></thead>
        <tbody id="qualityReviewRows"></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Checkpoint 预览</h2>
    <table>
      <thead><tr><th>ID</th><th>时间</th><th>源文/英文</th><th>中文</th></tr></thead>
      <tbody id="preview"></tbody>
    </table>
  </section>

  <section>
    <h2>日志</h2>
    <pre id="log"></pre>
  </section>
</main>

<script>
const FORM_STORAGE_KEY = 'subtitleFormState';
const ANALYSIS_STORAGE_KEY = 'subtitleLastAnalysis';
const DISPLAY_LIMITS_VERSION_KEY = 'subtitleDisplayLimitsVersion';
const OUTPUT_ROOT_MIGRATION_VERSION_KEY = 'subtitleOutputRootMigrationVersion';
const LEGACY_DEFAULT_OUTPUT_ROOT = __LEGACY_DEFAULT_OUTPUT_ROOT_JSON__;
const CONFIGURED_OUTPUT_ROOT = __INITIAL_OUTPUT_ROOT_JSON__;
const state = {
  pid: 0,
  running: false,
  lastAnalysis: null,
  currentData: {},
  analyzing: false,
  restoringSettings: false,
  serverSettings: {form: {}, remote: {}},
  remoteStatus: {checked: false, connected: false, status: 'stopped', name: '', error: '', persistentReconnect: false},
  qualityReviewItems: [],
  currentReviewItem: null,
  reviewPayloadOverride: null,
  queue: null,
  discoveredEpisodes: [],
  selectedPath: ''
};
let settingsSaveTimer = 0;
document.getElementById('outputRoot').value = CONFIGURED_OUTPUT_ROOT;

async function api(path, body = {}) {
  const res = await fetch(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

async function selectFile() {
  try {
    const data = await api('/api/select-file');
    if (data.path) {
      setSelectedPath(data.path);
      saveFormState();
    }
  } catch (e) {
    document.getElementById('log').textContent = `选择文件失败: ${e.message}`;
  }
}

async function selectFolder() {
  try {
    const data = await api('/api/select-folder');
    if (data.path) {
      setSelectedPath(data.path);
      saveFormState();
    }
  } catch (e) {
    document.getElementById('log').textContent = `选择文件夹失败: ${e.message}`;
  }
}

async function selectFont() {
  try {
    const data = await api('/api/select-font');
    if (data.path) {
      document.getElementById('subtitleFontFile').value = data.path;
      saveFormState();
    }
  } catch (e) {
    document.getElementById('log').textContent = `选择字体失败: ${e.message}`;
  }
}

const QUEUE_STATUS_LABELS = {
  queued: '等待',
  running: '运行中',
  success: '已完成',
  failed: '失败',
  cancelled: '已取消'
};

function queueModelLabel(item) {
  const model = String(item?.prepared_payload?.llm_model || item?.payload?.llm_model || '');
  if (!model) return '-';
  const parts = model.split(':');
  return model.startsWith('remote:') ? `远程 ${parts.slice(2).join(':')}` : `本地 ${model}`;
}

function renderEpisodeDiscovery() {
  const discovery = document.getElementById('queueDiscovery');
  discovery.hidden = state.discoveredEpisodes.length === 0;
  document.getElementById('episodeList').innerHTML = state.discoveredEpisodes.map((episode, index) => {
    const disabled = Boolean(episode.already_queued);
    return `<label class="episode-row">
      <input type="checkbox" class="episode-check" data-index="${index}" ${disabled ? 'disabled' : 'checked'}>
      <span>${escapeHtml(episode.relative_path || episode.name)}${disabled ? '（已在队列）' : ''}</span>
      <span class="episode-size">${escapeHtml(episode.size_mb)} MB</span>
    </label>`;
  }).join('');
  document.getElementById('queueSelectAll').checked = state.discoveredEpisodes.some(item => !item.already_queued);
}

function toggleAllEpisodes(checked) {
  document.querySelectorAll('.episode-check:not(:disabled)').forEach(input => {
    input.checked = Boolean(checked);
  });
}

async function scanSeriesQueue() {
  const button = document.getElementById('queueScanBtn');
  const selectedPath = document.getElementById('path').value.trim();
  if (!selectedPath) {
    document.getElementById('log').textContent = '请先选择系列中的一个视频或系列文件夹。';
    return;
  }
  button.disabled = true;
  button.textContent = '扫描中…';
  try {
    const data = await api('/api/queue/discover', {path: selectedPath});
    state.discoveredEpisodes = Array.isArray(data.episodes) ? data.episodes : [];
    renderEpisodeDiscovery();
    document.getElementById('log').textContent = `找到 ${state.discoveredEpisodes.length} 个视频文件。`;
  } catch (e) {
    document.getElementById('log').textContent = `扫描剧集失败：${e.message}`;
  } finally {
    button.disabled = false;
    button.textContent = '扫描同目录剧集';
  }
}

async function addSelectedEpisodes() {
  const selected = [...document.querySelectorAll('.episode-check:checked:not(:disabled)')]
    .map(input => state.discoveredEpisodes[Number(input.dataset.index)]?.path)
    .filter(Boolean);
  if (!selected.length) {
    document.getElementById('log').textContent = '没有选择可加入的剧集。';
    return;
  }
  const button = document.getElementById('queueAddBtn');
  button.disabled = true;
  try {
    const data = await api('/api/queue/add', {paths: selected, payload: payload(), start: false});
    state.queue = data;
    renderQueue();
    state.discoveredEpisodes = state.discoveredEpisodes.map(episode => ({
      ...episode,
      already_queued: episode.already_queued || selected.includes(episode.path)
    }));
    renderEpisodeDiscovery();
    document.getElementById('log').textContent = `已将 ${selected.length} 集加入持久化队列。`;
  } catch (e) {
    document.getElementById('log').textContent = `加入队列失败：${e.message}`;
  } finally {
    button.disabled = false;
  }
}

function queueItemActions(item) {
  const id = JSON.stringify(String(item.id));
  if (item.status === 'running') {
    return `<button class="danger compact-button" onclick='queueAction("cancel", ${id})'>终止</button>`;
  }
  if (item.status === 'queued') {
    return [
      `<button class="secondary compact-button" onclick='queueAction("move", ${id}, -1)' title="上移">上移</button>`,
      `<button class="secondary compact-button" onclick='queueAction("move", ${id}, 1)' title="下移">下移</button>`,
      `<button class="secondary compact-button" onclick='queueAction("cancel", ${id})'>取消</button>`
    ].join('');
  }
  const review = Number(item.review_pending_count || 0) > 0
    ? `<button class="secondary compact-button" onclick='loadQueueItem(${id})'>复核</button>`
    : `<button class="secondary compact-button" onclick='loadQueueItem(${id})'>查看</button>`;
  const retry = item.status === 'failed' || item.status === 'cancelled'
    ? `<button class="secondary compact-button" onclick='queueAction("retry", ${id})'>重试</button>`
    : '';
  return `${review}${retry}<button class="secondary compact-button" onclick='queueAction("remove", ${id})'>移除</button>`;
}

function renderQueue() {
  const data = state.queue || {items: [], summary: {}, worker: {}, paused: true};
  const items = Array.isArray(data.items) ? data.items : [];
  const summary = data.summary || {};
  document.getElementById('queueSummary').textContent = items.length
    ? `${summary.running || 0} 运行 / ${summary.queued || 0} 等待 / ${summary.success || 0} 完成 / ${summary.failed || 0} 失败`
    : '尚未加入剧集';
  const worker = data.worker || {};
  document.getElementById('queueWorkerState').textContent = worker.running
    ? `worker ${worker.pid || ''}：${worker.status || '运行中'}`
    : 'worker 未启动';
  const hasRunnableItem = items.some(item => item.status === 'queued' || item.status === 'running');
  document.getElementById('queueResumeBtn').disabled = !hasRunnableItem
    || (!data.paused && Boolean(worker.running));
  document.getElementById('queuePauseBtn').disabled = Boolean(data.paused) || !worker.running;
  document.getElementById('queueTableWrap').hidden = items.length === 0;
  document.getElementById('queueRows').innerHTML = items.map((item, index) => {
    const completed = Number(item.completed_count || 0);
    const total = Number(item.total_count || 0);
    const percent = Number(item.progress_percent || 0);
    const detail = total > 0 ? `${completed}/${total}` : (item.stage || '等待处理');
    const errorTitle = item.error ? ` title="${escapeHtml(item.error)}"` : '';
    return `<tr>
      <td>${index + 1}</td>
      <td class="queue-name"><b>${escapeHtml(item.movie_name || item.name || '')}</b><br><span class="muted">${escapeHtml(item.path || '')}</span></td>
      <td><span class="queue-status ${escapeHtml(item.status || '')}"${errorTitle}>${escapeHtml(QUEUE_STATUS_LABELS[item.status] || item.status || '-')}</span><br><span class="muted">${escapeHtml(item.stage || '')}</span></td>
      <td><progress max="100" value="${Math.max(0, Math.min(100, percent))}"></progress><br><span class="muted">${escapeHtml(detail)}</span></td>
      <td>${escapeHtml(queueModelLabel(item))}</td>
      <td><div class="queue-actions">${queueItemActions(item)}</div></td>
    </tr>`;
  }).join('');
}

async function refreshQueue() {
  try {
    state.queue = await api('/api/queue/list');
    renderQueue();
  } catch (e) {
    document.getElementById('queueWorkerState').textContent = `队列读取失败：${e.message}`;
  }
}

async function queueAction(action, id = '', direction = 0) {
  try {
    state.queue = await api('/api/queue/action', {action, id, direction});
    renderQueue();
  } catch (e) {
    document.getElementById('log').textContent = `队列操作失败：${e.message}`;
  }
}

async function loadQueueItem(id) {
  const item = (state.queue?.items || []).find(entry => String(entry.id) === String(id));
  if (!item) return;
  const prepared = {...(item.prepared_payload || item.payload || {}), pid: Number(item.pid || 0)};
  setSelectedPath(item.path);
  document.getElementById('outputRoot').value = prepared.output_root || item.output_root || '';
  document.getElementById('seriesName').value = prepared.series_name || item.series_name || '';
  document.getElementById('movieName').value = prepared.movie_name || item.movie_name || '';
  if (prepared.llm_model) restoreModelSelection(document.getElementById('llmModel'), prepared.llm_model);
  state.reviewPayloadOverride = prepared;
  state.pid = item.status === 'running' ? Number(item.pid || 0) : 0;
  try {
    const data = await api('/api/status', prepared);
    data.selected_path = item.path;
    data.video_path = item.path;
    render(data);
    setLastAnalysis(data);
    document.getElementById('qualityReview')?.scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (e) {
    document.getElementById('log').textContent = `加载队列剧集失败：${e.message}`;
  }
}

function setSelectedPath(path) {
  const input = document.getElementById('path');
  const changed = normalizePathForCompare(input.value) !== normalizePathForCompare(path)
    || (state.selectedPath && normalizePathForCompare(state.selectedPath) !== normalizePathForCompare(path));
  if (changed) {
    clearSourceSelections();
    clearLastAnalysis();
    state.pid = 0;
    state.reviewPayloadOverride = null;
    document.getElementById('seriesName').value = '';
    document.getElementById('movieName').value = '';
    resetAnalysisPresentation('已选择新视频，请重新分析字幕轨和处理方案。');
  }
  input.value = path;
  state.selectedPath = path;
}

const SOURCE_SELECTION_IDS = [
  'sidecarPath',
  'chineseSidecarPath',
  'englishSidecarPath',
  'subtitleStream',
  'chineseSubtitleStream',
  'englishSubtitleStream',
  'audioStream'
];

function clearSourceSelections() {
  SOURCE_SELECTION_IDS.forEach(id => {
    const element = document.getElementById(id);
    if (element) element.value = '';
  });
}

function activeSourceMode() {
  const selected = document.getElementById('source').value;
  if (selected !== 'auto') return selected;
  const analysis = getMatchingAnalysis();
  return analysis?.auto_source || 'auto';
}

function setHidden(id, hidden) {
  const element = document.getElementById(id);
  if (element) element.hidden = hidden;
}

function selectedEmbeddedAssets() {
  const analysis = getMatchingAnalysis();
  if (!analysis) return [];
  const items = (analysis.embedded_subtitles || []).filter(item => !item.error);
  const merge = document.getElementById('mergeExistingSubtitles').checked;
  const streamIds = (
    merge
      ? ['chineseSubtitleStream', 'englishSubtitleStream']
      : ['subtitleStream']
  ).map(id => document.getElementById(id)?.value).filter(Boolean);
  if (streamIds.length) {
    return streamIds
      .map(value => items.find(item => String(item.index) === String(value)))
      .filter(Boolean);
  }

  const plan = analysis.processing_plan;
  const lane = plan?.lanes || {};
  const selectedIds = [
    lane.chinese,
    lane.source,
    ...(lane.supplementary || [])
  ].filter(Boolean);
  return (plan?.selected_assets || []).filter(
    asset => asset.origin === 'embedded' && selectedIds.includes(asset.asset_id)
  );
}

function embeddedSelectionNeedsOcr() {
  const analysis = getMatchingAnalysis();
  if (!analysis) return true;
  const selected = selectedEmbeddedAssets();
  if (selected.length) {
    return selected.some(
      asset => asset.representation === 'bitmap_ocr' || asset.is_image === true
    );
  }
  return (analysis.embedded_subtitles || []).some(
    asset => asset.representation === 'bitmap_ocr' || asset.is_image === true
  );
}

function updateWorkflowVisibility() {
  const mode = activeSourceMode();
  const automaticSource = document.getElementById('source').value === 'auto';
  const usesSidecar = mode === 'sidecar';
  const usesEmbedded = mode === 'embedded';
  const usesAudio = mode === 'audio';
  const automaticPlanUsesExisting = automaticSource && (
    getMatchingAnalysis()?.processing_plan?.selected_assets || []
  ).some(asset => ['sidecar', 'embedded'].includes(asset.origin));
  const automaticPlanNeedsOcr = automaticSource && (
    getMatchingAnalysis()?.processing_plan?.selected_assets || []
  ).some(asset => asset.representation === 'bitmap_ocr');
  const usesExisting = usesSidecar || usesEmbedded || automaticPlanUsesExisting;
  const merge = document.getElementById('mergeExistingSubtitles').checked;
  const syncsExisting = usesExisting && document.getElementById('subtitleSync').value !== 'off';

  setHidden('existingSubtitleOptions', !usesExisting);
  setHidden('sidecarOptions', !usesSidecar || automaticSource);
  setHidden(
    'embeddedOptions',
    !((usesEmbedded && !automaticSource) || automaticPlanNeedsOcr)
  );
  setHidden('audioRecognitionOptions', !(usesAudio || syncsExisting));
  setHidden('asrLanguageGroup', !usesAudio);
  setHidden('sidecarPathGroup', !usesSidecar || merge);
  setHidden('chineseSidecarPathGroup', !usesSidecar || !merge);
  setHidden('englishSidecarPathGroup', !usesSidecar || !merge);
  setHidden('subtitleStreamGroup', automaticSource || !usesEmbedded || merge);
  setHidden('chineseSubtitleStreamGroup', automaticSource || !usesEmbedded || !merge);
  setHidden('englishSubtitleStreamGroup', automaticSource || !usesEmbedded || !merge);
  setHidden(
    'subtitleOcrLangGroup',
    !(usesEmbedded || automaticPlanNeedsOcr) || !embeddedSelectionNeedsOcr()
  );
  document.getElementById('audioRecognitionHeading').textContent =
    usesAudio ? '音频识别' : '字幕同步音轨';
  document.getElementById('audioStreamLabel').textContent =
    usesAudio ? '用于语音识别的音轨' : '用于字幕同步的音轨';
}

function analysisConfigurationFingerprint() {
  const source = document.getElementById('source').value;
  const ids = ['source', 'source_language', 'mergeExistingSubtitles', 'subtitleSync', 'subtitleOcrLang'];
  if (source === 'sidecar') {
    ids.push('sidecarPath', 'chineseSidecarPath', 'englishSidecarPath');
  }
  if (source === 'embedded') {
    ids.push('subtitleStream', 'chineseSubtitleStream', 'englishSubtitleStream');
  }
  if (source === 'audio') ids.push('asrLanguage');
  if (source === 'auto' || source === 'audio' || document.getElementById('subtitleSync').value !== 'off') {
    ids.push('audioStream');
  }
  return JSON.stringify(ids.map(id => {
    const element = document.getElementById(id);
    return [id, element?.type === 'checkbox' ? Boolean(element.checked) : String(element?.value || '')];
  }));
}

function payload() {
  const sourceLanguage = document.getElementById('source_language').value;
  const source = document.getElementById('source').value;
  const mode = activeSourceMode();
  const usesSidecar = mode === 'sidecar';
  const usesEmbedded = mode === 'embedded';
  const usesAudio = mode === 'audio';
  const usesExisting = usesSidecar || usesEmbedded;
  const explicitSource = source !== 'auto';
  const subtitleSync = document.getElementById('subtitleSync').value;
  const mayUseExisting = usesExisting || (source === 'auto' && mode === 'auto');
  const mergeExisting = mayUseExisting && document.getElementById('mergeExistingSubtitles').checked;
  const usesSelectedAudioTrack =
    usesAudio || (usesExisting && subtitleSync !== 'off') || (source === 'auto' && mode === 'auto');
  return {
    path: document.getElementById('path').value,
    output_root: document.getElementById('outputRoot').value,
    series_name: document.getElementById('seriesName').value,
    movie_name: document.getElementById('movieName').value,
    source,
    subtitle_sync: usesAudio ? 'off' : subtitleSync,
    subtitle_file: explicitSource && usesSidecar && !mergeExisting ? document.getElementById('sidecarPath').value : '',
    merge_existing_subtitles: mergeExisting,
    chinese_subtitle_file: explicitSource && usesSidecar && mergeExisting ? document.getElementById('chineseSidecarPath').value : '',
    english_subtitle_file: explicitSource && usesSidecar && mergeExisting ? document.getElementById('englishSidecarPath').value : '',
    subtitle_stream: explicitSource && usesEmbedded && !mergeExisting ? document.getElementById('subtitleStream').value : '',
    chinese_subtitle_stream: explicitSource && usesEmbedded && mergeExisting ? document.getElementById('chineseSubtitleStream').value : '',
    english_subtitle_stream: explicitSource && usesEmbedded && mergeExisting ? document.getElementById('englishSubtitleStream').value : '',
    audio_stream: usesSelectedAudioTrack ? document.getElementById('audioStream').value : '',
    source_language: sourceLanguage,
    asr_language: source === 'auto' || usesAudio ? resolveAsrLanguageForPayload() : 'auto',
    subtitle_ocr_lang: source === 'auto' || usesEmbedded ? document.getElementById('subtitleOcrLang').value : 'auto',
    llm_model: document.getElementById('llmModel').value,
    batch_size: Number(document.getElementById('batchSize').value || 5),
    context_lines: Number(document.getElementById('contextLines').value || 30),
    autonomous_review_rounds: document.getElementById('autonomousReview').checked ? 2 : 0,
    max_words: Number(document.getElementById('maxWords').value || 12),
    max_chars: Number(document.getElementById('maxChars').value || 42),
    max_duration: Number(document.getElementById('maxDuration').value || 5.5),
    subtitle_style_profile: document.getElementById('subtitleStyleProfile').value,
    subtitle_font_name: document.getElementById('subtitleFontName').value.trim(),
    subtitle_font_scale: Number(document.getElementById('subtitleFontScale').value || 100),
    subtitle_font_file: document.getElementById('subtitleFontFile').value.trim(),
    analysis_config_fingerprint: analysisConfigurationFingerprint()
  };
}

function resolveAsrLanguageForPayload() {
  const sourceLanguage = document.getElementById('source_language').value;
  const asrLanguage = document.getElementById('asrLanguage').value;
  if (asrLanguage === 'source') {
    return sourceLanguage && sourceLanguage !== 'auto' ? sourceLanguage : 'auto';
  }
  return asrLanguage || 'auto';
}

async function analyze() {
  setAnalyzeBusy(true);
  try {
    const base = payload();
    base.series_name = '';
    base.movie_name = '';
    const data = await api('/api/analyze', base);
    setLastAnalysis(data);
    document.getElementById('outputRoot').value = data.output_root || '';
    document.getElementById('seriesName').value = data.series_name;
    document.getElementById('movieName').value = data.movie_name;
    render(data);
    data.analysis_config_fingerprint = analysisConfigurationFingerprint();
    setLastAnalysis(data);
    updateRunControls(data);
    saveFormState();
    const outputNote = data.output_root_source === 'existing'
      ? `已找到已有输出：${data.output_root}`
      : `输出根目录：${data.output_root}`;
    if (data.name_source && data.name_source.includes('heuristic (')) {
      document.getElementById('log').textContent = `${outputNote}\n大模型提取片名失败，已回退到基础识别。报错信息: ${data.name_source}`;
    } else {
      document.getElementById('log').textContent = `分析完成。${outputNote}`;
    }
  } catch (e) {
    document.getElementById('log').textContent = `分析失败: ${e.message}`;
  } finally {
    setAnalyzeBusy(false);
  }
}

async function startRun() {
  const runBtn = document.getElementById('runBtn');
  if (state.pid) return;
  const blockReason = runBlockReason();
  if (blockReason) {
    updateRunControls();
    document.getElementById('log').textContent = blockReason;
    return;
  }
  runBtn.disabled = true;
  try {
    const base = payload();
    base.run_mode = document.querySelector('input[name="runMode"]:checked').value;
    base.restart = base.run_mode === 'restart';
    const data = await api('/api/start', base);
    state.pid = data.pid;
    document.getElementById('outputRoot').value = data.output_root || document.getElementById('outputRoot').value;
    document.getElementById('seriesName').value = data.series_name || document.getElementById('seriesName').value;
    document.getElementById('movieName').value = data.movie_name || document.getElementById('movieName').value;
    const merged = mergeWithAnalysis({...data, running: true, stage: '启动中', stage_index: 0});
    render(merged);
    setLastAnalysis(merged);
    saveFormState();
    document.getElementById('log').textContent = `已启动 PID ${data.pid}\n${data.command || ''}`;
    setTimeout(refreshStatus, 1500);
  } catch (e) {
    document.getElementById('log').textContent = `启动失败: ${e.message}`;
    updateRunControls();
  }
}

async function stopRun() {
  try {
    const base = payload();
    base.pid = state.pid;
    const data = await api('/api/stop', base);
    const merged = mergeWithAnalysis(data);
    if (data.stopped) state.pid = 0;
    render(merged);
    setLastAnalysis(merged);
    const tail = [data.stdout_tail || '', data.stderr_tail || ''].filter(Boolean).join('\n\n--- stderr ---\n');
    document.getElementById('log').textContent = `${data.message || '终止请求已发送'}\n${tail}`;
  } catch (e) {
    document.getElementById('log').textContent = `终止失败: ${e.message}`;
  }
}

async function refreshStatus() {
  const base = payload();
  base.pid = state.pid;
  const data = await api('/api/status', base);
  if (data.pid && data.running && state.pid === 0) {
      state.pid = data.pid;
  }
  const merged = mergeWithAnalysis(data);
  if (!merged.running) state.pid = 0;
  render(merged);
  setLastAnalysis(merged);
  document.getElementById('log').textContent = [data.stdout_tail || '', data.stderr_tail || ''].filter(Boolean).join('\n\n--- stderr ---\n');
}

function setAnalyzeBusy(isBusy) {
  state.analyzing = isBusy;
  const btn = document.getElementById('analyzeBtn');
  if (btn) {
    btn.textContent = isBusy ? '分析中...' : '分析视频';
    btn.disabled = isBusy;
  }
  if (isBusy) {
    document.getElementById('runningState').textContent = '分析中';
    document.getElementById('sidecarState').textContent = '分析中';
    document.getElementById('embeddedState').textContent = '分析中';
    document.getElementById('autoSourceState').textContent = '分析中';
    document.getElementById('paths').textContent = '正在识别片名、扫描字幕来源并读取 checkpoint...';
  }
  updateRunControls();
}

function mergeWithAnalysis(data) {
  const analysis = getMatchingAnalysis();
  if (!analysis) return data;
  const merged = {...analysis};
  Object.entries(data || {}).forEach(([key, value]) => {
    if (value !== undefined) merged[key] = value;
  });
  return merged;
}

function setLastAnalysis(data) {
  state.lastAnalysis = data;
  try {
    const persisted = {...data};
    delete persisted.stdout_tail;
    delete persisted.stderr_tail;
    localStorage.setItem(ANALYSIS_STORAGE_KEY, JSON.stringify(persisted));
  } catch (e) {}
}

function clearLastAnalysis() {
  state.lastAnalysis = null;
  try { localStorage.removeItem(ANALYSIS_STORAGE_KEY); } catch (e) {}
}

function resetAnalysisPresentation(message = '请先分析视频，确认字幕轨、同步音轨和处理方案。') {
  state.currentData = {};
  state.running = false;
  render({outcome: 'idle', stage_index: -1});
  const log = document.getElementById('log');
  if (log && message) log.textContent = message;
}

function getMatchingAnalysis() {
  if (state.lastAnalysis && analysisMatchesCurrentPath(state.lastAnalysis)) return state.lastAnalysis;
  try {
    const stored = JSON.parse(localStorage.getItem(ANALYSIS_STORAGE_KEY));
    if (stored && analysisMatchesCurrentPath(stored)) {
      state.lastAnalysis = stored;
      return stored;
    }
  } catch (e) {}
  return null;
}

function analysisMatchesCurrentPath(analysis) {
  const current = document.getElementById('path').value;
  const selected = analysis?.selected_path;
  const video = analysis?.video_path;
  const normalizedCurrent = normalizePathForCompare(current);
  return Boolean(normalizedCurrent)
    && (normalizePathForCompare(selected) === normalizedCurrent
      || normalizePathForCompare(video) === normalizedCurrent);
}

function normalizePathForCompare(value) {
  return String(value || '').replace(/\//g, '\\').replace(/\\+$/g, '').toLowerCase();
}

function hasCjk(value) {
  return /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]/.test(String(value || ''));
}

function normalizeLanguageTag(value) {
  const lang = String(value || '').trim().toLowerCase();
  if (!lang || ['auto', 'source', 'cached source', 'subtitle', 'embedded subtitle'].includes(lang)) return lang;
  if (['en', 'eng', 'english'].includes(lang) || lang.includes('english')) return 'en';
  if (['ja', 'jp', 'jpn', 'japanese'].includes(lang) || lang.includes('japanese')) return 'ja';
  if (['ko', 'kor', 'korean'].includes(lang) || lang.includes('korean')) return 'ko';
  if (['zh', 'zho', 'chi', 'chs', 'cht', 'cmn', 'cn', 'chinese'].includes(lang) || lang.includes('chinese')) return 'zh';
  return lang.split('-', 1)[0];
}

function sourceLanguageAllowsCjkPreview(item) {
  const lang = normalizeLanguageTag(item?.source_language);
  return Boolean(lang) && !['auto', 'source', 'cached source', 'subtitle', 'embedded subtitle', 'zh'].includes(lang);
}

function previewEnglishText(item) {
  const value = (item.en === undefined || item.en === null) ? (item.text || '') : item.en;
  if (!sourceLanguageAllowsCjkPreview(item) && hasCjk(value)) return '';
  return value;
}

const QUALITY_CHECK_LABELS = {
  pixel_width: '画面宽度',
  readability: '阅读速度',
  timing: '时间轴',
  completeness: '内容完整性',
  model_display_guard: '显示保护',
  terminology: '术语一致性',
  subtitle_sync: '字幕同步',
  source_completeness: '字幕源完整性',
  recognition_confidence: '识别置信度',
  movie_style: '整片风格分析',
  independent_final_qa: '独立终审',
  delivery_render: '成片渲染'
};

const QUALITY_ISSUE_LABELS = {
  invalid_duration: '时间范围无效',
  timeline_overlap: '字幕时间重叠',
  over_max_duration: '持续时间过长',
  under_min_duration: '持续时间过短',
  missing_chinese: '缺少中文',
  chinese_target_has_no_cjk: '中文轨没有中文',
  english_track_contains_cjk: '英文轨混入东亚文字',
  empty_visible_event: '可见字幕为空',
  low_ocr_confidence: 'OCR 置信度偏低',
  low_asr_log_probability: '语音识别置信度偏低',
  high_asr_compression_ratio: '语音识别疑似重复',
  high_no_speech_probability: '可能并非语音',
  low_word_confidence: '多个词识别不确定',
  chinese_cps: '中文阅读速度过快',
  english_cps: '英文阅读速度过快',
  east_asian_source_cps: '源文阅读速度过快'
};

function qualityIssueLabel(value) {
  return String(value || '')
    .split(',')
    .map(item => QUALITY_ISSUE_LABELS[item.trim()] || item.trim())
    .filter(Boolean)
    .join('、');
}

function renderQualityReview(data) {
  state.qualityReviewItems = Array.isArray(data.quality_review_items)
    ? data.quality_review_items
    : [];
  const section = document.getElementById('qualityReview');
  section.hidden = state.qualityReviewItems.length === 0;

  const filter = document.getElementById('qualityFilter');
  const selected = filter.value;
  const checks = [...new Set(state.qualityReviewItems.map(item => item.check).filter(Boolean))];
  filter.innerHTML = '<option value="">全部问题</option>' + checks.map(check =>
    `<option value="${escapeHtml(check)}">${escapeHtml(QUALITY_CHECK_LABELS[check] || check)}</option>`
  ).join('');
  filter.value = checks.includes(selected) ? selected : '';
  renderQualityReviewRows();
}

function renderQualityReviewRows() {
  const filter = document.getElementById('qualityFilter').value;
  const rows = state.qualityReviewItems
    .map((item, index) => ({item, index}))
    .filter(entry => !filter || entry.item.check === filter);
  document.getElementById('qualityReviewCount').textContent =
    `${rows.length} 条待复核`;
  document.getElementById('qualityReviewRows').innerHTML = rows.map(({item, index}) => {
    const segment = item.segment;
    const start = segment?.start ?? item.start ?? '';
    const end = segment?.end ?? item.end ?? '';
    const time = start === '' ? '-' : `${start} → ${end}`;
    const text = segment
      ? [previewEnglishText(segment), segment.zh || ''].filter(Boolean).join(' / ')
      : (item.details?.text || item.reasons?.join('；') || '-');
    const action = segment
      ? `<button class="secondary compact-button" onclick="openQualityReview(${index})">复核</button>`
      : '<span class="muted">查看报告</span>';
    return `<tr>
      <td><span class="quality-status ${escapeHtml(item.status)}">${item.status === 'fail' ? '未通过' : '复核'}</span></td>
      <td>${escapeHtml(QUALITY_CHECK_LABELS[item.check] || item.check)}</td>
      <td>${escapeHtml(time)}</td>
      <td>${escapeHtml(qualityIssueLabel(item.issue))}</td>
      <td>${escapeHtml(text)}</td>
      <td>${action}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="6" class="muted">当前筛选没有待复核项目</td></tr>';
}

function formatReviewTime(value) {
  const total = Math.max(0, Number(value || 0));
  const minutes = Math.floor(total / 60);
  const seconds = (total - minutes * 60).toFixed(3).padStart(6, '0');
  return `${minutes}:${seconds}`;
}

function resetReviewEvidence() {
  const frame = document.getElementById('reviewFrame');
  const waveform = document.getElementById('reviewWaveform');
  const audio = document.getElementById('reviewAudio');
  frame.hidden = true;
  waveform.hidden = true;
  audio.hidden = true;
  frame.removeAttribute('src');
  waveform.removeAttribute('src');
  audio.pause();
  audio.removeAttribute('src');
  audio.load();
  document.getElementById('reviewNeighborRows').innerHTML =
    '<tr><td colspan="3" class="muted">正在读取…</td></tr>';
}

async function openQualityReview(index) {
  const item = state.qualityReviewItems[index];
  if (!item?.segment) return;
  state.currentReviewItem = item;
  const segment = item.segment;
  document.getElementById('reviewIssue').textContent =
    `${QUALITY_CHECK_LABELS[item.check] || item.check}：${qualityIssueLabel(item.issue)}`;
  document.getElementById('reviewStart').value = segment.start ?? '';
  document.getElementById('reviewEnd').value = segment.end ?? '';
  document.getElementById('reviewEnglish').value = previewEnglishText(segment);
  document.getElementById('reviewChinese').value = segment.zh || '';
  document.getElementById('reviewDisplay').checked = segment.display !== false;
  document.getElementById('reviewError').hidden = true;
  resetReviewEvidence();
  const evidenceStatus = document.getElementById('reviewEvidenceStatus');
  evidenceStatus.textContent = '正在按当前时间点生成视频帧、音频和波形…';
  document.getElementById('reviewModal').style.display = 'flex';
  try {
    const request = {...(state.reviewPayloadOverride || payload())};
    request.id = segment.id;
    request.expected_start = segment.start;
    const evidence = await api('/api/review/evidence', request);
    if (state.currentReviewItem !== item) return;
    const frame = document.getElementById('reviewFrame');
    const waveform = document.getElementById('reviewWaveform');
    const audio = document.getElementById('reviewAudio');
    frame.src = evidence.frame_url;
    waveform.src = evidence.waveform_url;
    audio.src = evidence.audio_url;
    frame.hidden = false;
    waveform.hidden = false;
    audio.hidden = false;
    const neighbors = Array.isArray(evidence.neighbors) ? evidence.neighbors : [];
    document.getElementById('reviewNeighborRows').innerHTML = neighbors.map(neighbor =>
      `<tr class="${neighbor.is_current ? 'current' : ''}">
        <td data-label="时间">${escapeHtml(`${formatReviewTime(neighbor.start)} → ${formatReviewTime(neighbor.end)}`)}</td>
        <td data-label="英文">${escapeHtml(neighbor.en || '')}</td>
        <td data-label="中文">${escapeHtml(neighbor.zh || '')}</td>
      </tr>`
    ).join('') || '<tr><td colspan="3" class="muted">没有相邻字幕</td></tr>';
    evidenceStatus.textContent =
      `音频范围 ${formatReviewTime(evidence.clip_start)} → ${formatReviewTime(evidence.clip_end)}，目标字幕从音频第 ${Number(evidence.event_offset || 0).toFixed(2)} 秒开始`;
  } catch (e) {
    if (state.currentReviewItem !== item) return;
    evidenceStatus.textContent = `视听证据生成失败：${e.message}。仍可直接编辑字幕。`;
    document.getElementById('reviewNeighborRows').innerHTML =
      '<tr><td colspan="3" class="muted">无法读取相邻字幕</td></tr>';
  }
}

function openAdjacentReview(direction) {
  const current = state.currentReviewItem;
  if (!current) return;
  const currentIndex = state.qualityReviewItems.indexOf(current);
  for (
    let index = currentIndex + Number(direction);
    index >= 0 && index < state.qualityReviewItems.length;
    index += Number(direction)
  ) {
    if (state.qualityReviewItems[index]?.segment) {
      openQualityReview(index);
      return;
    }
  }
}

function closeReviewModal() {
  resetReviewEvidence();
  state.currentReviewItem = null;
  document.getElementById('reviewModal').style.display = 'none';
}

async function saveCheckpointReview() {
  const item = state.currentReviewItem;
  if (!item?.segment) return;
  const button = document.getElementById('reviewSaveBtn');
  const error = document.getElementById('reviewError');
  button.disabled = true;
  error.hidden = true;
  try {
    const request = {...(state.reviewPayloadOverride || payload())};
    Object.assign(request, {
      id: item.segment.id,
      expected_start: item.segment.start,
      start: Number(document.getElementById('reviewStart').value),
      end: Number(document.getElementById('reviewEnd').value),
      en: document.getElementById('reviewEnglish').value,
      zh: document.getElementById('reviewChinese').value,
      display: document.getElementById('reviewDisplay').checked
    });
    const result = await api('/api/checkpoint/update', request);
    closeReviewModal();
    document.getElementById('log').textContent = result.message || '复核已保存。';
    await refreshStatus();
  } catch (e) {
    error.textContent = `保存失败：${e.message}`;
    error.hidden = false;
  } finally {
    button.disabled = false;
  }
}

const SOURCE_ROUTE_LABELS = {
  proofread_existing_chinese: '已有中文：转简与保守校对',
  translate_existing_source: '已有源字幕：翻译为简体中文',
  transcribe_translate: '语音识别后翻译',
  no_safe_source: '没有可安全自动使用的完整对白来源'
};

const SOURCE_OPERATION_LABELS = {
  extract_bitmap_subtitles: '图像字幕 OCR',
  repair_low_confidence_ocr: '低置信 OCR 修复',
  transcribe_audio: 'Whisper 语音识别',
  repair_asr_risk_windows: 'ASR 风险片段修复',
  traditional_to_simplified: '繁体转简体',
  proofread_existing_chinese: '校对已有中文',
  translate_source: '翻译源字幕',
  synchronize_existing_tracks: '按音频校准时间',
  merge_language_lanes: '合并中英轨',
  translate_missing_chinese_only: '只补译中文缺失内容',
  final_bilingual_quality_review: '整片终审'
};

function selectedModelOption() {
  const select = document.getElementById('llmModel');
  return select && select.selectedIndex >= 0 ? select.options[select.selectedIndex] : null;
}

function isRemoteModel(value) {
  return String(value || '').startsWith('remote:');
}

function restoreModelSelection(select, value) {
  const requested = String(value || '').trim();
  if (!requested) {
    select.value = 'qwen3:14b';
    return;
  }
  const existing = [...select.options].find(option => option.value === requested);
  if (existing) {
    select.value = requested;
    return;
  }
  const modelName = requested.split(':').slice(2).join(':') || requested;
  const label = isRemoteModel(requested)
    ? `[需重新连接] ${modelName}`
    : `[当前不可用] ${requested}`;
  const unavailable = new Option(label, requested, true, true);
  unavailable.disabled = true;
  select.add(unavailable);
  select.value = requested;
}

function runOutputLabel(analysis) {
  const root = document.getElementById('outputRoot').value.trim();
  const series = document.getElementById('seriesName').value.trim();
  const movie = document.getElementById('movieName').value.trim();
  if (analysis?.output_dir && series === analysis.series_name && movie === analysis.movie_name) {
    return analysis.output_dir;
  }
  if (!root) return analysis?.output_dir || '自动选择';
  const separator = root.includes('\\') ? '\\' : '/';
  return [root.replace(/[\\/]+$/g, ''), series, movie].filter(Boolean).join(separator);
}

function runBlockReason() {
  if (state.analyzing) return '正在分析视频，请等待分析完成。';
  if (state.running || state.pid) return '任务正在运行。';
  if (!document.getElementById('path').value.trim()) return '请先选择视频文件或蓝光文件夹。';
  const analysis = getMatchingAnalysis();
  if (!analysis) return '请先分析视频，确认字幕轨、同步音轨和处理方案。';
  if (
    !analysis.analysis_config_fingerprint
    || analysis.analysis_config_fingerprint !== analysisConfigurationFingerprint()
  ) {
    return '来源或音轨设置已经变化，请重新分析后再运行。';
  }
  if (!analysis.processing_plan) return '来源设置已经变化，请重新分析后再运行。';
  if (analysis.processing_plan.route === 'no_safe_source') {
    return '没有可安全使用的完整对白来源，请调整字幕来源或音轨后重新分析。';
  }
  if (!document.getElementById('seriesName').value.trim() || !document.getElementById('movieName').value.trim()) {
    return '系列名和片名不能为空。';
  }
  const model = document.getElementById('llmModel').value.trim();
  const option = selectedModelOption();
  if (!model) return '请选择校对与翻译模型。';
  if (option?.disabled) return '上次选择的模型当前不可用，请重新连接远程服务器或明确选择本地模型。';
  if (isRemoteModel(model) && !state.remoteStatus.connected) {
    return '所选远程模型尚未连接。请先点击“连接”，本任务不会自动改用本地模型。';
  }
  return '';
}

function updateModelConnectionState() {
  const note = document.getElementById('modelConnectionState');
  const model = document.getElementById('llmModel').value;
  const option = selectedModelOption();
  note.className = 'field-note';
  if (isRemoteModel(model)) {
    if (state.remoteStatus.connected && !option?.disabled) {
      note.textContent = state.remoteStatus.persistentReconnect
        ? `已连接 ${state.remoteStatus.name || '远程服务器'}，独立 SSH 桥接会在网页退出或网络切换后自动重连`
        : `已连接 ${state.remoteStatus.name || '远程服务器'}，当前连接依赖此网页进程`;
      note.classList.add('success');
    } else if (state.remoteStatus.status === 'reconnecting') {
      note.textContent = `SSH 桥接正在自动重连：${state.remoteStatus.error || '等待网络恢复'}`;
      note.classList.add('warning');
    } else {
      note.textContent = '远程模型未连接，运行已锁定，不会静默改用本地 GPU';
      note.classList.add('warning');
    }
  } else if (model) {
    note.textContent = '本地模型会使用本机计算资源';
  } else {
    note.textContent = '尚未选择可用模型';
    note.classList.add('warning');
  }
}

function updateRunControls(data = state.currentData) {
  const analysis = getMatchingAnalysis();
  const source = analysis?.processing_plan
    ? (SOURCE_ROUTE_LABELS[analysis.processing_plan.route] || analysis.processing_plan.route || '自动推荐')
    : '等待分析';
  const option = selectedModelOption();
  const model = document.getElementById('llmModel').value;
  const modelLabel = option?.textContent?.trim() || '未选择';
  const reviewLabel = document.getElementById('autonomousReview').checked
    ? '标准终审 + 夜间复核 2 轮'
    : '标准终审';
  const hasCheckpoint = Boolean((data || {}).checkpoint_exists ?? analysis?.checkpoint_exists);
  const runModeGroup = document.getElementById('runModeGroup');
  const newTaskMode = document.getElementById('newTaskMode');
  runModeGroup.hidden = !hasCheckpoint;
  newTaskMode.hidden = hasCheckpoint;
  if (!hasCheckpoint) {
    const resume = document.querySelector('input[name="runMode"][value="resume"]');
    if (resume) resume.checked = true;
  }

  document.getElementById('runSummarySource').textContent = source;
  document.getElementById('runSummaryModel').textContent = modelLabel;
  document.getElementById('runSummaryReview').textContent = reviewLabel;
  document.getElementById('runSummaryOutput').textContent = runOutputLabel(analysis);
  updateModelConnectionState();

  const reason = runBlockReason();
  const preflightState = document.getElementById('preflightState');
  const preflightMessage = document.getElementById('preflightMessage');
  const runBtn = document.getElementById('runBtn');
  const ready = !reason;
  preflightState.textContent = state.running ? '运行中' : (ready ? '可以运行' : '需要处理');
  preflightState.className = ready ? 'preflight-state ready' : 'preflight-state';
  preflightMessage.textContent = reason || (
    isRemoteModel(model)
      ? '远程连接、字幕来源和输出位置已确认。'
      : '当前明确选择本地模型，翻译阶段可能占用本机 GPU。'
  );
  preflightMessage.className = ready ? 'preflight-message ready' : 'preflight-message';
  runBtn.disabled = Boolean(state.running || state.analyzing || reason);
  runBtn.title = reason || '按以上设置启动任务';
}

function sourceAssetLabel(asset) {
  if (!asset) return '不使用';
  const origins = {embedded: '内封', sidecar: '外置', audio: '音频'};
  const representations = {
    authored_text: '人工文本',
    bitmap_ocr: '图像字幕',
    speech_asr: '语音识别',
    unsupported: '不支持'
  };
  const roles = {dialogue: '完整对白', sdh: 'SDH', forced: '强制字幕', commentary: '评论', unknown: '角色待确认'};
  const scripts = {simplified: '简体', traditional: '繁体', unknown: ''};
  return [
    origins[asset.origin] || asset.origin,
    asset.label,
    asset.language || '语言待确认',
    scripts[asset.script] || '',
    representations[asset.representation] || asset.representation,
    roles[asset.role] || asset.role
  ].filter(Boolean).join(' · ');
}

function setPlannedSelectValue(id, value) {
  if (value === undefined || value === null || value === '') return;
  const select = document.getElementById(id);
  if (select && [...select.options].some(option => option.value === String(value) && !option.disabled)) {
    select.value = String(value);
  }
}

function applyProcessingPlanSelections(plan) {
  if (!plan || !Array.isArray(plan.selected_assets)) return;
  const assets = new Map(plan.selected_assets.map(asset => [asset.asset_id, asset]));
  const lane = plan.lanes || {};
  const chinese = assets.get(lane.chinese);
  const sourceAsset = assets.get(lane.source);
  const audio = assets.get(lane.audio_reference);
  const sourceMode = document.getElementById('source').value;
  const merge = document.getElementById('mergeExistingSubtitles').checked;

  if (sourceMode === 'sidecar') {
    if (merge) {
      if (chinese?.origin === 'sidecar') setPlannedSelectValue('chineseSidecarPath', chinese.path);
      if (sourceAsset?.origin === 'sidecar') setPlannedSelectValue('englishSidecarPath', sourceAsset.path);
    } else {
      const primary = chinese || sourceAsset;
      if (primary?.origin === 'sidecar') setPlannedSelectValue('sidecarPath', primary.path);
    }
  }
  if (sourceMode === 'embedded') {
    if (merge) {
      if (chinese?.origin === 'embedded') setPlannedSelectValue('chineseSubtitleStream', chinese.stream_index);
      if (sourceAsset?.origin === 'embedded') setPlannedSelectValue('englishSubtitleStream', sourceAsset.stream_index);
    } else {
      const primary = chinese || sourceAsset;
      if (primary?.origin === 'embedded') setPlannedSelectValue('subtitleStream', primary.stream_index);
    }
  }
  if (audio?.origin === 'audio') setPlannedSelectValue('audioStream', audio.stream_index);
}

function renderProcessingPlan(plan) {
  const section = document.getElementById('sourcePlan');
  if (!plan || !Array.isArray(plan.selected_assets)) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  const assets = new Map(plan.selected_assets.map(asset => [asset.asset_id, asset]));
  const lane = plan.lanes || {};
  const supplementary = (lane.supplementary || []).map(id => assets.get(id)).filter(Boolean);
  const execution = plan.execution || null;
  const localGpu = execution?.local_gpu || plan.compute?.local_gpu || [];
  const remoteLlm = execution?.remote_llm || plan.compute?.remote_llm || [];
  const localLabels = {subtitle_ocr: '字幕 OCR', speech_recognition: '语音识别'};
  const localPrefix = execution ? '本次本地 GPU' : '来源需重建时本地 GPU';
  const compute = [
    localGpu.length ? `${localPrefix}：${localGpu.map(item => localLabels[item] || item).join('、')}` : `${localPrefix}：不执行识别`,
    remoteLlm.length ? '远程/所选 Qwen：翻译、校对与终审' : 'Qwen：不调用'
  ];
  const rows = [
    ['中文轨', sourceAssetLabel(assets.get(lane.chinese))],
    ['源语言轨', sourceAssetLabel(assets.get(lane.source))],
    ['补充轨', supplementary.length ? supplementary.map(sourceAssetLabel).join('；') : '不使用'],
    ['时间基准', sourceAssetLabel(assets.get(lane.timing_reference))],
    ['同步音轨', sourceAssetLabel(assets.get(lane.audio_reference))],
    ['来源生成', execution?.source_cache_reused ? '复用已有来源缓存，不重新 OCR/ASR' : '按所选来源生成或读取'],
    ['处理步骤', (plan.operations || []).map(item => SOURCE_OPERATION_LABELS[item] || item).join(' → ')],
    ['计算位置', compute.join('；')]
  ];
  document.getElementById('sourcePlanRoute').textContent =
    SOURCE_ROUTE_LABELS[plan.route] || plan.route || '自动推荐';
  document.getElementById('sourcePlanRows').innerHTML = rows.map(([key, value]) =>
    `<div class="source-plan-row"><div class="source-plan-key">${escapeHtml(key)}</div><div class="source-plan-value">${escapeHtml(value)}</div></div>`
  ).join('');
  const warning = document.getElementById('sourcePlanWarning');
  const warnings = Array.isArray(plan.warnings) ? plan.warnings : [];
  warning.hidden = warnings.length === 0;
  warning.textContent = warnings.join('；');
}

function invalidateProcessingPlanPreview() {
  const analysis = getMatchingAnalysis();
  if (analysis) {
    setLastAnalysis({
      ...analysis,
      auto_source: 'auto',
      processing_plan: null
    });
  }
  renderProcessingPlan(null);
  updateRunControls();
}

function render(data) {
  state.currentData = data || {};
  state.running = Boolean(data.running);
  if (data.sidecar_subtitles) updateSidecarOptions(data.sidecar_subtitles);
  if (data.embedded_subtitles) updateEmbeddedOptions(data.embedded_subtitles);
  if (data.embedded_audio) updateAudioOptions(data.embedded_audio);
  applyProcessingPlanSelections(data.processing_plan);
  renderProcessingPlan(data.processing_plan);

  document.getElementById('videoSize').textContent = data.video_size_gb ? `${data.video_size_gb} GB` : '-';
  document.getElementById('totalCount').textContent = data.total_count ?? '未知';
  document.getElementById('completedCount').textContent = data.completed_count ?? 0;
  const outcomeLabels = {
    running: '运行中',
    success: '已完成',
    failed: '运行失败',
    interrupted: '已中断',
    stopped: '已终止',
    idle: '未运行'
  };
  document.getElementById('runningState').textContent = outcomeLabels[data.outcome] || (data.running ? '运行中' : '未运行');
  document.getElementById('stopBtn').disabled = !data.running;
  document.getElementById('sidecarState').textContent = data.has_sidecar_subtitles === undefined ? '未知' : (data.has_sidecar_subtitles ? `${(data.sidecar_subtitles || []).length} 个` : '无');
  const embeddedItems = (data.embedded_subtitles || []).filter(item => !item.error);
  document.getElementById('embeddedState').textContent = data.has_embedded_subtitles === undefined ? '未知' : (data.has_embedded_subtitles ? `${embeddedItems.length} 条轨道` : '无');
  document.getElementById('autoSourceState').textContent = sourceLabel(data.auto_source || '-');
  const qualityLabels = {pass: '通过', review: '建议复核', fail: '未通过'};
  document.getElementById('qualityState').textContent = data.running
    ? '待更新'
    : (qualityLabels[data.quality_status] || (data.quality_report_error ? '报告异常' : '未生成'));
  document.getElementById('paths').textContent = [
    data.video_path ? `主视频：${data.video_path}` : '',
    data.name_source ? `片名识别：${data.name_source}` : '',
    data.auto_source ? `自动判断：${sourceLabel(data.auto_source)}` : '',
    data.output_root ? `输出根目录：${data.output_root}${data.output_root_source === 'existing' ? '（已找到旧任务）' : ''}` : '',
    data.sidecar_subtitles ? `已有字幕：${subtitleListLabel(data.sidecar_subtitles)}` : '',
    data.embedded_subtitles ? `内封字幕：${subtitleListLabel(data.embedded_subtitles)}` : '',
    data.output_dir ? `输出：${data.output_dir}` : '',
    data.checkpoint_exists && data.checkpoint_path ? `checkpoint：${data.checkpoint_path}` : '',
    data.quality_report_exists && data.quality_report_path ? `质检报告：${data.quality_report_path}` : ''
  ].filter(Boolean).join('  |  ');

  const runMessage = document.getElementById('runMessage');
  if (data.outcome === 'failed') {
    const detail = data.error_summary || '程序异常退出，请查看下方日志。';
    const recovery = data.checkpoint_exists
      ? 'Checkpoint 已保留，可选择“继续任务”后重新运行。'
      : '请检查日志中的最后一条错误后重新运行。';
    runMessage.textContent = `运行失败：${detail} ${recovery}`;
    runMessage.className = 'status-message error';
    runMessage.hidden = false;
  } else if (data.outcome === 'interrupted') {
    runMessage.textContent = data.checkpoint_exists
      ? '上次运行未正常完成。Checkpoint 已保留，可选择“继续任务”恢复。'
      : '上次运行未正常完成，请查看日志后重新运行。';
    runMessage.className = 'status-message';
    runMessage.hidden = false;
  } else if (data.checkpoint_exists && data.checkpoint_compatible === false) {
    const checkpointModel = data.checkpoint_model || '旧模型';
    runMessage.textContent = `Checkpoint 来自 ${checkpointModel} 或旧处理策略。已保留字幕识别缓存，请使用“重新校对（保留识别）”。`;
    runMessage.className = 'status-message';
    runMessage.hidden = false;
    const reprocess = document.querySelector('input[name="runMode"][value="reprocess"]');
    const resume = document.querySelector('input[name="runMode"][value="resume"]');
    if (reprocess && resume?.checked) reprocess.checked = true;
  } else if (!data.running && data.quality_status === 'fail') {
    runMessage.textContent = `成片质检未通过。请先查看质检报告：${data.quality_report_path || ''}`;
    runMessage.className = 'status-message error';
    runMessage.hidden = false;
  } else if (!data.running && data.quality_status === 'review') {
    const summary = data.quality_summary || {};
    const reviewItems = summary.review_items ?? 0;
    const hideOverrides = summary.model_hide_overrides ?? 0;
    runMessage.textContent = `字幕已生成，质检建议复核 ${reviewItems} 项，其中模型误隐藏保护 ${hideOverrides} 项。`;
    runMessage.className = 'status-message';
    runMessage.hidden = false;
  } else {
    runMessage.textContent = '';
    runMessage.hidden = true;
  }

  const stageIndex = data.stage_index ?? -1;
  const isCompleted = stageIndex === 3;
  const isFailed = data.outcome === 'failed';
  const isInterrupted = data.outcome === 'interrupted' || data.outcome === 'stopped';
  for (let i = 0; i <= 3; i++) {
    const stepEl = document.getElementById(`step-${i}`);
    if (!stepEl) continue;
    if (stageIndex === -1) {
        stepEl.className = 'stage-step pending';
        if (i === 0) stepEl.querySelector('.step-text').textContent = '提取源';
        if (i === 1) stepEl.querySelector('.step-text').textContent = '字幕识别';
        if (i === 2) stepEl.querySelector('.step-text').textContent = '翻译与校对';
        if (i === 3) stepEl.querySelector('.step-text').textContent = '完成';
    } else {
        const currentState = i === stageIndex && isFailed
          ? 'failed'
          : (i === stageIndex && isInterrupted
            ? 'interrupted'
            : (isCompleted || i < stageIndex ? 'completed' : (i === stageIndex ? 'active' : 'pending')));
        stepEl.className = 'stage-step ' + currentState;
        if (i === stageIndex && data.stage) {
          stepEl.querySelector('.step-text').textContent = data.stage;
        } else {
          if (i === 0) stepEl.querySelector('.step-text').textContent = '提取源';
          if (i === 1) stepEl.querySelector('.step-text').textContent = '字幕识别';
          if (i === 2) stepEl.querySelector('.step-text').textContent = '翻译与校对';
          if (i === 3) stepEl.querySelector('.step-text').textContent = '完成';
        }
    }
  }

  const rows = (data.preview || []).map(item => {
    const time = `${item.start ?? ''} -> ${item.end ?? ''}`;
    return `<tr><td>${escapeHtml(item.id)}</td><td>${escapeHtml(time)}</td><td>${escapeHtml(previewEnglishText(item))}</td><td>${escapeHtml(item.zh || '')}</td></tr>`;
  }).join('');
  document.getElementById('preview').innerHTML = rows || '<tr><td colspan="4" class="muted">暂无 checkpoint 内容</td></tr>';
  renderQualityReview(data);
  updateWorkflowVisibility();
  updateRunControls(data);
}

function updateSidecarOptions(items) {
  const rows = (items || []).map(item => {
    const role = {dialogue: '完整对白', sdh: 'SDH', forced: '强制字幕', commentary: '评论', unknown: '角色待确认'}[item.role] || item.role || '';
    const label = `${item.name || item.path} (${item.language || 'und'} · ${role || '文本'} · ${item.size_kb ?? '?'} KB)`;
    return `<option value="${escapeHtml(item.path || '')}">${escapeHtml(label)}</option>`;
  }).join('');
  fillSubtitleSelect('sidecarPath', '自动选择', rows, null);
  fillSubtitleSelect('chineseSidecarPath', '自动选择', rows, value => {
    const item = (items || []).find(entry => entry.path === value);
    return languageHint(`${item?.language || ''} ${item?.name || item?.path || ''}`) === 'zh';
  });
  fillSubtitleSelect('englishSidecarPath', '可选：自动选择', rows, value => {
    const item = (items || []).find(entry => entry.path === value);
    return languageHint(`${item?.language || ''} ${item?.name || item?.path || ''}`) === 'en';
  });
}

function updateEmbeddedOptions(items) {
  const valid = (items || []).filter(item => !item.error);
  const rows = valid.map(item => {
    const supported = item.supported !== false;
    const kind = supported
      ? (item.is_image ? '图像' : (item.is_text ? '文本' : '字幕'))
      : `不支持编码 ${item.codec || ''}`;
    const role = {dialogue: '完整对白', sdh: 'SDH', forced: '强制字幕', commentary: '评论', unknown: '角色待确认'}[item.role] || item.role || '';
    const script = {simplified: '简体', traditional: '繁体', unknown: ''}[item.script] || '';
    const label = `${item.label || item.index} | ${kind} | ${item.language || 'und'} | ${script} ${role}`.trim();
    return `<option value="${escapeHtml(item.index)}"${supported ? '' : ' disabled'}>${escapeHtml(label)}</option>`;
  }).join('');
  fillSubtitleSelect('subtitleStream', '自动选择', rows, null);
  fillSubtitleSelect('chineseSubtitleStream', '自动选择', rows, value => {
    const item = valid.find(entry => String(entry.index) === String(value));
    return languageHint(`${item?.language || ''} ${item?.title || ''}`) === 'zh';
  });
  fillSubtitleSelect('englishSubtitleStream', '可选：自动选择', rows, value => {
    const item = valid.find(entry => String(entry.index) === String(value));
    return languageHint(`${item?.language || ''} ${item?.title || ''}`) === 'en';
  });
}

function fillSubtitleSelect(id, placeholder, rows, prefer) {
  const select = document.getElementById(id);
  if (!select) return;
  const current = select.value;
  select.innerHTML = `<option value="">${escapeHtml(placeholder)}</option>` + rows;
  const options = [...select.options];
  if (current && options.some(option => option.value === current && !option.disabled)) {
    select.value = current;
  } else if (prefer) {
    const preferred = options.find(
      option => option.value && !option.disabled && prefer(option.value)
    );
    if (preferred) select.value = preferred.value;
  }
}

function languageHint(text) {
  const lower = String(text || '').toLowerCase();
  const parts = lower.split(/[^0-9a-zA-Z\u4e00-\u9fff]+/).filter(Boolean);
  if (['zh-cn', 'zh-hans', 'zh-tw', 'zh-hant'].some(token => lower.includes(token))) return 'zh';
  if (parts.some(part => ['zh', 'zho', 'chi', 'chs', 'cht', 'cmn', 'cn', 'sc', 'tc', 'zh-cn', 'zh-hans', 'zh-tw', 'zh-hant'].includes(part))) return 'zh';
  if (/[中文简繁]/.test(lower) || lower.includes('chinese') || lower.includes('simplified') || lower.includes('traditional')) return 'zh';
  if (parts.some(part => ['en', 'eng', 'english'].includes(part)) || lower.includes('english')) return 'en';
  return '';
}

function updateAudioOptions(items) {
  const select = document.getElementById('audioStream');
  const current = select.value;
  const rows = (items || []).filter(item => !item.error).map(item => {
    let scoreInfo = item.language ? `[${item.language}]` : "";
    const selectable = item.role !== 'commentary';
    const roleInfo = selectable ? '' : ' · 评论/解说（不用于对白）';
    const label = `0:${item.index} ${item.codec} ${scoreInfo} ${item.title || ''}${roleInfo}`.trim();
    return `<option value="${escapeHtml(item.index)}"${selectable ? '' : ' disabled'}>${escapeHtml(label)}</option>`;
  }).join('');
  select.innerHTML = '<option value="">自动选择</option>' + rows;
  if ([...select.options].some(option => option.value === current && !option.disabled)) {
    select.value = current;
  }
}

function sourceLabel(value) {
  return ({auto: '自动', sidecar: '已有字幕', srt: '已有字幕', embedded: '内封字幕', audio: '音频识别'})[value] || value;
}

function subtitleListLabel(items) {
  if (!items || !items.length) return '无';
  const errors = items.filter(item => item.error).map(item => item.error);
  const valid = items.filter(item => !item.error);
  if (valid.length) return valid.slice(0, 4).map(item => item.name || item.label || item.path || item.index).join('；') + (valid.length > 4 ? ` 等 ${valid.length} 个` : '');
  return errors.length ? `读取失败：${errors[0]}` : '无';
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

const INPUT_IDS = [
  ...SOURCE_SELECTION_IDS,
  'path', 'outputRoot', 'seriesName', 'movieName', 'source',
  'mergeExistingSubtitles', 'subtitleSync',
  'source_language', 'asrLanguage', 'subtitleOcrLang', 'llmModel',
  'batchSize', 'contextLines', 'autonomousReview', 'maxWords', 'maxChars', 'maxDuration',
  'subtitleStyleProfile', 'subtitleFontName', 'subtitleFontScale', 'subtitleFontFile'
];

const REMOTE_STORAGE_KEY = 'sub_remote_config';

function updateRemoteAuthVisibility() {
  const authMethod = document.getElementById('remoteAuthMethod').value;
  document.getElementById('remoteKeyOptions').hidden = authMethod !== 'key';
  document.getElementById('remotePasswordOptions').hidden = authMethod !== 'password';
}

function openRemoteModal() {
  document.getElementById('remoteError').textContent = '';
  document.getElementById('remoteModal').style.display = 'flex';
  let localConf = {};
  try { localConf = JSON.parse(localStorage.getItem(REMOTE_STORAGE_KEY) || '{}'); } catch (e) {}
  const conf = {...(state.serverSettings.remote || {}), ...localConf};
  if (!localConf.password && state.serverSettings.remote?.password) {
    conf.password = state.serverSettings.remote.password;
  }
  if (conf.host) document.getElementById('remoteHost').value = conf.host;
  if (conf.port) document.getElementById('remotePort').value = conf.port;
  if (conf.user) document.getElementById('remoteUser').value = conf.user;
  if (conf.name) document.getElementById('remoteName').value = conf.name;
  document.getElementById('remoteAuthMethod').value = conf.auth_method || (conf.password ? 'password' : 'key');
  document.getElementById('remoteKeyPath').value = conf.key_filename || '';
  document.getElementById('remotePass').value = conf.password || '';
  document.getElementById('remoteRememberPassword').checked = Boolean(conf.remember_password || conf.password);
  updateRemoteAuthVisibility();
  document.getElementById('remoteHost').focus();
}

function closeRemoteModal() {
  document.getElementById('remoteModal').style.display = 'none';
  document.getElementById('remoteBtn')?.focus();
}

async function connectRemoteServer() {
  const btn = document.getElementById('remoteConnectBtn');
  btn.textContent = '连接中...';
  btn.disabled = true;
  document.getElementById('remoteError').textContent = '';
  
  const rememberPassword = document.getElementById('remoteRememberPassword').checked;
  const payload = {
    host: document.getElementById('remoteHost').value,
    port: document.getElementById('remotePort').value,
    user: document.getElementById('remoteUser').value,
    password: document.getElementById('remotePass').value,
    name: document.getElementById('remoteName').value || 'Remote',
    auth_method: document.getElementById('remoteAuthMethod').value,
    key_filename: document.getElementById('remoteKeyPath').value,
    remember_password: rememberPassword
  };
  
  try {
    const res = await fetch('/api/remote/connect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (res.ok && data.status === 'connected') {
      localStorage.setItem(REMOTE_STORAGE_KEY, JSON.stringify({
        host: payload.host,
        port: payload.port,
        user: payload.user,
        name: payload.name,
        auth_method: payload.auth_method,
        key_filename: payload.key_filename,
        remember_password: payload.auth_method === 'password' && rememberPassword
      }));
      state.serverSettings.remote = {
        host: payload.host,
        port: payload.port,
        user: payload.user,
        name: payload.name,
        auth_method: payload.auth_method,
        key_filename: payload.key_filename,
        remember_password: payload.auth_method === 'password' && rememberPassword,
        password: payload.auth_method === 'password' && rememberPassword ? payload.password : ''
      };
      if (!(payload.auth_method === 'password' && rememberPassword)) {
        document.getElementById('remotePass').value = '';
      }
      state.remoteStatus = {
        checked: true,
        connected: true,
        status: data.status || 'connected',
        name: data.name || payload.name,
        error: '',
        persistentReconnect: Boolean(data.persistent_reconnect)
      };
      updateLlmModels(data.models, data.name || payload.name);
      if (data.warning) {
        document.getElementById('remoteError').textContent = data.warning;
      }
      if (data.models && data.models.length > 0) {
        closeRemoteModal();
      } else {
        document.getElementById('remoteError').textContent = 'SSH 已连接，但远程 Ollama 没有返回可用模型。';
      }
    } else {
      state.remoteStatus = {checked: true, connected: false, status: 'stopped', name: '', error: data.error || '连接失败', persistentReconnect: false};
      document.getElementById('remoteError').textContent = data.error || '连接失败';
    }
  } catch(e) {
    state.remoteStatus = {checked: true, connected: false, status: 'stopped', name: '', error: String(e), persistentReconnect: false};
    document.getElementById('remoteError').textContent = String(e);
  } finally {
    btn.textContent = '连接';
    btn.disabled = false;
    updateRunControls();
  }
}

function updateLlmModels(remoteModels, remoteName) {
  const select = document.getElementById('llmModel');
  let current = select.value;
  try {
    const saved = JSON.parse(localStorage.getItem(FORM_STORAGE_KEY) || '{}');
    if (saved.llmModel) current = saved.llmModel;
  } catch (e) {}
  const safeRemoteName = String(remoteName || 'Remote').replaceAll(':', ' ').replace(/\s+/g, ' ').trim() || 'Remote';
  select.replaceChildren(new Option('[Local] qwen3:14b', 'qwen3:14b'));
  if (remoteModels && remoteModels.length) {
    remoteModels.forEach(m => {
      const modelName = String(m || '').trim();
      if (!modelName) return;
      select.add(new Option(`[Remote: ${safeRemoteName}] ${modelName}`, `remote:${safeRemoteName}:${modelName}`));
    });
  }
  if ([...select.options].some(o => o.value === current)) {
    select.value = current;
  } else if (current) {
    restoreModelSelection(select, current);
  }
  updateRunControls();
}

async function checkRemoteStatus() {
  try {
    const res = await fetch('/api/remote/status', {method: 'POST'});
    const data = await res.json();
    state.remoteStatus = {
      checked: true,
      connected: Boolean(res.ok && data.connected),
      status: data.status || (data.connected ? 'connected' : 'stopped'),
      name: data.name || '',
      error: data.error || '',
      persistentReconnect: Boolean(data.persistent_reconnect)
    };
    if (state.remoteStatus.connected && data.models) {
      updateLlmModels(data.models, data.name);
    }
  } catch(e) {
    state.remoteStatus = {checked: true, connected: false, status: 'stopped', name: '', error: String(e), persistentReconnect: false};
  } finally {
    updateRunControls();
  }
}

function saveFormState() {
  let analysisInvalidated = false;
  if (state.lastAnalysis && !analysisMatchesCurrentPath(state.lastAnalysis)) {
    clearSourceSelections();
    clearLastAnalysis();
    analysisInvalidated = true;
  }
  const data = {};
  INPUT_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) data[id] = el.type === 'checkbox' ? el.checked : el.value;
  });
  localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(data));
  if (!state.restoringSettings) scheduleServerFormSave(data);
  if (analysisInvalidated) {
    resetAnalysisPresentation('输入已变化，请重新分析视频。');
  } else {
    updateRunControls();
  }
}

function scheduleServerFormSave(data) {
  clearTimeout(settingsSaveTimer);
  settingsSaveTimer = setTimeout(() => {
    api('/api/settings/save', {form: data})
      .then(saved => { state.serverSettings = saved; })
      .catch(() => {});
  }, 300);
}

async function loadServerSettings() {
  try {
    const saved = await api('/api/settings/load');
    state.serverSettings = saved;
    return saved;
  } catch (e) {
    return {form: {}, remote: {}};
  }
}

function restoreFormState(serverForm = {}) {
  state.restoringSettings = true;
  try {
    const local = JSON.parse(localStorage.getItem(FORM_STORAGE_KEY) || '{}');
    const data = {...(local || {}), ...(serverForm || {})};
    if (data) {
      INPUT_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (el && data[id] !== undefined) {
          if (el.type === 'checkbox') el.checked = Boolean(data[id]);
          else if (id === 'llmModel') restoreModelSelection(el, data[id]);
          else el.value = data[id];
        }
      });
      migrateOutputRootDefault();
      migrateRecognitionLanguageState();
      const analysis = getMatchingAnalysis();
      if (analysis) {
        render(analysis);
        SOURCE_SELECTION_IDS.forEach(id => {
          const el = document.getElementById(id);
          if (el && data[id] !== undefined && [...el.options].some(option => option.value === String(data[id]))) {
            el.value = String(data[id]);
          }
        });
      } else {
        resetAnalysisPresentation('已恢复上次输入。请重新分析视频后再运行。');
      }
      if (data.seriesName && data.movieName) {
        refreshStatus().catch(() => {});
      }
    }
  } catch (e) {
  } finally {
    state.selectedPath = document.getElementById('path').value;
    state.restoringSettings = false;
    updateRunControls();
  }
}

async function migrateLegacyRemotePassword() {
  let legacy = {};
  try { legacy = JSON.parse(localStorage.getItem(REMOTE_STORAGE_KEY) || '{}'); } catch (e) {}
  if (!legacy.password) return;
  try {
    const authMethod = legacy.auth_method || 'password';
    const saved = await api('/api/settings/save', {
      remote: {...legacy, auth_method: authMethod, remember_password: true}
    });
    state.serverSettings = saved;
    delete legacy.password;
    legacy.auth_method = authMethod;
    legacy.remember_password = true;
    localStorage.setItem(REMOTE_STORAGE_KEY, JSON.stringify(legacy));
  } catch (e) {}
}

function migrateOutputRootDefault() {
  const outputRoot = document.getElementById('outputRoot');
  let changed = false;
  if (
    !normalizePathForCompare(CONFIGURED_OUTPUT_ROOT)
    && normalizePathForCompare(outputRoot.value) === normalizePathForCompare(LEGACY_DEFAULT_OUTPUT_ROOT)
  ) {
    outputRoot.value = '';
    changed = true;
  }
  if (Number(localStorage.getItem(OUTPUT_ROOT_MIGRATION_VERSION_KEY) || 0) < 2) {
    localStorage.setItem(OUTPUT_ROOT_MIGRATION_VERSION_KEY, '2');
    changed = true;
  }
  if (changed) saveFormState();
}

function migrateRecognitionLanguageState() {
  const source = document.getElementById('source_language');
  const asr = document.getElementById('asrLanguage');
  if (!source || !asr) return;
  if (source.value && source.value !== 'auto' && source.value !== 'en' && asr.value === 'en') {
    asr.value = 'source';
  }
}

function migrateDisplayLimitDefaults() {
  if (Number(localStorage.getItem(DISPLAY_LIMITS_VERSION_KEY) || 0) >= 3) return;
  const maxWords = document.getElementById('maxWords');
  const maxChars = document.getElementById('maxChars');
  const maxDuration = document.getElementById('maxDuration');
  if (maxWords.value === '14') maxWords.value = '12';
  if (maxChars.value === '82' || maxChars.value === '56') maxChars.value = '42';
  if (maxDuration.value === '6') maxDuration.value = '5.5';
  localStorage.setItem(DISPLAY_LIMITS_VERSION_KEY, '3');
  saveFormState();
}

document.addEventListener('DOMContentLoaded', async () => {
  const savedSettings = await loadServerSettings();
  restoreFormState(savedSettings.form || {});
  await migrateLegacyRemotePassword();
  migrateOutputRootDefault();
  migrateRecognitionLanguageState();
  migrateDisplayLimitDefaults();
  const source = document.getElementById('source');
  document.getElementById('path').addEventListener('input', event => {
    setSelectedPath(event.target.value);
  });
  document.getElementById('path').addEventListener('change', event => {
    setSelectedPath(event.target.value);
  });
  source.addEventListener('change', () => {
    invalidateProcessingPlanPreview();
    updateWorkflowVisibility();
  });
  document.getElementById('mergeExistingSubtitles').addEventListener('change', () => {
    invalidateProcessingPlanPreview();
    updateWorkflowVisibility();
  });
  document.getElementById('subtitleSync').addEventListener('change', () => {
    invalidateProcessingPlanPreview();
    updateWorkflowVisibility();
  });
  SOURCE_SELECTION_IDS.forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      invalidateProcessingPlanPreview();
      updateWorkflowVisibility();
    });
  });
  document.getElementById('source_language')?.addEventListener('change', () => {
    migrateRecognitionLanguageState();
    invalidateProcessingPlanPreview();
    updateWorkflowVisibility();
    saveFormState();
  });
  updateWorkflowVisibility();
  checkRemoteStatus();
  refreshQueue();
});
document.addEventListener('input', saveFormState);
document.addEventListener('change', saveFormState);
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  if (document.getElementById('reviewModal').style.display === 'flex') closeReviewModal();
  if (document.getElementById('remoteModal').style.display === 'flex') closeRemoteModal();
});

setInterval(() => { if (state.pid) refreshStatus().catch(() => {}); }, 5000);
setInterval(() => { refreshQueue().catch(() => {}); }, 5000);
setInterval(() => { checkRemoteStatus().catch(() => {}); }, 10000);
</script>
</body>
</html>""".replace(
    "__INITIAL_OUTPUT_ROOT_JSON__",
    json.dumps(str(CONFIGURED_OUTPUT_ROOT or ""), ensure_ascii=False),
).replace(
    "__LEGACY_DEFAULT_OUTPUT_ROOT_JSON__",
    json.dumps(str(LEGACY_DEFAULT_OUTPUT_ROOT), ensure_ascii=False),
)


LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_loopback_host(value: str) -> bool:
    raw = str(value or "").strip().casefold()
    if raw in LOOPBACK_HOSTS:
        return True
    try:
        hostname = urlparse(f"//{raw}").hostname
    except ValueError:
        return False
    return bool(hostname and hostname.casefold() in LOOPBACK_HOSTS)


class RequestValidationError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


class Handler(BaseHTTPRequestHandler):
    def reject_nonlocal_host(self) -> bool:
        if is_loopback_host(self.headers.get("Host", "")):
            return False
        self.send_json({"error": "This interface only accepts localhost requests"}, status=403)
        return True

    def do_GET(self) -> None:
        if self.reject_nonlocal_host():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(html_page(), "text/html; charset=utf-8")
        elif parsed.path == "/api/health":
            self.send_json(
                {
                    "app": "bilingual-subtitle-pipeline",
                    "status": "ok",
                    "source_sha256": FRONTEND_SOURCE_SHA256,
                }
            )
        elif parsed.path.startswith("/api/review/media/"):
            token = parsed.path.rsplit("/", 1)[-1]
            media_path = review_media_path(token)
            if media_path is None:
                self.send_json({"error": "Review media token is invalid or expired"}, status=404)
                return
            self.send_media(media_path)
        else:
            self.send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if self.reject_nonlocal_host():
            return
        try:
            if self.path == "/api/select-file":
                self.send_json({"path": choose_file()})
            elif self.path == "/api/select-folder":
                self.send_json({"path": choose_folder()})
            elif self.path == "/api/select-font":
                self.send_json({"path": choose_font()})
            elif self.path == "/api/settings/load":
                self.send_json(settings_store.load())
            elif self.path == "/api/settings/save":
                body = self.read_json_body()
                if "form" in body:
                    settings_store.save_form(body["form"])
                if "remote" in body:
                    settings_store.save_remote(body["remote"])
                self.send_json(settings_store.load())
            elif self.path == "/api/analyze":
                self.send_json(analyze_input(self.read_json_body()))
            elif self.path == "/api/start":
                self.send_json(start_processing(self.read_json_body()))
            elif self.path == "/api/stop":
                self.send_json(stop_processing(self.read_json_body()))
            elif self.path == "/api/status":
                self.send_json(run_status(self.read_json_body()))
            elif self.path == "/api/checkpoint/update":
                self.send_json(update_checkpoint_segment(self.read_json_body()))
            elif self.path == "/api/review/evidence":
                self.send_json(build_review_evidence(self.read_json_body()))
            elif self.path == "/api/queue/discover":
                body = self.read_json_body()
                episodes = discover_episode_files(str(body.get("path") or ""))
                queued_paths = {
                    os.path.normcase(str(item.get("path") or ""))
                    for item in queue_store.snapshot().get("items", [])
                    if item.get("status") in {"queued", "running"}
                }
                for episode in episodes:
                    episode["already_queued"] = (
                        os.path.normcase(str(episode["path"])) in queued_paths
                    )
                self.send_json({"episodes": episodes, "count": len(episodes)})
            elif self.path == "/api/queue/add":
                body = self.read_json_body()
                paths = body.get("paths")
                base_payload = body.get("payload")
                if not isinstance(paths, list) or not paths:
                    raise RequestValidationError("至少选择一个剧集视频")
                if not isinstance(base_payload, dict):
                    raise RequestValidationError("队列处理参数无效")
                queue_store.enqueue([str(path) for path in paths], base_payload)
                if body.get("start"):
                    queue_store.set_paused(False)
                    ensure_queue_worker()
                self.send_json(queue_store.snapshot())
            elif self.path == "/api/queue/list":
                snapshot = queue_store.snapshot()
                summary = snapshot.get("summary") or {}
                if not snapshot.get("paused") and (
                    int(summary.get("queued") or 0) > 0
                    or int(summary.get("running") or 0) > 0
                ):
                    ensure_queue_worker()
                    snapshot = queue_store.snapshot()
                self.send_json(snapshot)
            elif self.path == "/api/queue/action":
                body = self.read_json_body()
                action = str(body.get("action") or "")
                queue_store.action(
                    action,
                    str(body.get("id") or ""),
                    int(body.get("direction") or 0),
                )
                if action in {"resume", "retry"}:
                    if action == "resume":
                        queue_store.set_paused(False)
                    ensure_queue_worker()
                self.send_json(queue_store.snapshot())
            elif self.path == "/api/remote/connect":
                if not tunnel_manager:
                    self.send_json({"error": "paramiko not installed or ssh_tunnel not found"}, status=500)
                    return
                body = self.read_json_body()
                success = tunnel_manager.connect(
                    body.get("host", ""),
                    int(body.get("port", 22)),
                    body.get("user", ""),
                    body.get("password", ""),
                    body.get("name", "remote"),
                    body.get("auth_method", ""),
                    body.get("key_filename", ""),
                )
                if success:
                    models = tunnel_manager.fetch_models()
                    saved = settings_store.save_remote(body)
                    saved_remote = saved.get("remote") if isinstance(saved, dict) else {}
                    connection_status = tunnel_manager.status()
                    connection_status.update(
                        {
                            "status": "connected",
                            "models": models,
                            "persistent_reconnect": False,
                            "connection_owner": "frontend",
                        }
                    )
                    if isinstance(saved_remote, dict) and remote_settings_support_restart(saved_remote):
                        try:
                            bridge_connection = ensure_remote_bridge(
                                wait_for_connection=True,
                                timeout=30,
                                reload_config=True,
                            )
                            bridge_models = bridge_connection.get("models") or models
                            bridge_connection.update(
                                {
                                    "status": "connected",
                                    "models": bridge_models,
                                    "persistent_reconnect": True,
                                    "connection_owner": "bridge",
                                }
                            )
                            connection_status = bridge_connection
                            tunnel_manager.disconnect()
                        except Exception as exc:
                            connection_status["warning"] = (
                                "SSH 已验证，但独立自动重连桥接仍在启动：" + str(exc)
                            )
                    elif str(body.get("auth_method") or "").casefold() == "password":
                        connection_status["warning"] = (
                            "未保存密码；当前连接可用，但关闭网页后不能自动恢复。"
                        )
                    self.send_json(connection_status)
                else:
                    self.send_json({"error": tunnel_manager.last_error}, status=400)
            elif self.path == "/api/remote/status":
                self.send_json(persistent_remote_status())
            else:
                self.send_json({"error": "Not found"}, status=404)
        except RequestValidationError as exc:
            self.send_json({"error": str(exc)}, status=exc.status)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 1_048_576:
            raise RequestValidationError("Request body is too large", status=413)
        content_type = self.headers.get("Content-Type", "").partition(";")[0].strip().casefold()
        if content_type != "application/json":
            raise RequestValidationError("Content-Type must be application/json", status=415)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestValidationError("Request body must contain valid JSON") from exc
        if not isinstance(payload, dict):
            raise RequestValidationError("Request JSON must be an object")
        return payload

    def send_text(self, content: str, content_type: str) -> None:
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_media(self, path: Path) -> None:
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = 200
        range_header = str(self.headers.get("Range") or "").strip()
        if range_header.startswith("bytes="):
            raw_start, _separator, raw_end = range_header[6:].partition("-")
            try:
                start = int(raw_start) if raw_start else 0
                end = int(raw_end) if raw_end else size - 1
            except ValueError:
                self.send_error(416)
                return
            if start < 0 or start >= size or end < start:
                self.send_error(416)
                return
            end = min(end, size - 1)
            status = 206

        content_length = end - start + 1
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.casefold() == ".wav":
            content_type = "audio/wav"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "private, max-age=3600")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        remaining = content_length
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for subtitle OCR/ASR translation.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not is_loopback_host(args.host):
        parser.error("--host must be localhost, 127.0.0.1, or ::1")

    server_class = IPv6ThreadingHTTPServer if args.host == "::1" else ThreadingHTTPServer
    server = server_class((args.host, args.port), Handler)
    print(f"Subtitle frontend running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
