import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ass_styles import (
    STYLE_PROFILE_NAMES,
    ass_style_header,
    probe_video_duration_seconds,
    probe_video_play_resolution,
)
from font_delivery import font_family_name, package_ass_with_font, validate_font_path
from llm_policy import (
    generation_profile,
    indexed_subtitle_schema,
    indexed_terminology_schema,
    movie_name_schema,
    movie_style_schema,
)
from output_paths import OUTPUT_DIRECTORY_NAME, resolve_output_root, suggested_output_root
from pipeline_policy import (
    AUTONOMOUS_REVIEW_POLICY_VERSION,
    FINAL_QA_POLICY_VERSION,
    PROCESSING_POLICY_VERSION,
    STYLE_GUIDE_POLICY_VERSION,
    TERMINOLOGY_POLICY_VERSION,
    TERMINOLOGY_REVIEW_POLICY_VERSION,
)
from scene_timing import align_subtitles_to_scene_cuts, detect_scene_cuts
from subtitle_delivery import validate_ass_rendering, write_delivery_srts
from subtitle_sync import (
    SUBTITLE_SYNC_MODES,
    SYNC_POLICY_VERSION,
    synchronize_subtitle_segments,
)
from subtitle_quality import (
    SubtitleLayoutPolicy,
    build_layout_policy,
    build_quality_report,
    write_quality_report,
)
from subtitle_sources import (
    SubtitleAsset,
    assets_from_segments,
    attach_asset_to_segments,
    build_audio_asset,
    build_embedded_asset,
    build_processing_plan,
    build_sidecar_asset,
    is_authored_source,
    model_may_hide_segment,
    normalize_language,
    source_manifest,
)

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"jieba\._compat",
)
jieba = importlib.import_module("jieba")


Segment = Dict[str, Any]

jieba.setLogLevel(logging.WARNING)


@dataclass(frozen=True)
class ExistingSubtitleTracks:
    english: Optional[List[Segment]]
    chinese: List[Segment]
    source_label: str
    english_asset: Optional[SubtitleAsset] = None
    chinese_asset: Optional[SubtitleAsset] = None
    supplementary_assets: Tuple[SubtitleAsset, ...] = ()


VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m2ts", ".ts", ".mov", ".wmv"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}
UNKNOWN_SOURCE_LANGUAGE_TAGS = {"", "auto", "source", "cached source", "subtitle", "embedded subtitle"}
SOURCE_CACHE_VERSION = 11
COMPATIBLE_SYNC_POLICY_VERSIONS = (5,)
TARGET_CHINESE_CPS = 9.0
TARGET_ENGLISH_CPS = 20.0
MIN_SUBTITLE_DURATION = 20 / 24
MAX_READING_EXTENSION = 0.5
MIN_SUBTITLE_GAP = 2 / 24
MIN_OVERLAP_RETAIN_DURATION = 0.4
CHINESE_CHARS_PER_LINE = 16
ENGLISH_CHARS_PER_LINE = 42
BILINGUAL_LINES_PER_LANGUAGE = 1


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        for attempt in range(20):
            try:
                tmp_path.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.5))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def load_cached_source_segments(path: Path) -> List[Segment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("cached source segments must be a JSON list")

    segments: List[Segment] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"cached source segment {index} is not an object")
        if "start" not in item or "end" not in item or not any(key in item for key in ("text", "en", "zh")):
            raise ValueError(f"cached source segment {index} is missing start/end/text/en/zh")
        start = float(item["start"])
        end = float(item["end"])
        text = clean_subtitle_text(str(item.get("text") or item.get("en") or item.get("zh") or ""))
        if end <= start:
            raise ValueError(f"cached source segment {index} has invalid timing")
        if not text:
            continue
        normalized = {**item, "id": len(segments), "start": start, "end": end, "text": text}
        if normalized.get("en") is not None and should_clean_english_track(normalized):
            normalized["en"] = clean_english_track_text(str(normalized["en"]))
        if normalized.get("zh") is not None:
            normalized["zh"] = to_simplified_text(clean_subtitle_text(str(normalized["zh"])))
        segments.append(normalized)

    if not segments:
        raise ValueError("cached source segments are empty")
    fill_missing_source_languages(segments)
    return segments


def infer_source_language_from_segments(segments: List[Segment], fallback: str) -> str:
    if fallback and fallback != "auto":
        return fallback
    dominant = dominant_source_language(segments)
    if dominant:
        return dominant
    for segment in segments:
        language = segment.get("source_language")
        if language:
            return str(language)
    return "cached source"


def source_cache_uses_current_timing_policy(segments: List[Segment]) -> bool:
    has_existing_subtitle_timing = any(
        str(segment.get("timing_origin") or "").casefold() == "existing_subtitle"
        or "existing chinese" in str(segment.get("source_language") or "").casefold()
        for segment in segments
    )
    if not has_existing_subtitle_timing:
        return True
    return all(int(segment.get("source_cache_version") or 0) >= SOURCE_CACHE_VERSION for segment in segments)


def source_path_identity(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    identity: Dict[str, Any] = {"path": str(resolved)}
    if resolved.is_dir():
        identity["kind"] = "directory"
        return identity
    try:
        stat = resolved.stat()
        identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    except OSError:
        identity["missing"] = True
    return identity


def build_source_request_fingerprint(
    video_path: Path,
    input_path: Path,
    args: Any,
    sidecar_path: Optional[Path] = None,
    chinese_sidecar_path: Optional[Path] = None,
    english_sidecar_path: Optional[Path] = None,
    sync_policy_version: Optional[int] = None,
    processing_plan_fingerprint: Optional[str] = None,
) -> str:
    source_mode = str(getattr(args, "source", "auto") or "auto")
    discovered_sidecars: List[Dict[str, Any]] = []
    if source_mode in {"auto", "sidecar", "srt"}:
        for candidate in likely_sidecar_subtitles(video_path, input_path):
            identity = source_path_identity(candidate)
            if identity is not None:
                discovered_sidecars.append(identity)

    descriptor = {
        "version": 2,
        "video": source_path_identity(video_path),
        "input": source_path_identity(input_path),
        "source": source_mode,
        "merge_existing_subtitles": str(getattr(args, "merge_existing_subtitles", "yes")),
        "sidecar": source_path_identity(sidecar_path),
        "chinese_sidecar": source_path_identity(chinese_sidecar_path),
        "english_sidecar": source_path_identity(english_sidecar_path),
        "discovered_sidecars": discovered_sidecars,
        "subtitle_stream": getattr(args, "subtitle_stream", None),
        "chinese_subtitle_stream": getattr(args, "chinese_subtitle_stream", None),
        "english_subtitle_stream": getattr(args, "english_subtitle_stream", None),
        "audio_stream": getattr(args, "audio_stream", None),
        "subtitle_sync": str(getattr(args, "subtitle_sync", "auto") or "auto"),
        "subtitle_sync_policy_version": (
            SYNC_POLICY_VERSION
            if sync_policy_version is None
            else sync_policy_version
        ),
        "source_language": str(getattr(args, "source_language", "auto") or "auto"),
        "asr_language": str(getattr(args, "asr_language", "source") or "source"),
        "subtitle_ocr_lang": str(getattr(args, "subtitle_ocr_lang", "auto") or "auto"),
        "device": str(getattr(args, "device", "gpu:0") or "gpu:0"),
        "ocr_scale": float(getattr(args, "ocr_scale", 2.0) or 2.0),
        "crop_pad": int(getattr(args, "crop_pad", 8) or 8),
        "fast_ocr": bool(getattr(args, "fast_ocr", False)),
    }
    english_text_authority = str(
        getattr(args, "english_subtitle_text_authority", "authored")
        or "authored"
    )
    chinese_text_authority = str(
        getattr(args, "chinese_subtitle_text_authority", "authored")
        or "authored"
    )
    if english_text_authority != "authored":
        descriptor["english_subtitle_text_authority"] = english_text_authority
    if chinese_text_authority != "authored":
        descriptor["chinese_subtitle_text_authority"] = chinese_text_authority
    if processing_plan_fingerprint:
        descriptor["processing_plan_fingerprint"] = processing_plan_fingerprint
    serialized = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def source_cache_matches_request(
    segments: List[Segment],
    request_fingerprint: str,
    allow_legacy: bool = False,
) -> bool:
    cached_fingerprints = {
        str(segment.get("source_request_fingerprint") or "")
        for segment in segments
    }
    cached_fingerprints.discard("")
    if not cached_fingerprints:
        return allow_legacy
    return cached_fingerprints == {request_fingerprint}


def source_request_audio_stream_variants(
    requested_audio_stream: Optional[int],
    effective_audio_stream: Optional[int],
    subtitle_sync_mode: str,
    sync_report_path: Path,
) -> set[Optional[int]]:
    variants = {requested_audio_stream}
    if subtitle_sync_mode == "off":
        variants.add(None)
        return variants
    if effective_audio_stream is None or not sync_report_path.exists():
        return variants
    try:
        report = json.loads(sync_report_path.read_text(encoding="utf-8"))
        report_audio_stream = int(report.get("audio_stream"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return variants
    if report_audio_stream == int(effective_audio_stream):
        variants.update({None, effective_audio_stream})
    return variants


def dominant_source_language(segments: List[Segment]) -> str:
    counts: Dict[str, int] = {}
    for segment in segments:
        language = normalize_language_tag(str(segment.get("source_language") or ""))
        if language in UNKNOWN_SOURCE_LANGUAGE_TAGS:
            continue
        counts[language] = counts.get(language, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def fill_missing_source_languages(segments: List[Segment]) -> None:
    language = dominant_source_language(segments)
    if language in UNKNOWN_SOURCE_LANGUAGE_TAGS:
        return
    for segment in segments:
        current = normalize_language_tag(str(segment.get("source_language") or ""))
        if current in UNKNOWN_SOURCE_LANGUAGE_TAGS:
            segment["source_language"] = language


def normalize_language_tag(language: str) -> str:
    value = (language or "").strip().lower()
    if not value or value in {"auto", "source", "cached source"}:
        return value
    if value in {"en", "eng", "english"} or "english" in value:
        return "en"
    if value in {"ja", "jp", "jpn", "japanese"} or "japanese" in value:
        return "ja"
    if value in {"ko", "kor", "korean"} or "korean" in value:
        return "ko"
    if value in {"zh", "zho", "chi", "chs", "cht", "cmn", "cn", "chinese"} or "chinese" in value:
        return "zh"
    if value in {"fr", "fra", "fre", "french"} or "french" in value:
        return "fr"
    if value in {"de", "deu", "ger", "german"} or "german" in value:
        return "de"
    if value in {"es", "spa", "spanish"} or "spanish" in value:
        return "es"
    return value.split("-", 1)[0]


def source_cache_matches_requested_language(
    segments: List[Segment],
    source_language: str,
    asr_language: str,
) -> bool:
    desired = normalize_language_tag(source_language)
    if not desired or desired == "auto":
        desired = normalize_language_tag(resolve_asr_language(asr_language, source_language))
    if not desired or desired == "auto":
        return True

    cached = normalize_language_tag(infer_source_language_from_segments(segments, ""))
    return bool(cached and cached == desired)


def get_ffmpeg_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg or imageio-ffmpeg.") from exc


def find_main_video_in_folder(folder_path: Path) -> Path:
    if not folder_path.is_dir():
        raise NotADirectoryError(folder_path)

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
        raise FileNotFoundError(f"No supported video file found in folder: {folder_path}")

    return max(candidates, key=lambda path: path.stat().st_size)


def resolve_video_path(input_path: Path) -> Path:
    if input_path.is_dir():
        video_path = find_main_video_in_folder(input_path)
        print(f"Selected main video from folder: {video_path}")
        return video_path
    return input_path


def extract_audio(video_path: Path, temp_audio_path: Path, audio_stream: Optional[int] = None) -> None:
    print(f"Extracting audio from {video_path.name}...")
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
    ]
    if audio_stream is not None:
        cmd.extend(["-map", f"0:{audio_stream}"])
    cmd.extend([
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(temp_audio_path),
    ])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Audio extraction complete.")


def transcribe_audio(audio_path: Path, language: Optional[str] = "auto") -> List[Segment]:
    print("Loading faster-whisper large-v3 model with FP16...")
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
    )

    try:
        print("Transcribing audio with word timestamps...")
        requested_language = None if not language or language == "auto" else language
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            language=requested_language,
            multilingual=requested_language is None,
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )

        detected_language = str(getattr(info, "language", "") or requested_language or language or "auto")
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        print(f"Detected language '{detected_language}' with probability {language_probability}")

        results: List[Segment] = []
        for segment in segments:
            words = []
            word_probabilities: List[float] = []
            for word in getattr(segment, "words", None) or []:
                text = getattr(word, "word", "").strip()
                if text:
                    word_item = {
                        "start": float(getattr(word, "start", segment.start)),
                        "end": float(getattr(word, "end", segment.end)),
                        "word": text,
                    }
                    probability = getattr(word, "probability", None)
                    if probability is not None:
                        try:
                            normalized_probability = float(probability)
                        except (TypeError, ValueError):
                            normalized_probability = None
                        if (
                            normalized_probability is not None
                            and math.isfinite(normalized_probability)
                            and 0 <= normalized_probability <= 1
                        ):
                            word_item["probability"] = normalized_probability
                            word_probabilities.append(normalized_probability)
                    words.append(word_item)

            text = normalize_space(segment.text)
            if not text:
                continue
            item = {
                "id": len(results),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "source_language": detected_language,
                "asr_language_probability": language_probability,
            }
            for key, attribute in (
                ("asr_avg_logprob", "avg_logprob"),
                ("asr_compression_ratio", "compression_ratio"),
                ("asr_no_speech_prob", "no_speech_prob"),
                ("asr_temperature", "temperature"),
            ):
                value = getattr(segment, attribute, None)
                try:
                    normalized_value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(normalized_value):
                    item[key] = normalized_value
            if words:
                item["words"] = words
            if word_probabilities:
                item["asr_word_confidence"] = sum(word_probabilities) / len(word_probabilities)
            results.append(item)
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")

        return results
    finally:
        runtime_model = getattr(model, "model", None)
        unload_model = getattr(runtime_model, "unload_model", None)
        if callable(unload_model):
            try:
                unload_model()
                print("Released faster-whisper model from the local GPU.")
            except Exception as exc:
                print(f"Warning: could not unload faster-whisper from the local GPU: {exc}", file=sys.stderr)


def call_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = "qwen3:14b",
    *,
    role: str = "translation",
    response_schema: Optional[Dict[str, Any]] = None,
    seed_offset: int = 0,
) -> str:
    base_url = "http://localhost:11434"
    remote_model = model.startswith("remote:")
    if remote_model:
        parts = model.split(":", 2)
        if len(parts) != 3:
            raise ValueError("Invalid remote model identifier; reconnect the remote server.")
        remote_base_url = os.environ.get("SUBTITLE_REMOTE_OLLAMA_URL", "").rstrip("/")
        if not remote_base_url:
            raise RuntimeError("Remote model selected without a tunnel owned by this frontend process.")
        base_url = remote_base_url
        model = parts[2]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    profile = generation_profile(role)
    num_ctx = profile.num_ctx
    if remote_model:
        remote_num_ctx = os.environ.get(
            "SUBTITLE_REMOTE_OLLAMA_NUM_CTX",
            "",
        ).strip()
        if remote_num_ctx:
            try:
                num_ctx = int(remote_num_ctx)
            except ValueError as exc:
                raise ValueError(
                    "SUBTITLE_REMOTE_OLLAMA_NUM_CTX must be an integer."
                ) from exc
            if not 4096 <= num_ctx <= 1_048_576:
                raise ValueError(
                    "SUBTITLE_REMOTE_OLLAMA_NUM_CTX must be between 4096 and 1048576."
                )
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": profile.think,
        "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
        "options": {
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "top_k": profile.top_k,
            "seed": profile.seed + max(0, int(seed_offset)),
            "num_ctx": num_ctx,
            "num_predict": profile.max_tokens,
        },
    }
    if response_schema is not None:
        payload["format"] = response_schema

    response = post_ollama_chat(base_url, payload)
    message = response.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not content:
        raise ValueError("LLM returned an empty response")
    return str(content)


def post_ollama_chat(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        timeout = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "600"))
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Ollama request failed with HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Ollama at {base_url}: {exc.reason}") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid Ollama response from {base_url}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Ollama response must be a JSON object")
    if data.get("error"):
        raise RuntimeError(f"Ollama request failed: {data['error']}")
    return data


def strip_llm_noise(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _schema_string_property_names(response_schema: Dict[str, Any]) -> set[str]:
    names: set[str] = set()

    def collect(node: Any) -> None:
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, property_schema in properties.items():
                if isinstance(property_schema, dict) and property_schema.get("type") == "string":
                    names.add(str(name))
                collect(property_schema)
        collect(node.get("items"))

    collect(response_schema)
    return names


def escape_unescaped_json_quotes(value: str) -> str:
    escaped: List[str] = []
    backslash_run = 0
    for character in value:
        if character == '"' and backslash_run % 2 == 0:
            escaped.append('\\"')
        else:
            escaped.append(character)
        if character == "\\":
            backslash_run += 1
        else:
            backslash_run = 0
    return "".join(escaped)


def repair_unquoted_schema_strings(text: str, response_schema: Dict[str, Any]) -> str:
    string_properties = _schema_string_property_names(response_schema)
    if not string_properties:
        return text

    lines = text.splitlines(keepends=True)
    repaired_lines: List[str] = []
    changed = False
    property_pattern = re.compile(
        r'^(?P<prefix>\s*"(?P<key>(?:\\.|[^"\\])*)"\s*:\s*)(?P<value>.*)$'
    )
    for line_index, line in enumerate(lines):
        newline = ""
        body = line
        if body.endswith("\r\n"):
            body, newline = body[:-2], "\r\n"
        elif body.endswith("\n"):
            body, newline = body[:-1], "\n"

        match = property_pattern.match(body)
        if not match or match.group("key") not in string_properties:
            repaired_lines.append(line)
            continue

        value_with_spacing = match.group("value")
        trailing_spacing = value_with_spacing[len(value_with_spacing.rstrip()) :]
        value = value_with_spacing.rstrip()
        has_comma = value.endswith(",")
        if has_comma:
            value = value[:-1].rstrip()
        raw_value = value.strip()

        if not raw_value:
            repaired_lines.append(
                match.group("prefix")
                + '""'
                + ("," if has_comma else "")
                + trailing_spacing
                + newline
            )
            changed = True
            continue

        if raw_value.startswith('"'):
            try:
                json.loads(raw_value)
            except json.JSONDecodeError:
                if len(raw_value) < 2 or not raw_value.endswith('"'):
                    repaired_lines.append(line)
                    continue
                escaped_inner = escape_unescaped_json_quotes(raw_value[1:-1])
                try:
                    decoded_value = json.loads(f'"{escaped_inner}"')
                except json.JSONDecodeError:
                    repaired_lines.append(line)
                    continue
                repaired_lines.append(
                    match.group("prefix")
                    + json.dumps(decoded_value, ensure_ascii=False)
                    + ("," if has_comma else "")
                    + trailing_spacing
                    + newline
                )
                changed = True
                continue
            repaired_lines.append(line)
            continue

        if (
            raw_value in {"null", "true", "false"}
            or raw_value.startswith(("{", "["))
        ):
            repaired_lines.append(line)
            continue

        if raw_value.endswith('"'):
            raw_value = raw_value[:-1].rstrip()
        if len(raw_value) >= 2 and raw_value.startswith("'") and raw_value.endswith("'"):
            raw_value = raw_value[1:-1]
        if not raw_value:
            repaired_lines.append(line)
            continue

        if not has_comma:
            for following_line in lines[line_index + 1 :]:
                following_content = following_line.strip()
                if not following_content:
                    continue
                if following_content.startswith('"'):
                    has_comma = True
                break

        repaired_lines.append(
            match.group("prefix")
            + json.dumps(raw_value, ensure_ascii=False)
            + ("," if has_comma else "")
            + trailing_spacing
            + newline
        )
        changed = True

    return "".join(repaired_lines) if changed else text


def parse_concatenated_json_array(text: str) -> List[Any]:
    decoder = json.JSONDecoder()
    values: List[Any] = []
    position = 0
    while position < len(text):
        while position < len(text) and (
            text[position].isspace() or text[position] == ","
        ):
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        if isinstance(value, list):
            values.extend(value)
        elif isinstance(value, dict):
            values.append(value)
        else:
            raise json.JSONDecodeError(
                "Concatenated structured response contains a scalar",
                text,
                position,
            )
    if not values:
        raise json.JSONDecodeError(
            "Concatenated structured response is empty",
            text,
            0,
        )
    return values


def parse_json_response(
    text: str,
    response_schema: Optional[Dict[str, Any]] = None,
) -> Any:
    cleaned = strip_llm_noise(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as initial_error:
        expects_array = bool(
            response_schema
            and response_schema.get("type") == "array"
        )
        if expects_array:
            try:
                parsed = parse_concatenated_json_array(cleaned)
                print("Repaired concatenated structured LLM JSON values.")
                return parsed
            except json.JSONDecodeError:
                pass
        if response_schema is not None:
            repaired = repair_unquoted_schema_strings(cleaned, response_schema)
            if repaired != cleaned:
                try:
                    parsed = json.loads(repaired)
                    print("Repaired unquoted string values in structured LLM JSON.")
                    return parsed
                except json.JSONDecodeError:
                    if expects_array:
                        try:
                            parsed = parse_concatenated_json_array(repaired)
                            print(
                                "Repaired unquoted and concatenated "
                                "structured LLM JSON values."
                            )
                            return parsed
                        except json.JSONDecodeError:
                            pass
        start_candidates = [pos for pos in (cleaned.find("["), cleaned.find("{")) if pos >= 0]
        if not start_candidates:
            raise initial_error
        start = min(start_candidates)
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end <= start:
            raise initial_error
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            if response_schema is None:
                raise
            repaired = repair_unquoted_schema_strings(candidate, response_schema)
            if repaired == candidate:
                raise
            parsed = json.loads(repaired)
            print("Repaired unquoted string values in structured LLM JSON.")
            return parsed


def validate_indexed_batch_response(
    data: Any,
    expected_count: int,
    role: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    if role not in {None, "translation", "proofread"}:
        raise ValueError(f"Unsupported response validation role: {role}")
    if not isinstance(data, list):
        raise ValueError("LLM did not return a JSON list")
    if len(data) != expected_count:
        raise ValueError(f"LLM returned {len(data)} objects; expected {expected_count}")

    lookup: Dict[int, Dict[str, Any]] = {}
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"LLM response item {position} is not an object")
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"LLM response item {position} has no valid index") from exc
        if index < 0 or index >= expected_count:
            raise ValueError(f"LLM response index {index} is outside 0..{expected_count - 1}")
        if index in lookup:
            raise ValueError(f"LLM response contains duplicate index {index}")
        if role is not None:
            required_text_fields = (
                ("corrected_text", "chinese_translation")
                if role == "translation"
                else ("corrected_english", "corrected_chinese")
            )
            for field in required_text_fields:
                if not isinstance(item.get(field), str):
                    raise ValueError(
                        f"LLM response item {position} has no valid {field}"
                    )
            if not isinstance(item.get("display"), bool):
                raise ValueError(
                    f"LLM response item {position} has no boolean display flag"
                )
            terminology = item.get("terminology", [])
            if not isinstance(terminology, list):
                raise ValueError(
                    f"LLM response item {position} has invalid terminology"
                )
            for entry_index, entry in enumerate(terminology):
                if (
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("source"), str)
                    or not isinstance(entry.get("target"), str)
                ):
                    raise ValueError(
                        "LLM response item "
                        f"{position} has invalid terminology entry {entry_index}"
                    )
        lookup[index] = item

    expected_indexes = set(range(expected_count))
    if set(lookup) != expected_indexes:
        missing = sorted(expected_indexes - set(lookup))
        raise ValueError(f"LLM response is missing indexes: {missing}")
    return lookup


def normalize_final_qa_batch_response(
    data: Any,
    batch: List[Segment],
) -> Any:
    if not isinstance(data, list):
        return data
    normalized: List[Any] = []
    for item in data:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        output = dict(item)
        if "index" not in output and "local_index" in output:
            output["index"] = output["local_index"]
        try:
            index = int(output["index"])
        except (KeyError, TypeError, ValueError):
            index = -1
        aliases = {
            "corrected_english": (
                "correctedEnglish",
                "corrected_source",
                "source",
                "english",
                "en",
            ),
            "corrected_chinese": (
                "correctedChinese",
                "chinese",
                "zh",
            ),
        }
        for field, field_aliases in aliases.items():
            if isinstance(output.get(field), str):
                continue
            for alias in field_aliases:
                value = output.get(alias)
                if isinstance(value, str):
                    output[field] = value
                    break
        if "display" not in output and 0 <= index < len(batch):
            output["display"] = batch[index].get("display", True) is not False
        output.setdefault("terminology", [])
        normalized.append(output)
    return normalized


def parse_movie_name(filename: str, llm_model: str) -> Tuple[str, str]:
    system_prompt = "You extract clean movie and series names from release filenames."
    prompt = f"""
Given the filename "{filename}", extract:
1. "series_name": for a series or show, the show name; for a standalone movie, the movie name.
2. "movie_name": the clean episode/movie name. Preserve episode markers like 1of8 when present.

Return ONLY a valid JSON object with keys "series_name" and "movie_name".
"""
    try:
        response_schema = movie_name_schema()
        data = parse_json_response(
            call_llm(
                prompt,
                system_prompt,
                llm_model,
                role="metadata",
                response_schema=response_schema,
            ),
            response_schema=response_schema,
        )
        return data.get("series_name", "Unknown"), data.get("movie_name", "Unknown")
    except Exception as exc:
        print(f"Failed to parse movie name with LLM: {exc}")
        name = Path(filename).stem
        name = re.sub(r"\.(20\d\d|19\d\d).*", "", name).replace(".", " ")
        return name, name


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_subtitle_text(text: str) -> str:
    text = re.sub(r"\{\\[^}]*\}", "", text)
    text = re.sub(r"</?(i|b|u|font)[^>]*>", "", text, flags=re.I)
    text = normalize_space(text)
    return "" if text == "-" else text


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
AUTHORED_CHINESE_CUE_RE = re.compile(
    r"\[[^\[\]\n]{1,120}\]|\([^()\n]{1,120}\)|"
    r"\u3010[^\u3010\u3011\n]{1,120}\u3011|"
    r"\uff08[^\uff08\uff09\n]{1,120}\uff09"
)
AUTHORED_CHINESE_TITLE_CARD_RE = re.compile(
    r"(?:\u7247\u540d|\u5267\u540d|\u6807\u9898)\s*[:\uff1a]|"
    r"\u7b2c\s*(?:\d{1,3}|[\u96f6\u3007\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]{1,6})"
    r"\s*(?:\u96c6|\u5b63|\u7ae0|\u56de|\u90e8)"
)
EAST_ASIAN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
JAPANESE_SCRIPT_RE = re.compile(r"[\u3040-\u30ff]")
KOREAN_SCRIPT_RE = re.compile(r"[\uac00-\ud7af]")
SUBTITLE_TRACK_LABEL_RE = re.compile(
    r"^\s*[\(\[\{\uFF08\u3010]?\s*"
    r"(?:english|eng|chinese|chi|zho|mandarin|cantonese)"
    r"\s*(?:[-:\u2013\u2014\uFF1A]\s*)?"
    r"(?:sdh|cc|closed\s+captions?|subtitles?|captions?)"
    r"\s*[\)\]\}\uFF09\u3011]?\s*$",
    re.IGNORECASE,
)


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def contains_east_asian(text: str) -> bool:
    return bool(EAST_ASIAN_RE.search(text or ""))


def source_language_preserved(
    segment: Segment,
    original_source: str,
    corrected_source: str,
) -> bool:
    original = clean_subtitle_text(original_source)
    corrected = clean_subtitle_text(corrected_source)
    if not original:
        return not corrected
    if not corrected:
        return False

    language = normalize_language_tag(str(segment.get("source_language") or ""))
    if language == "en":
        return not contains_east_asian(corrected) or contains_east_asian(original)
    if language == "ja":
        if JAPANESE_SCRIPT_RE.search(original) and not JAPANESE_SCRIPT_RE.search(corrected):
            return False
        return not contains_east_asian(original) or contains_east_asian(corrected)
    if language == "ko":
        if KOREAN_SCRIPT_RE.search(original) and not KOREAN_SCRIPT_RE.search(corrected):
            return False
        return not contains_east_asian(original) or contains_east_asian(corrected)
    if language == "zh":
        return not contains_east_asian(original) or contains_east_asian(corrected)
    if JAPANESE_SCRIPT_RE.search(original):
        return bool(JAPANESE_SCRIPT_RE.search(corrected))
    if KOREAN_SCRIPT_RE.search(original):
        return bool(KOREAN_SCRIPT_RE.search(corrected))
    if contains_east_asian(original) and not contains_east_asian(corrected):
        return False
    return True


def clean_english_track_text(text: str) -> str:
    text = clean_subtitle_text(text)
    return "" if contains_east_asian(text) else text


def clean_ocr_english_track_text(text: str) -> str:
    text = clean_english_track_text(text)
    if text == "0":
        return ""
    return re.sub(r"\s+0$", "", text).rstrip()


def should_clean_english_track(segment: Segment) -> bool:
    language = normalize_language_tag(str(segment.get("source_language") or ""))
    if language == "zh":
        return True
    if language not in UNKNOWN_SOURCE_LANGUAGE_TAGS:
        return False
    return True


def source_track_text(segment: Segment) -> str:
    text = clean_subtitle_text(str(segment.get("en") or ""))
    return clean_english_track_text(text) if should_clean_english_track(segment) else text


SOURCE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?|%")
SOURCE_BOUNDARY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def source_boundary_tokens(text: str) -> List[str]:
    tokens = [
        token.replace("\u2019", "'").casefold()
        for token in SOURCE_TOKEN_RE.findall(clean_subtitle_text(text))
    ]
    return ["%" if token == "percent" else token for token in tokens]


def token_sequence_contains(haystack: List[str], needle: List[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def substantial_contiguous_token_overlap(
    haystack: List[str],
    needle: List[str],
    *,
    minimum_tokens: int = 3,
    minimum_ratio: float = 0.6,
) -> bool:
    required = max(minimum_tokens, math.ceil(len(needle) * minimum_ratio))
    if len(needle) < required:
        return False
    return any(
        token_sequence_contains(haystack, needle[start : start + required])
        for start in range(len(needle) - required + 1)
    )


def has_fuzzy_ocr_token_anchor(
    haystack: List[str],
    needle: List[str],
    *,
    minimum_similarity: float = 0.78,
) -> bool:
    return any(
        original != proposed
        and len(original) >= 4
        and len(proposed) >= 4
        and SequenceMatcher(None, original, proposed).ratio() >= minimum_similarity
        for original in haystack
        for proposed in needle
    )


def has_substantial_unordered_token_overlap(
    haystack: List[str],
    needle: List[str],
    *,
    minimum_ratio: float = 0.6,
) -> bool:
    if not needle:
        return False
    required = max(2, math.ceil(len(needle) * minimum_ratio))
    matched = len(set(needle) & set(haystack))
    return matched >= required


def adjacent_context_borrowed_phrase(
    segments: List[Segment],
    index: int,
    proposed_source: str,
    *,
    minimum_tokens: int = 3,
) -> str:
    if index < 0 or index >= len(segments):
        return ""
    segment = segments[index]
    authority = str(
        segment.get("source_text_authority")
        or segment.get("text_authority")
        or ""
    )
    original = clean_subtitle_text(str(segment.get("text") or ""))
    if authority != "ocr" or contains_east_asian(original):
        return ""
    original_tokens = source_boundary_tokens(original)
    proposed_tokens = source_boundary_tokens(proposed_source)
    if len(proposed_tokens) < minimum_tokens:
        return ""

    neighbor_tokens = [
        source_boundary_tokens(str(segments[neighbor_index].get("text") or ""))
        for neighbor_index in (index - 1, index + 1)
        if 0 <= neighbor_index < len(segments)
    ]
    for size in range(len(proposed_tokens), minimum_tokens - 1, -1):
        for start in range(len(proposed_tokens) - size + 1):
            phrase = proposed_tokens[start : start + size]
            if token_sequence_contains(original_tokens, phrase):
                continue
            if substantial_contiguous_token_overlap(
                original_tokens,
                phrase,
                minimum_tokens=minimum_tokens,
            ):
                continue
            content_tokens = [
                token
                for token in phrase
                if token not in SOURCE_BOUNDARY_STOPWORDS
            ]
            if len(content_tokens) < 3:
                continue
            if has_substantial_unordered_token_overlap(
                original_tokens,
                content_tokens,
            ):
                continue
            if has_fuzzy_ocr_token_anchor(original_tokens, content_tokens):
                continue
            if any(token_sequence_contains(neighbor, phrase) for neighbor in neighbor_tokens):
                return " ".join(phrase)
    return ""


def authored_chinese_content_retained(
    segment: Segment,
    original_chinese: str,
    proposed_chinese: str,
    *,
    minimum_original_characters: int = 32,
    minimum_ratio: float = 0.45,
) -> bool:
    authority = str(segment.get("chinese_text_authority") or "")
    if authority != "authored":
        return True
    if len(AUTHORED_CHINESE_CUE_RE.findall(proposed_chinese)) < len(
        AUTHORED_CHINESE_CUE_RE.findall(original_chinese)
    ):
        return False
    original_count = len(CJK_RE.findall(clean_subtitle_text(original_chinese)))
    if original_count < minimum_original_characters:
        return True
    proposed_count = len(CJK_RE.findall(clean_subtitle_text(proposed_chinese)))
    return proposed_count >= math.ceil(original_count * minimum_ratio)


def authored_chinese_title_card_allows_ocr_reconstruction(
    segment: Segment,
    original_chinese: str,
    proposed_source: str,
    borrowed_phrase: str,
    *,
    maximum_source_tokens: int = 20,
) -> bool:
    if not borrowed_phrase:
        return False
    source_authority = str(
        segment.get("source_text_authority")
        or segment.get("text_authority")
        or ""
    )
    if (
        source_authority != "ocr"
        or str(segment.get("chinese_text_authority") or "") != "authored"
        or not AUTHORED_CHINESE_TITLE_CARD_RE.search(original_chinese or "")
    ):
        return False
    proposed_tokens = source_boundary_tokens(proposed_source)
    return 2 <= len(proposed_tokens) <= maximum_source_tokens


def semantic_track_similarity(left: str, right: str) -> float:
    left_text = clean_subtitle_text(left)
    right_text = clean_subtitle_text(right)
    if not left_text or not right_text:
        return 0.0
    if contains_cjk(left_text) and contains_cjk(right_text):
        left_compact = "".join(CJK_RE.findall(left_text))
        right_compact = "".join(CJK_RE.findall(right_text))
        width = 2 if min(len(left_compact), len(right_compact)) >= 2 else 1
        left_units = {
            left_compact[index : index + width]
            for index in range(len(left_compact) - width + 1)
        }
        right_units = {
            right_compact[index : index + width]
            for index in range(len(right_compact) - width + 1)
        }
    else:
        left_units = {
            token
            for token in source_boundary_tokens(left_text)
            if token not in SOURCE_BOUNDARY_STOPWORDS
        }
        right_units = {
            token
            for token in source_boundary_tokens(right_text)
            if token not in SOURCE_BOUNDARY_STOPWORDS
        }
    if not left_units or not right_units:
        return 0.0
    return len(left_units & right_units) / len(left_units | right_units)


def previous_output_borrowed_phrase(
    source_segments: List[Segment],
    source_index: int,
    approved_segments: List[Segment],
    proposed_source: str,
    proposed_chinese: str,
) -> str:
    if source_index <= 0 or source_index >= len(source_segments):
        return ""
    previous_source = source_segments[source_index - 1]
    previous_id = previous_source.get("id")
    previous_output = next(
        (
            item
            for item in reversed(approved_segments)
            if item.get("id") == previous_id
        ),
        None,
    )
    if previous_output is None or previous_output.get("display", True) is False:
        return ""

    current_source = source_segments[source_index]
    current_original_source = str(current_source.get("text") or "")
    current_original_chinese = str(current_source.get("zh") or "")
    previous_original_source = str(previous_source.get("text") or "")
    previous_original_chinese = str(previous_source.get("zh") or "")
    source_repeat_score = max(
        semantic_track_similarity(
            current_original_source,
            previous_original_source,
        ),
        semantic_track_similarity(
            current_original_chinese,
            previous_original_chinese,
        ),
    )
    if source_repeat_score >= 0.55:
        return ""

    proposed_prior_score = max(
        semantic_track_similarity(
            proposed_source,
            str(previous_output.get("en") or previous_output.get("text") or ""),
        ),
        semantic_track_similarity(
            proposed_chinese,
            str(previous_output.get("zh") or ""),
        ),
    )
    proposed_own_score = max(
        semantic_track_similarity(proposed_source, current_original_source),
        semantic_track_similarity(
            proposed_chinese,
            current_original_chinese,
        ),
    )
    if (
        proposed_prior_score >= 0.28
        and proposed_own_score < 0.22
        and proposed_prior_score - proposed_own_score >= 0.15
    ):
        return normalize_space(proposed_source or proposed_chinese)[:120]
    return ""


def source_editing_policy(segments: List[Segment]) -> str:
    authorities = {
        str(
            segment.get("source_text_authority")
            or segment.get("text_authority")
            or "unknown"
        )
        for segment in segments
    }
    if authorities == {"authored"}:
        return (
            "The source text is professionally authored. Preserve it exactly after markup and "
            "whitespace normalization. Do not rewrite wording, merge repetitions, or hide events; "
            "only produce the Chinese translation or missing-content review."
        )
    if "ocr" in authorities:
        return (
            "The source includes OCR text. Repair only evident recognition errors, prioritizing "
            "low-confidence or impossible text and preserving credible authored wording. Keep "
            "each correction inside its original timestamp boundary; neighboring cues are context "
            "only and must never be copied into or completed inside the target cue."
        )
    if "asr" in authorities:
        return (
            "The source includes ASR text. You may repair recognition errors and proven adjacent "
            "hallucination loops while preserving unique spoken content."
        )
    return (
        "Preserve credible source wording. Only repair an error when the supplied source evidence "
        "supports the change."
    )


def preserve_authored_source(
    segment: Segment,
    original: str,
    proposed: str,
) -> str:
    if is_authored_source(segment):
        return clean_subtitle_text(original)
    return clean_subtitle_text(proposed)


def classify_subtitle_text(text: str) -> str:
    sample = text[:20_000]
    cjk_count = len(CJK_RE.findall(sample))
    latin_words = len(re.findall(r"\b[A-Za-z]{2,}\b", sample))
    if cjk_count >= 5:
        return "zh"
    if latin_words >= 5:
        return "en"
    return ""


def classify_subtitle_path(path: Path) -> str:
    label = path.stem.lower()
    zh_tokens = {
        "zh",
        "zho",
        "chi",
        "chs",
        "cht",
        "cmn",
        "cn",
        "sc",
        "tc",
        "zh-cn",
        "zh-hans",
        "zh-tw",
        "zh-hant",
        "chinese",
        "简",
        "简体",
        "繁",
        "繁体",
        "中文",
    }
    en_tokens = {"en", "eng", "english", "英文"}
    parts = set(re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", label))
    if parts & zh_tokens:
        return "zh"
    if parts & en_tokens:
        return "en"
    try:
        return classify_subtitle_text(read_text_with_fallback(path))
    except Exception:
        return ""


def to_simplified_text(text: str) -> str:
    if not contains_cjk(text):
        return text
    try:
        from opencc import OpenCC
    except Exception:
        converted = text
    else:
        if not hasattr(to_simplified_text, "_converter"):
            setattr(to_simplified_text, "_converter", OpenCC("t2s"))
        converted = getattr(to_simplified_text, "_converter").convert(text)
    cjk = r"\u3400-\u4dbf\u4e00-\u9fff"
    converted = re.sub(
        rf"(?<=[{cjk}，。！？；：、）》】」』])\s+(?=[{cjk}A-Za-z0-9%《【「『（])",
        "",
        converted,
    )
    converted = re.sub(
        rf"(?<=[A-Za-z0-9%》】」』）\]])\s+(?=[{cjk}，。！？；：、])",
        "",
        converted,
    )
    return converted


def to_simplified_segments(segments: List[Segment]) -> List[Segment]:
    output: List[Segment] = []
    for segment in segments:
        item = dict(segment)
        if item.get("zh"):
            item["zh"] = to_simplified_text(str(item["zh"]))
        item["text"] = to_simplified_text(str(item.get("text", "")))
        output.append(item)
    return output


def read_text_with_fallback(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def srt_time_to_seconds(value: str) -> float:
    match = re.match(r"(\d\d):(\d\d):(\d\d),(\d\d\d)", value.strip())
    if not match:
        raise ValueError(f"Bad SRT time: {value}")
    hours, minutes, seconds, millis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_srt_file(srt_path: Path) -> List[Segment]:
    print(f"Using sidecar SRT timing: {srt_path}")
    text = read_text_with_fallback(srt_path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: List[Segment] = []

    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines()]
        time_index = -1
        time_match = None
        for idx, line in enumerate(lines):
            time_match = re.search(
                r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)",
                line,
            )
            if time_match:
                time_index = idx
                break
        if not time_match:
            continue

        body = " ".join(line.strip() for line in lines[time_index + 1 :] if line.strip())
        body = clean_subtitle_text(body)
        if not body:
            continue

        start = srt_time_to_seconds(time_match.group(1))
        end = srt_time_to_seconds(time_match.group(2))
        if end <= start:
            continue

        segments.append(
            {
                "id": len(segments),
                "start": start,
                "end": end,
                "text": body,
            }
        )

    return segments


def ass_time_to_seconds(value: str) -> float:
    match = re.match(r"(\d+):(\d\d):(\d\d)[.](\d\d)", value.strip())
    if not match:
        raise ValueError(f"Bad ASS time: {value}")
    hours, minutes, seconds, centis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centis) / 100


def strip_ass_tags(text: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", text)
    text = text.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    return clean_subtitle_text(text)


def parse_ass_file(path: Path) -> List[Segment]:
    print(f"Using sidecar ASS/SSA timing: {path}")
    segments: List[Segment] = []
    text = read_text_with_fallback(path)
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        try:
            start = ass_time_to_seconds(parts[1])
            end = ass_time_to_seconds(parts[2])
        except ValueError:
            continue
        body = strip_ass_tags(parts[9])
        if not body or end <= start:
            continue
        segments.append({"id": len(segments), "start": start, "end": end, "text": body})
    return segments


def parse_vtt_file(path: Path) -> List[Segment]:
    print(f"Using sidecar WebVTT timing: {path}")
    text = read_text_with_fallback(path).replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: List[Segment] = []

    def vtt_time_to_seconds(value: str) -> float:
        value = value.strip().replace(",", ".")
        fields = value.split(":")
        if len(fields) == 2:
            minutes, rest = fields
            hours = 0
        else:
            hours, minutes, rest = fields[-3:]
        seconds, millis = (rest.split(".") + ["0"])[:2]
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis[:3].ljust(3, "0")) / 1000

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next((idx for idx, line in enumerate(lines) if "-->" in line), -1)
        if time_index < 0:
            continue
        match = re.search(r"([0-9:.]+)\s*-->\s*([0-9:.]+)", lines[time_index])
        if not match:
            continue
        body = clean_subtitle_text(" ".join(lines[time_index + 1 :]))
        if not body:
            continue
        start = vtt_time_to_seconds(match.group(1))
        end = vtt_time_to_seconds(match.group(2))
        if end > start:
            segments.append({"id": len(segments), "start": start, "end": end, "text": body})
    return segments


def parse_subtitle_file(path: Path) -> List[Segment]:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return parse_srt_file(path)
    if suffix in {".ass", ".ssa"}:
        return parse_ass_file(path)
    if suffix == ".vtt":
        return parse_vtt_file(path)
    raise ValueError(f"Unsupported sidecar subtitle format: {path.suffix}")


def find_sidecar_subtitles(video_path: Path, selected_path: Optional[Path] = None) -> List[Path]:
    candidates: List[Path] = []
    roots = [video_path.parent]
    if selected_path and selected_path.is_dir() and selected_path not in roots:
        roots.append(selected_path)
    for ext in SUBTITLE_EXTENSIONS:
        direct = video_path.with_suffix(ext)
        if direct.exists():
            candidates.append(direct)

    for root in roots:
        candidates.extend(sorted(root.glob("*.srt")))
        for ext in sorted(SUBTITLE_EXTENSIONS - {".srt"}):
            candidates.extend(sorted(root.glob(f"*{ext}")))

    scored: List[tuple[int, str, Path]] = []
    for candidate in candidates:
        stem = candidate.stem.lower()
        score = sidecar_match_score(video_path, candidate)
        if any(token in stem for token in ("en", "eng", "english")):
            score += 10
        scored.append((score, candidate.name.lower(), candidate))
    unique = {}
    for score, name, path in scored:
        if path not in unique or score > unique[path][0]:
            unique[path] = (score, name, path)
    return [item[2] for item in sorted(unique.values(), key=lambda item: (item[0], item[1]), reverse=True)]


def sidecar_match_score(video_path: Path, subtitle_path: Path) -> int:
    video_stem = video_path.stem.casefold()
    subtitle_stem = subtitle_path.stem.casefold()
    if subtitle_stem == video_stem:
        return 100
    if video_stem in subtitle_stem or subtitle_stem in video_stem:
        return 50
    return 0


def likely_sidecar_subtitles(video_path: Path, selected_path: Optional[Path] = None) -> List[Path]:
    candidates = find_sidecar_subtitles(video_path, selected_path)
    matched = [candidate for candidate in candidates if sidecar_match_score(video_path, candidate) > 0]
    if matched:
        return matched
    return candidates if len(candidates) == 1 else []


def find_sidecar_subtitle(video_path: Path, selected_path: Optional[Path] = None) -> Optional[Path]:
    subtitles = likely_sidecar_subtitles(video_path, selected_path)
    return subtitles[0] if subtitles else None


def sidecar_candidates(
    video_path: Path,
    selected_path: Optional[Path],
    explicit_path: Optional[Path] = None,
    explicit_zh_path: Optional[Path] = None,
    explicit_en_path: Optional[Path] = None,
) -> List[Path]:
    candidates: List[Path] = []
    candidates.extend(path for path in (explicit_path, explicit_zh_path, explicit_en_path) if path)
    candidates.extend(likely_sidecar_subtitles(video_path, selected_path))

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def segments_to_events(segments: List[Segment], key: str = "text") -> List[Any]:
    from subtitle_pipeline import SubtitleEvent

    events = []
    for segment in segments:
        text = clean_subtitle_text(str(segment.get(key) or segment.get("text") or ""))
        if not text:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            continue
        confidence = segment.get("ocr_confidence")
        metadata = (
            dict(segment.get("ocr_metadata"))
            if isinstance(segment.get("ocr_metadata"), dict)
            else {}
        )
        for field_name in (
            "source_asset_id",
            "source_origin",
            "source_representation",
            "source_role",
            "source_text_authority",
            "source_timing_authority",
            "chinese_asset_id",
            "chinese_origin",
            "chinese_representation",
            "chinese_role",
            "chinese_text_authority",
            "chinese_timing_authority",
            "supplementary_source",
        ):
            if segment.get(field_name) is not None:
                metadata[field_name] = segment[field_name]
        source_assets = [
            dict(item)
            for item in segment.get("source_assets") or []
            if isinstance(item, dict)
        ]
        if source_assets:
            metadata["source_assets"] = source_assets
        events.append(
            SubtitleEvent(
                start,
                end,
                text,
                float(confidence) if confidence is not None else None,
                metadata or None,
            )
        )
    return events


def merge_existing_subtitle_segments(
    en_segments: Optional[List[Segment]],
    zh_segments: List[Segment],
    source_label: str,
    *,
    english_asset: Optional[SubtitleAsset] = None,
    chinese_asset: Optional[SubtitleAsset] = None,
    supplementary_assets: Tuple[SubtitleAsset, ...] = (),
) -> List[Segment]:
    from subtitle_pipeline import is_subtitle_credit_text, pair_events

    zh_events = segments_to_events(to_simplified_segments(zh_segments))
    en_events = segments_to_events(en_segments or [])
    if not zh_events:
        return []

    pairs = pair_events(en_events, zh_events) if en_events else [(None, zh) for zh in zh_events]
    merged: List[Segment] = []
    for en_event, zh_event in pairs:
        if not en_event and not zh_event:
            continue
        timing_event = en_event or zh_event
        timing_asset = english_asset if en_event else chinese_asset
        timing_correction = ""
        if en_event and zh_event and english_asset and chinese_asset:
            english_timing_score = (
                english_asset.timing_authority == "authored",
                english_asset.origin == "embedded",
                english_asset.representation == "authored_text",
            )
            chinese_timing_score = (
                chinese_asset.timing_authority == "authored",
                chinese_asset.origin == "embedded",
                chinese_asset.representation == "authored_text",
            )
            if chinese_timing_score > english_timing_score:
                timing_event = zh_event
                timing_asset = chinese_asset
            if (
                english_asset.text_authority == "ocr"
                and chinese_asset.text_authority == "authored"
            ):
                timing_event = zh_event
                timing_asset = chinese_asset
                timing_correction = "authored_anchor_preferred_over_ocr"
            en_duration = max(0.01, en_event.end - en_event.start)
            zh_duration = max(0.01, zh_event.end - zh_event.start)
            if (
                english_asset.text_authority == "ocr"
                and chinese_asset.text_authority == "authored"
                and en_duration > 12.0
                and en_duration > zh_duration * 2.0
                and zh_duration <= 10.0
            ):
                timing_event = zh_event
                timing_asset = chinese_asset
                timing_correction = "authored_anchor_replaced_abnormal_ocr_span"
        start = timing_event.start
        end = timing_event.end
        en_text = clean_english_track_text(en_event.text) if en_event else ""
        if (
            en_text
            and english_asset is not None
            and english_asset.text_authority == "ocr"
        ):
            en_text = clean_ocr_english_track_text(en_text)
        zh_text = to_simplified_text(clean_subtitle_text(zh_event.text)) if zh_event else ""
        text = en_text or zh_text
        if not text and not zh_text:
            continue
        item: Segment = {
            "id": len(merged),
            "start": start,
            "end": end,
            "text": text,
            "en": en_text,
            "zh": zh_text,
            "source_language": (
                english_asset.language
                if en_text and english_asset
                else (chinese_asset.language if chinese_asset else source_label)
            ),
            "source_label": source_label,
            "display": True,
            "preserve_distinct_overlap": True,
        }
        if is_subtitle_credit_text(en_text) or is_subtitle_credit_text(zh_text):
            item["display"] = False
            item["suppression_reason"] = "subtitle_credit"
            item["non_content_credit"] = True
        if en_event and isinstance(en_event.metadata, dict):
            for field_name in (
                "source_asset_id",
                "source_origin",
                "source_representation",
                "source_role",
                "source_text_authority",
                "source_timing_authority",
                "supplementary_source",
            ):
                if en_event.metadata.get(field_name) is not None:
                    item[field_name] = en_event.metadata[field_name]
            source_assets = [
                dict(asset)
                for asset in en_event.metadata.get("source_assets") or []
                if isinstance(asset, dict)
            ]
            if source_assets:
                item["source_assets"] = source_assets
        if timing_asset is not None:
            item["timing_asset_id"] = timing_asset.asset_id
            item["timing_authority"] = timing_asset.timing_authority
        if timing_correction:
            item["timing_correction"] = timing_correction
        lane_confidences = {
            lane: event.confidence
            for lane, event in (("en", en_event), ("zh", zh_event))
            if event is not None and event.confidence is not None
        }
        if lane_confidences:
            item["ocr_confidence"] = min(lane_confidences.values())
            item["ocr_lane_confidences"] = lane_confidences
        merged.append(item)
    supplementary_by_id = {
        asset.asset_id: asset
        for asset in supplementary_assets
    }
    for segment in merged:
        if segment.get("supplementary_source"):
            source_asset_ids = {
                str(asset.get("asset_id"))
                for asset in segment.get("source_assets") or []
                if isinstance(asset, dict) and asset.get("asset_id")
            }
            direct_asset_id = str(segment.get("source_asset_id") or "")
            supplementary_asset = supplementary_by_id.get(direct_asset_id)
            if supplementary_asset is None:
                supplementary_asset = next(
                    (
                        asset
                        for asset_id, asset in supplementary_by_id.items()
                        if asset_id in source_asset_ids
                    ),
                    None,
                )
            if supplementary_asset is None and len(supplementary_assets) == 1:
                supplementary_asset = supplementary_assets[0]
            if supplementary_asset is not None:
                if not segment.get("source_asset_id"):
                    attach_asset_to_segments(
                        [segment],
                        supplementary_asset,
                        lane="source",
                    )
                attach_asset_to_segments(
                    [segment],
                    supplementary_asset,
                    lane="supplementary",
                )
        elif (
            segment.get("en")
            and english_asset is not None
            and not segment.get("source_asset_id")
        ):
            attach_asset_to_segments([segment], english_asset, lane="source")
        if segment.get("zh") and chinese_asset is not None:
            attach_asset_to_segments([segment], chinese_asset, lane="chinese")
    return merged


def find_existing_sidecar_subtitle_tracks(
    video_path: Path,
    selected_path: Optional[Path],
    explicit_path: Optional[Path] = None,
    explicit_zh_path: Optional[Path] = None,
    explicit_en_path: Optional[Path] = None,
    chinese_text_authority: str = "authored",
    english_text_authority: str = "authored",
) -> Optional[ExistingSubtitleTracks]:
    candidates = sidecar_candidates(video_path, selected_path, explicit_path, explicit_zh_path, explicit_en_path)
    if not candidates:
        return None

    explicit_candidates = {
        path.resolve()
        for path in (explicit_path, explicit_zh_path, explicit_en_path)
        if path is not None and path.exists()
    }
    zh_path: Optional[Path] = explicit_zh_path if explicit_zh_path and explicit_zh_path.exists() else None
    en_path: Optional[Path] = explicit_en_path if explicit_en_path and explicit_en_path.exists() else None
    if explicit_path is not None and explicit_path.exists():
        explicit_language = classify_subtitle_path(explicit_path)
        if explicit_language == "zh" and zh_path is None:
            zh_path = explicit_path
        elif explicit_language == "en" and en_path is None:
            en_path = explicit_path
    candidate_assets: List[Tuple[Path, SubtitleAsset]] = []
    for candidate in candidates:
        candidate_text_authority = "authored"
        if (
            explicit_en_path is not None
            and candidate.resolve() == explicit_en_path.resolve()
        ):
            candidate_text_authority = english_text_authority
        elif (
            explicit_zh_path is not None
            and candidate.resolve() == explicit_zh_path.resolve()
        ):
            candidate_text_authority = chinese_text_authority
        candidate_asset = build_sidecar_asset(
            video_path,
            candidate,
            likely_match=sidecar_match_score(video_path, candidate) > 0,
            score=sidecar_match_score(video_path, candidate),
            text_authority=candidate_text_authority,
        )
        if (
            candidate.resolve() not in explicit_candidates
            and candidate_asset.role in {"forced", "commentary"}
        ):
            continue
        candidate_assets.append((candidate, candidate_asset))

    automatic_plan = build_processing_plan(
        [asset for _, asset in candidate_assets],
        merge_existing=True,
    )
    assets_by_id = {
        asset.asset_id: path
        for path, asset in candidate_assets
    }
    lanes = automatic_plan.get("lanes") or {}
    if zh_path is None and lanes.get("chinese"):
        zh_path = assets_by_id.get(str(lanes["chinese"]))
    if en_path is None and lanes.get("source"):
        en_path = assets_by_id.get(str(lanes["source"]))

    if zh_path is None:
        return None

    print(f"Using existing Chinese sidecar subtitle: {zh_path}")
    zh_segments = parse_subtitle_file(zh_path)
    zh_asset = build_sidecar_asset(
        video_path,
        zh_path,
        language="zh",
        likely_match=sidecar_match_score(video_path, zh_path) > 0,
        score=sidecar_match_score(video_path, zh_path),
        text_authority=chinese_text_authority,
    )
    for segment in zh_segments:
        segment["source_language"] = "zh"
    attach_asset_to_segments(zh_segments, zh_asset, lane="chinese")
    en_segments: Optional[List[Segment]] = None
    en_asset: Optional[SubtitleAsset] = None
    if en_path and en_path != zh_path:
        print(f"Merging existing English sidecar subtitle: {en_path}")
        en_segments = parse_subtitle_file(en_path)
        en_asset = build_sidecar_asset(
            video_path,
            en_path,
            language="en",
            likely_match=sidecar_match_score(video_path, en_path) > 0,
            score=sidecar_match_score(video_path, en_path),
            text_authority=english_text_authority,
        )
        for segment in en_segments:
            segment["source_language"] = "en"
        attach_asset_to_segments(en_segments, en_asset, lane="source")

    return ExistingSubtitleTracks(
        english=en_segments,
        chinese=zh_segments,
        source_label="existing Chinese sidecar subtitle",
        english_asset=en_asset,
        chinese_asset=zh_asset,
    )


def find_existing_sidecar_subtitle_segments(
    video_path: Path,
    selected_path: Optional[Path],
    explicit_path: Optional[Path] = None,
    explicit_zh_path: Optional[Path] = None,
    explicit_en_path: Optional[Path] = None,
) -> Optional[List[Segment]]:
    tracks = find_existing_sidecar_subtitle_tracks(
        video_path,
        selected_path,
        explicit_path=explicit_path,
        explicit_zh_path=explicit_zh_path,
        explicit_en_path=explicit_en_path,
    )
    if tracks is None:
        return None
    merged = merge_existing_subtitle_segments(
        tracks.english,
        tracks.chinese,
        tracks.source_label,
        english_asset=tracks.english_asset,
        chinese_asset=tracks.chinese_asset,
        supplementary_assets=tracks.supplementary_assets,
    )
    return merged or None


def is_english_subtitle_stream(stream: Any) -> bool:
    label = f"{stream.lang} {stream.title}".lower()
    return stream.lang in {"eng", "en"} or "english" in label


def is_primary_subtitle_stream(stream: Any) -> bool:
    title = str(getattr(stream, "title", "") or "").casefold()
    if bool(getattr(stream, "is_forced", False)):
        return False
    if "forced" in title or "commentary" in title or "director comment" in title:
        return False
    return True


def is_supported_auto_subtitle_stream(stream: Any) -> bool:
    return bool(
        getattr(stream, "is_text_subtitle", False)
        or getattr(stream, "is_pgs", False)
    )


def choose_best_chinese_stream(streams: List[Any]) -> Tuple[Optional[Any], Optional[str]]:
    from subtitle_pipeline import classify_chinese, stream_score

    ranked = []
    for stream in streams:
        if (
            not is_primary_subtitle_stream(stream)
            or not is_supported_auto_subtitle_stream(stream)
        ):
            continue
        zh_kind = classify_chinese(stream)
        if not zh_kind:
            continue
        if zh_kind == "zh-Hans":
            rank = 40
        elif zh_kind == "zh-Hant":
            rank = 30
        elif zh_kind == "zh":
            rank = 20
        else:
            rank = 10
        ranked.append((rank, stream_score(stream), zh_kind, stream))
    if not ranked:
        return None, None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, zh_kind, stream = ranked[0]
    return stream, zh_kind


def choose_best_english_stream(streams: List[Any]) -> Optional[Any]:
    from subtitle_pipeline import stream_score

    english = [
        stream
        for stream in streams
        if is_english_subtitle_stream(stream)
        and is_primary_subtitle_stream(stream)
        and is_supported_auto_subtitle_stream(stream)
    ]
    if not english:
        return None
    english.sort(
        key=lambda stream: (
            bool(getattr(stream, "is_hearing_impaired", False))
            or "sdh" in str(getattr(stream, "title", "") or "").casefold(),
            stream_score(stream),
        ),
        reverse=True,
    )
    return english[0]


def find_existing_embedded_subtitle_tracks(
    video_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    chinese_stream_index: Optional[int] = None,
    english_stream_index: Optional[int] = None,
) -> Optional[ExistingSubtitleTracks]:
    from subtitle_pipeline import classify_chinese, find_ffmpeg, get_stream_events, probe_streams

    ffmpeg = find_ffmpeg(None)
    streams = [stream for stream in probe_streams(video_path, ffmpeg) if stream.is_subtitle]
    if not streams:
        return None

    selected = None
    if args.subtitle_stream is not None:
        matches = [stream for stream in streams if stream.index == args.subtitle_stream]
        if not matches:
            return None
        selected = matches[0]

    zh_stream, zh_kind = choose_best_chinese_stream(streams)
    en_stream = choose_best_english_stream(streams)
    if chinese_stream_index is not None:
        matches = [stream for stream in streams if stream.index == chinese_stream_index]
        if matches:
            zh_stream = matches[0]
            zh_kind = classify_chinese(zh_stream) or "zh"
    if english_stream_index is not None:
        matches = [stream for stream in streams if stream.index == english_stream_index]
        if matches:
            en_stream = matches[0]
    if selected is not None:
        selected_zh_kind = classify_chinese(selected)
        if selected_zh_kind:
            zh_stream, zh_kind = selected, selected_zh_kind
        elif is_english_subtitle_stream(selected):
            en_stream = selected

    if zh_stream is None or zh_kind is None:
        return None

    ocr_lang = args.subtitle_ocr_lang
    if not ocr_lang or ocr_lang == "auto":
        ocr_lang = choose_ocr_lang("zh-hant" if zh_kind in {"zh-Hant", "zh-yue"} else "zh-hans")

    print(
        "Using existing Chinese embedded subtitle stream "
        f"0:{zh_stream.index} {zh_stream.lang or '-'} {zh_stream.codec} {zh_stream.title}"
    )
    zh_asset = build_embedded_asset(video_path, zh_stream)
    zh_events = get_stream_events(
        video_path,
        zh_stream,
        ffmpeg,
        out_dir / "embedded" / f"stream_{zh_stream.index:02d}",
        ocr_lang,
        args,
    )
    zh_segments = [
        {
            "id": idx,
            "start": event.start,
            "end": event.end,
            "text": event.text,
            "source_language": zh_asset.language or "zh",
            **(
                {"ocr_confidence": event.confidence}
                if event.confidence is not None
                else {}
            ),
            **({"ocr_metadata": event.metadata} if event.metadata else {}),
        }
        for idx, event in enumerate(zh_events)
        if event.text.strip()
    ]
    if not zh_segments:
        return None
    attach_asset_to_segments(zh_segments, zh_asset, lane="chinese")

    en_segments: Optional[List[Segment]] = None
    en_asset: Optional[SubtitleAsset] = None
    if en_stream and en_stream.index != zh_stream.index:
        print(
            "Merging existing English embedded subtitle stream "
            f"0:{en_stream.index} {en_stream.lang or '-'} {en_stream.codec} {en_stream.title}"
        )
        en_asset = build_embedded_asset(video_path, en_stream)
        en_events = get_stream_events(
            video_path,
            en_stream,
            ffmpeg,
            out_dir / "embedded" / f"stream_{en_stream.index:02d}",
            "en",
            args,
        )
        en_segments = [
            {
                "id": idx,
                "start": event.start,
                "end": event.end,
                "text": event.text,
                "source_language": en_asset.language or "en",
                **(
                    {"ocr_confidence": event.confidence}
                    if event.confidence is not None
                    else {}
                ),
                **({"ocr_metadata": event.metadata} if event.metadata else {}),
            }
            for idx, event in enumerate(en_events)
            if event.text.strip()
        ]
        attach_asset_to_segments(en_segments, en_asset, lane="source")

    return ExistingSubtitleTracks(
        english=en_segments,
        chinese=zh_segments,
        source_label="existing Chinese embedded subtitle",
        english_asset=en_asset,
        chinese_asset=zh_asset,
    )


def find_existing_embedded_subtitle_segments(
    video_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    chinese_stream_index: Optional[int] = None,
    english_stream_index: Optional[int] = None,
) -> Optional[List[Segment]]:
    tracks = find_existing_embedded_subtitle_tracks(
        video_path,
        out_dir,
        args,
        chinese_stream_index=chinese_stream_index,
        english_stream_index=english_stream_index,
    )
    if tracks is None:
        return None
    merged = merge_existing_subtitle_segments(
        tracks.english,
        tracks.chinese,
        tracks.source_label,
        english_asset=tracks.english_asset,
        chinese_asset=tracks.chinese_asset,
        supplementary_assets=tracks.supplementary_assets,
    )
    return merged or None


def synchronize_existing_subtitle_tracks(
    video_path: Path,
    tracks: ExistingSubtitleTracks,
    *,
    mode: str,
    report_path: Optional[Path] = None,
    audio_stream: Optional[int] = None,
    ffmpeg_path: Optional[str] = None,
) -> Tuple[ExistingSubtitleTracks, Dict[str, Any]]:
    original_chinese = [dict(segment) for segment in tracks.chinese]
    original_english = (
        [dict(segment) for segment in tracks.english]
        if tracks.english is not None
        else None
    )

    synchronized_chinese, chinese_report = synchronize_subtitle_segments(
        video_path,
        original_chinese,
        mode=mode,
        report_path=None,
        audio_stream=audio_stream,
        ffmpeg_path=ffmpeg_path,
    )
    synchronized_english: Optional[List[Segment]] = None
    english_report: Optional[Dict[str, Any]] = None
    if original_english:
        synchronized_english, english_report = synchronize_subtitle_segments(
            video_path,
            original_english,
            mode=mode,
            report_path=None,
            audio_stream=audio_stream,
            ffmpeg_path=ffmpeg_path,
        )

    reports = [chinese_report]
    if english_report is not None:
        reports.append(english_report)
    quality_passed = mode == "off" or all(
        bool(report.get("pipeline_quality_passed"))
        for report in reports
    )
    applied = (
        mode == "auto"
        and quality_passed
        and any(bool(report.get("applied")) for report in reports)
    )

    if mode == "auto" and quality_passed:
        selected_chinese = synchronized_chinese
        selected_english = synchronized_english
    else:
        selected_chinese = original_chinese
        selected_english = original_english

    if mode == "off":
        status = "disabled"
    elif mode == "detect":
        status = "detected" if quality_passed else "detected_with_warnings"
    elif quality_passed:
        status = "aligned" if applied else "already_aligned"
    else:
        status = "rejected_partial" if english_report is not None else "rejected"

    report: Dict[str, Any] = {
        "version": SYNC_POLICY_VERSION,
        "mode": mode,
        "backend": "ffsubsync",
        "scope": "per_track",
        "video": str(video_path),
        "audio_stream": audio_stream,
        "event_count": {
            "chinese": len(original_chinese),
            "english": len(original_english or []),
        },
        "status": status,
        "applied": applied,
        "pipeline_quality_passed": quality_passed,
        "reverted_all_tracks": mode == "auto" and not quality_passed,
        "tracks": {
            "chinese": chinese_report,
            "english": english_report,
        },
    }
    if report_path is not None:
        write_json_atomic(report_path, report)

    return (
        ExistingSubtitleTracks(
            english=selected_english,
            chinese=selected_chinese,
            source_label=tracks.source_label,
            english_asset=tracks.english_asset,
            chinese_asset=tracks.chinese_asset,
            supplementary_assets=tracks.supplementary_assets,
        ),
        report,
    )


def choose_ocr_lang(language: str) -> str:
    lang = (language or "").lower()
    if lang in {"zh", "zho", "chi", "chs", "zh-cn", "zh-hans", "cmn"}:
        return "ch"
    if lang in {"cht", "zh-tw", "zh-hant"}:
        return "chinese_cht"
    if lang in {"ja", "jpn", "japanese"}:
        return "japan"
    if lang in {"ko", "kor", "korean"}:
        return "korean"
    if lang in {"fr", "fra", "fre", "french"}:
        return "fr"
    if lang in {"de", "deu", "ger", "german"}:
        return "german"
    return "en"


def resolve_asr_language(asr_language: str, source_language: str) -> str:
    requested = (asr_language or "").strip().lower()
    source = (source_language or "").strip().lower()
    if requested in {"", "source", "follow-source", "same"}:
        return source if source and source != "auto" else "auto"
    return requested


def load_embedded_subtitle_events(
    video_path: Path,
    stream_index: Optional[int],
    source_language: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> List[Segment]:
    from subtitle_pipeline import find_ffmpeg, get_stream_events, probe_streams, stream_score

    ffmpeg = find_ffmpeg(None)
    streams = [stream for stream in probe_streams(video_path, ffmpeg) if stream.is_subtitle]
    if not streams:
        raise RuntimeError("No embedded subtitle stream was found.")

    if stream_index is not None:
        matches = [stream for stream in streams if stream.index == stream_index]
        if not matches:
            raise RuntimeError(f"Embedded subtitle stream 0:{stream_index} was not found.")
        stream = matches[0]
    else:
        pool = [
            stream
            for stream in streams
            if is_primary_subtitle_stream(stream)
            and is_supported_auto_subtitle_stream(stream)
        ]
        if not pool:
            raise RuntimeError(
                "Only forced/commentary subtitle streams were found; no full-dialogue "
                "embedded subtitle is safe for automatic selection."
            )
        plan = build_processing_plan(
            [build_embedded_asset(video_path, candidate) for candidate in pool],
            preferred_source_language=source_language,
            merge_existing=True,
        )
        selected_asset_id = (plan.get("lanes") or {}).get("source")
        selected_by_id = {
            build_embedded_asset(video_path, candidate).asset_id: candidate
            for candidate in pool
        }
        stream = selected_by_id.get(str(selected_asset_id))
        if stream is None:
            pool.sort(key=stream_score, reverse=True)
            stream = pool[0]

    language_hint = source_language if source_language and source_language != "auto" else stream.lang
    ocr_lang = args.subtitle_ocr_lang
    if not ocr_lang or ocr_lang == "auto":
        ocr_lang = choose_ocr_lang(language_hint)

    print(f"Using embedded subtitle stream 0:{stream.index} {stream.lang or '-'} {stream.codec} {stream.title}")
    asset = build_embedded_asset(video_path, stream)
    events = get_stream_events(video_path, stream, ffmpeg, out_dir / "embedded" / f"stream_{stream.index:02d}", ocr_lang, args)
    segments = [
        {
            "id": idx,
            "start": event.start,
            "end": event.end,
            "text": event.text,
            "source_language": asset.language or stream.lang or source_language,
            **(
                {"ocr_confidence": event.confidence}
                if event.confidence is not None
                else {}
            ),
            **({"ocr_metadata": event.metadata} if event.metadata else {}),
        }
        for idx, event in enumerate(events)
        if event.text.strip()
    ]
    if not segments:
        raise RuntimeError(f"Embedded subtitle stream 0:{stream.index} did not produce usable subtitle events.")
    attach_asset_to_segments(segments, asset, lane="source")
    return segments


def discover_automatic_source_assets(
    video_path: Path,
    selected_path: Optional[Path],
) -> List[SubtitleAsset]:
    assets = [
        build_sidecar_asset(
            video_path,
            path,
            likely_match=sidecar_match_score(video_path, path) > 0,
            score=sidecar_match_score(video_path, path),
        )
        for path in likely_sidecar_subtitles(video_path, selected_path)
    ]
    try:
        from subtitle_pipeline import find_ffmpeg, probe_streams

        ffmpeg = find_ffmpeg(None)
        for stream in probe_streams(video_path, ffmpeg):
            if stream.is_subtitle:
                assets.append(build_embedded_asset(video_path, stream))
            elif stream.is_audio:
                assets.append(build_audio_asset(video_path, stream))
    except Exception as exc:
        print(f"Automatic source inventory could not read embedded streams: {exc}")
    return assets


def load_best_sidecar_source_track(
    video_path: Path,
    selected_path: Optional[Path],
    *,
    preferred_source_language: str = "auto",
) -> Tuple[Optional[List[Segment]], Optional[SubtitleAsset]]:
    candidates = likely_sidecar_subtitles(video_path, selected_path)
    candidate_assets = [
        (
            path,
            build_sidecar_asset(
                video_path,
                path,
                likely_match=sidecar_match_score(video_path, path) > 0,
                score=sidecar_match_score(video_path, path),
            ),
        )
        for path in candidates
    ]
    plan = build_processing_plan(
        [asset for _, asset in candidate_assets],
        preferred_source_language=preferred_source_language,
        merge_existing=True,
    )
    source_asset_id = (plan.get("lanes") or {}).get("source")
    selected = next(
        (
            (path, asset)
            for path, asset in candidate_assets
            if asset.asset_id == source_asset_id
        ),
        None,
    )
    if selected is None:
        return None, None
    path, asset = selected
    segments = parse_subtitle_file(path)
    for segment in segments:
        segment["source_language"] = asset.language or preferred_source_language
    attach_asset_to_segments(segments, asset, lane="source")
    return segments or None, asset


def load_best_embedded_source_track(
    video_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    preferred_source_language: str = "auto",
) -> Tuple[Optional[List[Segment]], Optional[SubtitleAsset]]:
    try:
        from subtitle_pipeline import find_ffmpeg, probe_streams

        ffmpeg = find_ffmpeg(None)
        streams = [
            stream
            for stream in probe_streams(video_path, ffmpeg)
            if stream.is_subtitle
            and is_primary_subtitle_stream(stream)
            and is_supported_auto_subtitle_stream(stream)
        ]
    except Exception as exc:
        print(f"Unable to inspect embedded source tracks for cross-source merge: {exc}")
        return None, None
    assets_by_id = {
        build_embedded_asset(video_path, stream).asset_id: (
            stream,
            build_embedded_asset(video_path, stream),
        )
        for stream in streams
    }
    plan = build_processing_plan(
        [asset for _, asset in assets_by_id.values()],
        preferred_source_language=preferred_source_language,
        merge_existing=True,
    )
    source_asset_id = (plan.get("lanes") or {}).get("source")
    selected = assets_by_id.get(str(source_asset_id))
    if selected is None:
        return None, None
    stream, asset = selected
    segments = load_embedded_subtitle_events(
        video_path,
        stream_index=stream.index,
        source_language=asset.language or preferred_source_language,
        out_dir=out_dir,
        args=args,
    )
    return segments or None, asset


def add_cross_origin_source_track(
    tracks: ExistingSubtitleTracks,
    video_path: Path,
    selected_path: Optional[Path],
    out_dir: Path,
    args: argparse.Namespace,
    *,
    preferred_source_language: str = "auto",
) -> ExistingSubtitleTracks:
    if tracks.english:
        return tracks
    chinese_origin = tracks.chinese_asset.origin if tracks.chinese_asset else ""
    if chinese_origin == "embedded":
        source_segments, source_asset = load_best_sidecar_source_track(
            video_path,
            selected_path,
            preferred_source_language=preferred_source_language,
        )
    else:
        source_segments, source_asset = load_best_embedded_source_track(
            video_path,
            out_dir,
            args,
            preferred_source_language=preferred_source_language,
        )
    if not source_segments or source_asset is None:
        return tracks
    print(
        "Merging cross-origin source subtitle track: "
        f"{source_asset.origin} {source_asset.label}"
    )
    return ExistingSubtitleTracks(
        english=source_segments,
        chinese=tracks.chinese,
        source_label=f"{tracks.source_label} + {source_asset.origin} source subtitle",
        english_asset=source_asset,
        chinese_asset=tracks.chinese_asset,
        supplementary_assets=tracks.supplementary_assets,
    )


def _source_event_key(text: str) -> str:
    return re.sub(
        r"[\W_]+",
        "",
        clean_subtitle_text(text).casefold(),
        flags=re.UNICODE,
    )


def merge_supplementary_source_segments(
    primary: Optional[List[Segment]],
    supplementary: List[Segment],
) -> List[Segment]:
    merged = [dict(segment) for segment in (primary or [])]
    for extra in supplementary:
        extra_key = _source_event_key(str(extra.get("text") or ""))
        if not extra_key:
            continue
        extra_start = float(extra["start"])
        extra_end = float(extra["end"])
        extra_duration = max(0.01, extra_end - extra_start)
        duplicate = False
        for existing in merged:
            existing_key = _source_event_key(str(existing.get("text") or ""))
            if not existing_key:
                continue
            overlap = max(
                0.0,
                min(extra_end, float(existing["end"]))
                - max(extra_start, float(existing["start"])),
            )
            overlap_ratio = overlap / min(
                extra_duration,
                max(0.01, float(existing["end"]) - float(existing["start"])),
            )
            text_ratio = min(len(extra_key), len(existing_key)) / max(
                len(extra_key),
                len(existing_key),
            )
            same_text = (
                extra_key == existing_key
                or (
                    text_ratio >= 0.85
                    and (
                        extra_key in existing_key
                        or existing_key in extra_key
                    )
                )
            )
            if overlap_ratio >= 0.5 and same_text:
                duplicate = True
                break
        if duplicate:
            continue
        merged.append(
            {
                **extra,
                "supplementary_source": True,
            }
        )
    merged.sort(
        key=lambda segment: (
            float(segment["start"]),
            float(segment["end"]),
            str(segment.get("text") or ""),
        )
    )
    for index, segment in enumerate(merged):
        segment["id"] = index
    return merged


def load_subtitle_asset_segments(
    asset: SubtitleAsset,
    video_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Optional[List[Segment]]:
    if asset.origin == "sidecar" and asset.path:
        path = Path(asset.path)
        if not path.exists():
            return None
        segments = parse_subtitle_file(path)
        for segment in segments:
            segment["source_language"] = asset.language or args.source_language
        attach_asset_to_segments(segments, asset, lane="source")
        return segments or None
    if asset.origin == "embedded" and asset.stream_index is not None:
        return load_embedded_subtitle_events(
            video_path,
            stream_index=asset.stream_index,
            source_language=asset.language or args.source_language,
            out_dir=out_dir,
            args=args,
        )
    return None


def merge_planned_supplementary_segments(
    primary: Optional[List[Segment]],
    assets: List[SubtitleAsset],
    plan: Optional[Dict[str, Any]],
    video_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[Segment], Tuple[SubtitleAsset, ...]]:
    supplementary_ids = set(
        (plan or {}).get("lanes", {}).get("supplementary") or []
    )
    selected_assets = [
        asset
        for asset in assets
        if asset.asset_id in supplementary_ids
        and normalize_language(asset.language) != "zh"
        and asset.role in {"forced", "sdh"}
    ]
    if not selected_assets:
        return [dict(segment) for segment in (primary or [])], ()

    merged = [dict(segment) for segment in (primary or [])]
    loaded_assets: List[SubtitleAsset] = []
    for asset in selected_assets:
        segments = load_subtitle_asset_segments(
            asset,
            video_path,
            out_dir,
            args,
        )
        if not segments:
            continue
        merged = merge_supplementary_source_segments(merged, segments)
        loaded_assets.append(asset)
        print(
            "Merged supplementary subtitle track: "
            f"{asset.origin} {asset.label}"
        )
    return merged, tuple(loaded_assets)


def add_supplementary_source_tracks(
    tracks: ExistingSubtitleTracks,
    assets: List[SubtitleAsset],
    plan: Optional[Dict[str, Any]],
    video_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> ExistingSubtitleTracks:
    english, loaded_assets = merge_planned_supplementary_segments(
        tracks.english,
        assets,
        plan,
        video_path,
        out_dir,
        args,
    )
    if not loaded_assets:
        return tracks

    supplementary_assets = {
        asset.asset_id: asset
        for asset in (*tracks.supplementary_assets, *loaded_assets)
    }
    return ExistingSubtitleTracks(
        english=english,
        chinese=tracks.chinese,
        source_label=f"{tracks.source_label} + supplementary source cues",
        english_asset=tracks.english_asset or loaded_assets[0],
        chinese_asset=tracks.chinese_asset,
        supplementary_assets=tuple(supplementary_assets.values()),
    )


def join_words(words: List[Dict[str, Any]]) -> str:
    text = " ".join(word["word"].strip() for word in words if word.get("word", "").strip())
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def split_plain_text(text: str, max_words: int, max_chars: int, min_chunks: int = 1) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) == 1 and contains_cjk(text):
        target_chunks = max(1, min_chunks, math.ceil(len(text) / max_chars))
        return split_text_balanced(text, target_chunks, joiner="")

    target_chunks = max(1, min_chunks)
    dynamic_max_words = max(3, min(max_words, math.ceil(len(words) / target_chunks)))
    chunks: List[str] = []
    current: List[str] = []

    def current_text() -> str:
        return normalize_space(" ".join(current))

    for word in words:
        current.append(word)
        text_now = current_text()
        sentence_end = bool(re.search(r"[.!?]$", word))
        hard_limit = len(current) >= dynamic_max_words or len(text_now) >= max_chars
        enough_for_target = len(chunks) + 1 < target_chunks and len(current) >= dynamic_max_words
        if sentence_end or hard_limit or enough_for_target:
            chunks.append(text_now)
            current = []

    if current:
        chunks.append(current_text())

    return [chunk for chunk in chunks if chunk]


def compact_split_units(text: str) -> List[str]:
    return re.findall(
        r"[\(\uff08][A-Za-z0-9][A-Za-z0-9 .'%:/+\-]{0,30}[\)\uff09]"
        r"|[A-Za-z0-9]+(?:[.'\u2019:/+\-][A-Za-z0-9]+)*%?[ \t]*"
        r"|[ \t]+|.",
        text,
        flags=re.DOTALL,
    )


def chinese_split_units(text: str) -> List[str]:
    tokens = [token for token in jieba.lcut(text, cut_all=False, HMM=True) if token]
    if "".join(tokens) != text:
        return compact_split_units(text)

    merged: List[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] in {"(", "（"}:
            closing = ")" if tokens[index] == "(" else "）"
            closing_index = next(
                (
                    candidate
                    for candidate in range(index + 1, min(len(tokens), index + 8))
                    if tokens[candidate] == closing
                ),
                -1,
            )
            if closing_index > index:
                candidate = "".join(tokens[index : closing_index + 1])
                if re.fullmatch(
                    r"[\(（][A-Za-z0-9][A-Za-z0-9 .'%:/+\-]{0,30}[\)）]",
                    candidate,
                ):
                    merged.append(candidate)
                    index = closing_index + 1
                    continue
        if (
            index + 2 < len(tokens)
            and tokens[index + 1] in {"·", "・"}
            and re.search(r"[\w\u3400-\u4dbf\u4e00-\u9fff]$", tokens[index])
            and re.match(r"^[\w\u3400-\u4dbf\u4e00-\u9fff]", tokens[index + 2])
        ):
            merged.append(tokens[index] + tokens[index + 1] + tokens[index + 2])
            index += 3
            continue
        merged.append(tokens[index])
        index += 1
    return merged


def split_text_balanced(
    text: str,
    chunk_count: int,
    joiner: str,
    *,
    max_chars: Optional[int] = None,
    max_words: Optional[int] = None,
) -> List[str]:
    use_chinese_words = (
        joiner == ""
        and contains_cjk(text)
        and not JAPANESE_SCRIPT_RE.search(text)
        and not KOREAN_SCRIPT_RE.search(text)
    )
    units = (
        chinese_split_units(text)
        if use_chinese_words
        else (
            compact_split_units(text)
            if joiner == ""
            else re.findall(r"[\[【][^\]】]+[\]】]|\S+", text)
        )
    )
    if not units:
        return [""] * max(1, chunk_count)

    chunk_count = max(1, min(chunk_count, len(units)))
    if chunk_count == 1:
        return [joiner.join(units).strip()]

    strong_punctuation = "。！？.!?"
    weak_punctuation = "，、；：,;:"
    no_break_after = {
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
    }
    chinese_no_break_after = {
        "与",
        "为",
        "从",
        "但",
        "到",
        "及",
        "和",
        "在",
        "将",
        "并",
        "或",
        "把",
        "由",
        "给",
        "而",
        "被",
        "让",
    }
    chinese_no_break_before = {
        "了",
        "吗",
        "吧",
        "呢",
        "呀",
        "啊",
        "地",
        "得",
        "的",
        "着",
        "过",
    }
    full_text = joiner.join(units).strip()
    target_length = max(1.0, len(full_text) / chunk_count)
    states: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]] = {
        (0, 0): (0.0, (0,))
    }
    for part_index in range(1, chunk_count + 1):
        minimum_end = part_index
        maximum_end = len(units) - (chunk_count - part_index)
        for end in range(minimum_end, maximum_end + 1):
            best: Optional[Tuple[float, Tuple[int, ...]]] = None
            for start in range(part_index - 1, end):
                previous_state = states.get((part_index - 1, start))
                if previous_state is None:
                    continue
                chunk = joiner.join(units[start:end]).strip()
                if not chunk:
                    continue
                bracketed_label = bool(
                    re.fullmatch(r"[\[【][^\]】]+[\]】]", chunk)
                )
                if (
                    max_chars is not None
                    and len(chunk) > max_chars
                    and not bracketed_label
                ):
                    continue
                if (
                    max_words is not None
                    and joiner
                    and len(chunk.split()) > max_words
                    and not bracketed_label
                ):
                    continue

                score = previous_state[0] + (len(chunk) - target_length) ** 2
                if end < len(units):
                    previous_unit = units[end - 1]
                    next_unit = units[end]
                    punctuation_weight = target_length * target_length
                    if previous_unit.endswith(tuple(strong_punctuation)):
                        score -= punctuation_weight * 0.8
                    elif previous_unit.endswith(tuple(weak_punctuation)):
                        score -= punctuation_weight * 0.35
                    if (
                        previous_unit.rstrip().endswith(("]", "】"))
                        and start == 0
                    ):
                        score += punctuation_weight * 0.5
                    if joiner:
                        normalized_previous = previous_unit.rstrip(".,;:!?").casefold()
                        if normalized_previous in no_break_after:
                            score += punctuation_weight * 1.5
                    else:
                        if previous_unit.strip() in chinese_no_break_after:
                            score += punctuation_weight * 1.5
                        if next_unit.strip() in chinese_no_break_before:
                            score += punctuation_weight * 1.5
                        if previous_unit in "（《「『【“‘":
                            score += punctuation_weight
                        if next_unit in "），。！？；：》」』】”’":
                            score += punctuation_weight
                        prefix = "".join(units[:end])
                        suffix = "".join(units[end:])
                        for opening, closing in (
                            ("《", "》"),
                            ("「", "」"),
                            ("『", "』"),
                            ("【", "】"),
                            ("（", "）"),
                            ("(", ")"),
                            ("[", "]"),
                        ):
                            if (
                                prefix.count(opening) > prefix.count(closing)
                                and closing in suffix
                            ):
                                score += punctuation_weight * 4.0
                candidate = (
                    score,
                    (*previous_state[1], end),
                )
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                states[(part_index, end)] = best

    final_state = states.get((chunk_count, len(units)))
    if final_state is None and (max_chars is not None or max_words is not None):
        return split_text_balanced(text, chunk_count, joiner)
    boundaries = list(
        final_state[1]
        if final_state is not None
        else tuple(round(len(units) * index / chunk_count) for index in range(chunk_count + 1))
    )

    chunks: List[str] = []
    for index in range(chunk_count):
        chunk = joiner.join(units[boundaries[index] : boundaries[index + 1]]).strip()
        chunks.append(chunk)
    return chunks


def track_chunk_count(text: str, max_words: int, max_chars: int, is_cjk: bool) -> int:
    if not text:
        return 1
    if is_cjk:
        return max(1, math.ceil(len(text) / max_chars))
    return max(
        1,
        math.ceil(len(text.split()) / max_words),
        math.ceil(len(text) / max_chars),
    )


def is_bracketed_label(text: str) -> bool:
    return bool(re.fullmatch(r"[\[【][^\]】]+[\]】]", text.strip()))


SEMANTIC_SENTENCE_ENDINGS = ".!?;。！？；"
SEMANTIC_CLOSING_MARKS = "\"'”’）)]】"
ENGLISH_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "st.",
    "vs.",
    "etc.",
    "u.k.",
    "u.s.",
}


def split_semantic_sentences(text: str) -> List[str]:
    cleaned = clean_subtitle_text(text)
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    index = 0
    bracket_depth = 0
    while index < len(cleaned):
        character = cleaned[index]
        if character in "[【":
            bracket_depth += 1
        elif character in "]】" and bracket_depth:
            bracket_depth -= 1
        if bracket_depth or character not in SEMANTIC_SENTENCE_ENDINGS:
            index += 1
            continue

        boundary_end = index + 1
        if character == ".":
            while (
                boundary_end < len(cleaned)
                and cleaned[boundary_end] == "."
            ):
                boundary_end += 1
            candidate = cleaned[start:boundary_end].strip()
            previous_token = (
                candidate.split()[-1].casefold()
                if candidate.split()
                else ""
            )
            decimal_point = (
                index > 0
                and index + 1 < len(cleaned)
                and cleaned[index - 1].isdigit()
                and cleaned[index + 1].isdigit()
            )
            if previous_token in ENGLISH_ABBREVIATIONS or decimal_point:
                index += 1
                continue

        while (
            boundary_end < len(cleaned)
            and cleaned[boundary_end] in SEMANTIC_CLOSING_MARKS
        ):
            boundary_end += 1
        candidate = cleaned[start:boundary_end].strip()
        if not re.search(r"[A-Za-z0-9\u3400-\u9fff]", candidate):
            index = boundary_end
            continue
        if cleaned[boundary_end:].strip():
            chunks.append(candidate)
            start = boundary_end
        index = boundary_end

    tail = cleaned[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks or [cleaned]


def split_aligned_bilingual_sentences(
    en_text: str,
    zh_text: str,
    target_chunk_count: int,
    max_chunk_count: int,
    *,
    max_words: int,
    en_max_chars: int,
    zh_max_chars: int,
    en_is_compact: bool,
    layout_policy: Optional[SubtitleLayoutPolicy],
) -> Optional[Tuple[List[str], List[str]]]:
    en_sentences = split_semantic_sentences(en_text)
    zh_sentences = split_semantic_sentences(zh_text)
    if (
        len(en_sentences) <= 1
        or len(en_sentences) != len(zh_sentences)
    ):
        return None

    bilingual = True
    pair_chunk_counts: List[int] = []
    for en_sentence, zh_sentence in zip(en_sentences, zh_sentences):
        required = max(
            track_chunk_count(
                en_sentence,
                max_words,
                en_max_chars,
                is_cjk=en_is_compact,
            ),
            track_chunk_count(
                zh_sentence,
                max_words,
                zh_max_chars,
                is_cjk=True,
            ),
        )
        if layout_policy is not None:
            required = max(
                required,
                layout_policy.required_chunks(
                    en_sentence,
                    "en",
                    bilingual=bilingual,
                ),
                layout_policy.required_chunks(
                    zh_sentence,
                    "zh",
                    bilingual=bilingual,
                ),
            )
        pair_chunk_counts.append(required)

    if sum(pair_chunk_counts) > max_chunk_count:
        return None
    while sum(pair_chunk_counts) < target_chunk_count:
        candidates = [
            (
                max(
                    len(en_sentences[index]) / pair_chunk_counts[index],
                    len(zh_sentences[index]) / pair_chunk_counts[index],
                ),
                index,
            )
            for index in range(len(pair_chunk_counts))
        ]
        _, selected = max(candidates)
        pair_chunk_counts[selected] += 1
        if sum(pair_chunk_counts) >= max_chunk_count:
            break

    en_chunks: List[str] = []
    zh_chunks: List[str] = []
    for en_sentence, zh_sentence, chunk_count in zip(
        en_sentences,
        zh_sentences,
        pair_chunk_counts,
    ):
        en_chunks.extend(
            split_text_balanced(
                en_sentence,
                chunk_count,
                joiner="" if en_is_compact else " ",
                max_chars=en_max_chars,
                max_words=None if en_is_compact else max_words,
            )
        )
        zh_chunks.extend(
            split_text_balanced(
                zh_sentence,
                chunk_count,
                joiner="",
                max_chars=zh_max_chars,
            )
        )
    if len(en_chunks) != len(zh_chunks):
        return None
    return en_chunks, zh_chunks


BRACKETED_SECTION_RE = re.compile(r"[\[【][^\]】]+[\]】]")


def split_bracketed_sections(text: str) -> List[Tuple[str, str]]:
    cleaned = clean_subtitle_text(text)
    matches = list(BRACKETED_SECTION_RE.finditer(cleaned))
    if not matches:
        return [("", cleaned)] if cleaned else []

    sections: List[Tuple[str, str]] = []
    prefix = cleaned[: matches[0].start()].strip()
    if prefix:
        sections.append(("", prefix))
    for index, match in enumerate(matches):
        body_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(cleaned)
        )
        sections.append(
            (
                match.group(0).strip(),
                cleaned[match.end() : body_end].strip(),
            )
        )
    return sections


def labeled_chunk_text(label: str, body: str, joiner: str) -> str:
    if not label:
        return body
    if not body:
        return label
    return f"{label}{joiner}{body}"


def aligned_chunk_fits(
    chunk: str,
    *,
    max_chars: int,
    max_words: Optional[int],
    layout_policy: Optional[SubtitleLayoutPolicy],
    track: str,
) -> bool:
    if not chunk or is_bracketed_label(chunk):
        return True
    leading_label = re.match(
        r"^[\[【][^\]】]+[\]】]\s*(?P<body>.+)$",
        chunk,
    )
    measured_text = (
        leading_label.group("body").strip()
        if leading_label is not None
        else chunk
    )
    if len(measured_text) > max_chars:
        return False
    if max_words is not None and len(measured_text.split()) > max_words:
        return False
    return layout_policy is None or layout_policy.fits(
        chunk,
        track,
        bilingual=True,
    )


def split_aligned_bilingual_sections(
    en_text: str,
    zh_text: str,
    target_chunk_count: int,
    max_chunk_count: int,
    *,
    max_words: int,
    en_max_chars: int,
    zh_max_chars: int,
    en_is_compact: bool,
    layout_policy: Optional[SubtitleLayoutPolicy],
) -> Optional[Tuple[List[str], List[str]]]:
    en_sections = split_bracketed_sections(en_text)
    zh_sections = split_bracketed_sections(zh_text)
    if (
        not any(label for label, _body in en_sections)
        or len(en_sections) != len(zh_sections)
        or any(
            bool(en_label) != bool(zh_label)
            for (en_label, _en_body), (zh_label, _zh_body) in zip(
                en_sections,
                zh_sections,
            )
        )
    ):
        return None

    section_chunk_counts: List[int] = []
    for (en_label, en_body), (zh_label, zh_body) in zip(
        en_sections,
        zh_sections,
    ):
        en_full = labeled_chunk_text(
            en_label,
            en_body,
            "" if en_is_compact else " ",
        )
        zh_full = labeled_chunk_text(zh_label, zh_body, "")
        required = max(
            track_chunk_count(
                en_full,
                max_words,
                en_max_chars,
                is_cjk=en_is_compact,
            ),
            track_chunk_count(
                zh_full,
                max_words,
                zh_max_chars,
                is_cjk=True,
            ),
        )
        if layout_policy is not None:
            required = max(
                required,
                layout_policy.required_chunks(
                    en_full,
                    "en",
                    bilingual=True,
                ),
                layout_policy.required_chunks(
                    zh_full,
                    "zh",
                    bilingual=True,
                ),
            )
        section_chunk_counts.append(required)

    if sum(section_chunk_counts) > max_chunk_count:
        return None
    while sum(section_chunk_counts) < target_chunk_count:
        candidates = [
            (
                max(
                    len(en_body) / section_chunk_counts[index],
                    len(zh_body) / section_chunk_counts[index],
                ),
                index,
            )
            for index, (
                (_en_label, en_body),
                (_zh_label, zh_body),
            ) in enumerate(zip(en_sections, zh_sections))
            if en_body or zh_body
        ]
        if not candidates:
            break
        _, selected = max(candidates)
        section_chunk_counts[selected] += 1
        if sum(section_chunk_counts) >= max_chunk_count:
            break

    en_chunks: List[str] = []
    zh_chunks: List[str] = []
    for (
        (en_label, en_body),
        (zh_label, zh_body),
        chunk_count,
    ) in zip(en_sections, zh_sections, section_chunk_counts):
        body_chunks = split_aligned_bilingual_sentences(
            en_body,
            zh_body,
            chunk_count,
            chunk_count,
            max_words=max_words,
            en_max_chars=en_max_chars,
            zh_max_chars=zh_max_chars,
            en_is_compact=en_is_compact,
            layout_policy=layout_policy,
        )
        if body_chunks is not None and len(body_chunks[0]) == chunk_count:
            section_en, section_zh = body_chunks
        else:
            section_en = (
                split_text_balanced(
                    en_body,
                    chunk_count,
                    joiner="" if en_is_compact else " ",
                    max_chars=en_max_chars,
                    max_words=None if en_is_compact else max_words,
                )
                if en_body
                else [""] * chunk_count
            )
            section_zh = (
                split_text_balanced(
                    zh_body,
                    chunk_count,
                    joiner="",
                    max_chars=zh_max_chars,
                )
                if zh_body
                else [""] * chunk_count
            )
        section_en += [""] * (chunk_count - len(section_en))
        section_zh += [""] * (chunk_count - len(section_zh))
        section_en[0] = labeled_chunk_text(
            en_label,
            section_en[0],
            "" if en_is_compact else " ",
        )
        section_zh[0] = labeled_chunk_text(
            zh_label,
            section_zh[0],
            "",
        )

        attached_fits = all(
            aligned_chunk_fits(
                chunk,
                max_chars=en_max_chars,
                max_words=None if en_is_compact else max_words,
                layout_policy=layout_policy,
                track="en",
            )
            for chunk in section_en
        ) and all(
            aligned_chunk_fits(
                chunk,
                max_chars=zh_max_chars,
                max_words=None,
                layout_policy=layout_policy,
                track="zh",
            )
            for chunk in section_zh
        )
        if not attached_fits:
            if not en_label or chunk_count < 2:
                return None
            remaining_count = chunk_count - 1
            section_en = [en_label, *split_text_balanced(
                en_body,
                remaining_count,
                joiner="" if en_is_compact else " ",
                max_chars=en_max_chars,
                max_words=None if en_is_compact else max_words,
            )]
            section_zh = [zh_label, *split_text_balanced(
                zh_body,
                remaining_count,
                joiner="",
                max_chars=zh_max_chars,
            )]
        en_chunks.extend(section_en)
        zh_chunks.extend(section_zh)

    if len(en_chunks) != len(zh_chunks):
        return None
    return en_chunks, zh_chunks


def valid_word_timestamps(segment: Segment) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for word in segment.get("words") or []:
        if not isinstance(word, dict) or not str(word.get("word") or "").strip():
            continue
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            words.append({**word, "start": start, "end": end})
    return sorted(words, key=lambda item: (float(item["start"]), float(item["end"])))


def allocate_words_to_chunks(
    words: List[Dict[str, Any]],
    chunks: List[str],
    *,
    compact_text: bool,
    min_duration: float = MIN_SUBTITLE_DURATION,
    max_duration: Optional[float] = None,
) -> List[List[Dict[str, Any]]]:
    if len(chunks) <= 1:
        return [words]
    if len(words) < len(chunks):
        return []
    weights = [
        max(1, len(chunk) if compact_text else len(chunk.split()))
        for chunk in chunks
    ]
    total_weight = sum(weights)
    cumulative_weights = [0]
    for weight in weights:
        cumulative_weights.append(cumulative_weights[-1] + weight)

    states: Dict[Tuple[int, int], Tuple[float, Tuple[int, ...]]] = {
        (0, 0): (0.0, (0,))
    }
    for chunk_index in range(1, len(chunks) + 1):
        minimum_end = chunk_index
        maximum_end = len(words) - (len(chunks) - chunk_index)
        expected_words = len(words) * weights[chunk_index - 1] / total_weight
        ideal_boundary = (
            len(words)
            * cumulative_weights[chunk_index]
            / total_weight
        )
        for end in range(minimum_end, maximum_end + 1):
            best: Optional[Tuple[float, Tuple[int, ...]]] = None
            for start in range(chunk_index - 1, end):
                previous = states.get((chunk_index - 1, start))
                if previous is None:
                    continue
                duration = (
                    float(words[end - 1]["end"])
                    - float(words[start]["start"])
                )
                score = previous[0]
                score += ((end - start) - expected_words) ** 2 * 2.0
                score += (end - ideal_boundary) ** 2 * 0.25
                if duration < min_duration:
                    score += (min_duration - duration) ** 2 * 1000.0
                if max_duration is not None and duration > max_duration:
                    score += (duration - max_duration) ** 2 * 2.0
                if end < len(words):
                    pause = max(
                        0.0,
                        float(words[end]["start"])
                        - float(words[end - 1]["end"]),
                    )
                    score -= min(pause, 3.0) ** 2 * 12.0
                    if re.search(
                        r"[.!?。！？;；]$",
                        str(words[end - 1].get("word") or "").strip(),
                    ):
                        score -= 8.0
                candidate = (score, (*previous[1], end))
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                states[(chunk_index, end)] = best

    final_state = states.get((len(chunks), len(words)))
    if final_state is None:
        return []
    boundaries = list(final_state[1])
    return [
        words[boundaries[index] : boundaries[index + 1]]
        for index in range(len(chunks))
    ]


def split_bilingual_segment(
    segment: Segment,
    max_words: int,
    max_chars: int,
    max_duration: float,
    layout_policy: Optional[SubtitleLayoutPolicy] = None,
) -> List[Segment]:
    en_text = source_track_text(segment)
    zh_text = to_simplified_text(clean_subtitle_text(str(segment.get("zh") or "")))
    source_text = clean_subtitle_text(str(segment.get("text") or en_text or zh_text))
    start = float(segment["start"])
    end = float(segment["end"])
    duration = end - start
    en_is_compact = contains_east_asian(en_text) and len(en_text.split()) <= 1
    en_max_chars = (
        min(max_chars, CHINESE_CHARS_PER_LINE)
        if en_is_compact
        else min(max_chars, ENGLISH_CHARS_PER_LINE)
    )
    zh_max_chars = min(max_chars, CHINESE_CHARS_PER_LINE)
    source_is_en = bool(en_text) and source_text == en_text
    source_is_zh = bool(zh_text) and source_text == zh_text
    source_is_cjk = contains_east_asian(source_text) and len(source_text.split()) <= 1
    source_char_limit = zh_max_chars if source_is_cjk else en_max_chars
    source_joiner = "" if source_is_cjk else " "
    bilingual = bool(en_text and zh_text)
    word_timestamps = valid_word_timestamps(segment)
    available_chunk_counts: List[int] = []
    if en_text:
        available_chunk_counts.append(
            max(1, len(en_text) if en_is_compact else len(en_text.split()))
        )
    if zh_text:
        available_chunk_counts.append(max(1, len(zh_text)))
    if source_text and not source_is_en and not source_is_zh:
        available_chunk_counts.append(
            max(1, len(source_text) if source_is_cjk else len(source_text.split()))
        )
    max_available_chunks = max(available_chunk_counts) if available_chunk_counts else 1
    if word_timestamps:
        max_available_chunks = min(max_available_chunks, len(word_timestamps))
    max_timed_chunks = max(
        1,
        math.floor((duration + 1e-9) / MIN_SUBTITLE_DURATION),
    )
    max_chunk_count = min(max_available_chunks, max_timed_chunks)
    chunk_count = max(
        1,
        math.ceil(duration / max_duration),
        track_chunk_count(en_text, max_words, en_max_chars, is_cjk=en_is_compact),
        track_chunk_count(zh_text, max_words, zh_max_chars, is_cjk=True),
        track_chunk_count(source_text, max_words, source_char_limit, is_cjk=source_is_cjk),
    )
    if layout_policy is not None:
        if en_text:
            chunk_count = max(
                chunk_count,
                layout_policy.required_chunks(en_text, "en", bilingual=bilingual),
            )
        if zh_text:
            chunk_count = max(
                chunk_count,
                layout_policy.required_chunks(zh_text, "zh", bilingual=bilingual),
            )
        if not en_text and not zh_text and source_text:
            source_track = "zh" if source_is_cjk else "en"
            chunk_count = max(
                chunk_count,
                layout_policy.required_chunks(
                    source_text,
                    source_track,
                    bilingual=False,
                ),
            )
    chunk_count = min(chunk_count, max_chunk_count)
    word_timing_track = str(
        segment.get("word_timing_track") or ""
    ).casefold()
    word_groups: List[List[Dict[str, Any]]] = []

    while True:
        aligned_chunks = None
        if en_text and zh_text:
            aligned_chunks = split_aligned_bilingual_sections(
                en_text,
                zh_text,
                chunk_count,
                max_chunk_count,
                max_words=max_words,
                en_max_chars=en_max_chars,
                zh_max_chars=zh_max_chars,
                en_is_compact=en_is_compact,
                layout_policy=layout_policy,
            )
            if aligned_chunks is None:
                aligned_chunks = split_aligned_bilingual_sentences(
                    en_text,
                    zh_text,
                    chunk_count,
                    max_chunk_count,
                    max_words=max_words,
                    en_max_chars=en_max_chars,
                    zh_max_chars=zh_max_chars,
                    en_is_compact=en_is_compact,
                    layout_policy=layout_policy,
                )
        if aligned_chunks is not None:
            en_chunks, zh_chunks = aligned_chunks
            chunk_count = len(en_chunks)
        else:
            en_chunks = (
                split_text_balanced(
                    en_text,
                    chunk_count,
                    joiner="" if en_is_compact else " ",
                    max_chars=en_max_chars,
                    max_words=None if en_is_compact else max_words,
                )
                if en_text
                else [""] * chunk_count
            )
            zh_chunks = (
                split_text_balanced(
                    zh_text,
                    chunk_count,
                    joiner="",
                    max_chars=zh_max_chars,
                )
                if zh_text
                else [""] * chunk_count
            )
        en_chunks += [""] * (chunk_count - len(en_chunks))
        zh_chunks += [""] * (chunk_count - len(zh_chunks))
        if source_is_en:
            text_chunks = en_chunks
        elif source_is_zh:
            text_chunks = zh_chunks
        else:
            text_chunks = split_text_balanced(
                source_text,
                chunk_count,
                joiner=source_joiner,
                max_chars=source_char_limit,
                max_words=None if source_is_cjk else max_words,
            )
            text_chunks += [""] * (chunk_count - len(text_chunks))
        limits_ok = all(
            aligned_chunk_fits(
                chunk,
                max_chars=en_max_chars,
                max_words=None if en_is_compact else max_words,
                layout_policy=layout_policy,
                track="en",
            )
            for chunk in en_chunks
        ) and all(
            aligned_chunk_fits(
                chunk,
                max_chars=zh_max_chars,
                max_words=None,
                layout_policy=layout_policy,
                track="zh",
            )
            for chunk in zh_chunks
        ) and (
            source_is_en
            or source_is_zh
            or all(
                not chunk
                or (
                    len(chunk) <= source_char_limit
                    and (source_is_cjk or len(chunk.split()) <= max_words)
                )
                or is_bracketed_label(chunk)
                for chunk in text_chunks
            )
        )
        if layout_policy is not None and not (source_is_en or source_is_zh):
            limits_ok = limits_ok and all(
                not chunk
                or is_bracketed_label(chunk)
                or layout_policy.fits(
                    chunk,
                    "zh" if source_is_cjk else "en",
                    bilingual=bilingual,
                )
                for chunk in text_chunks
            )
        timing_chunks = (
            en_chunks
            if word_timestamps and word_timing_track == "en"
            else (
                zh_chunks
                if word_timestamps and word_timing_track == "zh"
                else (
                    en_chunks
                    if source_is_en
                    else (zh_chunks if source_is_zh else text_chunks)
                )
            )
        )
        word_groups = allocate_words_to_chunks(
            word_timestamps,
            timing_chunks,
            compact_text=source_is_cjk,
            max_duration=max_duration,
        )
        if limits_ok or chunk_count >= max_chunk_count:
            break
        chunk_count += 1

    metadata = {
        key: value
        for key, value in segment.items()
        if key not in {"id", "start", "end", "text", "en", "zh", "words", "display_start", "display_end"}
    }
    if (
        chunk_count > 1
        and segment.get("timing_origin")
        and not word_timestamps
        and not (
            layout_policy is not None
            and str(
                segment.get("source_text_authority")
                or segment.get("text_authority")
                or ""
            )
            == "ocr"
        )
    ):
        return [
            {
                **metadata,
                "id": 0,
                "start": start,
                "end": end,
                "text": source_text or en_text or zh_text,
                "en": en_text,
                "zh": zh_text,
                "unanchored_split_avoided": True,
                "split_timing_basis": "source_cue",
            }
        ]

    output: List[Segment] = []
    for index in range(chunk_count):
        word_group = word_groups[index] if word_groups else []
        if word_group:
            piece_start = float(word_group[0]["start"])
            piece_end = float(word_group[-1]["end"])
        else:
            piece_start = start + duration * index / chunk_count
            piece_end = end if index == chunk_count - 1 else start + duration * (index + 1) / chunk_count
        piece_en = en_chunks[index]
        piece_zh = zh_chunks[index]
        piece_text = text_chunks[index] or piece_en or piece_zh
        if not piece_text and not piece_en and not piece_zh:
            continue
        output.append(
            {
                **metadata,
                "id": len(output),
                "start": piece_start,
                "end": piece_end,
                "text": piece_text,
                "en": piece_en,
                "zh": piece_zh,
                **({"words": word_group} if word_group else {}),
                "split_timing_basis": "word_timestamps" if word_group else "proportional",
            }
        )
    return output


def split_segment_by_words(segment: Segment, max_words: int, max_chars: int, max_duration: float) -> List[Segment]:
    words = segment.get("words") or []
    if not words:
        return []
    metadata = {
        key: value
        for key, value in segment.items()
        if key not in {"id", "start", "end", "text", "words"}
    }

    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for word in words:
        candidate = [*current, word]
        candidate_text = join_words(candidate)
        candidate_duration = float(candidate[-1]["end"]) - float(candidate[0]["start"])
        exceeds_limit = (
            len(candidate) > max_words
            or len(candidate_text) > max_chars
            or candidate_duration > max_duration
        )
        if current and exceeds_limit:
            chunks.append(current)
            current = [word]
        else:
            current = candidate
        sentence_end = bool(re.search(r"[.!?]$", word.get("word", "")))
        if current and sentence_end:
            chunks.append(current)
            current = []

    if current:
        chunks.append(current)

    output: List[Segment] = []
    for chunk in chunks:
        text = clean_subtitle_text(join_words(chunk))
        if not text:
            continue
        output.append(
            {
                **metadata,
                "id": len(output),
                "start": float(chunk[0]["start"]),
                "end": float(chunk[-1]["end"]),
                "text": text,
            }
        )
    return output


def split_segment_by_text(segment: Segment, max_words: int, max_chars: int, max_duration: float) -> List[Segment]:
    text = clean_subtitle_text(segment["text"])
    duration = float(segment["end"]) - float(segment["start"])
    min_chunks = max(1, math.ceil(duration / max_duration), math.ceil(len(text) / max_chars))
    chunks = split_plain_text(text, max_words=max_words, max_chars=max_chars, min_chunks=min_chunks)
    if len(chunks) <= 1:
        return [{**segment, "text": text}]
    metadata = {
        key: value
        for key, value in segment.items()
        if key not in {"id", "start", "end", "text", "words"}
    }

    weights = [max(1, len(chunk)) for chunk in chunks]
    total_weight = sum(weights)
    start = float(segment["start"])
    output: List[Segment] = []

    for index, (chunk, weight) in enumerate(zip(chunks, weights)):
        if index == len(chunks) - 1:
            end = float(segment["end"])
        else:
            end = start + duration * weight / total_weight
        output.append(
            {
                **metadata,
                "id": len(output),
                "start": start,
                "end": end,
                "text": chunk,
            }
        )
        start = end

    return output


def split_segments_for_subtitles(
    segments: List[Segment],
    max_words: int = 12,
    max_chars: int = ENGLISH_CHARS_PER_LINE,
    max_duration: float = 5.5,
    layout_policy: Optional[SubtitleLayoutPolicy] = None,
) -> List[Segment]:
    output: List[Segment] = []

    for segment in segments:
        text = clean_subtitle_text(segment.get("text", ""))
        en_text = source_track_text(segment)
        zh_text = to_simplified_text(clean_subtitle_text(str(segment.get("zh") or "")))
        if not text and not en_text and not zh_text:
            continue
        segment = {**segment, "text": text or en_text or zh_text}
        if "en" in segment or "zh" in segment:
            pieces = split_bilingual_segment(
                segment,
                max_words,
                max_chars,
                max_duration,
                layout_policy=layout_policy,
            )
        else:
            duration = float(segment["end"]) - float(segment["start"])
            needs_split = duration > max_duration or len(text) > max_chars or len(text.split()) > max_words

            if not needs_split:
                pieces = [segment]
            elif segment.get("words"):
                pieces = split_segment_by_words(segment, max_words, max_chars, max_duration)
                if not pieces:
                    pieces = split_segment_by_text(segment, max_words, max_chars, max_duration)
            else:
                pieces = split_segment_by_text(segment, max_words, max_chars, max_duration)

        for piece in pieces:
            if float(piece["end"]) <= float(piece["start"]):
                continue
            piece["id"] = len(output)
            output.append(piece)

    print(f"Prepared {len(output)} subtitle events from {len(segments)} source segments.")
    return output


def duplicate_display_key(segment: Segment) -> str:
    text = source_track_text(segment)
    if not text:
        text = to_simplified_text(clean_subtitle_text(str(segment.get("zh") or segment.get("text") or "")))
    normalized = "".join(character.casefold() for character in text if character.isalnum())
    if not normalized:
        return ""
    cjk_count = len(CJK_RE.findall(text))
    word_count = len(text.split())
    if cjk_count >= 8 or word_count >= 4 or len(normalized) >= 24:
        return normalized
    return ""


def overlap_content_key(segment: Segment) -> str:
    duplicate_key = duplicate_display_key(segment)
    if duplicate_key:
        return duplicate_key
    text = clean_subtitle_text(
        str(segment.get("en") or segment.get("zh") or segment.get("text") or "")
    )
    return "".join(character.casefold() for character in text if character.isalnum())


def model_guard_text_keys(segment: Segment) -> List[str]:
    keys: List[str] = []
    for field in ("text", "en", "zh"):
        value = clean_subtitle_text(str(segment.get(field) or ""))
        key = "".join(character.casefold() for character in value if character.isalnum())
        if key and key not in keys:
            keys.append(key)
    return keys


def model_guard_key_is_meaningful(key: str) -> bool:
    cjk_count = len(CJK_RE.findall(key))
    return cjk_count >= 2 or len(key) >= 6


def model_guard_temporal_gap(first: Segment, second: Segment) -> float:
    first_start = float(first["start"])
    first_end = float(first["end"])
    second_start = float(second["start"])
    second_end = float(second["end"])
    if first_start <= second_end and second_start <= first_end:
        return 0.0
    return min(abs(second_start - first_end), abs(first_start - second_end))


def ocr_fragment_covered_by_adjacent_chinese(
    source_segments: List[Segment],
    index: int,
    *,
    max_boundary_gap: float = 0.35,
    max_duration: float = 4.5,
) -> bool:
    if index <= 0 or index + 1 >= len(source_segments):
        return False
    source = source_segments[index]
    source_authority = str(
        source.get("source_text_authority")
        or source.get("text_authority")
        or ""
    )
    source_text = source_track_text(source)
    if (
        source_authority != "ocr"
        or not source_text
        or clean_subtitle_text(str(source.get("zh") or ""))
        or float(source["end"]) - float(source["start"]) > max_duration
    ):
        return False

    from subtitle_pipeline import is_sdh_sound_cue

    if is_sdh_sound_cue(source_text):
        return False
    if re.match(r"^[A-Z][A-Z0-9 .'-]{1,30}:", source_text):
        return False

    previous = source_segments[index - 1]
    following = source_segments[index + 1]
    if not clean_subtitle_text(str(previous.get("zh") or "")):
        return False
    if not clean_subtitle_text(str(following.get("zh") or "")):
        return False
    previous_gap = abs(float(source["start"]) - float(previous["end"]))
    following_gap = abs(float(following["start"]) - float(source["end"]))
    return (
        previous_gap <= max_boundary_gap
        and following_gap <= max_boundary_gap
    )


def ocr_tiny_fragment_covered_by_following_chinese(
    source_segments: List[Segment],
    index: int,
    *,
    max_boundary_gap: float = 0.12,
    max_duration: float = 0.75,
    max_words: int = 2,
) -> bool:
    if index < 0 or index + 1 >= len(source_segments):
        return False
    source = source_segments[index]
    source_authority = str(
        source.get("source_text_authority")
        or source.get("text_authority")
        or ""
    )
    source_text = source_track_text(source)
    source_words = re.findall(r"\b[\w'-]+\b", source_text)
    if (
        source_authority != "ocr"
        or not source_text
        or clean_subtitle_text(str(source.get("zh") or ""))
        or float(source["end"]) - float(source["start"]) > max_duration
        or not source_words
        or len(source_words) > max_words
    ):
        return False

    following = source_segments[index + 1]
    following_chinese = clean_subtitle_text(str(following.get("zh") or ""))
    following_authority = str(
        following.get("chinese_text_authority")
        or following.get("text_authority")
        or ""
    )
    boundary_gap = float(following["start"]) - float(source["end"])
    return (
        following_authority == "authored"
        and bool(following_chinese)
        and boundary_gap <= max_boundary_gap
        and boundary_gap >= -max_boundary_gap
    )


def model_hide_support_reason(
    source_segments: List[Segment],
    processed_segments: List[Segment],
    index: int,
    *,
    max_gap: float = 4.0,
) -> Optional[str]:
    source = source_segments[index]
    source_text = source_track_text(source)
    source_authority = str(
        source.get("source_text_authority")
        or source.get("text_authority")
        or ""
    )
    if (
        source_authority == "ocr"
        and source_text.startswith("%")
        and not clean_subtitle_text(str(source.get("zh") or ""))
    ):
        return "ocr_symbol_fragment"
    if ocr_fragment_covered_by_adjacent_chinese(source_segments, index):
        return "ocr_fragment_covered_by_adjacent_chinese"
    if ocr_tiny_fragment_covered_by_following_chinese(source_segments, index):
        return "ocr_tiny_fragment_covered_by_following_chinese"
    source_keys = model_guard_text_keys(source)
    meaningful_keys = [key for key in source_keys if model_guard_key_is_meaningful(key)]
    if not meaningful_keys:
        return "empty_or_nonsemantic_source"

    for neighbor_index in range(index - 1, max(-1, index - 51), -1):
        neighbor = processed_segments[neighbor_index]
        if neighbor.get("display", True) is False:
            continue
        if model_guard_temporal_gap(source, neighbor) > max_gap:
            continue

        represented_keys = model_guard_text_keys(neighbor)
        if neighbor_index < len(source_segments):
            represented_keys.extend(model_guard_text_keys(source_segments[neighbor_index]))
        for source_key in meaningful_keys:
            for represented_key in represented_keys:
                if not model_guard_key_is_meaningful(represented_key):
                    continue
                if source_key == represented_key:
                    return "adjacent_exact_duplicate"
                if source_key in represented_key:
                    return "covered_by_adjacent_visible_line"
                length_ratio = min(len(source_key), len(represented_key)) / max(
                    len(source_key),
                    len(represented_key),
                )
                similarity = SequenceMatcher(None, source_key, represented_key).ratio()
                if length_ratio >= 0.75 and similarity >= 0.88:
                    return "adjacent_near_duplicate"
    return None


def guard_model_hidden_segment_at(
    source_segments: List[Segment],
    processed_segments: List[Segment],
    index: int,
) -> None:
    segment = processed_segments[index]
    if (
        segment.get("non_content_credit")
        or segment.get("suppression_reason") == "subtitle_credit"
    ):
        segment["display"] = False
        segment["model_display_requested"] = False
        segment["display_guard"] = "accepted_hidden"
        segment["display_guard_reason"] = "subtitle_credit"
        return
    if (
        segment.get("manual_reviewed_at")
        and segment.get("display", True) is False
    ):
        segment["model_display_requested"] = False
        segment["display_guard"] = "accepted_hidden"
        segment["display_guard_reason"] = "manual_review"
        return
    if segment.get("display_suppression"):
        segment["display"] = False
        return
    model_requested_hidden = segment.get("model_display_requested") is False
    if not model_requested_hidden and segment.get("display", True) is False:
        model_requested_hidden = not bool(segment.get("display_suppression"))
    if not model_requested_hidden:
        return

    segment["model_display_requested"] = False
    if not model_may_hide_segment(source_segments[index]):
        segment["display"] = True
        segment["display_guard"] = "forced_visible"
        segment["display_guard_reason"] = "authored_source_preserved"
        return
    reason = model_hide_support_reason(source_segments, processed_segments, index)
    if reason:
        segment["display"] = False
        segment["display_guard"] = "accepted_hidden"
        segment["display_guard_reason"] = reason
    else:
        segment["display"] = True
        segment["display_guard"] = "forced_visible"
        segment["display_guard_reason"] = "unique_source_not_covered"


def guard_model_hidden_segments(
    source_segments: List[Segment],
    processed_segments: List[Segment],
) -> List[Segment]:
    if len(source_segments) != len(processed_segments):
        raise ValueError("Display guard requires one processed item per source subtitle event")

    guarded = []
    for processed in processed_segments:
        item = dict(processed)
        item.pop("display_start", None)
        item.pop("display_end", None)
        guarded.append(item)
    for index in range(len(guarded)):
        guard_model_hidden_segment_at(source_segments, guarded, index)
    return guarded


def is_subtitle_track_label_artifact(segment: Segment) -> bool:
    for field in ("text", "en"):
        value = normalize_space(str(segment.get(field) or ""))
        if value and SUBTITLE_TRACK_LABEL_RE.fullmatch(value):
            return True
    return False


def suppress_subtitle_track_label_artifacts(
    segments: List[Segment],
) -> Tuple[List[Segment], List[Dict[str, Any]]]:
    normalized: List[Segment] = []
    suppressed: List[Dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        if is_subtitle_track_label_artifact(item):
            item.update(
                {
                    "display": False,
                    "display_suppression": "subtitle_track_label_artifact",
                    "display_guard": "accepted_hidden",
                    "display_guard_reason": "subtitle_track_label_artifact",
                }
            )
            suppressed.append(
                {
                    "id": item.get("id"),
                    "start": item.get("start"),
                    "text": str(item.get("text") or item.get("en") or ""),
                }
            )
        normalized.append(item)
    return normalized, suppressed


def guarded_display_candidate(
    source_segments: List[Segment],
    previous_processed: List[Segment],
    segment: Segment,
    item: Dict[str, Any],
    context_start: float,
    context_end: float,
) -> Segment:
    candidate = apply_display_timing(
        segment,
        item,
        context_start,
        context_end,
    )
    source_prefix = source_segments[: len(previous_processed) + 1]
    combined = [*previous_processed, candidate]
    guard_model_hidden_segment_at(
        source_prefix,
        combined,
        len(combined) - 1,
    )
    return combined[-1]


def first_guarded_language_repair_index(
    source_segments: List[Segment],
    processed_segments: List[Segment],
    *,
    proofread: bool,
) -> Optional[int]:
    for index, processed in enumerate(processed_segments):
        if processed.get("display_guard") != "forced_visible":
            continue
        if proofread:
            original_en, original_zh = original_en_zh(source_segments[index])
            corrected = clean_english_track_text(str(processed.get("en") or original_en))
            translated = to_simplified_text(
                clean_subtitle_text(str(processed.get("zh") or original_zh))
            )
            valid = is_language_valid(
                original_en or original_zh,
                corrected,
                translated,
                require_corrected=bool(original_en),
                original_translation=original_zh,
            )
        else:
            corrected = clean_subtitle_text(
                str(processed.get("en") or processed.get("text") or "")
            )
            translated = clean_subtitle_text(str(processed.get("zh") or ""))
            valid = is_language_valid(
                str(source_segments[index].get("text") or ""),
                corrected,
                translated,
            )
        if not valid:
            return index
    return None


def suppress_recent_exact_duplicates(
    segments: List[Segment],
    max_gap: float = 4.0,
    max_duration: float = 5.5,
) -> List[Segment]:
    output = [dict(segment) for segment in segments]
    last_visible_by_key: Dict[str, Segment] = {}
    for segment in output:
        if segment.get("display", True) is False:
            continue
        key = duplicate_display_key(segment)
        if not key:
            continue
        previous = last_visible_by_key.get(key)
        if previous is not None:
            gap = float(segment["start"]) - float(previous["end"])
            if gap <= max_gap:
                segment["display"] = False
                segment["display_suppression"] = "recent_exact_duplicate"
                continue
        last_visible_by_key[key] = segment
    extend_display_over_hidden_segments(output, max_gap=2.0, max_duration=max_duration)
    return output


def prepare_segments_for_output(
    segments: List[Segment],
    max_words: int,
    max_chars: int,
    max_duration: float,
    layout_policy: Optional[SubtitleLayoutPolicy] = None,
    scene_cuts: Optional[List[float]] = None,
) -> List[Segment]:
    visible: List[Segment] = []
    deduplicated = suppress_recent_exact_duplicates(segments, max_duration=max_duration)
    for segment in deduplicated:
        if (
            segment.get("display", True) is False
            or segment.get("non_content_credit")
        ):
            continue
        item = dict(segment)
        item.setdefault("checkpoint_segment_id", item.get("id"))
        item["start"] = float(item.get("display_start", item["start"]))
        item["end"] = float(item.get("display_end", item["end"]))
        item.pop("display_start", None)
        item.pop("display_end", None)
        if item["end"] > item["start"]:
            visible.append(item)
    timeline_cleaned = resolve_timeline_overlaps(visible)
    split = split_segments_for_subtitles(
        timeline_cleaned,
        max_words,
        max_chars,
        max_duration,
        layout_policy=layout_policy,
    )
    sanitized = apply_timing_sanity_rules(split, max_duration=max_duration)
    ordered = sorted(sanitized, key=lambda item: (float(item["start"]), float(item["end"])))
    non_overlapping = resolve_timeline_overlaps(ordered)
    readable = optimize_readability_timing(non_overlapping, max_duration=max_duration)
    scene_aligned = align_subtitles_to_scene_cuts(
        readable,
        scene_cuts or [],
        max_duration_seconds=max_duration,
        min_duration_seconds=MIN_SUBTITLE_DURATION,
        min_gap_seconds=MIN_SUBTITLE_GAP,
    )
    return resolve_timeline_overlaps(scene_aligned)


def apply_timing_sanity_rules(segments: List[Segment], max_duration: float = 6.0) -> List[Segment]:
    output: List[Segment] = []
    for segment in segments:
        item = segment.copy()
        duration = float(item["end"]) - float(item["start"])
        word_count = len(str(item.get("text", "")).split())
        if duration > max_duration and word_count <= 4:
            item["end"] = float(item["start"]) + max_duration
        item["id"] = len(output)
        output.append(item)
    return output


def subtitle_character_count(text: Any, *, collapse_spaces: bool = False) -> int:
    cleaned = clean_subtitle_text(str(text or ""))
    if collapse_spaces:
        cleaned = re.sub(r"\s+", "", cleaned)
    return len(cleaned)


def required_reading_duration(segment: Segment) -> float:
    source = source_track_text(segment)
    chinese = to_simplified_text(clean_subtitle_text(str(segment.get("zh") or "")))
    if not source and not chinese:
        text = clean_subtitle_text(str(segment.get("text") or ""))
        if contains_cjk(text) and normalize_language_tag(
            str(segment.get("source_language") or "")
        ) == "zh":
            chinese = text
        else:
            source = text

    required = MIN_SUBTITLE_DURATION
    if chinese:
        required = max(
            required,
            subtitle_character_count(chinese, collapse_spaces=True) / TARGET_CHINESE_CPS,
        )
    if source:
        source_cps = (
            TARGET_CHINESE_CPS
            if contains_east_asian(source)
            else TARGET_ENGLISH_CPS
        )
        required = max(required, subtitle_character_count(source) / source_cps)
    return required


def optimize_readability_timing(
    segments: List[Segment],
    max_duration: float = 5.5,
) -> List[Segment]:
    output = [dict(segment) for segment in segments]
    for index, segment in enumerate(output):
        start = float(segment["start"])
        current_end = float(segment["end"])
        required_end = start + required_reading_duration(segment)
        if required_end <= current_end:
            continue

        latest_end = min(current_end + MAX_READING_EXTENSION, start + max_duration)
        if index + 1 < len(output):
            next_start = float(output[index + 1]["start"])
            latest_end = min(latest_end, next_start - MIN_SUBTITLE_GAP)
        extended_end = min(required_end, latest_end)
        if extended_end > current_end:
            segment["end"] = extended_end
    return output


def resolve_timeline_overlaps(segments: List[Segment]) -> List[Segment]:
    output: List[Segment] = []
    for source in segments:
        item = dict(source)
        while output and float(output[-1]["end"]) > float(item["start"]):
            previous = output[-1]
            if float(item["end"]) <= float(previous["start"]):
                break
            previous_key = overlap_content_key(previous)
            current_key = overlap_content_key(item)
            preserve_distinct = (
                bool(previous.get("preserve_distinct_overlap"))
                and bool(item.get("preserve_distinct_overlap"))
                and bool(previous_key)
                and bool(current_key)
                and previous_key != current_key
            )
            if preserve_distinct:
                break
            shortened_end = float(item["start"]) - MIN_SUBTITLE_GAP
            retained_duration = shortened_end - float(previous["start"])
            if retained_duration >= MIN_OVERLAP_RETAIN_DURATION:
                previous["end"] = shortened_end
                break
            output.pop()
        item["id"] = len(output)
        output.append(item)
    for index, item in enumerate(output):
        item["id"] = index
    return output


def readability_stats(segments: List[Segment]) -> Dict[str, float | int]:
    stats: Dict[str, float | int] = {
        "chinese_events": 0,
        "english_events": 0,
        "chinese_max_cps": 0.0,
        "english_max_cps": 0.0,
        "chinese_over_target": 0,
        "english_over_target": 0,
        "chinese_over_single_line": 0,
        "english_over_single_line": 0,
        "compact_source_events": 0,
        "compact_source_max_cps": 0.0,
        "compact_source_over_target": 0,
        "compact_source_over_single_line": 0,
    }
    for segment in segments:
        duration = max(0.01, float(segment["end"]) - float(segment["start"]))
        chinese = to_simplified_text(clean_subtitle_text(str(segment.get("zh") or "")))
        source = source_track_text(segment)
        if not chinese and not source:
            text = clean_subtitle_text(str(segment.get("text") or ""))
            if contains_cjk(text) and normalize_language_tag(
                str(segment.get("source_language") or "")
            ) == "zh":
                chinese = text
            else:
                source = text

        if chinese:
            count = subtitle_character_count(chinese, collapse_spaces=True)
            cps = count / duration
            stats["chinese_events"] += 1
            stats["chinese_max_cps"] = max(float(stats["chinese_max_cps"]), cps)
            stats["chinese_over_target"] += int(cps > TARGET_CHINESE_CPS + 1e-9)
            stats["chinese_over_single_line"] += int(count > CHINESE_CHARS_PER_LINE)
        if source:
            count = subtitle_character_count(source)
            cps = count / duration
            if contains_east_asian(source):
                stats["compact_source_events"] += 1
                stats["compact_source_max_cps"] = max(
                    float(stats["compact_source_max_cps"]),
                    cps,
                )
                stats["compact_source_over_target"] += int(
                    cps > TARGET_CHINESE_CPS + 1e-9
                )
                stats["compact_source_over_single_line"] += int(
                    count > CHINESE_CHARS_PER_LINE
                )
            else:
                stats["english_events"] += 1
                stats["english_max_cps"] = max(float(stats["english_max_cps"]), cps)
                stats["english_over_target"] += int(cps > TARGET_ENGLISH_CPS + 1e-9)
                stats["english_over_single_line"] += int(count > ENGLISH_CHARS_PER_LINE)
    return stats


def print_readability_report(segments: List[Segment]) -> None:
    stats = readability_stats(segments)
    print(
        "Readability report: "
        f"zh_max_cps={float(stats['chinese_max_cps']):.2f}, "
        f"zh>{TARGET_CHINESE_CPS:g}cps={stats['chinese_over_target']}, "
        f"zh>{CHINESE_CHARS_PER_LINE}chars={stats['chinese_over_single_line']}, "
        f"en_max_cps={float(stats['english_max_cps']):.2f}, "
        f"en>{TARGET_ENGLISH_CPS:g}cps={stats['english_over_target']}, "
        f"en>{ENGLISH_CHARS_PER_LINE}chars={stats['english_over_single_line']}, "
        f"east_asian_source_max_cps={float(stats['compact_source_max_cps']):.2f}, "
        f"east_asian_source>{TARGET_CHINESE_CPS:g}cps={stats['compact_source_over_target']}, "
        f"east_asian_source>{CHINESE_CHARS_PER_LINE}chars="
        f"{stats['compact_source_over_single_line']}"
    )


def build_context_text(segments: List[Segment], start: int, end: int, context_lines: int) -> Tuple[str, str]:
    before_start = max(0, start - context_lines)
    after_end = min(len(segments), end + context_lines)
    before = "\n".join(format_prompt_segment(index, segments[index]) for index in range(before_start, start))
    after = "\n".join(format_prompt_segment(index, segments[index]) for index in range(end, after_end))
    return before, after


def available_reading_duration(
    segment: Segment,
    next_start: Optional[float] = None,
    max_duration: float = 5.5,
) -> float:
    start = float(segment["start"])
    end = float(segment["end"])
    latest_end = min(end + MAX_READING_EXTENSION, start + max_duration)
    if next_start is not None:
        latest_end = min(latest_end, float(next_start) - MIN_SUBTITLE_GAP)
    return max(0.01, latest_end - start)


def reading_budgets(
    segment: Segment,
    next_start: Optional[float] = None,
) -> Tuple[int, int]:
    duration = available_reading_duration(segment, next_start)
    chinese_budget = max(
        4,
        min(
            CHINESE_CHARS_PER_LINE * BILINGUAL_LINES_PER_LANGUAGE,
            math.floor(duration * TARGET_CHINESE_CPS),
        ),
    )
    english_budget = max(
        4,
        min(
            ENGLISH_CHARS_PER_LINE * BILINGUAL_LINES_PER_LANGUAGE,
            math.floor(duration * TARGET_ENGLISH_CPS),
        ),
    )
    return chinese_budget, english_budget


def recognition_risk_labels(segment: Segment) -> List[str]:
    labels: List[str] = []
    authority = str(
        segment.get("source_text_authority")
        or segment.get("text_authority")
        or ""
    )
    if authority == "authored":
        return ["authored_source"]
    if authority == "ocr":
        try:
            confidence = float(segment.get("ocr_confidence"))
        except (TypeError, ValueError):
            confidence = None
        if confidence is None:
            labels.append("ocr_confidence_unknown")
        elif confidence < 0.75:
            labels.append("low_ocr_confidence")
    if authority == "asr":
        for key, threshold, label, comparison in (
            ("asr_avg_logprob", -1.0, "low_asr_log_probability", "lt"),
            ("asr_compression_ratio", 2.4, "high_asr_compression_ratio", "gt"),
            ("asr_no_speech_prob", 0.6, "high_no_speech_probability", "gt"),
            ("asr_word_confidence", 0.65, "low_word_confidence", "lt"),
        ):
            try:
                value = float(segment.get(key))
            except (TypeError, ValueError):
                continue
            if (comparison == "lt" and value < threshold) or (
                comparison == "gt" and value > threshold
            ):
                labels.append(label)
    return labels or ["no_recognition_risk"]


def format_prompt_segment(
    index: int,
    segment: Segment,
    *,
    next_start: Optional[float] = None,
    include_readability_limits: bool = False,
) -> str:
    suffix = ""
    if include_readability_limits:
        chinese_budget, english_budget = reading_budgets(segment, next_start)
        suffix = f" | LIMITS: ZH_MAX={chinese_budget}, SOURCE_MAX={english_budget}"
    authority = str(
        segment.get("source_text_authority")
        or segment.get("text_authority")
        or "unknown"
    )
    risk = ",".join(recognition_risk_labels(segment))
    return (
        f"[{index}] {float(segment['start']):.2f}->{float(segment['end']):.2f} "
        f"{segment['text']} | SOURCE_AUTHORITY={authority} "
        f"| RECOGNITION_RISK={risk}{suffix}"
    )


def format_bilingual_prompt_segment(
    index: int,
    segment: Segment,
    *,
    next_start: Optional[float] = None,
    include_readability_limits: bool = False,
) -> str:
    source_text, zh_text = original_en_zh(segment)
    source_language = normalize_language_tag(
        str(segment.get("source_language") or "")
    ) or "unknown"
    suffix = ""
    if include_readability_limits:
        chinese_budget, english_budget = reading_budgets(segment, next_start)
        suffix = f" | LIMITS: ZH_MAX={chinese_budget}, EN_MAX={english_budget}"
    authority = str(
        segment.get("source_text_authority")
        or segment.get("text_authority")
        or "unknown"
    )
    chinese_authority = str(
        segment.get("chinese_text_authority")
        or "unknown"
    )
    risk = ",".join(recognition_risk_labels(segment))
    return (
        f"[{index}] {float(segment['start']):.2f}->{float(segment['end']):.2f} "
        f"EN: {source_text or '-'} | ZH: {zh_text or '-'} "
        f"| SOURCE_LANGUAGE={source_language} "
        f"| SOURCE_AUTHORITY={authority} "
        f"| CHINESE_AUTHORITY={chinese_authority} "
        f"| RECOGNITION_RISK={risk}{suffix}"
    )


def build_bilingual_context_text(segments: List[Segment], start: int, end: int, context_lines: int) -> Tuple[str, str]:
    before_start = max(0, start - context_lines)
    after_end = min(len(segments), end + context_lines)
    before = "\n".join(format_bilingual_prompt_segment(index, segments[index]) for index in range(before_start, start))
    after = "\n".join(format_bilingual_prompt_segment(index, segments[index]) for index in range(end, after_end))
    return before, after


def build_approved_output_context(segments: List[Segment], context_lines: int) -> str:
    if context_lines <= 0:
        return "(none)"
    rows: List[str] = []
    for segment in segments[-context_lines:]:
        if segment.get("display", True) is False:
            continue
        source_text = clean_subtitle_text(str(segment.get("en") or segment.get("text") or ""))
        chinese_text = clean_subtitle_text(str(segment.get("zh") or ""))
        if source_text or chinese_text:
            rows.append(f"EN/SOURCE: {source_text or '-'} | ZH: {chinese_text or '-'}")
    return "\n".join(rows) or "(none)"


def extract_terminology_entries(item: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_entries = item.get("terminology", item.get("terms", []))
    if not isinstance(raw_entries, list):
        return []
    entries: List[Dict[str, str]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        source = normalize_space(
            str(raw.get("source") or raw.get("source_term") or raw.get("name") or "")
        )
        target = normalize_space(
            str(raw.get("target") or raw.get("chinese") or raw.get("translation") or "")
        )
        if not source or not target or len(source) > 100 or len(target) > 100:
            continue
        entries.append({"source": source, "target": to_simplified_text(target)})
    return entries


def terminology_from_segments(
    segments: List[Segment],
    seeded_entries: Any = None,
) -> Dict[str, Dict[str, str]]:
    glossary: Dict[str, Dict[str, str]] = {}
    register_terminology(glossary, seeded_entries)
    for segment in segments:
        register_terminology(glossary, segment.get("terminology"))
    return glossary


def terminology_review_candidates(
    segments: List[Segment],
    seeded_entries: Any = None,
) -> List[Dict[str, Any]]:
    glossary: Dict[str, Dict[str, str]] = {}
    for segment in segments:
        register_terminology(glossary, segment.get("terminology"))
    register_terminology(glossary, seeded_entries)

    candidates: List[Dict[str, Any]] = []
    for entry in sorted(
        glossary.values(),
        key=lambda item: item["source"].casefold(),
    ):
        contexts: List[Dict[str, Any]] = []
        for segment in segments:
            source, chinese = original_en_zh(segment)
            if not source_contains_term(source, entry["source"]):
                continue
            contexts.append(
                {
                    "id": segment.get("id"),
                    "source": source,
                    "chinese": chinese,
                    "ocr_confidence": segment.get("ocr_confidence"),
                }
            )
            if len(contexts) == 3:
                break
        candidates.append({**entry, "contexts": contexts})
    return candidates


def terminology_review_fingerprint(
    candidates: List[Dict[str, Any]],
    llm_model: str,
    title: str,
    review_mode: str,
) -> str:
    serialized = json.dumps(
        {
            "version": TERMINOLOGY_REVIEW_POLICY_VERSION,
            "llm_model": llm_model,
            "title": normalize_space(title),
            "review_mode": review_mode,
            "candidates": candidates,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def terminology_mapping_fingerprint(
    candidates: List[Dict[str, Any]],
    llm_model: str,
    title: str,
    review_mode: str,
) -> str:
    serialized = json.dumps(
        {
            "version": TERMINOLOGY_REVIEW_POLICY_VERSION,
            "llm_model": llm_model,
            "title": normalize_space(title),
            "review_mode": review_mode,
            "mappings": [
                {
                    "source": candidate["source"],
                    "target": candidate["target"],
                }
                for candidate in candidates
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def is_plausible_ocr_terminology_repair(
    source: str,
    original_target: str,
    proposed_target: str,
) -> bool:
    source_text = normalize_space(source).casefold()
    original = normalize_space(original_target).casefold()
    proposed = normalize_space(proposed_target).casefold()
    if not original or not proposed or original == proposed:
        return False
    if proposed == source_text:
        return True
    if abs(len(original) - len(proposed)) > 1:
        return False
    if len(original) == len(proposed):
        return sum(
            left != right
            for left, right in zip(original, proposed)
        ) == 1

    shorter, longer = (
        (original, proposed)
        if len(original) < len(proposed)
        else (proposed, original)
    )
    short_index = 0
    long_index = 0
    differences = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        differences += 1
        long_index += 1
        if differences > 1:
            return False
    return True


def validate_terminology_review_response(
    data: Any,
    expected_count: int,
) -> Dict[int, Dict[str, Any]]:
    if isinstance(data, dict):
        wrapped = next(
            (
                data.get(key)
                for key in ("items", "results", "terminology")
                if isinstance(data.get(key), list)
            ),
            None,
        )
        if wrapped is not None:
            data = wrapped
        elif expected_count == 1:
            data = [data]
    if not isinstance(data, list) or len(data) != expected_count:
        raise ValueError(
            "Terminology review response must contain exactly "
            f"{expected_count} items"
        )
    lookup: Dict[int, Dict[str, Any]] = {}
    for position, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(
                f"Terminology review item {position} is not an object"
            )
        item = dict(raw)
        if "index" not in item and "local_index" in item:
            item["index"] = item["local_index"]
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Terminology review item {position} has no valid index"
            ) from exc
        target = item.get("target")
        raw_confidence = item.get("confidence")
        decision = item.get("decision")
        if not isinstance(target, str) or not normalize_space(target):
            raise ValueError(
                f"Terminology review item {position} has no target"
            )
        if isinstance(raw_confidence, str):
            confidence = {
                "high": 1.0,
                "medium": 0.6,
                "low": 0.3,
            }.get(raw_confidence.casefold(), -1.0)
        else:
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = -1.0
        if not 0 <= confidence <= 1:
            raise ValueError(
                f"Terminology review item {position} has invalid confidence"
            )
        if decision not in {"keep", "ocr_repair", "retranslate"}:
            raise ValueError(
                f"Terminology review item {position} has invalid decision"
            )
        if index in lookup:
            raise ValueError(
                f"Terminology review response repeats index {index}"
            )
        item["confidence"] = confidence
        lookup[index] = item
    expected_indexes = set(range(expected_count))
    if set(lookup) != expected_indexes:
        raise ValueError(
            "Terminology review response is missing indexes: "
            f"{sorted(expected_indexes - set(lookup))}"
        )
    return lookup


def review_movie_terminology(
    segments: List[Segment],
    llm_model: str,
    artifact_path: Path,
    *,
    title: str = "",
    seeded_entries: Any = None,
    batch_size: int = 16,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    candidates = terminology_review_candidates(segments, seeded_entries)
    preserve_existing_edition = any(
        str(segment.get("processing_mode") or "")
        == "proofread_existing_chinese"
        for segment in segments
    )
    review_mode = (
        "preserve_existing_edition"
        if preserve_existing_edition
        else "review_generated_translation"
    )
    input_fingerprint = terminology_review_fingerprint(
        candidates,
        llm_model,
        title,
        review_mode,
    )
    mapping_fingerprint = terminology_mapping_fingerprint(
        candidates,
        llm_model,
        title,
        review_mode,
    )
    if not candidates:
        report = {
            "version": TERMINOLOGY_REVIEW_POLICY_VERSION,
            "status": "skipped_empty",
            "llm_model": llm_model,
            "input_fingerprint": input_fingerprint,
            "mapping_fingerprint": mapping_fingerprint,
            "review_mode": review_mode,
            "reviewed_count": 0,
            "correction_count": 0,
            "unapplied_count": 0,
            "entries": [],
            "resolved_entries": [],
        }
        write_json_atomic(artifact_path, report)
        return [], report

    cached: Optional[Dict[str, Any]] = None
    if artifact_path.exists():
        try:
            raw = json.loads(artifact_path.read_text(encoding="utf-8"))
            if (
                isinstance(raw, dict)
                and raw.get("version") == TERMINOLOGY_REVIEW_POLICY_VERSION
                and raw.get("llm_model") == llm_model
                and (
                    raw.get("input_fingerprint") == input_fingerprint
                    or raw.get("mapping_fingerprint") == mapping_fingerprint
                )
            ):
                cached = raw
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    if (
        cached
        and cached.get("status") in {"pass", "review"}
        and isinstance(cached.get("resolved_entries"), list)
    ):
        if cached.get("mapping_fingerprint") != mapping_fingerprint:
            cached = {
                **cached,
                "mapping_fingerprint": mapping_fingerprint,
            }
            write_json_atomic(artifact_path, cached)
        print(f"Using cached movie-wide terminology review: {artifact_path}")
        return [
            dict(entry)
            for entry in cached["resolved_entries"]
            if isinstance(entry, dict)
        ], cached

    reviewed: List[Dict[str, Any]] = []
    if (
        cached
        and cached.get("status") == "running"
        and isinstance(cached.get("entries"), list)
    ):
        reviewed = [
            dict(entry)
            for entry in cached["entries"]
            if isinstance(entry, dict)
        ]
        if len(reviewed) > len(candidates):
            reviewed = []
        elif reviewed:
            print(f"Resuming terminology review at item {len(reviewed) + 1}.")

    effective_batch_size = max(1, batch_size)
    start_index = len(reviewed)
    start_index -= start_index % effective_batch_size
    reviewed = reviewed[:start_index]
    for batch_start in range(
        start_index,
        len(candidates),
        effective_batch_size,
    ):
        batch = candidates[batch_start : batch_start + effective_batch_size]
        item_text = "\n\n".join(
            (
                f"ITEM {index}\n"
                f"SOURCE TERM: {candidate['source']}\n"
                f"CURRENT CHINESE TARGET: {candidate['target']}\n"
                "CONTEXTS:\n"
                + (
                    "\n".join(
                        "- SOURCE: "
                        f"{context['source']}\n"
                        f"  CHINESE OCR/SUBTITLE: {context['chinese']}\n"
                        f"  OCR CONFIDENCE: {context.get('ocr_confidence')}"
                        for context in candidate["contexts"]
                    )
                    or "- (no direct context retained)"
                )
            )
            for index, candidate in enumerate(batch)
        )
        prompt = f"""
Review the terminology mappings for the film "{title or '(unknown title)'}".
{
    "The Chinese track is an existing professional subtitle edition. Preserve "
    "its wording and regional name choices; only OCR repairs may be applied."
    if preserve_existing_edition
    else
    "The Chinese track is generated or provisional. High-confidence terminology "
    "corrections and retranslations may be applied."
}
Some text may have come through bitmap OCR, so individual look-alike characters
or Latin handles can be misrecognized.

{item_text}

Return exactly {len(batch)} JSON objects using the provided schema.
- index is the zero-based ITEM number in this batch.
- target is the best concise Simplified Chinese rendering.
- confidence is a number from 0.0 to 1.0. Use at least 0.9 only when the
  classification and target are both clear.
- decision must be "keep", "ocr_repair", or "retranslate".
- Use "ocr_repair" only when preserving the same edition wording while correcting
  recognition damage such as a look-alike character, missing stroke, or corrupted
  Latin identifier.
- Use "retranslate" for any different localization, synonym, transliteration,
  Mainland/Taiwan terminology swap, or preferred spelling. Retranslations will not
  be applied, even at high confidence.
- Keep valid edition choices such as regional transliterations and vocabulary.
- For a stylized avatar handle, recover the edition's existing spelling or localized
  pun; do not invent a new transliteration.
- If a repeated CJK target appears to be a one-character OCR corruption of a
  localized handle or pun, repair that CJK target instead of falling back to the
  Latin source spelling.
- Do not output explanations or extra keys.
"""
        response_schema = indexed_terminology_schema(len(batch))
        lookup: Optional[Dict[int, Dict[str, Any]]] = None
        for attempt in range(3):
            try:
                data = parse_json_response(
                    call_llm(
                        prompt,
                        (
                            "You are the senior terminology editor for a "
                            "professional Simplified Chinese film subtitle."
                        ),
                        llm_model,
                        role="proofread",
                        response_schema=response_schema,
                        seed_offset=attempt,
                    ),
                    response_schema=response_schema,
                )
                lookup = validate_terminology_review_response(
                    data,
                    len(batch),
                )
                break
            except Exception as exc:
                print(
                    "Terminology review batch "
                    f"{batch_start + 1}-{batch_start + len(batch)} failed "
                    f"(attempt {attempt + 1}/3): {exc}"
                )
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise
        if lookup is None:
            raise RuntimeError(
                "Terminology review did not produce a complete batch"
            )

        for local_index, candidate in enumerate(batch):
            item = lookup[local_index]
            original_target = to_simplified_text(candidate["target"])
            proposed_target = to_simplified_text(
                clean_subtitle_text(item["target"])
            )
            confidence = float(item["confidence"])
            decision = str(item["decision"])
            proposed_changed = (
                normalize_space(proposed_target).casefold()
                != normalize_space(original_target).casefold()
            )
            plausible_ocr_repair = is_plausible_ocr_terminology_repair(
                candidate["source"],
                original_target,
                proposed_target,
            )
            applied = bool(
                proposed_changed
                and confidence >= 0.9
                and (
                    (
                        decision == "ocr_repair"
                        and (
                            not preserve_existing_edition
                            or plausible_ocr_repair
                        )
                    )
                    or (
                        not preserve_existing_edition
                        and decision == "retranslate"
                    )
                )
                and proposed_target
            )
            reviewed.append(
                {
                    "source": candidate["source"],
                    "original_target": original_target,
                    "proposed_target": proposed_target,
                    "target": proposed_target if applied else original_target,
                    "confidence": confidence,
                    "decision": decision,
                    "plausible_ocr_repair": plausible_ocr_repair,
                    "changed": applied,
                    "unapplied": proposed_changed and not applied,
                }
            )
        write_json_atomic(
            artifact_path,
            {
                "version": TERMINOLOGY_REVIEW_POLICY_VERSION,
                "status": "running",
                "llm_model": llm_model,
                "input_fingerprint": input_fingerprint,
                "mapping_fingerprint": mapping_fingerprint,
                "review_mode": review_mode,
                "reviewed_count": len(reviewed),
                "entries": reviewed,
            },
        )

    resolved_entries = [
        {"source": entry["source"], "target": entry["target"]}
        for entry in reviewed
    ]
    correction_count = sum(bool(entry.get("changed")) for entry in reviewed)
    unapplied_count = sum(bool(entry.get("unapplied")) for entry in reviewed)
    report = {
        "version": TERMINOLOGY_REVIEW_POLICY_VERSION,
        "status": "review" if unapplied_count else "pass",
        "llm_model": llm_model,
        "input_fingerprint": input_fingerprint,
        "mapping_fingerprint": mapping_fingerprint,
        "review_mode": review_mode,
        "reviewed_count": len(reviewed),
        "correction_count": correction_count,
        "unapplied_count": unapplied_count,
        "entries": reviewed,
        "resolved_entries": resolved_entries,
    }
    write_json_atomic(artifact_path, report)
    return resolved_entries, report


def register_terminology(
    glossary: Dict[str, Dict[str, str]],
    entries: Any,
) -> List[Dict[str, str]]:
    if not isinstance(entries, list):
        return []
    accepted: List[Dict[str, str]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        source = normalize_space(str(raw.get("source") or ""))
        target = to_simplified_text(normalize_space(str(raw.get("target") or "")))
        if not source or not target:
            continue
        key = source.casefold()
        if key in CONTEXT_DEPENDENT_TERMINOLOGY_SOURCES:
            continue
        existing = glossary.get(key)
        if existing is None:
            existing = {"source": source, "target": target}
            glossary[key] = existing
        accepted.append(dict(existing))
    return accepted


def load_terminology_overrides(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("terminology overrides must contain an entries list")
    overrides: List[Dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        source = normalize_space(str(raw_entry.get("source") or ""))
        if not source:
            continue
        if raw_entry.get("remove") is True:
            overrides.append({"source": source, "remove": True})
            continue
        target = to_simplified_text(
            normalize_space(str(raw_entry.get("target") or ""))
        )
        if target:
            overrides.append({"source": source, "target": target})
    return overrides


def apply_terminology_overrides(
    entries: Any,
    overrides: Any,
) -> List[Dict[str, str]]:
    glossary: Dict[str, Dict[str, str]] = {}
    register_terminology(glossary, entries)
    if not isinstance(overrides, list):
        return list(glossary.values())
    for raw in overrides:
        if not isinstance(raw, dict):
            continue
        source = normalize_space(str(raw.get("source") or ""))
        if source and raw.get("remove") is True:
            glossary.pop(source.casefold(), None)
            continue
        target = to_simplified_text(
            normalize_space(str(raw.get("target") or ""))
        )
        if not source or not target:
            continue
        glossary[source.casefold()] = {
            "source": source,
            "target": target,
        }
    return [dict(entry) for entry in glossary.values()]


def source_term_spans(source_text: str, source_term: str) -> List[Tuple[int, int]]:
    source = normalize_space(source_term)
    if not source:
        return []
    source_pattern = re.escape(source.casefold())
    if source.isascii() and any(character.isalnum() for character in source):
        source_pattern = rf"(?<!\w){source_pattern}(?!\w)"
    normalized_text = normalize_space(source_text).casefold()
    return [match.span() for match in re.finditer(source_pattern, normalized_text)]


def source_contains_term(source_text: str, source_term: str) -> bool:
    return bool(source_term_spans(source_text, source_term))


CONTEXT_DEPENDENT_TERMINOLOGY_SOURCES = {
    "sex",
}


def applicable_terminology_entries(
    entries: List[Dict[str, str]],
    source_text: str,
) -> List[Dict[str, str]]:
    candidates: List[Tuple[int, Dict[str, str], List[Tuple[int, int]], int]] = []
    for index, entry in enumerate(entries):
        source = normalize_space(str(entry.get("source") or ""))
        if source.casefold() in CONTEXT_DEPENDENT_TERMINOLOGY_SOURCES:
            continue
        spans = source_term_spans(source_text, source)
        if spans:
            candidates.append((index, entry, spans, len(source)))

    covered_spans: List[Tuple[int, int]] = []
    selected_indexes: set[int] = set()
    for index, _entry, spans, _source_length in sorted(
        candidates,
        key=lambda candidate: (-candidate[3], candidate[0]),
    ):
        independent_spans = [
            span
            for span in spans
            if not any(
                span[0] >= covered[0] and span[1] <= covered[1]
                for covered in covered_spans
            )
        ]
        if not independent_spans:
            continue
        selected_indexes.add(index)
        covered_spans.extend(independent_spans)

    return [
        entry
        for index, entry in enumerate(entries)
        if index in selected_indexes
    ]


def approved_terminology_entries(
    glossary: Dict[str, Dict[str, str]],
    proposed_entries: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    approved: List[Dict[str, str]] = []
    approved_keys: set[str] = set()
    for entry in proposed_entries:
        key = normalize_space(str(entry.get("source") or "")).casefold()
        resolved = glossary.get(key)
        if not key or resolved is None or key in approved_keys:
            continue
        approved.append(dict(resolved))
        approved_keys.add(key)
    return approved


def format_terminology_glossary(
    glossary: Dict[str, Dict[str, str]],
    relevant_source_text: str = "",
) -> str:
    if not glossary:
        return "(none yet)"
    values = list(glossary.values())
    relevant = applicable_terminology_entries(values, relevant_source_text)
    relevant_keys = {
        normalize_space(entry["source"]).casefold()
        for entry in relevant
    }
    shadowed_relevant_keys = {
        normalize_space(entry["source"]).casefold()
        for entry in values
        if source_contains_term(relevant_source_text, entry["source"])
    } - relevant_keys
    selected: List[Dict[str, str]] = []
    selected_keys: set[str] = set()
    for entry in relevant + values[-250:]:
        key = entry["source"].casefold()
        if key in selected_keys or key in shadowed_relevant_keys:
            continue
        selected.append(entry)
        selected_keys.add(key)
    return "\n".join(
        f"- {entry['source']} => {entry['target']}"
        for entry in selected
    )


def terminology_conflicts(
    glossary: Dict[str, Dict[str, str]],
    source_text: str,
    target_text: str,
    proposed_entries: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    combined: Dict[str, Dict[str, str]] = dict(glossary)
    for entry in proposed_entries or []:
        key = str(entry.get("source") or "").casefold()
        if key and key not in combined:
            combined[key] = entry

    conflicts: List[Dict[str, str]] = []
    for entry in applicable_terminology_entries(
        list(combined.values()),
        source_text,
    ):
        source = normalize_space(entry["source"])
        target = normalize_space(entry["target"])
        if not source or not target:
            continue
        if not terminology_target_present(target_text, target):
            conflicts.append(dict(entry))
    return conflicts


def terminology_target_present(target_text: str, required_target: str) -> bool:
    compact_text = re.sub(r"\s+", "", normalize_space(target_text)).casefold()
    compact_target = re.sub(r"\s+", "", normalize_space(required_target)).casefold()
    return bool(compact_target and compact_target in compact_text)


def canonicalize_approved_terminology_targets(
    source_text: str,
    target_text: str,
    glossary: Dict[str, Dict[str, str]],
) -> str:
    output = normalize_space(target_text)
    for entry in applicable_terminology_entries(
        list(glossary.values()),
        source_text,
    ):
        target = normalize_space(str(entry.get("target") or ""))
        compact_target = re.sub(r"\s+", "", target)
        if not compact_target:
            continue
        loose_pattern = r"\s*".join(re.escape(character) for character in compact_target)
        output = re.sub(loose_pattern, target, output, flags=re.IGNORECASE)
    return normalize_space(output)


def preserve_authored_chinese_terminology(
    segment: Segment,
    original_chinese: str,
    proposed_chinese: str,
    source_text: str,
    glossary: Dict[str, Dict[str, str]],
) -> str:
    proposed = canonicalize_approved_terminology_targets(
        source_text,
        proposed_chinese,
        glossary,
    )
    required = applicable_terminology_entries(
        list(glossary.values()),
        source_text,
    )
    missing = [
        entry
        for entry in required
        if not terminology_target_present(
            proposed,
            str(entry.get("target") or ""),
        )
    ]
    if (
        missing
        and str(segment.get("chinese_text_authority") or "") == "authored"
        and all(
            terminology_target_present(
                original_chinese,
                str(entry.get("target") or ""),
            )
            or authored_chinese_satisfies_technical_term(
                segment,
                original_chinese,
                entry,
            )
            for entry in missing
        )
    ):
        return canonicalize_approved_terminology_targets(
            source_text,
            original_chinese,
            glossary,
        )
    return proposed


def authored_chinese_satisfies_technical_term(
    segment: Segment,
    original_chinese: str,
    entry: Dict[str, str],
) -> bool:
    if str(segment.get("chinese_text_authority") or "") != "authored":
        return False
    source = normalize_space(str(entry.get("source") or ""))
    target = re.sub(r"[\W_]+", "", str(entry.get("target") or ""), flags=re.UNICODE)
    original = re.sub(r"[\W_]+", "", original_chinese, flags=re.UNICODE)
    if (
        not source
        or source != source.casefold()
        or not any("a" <= character <= "z" for character in source)
        or len(target) < 4
        or not original
    ):
        return False

    positions: List[int] = []
    cursor = 0
    for character in target.casefold():
        position = original.casefold().find(character, cursor)
        if position < 0:
            return False
        positions.append(position)
        cursor = position + 1
    return positions[-1] - positions[0] + 1 <= len(target) + 8


def filter_authored_chinese_terminology_conflicts(
    segment: Segment,
    original_chinese: str,
    conflicts: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    return [
        entry
        for entry in conflicts
        if not authored_chinese_satisfies_technical_term(
            segment,
            original_chinese,
            entry,
        )
    ]


SPEAKER_LABEL_RE = re.compile(
    r"^\s*[-\u2013\u2014]?\s*(?P<label>[A-Za-z][A-Za-z0-9 .'\-]{0,48}):"
)
CJK_SPEAKER_LABEL_RE = re.compile(
    r"^\s*[-\u2013\u2014]?\s*[^:：\n]{1,24}[：:]"
)
SUBTITLE_CUE_RE = re.compile(
    r"\([^()\n]{1,96}\)|\[[^\[\]\n]{1,96}\]|"
    r"（[^（）\n]{1,96}）|【[^【】\n]{1,96}】"
)
OUTER_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
)


def restore_glossary_speaker_label(
    source_text: str,
    target_text: str,
    glossary: Dict[str, Dict[str, str]],
) -> str:
    target = normalize_space(target_text)
    match = SPEAKER_LABEL_RE.match(normalize_space(source_text))
    if not target or not match or not glossary:
        return target

    speaker_label = match.group("label")
    mappings = [
        entry
        for entry in glossary.values()
        if source_contains_term(speaker_label, entry["source"])
    ]
    missing_targets = [
        normalize_space(entry["target"])
        for entry in mappings
        if normalize_space(entry["target"])
        and normalize_space(entry["target"]).casefold() not in target.casefold()
    ]
    if not missing_targets:
        return target
    speaker_prefix = "\u3001".join(missing_targets)
    return f"{speaker_prefix}\uff1a{target}"


def has_subtitle_speaker_label(text: str) -> bool:
    normalized = normalize_space(text)
    return bool(
        SPEAKER_LABEL_RE.match(normalized)
        or CJK_SPEAKER_LABEL_RE.match(normalized)
    )


def strip_added_outer_quotes(original: str, proposed: str) -> str:
    original_text = normalize_space(original)
    proposed_text = normalize_space(proposed)
    for opening, closing in OUTER_QUOTE_PAIRS:
        original_wrapped = (
            len(original_text) >= len(opening) + len(closing)
            and original_text.startswith(opening)
            and original_text.endswith(closing)
        )
        proposed_wrapped = (
            len(proposed_text) >= len(opening) + len(closing)
            and proposed_text.startswith(opening)
            and proposed_text.endswith(closing)
        )
        if proposed_wrapped and not original_wrapped:
            return normalize_space(
                proposed_text[len(opening) : len(proposed_text) - len(closing)]
            )
    return proposed_text


def preserve_final_qa_track_structure(original: str, proposed: str) -> str:
    original_text = clean_subtitle_text(original)
    proposed_text = strip_added_outer_quotes(
        original_text,
        clean_subtitle_text(proposed),
    )
    if not original_text:
        return proposed_text
    if (
        has_subtitle_speaker_label(original_text)
        and not has_subtitle_speaker_label(proposed_text)
    ):
        return original_text
    if len(SUBTITLE_CUE_RE.findall(proposed_text)) < len(
        SUBTITLE_CUE_RE.findall(original_text)
    ):
        return original_text
    return proposed_text


def write_terminology_artifact(
    path: Optional[Path],
    glossary: Dict[str, Dict[str, str]],
    llm_model: str,
) -> None:
    if path is None:
        return
    write_json_atomic(
        path,
        {
            "version": TERMINOLOGY_POLICY_VERSION,
            "processing_policy_version": PROCESSING_POLICY_VERSION,
            "llm_model": llm_model,
            "entries": list(glossary.values()),
        },
    )


def style_guide_fingerprint(
    segments: List[Segment],
    llm_model: str,
) -> str:
    source = [
        {
            "id": segment.get("id"),
            "text": clean_subtitle_text(str(segment.get("text") or "")),
            "en": clean_subtitle_text(str(segment.get("en") or "")),
            "zh": clean_subtitle_text(str(segment.get("zh") or "")),
            "source_language": str(segment.get("source_language") or ""),
        }
        for segment in segments
    ]
    serialized = json.dumps(
        {
            "policy_version": STYLE_GUIDE_POLICY_VERSION,
            "llm_model": llm_model,
            "segments": source,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def style_guide_sample(
    segments: List[Segment],
    *,
    max_items: int = 180,
    max_chars: int = 24000,
) -> str:
    if not segments:
        return "(no subtitles)"
    if len(segments) <= max_items:
        indices = list(range(len(segments)))
    else:
        indices = sorted(
            {
                round(index * (len(segments) - 1) / (max_items - 1))
                for index in range(max_items)
            }
        )
    rows: List[str] = []
    used = 0
    for index in indices:
        segment = segments[index]
        source = clean_subtitle_text(
            str(segment.get("en") or segment.get("text") or "")
        )
        chinese = clean_subtitle_text(str(segment.get("zh") or ""))
        row = f"[{index}] SOURCE: {source or '-'} | ZH: {chinese or '-'}"
        if rows and used + len(row) + 1 > max_chars:
            break
        rows.append(row)
        used += len(row) + 1
    return "\n".join(rows)


def normalize_style_guide(
    raw: Dict[str, Any],
    *,
    llm_model: str,
    input_fingerprint: str,
) -> Dict[str, Any]:
    def text_field(name: str, fallback: str) -> str:
        value = normalize_space(str(raw.get(name) or fallback))
        return value[:600]

    terminology = extract_terminology_entries(
        {"terminology": raw.get("terminology")}
    )
    return {
        "version": STYLE_GUIDE_POLICY_VERSION,
        "status": "ready",
        "llm_model": llm_model,
        "input_fingerprint": input_fingerprint,
        "register": text_field(
            "register",
            "Natural, concise spoken Simplified Chinese suitable for film subtitles.",
        ),
        "name_policy": text_field(
            "name_policy",
            "Use one movie-wide rendering for every recurring personal name.",
        ),
        "address_policy": text_field(
            "address_policy",
            "Translate forms of address consistently according to relationships.",
        ),
        "sdh_policy": text_field(
            "sdh_policy",
            "When Chinese lacks an English-only SDH cue, add a concise Chinese translation.",
        ),
        "punctuation_policy": text_field(
            "punctuation_policy",
            "Use concise Chinese subtitle punctuation and avoid unnecessary sentence-final marks.",
        ),
        "terminology": terminology,
    }


def analyze_movie_style(
    segments: List[Segment],
    llm_model: str,
    artifact_path: Path,
) -> Dict[str, Any]:
    input_fingerprint = style_guide_fingerprint(segments, llm_model)
    if artifact_path.exists():
        try:
            cached = json.loads(artifact_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("version") == STYLE_GUIDE_POLICY_VERSION
                and cached.get("status") in {"ready", "unavailable"}
                and cached.get("llm_model") == llm_model
                and cached.get("input_fingerprint") == input_fingerprint
            ):
                print(f"Using cached movie-wide style guide: {artifact_path}")
                return cached
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    prompt = f"""
Analyze this representative, movie-wide subtitle sample before line-by-line work.

Return one JSON object that defines a concise professional style guide for the entire movie.
Identify recurring personal names and important recurring terms only when the sample supports them.

=== MOVIE-WIDE SAMPLE ===
{style_guide_sample(segments)}

Required decisions:
- register: conversational/formal/technical tone and how concise the Chinese should be.
- name_policy: one consistent treatment for names; never alternate translated and untranslated forms.
- address_policy: consistent Chinese forms of address and relationship-sensitive pronouns.
- sdh_policy: English-only SDH, music, laughter, speaker labels, and sound descriptions must be translated into concise Chinese only when Chinese lacks that information.
- punctuation_policy: punctuation suitable for professional on-screen Simplified Chinese.
- terminology: only recurring names or terms with a stable source-to-Chinese mapping.

    Do not edit individual subtitle lines. Do not add markdown or explanations.
"""
    response_schema = movie_style_schema()
    raw: Any = None
    for attempt in range(3):
        try:
            raw = parse_json_response(
                call_llm(
                    prompt,
                    (
                        "You are the lead subtitle editor establishing a movie-wide style bible "
                        "before independent translators begin."
                    ),
                    llm_model,
                    role="proofread",
                    response_schema=response_schema,
                    seed_offset=attempt,
                ),
                response_schema=response_schema,
            )
            break
        except Exception as exc:
            print(
                "Movie-wide style analysis failed "
                f"(attempt {attempt + 1}/3): {exc}"
            )
            if attempt == 2:
                raise
            time.sleep(2)
    if not isinstance(raw, dict):
        raise ValueError("Movie-wide style analysis did not return a JSON object")
    style_guide = normalize_style_guide(
        raw,
        llm_model=llm_model,
        input_fingerprint=input_fingerprint,
    )
    write_json_atomic(artifact_path, style_guide)
    return style_guide


def format_style_guide(style_guide: Optional[Dict[str, Any]]) -> str:
    if not isinstance(style_guide, dict) or style_guide.get("status") != "ready":
        return "(default professional subtitle style)"
    return "\n".join(
        [
            f"- Register: {style_guide.get('register') or '-'}",
            f"- Names: {style_guide.get('name_policy') or '-'}",
            f"- Address: {style_guide.get('address_policy') or '-'}",
            f"- SDH: {style_guide.get('sdh_policy') or '-'}",
            f"- Punctuation: {style_guide.get('punctuation_policy') or '-'}",
        ]
    )


def parse_display_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "hidden", "hide", "omit", "skip"}
    return True


def apply_display_timing(
    segment: Segment,
    item: Dict[str, Any],
    context_start: float,
    context_end: float,
) -> Segment:
    output = dict(segment)
    display = (
        False
        if segment.get("non_content_credit")
        else parse_display_flag(
            item.get("display", item.get("show", item.get("keep", True)))
        )
    )
    output["display"] = display
    if display:
        output.pop("model_display_requested", None)
        output.pop("display_guard", None)
        output.pop("display_guard_reason", None)
    else:
        output["model_display_requested"] = False
    output.pop("display_start", None)
    output.pop("display_end", None)
    return output


def extend_display_over_hidden_segments(
    segments: List[Segment],
    max_gap: float = 2.0,
    max_duration: float = 6.0,
) -> None:
    last_visible: Optional[Segment] = None
    for segment in segments:
        if segment.get("display", True) is False:
            if last_visible is not None:
                visible_end = float(last_visible.get("display_end", last_visible["end"]))
                gap = float(segment["start"]) - visible_end
                if gap <= max_gap:
                    latest_end = float(last_visible.get("display_start", last_visible["start"])) + max_duration
                    extended_end = min(float(segment["end"]), latest_end)
                    if extended_end > visible_end:
                        last_visible["display_end"] = extended_end
            continue
        last_visible = segment


def sanitize_checkpoint_display_timing(segments: List[Segment]) -> List[Segment]:
    sanitized: List[Segment] = []
    for segment in segments:
        item = dict(segment)
        item.pop("display_start", None)
        item.pop("display_end", None)
        sanitized.append(item)
    extend_display_over_hidden_segments(sanitized)
    return sanitized


def is_language_valid(
    original_text: str,
    corrected: str,
    translated: str,
    require_corrected: bool = True,
    original_translation: str = "",
) -> bool:
    original_text = clean_subtitle_text(original_text)
    corrected = clean_subtitle_text(corrected)
    translated = clean_subtitle_text(translated)
    original_translation = clean_subtitle_text(original_translation)
    if require_corrected and not corrected:
        return False
    if not translated:
        return False

    orig_cjk_count = len(CJK_RE.findall(original_text))
    corr_cjk_count = len(CJK_RE.findall(corrected))
    trans_has_cjk = contains_cjk(translated)

    # 英文不是英文：如果修正后的中文字符数量超过了原始源文中的中文字符数量，说明LLM在擅自翻译或添加中文
    if corr_cjk_count > orig_cjk_count:
        return False
        
    # 中文不是中文：译文无中文字符，但包含多个字母（说明可能是生硬复制了外文句子或根本没翻译）
    if not trans_has_cjk and any(character.isalpha() for character in translated):
        original_token = "".join(
            character.casefold()
            for character in normalize_space(original_translation)
            if character.isalnum()
        )
        translated_token = "".join(
            character.casefold()
            for character in normalize_space(translated)
            if character.isalnum()
        )
        preserved_existing_token = (
            bool(original_token)
            and original_token == translated_token
            and len(original_token) <= 12
            and source_contains_term(original_text, original_token)
        )
        if not preserved_existing_token:
            return False
        
    return True


def retry_single_segment_llm(
    segment: Segment,
    before_text: str,
    after_text: str,
    next_start: Optional[float],
    context_start: float,
    context_end: float,
    system_prompt: str,
    llm_model: str,
    language_label: str,
    is_proofread: bool = False,
    terminology_text: str = "(none yet)",
    terminology_glossary: Optional[Dict[str, Dict[str, str]]] = None,
    require_display: bool = False,
) -> Dict[str, Any]:
    print(
        "Detected an invalid visible subtitle; retrying one item: "
        f"{ascii(str(segment.get('text') or ''))}"
    )
    display_requirement = (
        "- This line is not covered by another visible subtitle. "
        'You MUST set "display": true and return complete visible text.'
        if require_display
        else ""
    )
    if is_proofread:
        batch_text = format_bilingual_prompt_segment(
            0,
            segment,
            next_start=next_start,
        )
        prompt = f"""
You will proofread only the TARGET LINE(S).

These subtitles may already contain Chinese. Proofread the existing source-language and Chinese tracks.
The source track may also be OCR output and may contain recognition errors.
Do not retranslate a non-empty Chinese line. Translate source-track information only when the Chinese line is empty or incomplete.

=== PREVIOUS CONTEXT, REFERENCE ONLY ===
{before_text}

=== APPROVED MOVIE-WIDE TERMINOLOGY, USE EXACTLY ===
{terminology_text}

=== TARGET LINE(S) TO OUTPUT ===
{batch_text}

=== FOLLOWING CONTEXT, REFERENCE ONLY ===
{after_text}

Return a raw JSON list with exactly 1 objects.
Each object must correspond to one TARGET line and keep the same local index.
Schema:
[
  {{
    "index": 0,
    "corrected_english": "...",
    "corrected_chinese": "...",
    "display": true,
    "terminology": [{{"source": "John", "target": "约翰"}}]
  }}
]

Rules:
- One input line must still produce one output object for checkpointing.
{display_requirement}
- Correct obvious source-language and Chinese OCR/subtitle recognition errors.
- When SOURCE_AUTHORITY=ocr and the Chinese lane is authored, use the authored Chinese meaning to reconstruct fluent, grammatical source-language text instead of preserving nonsensical OCR wording.
- Correct only the content present inside this TARGET timestamp. Fragments are valid: never copy, complete, or move words or meaning from PREVIOUS/FOLLOWING CONTEXT into the target.
- When SOURCE_AUTHORITY=authored, copy the source-language text unchanged into corrected_english.
- Only ASR/OCR events may be hidden as proven duplicate loops; authored subtitle events must remain visible.
- Convert Traditional Chinese to natural Simplified Chinese.
- If corrected_chinese is already non-empty and complete, preserve its meaning and only proofread it. Do not replace it with a fresh translation.
- Translate source-track information into corrected_chinese only when the original Chinese is empty or clearly incomplete.
- Preserve every unique fact, clause, speaker label, number, and sound cue inside this TARGET timestamp. Never summarize or omit content to make the object shorter.
- This editing object may span several seconds and will be split into readable display cues after proofreading. Do not enforce final line-length limits here.
- `corrected_english` is a legacy field name for the source track. Keep it in SOURCE_LANGUAGE and never translate it into English.
- If the source contains SDH/non-speech cues such as music, applause, laughter, or speaker labels and the Chinese line lacks that information, add only that missing cue in concise Simplified Chinese.
- You MUST translate uppercase descriptive text in parentheses or brackets (e.g. "(SIGHS)" or "[MUSIC]") into Simplified Chinese.
- Keep names and recurring terminology consistent across the context. Do not translate a name in one line and leave the same name untranslated in another unless the context clearly requires it.
- Reuse every applicable APPROVED MOVIE-WIDE TERMINOLOGY mapping exactly.
- In terminology, return only personal names or recurring terms that occur in this target line. Use an empty list when there are none.
- Never output or change timestamps. Timing is enforced by the subtitle pipeline.
- Keep corrected_english empty only when no source track exists.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
"""
    else:
        batch_text = format_prompt_segment(
            0,
            segment,
            next_start=next_start,
        )
        prompt = f"""
You will correct and translate only the TARGET LINE(S).

Use the previous and following context to understand names, pronouns, topic continuity, and terminology.
Do not translate the context sections. They are reference only.
The source subtitle/audio language is: {language_label}.

=== PREVIOUS CONTEXT, REFERENCE ONLY ===
{before_text}

=== APPROVED MOVIE-WIDE TERMINOLOGY, USE EXACTLY ===
{terminology_text}

=== TARGET LINE(S) TO OUTPUT ===
{batch_text}

=== FOLLOWING CONTEXT, REFERENCE ONLY ===
{after_text}

Return a raw JSON list with exactly 1 objects.
Each object must correspond to one TARGET line and keep the same local index.
Schema:
[
  {{
    "index": 0,
    "corrected_text": "...",
    "chinese_translation": "...",
    "display": true,
    "terminology": [{{"source": "John", "target": "约翰"}}]
  }}
]

Rules:
- One input line must still produce one output object for checkpointing.
{display_requirement}
- Never output or change timestamps. Timing is enforced by the subtitle pipeline.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
- Correct obvious ASR/OCR/subtitle errors in the source text before translating.
- Correct only the content present inside this TARGET timestamp. Fragments are valid: never copy, complete, or move words or meaning from PREVIOUS/FOLLOWING CONTEXT into the target.
- When SOURCE_AUTHORITY=authored, copy the source text unchanged into corrected_text.
- When SOURCE_AUTHORITY=ocr but RECOGNITION_RISK=no_recognition_risk, preserve the source unless the context proves an error.
- Only ASR/OCR events may be hidden as proven duplicate loops; authored subtitle events must remain visible.
- Keep corrected_text in the original source language.
- Translate into natural Simplified Chinese.
- Preserve every unique fact, clause, speaker label, number, and sound cue inside this TARGET timestamp. Never summarize or omit content to make the object shorter.
- This editing object may span several seconds and will be split into readable display cues after translation. Do not enforce final line-length limits here.
- Keep personal names and recurring terminology consistent across the context. Do not translate a name in one line and leave the same name untranslated in another unless the context clearly requires it.
- Reuse every applicable APPROVED MOVIE-WIDE TERMINOLOGY mapping exactly.
- In terminology, return only personal names or recurring terms that occur in this target line. Use an empty list when there are none.
"""
    if is_proofread:
        original_source, original_chinese = original_en_zh(segment)
        terminology_source = original_source or original_chinese
    else:
        terminology_source = str(segment.get("text") or "")
    required_terminology = applicable_terminology_entries(
        list((terminology_glossary or {}).values()),
        terminology_source,
    )
    if required_terminology:
        required_terminology_text = "\n".join(
            f"- {entry['source']} => {entry['target']}"
            for entry in required_terminology
        )
        prompt += f"""

=== REQUIRED TERMINOLOGY FOR THIS TARGET ===
{required_terminology_text}

Every exact target string above must appear in the Chinese output. Do not shorten
a full name to only a surname. These required names take precedence over the
soft character target when both cannot be satisfied.
"""
    required_authored_cues: List[str] = []
    if is_proofread:
        _, original_chinese = original_en_zh(segment)
        if str(segment.get("chinese_text_authority") or "") == "authored":
            required_authored_cues = AUTHORED_CHINESE_CUE_RE.findall(
                original_chinese
            )
    if required_authored_cues:
        cue_text = "\n".join(f"- {cue}" for cue in required_authored_cues)
        prompt += f"""

=== REQUIRED AUTHORED CHINESE LABELS OR CUES ===
{cue_text}

Preserve the complete meaning of every label or cue above. You may normalize
brackets and punctuation, but must not omit a speaker name, role, or sound cue.
"""
    response_role = "proofread" if is_proofread else "translation"
    response_schema = indexed_subtitle_schema(1, response_role)
    validation_feedback = ""
    last_safe_proofread_item: Optional[Dict[str, Any]] = None
    for attempt in range(10):
        try:
            attempt_prompt = prompt
            if validation_feedback:
                attempt_prompt += f"""

=== AUTOMATIC VALIDATION FEEDBACK ===
{validation_feedback}

Correct the listed defect in this attempt. Return the complete subtitle object,
not an explanation.
"""
            data = parse_json_response(
                call_llm(
                    attempt_prompt,
                    system_prompt,
                    llm_model,
                    role=response_role,
                    response_schema=response_schema,
                    seed_offset=attempt,
                ),
                response_schema=response_schema,
            )
            lookup = validate_indexed_batch_response(
                data,
                1,
                response_role,
            )
            if lookup:
                item = lookup[0]
                if not parse_display_flag(item.get("display", True)):
                    if require_display:
                        print(
                            f"Single-item retry {attempt + 1} still requested "
                            "hiding a unique subtitle; retrying."
                        )
                        validation_feedback = (
                            'The previous output set "display" to false. This '
                            'unique subtitle must use "display": true.'
                        )
                        continue
                    return item
                if is_proofread:
                    orig_en, orig_zh = original_en_zh(segment)
                    corrected = (
                        preserve_authored_source(
                            segment,
                            orig_en,
                            clean_subtitle_text(
                                str(item.get("corrected_english") or orig_en)
                            ),
                        )
                        if orig_en
                        else ""
                    )
                    translated = to_simplified_text(
                        clean_subtitle_text(
                            str(
                                item.get("corrected_chinese")
                                or item.get("chinese_translation")
                                or ""
                            )
                        )
                    )
                    translated = restore_glossary_speaker_label(
                        corrected or orig_en,
                        translated,
                        terminology_glossary or {},
                    )
                    translated = preserve_authored_chinese_terminology(
                        segment,
                        orig_zh,
                        translated,
                        corrected or orig_en or orig_zh,
                        terminology_glossary or {},
                    )
                    item["corrected_chinese"] = translated
                    proposed_terminology = applicable_terminology_entries(
                        extract_terminology_entries(item),
                        corrected or orig_en or orig_zh,
                    )
                    language_preserved = source_language_preserved(
                        segment,
                        orig_en,
                        corrected,
                    )
                    language_valid = is_language_valid(
                        orig_en or orig_zh,
                        corrected,
                        translated,
                        require_corrected=bool(orig_en),
                        original_translation=orig_zh,
                    )
                    content_retained = authored_chinese_content_retained(
                        segment,
                        orig_zh,
                        translated,
                    )
                    conflicts = terminology_conflicts(
                        terminology_glossary or {},
                        corrected or orig_en or orig_zh,
                        translated,
                        proposed_terminology,
                    )
                    conflicts = filter_authored_chinese_terminology_conflicts(
                        segment,
                        orig_zh,
                        conflicts,
                    )
                    if language_preserved and language_valid and not conflicts:
                        last_safe_proofread_item = dict(item)
                    if (
                        language_preserved
                        and language_valid
                        and content_retained
                        and not conflicts
                    ):
                        return item
                else:
                    corrected = clean_subtitle_text(
                        str(
                            item.get("corrected_text")
                            or item.get("corrected_source")
                            or segment.get("text", "")
                        )
                    )
                    translated = to_simplified_text(
                        clean_subtitle_text(
                            str(item.get("chinese_translation") or "")
                        )
                    )
                    translated = restore_glossary_speaker_label(
                        corrected,
                        translated,
                        terminology_glossary or {},
                    )
                    item["chinese_translation"] = translated
                    proposed_terminology = applicable_terminology_entries(
                        extract_terminology_entries(item),
                        corrected,
                    )
                    language_preserved = source_language_preserved(
                        segment,
                        str(segment.get("text") or ""),
                        corrected,
                    )
                    language_valid = is_language_valid(
                        str(segment.get("text") or ""),
                        corrected,
                        translated,
                    )
                    conflicts = terminology_conflicts(
                        terminology_glossary or {},
                        corrected,
                        translated,
                        proposed_terminology,
                    )
                    if language_preserved and language_valid and not conflicts:
                        return item
                feedback: List[str] = []
                if not language_preserved:
                    feedback.append(
                        "The source-language text changed language or script; "
                        "keep it in the original language."
                    )
                if not language_valid:
                    feedback.append(
                        "The Chinese output or source-language output failed "
                        "language validation; return both complete tracks."
                    )
                if is_proofread and not content_retained:
                    cue_detail = (
                        " Required labels or cues: "
                        + "; ".join(required_authored_cues)
                        + "."
                        if required_authored_cues
                        else ""
                    )
                    feedback.append(
                        "The previous Chinese output omitted too much authored "
                        "content. Preserve every unique fact, clause, label, "
                        "number, and cue from this target."
                        + cue_detail
                    )
                if conflicts:
                    missing = ", ".join(
                        f"{entry['source']} => {entry['target']}"
                        for entry in conflicts
                    )
                    print(
                        f"Single-item retry {attempt + 1} omitted required "
                        f"terminology: {missing}"
                    )
                    feedback.append(
                        "The Chinese output omitted these required exact "
                        f"mappings: {missing}."
                    )
                validation_feedback = "\n".join(feedback)
            print(
                f"Single-item retry {attempt + 1} still failed language "
                "validation; retrying."
            )
        except Exception as exc:
            print(f"Single-item retry request failed: {ascii(str(exc))}")
            validation_feedback = (
                "The previous response was not valid against the required JSON "
                "schema. Return only one complete raw JSON object in the list."
            )
    
    if is_proofread and last_safe_proofread_item:
        original_source, original_chinese = original_en_zh(segment)
        fallback_chinese = preserve_authored_chinese_terminology(
            segment,
            original_chinese,
            to_simplified_text(clean_subtitle_text(original_chinese)),
            str(
                last_safe_proofread_item.get("corrected_english")
                or original_source
                or original_chinese
            ),
            terminology_glossary or {},
        )
        fallback_item = dict(last_safe_proofread_item)
        fallback_item["corrected_chinese"] = fallback_chinese
        if authored_chinese_content_retained(
            segment,
            original_chinese,
            fallback_chinese,
        ):
            print(
                "All 10 single-item retries omitted authored content; "
                "preserving the original Chinese track and continuing."
            )
            return fallback_item
    print(
        "All 10 single-item retries failed; stopping the current batch "
        "and preserving the latest checkpoint."
    )
    return {}


def translate_and_correct_segments(
    segments: List[Segment],
    llm_model: str,
    batch_size: int,
    context_lines: int,
    source_language: str,
    checkpoint_path: Optional[Path] = None,
    terminology_path: Optional[Path] = None,
    style_guide: Optional[Dict[str, Any]] = None,
) -> List[Segment]:
    print("Starting sentence-level LLM correction and translation...")
    language_label = "the source language" if not source_language or source_language == "auto" else source_language
    style_text = format_style_guide(style_guide)
    editing_policy = source_editing_policy(segments)
    system_prompt = (
        "You are a professional subtitle editor and Chinese translator. "
        f"{editing_policy} "
        "Preserve source timing anchors. "
        f"Follow this movie-wide style guide exactly:\n{style_text}"
    )

    processed_segments: List[Segment] = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, list)
                and len(cached) <= len(segments)
                and checkpoint_matches_segments(
                    cached,
                    segments,
                    expected_mode="translate",
                    expected_model=llm_model,
                    expected_policy_version=PROCESSING_POLICY_VERSION,
                    expected_terminology_policy_version=TERMINOLOGY_POLICY_VERSION,
                    expected_plan_fingerprint=str(
                        segments[0].get("processing_plan_fingerprint") or ""
                    )
                    if segments
                    else "",
                )
            ):
                processed_segments = sanitize_checkpoint_display_timing(cached)
                processed_segments = guard_model_hidden_segments(
                    segments[: len(processed_segments)],
                    processed_segments,
                )
                print(f"Resuming from checkpoint with {len(processed_segments)} completed segments.")
            else:
                print("Ignoring checkpoint because it does not match the current subtitle segmentation.")
        except Exception as exc:
            print(f"Ignoring unreadable checkpoint {checkpoint_path}: {exc}")

    start_index = len(processed_segments)
    repair_index = first_guarded_language_repair_index(
        segments,
        processed_segments,
        proofread=False,
    )
    if repair_index is not None:
        start_index = min(start_index, repair_index)
        print(
            "Checkpoint contains a restored unique subtitle without a valid "
            f"Chinese translation; reprocessing from item {start_index + 1}."
        )
    if start_index < len(segments):
        start_index -= start_index % batch_size
        processed_segments = processed_segments[:start_index]
    glossary = terminology_from_segments(
        processed_segments,
        (style_guide or {}).get("terminology"),
    )
    write_terminology_artifact(terminology_path, glossary, llm_model)

    for i in range(start_index, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        print(f"Processing segments {i + 1} to {i + len(batch)} of {len(segments)}...")

        before_text, after_text = build_context_text(segments, i, i + len(batch), context_lines)
        approved_output_context = build_approved_output_context(processed_segments, context_lines)
        batch_source_text = "\n".join(str(segment.get("text") or "") for segment in batch)
        terminology_text = format_terminology_glossary(glossary, batch_source_text)
        batch_text = "\n".join(
            format_prompt_segment(
                j,
                segment,
                next_start=(
                    float(segments[i + j + 1]["start"])
                    if i + j + 1 < len(segments)
                    else None
                ),
            )
            for j, segment in enumerate(batch)
        )
        context_start_index = max(0, i - context_lines)
        context_end_index = min(len(segments), i + len(batch) + context_lines)
        context_start = float(segments[context_start_index]["start"])
        context_end = float(segments[context_end_index - 1]["end"])

        prompt = f"""
You will correct and translate only the TARGET LINE(S).

Use the previous and following context to understand names, pronouns, topic continuity, and terminology.
Do not translate the context sections. They are reference only.
The source subtitle/audio language is: {language_label}.

=== PREVIOUS CONTEXT, REFERENCE ONLY ===
{before_text}

=== PREVIOUS APPROVED OUTPUT, MATCH NAMES AND TERMINOLOGY ===
{approved_output_context}

=== MOVIE-WIDE STYLE GUIDE, FOLLOW THROUGHOUT ===
{style_text}

=== SOURCE AUTHORITY POLICY ===
{editing_policy}

=== APPROVED MOVIE-WIDE TERMINOLOGY, USE EXACTLY ===
{terminology_text}

=== TARGET LINE(S) TO OUTPUT ===
{batch_text}

=== FOLLOWING CONTEXT, REFERENCE ONLY ===
{after_text}

Return a raw JSON list with exactly {len(batch)} objects.
Each object must correspond to one TARGET line and keep the same local index.
Schema:
[
  {{
    "index": 0,
    "corrected_text": "...",
    "chinese_translation": "...",
    "display": true,
    "terminology": [{{"source": "John", "target": "约翰"}}]
  }}
]

Rules:
- One input line must still produce one output object for checkpointing.
- Hide a later TARGET line only when it is a proven near-exact duplicate already fully represented by an overlapping visible cue.
- Keep genuine continuations as separate timed fragments. Never complete an earlier object with words or meaning from a later TARGET line.
- Never output or change timestamps. Timing is enforced by the subtitle pipeline.
- If a TARGET line only repeats or continues a sentence already clearly represented in PREVIOUS CONTEXT, set "display": false.
- If two TARGET lines are genuine consecutive new information, keep both visible.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
- Correct obvious ASR/OCR/subtitle errors in the source text before translating.
- Correct only the content present inside each TARGET timestamp. Context is evidence for recognition and terminology, not text to copy into the target.
- When SOURCE_AUTHORITY=authored, copy the source text unchanged into corrected_text.
- When SOURCE_AUTHORITY=ocr but RECOGNITION_RISK=no_recognition_risk, preserve the source unless the context proves an error.
- Only ASR/OCR events may be hidden as proven duplicate loops; authored subtitle events must remain visible.
- Keep corrected_text in the original source language.
- Translate into natural Simplified Chinese.
- You MUST translate uppercase descriptive text in parentheses or brackets (e.g. "(SIGHS)" or "[MUSIC]") into Simplified Chinese.
- Preserve every unique fact, clause, speaker label, number, and sound cue inside each TARGET timestamp. Never summarize or omit content to make an object shorter.
- A TARGET object may span several seconds and will be split into readable display cues after translation. Do not enforce final line-length limits in this editing response.
- Before returning, verify that object index N contains only the meaning of TARGET [N], never the preceding or following target.
- Keep personal names and recurring terminology consistent across the context. Do not translate a name in one line and leave the same name untranslated in another unless the context clearly requires it.
- Reuse every applicable APPROVED MOVIE-WIDE TERMINOLOGY mapping exactly.
- In terminology, return only personal names or recurring terms that occur in that target line. Once a mapping is approved, never propose a different target. Use an empty list when there are none.
"""

        try:
            lookup: Optional[Dict[int, Dict[str, Any]]] = None
            response_schema = indexed_subtitle_schema(
                len(batch),
                "translation",
            )
            for attempt in range(3):
                try:
                    data = parse_json_response(
                        call_llm(
                            prompt,
                            system_prompt,
                            llm_model,
                            role="translation",
                            response_schema=response_schema,
                            seed_offset=attempt,
                        ),
                        response_schema=response_schema,
                    )
                    lookup = validate_indexed_batch_response(
                        data,
                        len(batch),
                        "translation",
                    )
                    break
                except Exception as exc:
                    print(f"LLM request failed (attempt {attempt+1}/3): {exc}")
                    if attempt < 2:
                        import time
                        time.sleep(2)
                    else:
                        raise RuntimeError(
                            "网络、大模型服务或响应格式连续失败，已中断翻译；"
                            "修复后再次运行可从最近 checkpoint 继续。"
                        ) from exc
            if lookup is None:
                raise RuntimeError("LLM response validation did not produce a complete batch")

            batch_processed: List[Segment] = []
            for j, segment in enumerate(batch):
                item = lookup[j]
                corrected_val = str(
                    item.get("corrected_text")
                    or item.get("corrected_source")
                    or item.get("corrected_english")
                    or segment["text"]
                )
                corrected = preserve_authored_source(
                    segment,
                    str(segment.get("text") or ""),
                    corrected_val,
                )
                translated_val = str(item.get("chinese_translation") or "")
                translated = restore_glossary_speaker_label(
                    corrected,
                    clean_subtitle_text(translated_val),
                    glossary,
                )
                candidate_metadata = {
                    **segment,
                    "en": corrected,
                    "zh": translated,
                    "processing_mode": "translate",
                    "processing_policy_version": PROCESSING_POLICY_VERSION,
                    "terminology_policy_version": TERMINOLOGY_POLICY_VERSION,
                    "llm_model": llm_model,
                    "llm_role": "translation",
                    "llm_generation_profile": asdict(
                        generation_profile("translation")
                    ),
                }
                guarded_candidate = guarded_display_candidate(
                    segments,
                    [*processed_segments, *batch_processed],
                    candidate_metadata,
                    item,
                    context_start,
                    context_end,
                )
                display = guarded_candidate.get("display", True) is not False
                proposed_terminology = applicable_terminology_entries(
                    extract_terminology_entries(item),
                    corrected,
                )
                conflicts = terminology_conflicts(
                    glossary,
                    corrected,
                    translated,
                    proposed_terminology,
                )

                if display and (
                    not source_language_preserved(
                        segment,
                        str(segment.get("text") or ""),
                        corrected,
                    )
                    or not is_language_valid(segment["text"], corrected, translated)
                    or conflicts
                ):
                    forced_model_hide = (
                        guarded_candidate.get("display_guard") == "forced_visible"
                    )
                    retry_item = retry_single_segment_llm(
                        segment,
                        f"{before_text}\n\nPREVIOUS APPROVED OUTPUT:\n{approved_output_context}",
                        after_text,
                        (
                            float(segments[i + j + 1]["start"])
                            if i + j + 1 < len(segments)
                            else None
                        ),
                        context_start,
                        context_end,
                        system_prompt,
                        llm_model,
                        language_label,
                        is_proofread=False,
                        terminology_text=format_terminology_glossary(glossary, corrected),
                        terminology_glossary=glossary,
                        require_display=forced_model_hide,
                    )
                    if not retry_item:
                        raise RuntimeError(f"LLM could not produce a valid translation for segment {i + j}")
                    item.update(retry_item)
                    corrected_val = str(
                        item.get("corrected_text")
                        or item.get("corrected_source")
                        or item.get("corrected_english")
                        or segment["text"]
                    )
                    corrected = preserve_authored_source(
                        segment,
                        str(segment.get("text") or ""),
                        corrected_val,
                    )
                    translated_val = str(item.get("chinese_translation") or "")
                    translated = restore_glossary_speaker_label(
                        corrected,
                        clean_subtitle_text(translated_val),
                        glossary,
                    )
                    candidate_metadata.update(
                        {
                            "en": corrected,
                            "zh": translated,
                        }
                    )
                    guarded_candidate = guarded_display_candidate(
                        segments,
                        [*processed_segments, *batch_processed],
                        candidate_metadata,
                        item,
                        context_start,
                        context_end,
                    )
                    if forced_model_hide:
                        guarded_candidate.update(
                            {
                                "model_display_requested": False,
                                "display_guard": "forced_visible",
                                "display_guard_reason": "unique_source_not_covered",
                            }
                        )
                    display = guarded_candidate.get("display", True) is not False
                    proposed_terminology = applicable_terminology_entries(
                        extract_terminology_entries(item),
                        corrected,
                    )
                    conflicts = terminology_conflicts(
                        glossary,
                        corrected,
                        translated,
                        proposed_terminology,
                    )
                    if display and (
                        not source_language_preserved(
                            segment,
                            str(segment.get("text") or ""),
                            corrected,
                        )
                        or not is_language_valid(segment["text"], corrected, translated)
                        or conflicts
                    ):
                        raise RuntimeError(f"LLM returned an invalid translation for segment {i + j}")

                terminology = (
                    register_terminology(glossary, proposed_terminology)
                    if display
                    else []
                )
                guarded_candidate["terminology"] = terminology
                batch_processed.append(guarded_candidate)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process subtitle batch {i + 1}-{i + len(batch)}; "
                "the last valid checkpoint was preserved."
            ) from exc

        validate_timing_preserved(batch, batch_processed, i)
        processed_segments.extend(batch_processed)
        extend_display_over_hidden_segments(processed_segments)

        if checkpoint_path:
            write_json_atomic(checkpoint_path, processed_segments)
        write_terminology_artifact(terminology_path, glossary, llm_model)

    return processed_segments


def original_en_zh(segment: Segment) -> Tuple[str, str]:
    text = clean_subtitle_text(str(segment.get("text", "")))
    language = normalize_language_tag(str(segment.get("source_language") or ""))
    source_text = source_track_text(segment)
    if not source_text and not segment.get("en") and language != "zh":
        source_text = text
    chinese_text = clean_subtitle_text(str(segment.get("zh") or ""))
    if (
        not chinese_text
        and text
        and (
            language == "zh"
            or (
                language in UNKNOWN_SOURCE_LANGUAGE_TAGS
                and contains_cjk(text)
                and not source_text
            )
        )
    ):
        chinese_text = text
    return source_text, to_simplified_text(chinese_text)


def proofread_existing_chinese_segments(
    segments: List[Segment],
    llm_model: str,
    batch_size: int,
    context_lines: int,
    checkpoint_path: Optional[Path] = None,
    terminology_path: Optional[Path] = None,
    style_guide: Optional[Dict[str, Any]] = None,
) -> List[Segment]:
    print("Starting LLM proofreading for existing Chinese subtitles; translation is skipped.")
    style_text = format_style_guide(style_guide)
    editing_policy = source_editing_policy(segments)
    system_prompt = (
        "You are a professional bilingual subtitle proofreader. "
        f"{editing_policy} "
        "Fix Chinese OCR/subtitle recognition errors conservatively. "
        "Do not translate when Chinese subtitles are already provided. "
        f"Follow this movie-wide style guide exactly:\n{style_text}"
    )

    processed_segments: List[Segment] = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, list)
                and len(cached) <= len(segments)
                and checkpoint_matches_segments(
                    cached,
                    segments,
                    expected_mode="proofread_existing_chinese",
                    expected_model=llm_model,
                    expected_policy_version=PROCESSING_POLICY_VERSION,
                    expected_terminology_policy_version=TERMINOLOGY_POLICY_VERSION,
                    expected_plan_fingerprint=str(
                        segments[0].get("processing_plan_fingerprint") or ""
                    )
                    if segments
                    else "",
                )
            ):
                processed_segments = sanitize_checkpoint_display_timing(to_simplified_segments(cached))
                processed_segments = guard_model_hidden_segments(
                    segments[: len(processed_segments)],
                    processed_segments,
                )
                print(f"Resuming from checkpoint with {len(processed_segments)} completed segments.")
            else:
                print("Ignoring checkpoint because it does not match the current subtitle segmentation.")
        except Exception as exc:
            print(f"Ignoring unreadable checkpoint {checkpoint_path}: {exc}")

    start_index = len(processed_segments)
    repair_index = first_guarded_language_repair_index(
        segments,
        processed_segments,
        proofread=True,
    )
    if repair_index is not None:
        start_index = min(start_index, repair_index)
        print(
            "Checkpoint contains a restored unique subtitle without valid "
            f"bilingual text; reprocessing from item {start_index + 1}."
        )
    if start_index < len(segments):
        start_index -= start_index % batch_size
        processed_segments = processed_segments[:start_index]
    glossary = terminology_from_segments(
        processed_segments,
        (style_guide or {}).get("terminology"),
    )
    write_terminology_artifact(terminology_path, glossary, llm_model)

    for i in range(start_index, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        print(f"Proofreading segments {i + 1} to {i + len(batch)} of {len(segments)}...")

        before_text, after_text = build_bilingual_context_text(segments, i, i + len(batch), context_lines)
        approved_output_context = build_approved_output_context(processed_segments, context_lines)
        batch_source_text = "\n".join(
            " ".join(part for part in original_en_zh(segment) if part)
            for segment in batch
        )
        terminology_text = format_terminology_glossary(glossary, batch_source_text)
        batch_text = "\n".join(
            format_bilingual_prompt_segment(
                j,
                segment,
                next_start=(
                    float(segments[i + j + 1]["start"])
                    if i + j + 1 < len(segments)
                    else None
                ),
            )
            for j, segment in enumerate(batch)
        )
        context_start_index = max(0, i - context_lines)
        context_end_index = min(len(segments), i + len(batch) + context_lines)
        context_start = float(segments[context_start_index]["start"])
        context_end = float(segments[context_end_index - 1]["end"])

        prompt = f"""
You will proofread only the TARGET LINE(S).

These subtitles may already contain Chinese. Proofread the existing source-language and Chinese tracks.
The source track may also be OCR output and may contain recognition errors.
Do not retranslate a non-empty Chinese line. Translate source-track information only when the Chinese line is empty or incomplete.

=== PREVIOUS CONTEXT, REFERENCE ONLY ===
{before_text}

=== PREVIOUS APPROVED OUTPUT, MATCH NAMES AND TERMINOLOGY ===
{approved_output_context}

=== MOVIE-WIDE STYLE GUIDE, FOLLOW THROUGHOUT ===
{style_text}

=== SOURCE AUTHORITY POLICY ===
{editing_policy}

=== APPROVED MOVIE-WIDE TERMINOLOGY, USE EXACTLY ===
{terminology_text}

=== TARGET LINE(S) TO OUTPUT ===
{batch_text}

=== FOLLOWING CONTEXT, REFERENCE ONLY ===
{after_text}

Return a raw JSON list with exactly {len(batch)} objects.
Each object must correspond to one TARGET line and keep the same local index.
Schema:
[
  {{
    "index": 0,
    "corrected_english": "...",
    "corrected_chinese": "...",
    "display": true,
    "terminology": [{{"source": "John", "target": "约翰"}}]
  }}
]

Rules:
- One input line must still produce one output object for checkpointing.
- Correct obvious source-language and Chinese OCR/subtitle recognition errors.
- When SOURCE_AUTHORITY=ocr and the Chinese lane is authored, use the authored Chinese meaning to reconstruct fluent, grammatical source-language text instead of preserving nonsensical OCR wording.
- Correct only the content present inside each TARGET timestamp. Fragments are valid: never copy, complete, or move words or meaning from neighboring TARGET or context lines.
- When SOURCE_AUTHORITY=authored, copy the source-language text unchanged into corrected_english.
- Only ASR/OCR events may be hidden as proven duplicate loops; authored subtitle events must remain visible.
- Convert Traditional Chinese to natural Simplified Chinese.
- If corrected_chinese is already non-empty and complete, preserve its meaning and only proofread it. Do not replace it with a fresh translation.
- Translate source-track information into corrected_chinese only when the original Chinese is empty or clearly incomplete.
- Preserve every unique fact, clause, speaker label, number, and sound cue inside each TARGET timestamp. Never summarize or omit content to make an object shorter.
- A TARGET object may span several seconds and will be split into readable display cues after proofreading. Do not enforce final line-length limits in this editing response.
- Before returning, verify that object index N contains only the meaning of TARGET [N], never the preceding or following target.
- `corrected_english` is a legacy field name for the source track. Keep it in SOURCE_LANGUAGE and never translate it into English.
- If the source contains SDH/non-speech cues such as music, applause, laughter, or speaker labels and the Chinese line lacks that information, add only that missing cue in concise Simplified Chinese.
- You MUST translate uppercase descriptive text in parentheses or brackets (e.g. "(SIGHS)" or "[MUSIC]") into Simplified Chinese.
- Keep names and recurring terminology consistent across the context. Do not translate a name in one line and leave the same name untranslated in another unless the context clearly requires it.
- Reuse every applicable APPROVED MOVIE-WIDE TERMINOLOGY mapping exactly.
- In terminology, return only personal names or recurring terms that occur in that target line. Once a mapping is approved, never propose a different target. Use an empty list when there are none.
- Hide a later TARGET line only when it is a proven near-exact duplicate already fully represented by an overlapping visible cue. Keep genuine continuations separately timed.
- Never output or change timestamps. Timing is enforced by the subtitle pipeline.
- Keep corrected_english empty only when no source track exists.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
"""

        try:
            lookup: Optional[Dict[int, Dict[str, Any]]] = None
            response_schema = indexed_subtitle_schema(
                len(batch),
                "proofread",
            )
            for attempt in range(3):
                try:
                    data = parse_json_response(
                        call_llm(
                            prompt,
                            system_prompt,
                            llm_model,
                            role="proofread",
                            response_schema=response_schema,
                            seed_offset=attempt,
                        ),
                        response_schema=response_schema,
                    )
                    lookup = validate_indexed_batch_response(
                        data,
                        len(batch),
                        "proofread",
                    )
                    break
                except Exception as exc:
                    print(f"LLM request failed (attempt {attempt+1}/3): {exc}")
                    if attempt < 2:
                        import time
                        time.sleep(2)
                    else:
                        raise RuntimeError(
                            "网络、大模型服务或响应格式连续失败，已中断校对；"
                            "修复后再次运行可从最近 checkpoint 继续。"
                        ) from exc
            if lookup is None:
                raise RuntimeError("LLM response validation did not produce a complete batch")

            batch_processed: List[Segment] = []
            for j, segment in enumerate(batch):
                item = lookup[j]
                original_en, original_zh = original_en_zh(segment)
                corrected_en_val = str(item.get("corrected_english") or item.get("en") or original_en)
                corrected_en = (
                    preserve_authored_source(
                        segment,
                        original_en,
                        corrected_en_val,
                    )
                    if original_en
                    else ""
                )
                corrected_zh_val = str(
                    item.get("corrected_chinese")
                    or item.get("chinese")
                    or item.get("zh")
                    or original_zh
                )
                corrected_zh = restore_glossary_speaker_label(
                    corrected_en or original_en,
                    to_simplified_text(clean_subtitle_text(corrected_zh_val)),
                    glossary,
                )
                corrected_zh = preserve_authored_chinese_terminology(
                    segment,
                    original_zh,
                    corrected_zh,
                    corrected_en or original_en or original_zh,
                    glossary,
                )
                candidate_metadata = {
                    **segment,
                    "en": corrected_en,
                    "zh": corrected_zh,
                    "processing_mode": "proofread_existing_chinese",
                    "processing_policy_version": PROCESSING_POLICY_VERSION,
                    "terminology_policy_version": TERMINOLOGY_POLICY_VERSION,
                    "llm_model": llm_model,
                    "llm_role": "proofread",
                    "llm_generation_profile": asdict(
                        generation_profile("proofread")
                    ),
                }
                guarded_candidate = guarded_display_candidate(
                    segments,
                    [*processed_segments, *batch_processed],
                    candidate_metadata,
                    item,
                    context_start,
                    context_end,
                )
                display = guarded_candidate.get("display", True) is not False
                terminology_source = corrected_en or original_en or original_zh
                proposed_terminology = applicable_terminology_entries(
                    extract_terminology_entries(item),
                    terminology_source,
                )
                conflicts = terminology_conflicts(
                    glossary,
                    terminology_source,
                    corrected_zh,
                    proposed_terminology,
                )
                conflicts = filter_authored_chinese_terminology_conflicts(
                    segment,
                    original_zh,
                    conflicts,
                )
                borrowed_phrase = adjacent_context_borrowed_phrase(
                    segments,
                    i + j,
                    corrected_en,
                )
                if not borrowed_phrase:
                    borrowed_phrase = previous_output_borrowed_phrase(
                        segments,
                        i + j,
                        [*processed_segments, *batch_processed],
                        corrected_en,
                        corrected_zh,
                    )
                if authored_chinese_title_card_allows_ocr_reconstruction(
                    segment,
                    original_zh,
                    corrected_en,
                    borrowed_phrase,
                ):
                    borrowed_phrase = ""
                content_retained = authored_chinese_content_retained(
                    segment,
                    original_zh,
                    corrected_zh,
                )

                if display and (
                    not source_language_preserved(
                        segment,
                        original_en,
                        corrected_en,
                    )
                    or not is_language_valid(
                        original_en or original_zh,
                        corrected_en,
                        corrected_zh,
                        require_corrected=bool(original_en),
                        original_translation=original_zh,
                    )
                    or not content_retained
                    or conflicts
                    or borrowed_phrase
                ):
                    forced_model_hide = (
                        guarded_candidate.get("display_guard") == "forced_visible"
                    )
                    retry_item = retry_single_segment_llm(
                        segment,
                        f"{before_text}\n\nPREVIOUS APPROVED OUTPUT:\n{approved_output_context}",
                        after_text,
                        (
                            float(segments[i + j + 1]["start"])
                            if i + j + 1 < len(segments)
                            else None
                        ),
                        context_start,
                        context_end,
                        system_prompt,
                        llm_model,
                        "",
                        is_proofread=True,
                        terminology_text=format_terminology_glossary(
                            glossary,
                            terminology_source,
                        ),
                        terminology_glossary=glossary,
                        require_display=forced_model_hide,
                    )
                    if not retry_item:
                        raise RuntimeError(f"LLM could not produce a valid proofread result for segment {i + j}")
                    item.update(retry_item)
                    corrected_en_val = str(item.get("corrected_english") or item.get("en") or original_en)
                    corrected_en = (
                        preserve_authored_source(
                            segment,
                            original_en,
                            corrected_en_val,
                        )
                        if original_en
                        else ""
                    )
                    corrected_zh_val = str(
                        item.get("corrected_chinese")
                        or item.get("chinese")
                        or item.get("zh")
                        or original_zh
                    )
                    corrected_zh = restore_glossary_speaker_label(
                        corrected_en or original_en,
                        to_simplified_text(clean_subtitle_text(corrected_zh_val)),
                        glossary,
                    )
                    corrected_zh = preserve_authored_chinese_terminology(
                        segment,
                        original_zh,
                        corrected_zh,
                        corrected_en or original_en or original_zh,
                        glossary,
                    )
                    candidate_metadata.update(
                        {
                            "en": corrected_en,
                            "zh": corrected_zh,
                        }
                    )
                    guarded_candidate = guarded_display_candidate(
                        segments,
                        [*processed_segments, *batch_processed],
                        candidate_metadata,
                        item,
                        context_start,
                        context_end,
                    )
                    if forced_model_hide:
                        guarded_candidate.update(
                            {
                                "model_display_requested": False,
                                "display_guard": "forced_visible",
                                "display_guard_reason": "unique_source_not_covered",
                            }
                        )
                    display = guarded_candidate.get("display", True) is not False
                    terminology_source = corrected_en or original_en or original_zh
                    proposed_terminology = applicable_terminology_entries(
                        extract_terminology_entries(item),
                        terminology_source,
                    )
                    conflicts = terminology_conflicts(
                        glossary,
                        terminology_source,
                        corrected_zh,
                        proposed_terminology,
                    )
                    conflicts = filter_authored_chinese_terminology_conflicts(
                        segment,
                        original_zh,
                        conflicts,
                    )
                    borrowed_phrase = adjacent_context_borrowed_phrase(
                        segments,
                        i + j,
                        corrected_en,
                    )
                    if not borrowed_phrase:
                        borrowed_phrase = previous_output_borrowed_phrase(
                            segments,
                            i + j,
                            [*processed_segments, *batch_processed],
                            corrected_en,
                            corrected_zh,
                        )
                    if authored_chinese_title_card_allows_ocr_reconstruction(
                        segment,
                        original_zh,
                        corrected_en,
                        borrowed_phrase,
                    ):
                        borrowed_phrase = ""
                    content_retained = authored_chinese_content_retained(
                        segment,
                        original_zh,
                        corrected_zh,
                    )
                    if display and (
                        not source_language_preserved(
                            segment,
                            original_en,
                            corrected_en,
                        )
                        or not is_language_valid(
                            original_en or original_zh,
                            corrected_en,
                            corrected_zh,
                            require_corrected=bool(original_en),
                            original_translation=original_zh,
                        )
                        or not content_retained
                        or conflicts
                        or borrowed_phrase
                    ):
                        suppression_reason = None
                        if ocr_fragment_covered_by_adjacent_chinese(
                            segments,
                            i + j,
                        ):
                            suppression_reason = (
                                "ocr_fragment_covered_by_adjacent_chinese"
                            )
                        elif ocr_tiny_fragment_covered_by_following_chinese(
                            segments,
                            i + j,
                        ):
                            suppression_reason = (
                                "ocr_tiny_fragment_covered_by_following_chinese"
                            )
                        if suppression_reason:
                            guarded_candidate.update(
                                {
                                    "en": "",
                                    "zh": "",
                                    "display": False,
                                    "display_suppression": suppression_reason,
                                    "display_guard": "accepted_hidden",
                                    "display_guard_reason": suppression_reason,
                                }
                            )
                            proposed_terminology = []
                            display = False
                        else:
                            validation_failures = []
                            if not source_language_preserved(
                                segment,
                                original_en,
                                corrected_en,
                            ):
                                validation_failures.append(
                                    "source_language_changed"
                                )
                            if not is_language_valid(
                                original_en or original_zh,
                                corrected_en,
                                corrected_zh,
                                require_corrected=bool(original_en),
                                original_translation=original_zh,
                            ):
                                validation_failures.append(
                                    "invalid_language"
                                )
                            if not content_retained:
                                validation_failures.append(
                                    "authored_content_omitted"
                                )
                            if conflicts:
                                validation_failures.append(
                                    "terminology_conflict"
                                )
                            if borrowed_phrase:
                                validation_failures.append(
                                    "adjacent_context_borrowed"
                                )
                            raise RuntimeError(
                                "LLM returned an invalid proofread result "
                                f"for segment {i + j}: "
                                f"{','.join(validation_failures) or 'unknown'}"
                                + (
                                    f"; copied adjacent context: {borrowed_phrase!r}"
                                    if borrowed_phrase
                                    else ""
                                )
                            )

                terminology = (
                    register_terminology(glossary, proposed_terminology)
                    if display
                    else []
                )
                guarded_candidate["terminology"] = terminology
                batch_processed.append(guarded_candidate)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to proofread subtitle batch {i + 1}-{i + len(batch)}; "
                "the last valid checkpoint was preserved."
            ) from exc

        validate_timing_preserved(batch, batch_processed, i)
        processed_segments.extend(batch_processed)
        extend_display_over_hidden_segments(processed_segments)

        if checkpoint_path:
            write_json_atomic(checkpoint_path, processed_segments)
        write_terminology_artifact(terminology_path, glossary, llm_model)

    return processed_segments


def final_qa_segment_fingerprint_row(segment: Segment) -> Dict[str, Any]:
    return {
        "id": segment.get("id"),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "en": segment.get("en"),
        "zh": segment.get("zh"),
        "display": segment.get("display", True),
        "source_language": segment.get("source_language"),
        "source_text_authority": segment.get("source_text_authority"),
        "chinese_text_authority": segment.get("chinese_text_authority"),
        "processing_plan_fingerprint": segment.get(
            "processing_plan_fingerprint"
        ),
        "manual_reviewed_at": segment.get("manual_reviewed_at"),
    }


DEFAULT_FINAL_QA_REVIEW_PROFILE = "independent_final"
AUTONOMOUS_REVIEW_PASSES = (
    {
        "profile": "semantic_integrity",
        "directive": (
            "Audit semantic fidelity and source-boundary integrity. Prioritize residual "
            "ASR/OCR mistakes, missing or invented facts, numbers, names, forms of address, "
            "speaker labels, and source-only SDH/music/sound cues that Chinese still lacks. "
            "Change text only when there is a concrete defect; do not make preference-only "
            "rewrites."
        ),
    },
    {
        "profile": "professional_readability",
        "directive": (
            "Audit professional subtitle language and movie-wide consistency. Correct only "
            "clear unnatural Chinese, inconsistent names or register, ambiguous punctuation, "
            "and wording that is unnecessarily hard to read. Preserve every fact and cue, "
            "never shorten by omission, and never rewrite an authored source track."
        ),
    },
    {
        "profile": "adversarial_residual",
        "directive": (
            "Perform an adversarial residual-error audit. Look specifically for context "
            "leakage between adjacent timestamps, duplicated recognition loops, language or "
            "script changes, contradictory terminology, and content present in only one lane. "
            "Make no stylistic changes unless they fix a demonstrable delivery defect."
        ),
    },
)


def final_qa_fingerprint(
    segments: List[Segment],
    llm_model: str,
    style_guide: Optional[Dict[str, Any]],
    *,
    review_profile: str = DEFAULT_FINAL_QA_REVIEW_PROFILE,
    review_directive: str = "",
) -> str:
    rows = [final_qa_segment_fingerprint_row(segment) for segment in segments]
    descriptor: Dict[str, Any] = {
        "policy_version": FINAL_QA_POLICY_VERSION,
        "llm_model": llm_model,
        "style_fingerprint": (style_guide or {}).get("input_fingerprint"),
        "terminology_review_fingerprint": (style_guide or {}).get(
            "terminology_review_fingerprint"
        ),
        "style_terminology": (style_guide or {}).get("terminology") or [],
        "segments": rows,
    }
    if review_profile != DEFAULT_FINAL_QA_REVIEW_PROFILE:
        descriptor["review_profile"] = review_profile
    if review_directive:
        descriptor["review_directive"] = review_directive
    serialized = json.dumps(
        descriptor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def final_qa_cache_accepts_manual_overrides(
    cached_segments: Any,
    segments: List[Segment],
) -> bool:
    if not isinstance(cached_segments, list) or len(cached_segments) != len(segments):
        return False
    authority_fields = (
        "id",
        "source_language",
        "source_text_authority",
        "chinese_text_authority",
        "processing_plan_fingerprint",
    )
    manual_override_found = False
    for cached_segment, segment in zip(cached_segments, segments):
        if not isinstance(cached_segment, dict) or not isinstance(segment, dict):
            return False
        if segment.get("manual_reviewed_at"):
            manual_override_found = True
            if any(
                cached_segment.get(field) != segment.get(field)
                for field in authority_fields
            ):
                return False
            continue
        if (
            final_qa_segment_fingerprint_row(cached_segment)
            != final_qa_segment_fingerprint_row(segment)
        ):
            return False
    return manual_override_found


def run_final_subtitle_qa(
    segments: List[Segment],
    llm_model: str,
    artifact_path: Path,
    *,
    style_guide: Optional[Dict[str, Any]] = None,
    batch_size: int = 16,
    review_profile: str = DEFAULT_FINAL_QA_REVIEW_PROFILE,
    review_directive: str = "",
    seed_base: int = 0,
) -> Tuple[List[Segment], Dict[str, Any]]:
    input_fingerprint = final_qa_fingerprint(
        segments,
        llm_model,
        style_guide,
        review_profile=review_profile,
        review_directive=review_directive,
    )
    cached: Optional[Dict[str, Any]] = None
    if artifact_path.exists():
        try:
            raw = json.loads(artifact_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("version") == FINAL_QA_POLICY_VERSION:
                cached = raw
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    if (
        cached
        and cached.get("status") in {"pass", "review"}
        and cached.get("llm_model") == llm_model
        and cached.get("output_fingerprint") == input_fingerprint
    ):
        print(f"Final subtitle QA already matches the checkpoint: {artifact_path}")
        return [dict(segment) for segment in segments], cached
    if (
        cached
        and cached.get("status") in {"pass", "review"}
        and cached.get("llm_model") == llm_model
        and cached.get("input_fingerprint") == input_fingerprint
        and isinstance(cached.get("segments"), list)
        and len(cached["segments"]) == len(segments)
    ):
        print(f"Using cached independent final subtitle QA: {artifact_path}")
        return [dict(segment) for segment in cached["segments"]], cached
    cached_segments = cached.get("segments") if cached else None
    if (
        cached
        and cached.get("status") in {"pass", "review"}
        and cached.get("llm_model") == llm_model
        and cached.get("output_fingerprint")
        == final_qa_fingerprint(
            cached_segments or [],
            llm_model,
            style_guide,
            review_profile=review_profile,
            review_directive=review_directive,
        )
        and final_qa_cache_accepts_manual_overrides(cached_segments, segments)
    ):
        reviewed = [dict(segment) for segment in segments]
        manual_ids = {
            segment.get("id")
            for segment in reviewed
            if segment.get("manual_reviewed_at")
        }
        for segment in reviewed:
            if segment.get("id") in manual_ids:
                segment["final_qa_manual_preserved"] = True
        rejected_items = [
            dict(item)
            for item in cached.get("rejected_items") or []
            if isinstance(item, dict) and item.get("id") not in manual_ids
        ]
        report = dict(cached)
        report.update(
            {
                "status": "review" if rejected_items else "pass",
                "input_fingerprint": input_fingerprint,
                "output_fingerprint": final_qa_fingerprint(
                    reviewed,
                    llm_model,
                    style_guide,
                    review_profile=review_profile,
                    review_directive=review_directive,
                ),
                "reviewed_count": len(reviewed),
                "manual_preserved_count": len(manual_ids),
                "manual_override_count": len(manual_ids),
                "rejected_count": len(rejected_items),
                "rejected_items": rejected_items,
                "segments": reviewed,
                "review_profile": review_profile,
                "review_directive": review_directive,
            }
        )
        write_json_atomic(artifact_path, report)
        print(
            "Applied manually reviewed checkpoint overrides to cached "
            f"independent final subtitle QA: {artifact_path}"
        )
        return reviewed, report

    reviewed: List[Segment] = []
    changes: List[Dict[str, Any]] = []
    rejected_items: List[Dict[str, Any]] = []
    if (
        cached
        and cached.get("status") == "running"
        and cached.get("llm_model") == llm_model
        and cached.get("input_fingerprint") == input_fingerprint
        and isinstance(cached.get("segments"), list)
    ):
        reviewed = [dict(segment) for segment in cached["segments"]]
        changes = [
            dict(item)
            for item in cached.get("changes") or []
            if isinstance(item, dict)
        ]
        rejected_items = [
            dict(item)
            for item in cached.get("rejected_items") or []
            if isinstance(item, dict)
        ]
        if len(reviewed) > len(segments):
            reviewed = []
            changes = []
            rejected_items = []
        elif reviewed:
            print(f"Resuming independent final QA at item {len(reviewed) + 1}.")

    style_text = format_style_guide(style_guide)
    editing_policy = source_editing_policy(segments)
    review_focus = review_directive or (
        "Review all residual content, language, terminology, and subtitle-delivery defects."
    )
    glossary = terminology_from_segments(
        reviewed or segments,
        (style_guide or {}).get("terminology"),
    )
    start_index = len(reviewed)
    start_index -= start_index % max(1, batch_size)
    reviewed = reviewed[:start_index]

    for batch_start in range(start_index, len(segments), max(1, batch_size)):
        batch = segments[batch_start : batch_start + max(1, batch_size)]
        before = build_approved_output_context(
            segments[max(0, batch_start - 8) : batch_start],
            8,
        )
        after = "\n".join(
            format_bilingual_prompt_segment(index, segment)
            for index, segment in enumerate(
                segments[
                    batch_start + len(batch) : batch_start + len(batch) + 8
                ]
            )
        ) or "(none)"
        batch_text = "\n".join(
            format_bilingual_prompt_segment(
                index,
                segment,
                next_start=(
                    float(segments[batch_start + index + 1]["start"])
                    if batch_start + index + 1 < len(segments)
                    else None
                ),
            )
            for index, segment in enumerate(batch)
        )
        prompt = f"""
Perform an independent final quality review of only the TARGET LINE(S).
 Another editor already translated and proofread them. Do not trust earlier wording blindly.

=== REVIEW PASS ===
PROFILE: {review_profile}
FOCUS: {review_focus}

=== MOVIE-WIDE STYLE GUIDE ===
{style_text}

=== SOURCE AUTHORITY POLICY ===
{editing_policy}

=== APPROVED TERMINOLOGY ===
{format_terminology_glossary(glossary, batch_text)}

=== PREVIOUS CONTEXT, REFERENCE ONLY ===
{before}

=== TARGET LINE(S) ===
{batch_text}

=== FOLLOWING CONTEXT, REFERENCE ONLY ===
{after}

Return exactly {len(batch)} JSON objects using the provided schema.
Each object must use the key "index" with the zero-based local index. Do not use
"local_index", and do not return timestamps.
Schema:
[
  {{
    "index": 0,
    "corrected_english": "...",
    "corrected_chinese": "...",
    "display": true,
    "terminology": [{{"source": "John", "target": "约翰"}}]
  }}
]
Rules:
- Preserve every local index, timestamp, and display decision.
- Correct residual ASR/OCR errors, grammar, mistranslation, inconsistent names, forms of address, and terminology.
- When SOURCE_AUTHORITY=ocr and the Chinese lane is authored, use the authored Chinese meaning to reconstruct fluent, grammatical source-language text instead of preserving nonsensical OCR wording.
- Correct only content present inside each TARGET timestamp. Context is reference only; never copy, complete, or move words or meaning from neighboring cues.
- Subtitle fragments are valid. Keep genuine continuations separately timed and remove only proven near-exact duplicate loops.
- Never remove an existing speaker label or SDH/music/sound cue from either language track.
- Never wrap an entire subtitle in quotation marks unless the original line was already wrapped.
- When SOURCE_AUTHORITY=authored, preserve the source-language text exactly and review only the Chinese lane.
- Treat recognition confidence as review priority, not proof that text is correct or incorrect.
- `corrected_english` is a legacy schema field name for the SOURCE track. The source may be English, Japanese, Korean, or another language.
- Keep `corrected_english` in the same language and script as SOURCE_LANGUAGE. Never translate the source track into English.
- Keep a non-empty Chinese subtitle's meaning; translate only source-track information missing from Chinese.
- Translate source-only SDH/music/sound/speaker cues into concise Simplified Chinese when Chinese lacks them.
- Apply one rendering for every recurring personal name throughout the movie.
- Preserve every unique fact, clause, speaker label, number, and sound cue inside each TARGET timestamp. Never summarize or omit content to make an object shorter.
- A TARGET object may span several seconds and will be split into readable display cues after this review. Do not enforce final line-length limits in the editing response.
- Before returning, verify that object index N contains only the meaning of TARGET [N], never the preceding or following target.
- Never add commentary or keys outside the schema.
"""
        response_schema = indexed_subtitle_schema(len(batch), "proofread")
        lookup: Optional[Dict[int, Dict[str, Any]]] = None
        for attempt in range(3):
            try:
                data = parse_json_response(
                    call_llm(
                        prompt,
                         (
                            "You are the independent senior subtitle QC editor for the "
                            f"{review_profile} pass. Review the completed bilingual "
                            "deliverable against the movie-wide style guide."
                         ),
                        llm_model,
                        role="proofread",
                        response_schema=response_schema,
                        seed_offset=max(0, int(seed_base)) + attempt,
                    ),
                    response_schema=response_schema,
                )
                data = normalize_final_qa_batch_response(data, batch)
                lookup = validate_indexed_batch_response(
                    data,
                    len(batch),
                    "proofread",
                )
                break
            except Exception as exc:
                print(
                    "Independent final QA batch "
                    f"{batch_start + 1}-{batch_start + len(batch)} failed "
                    f"(attempt {attempt + 1}/3): {exc}"
                )
                if attempt < 2:
                    import time

                    time.sleep(2)
                else:
                    raise
        if lookup is None:
            raise RuntimeError(
                "Independent final QA response validation did not produce "
                "a complete batch"
            )
        batch_reviewed: List[Segment] = []
        for local_index, segment in enumerate(batch):
            item = lookup[local_index]
            if segment.get("manual_reviewed_at"):
                batch_reviewed.append(
                    {
                        **segment,
                        "final_qa_policy_version": FINAL_QA_POLICY_VERSION,
                        "final_qa_model": llm_model,
                        "final_qa_manual_preserved": True,
                    }
                )
                continue
            original_source, original_zh = original_en_zh(segment)
            corrected_source = (
                preserve_final_qa_track_structure(
                    original_source,
                    preserve_authored_source(
                        segment,
                        original_source,
                        str(item.get("corrected_english") or original_source),
                    ),
                )
                if original_source
                else ""
            )
            corrected_zh = to_simplified_text(
                clean_subtitle_text(
                    str(item.get("corrected_chinese") or original_zh)
                )
            )
            corrected_zh = restore_glossary_speaker_label(
                corrected_source or original_source,
                corrected_zh,
                glossary,
            )
            corrected_zh = preserve_final_qa_track_structure(
                original_zh,
                corrected_zh,
            )
            corrected_zh = preserve_authored_chinese_terminology(
                segment,
                original_zh,
                corrected_zh,
                corrected_source or original_source or original_zh,
                glossary,
            )
            proposed_terminology = applicable_terminology_entries(
                extract_terminology_entries(item),
                corrected_source or original_source,
            )
            proposed_terminology = approved_terminology_entries(
                glossary,
                proposed_terminology,
            )
            conflicts = terminology_conflicts(
                glossary,
                corrected_source or original_source,
                corrected_zh,
                proposed_terminology,
            )
            conflicts = filter_authored_chinese_terminology_conflicts(
                segment,
                original_zh,
                conflicts,
            )
            borrowed_phrase = adjacent_context_borrowed_phrase(
                segments,
                batch_start + local_index,
                corrected_source,
            )
            if authored_chinese_title_card_allows_ocr_reconstruction(
                segment,
                original_zh,
                corrected_source,
                borrowed_phrase,
            ):
                borrowed_phrase = ""
            language_preserved = source_language_preserved(
                segment,
                original_source,
                corrected_source,
            )
            content_retained = authored_chinese_content_retained(
                segment,
                original_zh,
                corrected_zh,
            )
            valid = language_preserved and is_language_valid(
                original_source or original_zh,
                corrected_source,
                corrected_zh,
                require_corrected=bool(original_source),
                original_translation=original_zh,
            ) and content_retained and not conflicts and not borrowed_phrase
            if not valid:
                batch_reviewed.append(dict(segment))
                if segment.get("display", True) is not False:
                    rejected_items.append(
                        {
                            "id": segment.get("id"),
                            "start": segment.get("start"),
                            "reason": (
                                "source_language_changed"
                                if not language_preserved
                                else (
                                    "adjacent_context_borrowed"
                                    if borrowed_phrase
                                    else (
                                        "authored_content_omitted"
                                        if not content_retained
                                        else "invalid_language_or_terminology"
                                    )
                                )
                            ),
                            **(
                                {"detail": borrowed_phrase}
                                if borrowed_phrase
                                else {}
                            ),
                        }
                    )
                continue

            output = {
                **segment,
                "en": corrected_source,
                "zh": corrected_zh,
                "final_qa_policy_version": FINAL_QA_POLICY_VERSION,
                "final_qa_model": llm_model,
            }
            changed_fields = [
                field
                for field, before_value, after_value in (
                    ("en", original_source, corrected_source),
                    ("zh", original_zh, corrected_zh),
                )
                if normalize_space(str(before_value or ""))
                != normalize_space(str(after_value or ""))
            ]
            if changed_fields:
                output["final_qa_changed"] = True
                changes.append(
                    {
                        "id": segment.get("id"),
                        "start": segment.get("start"),
                        "fields": changed_fields,
                    }
                )
            resolved_terminology: List[Dict[str, str]] = []
            resolved_keys: set[str] = set()
            for entry in [
                *(segment.get("terminology") or []),
                *proposed_terminology,
            ]:
                if not isinstance(entry, dict):
                    continue
                key = normalize_space(str(entry.get("source") or "")).casefold()
                resolved = glossary.get(key)
                if not key or resolved is None or key in resolved_keys:
                    continue
                resolved_terminology.append(dict(resolved))
                resolved_keys.add(key)
            output["terminology"] = resolved_terminology
            batch_reviewed.append(output)

        validate_timing_preserved(batch, batch_reviewed, batch_start)
        reviewed.extend(batch_reviewed)
        write_json_atomic(
            artifact_path,
            {
                "version": FINAL_QA_POLICY_VERSION,
                "status": "running",
                "llm_model": llm_model,
                "input_fingerprint": input_fingerprint,
                "reviewed_count": len(reviewed),
                "manual_preserved_count": sum(
                    bool(item.get("final_qa_manual_preserved"))
                    for item in reviewed
                ),
                "changed_count": len(changes),
                "rejected_count": len(rejected_items),
                "changes": changes,
                "rejected_items": rejected_items,
                "segments": reviewed,
                "review_profile": review_profile,
                "review_directive": review_directive,
                "seed_base": max(0, int(seed_base)),
            },
        )

    report = {
        "version": FINAL_QA_POLICY_VERSION,
        "status": "review" if rejected_items else "pass",
        "llm_model": llm_model,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": final_qa_fingerprint(
            reviewed,
            llm_model,
            style_guide,
            review_profile=review_profile,
            review_directive=review_directive,
        ),
        "reviewed_count": len(reviewed),
        "manual_preserved_count": sum(
            bool(item.get("final_qa_manual_preserved"))
            for item in reviewed
        ),
        "changed_count": len(changes),
        "rejected_count": len(rejected_items),
        "changes": changes,
        "rejected_items": rejected_items,
        "segments": reviewed,
        "review_profile": review_profile,
        "review_directive": review_directive,
        "seed_base": max(0, int(seed_base)),
    }
    write_json_atomic(artifact_path, report)
    return reviewed, report


def autonomous_review_fingerprint(
    segments: List[Segment],
    llm_model: str,
    style_guide: Optional[Dict[str, Any]],
    rounds: int,
) -> str:
    selected_passes = AUTONOMOUS_REVIEW_PASSES[:rounds]
    serialized = json.dumps(
        {
            "policy_version": AUTONOMOUS_REVIEW_POLICY_VERSION,
            "llm_model": llm_model,
            "rounds": rounds,
            "passes": list(selected_passes),
            "input_fingerprint": final_qa_fingerprint(
                segments,
                llm_model,
                style_guide,
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def autonomous_review_round_artifact_path(
    artifact_path: Path,
    round_number: int,
) -> Path:
    return artifact_path.with_name(
        f"{artifact_path.stem}.round-{round_number}{artifact_path.suffix}"
    )


def run_autonomous_subtitle_review(
    segments: List[Segment],
    llm_model: str,
    artifact_path: Path,
    *,
    style_guide: Optional[Dict[str, Any]] = None,
    batch_size: int = 16,
    rounds: int = 2,
) -> Tuple[List[Segment], Dict[str, Any]]:
    if rounds < 1 or rounds > len(AUTONOMOUS_REVIEW_PASSES):
        raise ValueError(
            "Autonomous review rounds must be between 1 and "
            f"{len(AUTONOMOUS_REVIEW_PASSES)}."
        )

    input_fingerprint = autonomous_review_fingerprint(
        segments,
        llm_model,
        style_guide,
        rounds,
    )
    current_output_fingerprint = final_qa_fingerprint(
        segments,
        llm_model,
        style_guide,
    )
    cached: Optional[Dict[str, Any]] = None
    if artifact_path.exists():
        try:
            raw = json.loads(artifact_path.read_text(encoding="utf-8"))
            if (
                isinstance(raw, dict)
                and raw.get("version") == AUTONOMOUS_REVIEW_POLICY_VERSION
            ):
                cached = raw
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    if (
        cached
        and cached.get("status") in {"pass", "review"}
        and cached.get("llm_model") == llm_model
        and cached.get("rounds_requested") == rounds
        and cached.get("output_fingerprint") == current_output_fingerprint
    ):
        print(f"Autonomous review already matches the checkpoint: {artifact_path}")
        return [dict(segment) for segment in segments], cached

    cached_segments = cached.get("segments") if cached else None
    if (
        cached
        and cached.get("status") in {"pass", "review"}
        and cached.get("llm_model") == llm_model
        and cached.get("rounds_requested") == rounds
        and cached.get("input_fingerprint") == input_fingerprint
        and isinstance(cached_segments, list)
        and len(cached_segments) == len(segments)
        and cached.get("output_fingerprint")
        == final_qa_fingerprint(cached_segments, llm_model, style_guide)
    ):
        print(f"Using cached autonomous subtitle review: {artifact_path}")
        return [dict(segment) for segment in cached_segments], cached

    reviewed = [dict(segment) for segment in segments]
    round_summaries: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []
    rejected_items: List[Dict[str, Any]] = []
    manual_preserved_ids: set[Any] = set()

    for round_index, review_pass in enumerate(
        AUTONOMOUS_REVIEW_PASSES[:rounds],
        start=1,
    ):
        profile = str(review_pass["profile"])
        directive = str(review_pass["directive"])
        round_artifact = autonomous_review_round_artifact_path(
            artifact_path,
            round_index,
        )
        print(
            "Autonomous overnight review: "
            f"round={round_index}/{rounds}, profile={profile}",
            flush=True,
        )
        try:
            reviewed, round_report = run_final_subtitle_qa(
                reviewed,
                llm_model,
                round_artifact,
                style_guide=style_guide,
                batch_size=batch_size,
                review_profile=profile,
                review_directive=directive,
                seed_base=round_index * 1000,
            )
        except Exception as exc:
            failure_report = {
                "version": AUTONOMOUS_REVIEW_POLICY_VERSION,
                "status": "failed",
                "llm_model": llm_model,
                "rounds_requested": rounds,
                "rounds_completed": len(round_summaries),
                "failed_round": round_index,
                "failed_profile": profile,
                "input_fingerprint": input_fingerprint,
                "output_fingerprint": final_qa_fingerprint(
                    reviewed,
                    llm_model,
                    style_guide,
                ),
                "reviewed_count": len(reviewed),
                "manual_preserved_count": len(manual_preserved_ids),
                "changed_count": len(changes),
                "rejected_count": len(rejected_items),
                "changes": changes,
                "rejected_items": rejected_items,
                "round_reports": round_summaries,
                "segments": reviewed,
                "error": str(exc),
            }
            write_json_atomic(artifact_path, failure_report)
            print(
                "Autonomous overnight review stopped after the last valid round: "
                f"{exc}"
            )
            return reviewed, failure_report

        tagged_changes = [
            {
                **dict(item),
                "round": round_index,
                "review_profile": profile,
            }
            for item in round_report.get("changes") or []
            if isinstance(item, dict)
        ]
        tagged_rejected = [
            {
                **dict(item),
                "round": round_index,
                "review_profile": profile,
            }
            for item in round_report.get("rejected_items") or []
            if isinstance(item, dict)
        ]
        changes.extend(tagged_changes)
        rejected_items.extend(tagged_rejected)
        manual_preserved_ids.update(
            segment.get("id")
            for segment in reviewed
            if segment.get("manual_reviewed_at")
        )
        round_summaries.append(
            {
                "round": round_index,
                "profile": profile,
                "status": round_report.get("status"),
                "artifact_path": str(round_artifact),
                "reviewed_count": round_report.get("reviewed_count", len(reviewed)),
                "changed_count": round_report.get("changed_count", 0),
                "rejected_count": round_report.get("rejected_count", 0),
                "manual_preserved_count": round_report.get(
                    "manual_preserved_count",
                    0,
                ),
            }
        )
        write_json_atomic(
            artifact_path,
            {
                "version": AUTONOMOUS_REVIEW_POLICY_VERSION,
                "status": "running",
                "llm_model": llm_model,
                "rounds_requested": rounds,
                "rounds_completed": round_index,
                "input_fingerprint": input_fingerprint,
                "output_fingerprint": final_qa_fingerprint(
                    reviewed,
                    llm_model,
                    style_guide,
                ),
                "reviewed_count": len(reviewed),
                "manual_preserved_count": len(manual_preserved_ids),
                "changed_count": len(changes),
                "rejected_count": len(rejected_items),
                "changes": changes,
                "rejected_items": rejected_items,
                "round_reports": round_summaries,
                "segments": reviewed,
            },
        )

    report = {
        "version": AUTONOMOUS_REVIEW_POLICY_VERSION,
        "status": "review" if rejected_items else "pass",
        "llm_model": llm_model,
        "rounds_requested": rounds,
        "rounds_completed": rounds,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": final_qa_fingerprint(
            reviewed,
            llm_model,
            style_guide,
        ),
        "reviewed_count": len(reviewed),
        "manual_preserved_count": len(manual_preserved_ids),
        "changed_count": len(changes),
        "changed_segment_count": len(
            {item.get("id") for item in changes if item.get("id") is not None}
        ),
        "rejected_count": len(rejected_items),
        "changes": changes,
        "rejected_items": rejected_items,
        "round_reports": round_summaries,
        "segments": reviewed,
    }
    write_json_atomic(artifact_path, report)
    return reviewed, report


def checkpoint_matches_segments(
    cached: List[Segment],
    segments: List[Segment],
    expected_mode: Optional[str] = None,
    expected_model: Optional[str] = None,
    expected_policy_version: Optional[int] = None,
    expected_terminology_policy_version: Optional[int] = None,
    expected_plan_fingerprint: Optional[str] = None,
) -> bool:
    for index, cached_segment in enumerate(cached):
        if index >= len(segments):
            print("Ignoring checkpoint because it contains more items than the source.")
            return False
        current = segments[index]
        if "display" not in cached_segment:
            print("Ignoring checkpoint because it lacks duplicate-cleanup display metadata.")
            return False
        if expected_mode and cached_segment.get("processing_mode") != expected_mode:
            print(
                "Ignoring checkpoint because it was generated by a different processing mode "
                f"at item {index}: expected {expected_mode}, got {cached_segment.get('processing_mode')!r}."
            )
            return False
        if (
            expected_terminology_policy_version is not None
            and int(cached_segment.get("terminology_policy_version") or 0)
            != expected_terminology_policy_version
        ):
            print(
                "Ignoring checkpoint because its terminology policy is outdated "
                f"at item {index}: expected {expected_terminology_policy_version}, "
                f"got {cached_segment.get('terminology_policy_version')!r}."
            )
            return False
        if expected_model and cached_segment.get("llm_model") != expected_model:
            print(
                "Ignoring checkpoint because it was generated by a different model "
                f"at item {index}: expected {expected_model}, got {cached_segment.get('llm_model')!r}."
            )
            return False
        if (
            expected_policy_version is not None
            and int(cached_segment.get("processing_policy_version") or 0)
            != expected_policy_version
        ):
            print(
                "Ignoring checkpoint because its processing policy is outdated "
                f"at item {index}: expected {expected_policy_version}, "
                f"got {cached_segment.get('processing_policy_version')!r}."
            )
            return False
        if (
            expected_plan_fingerprint
            and str(cached_segment.get("processing_plan_fingerprint") or "")
            != expected_plan_fingerprint
        ):
            print(
                "Ignoring checkpoint because its processing plan does not match "
                f"the current source plan at item {index}."
            )
            return False
        current_fingerprint = str(current.get("source_request_fingerprint") or "")
        cached_fingerprint = str(cached_segment.get("source_request_fingerprint") or "")
        if current_fingerprint and cached_fingerprint != current_fingerprint:
            print(
                "Ignoring checkpoint because its source request fingerprint does not match "
                f"the current source at item {index}."
            )
            return False
        manually_reviewed = bool(cached_segment.get("manual_reviewed_at"))
        has_manual_anchors = (
            cached_segment.get("manual_source_start") is not None
            and cached_segment.get("manual_source_end") is not None
        )
        cached_start = cached_segment.get(
            "manual_source_start",
            cached_segment["start"],
        )
        cached_end = cached_segment.get(
            "manual_source_end",
            cached_segment["end"],
        )
        same_start = (
            manually_reviewed and not has_manual_anchors
        ) or abs(float(cached_start) - float(current["start"])) <= 0.02
        same_end = (
            manually_reviewed and not has_manual_anchors
        ) or abs(float(cached_end) - float(current["end"])) <= 0.02
        cached_text = cached_segment.get(
            "manual_source_text",
            cached_segment.get("text", ""),
        )
        same_text = clean_subtitle_text(str(cached_text)) == clean_subtitle_text(
            str(current.get("text", ""))
        )
        if not (same_start and same_end and same_text):
            print(
                "Checkpoint mismatch at item "
                f"{index}: cached {cached_segment.get('start')}->{cached_segment.get('end')} "
                f"{cached_segment.get('text')!r}, current {current.get('start')}->{current.get('end')} "
                f"{current.get('text')!r}"
            )
            return False
    return True


def validate_timing_preserved(source_batch: List[Segment], processed_batch: List[Segment], global_start: int) -> None:
    if len(source_batch) != len(processed_batch):
        raise ValueError(
            f"Timing validation failed at batch {global_start}: "
            f"expected {len(source_batch)} items, got {len(processed_batch)}"
        )

    for offset, (source, processed) in enumerate(zip(source_batch, processed_batch)):
        if float(source["start"]) != float(processed["start"]) or float(source["end"]) != float(processed["end"]):
            raise ValueError(
                f"Timing validation failed at item {global_start + offset}: "
                f"{source['start']}->{source['end']} became {processed['start']}->{processed['end']}"
            )


def seconds_to_ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    hours, rem = divmod(cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    seconds, cs = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def escape_ass_text(text: str) -> str:
    text = normalize_space(text).replace("\n", " ")
    text = text.replace("{", "(").replace("}", ")")
    return text


def generate_ass(
    segments: List[Segment],
    out_path: Path,
    mode: str,
    *,
    play_resolution: tuple[int, int] = (1920, 1080),
    style_profile: str = "adaptive",
    font_name: str = "",
    font_scale: int | float = 100,
) -> None:
    header = ass_style_header(
        *play_resolution,
        profile_name=style_profile,
        font_name=font_name,
        font_scale=font_scale,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for segment in segments:
            if segment.get("display", True) is False:
                continue
            start = seconds_to_ass_time(float(segment.get("display_start", segment["start"])))
            end = seconds_to_ass_time(float(segment.get("display_end", segment["end"])))
            raw_source_text = str(segment.get("en", segment["text"]))
            if should_clean_english_track(segment):
                raw_source_text = clean_english_track_text(raw_source_text)
            en_text = escape_ass_text(raw_source_text)
            zh_text = escape_ass_text(segment.get("zh", ""))
            if mode == "en" and not en_text:
                continue
            if mode == "zh" and not zh_text:
                continue
            if mode == "bilingual" and not (en_text or zh_text):
                continue

            if mode == "en":
                handle.write(f"Dialogue: 0,{start},{end},SourceOnly,,0,0,0,,{en_text}\n")
            elif mode == "zh":
                handle.write(f"Dialogue: 0,{start},{end},Chinese,,0,0,0,,{zh_text}\n")
            elif mode == "bilingual":
                if zh_text and en_text:
                    text = f"{{\\rChinese}}{zh_text}\\N{{\\rSource}}{en_text}"
                    style = "Chinese"
                elif zh_text:
                    text = zh_text
                    style = "Chinese"
                else:
                    text = en_text
                    style = "SourceOnly"
                handle.write(f"Dialogue: 0,{start},{end},{style},,0,0,0,,{text}\n")


def print_timing_report(segments: List[Segment]) -> None:
    if not segments:
        print("No subtitle events generated.")
        return

    durations = [float(segment["end"]) - float(segment["start"]) for segment in segments]
    gaps = [
        float(segments[index]["start"]) - float(segments[index - 1]["end"])
        for index in range(1, len(segments))
    ]
    intentional_overlaps = sum(
        gap < 0
        and bool(segments[index - 1].get("preserve_distinct_overlap"))
        and bool(segments[index].get("preserve_distinct_overlap"))
        and overlap_content_key(segments[index - 1]) != overlap_content_key(segments[index])
        for index, gap in enumerate(gaps, start=1)
    )
    total_overlaps = sum(gap < 0 for gap in gaps)
    out_of_order = sum(
        float(segments[index]["start"]) < float(segments[index - 1]["start"])
        for index in range(1, len(segments))
    )
    print(
        "Timing report: "
        f"events={len(segments)}, "
        f"max_duration={max(durations):.2f}s, "
        f">6s={sum(duration > 6 for duration in durations)}, "
        f">8s={sum(duration > 8 for duration in durations)}, "
        f"unexpected_overlaps={total_overlaps - intentional_overlaps}, "
        f"intentional_overlaps={intentional_overlaps}, "
        f"out_of_order={out_of_order}, "
        f"gaps>5s={sum(gap > 5 for gap in gaps)}, "
        f"max_gap={max(gaps) if gaps else 0:.2f}s"
    )


def default_output_root(video_path: Optional[Path] = None) -> Path:
    configured = os.environ.get("SUBTITLE_OUTPUT_ROOT")
    if configured:
        return Path(configured).expanduser()
    if video_path is not None:
        return suggested_output_root(video_path)
    return Path.cwd() / OUTPUT_DIRECTORY_NAME


def segments_have_chinese(segments: List[Segment]) -> bool:
    zh_text = " ".join(str(segment.get("zh") or "") for segment in segments)
    if len(CJK_RE.findall(zh_text)) >= 3:
        return True
    dominant_language = dominant_source_language(segments)
    if dominant_language and dominant_language != "zh":
        return False
    text_without_english = []
    for segment in segments:
        if segment.get("en"):
            continue
        language = normalize_language_tag(str(segment.get("source_language") or ""))
        if language not in UNKNOWN_SOURCE_LANGUAGE_TAGS and language != "zh":
            continue
        text_without_english.append(str(segment.get("text") or ""))
    return len(CJK_RE.findall(" ".join(text_without_english))) >= 3


def require_safe_automatic_audio_selection(
    audio_assets: List[SubtitleAsset],
    *,
    explicit_stream: Optional[int],
    selected_asset: Optional[SubtitleAsset],
) -> None:
    if explicit_stream is not None or selected_asset is not None or not audio_assets:
        return
    rejected = ", ".join(
        asset.label
        for asset in audio_assets
        if asset.role == "commentary"
    )
    detail = f" Detected tracks: {rejected}." if rejected else ""
    raise RuntimeError(
        "Only commentary or audio-description tracks are available for automatic "
        "selection. Choose the intended dialogue audio track explicitly before "
        f"running speech recognition.{detail}"
    )


def main() -> None:
    configure_output_encoding()
    parser = argparse.ArgumentParser(description="Transcribe, import, OCR, correct, and translate subtitles to ASS.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file or Blu-ray folder")
    parser.add_argument("--source", choices=["auto", "sidecar", "srt", "embedded", "audio"], default="auto", help="Subtitle source")
    parser.add_argument("--srt", type=str, help="Explicit sidecar SRT path. Kept for compatibility.")
    parser.add_argument("--subtitle-file", type=str, help="Explicit sidecar subtitle path: srt, ass, ssa, or vtt.")
    parser.add_argument("--chinese-subtitle-file", type=str, help="Explicit Chinese sidecar subtitle path for bilingual merge.")
    parser.add_argument("--english-subtitle-file", type=str, help="Explicit English sidecar subtitle path for bilingual merge.")
    parser.add_argument(
        "--chinese-subtitle-text-authority",
        choices=["authored", "ocr"],
        default="authored",
        help="Whether the selected Chinese sidecar is authored text or OCR text.",
    )
    parser.add_argument(
        "--english-subtitle-text-authority",
        choices=["authored", "ocr"],
        default="authored",
        help="Whether the selected English sidecar is authored text or OCR text.",
    )
    parser.add_argument("--subtitle-stream", type=int, help="Embedded subtitle stream index, e.g. 2 for 0:2.")
    parser.add_argument("--chinese-subtitle-stream", type=int, help="Embedded Chinese subtitle stream index for bilingual merge.")
    parser.add_argument("--english-subtitle-stream", type=int, help="Embedded English subtitle stream index for bilingual merge.")
    parser.add_argument("--merge-existing-subtitles", choices=["yes", "no"], default="yes", help="Merge existing Chinese and English subtitles before proofreading.")
    parser.add_argument("--audio-stream", type=int, help="Audio stream index, e.g. 1 for 0:1.")
    parser.add_argument(
        "--subtitle-sync",
        choices=SUBTITLE_SYNC_MODES,
        default="auto",
        help="Align existing subtitle timing to the selected audio: auto, detect, or off.",
    )
    parser.add_argument("--source-language", default="auto", help="Source subtitle language for correction/translation context.")
    parser.add_argument("--asr-language", default="source", help="Whisper language code, source to follow --source-language, or auto for detection.")
    parser.add_argument("--subtitle-ocr-lang", default="auto", help="PaddleOCR language for image subtitles, or auto.")
    parser.add_argument("--device", default="gpu:0", help="OCR device for embedded image subtitles.")
    parser.add_argument("--ocr-scale", type=float, default=2.0)
    parser.add_argument("--crop-pad", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for PGS images per stream.")
    parser.add_argument("--fast-ocr", action="store_true", help="Allow faster OCR settings instead of best-quality defaults.")
    parser.add_argument("--output-root", type=str, help="Output root. Defaults to sibling '1 字幕' folder.")
    parser.add_argument("--series-name", type=str, help="Override output series folder name")
    parser.add_argument("--movie-name", type=str, help="Override output movie/episode file name")
    parser.add_argument("--llm-model", type=str, default="qwen3:14b", help="Ollama model name")
    parser.add_argument("--batch-size", type=int, default=5, help="LLM batch size in subtitle sentence units.")
    parser.add_argument("--context-lines", type=int, default=30, help="Reference this many subtitle lines before and after each target batch.")
    parser.add_argument(
        "--autonomous-review-rounds",
        type=int,
        choices=range(0, len(AUTONOMOUS_REVIEW_PASSES) + 1),
        default=0,
        help=(
            "Run zero to three checkpointed independent overnight review passes "
            "before final layout and delivery validation."
        ),
    )
    parser.add_argument("--max-words", type=int, default=12, help="Maximum English words per subtitle event")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=ENGLISH_CHARS_PER_LINE,
        help="Requested English characters per event, capped at the bilingual one-line limit.",
    )
    parser.add_argument("--max-duration", type=float, default=5.5, help="Maximum seconds per subtitle event before splitting")
    parser.add_argument(
        "--subtitle-style-profile",
        choices=STYLE_PROFILE_NAMES,
        default="adaptive",
        help="ASS display profile: adaptive, mobile, or compact.",
    )
    parser.add_argument("--subtitle-font-name", default="", help="ASS font family name. Defaults to Arial.")
    parser.add_argument(
        "--subtitle-font-file",
        default="",
        help="Optional TTF/OTF/TTC file to attach to the bilingual Matroska subtitle bundle.",
    )
    parser.add_argument(
        "--subtitle-font-scale",
        type=int,
        default=100,
        help="ASS font scale percentage from 70 to 160.",
    )
    parser.add_argument("--run-state-file", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    input_path = Path(args.video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    video_path = resolve_video_path(input_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if args.series_name and args.movie_name:
        series_name, movie_name = args.series_name, args.movie_name
    else:
        series_name, movie_name = parse_movie_name(video_path.name, args.llm_model)
    print(f"Identified Series: {series_name}, Movie: {movie_name}")

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root, _ = resolve_output_root(video_path, series_name, movie_name)
    out_dir = output_root / series_name / movie_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{movie_name}.segments.checkpoint.json"
    source_segments_path = out_dir / f"{movie_name}.segments.source.json"
    source_manifest_path = out_dir / f"{movie_name}.source-manifest.json"
    processing_plan_path = out_dir / f"{movie_name}.processing-plan.json"
    sync_report_path = out_dir / f"{movie_name}.subtitle-sync.report.json"
    terminology_path = out_dir / f"{movie_name}.terminology.json"
    terminology_review_path = out_dir / f"{movie_name}.terminology-review.json"
    terminology_overrides_path = (
        out_dir / f"{movie_name}.terminology-overrides.json"
    )
    style_guide_path = out_dir / f"{movie_name}.style-guide.json"
    final_qa_path = out_dir / f"{movie_name}.final-qa.json"
    autonomous_review_path = out_dir / f"{movie_name}.autonomous-review.json"
    run_state_file = Path(args.run_state_file) if args.run_state_file else None

    sidecar_path = Path(args.subtitle_file or args.srt) if (args.subtitle_file or args.srt) else None
    chinese_sidecar_path = Path(args.chinese_subtitle_file) if args.chinese_subtitle_file else None
    english_sidecar_path = Path(args.english_subtitle_file) if args.english_subtitle_file else None
    if sidecar_path is None:
        sidecar_path = english_sidecar_path or chinese_sidecar_path
    merge_existing_subtitles = args.merge_existing_subtitles == "yes"
    automatic_inventory_plan: Optional[Dict[str, Any]] = None
    automatic_assets: List[SubtitleAsset] = []
    preferred_auto_existing_origin = ""
    preferred_auto_primary_asset: Optional[SubtitleAsset] = None
    if args.source == "auto":
        automatic_assets = discover_automatic_source_assets(
            video_path,
            input_path,
        )
        automatic_inventory_plan = build_processing_plan(
            automatic_assets,
            preferred_source_language=args.source_language,
            merge_existing=merge_existing_subtitles,
            synchronize_existing=args.subtitle_sync != "off",
        )
        automatic_lanes = automatic_inventory_plan.get("lanes") or {}
        chinese_asset_id = automatic_lanes.get("chinese")
        chinese_asset = next(
            (
                asset
                for asset in automatic_assets
                if asset.asset_id == chinese_asset_id
            ),
            None,
        )
        if chinese_asset is not None:
            preferred_auto_existing_origin = chinese_asset.origin
        preferred_auto_primary_asset_id = (
            chinese_asset_id or automatic_lanes.get("source")
        )
        preferred_auto_primary_asset = next(
            (
                asset
                for asset in automatic_assets
                if asset.asset_id == preferred_auto_primary_asset_id
            ),
            None,
        )
        explicit_sidecar_primary = (
            Path(args.subtitle_file or args.srt)
            if (args.subtitle_file or args.srt)
            else (
                chinese_sidecar_path
                or (
                    english_sidecar_path
                    if not merge_existing_subtitles
                    else None
                )
            )
        )
        explicit_stream_index = (
            args.subtitle_stream
            if args.subtitle_stream is not None
            else (
                args.chinese_subtitle_stream
                if args.chinese_subtitle_stream is not None
                else (
                    args.english_subtitle_stream
                    if not merge_existing_subtitles
                    else None
                )
            )
        )
        if (
            explicit_sidecar_primary is not None
            and explicit_sidecar_primary.exists()
        ):
            preferred_auto_primary_asset = build_sidecar_asset(
                video_path,
                explicit_sidecar_primary,
                likely_match=True,
                score=max(
                    100,
                    sidecar_match_score(
                        video_path,
                        explicit_sidecar_primary,
                    ),
                ),
            )
            preferred_auto_existing_origin = "sidecar"
        elif explicit_stream_index is not None:
            explicit_embedded_asset = next(
                (
                    asset
                    for asset in automatic_assets
                    if asset.origin == "embedded"
                    and asset.stream_index == explicit_stream_index
                ),
                None,
            )
            if explicit_embedded_asset is not None:
                preferred_auto_primary_asset = explicit_embedded_asset
                preferred_auto_existing_origin = "embedded"
        print(
            "Automatic source inventory recommendation: "
            f"{automatic_inventory_plan.get('route')}, "
            f"existing_origin={preferred_auto_existing_origin or 'none'}"
        )
    audio_inventory_assets = automatic_assets
    if args.source != "auto" and not any(
        asset.origin == "audio"
        for asset in audio_inventory_assets
    ):
        audio_inventory_assets = discover_automatic_source_assets(
            video_path,
            input_path,
        )
    audio_assets = [
        asset
        for asset in audio_inventory_assets
        if asset.origin == "audio"
    ]
    preferred_audio_asset: Optional[SubtitleAsset] = None
    if args.audio_stream is None:
        if audio_assets:
            audio_plan = build_processing_plan(
                audio_assets,
                preferred_source_language=args.source_language,
                merge_existing=False,
            )
            audio_asset_id = (audio_plan.get("lanes") or {}).get("source")
            preferred_audio_asset = next(
                (
                    asset
                    for asset in audio_assets
                    if asset.asset_id == audio_asset_id
                ),
                None,
            )
    effective_audio_stream = (
        args.audio_stream
        if args.audio_stream is not None
        else (
            preferred_audio_asset.stream_index
            if preferred_audio_asset is not None
            else None
        )
    )
    effective_audio_asset = next(
        (
            asset
            for asset in audio_assets
            if asset.stream_index == effective_audio_stream
        ),
        None,
    )
    if effective_audio_asset is None and effective_audio_stream is not None:
        effective_audio_asset = build_audio_asset(
            video_path,
            stream_index=effective_audio_stream,
            language=(
                args.source_language
                if args.source_language not in {"", "auto"}
                else ""
            ),
        )
    automatic_audio_selection_blocked = (
        args.audio_stream is None
        and bool(audio_assets)
        and preferred_audio_asset is None
    )
    effective_subtitle_sync_mode = args.subtitle_sync
    subtitle_sync_skip_reason = ""
    if automatic_audio_selection_blocked and args.subtitle_sync != "off":
        effective_subtitle_sync_mode = "off"
        subtitle_sync_skip_reason = (
            "Only commentary or audio-description tracks were available for "
            "automatic audio selection."
        )
        print(
            "Subtitle sync disabled: "
            f"{subtitle_sync_skip_reason} Select an audio stream explicitly to sync."
        )
    automatic_plan_cache_fingerprint = (
        str(automatic_inventory_plan.get("fingerprint"))
        if automatic_inventory_plan
        and any(
            str(asset.get("origin") or "") in {"sidecar", "embedded"}
            for asset in automatic_inventory_plan.get("selected_assets") or []
            if isinstance(asset, dict)
        )
        else None
    )
    if args.source in ("auto", "sidecar", "srt") and not sidecar_path:
        if (
            args.source == "auto"
            and preferred_auto_primary_asset is not None
            and preferred_auto_primary_asset.origin == "sidecar"
            and preferred_auto_primary_asset.path
        ):
            sidecar_path = Path(preferred_auto_primary_asset.path)
        else:
            sidecar_path = find_sidecar_subtitle(video_path, input_path)
    if args.source == "auto" and sidecar_path and sidecar_path.exists():
        auto_sidecar_asset = build_sidecar_asset(
            video_path,
            sidecar_path,
            likely_match=sidecar_match_score(video_path, sidecar_path) > 0,
            score=sidecar_match_score(video_path, sidecar_path),
        )
        if auto_sidecar_asset.role in {"forced", "commentary"}:
            print(
                "Skipping automatic sidecar selection because it is a "
                f"{auto_sidecar_asset.role} track: {sidecar_path}"
            )
            sidecar_path = None
    source_request_fingerprint = build_source_request_fingerprint(
        video_path,
        input_path,
        args,
        sidecar_path=sidecar_path,
        chinese_sidecar_path=chinese_sidecar_path,
        english_sidecar_path=english_sidecar_path,
        processing_plan_fingerprint=automatic_plan_cache_fingerprint,
    )
    compatible_source_request_fingerprints: set[str] = set()
    for audio_stream_variant in source_request_audio_stream_variants(
        args.audio_stream,
        effective_audio_stream,
        effective_subtitle_sync_mode,
        sync_report_path,
    ):
        fingerprint_args = argparse.Namespace(**vars(args))
        fingerprint_args.audio_stream = audio_stream_variant
        compatible_source_request_fingerprints.update(
            build_source_request_fingerprint(
                video_path,
                input_path,
                fingerprint_args,
                sidecar_path=sidecar_path,
                chinese_sidecar_path=chinese_sidecar_path,
                english_sidecar_path=english_sidecar_path,
                sync_policy_version=version,
                processing_plan_fingerprint=automatic_plan_cache_fingerprint,
            )
            for version in {
                SYNC_POLICY_VERSION,
                *COMPATIBLE_SYNC_POLICY_VERSIONS,
            }
        )
    has_explicit_source_selection = any(
        value is not None and value != ""
        for value in (
            args.subtitle_file,
            args.srt,
            args.chinese_subtitle_file,
            args.english_subtitle_file,
            args.subtitle_stream,
            args.chinese_subtitle_stream,
            args.english_subtitle_stream,
            args.audio_stream,
        )
    )
    allow_legacy_source_cache = (
        args.source in ("auto", "audio")
        and not has_explicit_source_selection
        and not (args.source == "auto" and sidecar_path is not None)
    )

    temp_audio = out_dir / f".{movie_name}.subtitle-audio.{os.getpid()}.wav"
    try:
        actual_source = args.source
        source_language = args.source_language
        asr_language = resolve_asr_language(args.asr_language, args.source_language)
        subtitle_segments: Optional[List[Segment]] = None
        existing_tracks: Optional[ExistingSubtitleTracks] = None
        sync_report: Optional[Dict[str, Any]] = None
        loaded_legacy_source_cache = False
        reused_source_cache = False
        if source_segments_path.exists():
            try:
                cached_segments = load_cached_source_segments(source_segments_path)
                current_request_matches = source_cache_matches_request(
                    cached_segments,
                    source_request_fingerprint,
                    allow_legacy=allow_legacy_source_cache,
                )
                compatible_request_matches = any(
                    source_cache_matches_request(
                        cached_segments,
                        fingerprint,
                        allow_legacy=False,
                    )
                    for fingerprint in compatible_source_request_fingerprints
                )
                if not current_request_matches and not compatible_request_matches:
                    print(
                        "Ignoring cached source segments because the selected source, file, stream, "
                        f"or input video changed: {source_segments_path}"
                    )
                elif not source_cache_uses_current_timing_policy(cached_segments):
                    print(
                        "Ignoring legacy merged-subtitle source cache because it predates the current "
                        f"timing-anchor policy: {source_segments_path}"
                    )
                elif source_cache_matches_requested_language(cached_segments, args.source_language, args.asr_language):
                    subtitle_segments = cached_segments
                    loaded_legacy_source_cache = (
                        compatible_request_matches
                        or not any(
                            segment.get("source_request_fingerprint")
                            for segment in cached_segments
                        )
                    )
                    actual_source = "cached source segments"
                    reused_source_cache = True
                    source_language = infer_source_language_from_segments(subtitle_segments, source_language)
                    if sync_report_path.exists():
                        try:
                            cached_sync_report = json.loads(
                                sync_report_path.read_text(encoding="utf-8")
                            )
                            if isinstance(cached_sync_report, dict):
                                sync_report = cached_sync_report
                        except (OSError, ValueError, json.JSONDecodeError) as exc:
                            print(f"Ignoring unreadable subtitle sync report {sync_report_path}: {exc}")
                    print(f"Reusing cached source segments from {source_segments_path} ({len(subtitle_segments)} events).")
                else:
                    print(
                        "Ignoring cached source segments because their language does not match "
                        f"the requested source/asr language: {source_segments_path}"
                    )
            except Exception as exc:
                print(f"Ignoring unreadable source cache {source_segments_path}: {exc}")

        if subtitle_segments is None:
            if (
                merge_existing_subtitles
                and args.source in ("auto", "sidecar", "srt")
                and not (
                    args.source == "auto"
                    and preferred_auto_existing_origin == "embedded"
                )
            ):
                existing_tracks = find_existing_sidecar_subtitle_tracks(
                    video_path,
                    input_path,
                    explicit_path=sidecar_path if (args.subtitle_file or args.srt) else None,
                    explicit_zh_path=chinese_sidecar_path,
                    explicit_en_path=english_sidecar_path,
                    chinese_text_authority=args.chinese_subtitle_text_authority,
                    english_text_authority=args.english_subtitle_text_authority,
                )
                if existing_tracks:
                    actual_source = "existing Chinese sidecar subtitles"
                    source_language = (
                        existing_tracks.english_asset.language
                        if existing_tracks.english_asset
                        else "zh"
                    )

            if existing_tracks is None and merge_existing_subtitles and args.source in ("auto", "embedded"):
                try:
                    existing_tracks = find_existing_embedded_subtitle_tracks(
                        video_path,
                        out_dir,
                        args,
                        chinese_stream_index=args.chinese_subtitle_stream,
                        english_stream_index=args.english_subtitle_stream,
                    )
                    if existing_tracks:
                        actual_source = "existing Chinese embedded subtitles"
                        source_language = (
                            existing_tracks.english_asset.language
                            if existing_tracks.english_asset
                            else "zh"
                        )
                except Exception as exc:
                    if args.source == "embedded":
                        print(f"Failed to load embedded Chinese subtitles: {exc}", file=sys.stderr)
                        raise
                    print(f"No existing Chinese embedded subtitles available ({exc}).")

            if (
                existing_tracks is None
                and merge_existing_subtitles
                and args.source == "auto"
                and preferred_auto_existing_origin == "embedded"
            ):
                existing_tracks = find_existing_sidecar_subtitle_tracks(
                    video_path,
                    input_path,
                    explicit_path=None,
                    explicit_zh_path=chinese_sidecar_path,
                    explicit_en_path=english_sidecar_path,
                    chinese_text_authority=args.chinese_subtitle_text_authority,
                    english_text_authority=args.english_subtitle_text_authority,
                )
                if existing_tracks:
                    actual_source = "existing Chinese sidecar subtitles"
                    source_language = (
                        existing_tracks.english_asset.language
                        if existing_tracks.english_asset
                        else "zh"
                    )

            if existing_tracks and args.source == "auto":
                existing_tracks = add_cross_origin_source_track(
                    existing_tracks,
                    video_path,
                    input_path,
                    out_dir,
                    args,
                    preferred_source_language=args.source_language,
                )
                if existing_tracks.english_asset:
                    source_language = (
                        existing_tracks.english_asset.language
                        or source_language
                    )
                existing_tracks = add_supplementary_source_tracks(
                    existing_tracks,
                    automatic_assets,
                    automatic_inventory_plan,
                    video_path,
                    out_dir,
                    args,
                )

            if existing_tracks:
                synchronized_tracks, sync_report = synchronize_existing_subtitle_tracks(
                    video_path,
                    existing_tracks,
                    mode=effective_subtitle_sync_mode,
                    report_path=sync_report_path,
                    audio_stream=effective_audio_stream,
                )
                if subtitle_sync_skip_reason:
                    sync_report.update(
                        {
                            "requested_mode": args.subtitle_sync,
                            "status": "skipped_unsafe_audio_selection",
                            "skip_reason": subtitle_sync_skip_reason,
                        }
                    )
                    write_json_atomic(sync_report_path, sync_report)
                subtitle_segments = merge_existing_subtitle_segments(
                    synchronized_tracks.english,
                    synchronized_tracks.chinese,
                    synchronized_tracks.source_label,
                    english_asset=synchronized_tracks.english_asset,
                    chinese_asset=synchronized_tracks.chinese_asset,
                    supplementary_assets=synchronized_tracks.supplementary_assets,
                )
                track_reports = sync_report.get("tracks") or {}
                track_statuses = ", ".join(
                    f"{name}={report.get('status')}"
                    for name, report in track_reports.items()
                    if isinstance(report, dict)
                )
                print(
                    "Subtitle sync: "
                    f"scope=per_track, status={sync_report.get('status')}, "
                    f"applied={sync_report.get('applied')}"
                    + (f", {track_statuses}" if track_statuses else "")
                )
                print("Existing Chinese subtitles detected; translation will be skipped after bilingual proofreading.")
            else:
                auto_primary_origin = (
                    preferred_auto_primary_asset.origin
                    if preferred_auto_primary_asset is not None
                    else ""
                )
                auto_may_use_sidecar = auto_primary_origin in {"", "sidecar"}
                auto_may_use_embedded = auto_primary_origin in {"", "embedded"}
                if (
                    args.source in ("sidecar", "srt")
                    or (
                        args.source == "auto"
                        and auto_may_use_sidecar
                    )
                ) and sidecar_path and sidecar_path.exists():
                    actual_source = "sidecar"
                    source_segments = parse_subtitle_file(sidecar_path)
                    sidecar_asset = build_sidecar_asset(
                        video_path,
                        sidecar_path,
                        language="" if source_language == "auto" else source_language,
                        likely_match=sidecar_match_score(video_path, sidecar_path) > 0,
                        score=sidecar_match_score(video_path, sidecar_path),
                    )
                    attach_asset_to_segments(source_segments, sidecar_asset, lane="source")
                    source_language = sidecar_asset.language or source_language
                elif args.source in ("sidecar", "srt"):
                    raise FileNotFoundError("Sidecar subtitle source requested, but no subtitle file was found.")
                elif args.source == "embedded" or (
                    args.source == "auto"
                    and auto_may_use_embedded
                ):
                    try:
                        actual_source = "embedded"
                        fallback_stream_index = args.subtitle_stream
                        if (
                            fallback_stream_index is None
                            and args.source == "auto"
                            and preferred_auto_primary_asset is not None
                            and preferred_auto_primary_asset.origin == "embedded"
                        ):
                            fallback_stream_index = (
                                preferred_auto_primary_asset.stream_index
                            )
                        if fallback_stream_index is None:
                            fallback_stream_index = args.english_subtitle_stream
                        if fallback_stream_index is None:
                            fallback_stream_index = args.chinese_subtitle_stream
                        source_segments = load_embedded_subtitle_events(
                            video_path,
                            stream_index=fallback_stream_index,
                            source_language=args.source_language,
                            out_dir=out_dir,
                            args=args,
                        )
                        if source_language == "auto":
                            source_language = source_segments[0].get("source_language") or "embedded subtitle"
                    except Exception as exc:
                        if args.source == "embedded":
                            print(f"Failed to load embedded subtitles: {exc}", file=sys.stderr)
                            raise
                        print(f"No embedded subtitles available ({exc}). Falling back to audio extraction...")
                        actual_source = "audio"
                        require_safe_automatic_audio_selection(
                            audio_assets,
                            explicit_stream=args.audio_stream,
                            selected_asset=preferred_audio_asset,
                        )
                        extract_audio(
                            video_path,
                            temp_audio,
                            audio_stream=effective_audio_stream,
                        )
                        source_segments = transcribe_audio(temp_audio, language=asr_language)
                        if source_language == "auto":
                            source_language = infer_source_language_from_segments(source_segments, asr_language)
                        audio_asset = effective_audio_asset or build_audio_asset(
                            video_path,
                            stream_index=effective_audio_stream,
                            language=source_language,
                        )
                        attach_asset_to_segments(source_segments, audio_asset, lane="source")
                else:
                    actual_source = "audio"
                    require_safe_automatic_audio_selection(
                        audio_assets,
                        explicit_stream=args.audio_stream,
                        selected_asset=preferred_audio_asset,
                    )
                    extract_audio(
                        video_path,
                        temp_audio,
                        audio_stream=effective_audio_stream,
                    )
                    source_segments = transcribe_audio(temp_audio, language=asr_language)
                    if source_language == "auto":
                        source_language = infer_source_language_from_segments(source_segments, asr_language)
                    audio_asset = effective_audio_asset or build_audio_asset(
                        video_path,
                        stream_index=effective_audio_stream,
                        language=source_language,
                    )
                    attach_asset_to_segments(source_segments, audio_asset, lane="source")

                if (
                    args.source == "auto"
                    and merge_existing_subtitles
                    and automatic_inventory_plan is not None
                ):
                    source_segments, _ = merge_planned_supplementary_segments(
                        source_segments,
                        automatic_assets,
                        automatic_inventory_plan,
                        video_path,
                        out_dir,
                        args,
                    )
                subtitle_segments = source_segments

        selected_assets = assets_from_segments(subtitle_segments)
        existing_timing_source = any(
            asset.origin in {"sidecar", "embedded"}
            for asset in selected_assets
        ) or actual_source in {
            "sidecar",
            "embedded",
            "existing Chinese sidecar subtitles",
            "existing Chinese embedded subtitles",
        }
        if not selected_assets and not existing_timing_source:
            cached_audio_asset = effective_audio_asset or build_audio_asset(
                video_path,
                stream_index=effective_audio_stream,
                language=infer_source_language_from_segments(
                    subtitle_segments,
                    source_language,
                ),
            )
            attach_asset_to_segments(
                subtitle_segments,
                cached_audio_asset,
                lane="source",
            )
            selected_assets = [cached_audio_asset]
        if existing_timing_source:
            for segment in subtitle_segments:
                segment["preserve_distinct_overlap"] = True
                segment["timing_origin"] = "existing_subtitle"
        else:
            for segment in subtitle_segments:
                segment.setdefault("timing_origin", "audio_asr")

        if existing_timing_source and not reused_source_cache and existing_tracks is None:
            subtitle_segments, sync_report = synchronize_subtitle_segments(
                video_path,
                subtitle_segments,
                mode=effective_subtitle_sync_mode,
                report_path=sync_report_path,
                audio_stream=effective_audio_stream,
            )
            if subtitle_sync_skip_reason:
                sync_report.update(
                    {
                        "requested_mode": args.subtitle_sync,
                        "status": "skipped_unsafe_audio_selection",
                        "skip_reason": subtitle_sync_skip_reason,
                    }
                )
                write_json_atomic(sync_report_path, sync_report)
            print(
                "Subtitle sync: "
                f"status={sync_report.get('status')}, "
                f"applied={sync_report.get('applied')}, "
                f"median_shift={sync_report.get('median_start_shift_seconds', 0)}s"
            )

        subtitle_segments = split_segments_for_subtitles(
            subtitle_segments,
            max_words=args.max_words,
            max_chars=args.max_chars,
            max_duration=args.max_duration,
        )
        subtitle_segments = apply_timing_sanity_rules(subtitle_segments, max_duration=args.max_duration)
        selected_assets = assets_from_segments(subtitle_segments)
        processing_assets = list(selected_assets)
        should_record_audio_reference = (
            effective_audio_asset is not None
            and (
                any(asset.origin == "audio" for asset in selected_assets)
                or (
                    existing_timing_source
                    and effective_subtitle_sync_mode != "off"
                )
            )
        )
        if (
            should_record_audio_reference
            and effective_audio_asset is not None
            and all(
                asset.asset_id != effective_audio_asset.asset_id
                for asset in processing_assets
            )
        ):
            processing_assets.append(effective_audio_asset)
        processing_plan = build_processing_plan(
            processing_assets,
            preferred_source_language=source_language,
            merge_existing=merge_existing_subtitles,
            synchronize_existing=effective_subtitle_sync_mode != "off",
            allow_audio_source=args.source in {"auto", "audio"},
        )
        processing_plan["execution"] = {
            "source_cache_reused": reused_source_cache,
            "source_generation": "cache" if reused_source_cache else "generated",
            "local_gpu": (
                []
                if reused_source_cache
                else list(
                    processing_plan.get("compute", {}).get("local_gpu") or []
                )
            ),
            "remote_llm": list(
                processing_plan.get("compute", {}).get("remote_llm") or []
            ),
        }
        for segment in subtitle_segments:
            segment["source_cache_version"] = SOURCE_CACHE_VERSION
            segment["processing_plan_fingerprint"] = processing_plan["fingerprint"]
            if not loaded_legacy_source_cache:
                segment["source_request_fingerprint"] = source_request_fingerprint
        write_json_atomic(source_segments_path, subtitle_segments)
        write_json_atomic(
            source_manifest_path,
            source_manifest(
                processing_assets,
                processing_plan,
                video_path=video_path,
            ),
        )
        write_json_atomic(processing_plan_path, processing_plan)

        print(f"Actual subtitle source: {actual_source}")
        print(
            "Processing route: "
            f"{processing_plan.get('route')}; "
            f"source_generation={processing_plan['execution']['source_generation']}; "
            f"local_gpu={','.join(processing_plan['execution']['local_gpu']) or 'none'}; "
            f"llm={','.join(processing_plan['execution']['remote_llm']) or 'none'}"
        )
        print_timing_report(subtitle_segments)

        if len(subtitle_segments) >= 12:
            try:
                style_guide = analyze_movie_style(
                    subtitle_segments,
                    args.llm_model,
                    style_guide_path,
                )
                print(
                    "Movie-wide style guide: "
                    f"status={style_guide.get('status')}, path={style_guide_path}"
                )
            except Exception as exc:
                style_guide = {
                    "version": STYLE_GUIDE_POLICY_VERSION,
                    "status": "unavailable",
                    "llm_model": args.llm_model,
                    "input_fingerprint": style_guide_fingerprint(
                        subtitle_segments,
                        args.llm_model,
                    ),
                    "error": str(exc),
                    "terminology": [],
                }
                write_json_atomic(style_guide_path, style_guide)
                print(f"Movie-wide style analysis unavailable; using defaults: {exc}")
        else:
            style_guide = {
                "version": STYLE_GUIDE_POLICY_VERSION,
                "status": "skipped_short",
                "llm_model": args.llm_model,
                "input_fingerprint": style_guide_fingerprint(
                    subtitle_segments,
                    args.llm_model,
                ),
                "reason": "Fewer than 12 subtitle events.",
                "terminology": [],
            }
            write_json_atomic(style_guide_path, style_guide)

        if (
            processing_plan.get("route") == "proofread_existing_chinese"
            or segments_have_chinese(subtitle_segments)
        ):
            processed_segments = proofread_existing_chinese_segments(
                subtitle_segments,
                llm_model=args.llm_model,
                batch_size=args.batch_size,
                context_lines=args.context_lines,
                checkpoint_path=checkpoint_path,
                terminology_path=terminology_path,
                style_guide=style_guide,
            )
        else:
            processed_segments = translate_and_correct_segments(
                subtitle_segments,
                llm_model=args.llm_model,
                batch_size=args.batch_size,
                context_lines=args.context_lines,
                source_language=source_language,
                checkpoint_path=checkpoint_path,
                terminology_path=terminology_path,
                style_guide=style_guide,
            )

        try:
            reviewed_terminology, terminology_review_report = (
                review_movie_terminology(
                    processed_segments,
                    args.llm_model,
                    terminology_review_path,
                    title=movie_name,
                    seeded_entries=style_guide.get("terminology"),
                )
            )
            manual_terminology = load_terminology_overrides(
                terminology_overrides_path
            )
            reviewed_terminology = apply_terminology_overrides(
                reviewed_terminology,
                manual_terminology,
            )
            terminology_review_report = {
                **terminology_review_report,
                "manual_override_count": len(manual_terminology),
                "resolved_entries": reviewed_terminology,
            }
            write_json_atomic(
                terminology_review_path,
                terminology_review_report,
            )
            style_guide = {
                **style_guide,
                "terminology": reviewed_terminology,
                "terminology_review_fingerprint": (
                    terminology_review_report.get("input_fingerprint")
                ),
                "terminology_review_status": terminology_review_report.get(
                    "status"
                ),
            }
            write_json_atomic(style_guide_path, style_guide)
            authoritative_glossary: Dict[str, Dict[str, str]] = {}
            register_terminology(
                authoritative_glossary,
                reviewed_terminology,
            )
            write_terminology_artifact(
                terminology_path,
                authoritative_glossary,
                args.llm_model,
            )
            print(
                "Movie-wide terminology review: "
                f"status={terminology_review_report.get('status')}, "
                f"reviewed={terminology_review_report.get('reviewed_count', 0)}, "
                f"corrected={terminology_review_report.get('correction_count', 0)}, "
                f"unapplied={terminology_review_report.get('unapplied_count', 0)}, "
                f"manual={terminology_review_report.get('manual_override_count', 0)}"
            )
        except Exception as exc:
            terminology_review_report = {
                "version": TERMINOLOGY_REVIEW_POLICY_VERSION,
                "status": "failed",
                "llm_model": args.llm_model,
                "error": str(exc),
            }
            write_json_atomic(
                terminology_review_path,
                terminology_review_report,
            )
            print(
                "Movie-wide terminology review failed; "
                f"using first-pass terminology: {exc}"
            )

        if len(processed_segments) >= 12:
            try:
                processed_segments, final_qa_report = run_final_subtitle_qa(
                    processed_segments,
                    args.llm_model,
                    final_qa_path,
                    style_guide=style_guide,
                    batch_size=max(8, args.batch_size * 2),
                )
                (
                    processed_segments,
                    suppressed_track_labels,
                ) = suppress_subtitle_track_label_artifacts(processed_segments)
                if suppressed_track_labels:
                    final_qa_report = {
                        **final_qa_report,
                        "segments": processed_segments,
                        "output_fingerprint": final_qa_fingerprint(
                            processed_segments,
                            args.llm_model,
                            style_guide,
                        ),
                        "deterministic_hidden_count": len(
                            suppressed_track_labels
                        ),
                        "deterministic_hidden_items": suppressed_track_labels,
                    }
                    write_json_atomic(final_qa_path, final_qa_report)
                    print(
                        "Suppressed subtitle track label artifacts: "
                        f"{len(suppressed_track_labels)}"
                    )
                write_json_atomic(checkpoint_path, processed_segments)
                print(
                    "Independent final QA: "
                    f"status={final_qa_report.get('status')}, "
                    f"changed={final_qa_report.get('changed_count', 0)}, "
                    f"rejected={final_qa_report.get('rejected_count', 0)}"
                )
            except Exception as exc:
                final_qa_report = {
                    "version": FINAL_QA_POLICY_VERSION,
                    "status": "failed",
                    "llm_model": args.llm_model,
                    "error": str(exc),
                }
                write_json_atomic(final_qa_path, final_qa_report)
                print(f"Independent final QA failed; preserving first-pass subtitles: {exc}")
        else:
            final_qa_report = {
                "version": FINAL_QA_POLICY_VERSION,
                "status": "skipped_short",
                "llm_model": args.llm_model,
                "reason": "Fewer than 12 processed subtitle events.",
            }
            write_json_atomic(final_qa_path, final_qa_report)

        delivery_qa_report = final_qa_report
        if args.autonomous_review_rounds and len(processed_segments) >= 12:
            processed_segments, autonomous_review_report = (
                run_autonomous_subtitle_review(
                    processed_segments,
                    args.llm_model,
                    autonomous_review_path,
                    style_guide=style_guide,
                    batch_size=max(8, args.batch_size * 2),
                    rounds=args.autonomous_review_rounds,
                )
            )
            delivery_qa_report = autonomous_review_report
            autonomous_status = str(autonomous_review_report.get("status") or "")
            combined_status = (
                autonomous_status
                if autonomous_status in {"pass", "review"}
                else final_qa_report.get("status", "failed")
            )
            final_qa_report = {
                **final_qa_report,
                "status": combined_status,
                "segments": processed_segments,
                "output_fingerprint": final_qa_fingerprint(
                    processed_segments,
                    args.llm_model,
                    style_guide,
                ),
                "autonomous_review": {
                    "artifact_path": str(autonomous_review_path),
                    "status": autonomous_status,
                    "rounds_requested": autonomous_review_report.get(
                        "rounds_requested",
                        args.autonomous_review_rounds,
                    ),
                    "rounds_completed": autonomous_review_report.get(
                        "rounds_completed",
                        0,
                    ),
                    "changed_count": autonomous_review_report.get(
                        "changed_count",
                        0,
                    ),
                    "rejected_count": autonomous_review_report.get(
                        "rejected_count",
                        0,
                    ),
                    **(
                        {"error": autonomous_review_report.get("error")}
                        if autonomous_review_report.get("error")
                        else {}
                    ),
                },
            }
            write_json_atomic(checkpoint_path, processed_segments)
            write_json_atomic(final_qa_path, final_qa_report)
            print(
                "Autonomous overnight review complete: "
                f"status={autonomous_status}, "
                f"rounds={autonomous_review_report.get('rounds_completed', 0)}/"
                f"{args.autonomous_review_rounds}, "
                f"changed={autonomous_review_report.get('changed_count', 0)}, "
                f"rejected={autonomous_review_report.get('rejected_count', 0)}"
            )

        subtitle_font_file = (
            validate_font_path(Path(args.subtitle_font_file))
            if args.subtitle_font_file
            else None
        )
        resolved_font_name = args.subtitle_font_name
        if subtitle_font_file is not None:
            resolved_font_name = font_family_name(subtitle_font_file)
            if args.subtitle_font_name and args.subtitle_font_name != resolved_font_name:
                print(
                    "Using the selected font file's internal family name: "
                    f"{resolved_font_name} (requested {args.subtitle_font_name})"
                )
        play_resolution = probe_video_play_resolution(video_path)
        print(
            "ASS display: "
            f"profile={args.subtitle_style_profile}, "
            f"font={resolved_font_name or 'Arial'}, "
            f"scale={args.subtitle_font_scale}%, "
            f"PlayRes={play_resolution[0]}x{play_resolution[1]}"
        )
        layout_policy = build_layout_policy(
            play_resolution,
            profile_name=args.subtitle_style_profile,
            font_name=resolved_font_name,
            font_scale=args.subtitle_font_scale,
            font_path=subtitle_font_file,
        )
        scene_cut_cache_path = out_dir / f"{movie_name}.scene-cuts.json"
        scene_cuts = detect_scene_cuts(
            video_path,
            scene_cut_cache_path,
        )
        if scene_cuts:
            print(f"Scene-aware timing: loaded {len(scene_cuts)} cached or detected cuts.")
        output_segments = prepare_segments_for_output(
            processed_segments,
            max_words=args.max_words,
            max_chars=args.max_chars,
            max_duration=args.max_duration,
            layout_policy=layout_policy,
            scene_cuts=scene_cuts,
        )
        en_ass = out_dir / f"{movie_name}.en.ass"
        zh_ass = out_dir / f"{movie_name}.zh.ass"
        bi_ass = out_dir / f"{movie_name}.bilingual.ass"
        quality_report_path = out_dir / f"{movie_name}.quality-report.json"
        render_validation_path = out_dir / f"{movie_name}.render-validation.json"
        render_preview_path = out_dir / f"{movie_name}.render-preview.png"

        print("Generating bilingual subtitles...", flush=True)
        ass_options = {
            "play_resolution": play_resolution,
            "style_profile": args.subtitle_style_profile,
            "font_name": resolved_font_name,
            "font_scale": args.subtitle_font_scale,
        }
        generate_ass(output_segments, en_ass, "en", **ass_options)
        generate_ass(output_segments, zh_ass, "zh", **ass_options)
        generate_ass(output_segments, bi_ass, "bilingual", **ass_options)
        srt_artifacts = write_delivery_srts(
            output_segments,
            out_dir,
            movie_name,
        )
        render_validation = validate_ass_rendering(
            bi_ass,
            output_segments,
            render_preview_path,
            play_resolution=play_resolution,
            font_directory=(
                subtitle_font_file.parent
                if subtitle_font_file is not None
                else None
            ),
        )
        write_json_atomic(render_validation_path, render_validation)
        print(
            "ASS render validation: "
            f"status={render_validation.get('status')}, "
            f"preview={render_validation.get('preview_path', '-')}"
        )
        artifact_paths = {
            "english_ass": str(en_ass),
            "chinese_ass": str(zh_ass),
            "bilingual_ass": str(bi_ass),
            "quality_report": str(quality_report_path),
            "source_manifest": str(source_manifest_path),
            "processing_plan": str(processing_plan_path),
            "style_guide": str(style_guide_path),
            "terminology_review": str(terminology_review_path),
            "final_qa": str(final_qa_path),
            "render_validation": str(render_validation_path),
            **srt_artifacts,
        }
        if args.autonomous_review_rounds and autonomous_review_path.exists():
            artifact_paths["autonomous_review"] = str(autonomous_review_path)
        if render_preview_path.exists():
            artifact_paths["render_preview"] = str(render_preview_path)
        if sync_report_path.exists():
            artifact_paths["subtitle_sync_report"] = str(sync_report_path)
        if terminology_overrides_path.exists():
            artifact_paths["terminology_overrides"] = str(
                terminology_overrides_path
            )
        if scene_cut_cache_path.exists():
            artifact_paths["scene_cut_cache"] = str(scene_cut_cache_path)
        if subtitle_font_file is not None:
            font_manifest = package_ass_with_font(
                bi_ass,
                subtitle_font_file,
                font_family=resolved_font_name,
            )
            print(
                "Font delivery bundle: "
                f"{font_manifest['matroska_subtitle_bundle']}"
            )
            artifact_paths["bilingual_font_bundle"] = str(
                font_manifest["matroska_subtitle_bundle"]
            )

        if loaded_legacy_source_cache:
            for segment in subtitle_segments:
                segment["source_request_fingerprint"] = source_request_fingerprint
            for segment in processed_segments:
                segment["source_request_fingerprint"] = source_request_fingerprint
            write_json_atomic(source_segments_path, subtitle_segments)
            write_json_atomic(checkpoint_path, processed_segments)

        quality_report = build_quality_report(
            subtitle_segments,
            processed_segments,
            output_segments,
            layout_policy,
            max_duration=args.max_duration,
            target_chinese_cps=TARGET_CHINESE_CPS,
            target_english_cps=TARGET_ENGLISH_CPS,
            artifacts=artifact_paths,
            sync_report=sync_report,
            source_kind=actual_source,
            video_duration_seconds=probe_video_duration_seconds(video_path),
            min_duration_seconds=MIN_SUBTITLE_DURATION,
            style_guide=style_guide,
            final_qa_report=delivery_qa_report,
            render_validation=render_validation,
        )
        write_quality_report(quality_report_path, quality_report)
        print_timing_report(output_segments)
        print_readability_report(output_segments)
        print(
            "Final quality report: "
            f"status={quality_report['status']}, "
            f"review_items={quality_report['summary']['review_items']}, "
            f"pixel_overflow={quality_report['summary']['pixel_overflow_events']}, "
            f"model_hide_overrides={quality_report['summary']['model_hide_overrides']}, "
            f"path={quality_report_path}"
        )
        if quality_report["status"] == "fail":
            raise RuntimeError(
                "Final subtitle quality validation failed. "
                f"Review {quality_report_path} before delivery."
            )
        print(f"Success! Subtitles saved to {out_dir}")
    finally:
        if temp_audio.exists():
            try:
                temp_audio.unlink()
            except OSError as exc:
                print(f"Warning: could not remove temporary audio {temp_audio}: {exc}", file=sys.stderr)
        if run_state_file and run_state_file.exists():
            try:
                run_state_file.unlink()
            except OSError as exc:
                print(f"Warning: could not remove run state {run_state_file}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
