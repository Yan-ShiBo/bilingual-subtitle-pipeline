import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ass_styles import STYLE_PROFILE_NAMES, ass_style_header, probe_video_play_resolution
from font_delivery import font_family_name, package_ass_with_font, validate_font_path
from llm_policy import (
    generation_profile,
    indexed_subtitle_schema,
    movie_name_schema,
)
from output_paths import OUTPUT_DIRECTORY_NAME, resolve_output_root, suggested_output_root
from pipeline_policy import PROCESSING_POLICY_VERSION, TERMINOLOGY_POLICY_VERSION
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


Segment = Dict[str, Any]


@dataclass(frozen=True)
class ExistingSubtitleTracks:
    english: Optional[List[Segment]]
    chinese: List[Segment]
    source_label: str


VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m2ts", ".ts", ".mov", ".wmv"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}
UNKNOWN_SOURCE_LANGUAGE_TAGS = {"", "auto", "source", "cached source", "subtitle", "embedded subtitle"}
SOURCE_CACHE_VERSION = 4
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
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


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
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )

        detected_language = str(getattr(info, "language", "") or requested_language or language or "auto")
        print(f"Detected language '{detected_language}' with probability {info.language_probability}")

        results: List[Segment] = []
        for segment in segments:
            words = []
            for word in getattr(segment, "words", None) or []:
                text = getattr(word, "word", "").strip()
                if text:
                    words.append(
                        {
                            "start": float(getattr(word, "start", segment.start)),
                            "end": float(getattr(word, "end", segment.end)),
                            "word": text,
                        }
                    )

            text = normalize_space(segment.text)
            if not text:
                continue
            item = {
                "id": len(results),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "source_language": detected_language,
            }
            if words:
                item["words"] = words
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
) -> str:
    base_url = "http://localhost:11434"
    if model.startswith("remote:"):
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
            "seed": profile.seed,
            "num_ctx": profile.num_ctx,
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


def parse_movie_name(filename: str, llm_model: str) -> Tuple[str, str]:
    system_prompt = "You extract clean movie and series names from release filenames."
    prompt = f"""
Given the filename "{filename}", extract:
1. "series_name": for a series or show, the show name; for a standalone movie, the movie name.
2. "movie_name": the clean episode/movie name. Preserve episode markers like 1of8 when present.

Return ONLY a valid JSON object with keys "series_name" and "movie_name".
"""
    try:
        data = parse_json_response(
            call_llm(
                prompt,
                system_prompt,
                llm_model,
                role="metadata",
                response_schema=movie_name_schema(),
            )
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
EAST_ASIAN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def contains_east_asian(text: str) -> bool:
    return bool(EAST_ASIAN_RE.search(text or ""))


def clean_english_track_text(text: str) -> str:
    text = clean_subtitle_text(text)
    return "" if contains_east_asian(text) else text


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
        return text
    if not hasattr(to_simplified_text, "_converter"):
        setattr(to_simplified_text, "_converter", OpenCC("t2s"))
    return getattr(to_simplified_text, "_converter").convert(text)


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
        events.append(SubtitleEvent(start, end, text))
    return events


def merge_existing_subtitle_segments(
    en_segments: Optional[List[Segment]],
    zh_segments: List[Segment],
    source_label: str,
) -> List[Segment]:
    from subtitle_pipeline import pair_events

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
        start = timing_event.start
        end = timing_event.end
        en_text = clean_english_track_text(en_event.text) if en_event else ""
        zh_text = to_simplified_text(clean_subtitle_text(zh_event.text)) if zh_event else ""
        text = en_text or zh_text
        if not text and not zh_text:
            continue
        merged.append(
            {
                "id": len(merged),
                "start": start,
                "end": end,
                "text": text,
                "en": en_text,
                "zh": zh_text,
                "source_language": source_label,
                "display": True,
                "preserve_distinct_overlap": True,
            }
        )
    return merged


def find_existing_sidecar_subtitle_tracks(
    video_path: Path,
    selected_path: Optional[Path],
    explicit_path: Optional[Path] = None,
    explicit_zh_path: Optional[Path] = None,
    explicit_en_path: Optional[Path] = None,
) -> Optional[ExistingSubtitleTracks]:
    candidates = sidecar_candidates(video_path, selected_path, explicit_path, explicit_zh_path, explicit_en_path)
    if not candidates:
        return None

    zh_path: Optional[Path] = explicit_zh_path if explicit_zh_path and explicit_zh_path.exists() else None
    en_path: Optional[Path] = explicit_en_path if explicit_en_path and explicit_en_path.exists() else None
    for candidate in candidates:
        language = classify_subtitle_path(candidate)
        if language == "zh" and zh_path is None:
            zh_path = candidate
        elif language == "en" and en_path is None:
            en_path = candidate

    if zh_path is None:
        return None

    print(f"Using existing Chinese sidecar subtitle: {zh_path}")
    zh_segments = parse_subtitle_file(zh_path)
    en_segments: Optional[List[Segment]] = None
    if en_path and en_path != zh_path:
        print(f"Merging existing English sidecar subtitle: {en_path}")
        en_segments = parse_subtitle_file(en_path)

    return ExistingSubtitleTracks(
        english=en_segments,
        chinese=zh_segments,
        source_label="existing Chinese sidecar subtitle",
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
    )
    return merged or None


def is_english_subtitle_stream(stream: Any) -> bool:
    label = f"{stream.lang} {stream.title}".lower()
    return stream.lang in {"eng", "en"} or "english" in label


def choose_best_chinese_stream(streams: List[Any]) -> Tuple[Optional[Any], Optional[str]]:
    from subtitle_pipeline import classify_chinese, stream_score

    ranked = []
    for stream in streams:
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

    english = [stream for stream in streams if is_english_subtitle_stream(stream)]
    if not english:
        return None
    english.sort(key=stream_score, reverse=True)
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
            "source_language": "existing Chinese embedded subtitle",
        }
        for idx, event in enumerate(zh_events)
        if event.text.strip()
    ]
    if not zh_segments:
        return None

    en_segments: Optional[List[Segment]] = None
    if en_stream and en_stream.index != zh_stream.index:
        print(
            "Merging existing English embedded subtitle stream "
            f"0:{en_stream.index} {en_stream.lang or '-'} {en_stream.codec} {en_stream.title}"
        )
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
                "source_language": "existing English embedded subtitle",
            }
            for idx, event in enumerate(en_events)
            if event.text.strip()
        ]

    return ExistingSubtitleTracks(
        english=en_segments,
        chinese=zh_segments,
        source_label="existing Chinese embedded subtitle",
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
        requested = (source_language or "").lower()
        preferred = [
            stream
            for stream in streams
            if requested and (stream.lang.lower() == requested or requested in stream.title.lower())
        ]
        pool = preferred or streams
        pool.sort(key=stream_score, reverse=True)
        stream = pool[0]

    language_hint = source_language if source_language and source_language != "auto" else stream.lang
    ocr_lang = args.subtitle_ocr_lang
    if not ocr_lang or ocr_lang == "auto":
        ocr_lang = choose_ocr_lang(language_hint)

    print(f"Using embedded subtitle stream 0:{stream.index} {stream.lang or '-'} {stream.codec} {stream.title}")
    events = get_stream_events(video_path, stream, ffmpeg, out_dir / "embedded" / f"stream_{stream.index:02d}", ocr_lang, args)
    segments = [
        {"id": idx, "start": event.start, "end": event.end, "text": event.text, "source_language": stream.lang or source_language}
        for idx, event in enumerate(events)
        if event.text.strip()
    ]
    if not segments:
        raise RuntimeError(f"Embedded subtitle stream 0:{stream.index} did not produce usable subtitle events.")
    return segments


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


def split_text_balanced(text: str, chunk_count: int, joiner: str) -> List[str]:
    units = list(text) if joiner == "" else text.split()
    if not units:
        return [""] * max(1, chunk_count)

    chunk_count = max(1, min(chunk_count, len(units)))
    if chunk_count == 1:
        return [joiner.join(units).strip()]

    strong_punctuation = "。！？!?"
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
    boundaries = [0]
    for boundary_index in range(1, chunk_count):
        previous = boundaries[-1]
        remaining_chunks = chunk_count - boundary_index
        minimum = previous + 1
        maximum = len(units) - remaining_chunks
        ideal = round(len(units) * boundary_index / chunk_count)

        def boundary_score(end: int) -> Tuple[int, int, int]:
            previous_unit = units[end - 1]
            next_unit = units[end] if end < len(units) else ""
            score = abs(end - ideal) * 4
            if previous_unit.endswith(tuple(strong_punctuation)):
                score -= 12
            elif previous_unit.endswith(tuple(weak_punctuation)):
                score -= 6
            if joiner:
                normalized_previous = previous_unit.rstrip(".,;:!?").casefold()
                if normalized_previous in no_break_after:
                    score += 10
            else:
                if previous_unit in "（《「『【“‘":
                    score += 12
                if next_unit in "），。！？；：》」』】”’":
                    score += 12
            return score, abs(end - ideal), end

        selected = min(range(minimum, maximum + 1), key=boundary_score)
        boundaries.append(selected)
    boundaries.append(len(units))

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
    max_available_chunks = min(available_chunk_counts) if available_chunk_counts else 1
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
    chunk_count = min(chunk_count, max_available_chunks)

    while True:
        en_chunks = (
            split_text_balanced(
                en_text,
                chunk_count,
                joiner="" if en_is_compact else " ",
            )
            if en_text
            else [""] * chunk_count
        )
        zh_chunks = split_text_balanced(zh_text, chunk_count, joiner="") if zh_text else [""] * chunk_count
        en_chunks += [""] * (chunk_count - len(en_chunks))
        zh_chunks += [""] * (chunk_count - len(zh_chunks))
        if source_is_en:
            text_chunks = en_chunks
        elif source_is_zh:
            text_chunks = zh_chunks
        else:
            text_chunks = split_text_balanced(source_text, chunk_count, joiner=source_joiner)
            text_chunks += [""] * (chunk_count - len(text_chunks))
        limits_ok = all(
            (
                not chunk
                or (
                    len(chunk) <= en_max_chars
                    and (en_is_compact or len(chunk.split()) <= max_words)
                )
            )
            for chunk in en_chunks
        ) and all(not chunk or len(chunk) <= zh_max_chars for chunk in zh_chunks) and all(
            not chunk
            or (
                len(chunk) <= source_char_limit
                and (source_is_cjk or len(chunk.split()) <= max_words)
            )
            for chunk in text_chunks
        )
        if layout_policy is not None:
            limits_ok = limits_ok and all(
                not chunk or layout_policy.fits(chunk, "en", bilingual=bilingual)
                for chunk in en_chunks
            ) and all(
                not chunk or layout_policy.fits(chunk, "zh", bilingual=bilingual)
                for chunk in zh_chunks
            )
        if limits_ok or chunk_count >= max_available_chunks:
            break
        chunk_count += 1

    metadata = {
        key: value
        for key, value in segment.items()
        if key not in {"id", "start", "end", "text", "en", "zh", "words", "display_start", "display_end"}
    }
    output: List[Segment] = []
    for index in range(chunk_count):
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


def model_hide_support_reason(
    source_segments: List[Segment],
    processed_segments: List[Segment],
    index: int,
    *,
    max_gap: float = 4.0,
) -> Optional[str]:
    source = source_segments[index]
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
    model_requested_hidden = segment.get("model_display_requested") is False
    if not model_requested_hidden and segment.get("display", True) is False:
        model_requested_hidden = not bool(segment.get("display_suppression"))
    if not model_requested_hidden:
        return

    segment["model_display_requested"] = False
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
) -> List[Segment]:
    visible: List[Segment] = []
    deduplicated = suppress_recent_exact_duplicates(segments, max_duration=max_duration)
    for segment in deduplicated:
        if segment.get("display", True) is False:
            continue
        item = dict(segment)
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
    return optimize_readability_timing(non_overlapping, max_duration=max_duration)


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
    return (
        f"[{index}] {float(segment['start']):.2f}->{float(segment['end']):.2f} "
        f"{segment['text']}{suffix}"
    )


def format_bilingual_prompt_segment(
    index: int,
    segment: Segment,
    *,
    next_start: Optional[float] = None,
    include_readability_limits: bool = False,
) -> str:
    text = str(segment.get("text", ""))
    en_text = clean_english_track_text(str(segment.get("en") or ("" if contains_cjk(text) else text)))
    zh_text = to_simplified_text(clean_subtitle_text(str(segment.get("zh") or (text if contains_cjk(text) else ""))))
    suffix = ""
    if include_readability_limits:
        chinese_budget, english_budget = reading_budgets(segment, next_start)
        suffix = f" | LIMITS: ZH_MAX={chinese_budget}, EN_MAX={english_budget}"
    return (
        f"[{index}] {float(segment['start']):.2f}->{float(segment['end']):.2f} "
        f"EN: {en_text or '-'} | ZH: {zh_text or '-'}{suffix}"
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


def terminology_from_segments(segments: List[Segment]) -> Dict[str, Dict[str, str]]:
    glossary: Dict[str, Dict[str, str]] = {}
    for segment in segments:
        register_terminology(glossary, segment.get("terminology"))
    return glossary


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
        existing = glossary.get(key)
        if existing is None:
            existing = {"source": source, "target": target}
            glossary[key] = existing
        accepted.append(dict(existing))
    return accepted


def source_contains_term(source_text: str, source_term: str) -> bool:
    source = normalize_space(source_term)
    if not source:
        return False
    source_pattern = re.escape(source.casefold())
    if source.isascii() and any(character.isalnum() for character in source):
        source_pattern = rf"(?<!\w){source_pattern}(?!\w)"
    return bool(re.search(source_pattern, normalize_space(source_text).casefold()))


def applicable_terminology_entries(
    entries: List[Dict[str, str]],
    source_text: str,
) -> List[Dict[str, str]]:
    return [
        entry
        for entry in entries
        if source_contains_term(source_text, str(entry.get("source") or ""))
    ]


def format_terminology_glossary(
    glossary: Dict[str, Dict[str, str]],
    relevant_source_text: str = "",
) -> str:
    if not glossary:
        return "(none yet)"
    values = list(glossary.values())
    relevant = [
        entry
        for entry in values
        if source_contains_term(relevant_source_text, entry["source"])
    ]
    selected: List[Dict[str, str]] = []
    selected_keys: set[str] = set()
    for entry in relevant + values[-250:]:
        key = entry["source"].casefold()
        if key in selected_keys:
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

    target_folded = normalize_space(target_text).casefold()
    conflicts: List[Dict[str, str]] = []
    for entry in combined.values():
        source = normalize_space(entry["source"])
        target = normalize_space(entry["target"])
        if not source or not target:
            continue
        if source_contains_term(source_text, source) and target.casefold() not in target_folded:
            conflicts.append(dict(entry))
    return conflicts


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
    display = parse_display_flag(item.get("display", item.get("show", item.get("keep", True))))
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
) -> bool:
    if require_corrected and not normalize_space(corrected):
        return False
    if not normalize_space(translated):
        return False

    orig_cjk_count = len(CJK_RE.findall(original_text))
    corr_cjk_count = len(CJK_RE.findall(corrected))
    trans_has_cjk = contains_cjk(translated)

    # 英文不是英文：如果修正后的中文字符数量超过了原始源文中的中文字符数量，说明LLM在擅自翻译或添加中文
    if corr_cjk_count > orig_cjk_count:
        return False
        
    # 中文不是中文：译文无中文字符，但包含多个字母（说明可能是生硬复制了外文句子或根本没翻译）
    if not trans_has_cjk and any(character.isalpha() for character in translated):
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
    require_display: bool = False,
) -> Dict[str, Any]:
    print(f"检测到语言异常，正在重试单句: {segment.get('text', '')}")
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
            include_readability_limits=True,
        )
        prompt = f"""
You will proofread only the TARGET LINE(S).

These subtitles may already contain Chinese. Proofread existing English and Chinese text.
English may also be OCR output and may contain recognition errors.
Do not retranslate a non-empty Chinese line. Translate from English only when the Chinese line is empty or misses information that exists only in English.

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
- Correct obvious English and Chinese OCR/subtitle recognition errors.
- Convert Traditional Chinese to natural Simplified Chinese.
- If corrected_chinese is already non-empty and complete, preserve its meaning and only proofread it. Do not replace it with a fresh translation.
- Translate from English into corrected_chinese only when the original Chinese is empty or clearly lacks information present in the English line.
- Keep corrected_english in natural English when an English source exists.
- IMPORTANT: The `corrected_english` field MUST be purely English. If the input `English:` field contains Chinese characters, you MUST translate them into English, or leave the field empty. Do not put Chinese characters in `corrected_english`.
- If English contains SDH/non-speech cues such as music, applause, laughter, or speaker labels and the Chinese line lacks that information, add only that missing cue in concise Simplified Chinese.
- You MUST translate uppercase descriptive text in parentheses or brackets (e.g. "(SIGHS)" or "[MUSIC]") into Simplified Chinese.
- LIMITS are maximum visible character budgets calculated from the available display time. If Chinese exceeds ZH_MAX, condense it without losing the intended meaning. Otherwise preserve the existing Chinese meaning and wording as much as possible.
- Keep each Chinese line within 16 characters and each English line within 42 characters where the text permits.
- Keep names and recurring terminology consistent across the context. Do not translate a name in one line and leave the same name untranslated in another unless the context clearly requires it.
- Reuse every applicable APPROVED MOVIE-WIDE TERMINOLOGY mapping exactly.
- In terminology, return only personal names or recurring terms that occur in this target line. Use an empty list when there are none.
- Never output or change timestamps. Timing is enforced by the subtitle pipeline.
- Keep corrected_english empty only when no English source exists or when the original English text contains no useful translatable English.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
"""
    else:
        batch_text = format_prompt_segment(
            0,
            segment,
            next_start=next_start,
            include_readability_limits=True,
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
- Keep corrected_text in the original source language.
- Translate into natural Simplified Chinese.
- LIMITS are maximum visible character budgets calculated from the available display time. Keep chinese_translation within ZH_MAX by using concise natural Chinese without dropping the intended meaning.
- Keep each Chinese line within 16 characters and each source line within 42 characters where the text permits.
- Keep personal names and recurring terminology consistent across the context. Do not translate a name in one line and leave the same name untranslated in another unless the context clearly requires it.
- Reuse every applicable APPROVED MOVIE-WIDE TERMINOLOGY mapping exactly.
- In terminology, return only personal names or recurring terms that occur in this target line. Use an empty list when there are none.
"""
    
    for attempt in range(10):
        try:
            response_role = "proofread" if is_proofread else "translation"
            data = parse_json_response(
                call_llm(
                    prompt,
                    system_prompt,
                    llm_model,
                    role=response_role,
                    response_schema=indexed_subtitle_schema(1, response_role),
                )
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
                        print(f"重试单句第 {attempt + 1} 次仍请求隐藏唯一字幕，继续重试...")
                        continue
                    return item
                if is_proofread:
                    orig_en, orig_zh = original_en_zh(segment)
                    corrected = clean_english_track_text(str(item.get("corrected_english") or orig_en))
                    translated = str(item.get("corrected_chinese") or item.get("chinese_translation") or "")
                    if is_language_valid(
                        orig_en or orig_zh,
                        corrected,
                        translated,
                        require_corrected=bool(orig_en),
                    ):
                        return item
                else:
                    corrected = str(item.get("corrected_text") or item.get("corrected_source") or segment.get("text", ""))
                    translated = str(item.get("chinese_translation") or "")
                    if is_language_valid(segment.get("text", ""), corrected, translated):
                        return item
            print(f"重试单句第 {attempt + 1} 次依然语言异常，继续重试...")
        except Exception as exc:
            print(f"重试单句请求异常: {exc}")
    
    print("10次重试均失败，将中断当前批次并保留最近 checkpoint。")
    return {}


def translate_and_correct_segments(
    segments: List[Segment],
    llm_model: str,
    batch_size: int,
    context_lines: int,
    source_language: str,
    checkpoint_path: Optional[Path] = None,
    terminology_path: Optional[Path] = None,
) -> List[Segment]:
    print("Starting sentence-level LLM correction and translation...")
    language_label = "the source language" if not source_language or source_language == "auto" else source_language
    system_prompt = (
        "You are a professional subtitle editor and Chinese translator. "
        "Repair ASR/OCR errors, repeated hallucinated fragments, and duplicate subtitle loops while preserving source timing anchors."
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
    glossary = terminology_from_segments(processed_segments)
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
                include_readability_limits=True,
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
- If adjacent TARGET lines are ASR repetitions or overlapping fragments of the same spoken sentence, keep the earliest suitable object visible and set only later redundant objects to "display": false.
- For the kept object, use corrected_text and chinese_translation for the complete sentence.
- Never output or change timestamps. Timing is enforced by the subtitle pipeline.
- If a TARGET line only repeats or continues a sentence already clearly represented in PREVIOUS CONTEXT, set "display": false.
- If two TARGET lines are genuine consecutive new information, keep both visible.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
- Correct obvious ASR/OCR/subtitle errors in the source text before translating.
- Keep corrected_text in the original source language.
- Translate into natural Simplified Chinese.
- You MUST translate uppercase descriptive text in parentheses or brackets (e.g. "(SIGHS)" or "[MUSIC]") into Simplified Chinese.
- LIMITS are maximum visible character budgets calculated from the available display time. Keep chinese_translation within ZH_MAX by using concise natural Chinese without dropping the intended meaning.
- Keep each Chinese line within 16 characters and each source line within 42 characters where the text permits.
- Keep personal names and recurring terminology consistent across the context. Do not translate a name in one line and leave the same name untranslated in another unless the context clearly requires it.
- Reuse every applicable APPROVED MOVIE-WIDE TERMINOLOGY mapping exactly.
- In terminology, return only personal names or recurring terms that occur in that target line. Once a mapping is approved, never propose a different target. Use an empty list when there are none.
"""

        try:
            lookup: Optional[Dict[int, Dict[str, Any]]] = None
            for attempt in range(3):
                try:
                    data = parse_json_response(
                        call_llm(
                            prompt,
                            system_prompt,
                            llm_model,
                            role="translation",
                            response_schema=indexed_subtitle_schema(
                                len(batch),
                                "translation",
                            ),
                        )
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
                corrected = clean_subtitle_text(corrected_val)
                translated_val = str(item.get("chinese_translation") or "")
                translated = clean_subtitle_text(translated_val)
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
                    not is_language_valid(segment["text"], corrected, translated)
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
                    corrected = clean_subtitle_text(corrected_val)
                    translated_val = str(item.get("chinese_translation") or "")
                    translated = clean_subtitle_text(translated_val)
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
                        not is_language_valid(segment["text"], corrected, translated)
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
) -> List[Segment]:
    print("Starting LLM proofreading for existing Chinese subtitles; translation is skipped.")
    system_prompt = (
        "You are a professional bilingual subtitle proofreader. "
        "Fix English and Chinese OCR/subtitle recognition errors, repeated lines, and duplicate loops. "
        "Do not translate when Chinese subtitles are already provided."
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
    glossary = terminology_from_segments(processed_segments)
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
                include_readability_limits=True,
            )
            for j, segment in enumerate(batch)
        )
        context_start_index = max(0, i - context_lines)
        context_end_index = min(len(segments), i + len(batch) + context_lines)
        context_start = float(segments[context_start_index]["start"])
        context_end = float(segments[context_end_index - 1]["end"])

        prompt = f"""
You will proofread only the TARGET LINE(S).

These subtitles may already contain Chinese. Proofread existing English and Chinese text.
English may also be OCR output and may contain recognition errors.
Do not retranslate a non-empty Chinese line. Translate from English only when the Chinese line is empty or misses information that exists only in English.

=== PREVIOUS CONTEXT, REFERENCE ONLY ===
{before_text}

=== PREVIOUS APPROVED OUTPUT, MATCH NAMES AND TERMINOLOGY ===
{approved_output_context}

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
- Correct obvious English and Chinese OCR/subtitle recognition errors.
- Convert Traditional Chinese to natural Simplified Chinese.
- If corrected_chinese is already non-empty and complete, preserve its meaning and only proofread it. Do not replace it with a fresh translation.
- Translate from English into corrected_chinese only when the original Chinese is empty or clearly lacks information present in the English line.
- Keep corrected_english in natural English when an English source exists.
- IMPORTANT: The `corrected_english` field MUST be purely English. If the input `English:` field contains Chinese characters, you MUST translate them into English, or leave the field empty. Do not put Chinese characters in `corrected_english`.
- If English contains SDH/non-speech cues such as music, applause, laughter, or speaker labels and the Chinese line lacks that information, add only that missing cue in concise Simplified Chinese.
- You MUST translate uppercase descriptive text in parentheses or brackets (e.g. "(SIGHS)" or "[MUSIC]") into Simplified Chinese.
- LIMITS are maximum visible character budgets calculated from the available display time. If Chinese exceeds ZH_MAX, condense it without losing the intended meaning. Otherwise preserve the existing Chinese meaning and wording as much as possible.
- Keep each Chinese line within 16 characters and each English line within 42 characters where the text permits.
- Keep names and recurring terminology consistent across the context. Do not translate a name in one line and leave the same name untranslated in another unless the context clearly requires it.
- Reuse every applicable APPROVED MOVIE-WIDE TERMINOLOGY mapping exactly.
- In terminology, return only personal names or recurring terms that occur in that target line. Once a mapping is approved, never propose a different target. Use an empty list when there are none.
- If adjacent TARGET lines are repeated, overlapping, or fragments of the same subtitle, keep the earliest suitable object visible and set only later redundant objects to "display": false.
- Never output or change timestamps. Timing is enforced by the subtitle pipeline.
- Keep corrected_english empty only when no English source exists or when the original English text contains no useful translatable English.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
"""

        try:
            lookup: Optional[Dict[int, Dict[str, Any]]] = None
            for attempt in range(3):
                try:
                    data = parse_json_response(
                        call_llm(
                            prompt,
                            system_prompt,
                            llm_model,
                            role="proofread",
                            response_schema=indexed_subtitle_schema(
                                len(batch),
                                "proofread",
                            ),
                        )
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
                corrected_en = clean_english_track_text(corrected_en_val)
                corrected_zh_val = str(
                    item.get("corrected_chinese")
                    or item.get("chinese")
                    or item.get("zh")
                    or original_zh
                )
                corrected_zh = to_simplified_text(clean_subtitle_text(corrected_zh_val))
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

                if display and (
                    not is_language_valid(
                        original_en or original_zh,
                        corrected_en,
                        corrected_zh,
                        require_corrected=bool(original_en),
                    )
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
                        "",
                        is_proofread=True,
                        terminology_text=format_terminology_glossary(
                            glossary,
                            terminology_source,
                        ),
                        require_display=forced_model_hide,
                    )
                    if not retry_item:
                        raise RuntimeError(f"LLM could not produce a valid proofread result for segment {i + j}")
                    item.update(retry_item)
                    corrected_en_val = str(item.get("corrected_english") or item.get("en") or original_en)
                    corrected_en = clean_english_track_text(corrected_en_val)
                    corrected_zh_val = str(
                        item.get("corrected_chinese")
                        or item.get("chinese")
                        or item.get("zh")
                        or original_zh
                    )
                    corrected_zh = to_simplified_text(clean_subtitle_text(corrected_zh_val))
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
                    if display and (
                        not is_language_valid(
                            original_en or original_zh,
                            corrected_en,
                            corrected_zh,
                            require_corrected=bool(original_en),
                        )
                        or conflicts
                    ):
                        raise RuntimeError(f"LLM returned an invalid proofread result for segment {i + j}")

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


def checkpoint_matches_segments(
    cached: List[Segment],
    segments: List[Segment],
    expected_mode: Optional[str] = None,
    expected_model: Optional[str] = None,
    expected_policy_version: Optional[int] = None,
    expected_terminology_policy_version: Optional[int] = None,
) -> bool:
    for index, cached_segment in enumerate(cached):
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
        current_fingerprint = str(current.get("source_request_fingerprint") or "")
        cached_fingerprint = str(cached_segment.get("source_request_fingerprint") or "")
        if current_fingerprint and cached_fingerprint != current_fingerprint:
            print(
                "Ignoring checkpoint because its source request fingerprint does not match "
                f"the current source at item {index}."
            )
            return False
        same_start = abs(float(cached_segment["start"]) - float(current["start"])) <= 0.02
        same_end = abs(float(cached_segment["end"]) - float(current["end"])) <= 0.02
        same_text = clean_subtitle_text(str(cached_segment.get("text", ""))) == clean_subtitle_text(str(current.get("text", "")))
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


def main() -> None:
    configure_output_encoding()
    parser = argparse.ArgumentParser(description="Transcribe, import, OCR, correct, and translate subtitles to ASS.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file or Blu-ray folder")
    parser.add_argument("--source", choices=["auto", "sidecar", "srt", "embedded", "audio"], default="auto", help="Subtitle source")
    parser.add_argument("--srt", type=str, help="Explicit sidecar SRT path. Kept for compatibility.")
    parser.add_argument("--subtitle-file", type=str, help="Explicit sidecar subtitle path: srt, ass, ssa, or vtt.")
    parser.add_argument("--chinese-subtitle-file", type=str, help="Explicit Chinese sidecar subtitle path for bilingual merge.")
    parser.add_argument("--english-subtitle-file", type=str, help="Explicit English sidecar subtitle path for bilingual merge.")
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
    sync_report_path = out_dir / f"{movie_name}.subtitle-sync.report.json"
    terminology_path = out_dir / f"{movie_name}.terminology.json"
    run_state_file = Path(args.run_state_file) if args.run_state_file else None

    sidecar_path = Path(args.subtitle_file or args.srt) if (args.subtitle_file or args.srt) else None
    chinese_sidecar_path = Path(args.chinese_subtitle_file) if args.chinese_subtitle_file else None
    english_sidecar_path = Path(args.english_subtitle_file) if args.english_subtitle_file else None
    if sidecar_path is None:
        sidecar_path = english_sidecar_path or chinese_sidecar_path
    merge_existing_subtitles = args.merge_existing_subtitles == "yes"
    if args.source in ("auto", "sidecar", "srt") and not sidecar_path:
        sidecar_path = find_sidecar_subtitle(video_path, input_path)
    source_request_fingerprint = build_source_request_fingerprint(
        video_path,
        input_path,
        args,
        sidecar_path=sidecar_path,
        chinese_sidecar_path=chinese_sidecar_path,
        english_sidecar_path=english_sidecar_path,
    )
    compatible_source_request_fingerprints = {
        build_source_request_fingerprint(
            video_path,
            input_path,
            args,
            sidecar_path=sidecar_path,
            chinese_sidecar_path=chinese_sidecar_path,
            english_sidecar_path=english_sidecar_path,
            sync_policy_version=version,
        )
        for version in COMPATIBLE_SYNC_POLICY_VERSIONS
    }
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
                    print(f"Reusing cached source segments from {source_segments_path} ({len(subtitle_segments)} events).")
                else:
                    print(
                        "Ignoring cached source segments because their language does not match "
                        f"the requested source/asr language: {source_segments_path}"
                    )
            except Exception as exc:
                print(f"Ignoring unreadable source cache {source_segments_path}: {exc}")

        if subtitle_segments is None:
            if merge_existing_subtitles and args.source in ("auto", "sidecar", "srt"):
                existing_tracks = find_existing_sidecar_subtitle_tracks(
                    video_path,
                    input_path,
                    explicit_path=sidecar_path if (args.subtitle_file or args.srt) else None,
                    explicit_zh_path=chinese_sidecar_path,
                    explicit_en_path=english_sidecar_path,
                )
                if existing_tracks:
                    actual_source = "existing Chinese sidecar subtitles"
                    source_language = "existing Chinese subtitle"

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
                        source_language = "existing Chinese subtitle"
                except Exception as exc:
                    if args.source == "embedded":
                        print(f"Failed to load embedded Chinese subtitles: {exc}", file=sys.stderr)
                        raise
                    print(f"No existing Chinese embedded subtitles available ({exc}).")

            if existing_tracks:
                synchronized_tracks, sync_report = synchronize_existing_subtitle_tracks(
                    video_path,
                    existing_tracks,
                    mode=args.subtitle_sync,
                    report_path=sync_report_path,
                    audio_stream=args.audio_stream,
                )
                subtitle_segments = merge_existing_subtitle_segments(
                    synchronized_tracks.english,
                    synchronized_tracks.chinese,
                    synchronized_tracks.source_label,
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
                if args.source in ("auto", "sidecar", "srt") and sidecar_path and sidecar_path.exists():
                    actual_source = "sidecar"
                    source_segments = parse_subtitle_file(sidecar_path)
                    if source_language == "auto":
                        source_language = "subtitle"
                elif args.source in ("sidecar", "srt"):
                    raise FileNotFoundError("Sidecar subtitle source requested, but no subtitle file was found.")
                elif args.source in ("auto", "embedded"):
                    try:
                        actual_source = "embedded"
                        fallback_stream_index = args.subtitle_stream
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
                        extract_audio(video_path, temp_audio, audio_stream=args.audio_stream)
                        source_segments = transcribe_audio(temp_audio, language=asr_language)
                        if source_language == "auto":
                            source_language = infer_source_language_from_segments(source_segments, asr_language)
                else:
                    actual_source = "audio"
                    extract_audio(video_path, temp_audio, audio_stream=args.audio_stream)
                    source_segments = transcribe_audio(temp_audio, language=asr_language)
                    if source_language == "auto":
                        source_language = infer_source_language_from_segments(source_segments, asr_language)

                subtitle_segments = source_segments

        existing_timing_source = actual_source in {
            "sidecar",
            "embedded",
            "existing Chinese sidecar subtitles",
            "existing Chinese embedded subtitles",
        }
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
                mode=args.subtitle_sync,
                report_path=sync_report_path,
                audio_stream=args.audio_stream,
            )
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
        for segment in subtitle_segments:
            segment["source_cache_version"] = SOURCE_CACHE_VERSION
            if not loaded_legacy_source_cache:
                segment["source_request_fingerprint"] = source_request_fingerprint
        write_json_atomic(source_segments_path, subtitle_segments)

        print(f"Actual subtitle source: {actual_source}")
        print_timing_report(subtitle_segments)

        if segments_have_chinese(subtitle_segments):
            processed_segments = proofread_existing_chinese_segments(
                subtitle_segments,
                llm_model=args.llm_model,
                batch_size=args.batch_size,
                context_lines=args.context_lines,
                checkpoint_path=checkpoint_path,
                terminology_path=terminology_path,
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
        output_segments = prepare_segments_for_output(
            processed_segments,
            max_words=args.max_words,
            max_chars=args.max_chars,
            max_duration=args.max_duration,
            layout_policy=layout_policy,
        )
        en_ass = out_dir / f"{movie_name}.en.ass"
        zh_ass = out_dir / f"{movie_name}.zh.ass"
        bi_ass = out_dir / f"{movie_name}.bilingual.ass"
        quality_report_path = out_dir / f"{movie_name}.quality-report.json"

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
        artifact_paths = {
            "english_ass": str(en_ass),
            "chinese_ass": str(zh_ass),
            "bilingual_ass": str(bi_ass),
            "quality_report": str(quality_report_path),
        }
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
