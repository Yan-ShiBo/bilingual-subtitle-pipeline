import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


Segment = Dict[str, Any]
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m2ts", ".ts", ".mov", ".wmv"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}


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


def transcribe_audio(audio_path: Path, language: Optional[str] = "en") -> List[Segment]:
    print("Loading faster-whisper large-v3 model with FP16...")
    from faster_whisper import WhisperModel
    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
    )

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

    print(f"Detected language '{info.language}' with probability {info.language_probability}")

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
        }
        if words:
            item["words"] = words
        results.append(item)
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")

    return results


def call_llm(prompt: str, system_prompt: str = "", model: str = "qwen3:14b") -> str:
    base_url = "http://localhost:11434/v1"
    if model.startswith("remote:"):
        parts = model.split(":", 2)
        if len(parts) == 3:
            base_url = "http://localhost:11435/v1"
            model = parts[2]

    client = OpenAI(
        base_url=base_url,
        api_key="ollama",
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        extra_body={
            "think": False,
            "keep_alive": -1,
        },
    )
    return response.choices[0].message.content


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


def parse_movie_name(filename: str, llm_model: str) -> Tuple[str, str]:
    system_prompt = "You extract clean movie and series names from release filenames."
    prompt = f"""
Given the filename "{filename}", extract:
1. "series_name": for a series or show, the show name; for a standalone movie, the movie name.
2. "movie_name": the clean episode/movie name. Preserve episode markers like 1of8 when present.

Return ONLY a valid JSON object with keys "series_name" and "movie_name".
"""
    try:
        data = parse_json_response(call_llm(prompt, system_prompt, llm_model))
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
    return normalize_space(text)


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
    direct_stems = {video_path.stem.lower()}
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
        score = 0
        if stem in direct_stems:
            score += 100
        if video_path.stem.lower() in stem or stem in video_path.stem.lower():
            score += 50
        if any(token in stem for token in ("en", "eng", "english")):
            score += 10
        scored.append((score, candidate.name.lower(), candidate))
    unique = {}
    for score, name, path in scored:
        if path not in unique or score > unique[path][0]:
            unique[path] = (score, name, path)
    return [item[2] for item in sorted(unique.values(), key=lambda item: (item[0], item[1]), reverse=True)]


def find_sidecar_subtitle(video_path: Path, selected_path: Optional[Path] = None) -> Optional[Path]:
    subtitles = find_sidecar_subtitles(video_path, selected_path)
    return subtitles[0] if subtitles else None


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


def split_segment_by_words(segment: Segment, max_words: int, max_chars: int, max_duration: float) -> List[Segment]:
    words = segment.get("words") or []
    if not words:
        return []

    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for word in words:
        current.append(word)
        text_now = join_words(current)
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        sentence_end = bool(re.search(r"[.!?]$", word.get("word", "")))
        hard_limit = len(current) >= max_words or len(text_now) >= max_chars or duration >= max_duration
        if current and (sentence_end or hard_limit):
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
    max_words: int = 14,
    max_chars: int = 82,
    max_duration: float = 6.0,
) -> List[Segment]:
    output: List[Segment] = []

    for segment in segments:
        text = clean_subtitle_text(segment.get("text", ""))
        if not text:
            continue
        segment = {**segment, "text": text}
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


def build_context_text(segments: List[Segment], start: int, end: int, context_lines: int) -> Tuple[str, str]:
    before_start = max(0, start - context_lines)
    after_end = min(len(segments), end + context_lines)
    before = "\n".join(f"[{index}] {segments[index]['text']}" for index in range(before_start, start))
    after = "\n".join(f"[{index}] {segments[index]['text']}" for index in range(end, after_end))
    return before, after


def translate_and_correct_segments(
    segments: List[Segment],
    llm_model: str,
    batch_size: int,
    context_lines: int,
    source_language: str,
    checkpoint_path: Optional[Path] = None,
) -> List[Segment]:
    print("Starting sentence-level LLM correction and translation...")
    language_label = "the source language" if not source_language or source_language == "auto" else source_language
    system_prompt = (
        "You are a professional subtitle editor and Chinese translator. "
        "Keep timing granularity fixed: never merge, split, reorder, omit, or add subtitle items."
    )

    processed_segments: List[Segment] = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and len(cached) <= len(segments) and checkpoint_matches_segments(cached, segments):
                processed_segments = cached
                print(f"Resuming from checkpoint with {len(processed_segments)} completed segments.")
            else:
                print(f"Ignoring checkpoint because it does not match the current subtitle segmentation.")
        except Exception as exc:
            print(f"Ignoring unreadable checkpoint {checkpoint_path}: {exc}")

    start_index = len(processed_segments)
    start_index -= start_index % batch_size
    processed_segments = processed_segments[:start_index]

    for i in range(start_index, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        print(f"Processing segments {i + 1} to {i + len(batch)} of {len(segments)}...")

        before_text, after_text = build_context_text(segments, i, i + len(batch), context_lines)
        batch_text = "\n".join(f"[{j}] {segment['text']}" for j, segment in enumerate(batch))

        prompt = f"""
You will correct and translate only the TARGET LINE(S).

Use the previous and following context to understand names, pronouns, topic continuity, and terminology.
Do not translate the context sections. They are reference only.
The source subtitle/audio language is: {language_label}.

=== PREVIOUS CONTEXT, REFERENCE ONLY ===
{before_text}

=== TARGET LINE(S) TO OUTPUT ===
{batch_text}

=== FOLLOWING CONTEXT, REFERENCE ONLY ===
{after_text}

Return a raw JSON list with exactly {len(batch)} objects.
Each object must correspond to one TARGET line and keep the same local index.
Schema:
[
  {{"index": 0, "corrected_text": "...", "chinese_translation": "..."}}
]

Rules:
- One input line must produce one output object.
- Do not merge two lines.
- Do not split one line into multiple objects.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
- Correct obvious ASR/OCR/subtitle errors in the source text before translating.
- Keep corrected_text in the original source language.
- Translate into natural Simplified Chinese.
"""

        try:
            data = parse_json_response(call_llm(prompt, system_prompt, llm_model))
            if not isinstance(data, list):
                raise ValueError("LLM did not return a JSON list")

            lookup: Dict[int, Dict[str, Any]] = {}
            for item in data:
                if isinstance(item, dict) and "index" in item:
                    try:
                        lookup[int(item["index"])] = item
                    except (TypeError, ValueError):
                        pass

            batch_processed: List[Segment] = []
            for j, segment in enumerate(batch):
                item = lookup.get(j, {})
                corrected = clean_subtitle_text(
                    str(
                        item.get("corrected_text")
                        or item.get("corrected_source")
                        or item.get("corrected_english")
                        or segment["text"]
                    )
                )
                translated = clean_subtitle_text(str(item.get("chinese_translation") or ""))
                batch_processed.append(
                    {
                        **segment,
                        "en": corrected,
                        "zh": translated or corrected,
                    }
                )
        except Exception as exc:
            print(f"Warning: failed to process batch via LLM ({exc}). Falling back to original text.")
            batch_processed = [
                {**segment, "en": segment["text"], "zh": segment["text"]}
                for segment in batch
            ]

        validate_timing_preserved(batch, batch_processed, i)
        processed_segments.extend(batch_processed)

        if checkpoint_path:
            write_json_atomic(checkpoint_path, processed_segments)

    return processed_segments


def checkpoint_matches_segments(cached: List[Segment], segments: List[Segment]) -> bool:
    for index, cached_segment in enumerate(cached):
        current = segments[index]
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


def generate_ass(segments: List[Segment], out_path: Path, mode: str) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,20,1
Style: English,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,15,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for segment in segments:
            start = seconds_to_ass_time(float(segment["start"]))
            end = seconds_to_ass_time(float(segment["end"]))
            en_text = escape_ass_text(segment.get("en", segment["text"]))
            zh_text = escape_ass_text(segment.get("zh", ""))

            if mode == "en":
                handle.write(f"Dialogue: 0,{start},{end},English,,0,0,0,,{en_text}\n")
            elif mode == "zh":
                handle.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{zh_text}\n")
            elif mode == "bilingual":
                handle.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{zh_text}{{\\rEnglish}}\\N{en_text}\n")


def print_timing_report(segments: List[Segment]) -> None:
    if not segments:
        print("No subtitle events generated.")
        return

    durations = [float(segment["end"]) - float(segment["start"]) for segment in segments]
    gaps = [
        float(segments[index]["start"]) - float(segments[index - 1]["end"])
        for index in range(1, len(segments))
    ]
    print(
        "Timing report: "
        f"events={len(segments)}, "
        f"max_duration={max(durations):.2f}s, "
        f">6s={sum(duration > 6 for duration in durations)}, "
        f">8s={sum(duration > 8 for duration in durations)}, "
        f"gaps>5s={sum(gap > 5 for gap in gaps)}, "
        f"max_gap={max(gaps) if gaps else 0:.2f}s"
    )


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / ("1 " + "\u5b57\u5e55")


def main() -> None:
    configure_output_encoding()
    parser = argparse.ArgumentParser(description="Transcribe, import, OCR, correct, and translate subtitles to ASS.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file or Blu-ray folder")
    parser.add_argument("--source", choices=["auto", "sidecar", "srt", "embedded", "audio"], default="auto", help="Subtitle source")
    parser.add_argument("--srt", type=str, help="Explicit sidecar SRT path. Kept for compatibility.")
    parser.add_argument("--subtitle-file", type=str, help="Explicit sidecar subtitle path: srt, ass, ssa, or vtt.")
    parser.add_argument("--subtitle-stream", type=int, help="Embedded subtitle stream index, e.g. 2 for 0:2.")
    parser.add_argument("--audio-stream", type=int, help="Audio stream index, e.g. 1 for 0:1.")
    parser.add_argument("--source-language", default="auto", help="Source subtitle language for correction/translation context.")
    parser.add_argument("--asr-language", default="en", help="Whisper language code, or auto for detection.")
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
    parser.add_argument("--max-words", type=int, default=14, help="Maximum English words per subtitle event")
    parser.add_argument("--max-chars", type=int, default=82, help="Maximum English characters per subtitle event")
    parser.add_argument("--max-duration", type=float, default=6.0, help="Maximum seconds per subtitle event before splitting")
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

    output_root = Path(args.output_root) if args.output_root else default_output_root()
    out_dir = output_root / series_name / movie_name
    checkpoint_path = out_dir / f"{movie_name}.segments.checkpoint.json"
    source_segments_path = out_dir / f"{movie_name}.segments.source.json"

    sidecar_path = Path(args.subtitle_file or args.srt) if (args.subtitle_file or args.srt) else None
    if args.source in ("auto", "sidecar", "srt") and not sidecar_path:
        sidecar_path = find_sidecar_subtitle(video_path, input_path)

    temp_audio = video_path.with_suffix(".wav")
    try:
        actual_source = args.source
        source_language = args.source_language
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
                source_segments = load_embedded_subtitle_events(
                    video_path,
                    stream_index=args.subtitle_stream,
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
                source_segments = transcribe_audio(temp_audio, language=args.asr_language)
                if source_language == "auto":
                    source_language = args.asr_language
        else:
            actual_source = "audio"
            extract_audio(video_path, temp_audio, audio_stream=args.audio_stream)
            source_segments = transcribe_audio(temp_audio, language=args.asr_language)
            if source_language == "auto":
                source_language = args.asr_language

        print(f"Actual subtitle source: {actual_source}")

        subtitle_segments = split_segments_for_subtitles(
            source_segments,
            max_words=args.max_words,
            max_chars=args.max_chars,
            max_duration=args.max_duration,
        )
        subtitle_segments = apply_timing_sanity_rules(subtitle_segments, max_duration=args.max_duration)
        write_json_atomic(source_segments_path, subtitle_segments)
        print_timing_report(subtitle_segments)

        processed_segments = translate_and_correct_segments(
            subtitle_segments,
            llm_model=args.llm_model,
            batch_size=args.batch_size,
            context_lines=args.context_lines,
            source_language=source_language,
            checkpoint_path=checkpoint_path,
        )

        en_ass = out_dir / f"{movie_name}.en.ass"
        zh_ass = out_dir / f"{movie_name}.zh.ass"
        bi_ass = out_dir / f"{movie_name}.bilingual.ass"

        generate_ass(processed_segments, en_ass, "en")
        generate_ass(processed_segments, zh_ass, "zh")
        generate_ass(processed_segments, bi_ass, "bilingual")

        print_timing_report(processed_segments)
        print(f"Success! Subtitles saved to {out_dir}")
    finally:
        if temp_audio.exists():
            temp_audio.unlink()


if __name__ == "__main__":
    main()
