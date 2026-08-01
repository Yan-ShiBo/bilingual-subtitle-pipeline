from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import site
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFilter

from ass_styles import STYLE_PROFILE_NAMES, ass_style_header


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


TEXT_CODECS = {
    "ass",
    "ssa",
    "subrip",
    "srt",
    "webvtt",
    "mov_text",
    "text",
}

EXTRACT_CACHE_VERSION = 1
PGS_IMAGE_CACHE_VERSION = 2
OCR_CACHE_VERSION = 3
COMPATIBLE_OCR_CACHE_VERSIONS = {2, OCR_CACHE_VERSION}


@dataclass
class StreamInfo:
    index: int
    kind: str
    codec: str
    lang: str = ""
    title: str = ""
    metadata: dict[str, str] | None = None
    disposition: dict[str, int] | None = None

    @property
    def is_subtitle(self) -> bool:
        return self.kind.lower() == "subtitle"

    @property
    def is_audio(self) -> bool:
        return self.kind.lower() == "audio"

    @property
    def is_pgs(self) -> bool:
        return "pgs" in self.codec.lower() or "hdmv_pgs" in self.codec.lower()

    @property
    def is_text_subtitle(self) -> bool:
        codec = self.codec.lower()
        codec_tokens = set(re.split(r"[^a-z0-9_]+", codec))
        return bool(codec_tokens & TEXT_CODECS)

    @property
    def is_forced(self) -> bool:
        return bool((self.disposition or {}).get("forced"))

    @property
    def is_hearing_impaired(self) -> bool:
        return bool((self.disposition or {}).get("hearing_impaired"))

    @property
    def is_default(self) -> bool:
        return bool((self.disposition or {}).get("default"))


@dataclass
class SubtitleEvent:
    start: float
    end: float
    text: str
    confidence: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OcrObservation:
    text: str
    confidence: float | None = None
    line_confidences: list[float] | None = None


@dataclass
class PgsCompositionObject:
    object_id: int
    x: int
    y: int


@dataclass
class PgsComposition:
    pts: float
    width: int
    height: int
    palette_id: int
    objects: list[PgsCompositionObject]


@dataclass
class PgsObject:
    width: int
    height: int
    rle: bytes


@dataclass
class PgsImageEvent:
    start: float
    end: float
    image_path: Path
    image_hash: str


def log(message: str) -> None:
    print(message, flush=True)


def configure_windows_cuda_dll_paths() -> None:
    if os.name != "nt":
        return
    roots: list[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    
    dll_dirs: list[Path] = []
    
    # 1. Add nvidia python package DLL paths
    for root in roots:
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.exists():
            continue
        for candidate in nvidia_root.glob("**/bin"):
            dll_dirs.append(candidate)
            for child in candidate.iterdir():
                if child.is_dir():
                    dll_dirs.append(child)
                    
    # 2. Add system CUDA paths from environment variables
    for env_name, env_val in os.environ.items():
        if env_name.startswith("CUDA_PATH"):
            cuda_bin = Path(env_val) / "bin"
            if cuda_bin.exists():
                dll_dirs.append(cuda_bin)
                
    # 3. Add default system CUDA paths if not in env but exist
    default_cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if default_cuda_root.exists():
        try:
            for version_dir in default_cuda_root.iterdir():
                if version_dir.is_dir():
                    cuda_bin = version_dir / "bin"
                    if cuda_bin.exists():
                        dll_dirs.append(cuda_bin)
        except Exception:
            pass

    existing_path = os.environ.get("PATH", "")
    added_paths = set()
    for directory in dll_dirs:
        try:
            resolved = directory.resolve()
            text = str(resolved)
        except Exception:
            text = str(directory)
            
        if text in added_paths:
            continue
        added_paths.add(text)
        
        if text not in existing_path:
            os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(text)
        except Exception:
            pass




def suppress_paddlex_optional_langchain_imports() -> None:
    if "langchain_text_splitters" in sys.modules:
        return
    module = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:  # pragma: no cover - only for optional PaddleX import shim
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    module.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain_text_splitters"] = module
    if "modelscope" not in sys.modules:
        modelscope = types.ModuleType("modelscope")
        hub = types.ModuleType("modelscope.hub")
        errors = types.ModuleType("modelscope.hub.errors")

        class NotExistError(Exception):  # pragma: no cover - only for optional PaddleX source
            pass

        def snapshot_download(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - only for optional PaddleX source
            raise NotExistError("ModelScope is disabled in this OCR process.")

        modelscope.snapshot_download = snapshot_download
        errors.NotExistError = NotExistError
        hub.errors = errors
        modelscope.hub = hub
        sys.modules["modelscope"] = modelscope
        sys.modules["modelscope.hub"] = hub
        sys.modules["modelscope.hub.errors"] = errors


def run(cmd: list[str], check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + proc.stdout)
    return proc


def find_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - message path
        raise RuntimeError("ffmpeg was not found. Install ffmpeg or imageio-ffmpeg.") from exc


def find_ffprobe(ffmpeg: str) -> str | None:
    ffmpeg_path = Path(ffmpeg)
    executable = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    sibling = ffmpeg_path.with_name(executable)
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe")


def probe_streams_json(video: Path, ffprobe: str) -> list[StreamInfo]:
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(video),
        ],
        check=False,
        timeout=90,
    )
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        return []

    streams: list[StreamInfo] = []
    for raw in raw_streams:
        if not isinstance(raw, dict):
            continue
        tags = raw.get("tags") if isinstance(raw.get("tags"), dict) else {}
        metadata = {str(key): str(value) for key, value in tags.items()}
        for key in ("nb_frames", "bit_rate", "duration"):
            if raw.get(key) is not None:
                metadata[key] = str(raw[key])
        disposition = {
            str(key): int(bool(value))
            for key, value in (raw.get("disposition") or {}).items()
        }
        try:
            index = int(raw["index"])
        except (KeyError, TypeError, ValueError):
            continue
        streams.append(
            StreamInfo(
                index=index,
                lang=str(tags.get("language") or "").casefold(),
                kind=str(raw.get("codec_type") or ""),
                codec=str(raw.get("codec_name") or raw.get("codec_long_name") or ""),
                title=str(tags.get("title") or ""),
                metadata=metadata,
                disposition=disposition,
            )
        )
    return streams


def probe_streams(video: Path, ffmpeg: str) -> list[StreamInfo]:
    ffprobe = find_ffprobe(ffmpeg)
    if ffprobe:
        streams = probe_streams_json(video, ffprobe)
        if streams:
            return streams

    proc = run([ffmpeg, "-hide_banner", "-i", str(video)], check=False, timeout=90)
    streams: list[StreamInfo] = []
    current: StreamInfo | None = None
    stream_re = re.compile(r"^\s*Stream #0:(\d+)(?:\(([^)]*)\))?(?:\[[^\]]+\])?:\s*([^:]+):\s*([^,\n]+)")
    metadata_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$")
    in_metadata = False
    for line in proc.stdout.splitlines():
        match = stream_re.match(line)
        if match:
            if current:
                streams.append(current)
            current = StreamInfo(
                index=int(match.group(1)),
                lang=(match.group(2) or "").lower(),
                kind=match.group(3).strip(),
                codec=match.group(4).strip(),
                metadata={},
                disposition={
                    "default": int("(default)" in line.casefold()),
                    "forced": int("(forced)" in line.casefold()),
                    "hearing_impaired": int(
                        "(hearing impaired)" in line.casefold()
                        or "(hearing_impaired)" in line.casefold()
                    ),
                },
            )
            in_metadata = False
            continue
        if current and "Metadata:" in line:
            in_metadata = True
            continue
        if current and in_metadata:
            meta = metadata_re.match(line)
            if meta:
                key = meta.group(1).strip()
                value = meta.group(2).strip()
                current.metadata = current.metadata or {}
                current.metadata[key] = value
                if key.lower() == "title":
                    current.title = value
            elif line.startswith("  Stream #"):
                in_metadata = False
    if current:
        streams.append(current)
    return streams


def stream_score(stream: StreamInfo) -> int:
    title = stream.title.lower()
    score = 0
    if stream.is_forced or "forced" in title:
        score -= 10_000
    if "commentary" in title or "director comment" in title:
        score -= 20_000
    if "full" in title:
        score += 2_000
    if stream.is_hearing_impaired or "sdh" in title:
        score += 500
    if stream.is_default:
        score += 100
    if stream.metadata:
        frames = stream.metadata.get("NUMBER_OF_FRAMES") or stream.metadata.get("number_of_frames")
        bps = stream.metadata.get("BPS") or stream.metadata.get("bps")
        if frames and frames.isdigit():
            score += min(int(frames), 10_000)
        if bps and bps.isdigit():
            score += min(int(bps) // 100, 1_000)
    return score


def classify_chinese(stream: StreamInfo) -> str:
    label = f"{stream.lang} {stream.title}".lower()
    if "cantonese" in label or "yue" in label:
        return "zh-yue"
    if "simplified" in label or "chs" in label or "简" in label:
        return "zh-Hans"
    if "traditional" in label or "cht" in label or "繁" in label:
        return "zh-Hant"
    if stream.lang in {"chs", "zh-hans"}:
        return "zh-Hans"
    if stream.lang in {"cht", "zh-hant"}:
        return "zh-Hant"
    if stream.lang in {"chi", "zho", "zh", "cmn"}:
        return "zh"
    return ""


def choose_streams(streams: list[StreamInfo]) -> tuple[StreamInfo, StreamInfo | None, str | None]:
    subs = [s for s in streams if s.is_subtitle]
    english = [
        s
        for s in subs
        if s.lang in {"eng", "en"} or "english" in s.title.lower()
    ]
    if not english:
        raise RuntimeError("No English subtitle stream was found.")
    english.sort(key=stream_score, reverse=True)
    en_stream = english[0]

    chinese: list[tuple[int, int, str, StreamInfo]] = []
    for s in subs:
        zh_kind = classify_chinese(s)
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
        chinese.append((rank, stream_score(s), zh_kind, s))
    if not chinese:
        return en_stream, None, None
    chinese.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, zh_kind, zh_stream = chinese[0]
    return en_stream, zh_stream, zh_kind


def extract_subtitle(video: Path, stream_index: int, ffmpeg: str, out_path: Path, codec: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if codec == "copy":
        cmd = [ffmpeg, "-y", "-i", str(video), "-map", f"0:{stream_index}", "-c:s", "copy", str(out_path)]
    else:
        cmd = [ffmpeg, "-y", "-i", str(video), "-map", f"0:{stream_index}", "-c:s", codec, str(out_path)]
    run(cmd, check=True)


def seconds_to_srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def srt_time_to_seconds(value: str) -> float:
    match = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)", value)
    if not match:
        raise ValueError(f"Invalid subtitle time: {value!r}")
    h, m, s, frac = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")[:3]) / 1000


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?(?:i|b|u|s|font|span|c|v|ruby|rt)(?:[.\s][^>]*)?>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\{\\.*?\}", "", text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def parse_srt(path: Path) -> list[SubtitleEvent]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n"))
    events: list[SubtitleEvent] = []
    time_re = re.compile(
        r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)"
    )
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_idx = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if time_idx < 0:
            continue
        match = time_re.search(lines[time_idx])
        if not match:
            continue
        nums = [int(x) for x in match.groups()]
        start = nums[0] * 3600 + nums[1] * 60 + nums[2] + nums[3] / 1000
        end = nums[4] * 3600 + nums[5] * 60 + nums[6] + nums[7] / 1000
        text = clean_text("\n".join(lines[time_idx + 1 :]))
        if text:
            events.append(SubtitleEvent(start, end, text))
    return events


def write_srt(events: list[SubtitleEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for idx, event in enumerate(events, 1):
            f.write(f"{idx}\n")
            f.write(f"{seconds_to_srt_time(event.start)} --> {seconds_to_srt_time(event.end)}\n")
            f.write(event.text.strip() + "\n\n")


def convert_traditional_to_simplified(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    try:
        from opencc import OpenCC
    except Exception as exc:
        raise RuntimeError("opencc is required for Traditional Chinese conversion.") from exc
    converter = OpenCC("t2s")
    return [
        SubtitleEvent(
            e.start,
            e.end,
            converter.convert(e.text),
            e.confidence,
            e.metadata,
        )
        for e in events
    ]


def ycbcr_to_rgba(y: int, cr: int, cb: int, alpha: int) -> tuple[int, int, int, int]:
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return (
        max(0, min(255, int(round(r)))),
        max(0, min(255, int(round(g)))),
        max(0, min(255, int(round(b)))),
        max(0, min(255, alpha)),
    )


def parse_palette(payload: bytes) -> tuple[int, dict[int, tuple[int, int, int, int]]]:
    if len(payload) < 2:
        return 0, {}
    palette_id = payload[0]
    entries: dict[int, tuple[int, int, int, int]] = {}
    pos = 2
    while pos + 5 <= len(payload):
        entry_id = payload[pos]
        y = payload[pos + 1]
        cr = payload[pos + 2]
        cb = payload[pos + 3]
        alpha = payload[pos + 4]
        entries[entry_id] = ycbcr_to_rgba(y, cr, cb, alpha)
        pos += 5
    return palette_id, entries


def parse_composition(payload: bytes, pts: float) -> PgsComposition | None:
    if len(payload) < 11:
        return None
    width = int.from_bytes(payload[0:2], "big")
    height = int.from_bytes(payload[2:4], "big")
    palette_id = payload[9]
    object_count = payload[10]
    pos = 11
    objects: list[PgsCompositionObject] = []
    for _ in range(object_count):
        if pos + 8 > len(payload):
            break
        object_id = int.from_bytes(payload[pos : pos + 2], "big")
        flags = payload[pos + 3]
        x = int.from_bytes(payload[pos + 4 : pos + 6], "big")
        y = int.from_bytes(payload[pos + 6 : pos + 8], "big")
        pos += 8
        if flags & 0x40 and pos + 8 <= len(payload):
            pos += 8
        objects.append(PgsCompositionObject(object_id, x, y))
    return PgsComposition(pts, width, height, palette_id, objects)


def parse_object_segment(
    payload: bytes,
    pending: dict[tuple[int, int], dict[str, Any]],
    objects: dict[int, PgsObject],
) -> None:
    if len(payload) < 4:
        return
    object_id = int.from_bytes(payload[0:2], "big")
    version = payload[2]
    sequence = payload[3]
    key = (object_id, version)
    if sequence & 0x80:
        if len(payload) < 11:
            return
        data_len = int.from_bytes(payload[4:7], "big")
        width = int.from_bytes(payload[7:9], "big")
        height = int.from_bytes(payload[9:11], "big")
        pending[key] = {
            "width": width,
            "height": height,
            "expected": max(0, data_len - 4),
            "data": bytearray(payload[11:]),
        }
    else:
        if key not in pending:
            matches = [k for k in pending if k[0] == object_id]
            if not matches:
                return
            key = matches[-1]
        pending[key]["data"].extend(payload[4:])
    if sequence & 0x40 and key in pending:
        entry = pending.pop(key)
        objects[object_id] = PgsObject(
            width=int(entry["width"]),
            height=int(entry["height"]),
            rle=bytes(entry["data"]),
        )


def decode_rle_to_image(obj: PgsObject, palette: dict[int, tuple[int, int, int, int]]) -> Image.Image:
    width, height = obj.width, obj.height
    rgba = bytearray(width * height * 4)
    x = 0
    y = 0
    pos = 0

    def put(color_index: int, count: int) -> None:
        nonlocal x, y
        color = palette.get(color_index, (0, 0, 0, 0))
        for _ in range(count):
            if y >= height:
                return
            offset = (y * width + x) * 4
            rgba[offset : offset + 4] = bytes(color)
            x += 1
            if x >= width:
                x = 0
                y += 1

    data = obj.rle
    while pos < len(data) and y < height:
        value = data[pos]
        pos += 1
        if value:
            put(value, 1)
            continue
        if pos >= len(data):
            break
        command = data[pos]
        pos += 1
        if command == 0:
            x = 0
            y += 1
        elif command < 0x40:
            put(0, command)
        elif command < 0x80:
            if pos >= len(data):
                break
            count = ((command & 0x3F) << 8) | data[pos]
            pos += 1
            put(0, count)
        elif command < 0xC0:
            if pos >= len(data):
                break
            count = command & 0x3F
            color = data[pos]
            pos += 1
            put(color, count)
        else:
            if pos + 1 >= len(data):
                break
            count = ((command & 0x3F) << 8) | data[pos]
            color = data[pos + 1]
            pos += 2
            put(color, count)
    return Image.frombytes("RGBA", (width, height), bytes(rgba))


def expand_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int], pad: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def render_composition(
    comp: PgsComposition,
    objects: dict[int, PgsObject],
    palette: dict[int, tuple[int, int, int, int]],
    out_path: Path,
    scale: float,
    pad: int,
) -> tuple[Path, str] | None:
    if not comp.objects:
        return None
    canvas = Image.new("RGBA", (comp.width, comp.height), (0, 0, 0, 0))
    for item in comp.objects:
        obj = objects.get(item.object_id)
        if not obj:
            continue
        obj_img = decode_rle_to_image(obj, palette)
        canvas.paste(obj_img, (item.x, item.y), obj_img)
    bbox = canvas.getbbox()
    if not bbox:
        return None
    bbox = expand_bbox(bbox, canvas.size, pad)
    crop = canvas.crop(bbox)
    background = Image.new("RGBA", crop.size, (0, 0, 0, 255))
    background.alpha_composite(crop)
    if scale and not math.isclose(scale, 1.0):
        new_size = (
            max(1, int(round(background.width * scale))),
            max(1, int(round(background.height * scale))),
        )
        background = background.resize(new_size, Image.Resampling.LANCZOS)
    background = background.convert("RGB").filter(ImageFilter.MaxFilter(3))
    digest = hashlib.sha1(background.tobytes()).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(out_path)
    return out_path, digest


def parse_pgs_to_images(
    sup_path: Path,
    image_dir: Path,
    scale: float,
    pad: int,
    limit: int = 0,
) -> list[PgsImageEvent]:
    try:
        events = parse_pgs_to_images_with_pgsrip(sup_path, image_dir, scale=scale, pad=pad, limit=limit)
        if events:
            return events
    except Exception as exc:
        log(f"pgsrip PGS parser failed, falling back to internal parser: {exc}")

    image_dir.mkdir(parents=True, exist_ok=True)
    palettes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    active_palette: dict[int, tuple[int, int, int, int]] = {}
    objects: dict[int, PgsObject] = {}
    pending_objects: dict[tuple[int, int], dict[str, Any]] = {}
    current_comp: PgsComposition | None = None
    current_event: PgsImageEvent | None = None
    events: list[PgsImageEvent] = []
    image_index = 0

    def close_event(end_time: float) -> None:
        nonlocal current_event
        if current_event and end_time > current_event.start:
            current_event.end = end_time
            events.append(current_event)
        current_event = None

    with sup_path.open("rb") as f:
        while True:
            signature = f.read(2)
            if not signature:
                break
            if signature != b"PG":
                raise RuntimeError(f"Invalid PGS signature at byte {f.tell() - 2}.")
            pts = int.from_bytes(f.read(4), "big") / 90_000.0
            f.read(4)  # DTS
            segment_type_raw = f.read(1)
            if not segment_type_raw:
                break
            segment_type = segment_type_raw[0]
            length = int.from_bytes(f.read(2), "big")
            payload = f.read(length)

            if segment_type == 0x14:
                palette_id, palette = parse_palette(payload)
                if palette:
                    palettes[palette_id] = palette
                    active_palette = palette
            elif segment_type == 0x15:
                parse_object_segment(payload, pending_objects, objects)
            elif segment_type == 0x16:
                current_comp = parse_composition(payload, pts)
            elif segment_type == 0x80 and current_comp:
                if not current_comp.objects:
                    close_event(current_comp.pts)
                    current_comp = None
                    continue
                palette = palettes.get(current_comp.palette_id) or active_palette
                image_path = image_dir / f"{image_index:06d}_{current_comp.pts:.3f}.png"
                rendered = render_composition(current_comp, objects, palette, image_path, scale, pad)
                if rendered:
                    image_path, digest = rendered
                    if current_event and current_event.image_hash == digest:
                        current_comp = None
                        continue
                    close_event(current_comp.pts)
                    current_event = PgsImageEvent(current_comp.pts, current_comp.pts + 3.0, image_path, digest)
                    image_index += 1
                    if limit and image_index >= limit:
                        break
                else:
                    close_event(current_comp.pts)
                current_comp = None
    if current_event:
        current_event.end = current_event.start + 3.0
        events.append(current_event)
    return events


def parse_pgs_to_images_with_pgsrip(
    sup_path: Path,
    image_dir: Path,
    scale: float,
    pad: int,
    limit: int = 0,
) -> list[PgsImageEvent]:
    from pgsrip.media import Pgs
    from pgsrip.media_path import MediaPath
    from pgsrip.options import Options

    image_dir.mkdir(parents=True, exist_ok=True)
    data = sup_path.read_bytes()
    pgs = Pgs(MediaPath(str(sup_path)), options=Options(), data_reader=lambda: data, temp_folder=str(image_dir))
    events: list[PgsImageEvent] = []
    for idx, item in enumerate(pgs.items):
        if limit and idx >= limit:
            break
        if item.image is None:
            continue
        arr = item.image.data
        img = Image.fromarray(arr).convert("RGB")
        if pad > 0:
            padded = Image.new("RGB", (img.width + pad * 2, img.height + pad * 2), "white")
            padded.paste(img, (pad, pad))
            img = padded
        if scale and not math.isclose(scale, 1.0):
            new_size = (
                max(1, int(round(img.width * scale))),
                max(1, int(round(img.height * scale))),
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        digest = hashlib.sha1(img.tobytes()).hexdigest()
        path = image_dir / f"{idx:06d}_{srt_time_to_seconds(str(item.start)):.3f}.png"
        img.save(path)
        events.append(
            PgsImageEvent(
                start=srt_time_to_seconds(str(item.start)),
                end=srt_time_to_seconds(str(item.end)),
                image_path=path,
                image_hash=digest,
            )
        )
    return events


class PaddleOcrEngine:
    def __init__(self, lang: str, device: str, prefer_accuracy: bool = True) -> None:
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        configure_windows_cuda_dll_paths()
        suppress_paddlex_optional_langchain_imports()
        try:
            import paddle
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR could not be imported. Run setup_gpu_ocr.ps1 first, or check Python package conflicts."
            ) from exc
        self.lang = lang
        if device.startswith("cuda"):
            device = device.replace("cuda", "gpu")
        self.device = device
        self.paddle = paddle
        try:
            paddle.set_device(device)
        except Exception:
            pass
        kwargs = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if prefer_accuracy:
            kwargs["ocr_version"] = "PP-OCRv5"
            if lang == "chinese_cht":
                kwargs["text_detection_model_name"] = "PP-OCRv5_server_det"
                kwargs["text_recognition_model_name"] = "chinese_cht_PP-OCRv3_mobile_rec"
        try:
            self.ocr = PaddleOCR(device=device, **kwargs)
        except TypeError:
            kwargs.pop("ocr_version", None)
            try:
                self.ocr = PaddleOCR(device=device, **kwargs)
            except TypeError:
                self.ocr = PaddleOCR(
                    lang=lang,
                    use_gpu=device.startswith("gpu"),
                    use_angle_cls=False,
                    show_log=False,
                )

    def assert_gpu(self) -> None:
        try:
            device = self.paddle.get_device()
        except Exception:
            device = ""
        if self.device.startswith("gpu") and "gpu" not in str(device).lower():
            raise RuntimeError(f"Paddle is not using GPU. Current device: {device!r}")

    def recognize(self, image_path: Path) -> OcrObservation:
        if hasattr(self.ocr, "predict"):
            result = self.ocr.predict(str(image_path))
        else:
            result = self.ocr.ocr(str(image_path), cls=False)
        return extract_ocr_observation(result, self.lang)


def extract_ocr_observation(result: Any, lang: str) -> OcrObservation:
    lines: list[tuple[float, float, str, float | None]] = []

    def add_line(box: Any, text: str, confidence: Any = None) -> None:
        if not text:
            return
        try:
            ys = [float(p[1]) for p in box]
            xs = [float(p[0]) for p in box]
            y = sum(ys) / len(ys)
            x = sum(xs) / len(xs)
        except Exception:
            y = float(len(lines))
            x = 0.0
        try:
            normalized_confidence = float(confidence)
            if not math.isfinite(normalized_confidence):
                normalized_confidence = None
            elif normalized_confidence < 0 or normalized_confidence > 1:
                normalized_confidence = None
        except (TypeError, ValueError):
            normalized_confidence = None
        lines.append((y, x, clean_text(str(text)), normalized_confidence))

    def walk(obj: Any) -> None:
        if obj is None:
            return
        if hasattr(obj, "json") and not isinstance(obj, (dict, list, tuple, str)):
            try:
                payload = obj.json
                if callable(payload):
                    payload = payload()
                walk(payload)
                return
            except Exception:
                pass
        if isinstance(obj, dict):
            if "rec_texts" in obj:
                texts = obj.get("rec_texts")
                if texts is None:
                    texts = []
                scores = obj.get("rec_scores")
                if scores is None:
                    scores = []
                boxes = obj.get("rec_boxes")
                if boxes is None or (hasattr(boxes, "__len__") and len(boxes) == 0):
                    boxes = obj.get("rec_polys")
                if boxes is None:
                    boxes = []
                for idx, text in enumerate(texts):
                    box = boxes[idx] if idx < len(boxes) else None
                    if box is not None and hasattr(box, "tolist"):
                        box = box.tolist()
                    score = scores[idx] if idx < len(scores) else None
                    add_line(box, text, score)
                return
            if "res" in obj:
                walk(obj["res"])
                return
            for value in obj.values():
                walk(value)
            return
        if isinstance(obj, str):
            return
        if isinstance(obj, (list, tuple)):
            if len(obj) == 2 and isinstance(obj[1], (list, tuple)) and len(obj[1]) >= 1:
                text = obj[1][0]
                confidence = obj[1][1] if len(obj[1]) > 1 else None
                add_line(obj[0], text, confidence)
                return
            for item in obj:
                walk(item)

    walk(result)
    if not lines:
        return OcrObservation(text="")
    lines.sort(key=lambda item: (round(item[0] / 12), item[1]))
    texts = [item[2] for item in lines if item[2]]
    confidences = [item[3] for item in lines if item[2] and item[3] is not None]
    if lang in {"ch", "chinese_cht"}:
        text = clean_text("".join(texts))
    else:
        text = clean_text(" ".join(texts))
    return OcrObservation(
        text=text,
        confidence=round(sum(confidences) / len(confidences), 6) if confidences else None,
        line_confidences=[round(value, 6) for value in confidences] or None,
    )


def extract_ocr_text(result: Any, lang: str) -> str:
    return extract_ocr_observation(result, lang).text


def ocr_pgs_events(
    image_events: list[PgsImageEvent],
    lang: str,
    device: str,
    cache_path: Path,
    prefer_accuracy: bool,
) -> list[SubtitleEvent]:
    config = {
        "lang": lang,
        "device": device,
        "prefer_accuracy": prefer_accuracy,
    }
    image_signatures = [
        {
            "start": item.start,
            "end": item.end,
            "image_hash": item.image_hash,
        }
        for item in image_events
    ]
    out: list[SubtitleEvent] = []
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("version") in COMPATIBLE_OCR_CACHE_VERSIONS
            and cached.get("config") == config
            and cached.get("images") == image_signatures
            and isinstance(cached.get("events"), list)
            and len(cached["events"]) <= len(image_events)
        ):
            try:
                out = [
                    SubtitleEvent(
                        float(item["start"]),
                        float(item["end"]),
                        str(item["text"]),
                        (
                            float(item["confidence"])
                            if item.get("confidence") is not None
                            else None
                        ),
                        item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
                    )
                    for item in cached["events"]
                ]
            except (KeyError, TypeError, ValueError):
                out = []
            if len(out) == len(image_events):
                log(f"Using validated OCR cache: {cache_path.name} ({len(out)} events)")
                validate_ocr_results(out)
                return out
            if out:
                log(f"Resuming OCR cache at image {len(out) + 1}/{len(image_events)}")

    if len(out) >= len(image_events):
        return out
    engine = PaddleOcrEngine(lang=lang, device=device, prefer_accuracy=prefer_accuracy)
    engine.assert_gpu()
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    iterator: Iterable[PgsImageEvent]
    remaining = image_events[len(out) :]
    iterator = tqdm(remaining, desc=f"OCR {lang}", unit="line") if tqdm else remaining
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(iterator, len(out) + 1):
        observation = engine.recognize(item.image_path)
        if isinstance(observation, OcrObservation):
            event = SubtitleEvent(
                item.start,
                item.end,
                observation.text,
                observation.confidence,
                (
                    {"line_confidences": observation.line_confidences}
                    if observation.line_confidences
                    else None
                ),
            )
        else:
            event = SubtitleEvent(item.start, item.end, str(observation))
        out.append(event)
        if idx % 20 == 0:
            write_json_atomic(
                cache_path,
                {
                    "version": OCR_CACHE_VERSION,
                    "config": config,
                    "images": image_signatures,
                    "events": [event.__dict__ for event in out],
                },
            )
    write_json_atomic(
        cache_path,
        {
            "version": OCR_CACHE_VERSION,
            "config": config,
            "images": image_signatures,
            "events": [event.__dict__ for event in out],
        },
    )
    validate_ocr_results(out)
    return out


def validate_ocr_results(events: list[SubtitleEvent]) -> None:
    recognized_count = sum(1 for event in events if clean_text(event.text))
    log(f"OCR recognized text in {recognized_count}/{len(events)} images")
    if events and recognized_count == 0:
        raise RuntimeError(
            f"OCR produced no text for {len(events)} images. Check the OCR language and source subtitle track."
        )


def file_cache_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def extracted_stream_cache_matches(
    meta_path: Path,
    video: Path,
    stream_index: int,
    codec: str,
) -> bool:
    if not meta_path.exists():
        return False
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return payload == {
            "version": EXTRACT_CACHE_VERSION,
            "video": file_cache_identity(video),
            "stream_index": stream_index,
            "codec": codec,
        }
    except (OSError, ValueError, TypeError):
        return False


def ensure_extracted_subtitle(
    video: Path,
    stream_index: int,
    ffmpeg: str,
    out_path: Path,
    codec: str,
) -> None:
    meta_path = out_path.with_name(f"{out_path.name}.meta.json")
    if out_path.exists() and extracted_stream_cache_matches(meta_path, video, stream_index, codec):
        return
    extract_subtitle(video, stream_index, ffmpeg, out_path, codec)
    write_json_atomic(
        meta_path,
        {
            "version": EXTRACT_CACHE_VERSION,
            "video": file_cache_identity(video),
            "stream_index": stream_index,
            "codec": codec,
        },
    )


def read_or_extract_text_events(
    video: Path,
    stream: StreamInfo,
    ffmpeg: str,
    work_dir: Path,
) -> list[SubtitleEvent]:
    srt_path = work_dir / f"stream_{stream.index:02d}.srt"
    ensure_extracted_subtitle(video, stream.index, ffmpeg, srt_path, "srt")
    return parse_srt(srt_path)


def read_or_ocr_pgs_events(
    video: Path,
    stream: StreamInfo,
    ffmpeg: str,
    work_dir: Path,
    ocr_lang: str,
    device: str,
    scale: float,
    pad: int,
    limit: int,
    prefer_accuracy: bool,
) -> list[SubtitleEvent]:
    sup_path = work_dir / f"stream_{stream.index:02d}.sup"
    cache_suffix = f"_limit{limit}" if limit else ""
    image_dir = work_dir / f"stream_{stream.index:02d}_images{cache_suffix}"
    image_events_path = work_dir / f"stream_{stream.index:02d}_image_events{cache_suffix}.json"
    ocr_cache_path = work_dir / f"stream_{stream.index:02d}_{ocr_lang}_ocr{cache_suffix}.json"
    sup_was_cached = sup_path.exists() and extracted_stream_cache_matches(
        sup_path.with_name(f"{sup_path.name}.meta.json"),
        video,
        stream.index,
        "copy",
    )
    if not sup_was_cached:
        log(f"Extracting PGS stream 0:{stream.index} -> {sup_path}")
    ensure_extracted_subtitle(video, stream.index, ffmpeg, sup_path, "copy")
    render_config = {
        "sup": file_cache_identity(sup_path),
        "scale": scale,
        "pad": pad,
        "limit": limit,
    }
    image_events: list[PgsImageEvent] | None = None
    if image_events_path.exists():
        try:
            raw = json.loads(image_events_path.read_text(encoding="utf-8"))
            if (
                isinstance(raw, dict)
                and raw.get("version") == PGS_IMAGE_CACHE_VERSION
                and raw.get("render") == render_config
                and isinstance(raw.get("events"), list)
            ):
                cached_events = [
                    PgsImageEvent(
                        float(item["start"]),
                        float(item["end"]),
                        Path(item["image_path"]),
                        str(item["image_hash"]),
                    )
                    for item in raw["events"]
                ]
                if all(item.image_path.exists() for item in cached_events):
                    image_events = cached_events
        except (KeyError, OSError, TypeError, ValueError):
            image_events = None
    if image_events is None:
        log(f"Rendering PGS images from {sup_path.name}")
        image_events = parse_pgs_to_images(sup_path, image_dir, scale=scale, pad=pad, limit=limit)
        write_json_atomic(
            image_events_path,
            {
                "version": PGS_IMAGE_CACHE_VERSION,
                "render": render_config,
                "events": [
                    {
                        "start": event.start,
                        "end": event.end,
                        "image_path": str(event.image_path),
                        "image_hash": event.image_hash,
                    }
                    for event in image_events
                ],
            },
        )
    log(f"OCR source images: {len(image_events)}")
    return ocr_pgs_events(image_events, ocr_lang, device, ocr_cache_path, prefer_accuracy)


def get_stream_events(
    video: Path,
    stream: StreamInfo,
    ffmpeg: str,
    work_dir: Path,
    ocr_lang: str,
    args: argparse.Namespace,
) -> list[SubtitleEvent]:
    if stream.is_pgs:
        return read_or_ocr_pgs_events(
            video,
            stream,
            ffmpeg,
            work_dir,
            ocr_lang=ocr_lang,
            device=args.device,
            scale=args.ocr_scale,
            pad=args.crop_pad,
            limit=args.limit,
            prefer_accuracy=not args.fast_ocr,
        )
    if stream.is_text_subtitle:
        return read_or_extract_text_events(video, stream, ffmpeg, work_dir)
    raise RuntimeError(f"Unsupported subtitle codec for stream 0:{stream.index}: {stream.codec}")


def is_sdh_sound_cue(text: str) -> bool:
    raw = clean_text(text)
    if not raw:
        return False
    t = raw.upper()
    cue_words = {
        "MUSIC",
        "PLAYING",
        "LAUGH",
        "LAUGHS",
        "LAUGHING",
        "SIGHS",
        "GROANS",
        "SCREAM",
        "SCREAMING",
        "CHEERING",
        "APPLAUSE",
        "BEEP",
        "BEEPING",
        "RINGING",
        "WHIRRING",
        "EXPLOSION",
        "THUNDER",
        "GUNSHOT",
    }
    has_cue_word = any(word in t for word in cue_words)
    letters = [c for c in raw if c.isalpha()]
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
    return has_cue_word and upper_ratio > 0.85


def sdh_cue_categories(text: str) -> set[str]:
    raw = clean_text(text)
    if not raw:
        return set()
    upper = raw.upper()
    categories: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "music": (
            ("MUSIC", "SONG", "SINGING"),
            ("音乐", "歌声", "演奏", "唱歌"),
        ),
        "laughter": (
            ("LAUGH", "LAUGHS", "LAUGHING"),
            ("笑声", "大笑", "发笑"),
        ),
        "sigh": (("SIGHS", "SIGHING"), ("叹气", "叹息")),
        "groan": (("GROANS", "GROANING"), ("呻吟",)),
        "scream": (("SCREAM", "SCREAMING"), ("尖叫",)),
        "cheering": (("CHEERING",), ("欢呼",)),
        "applause": (("APPLAUSE", "CLAPPING"), ("掌声", "鼓掌")),
        "beep": (("BEEP", "BEEPING"), ("哔", "蜂鸣")),
        "ringing": (("RINGING",), ("铃声", "电话响")),
        "mechanical": (("WHIRRING",), ("嗡嗡", "机械声")),
        "explosion": (("EXPLOSION",), ("爆炸",)),
        "thunder": (("THUNDER",), ("雷声",)),
        "gunshot": (("GUNSHOT", "GUNFIRE"), ("枪声", "枪响")),
    }
    has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", raw))
    if has_cjk:
        compact = re.sub(r"\s+", "", raw)
        bracketed = bool(
            re.match(r"^[\[(（【<].*[\])）】>]$", compact)
        )
        cue_signals = (
            "响起",
            "播放",
            "歌声",
            "演奏",
            "唱歌",
            "笑声",
            "大笑",
            "发笑",
            "叹气",
            "叹息",
            "呻吟",
            "尖叫",
            "欢呼",
            "掌声",
            "鼓掌",
            "哔",
            "蜂鸣",
            "铃声",
            "电话响",
            "嗡嗡",
            "机械声",
            "爆炸",
            "雷声",
            "枪声",
            "枪响",
        )
        if not bracketed and not (
            len(compact) <= 12
            and any(signal in compact for signal in cue_signals)
        ):
            return set()
    elif not is_sdh_sound_cue(raw):
        return set()

    result: set[str] = set()
    for category, (english_tokens, chinese_tokens) in categories.items():
        if any(token in upper for token in english_tokens) or any(
            token in raw
            for token in chinese_tokens
        ):
            result.add(category)
    return result


def best_overlap(base: SubtitleEvent, candidates: list[tuple[int, SubtitleEvent]]) -> tuple[int, SubtitleEvent] | None:
    best: tuple[float, float, SubtitleEvent] | None = None
    best_idx = -1
    base_center = (base.start + base.end) / 2
    for idx, cand in candidates:
        overlap = max(0.0, min(base.end, cand.end) - max(base.start, cand.start))
        cand_center = (cand.start + cand.end) / 2
        distance = abs(base_center - cand_center)
        if overlap <= 0:
            gap = max(base.start - cand.end, cand.start - base.end)
            if gap > 0.75:
                continue
            score = -gap * 10 - distance
        else:
            score = overlap * 10 - distance
        if best is None or score > best[0]:
            best = (score, distance, cand)
            best_idx = idx
    if not best:
        return None
    _, _, cand = best
    return best_idx, cand


def strong_overlap(first: SubtitleEvent, second: SubtitleEvent) -> bool:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    shorter_duration = min(
        max(0.01, first.end - first.start),
        max(0.01, second.end - second.start),
    )
    return overlap / shorter_duration >= 0.5


def is_subtitle_credit_text(text: str) -> bool:
    raw = clean_text(text)
    if not raw or len(raw) > 96:
        return False
    compact = re.sub(r"\s+", "", raw)
    if re.search(
        r"(?:\u5b57\u5e55(?:\u5236\u4f5c|\u7ffb\u8bd1|\u6821\u5bf9|\u65f6\u95f4\u8f74|\u538b\u5236|\u542c\u8bd1|\u7ec4)|"
        r"(?:\u7ffb\u8bd1|\u6821\u5bf9|\u65f6\u95f4\u8f74|\u538b\u5236|\u542c\u8bd1)[\uff1a:])",
        compact,
    ):
        return True
    lowered = raw.casefold()
    return bool(
        re.search(
            r"\b(?:subtitles?|captions?)\s+(?:by|translated|synced|edited)\b",
            lowered,
        )
        or re.search(
            r"\b(?:translated|captioned|subtitled|synced)\s+by\b",
            lowered,
        )
    )


def join_subtitle_events(events: list[SubtitleEvent], language: str) -> SubtitleEvent:
    ordered = sorted(events, key=lambda event: (event.start, event.end))
    texts: list[str] = []
    for event in ordered:
        text = clean_text(event.text)
        normalized = re.sub(r"\s+", "", text).casefold()
        if not text:
            continue
        if not texts:
            texts.append(text)
            continue
        previous_normalized = re.sub(r"\s+", "", texts[-1]).casefold()
        if normalized == previous_normalized or normalized in previous_normalized:
            continue
        if previous_normalized in normalized:
            texts[-1] = text
            continue
        texts.append(text)
    separator = "" if language == "zh" else " "
    confidences = [event.confidence for event in ordered if event.confidence is not None]
    event_metadata = [
        event.metadata
        for event in ordered
        if isinstance(event.metadata, dict)
    ]
    metadata: dict[str, Any] = {}
    source_assets: dict[str, dict[str, Any]] = {}
    for item in event_metadata:
        for asset in item.get("source_assets") or []:
            if isinstance(asset, dict) and asset.get("asset_id"):
                source_assets[str(asset["asset_id"])] = dict(asset)
    if source_assets:
        metadata["source_assets"] = list(source_assets.values())
    authority_field = (
        "chinese_text_authority"
        if language == "zh"
        else "source_text_authority"
    )
    authorities = [
        str(
            item.get(authority_field)
            or item.get("source_text_authority")
            or item.get("chinese_text_authority")
            or ""
        )
        for item in event_metadata
        if (
            item.get(authority_field)
            or item.get("source_text_authority")
            or item.get("chinese_text_authority")
        )
    ]
    if authorities:
        authority_rank = {"asr": 0, "ocr": 1, "unknown": 2, "authored": 3}
        metadata[authority_field] = min(
            authorities,
            key=lambda value: authority_rank.get(value, 2),
        )
    for field_name in (
        "source_asset_id",
        "source_origin",
        "source_representation",
        "source_role",
        "source_timing_authority",
    ):
        values = {
            str(item.get(field_name))
            for item in event_metadata
            if item.get(field_name) is not None
        }
        if len(values) == 1:
            metadata[field_name] = values.pop()
    if any(bool(item.get("supplementary_source")) for item in event_metadata):
        metadata["supplementary_source"] = True
    joined_text = clean_text(separator.join(texts))
    if language == "zh":
        seen_labels: set[str] = set()

        def keep_first_label(match: re.Match[str]) -> str:
            label = re.sub(r"\s+", " ", match.group(0)).strip()
            key = re.sub(r"\s+", "", label).casefold()
            if key in seen_labels:
                return ""
            seen_labels.add(key)
            return label

        joined_text = clean_text(
            re.sub(r"\[[^\]]{1,160}\]", keep_first_label, joined_text)
        )
    return SubtitleEvent(
        min(event.start for event in ordered),
        max(event.end for event in ordered),
        joined_text,
        min(confidences) if confidences else None,
        metadata or None,
    )


def event_text_authority(event: SubtitleEvent) -> str:
    if not isinstance(event.metadata, dict):
        return "unknown"
    return str(
        event.metadata.get("source_text_authority")
        or event.metadata.get("chinese_text_authority")
        or "unknown"
    )


def collapse_rolling_authored_events(
    events: list[SubtitleEvent],
    *,
    boundary_tolerance: float = 0.08,
) -> list[SubtitleEvent]:
    """Collapse cumulative subtitle frames into one appearance per text unit."""
    output: list[SubtitleEvent] = []
    recent_units: dict[str, tuple[int, float]] = {}
    recent_labels: dict[str, tuple[int, float]] = {}
    label_pattern = re.compile(r"\[[^\]]{1,160}\]", re.DOTALL)

    def compact_key(value: str) -> str:
        return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()

    def normalized_labels(value: str) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for match in label_pattern.finditer(value):
            label = re.sub(r"\s+", " ", match.group(0)).strip()
            key = compact_key(label)
            if key and key not in seen:
                labels.append(label)
                seen.add(key)
        return labels

    for event in sorted(events, key=lambda item: (item.start, item.end)):
        raw_text = clean_text(event.text)
        if not raw_text:
            continue
        labels = normalized_labels(raw_text)
        content_text = clean_text(label_pattern.sub("\n", raw_text))
        units = [
            clean_text(unit)
            for unit in content_text.splitlines()
            if clean_text(unit)
        ]
        duplicate_indices: list[int] = []
        unique_units: list[str] = []
        for unit in units:
            key = compact_key(unit)
            previous = recent_units.get(key)
            if (
                key
                and previous is not None
                and event.start <= previous[1] + boundary_tolerance
            ):
                duplicate_indices.append(previous[0])
                continue
            unique_units.append(unit)

        if units and not unique_units and duplicate_indices:
            target_index = max(duplicate_indices)
            target = output[target_index]
            existing_label_keys = {
                compact_key(label)
                for label in normalized_labels(target.text)
            }
            new_labels = [
                label
                for label in labels
                if compact_key(label) not in existing_label_keys
            ]
            if new_labels:
                target.text = clean_text(
                    "\n".join([*new_labels, target.text])
                )
            target.end = max(target.end, event.end)
            for unit in units:
                key = compact_key(unit)
                if key:
                    recent_units[key] = (target_index, target.end)
            for label in labels:
                key = compact_key(label)
                if key:
                    recent_labels[key] = (target_index, target.end)
            continue

        if not units and labels and not sdh_cue_categories(raw_text):
            candidate_indices = [
                recent_labels[compact_key(label)][0]
                for label in labels
                if compact_key(label) in recent_labels
                and event.start
                <= recent_labels[compact_key(label)][1] + boundary_tolerance
            ]
            if candidate_indices:
                target_index = max(candidate_indices)
                target = output[target_index]
                target.end = max(target.end, event.end)
                for label in labels:
                    key = compact_key(label)
                    if key and key in recent_labels:
                        recent_labels[key] = (target_index, target.end)
                continue

        kept_labels: list[str] = []
        for label in labels:
            key = compact_key(label)
            previous = recent_labels.get(key)
            if (
                key
                and previous is not None
                and event.start <= previous[1] + boundary_tolerance
            ):
                continue
            kept_labels.append(label)

        parts = [*kept_labels, *unique_units]
        if not parts:
            if labels and sdh_cue_categories(raw_text):
                parts = labels
            else:
                continue
        item = SubtitleEvent(
            event.start,
            event.end,
            clean_text("\n".join(parts)),
            event.confidence,
            dict(event.metadata) if isinstance(event.metadata, dict) else event.metadata,
        )
        output_index = len(output)
        output.append(item)
        for unit in unique_units:
            key = compact_key(unit)
            if key:
                recent_units[key] = (output_index, item.end)
        for label in kept_labels:
            key = compact_key(label)
            if key:
                recent_labels[key] = (output_index, item.end)
    return output


def pair_ocr_events_to_authored_anchors(
    ocr_events: list[SubtitleEvent],
    authored_events: list[SubtitleEvent],
) -> list[tuple[SubtitleEvent | None, SubtitleEvent | None]]:
    authored_events = collapse_rolling_authored_events(authored_events)
    pairs: list[tuple[SubtitleEvent | None, SubtitleEvent | None]] = []
    reserved_ocr: set[int] = {
        index
        for index, event in enumerate(ocr_events)
        if is_subtitle_credit_text(event.text)
    }
    reserved_authored: set[int] = {
        index
        for index, event in enumerate(authored_events)
        if is_subtitle_credit_text(event.text)
    }
    cue_candidates: list[tuple[float, float, int, int]] = []
    for ocr_index, ocr_event in enumerate(ocr_events):
        categories = sdh_cue_categories(ocr_event.text)
        if not categories:
            continue
        for authored_index, authored_event in enumerate(authored_events):
            if not (categories & sdh_cue_categories(authored_event.text)):
                continue
            if not strong_overlap(ocr_event, authored_event):
                continue
            overlap = max(
                0.0,
                min(ocr_event.end, authored_event.end)
                - max(ocr_event.start, authored_event.start),
            )
            shorter_duration = min(
                max(0.01, ocr_event.end - ocr_event.start),
                max(0.01, authored_event.end - authored_event.start),
            )
            cue_candidates.append(
                (
                    -(overlap / shorter_duration),
                    -overlap,
                    ocr_index,
                    authored_index,
                )
            )
    for _ratio, _overlap, ocr_index, authored_index in sorted(cue_candidates):
        if ocr_index in reserved_ocr or authored_index in reserved_authored:
            continue
        reserved_ocr.add(ocr_index)
        reserved_authored.add(authored_index)
        pairs.append((ocr_events[ocr_index], authored_events[authored_index]))

    eligible_ocr = [
        index
        for index, event in enumerate(ocr_events)
        if index not in reserved_ocr
        and not is_sdh_sound_cue(event.text)
    ]
    eligible_authored = [
        index
        for index, event in enumerate(authored_events)
        if index not in reserved_authored
        and not is_sdh_sound_cue(event.text)
    ]
    parent = {index: index for index in eligible_ocr}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        members = [
            index
            for index in eligible_ocr
            if find(index) in {first_root, second_root}
        ]
        combined_start = min(ocr_events[index].start for index in members)
        combined_end = max(ocr_events[index].end for index in members)
        if combined_end - combined_start > 20.0:
            return
        parent[second_root] = first_root

    for authored_index in eligible_authored:
        anchor = authored_events[authored_index]
        overlaps: list[tuple[float, int]] = []
        for ocr_index in eligible_ocr:
            event = ocr_events[ocr_index]
            overlap = max(
                0.0,
                min(anchor.end, event.end) - max(anchor.start, event.start),
            )
            if overlap >= 0.2:
                overlaps.append((overlap, ocr_index))
        if len(overlaps) < 2:
            continue
        overlaps.sort(reverse=True)
        strongest = overlaps[0][0]
        selected = sorted(
            index
            for overlap, index in overlaps
            if overlap >= strongest * 0.05
        )
        for first, second in zip(selected, selected[1:]):
            if second != first + 1:
                continue
            union(first, second)

    components_by_root: dict[int, list[int]] = {}
    for index in eligible_ocr:
        components_by_root.setdefault(find(index), []).append(index)
    components = sorted(
        components_by_root.values(),
        key=lambda indices: min(ocr_events[index].start for index in indices),
    )

    authored_by_component: dict[int, list[int]] = {}
    unassigned_authored: set[int] = set(eligible_authored)
    for authored_index in eligible_authored:
        anchor = authored_events[authored_index]
        best: tuple[float, float, int] | None = None
        for component_index, indices in enumerate(components):
            overlap = sum(
                max(
                    0.0,
                    min(anchor.end, ocr_events[index].end)
                    - max(anchor.start, ocr_events[index].start),
                )
                for index in indices
            )
            component_start = min(ocr_events[index].start for index in indices)
            component_end = max(ocr_events[index].end for index in indices)
            gap = max(component_start - anchor.end, anchor.start - component_end, 0.0)
            if overlap <= 0.0 and gap > 0.75:
                continue
            score = overlap * 10.0 - gap
            center_distance = abs(
                (anchor.start + anchor.end) / 2
                - (component_start + component_end) / 2
            )
            candidate = (score, -center_distance, component_index)
            if best is None or candidate > best:
                best = candidate
        if best is not None:
            component_index = best[2]
            authored_by_component.setdefault(component_index, []).append(authored_index)
            unassigned_authored.discard(authored_index)

    for component_index, indices in enumerate(components):
        ocr_event = join_subtitle_events(
            [ocr_events[index] for index in indices],
            "en",
        )
        anchor_indices = authored_by_component.get(component_index, [])
        authored_event = (
            join_subtitle_events(
                [authored_events[index] for index in anchor_indices],
                "zh",
            )
            if anchor_indices
            else None
        )
        pairs.append((ocr_event, authored_event))

    paired_ocr = set(eligible_ocr) | reserved_ocr
    for index, event in enumerate(ocr_events):
        if index not in paired_ocr:
            pairs.append((event, None))
    paired_authored = (set(eligible_authored) - unassigned_authored) | reserved_authored
    for index, event in enumerate(authored_events):
        if index not in paired_authored:
            pairs.append((None, event))
    for index in sorted(reserved_ocr):
        if is_subtitle_credit_text(ocr_events[index].text):
            pairs.append((ocr_events[index], None))
    for index in sorted(reserved_authored):
        if is_subtitle_credit_text(authored_events[index].text):
            pairs.append((None, authored_events[index]))
    pairs.sort(key=lambda item: min(event.start for event in item if event is not None))
    return pairs


def pair_ambiguous_component(
    en_events: list[SubtitleEvent],
    zh_events: list[SubtitleEvent],
    component_en: set[int],
    component_zh: set[int],
    en_edges: dict[int, set[int]],
) -> list[tuple[SubtitleEvent | None, SubtitleEvent | None]]:
    candidates: list[tuple[float, float, float, float, int, int]] = []
    for en_idx in component_en:
        en = en_events[en_idx]
        en_duration = max(0.01, en.end - en.start)
        en_center = (en.start + en.end) / 2
        for zh_idx in en_edges.get(en_idx, set()) & component_zh:
            zh = zh_events[zh_idx]
            overlap = max(0.0, min(en.end, zh.end) - max(en.start, zh.start))
            shorter_duration = min(en_duration, max(0.01, zh.end - zh.start))
            overlap_ratio = overlap / shorter_duration
            center_distance = abs(en_center - (zh.start + zh.end) / 2)
            start_distance = abs(en.start - zh.start)
            candidates.append(
                (
                    -overlap_ratio,
                    -overlap,
                    center_distance,
                    start_distance,
                    en_idx,
                    zh_idx,
                )
            )

    matched_en: set[int] = set()
    matched_zh: set[int] = set()
    pairs: list[tuple[SubtitleEvent | None, SubtitleEvent | None]] = []
    for _ratio, _overlap, _center, _start, en_idx, zh_idx in sorted(candidates):
        if en_idx in matched_en or zh_idx in matched_zh:
            continue
        matched_en.add(en_idx)
        matched_zh.add(zh_idx)
        pairs.append((en_events[en_idx], zh_events[zh_idx]))

    pairs.extend((en_events[idx], None) for idx in sorted(component_en - matched_en))
    pairs.extend((None, zh_events[idx]) for idx in sorted(component_zh - matched_zh))
    pairs.sort(key=lambda item: min(event.start for event in item if event is not None))
    return pairs


def rebalance_fragment_pairs(
    pairs: list[tuple[SubtitleEvent | None, SubtitleEvent | None]],
) -> list[tuple[SubtitleEvent | None, SubtitleEvent | None]]:
    working = sorted(
        pairs,
        key=lambda item: min(event.start for event in item if event is not None),
    )
    while True:
        merged_fragment = False
        for index, (en_event, zh_event) in enumerate(working):
            if en_event is None or zh_event is None:
                continue
            if is_subtitle_credit_text(en_event.text) or is_subtitle_credit_text(zh_event.text):
                continue
            if is_sdh_sound_cue(en_event.text) or is_sdh_sound_cue(zh_event.text):
                continue

            en_duration = max(0.01, en_event.end - en_event.start)
            zh_duration = max(0.01, zh_event.end - zh_event.start)
            if en_duration <= 1.0 and zh_duration >= max(1.5, en_duration * 2.0):
                fragment_lane = 0
                covering_event = zh_event
            elif zh_duration <= 1.0 and en_duration >= max(1.5, zh_duration * 2.0):
                fragment_lane = 1
                covering_event = en_event
            else:
                continue

            candidates: list[tuple[float, int]] = []
            for neighbor_index in (index - 1, index + 1):
                if neighbor_index < 0 or neighbor_index >= len(working):
                    continue
                neighbor = working[neighbor_index]
                lane_event = neighbor[fragment_lane]
                if lane_event is None:
                    continue
                if any(
                    event is not None
                    and (
                        is_subtitle_credit_text(event.text)
                        or is_sdh_sound_cue(event.text)
                    )
                    for event in neighbor
                ):
                    continue
                overlap = max(
                    0.0,
                    min(covering_event.end, lane_event.end)
                    - max(covering_event.start, lane_event.start),
                )
                if overlap > 0.05:
                    candidates.append((overlap, neighbor_index))
            if not candidates:
                continue

            selected = {index}
            for _overlap, neighbor_index in sorted(candidates, reverse=True):
                proposed = selected | {neighbor_index}
                proposed_events = [
                    event
                    for pair_index in proposed
                    for event in working[pair_index]
                    if event is not None
                ]
                span = max(event.end for event in proposed_events) - min(
                    event.start for event in proposed_events
                )
                if span <= 20.0:
                    selected = proposed
            if len(selected) == 1:
                continue

            first = min(selected)
            last = max(selected)
            selected_pairs = working[first : last + 1]
            english_events = [event for event, _ in selected_pairs if event is not None]
            chinese_events = [event for _, event in selected_pairs if event is not None]
            replacement = (
                join_subtitle_events(english_events, "en") if english_events else None,
                join_subtitle_events(chinese_events, "zh") if chinese_events else None,
            )
            working[first : last + 1] = [replacement]
            merged_fragment = True
            break
        if not merged_fragment:
            return working


def pair_events(en_events: list[SubtitleEvent], zh_events: list[SubtitleEvent]) -> list[tuple[SubtitleEvent | None, SubtitleEvent | None]]:
    en_authorities = {
        event_text_authority(event)
        for event in en_events
        if not is_subtitle_credit_text(event.text)
    }
    zh_authorities = {
        event_text_authority(event)
        for event in zh_events
        if not is_subtitle_credit_text(event.text)
    }
    if en_authorities == {"ocr"} and zh_authorities == {"authored"}:
        return rebalance_fragment_pairs(
            pair_ocr_events_to_authored_anchors(en_events, zh_events)
        )

    paired: list[tuple[SubtitleEvent | None, SubtitleEvent | None]] = []
    used_en: set[int] = set()
    used_zh: set[int] = set()
    for en_idx, en in enumerate(en_events):
        if is_subtitle_credit_text(en.text):
            used_en.add(en_idx)
            paired.append((en, None))
    for zh_idx, zh in enumerate(zh_events):
        if is_subtitle_credit_text(zh.text):
            used_zh.add(zh_idx)
            paired.append((None, zh))
    cue_candidates: list[tuple[float, float, float, int, int]] = []
    for en_idx, en in enumerate(en_events):
        en_categories = sdh_cue_categories(en.text)
        if not en_categories:
            continue
        en_center = (en.start + en.end) / 2
        for zh_idx, zh in enumerate(zh_events):
            if not (en_categories & sdh_cue_categories(zh.text)):
                continue
            if not strong_overlap(en, zh):
                continue
            overlap = max(
                0.0,
                min(en.end, zh.end) - max(en.start, zh.start),
            )
            shorter_duration = min(
                max(0.01, en.end - en.start),
                max(0.01, zh.end - zh.start),
            )
            cue_candidates.append(
                (
                    -(overlap / shorter_duration),
                    -overlap,
                    abs(en_center - (zh.start + zh.end) / 2),
                    en_idx,
                    zh_idx,
                )
            )
    for _ratio, _overlap, _center, en_idx, zh_idx in sorted(cue_candidates):
        if en_idx in used_en or zh_idx in used_zh:
            continue
        used_en.add(en_idx)
        used_zh.add(zh_idx)
        paired.append((en_events[en_idx], zh_events[zh_idx]))

    en_edges: dict[int, set[int]] = {}
    zh_edges: dict[int, set[int]] = {}
    active_zh_start = 0
    for en_idx, en in enumerate(en_events):
        if en_idx in used_en:
            continue
        if is_sdh_sound_cue(en.text) or bool(
            (en.metadata or {}).get("supplementary_source")
        ):
            continue
        while active_zh_start < len(zh_events) and zh_events[active_zh_start].end <= en.start:
            active_zh_start += 1
        for zh_idx in range(active_zh_start, len(zh_events)):
            zh = zh_events[zh_idx]
            if zh.start >= en.end:
                break
            if zh_idx in used_zh:
                continue
            if strong_overlap(en, zh):
                en_edges.setdefault(en_idx, set()).add(zh_idx)
                zh_edges.setdefault(zh_idx, set()).add(en_idx)

    for initial_en_idx in sorted(en_edges):
        if initial_en_idx in used_en:
            continue
        component_en: set[int] = set()
        component_zh: set[int] = set()
        pending_en = [initial_en_idx]
        while pending_en:
            en_idx = pending_en.pop()
            if en_idx in component_en:
                continue
            component_en.add(en_idx)
            for zh_idx in en_edges.get(en_idx, set()):
                if zh_idx not in component_zh:
                    component_zh.add(zh_idx)
                    pending_en.extend(zh_edges.get(zh_idx, set()) - component_en)
        used_en.update(component_en)
        used_zh.update(component_zh)
        if len(component_en) > 1 and len(component_zh) > 1:
            paired.extend(
                pair_ambiguous_component(
                    en_events,
                    zh_events,
                    component_en,
                    component_zh,
                    en_edges,
                )
            )
        else:
            paired.append(
                (
                    join_subtitle_events([en_events[idx] for idx in component_en], "en"),
                    join_subtitle_events([zh_events[idx] for idx in component_zh], "zh"),
                )
            )

    start = 0
    for en_idx, en in enumerate(en_events):
        if en_idx in used_en:
            continue
        while start < len(zh_events) and zh_events[start].end < en.start - 5:
            start += 1
        if is_sdh_sound_cue(en.text) or bool(
            (en.metadata or {}).get("supplementary_source")
        ):
            paired.append((en, None))
            continue
        window = [
            (idx, zh_events[idx])
            for idx in range(start, min(len(zh_events), start + 12))
            if idx not in used_zh
        ]
        match = best_overlap(en, window)
        if match:
            zh_idx, zh = match
            used_zh.add(zh_idx)
            paired.append((en, zh))
        else:
            paired.append((en, None))
    for idx, zh in enumerate(zh_events):
        if idx not in used_zh:
            paired.append((None, zh))
    paired.sort(key=lambda item: min(e.start for e in item if e is not None))
    return rebalance_fragment_pairs(paired)


def ass_escape(text: str) -> str:
    text = clean_text(text)
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", r"\\N", text)
    return text


def write_bilingual_ass(
    pairs: list[tuple[SubtitleEvent | None, SubtitleEvent | None]],
    out_path: Path,
    *,
    play_resolution: tuple[int, int] = (1920, 1080),
    style_profile: str = "adaptive",
    font_name: str = "",
    font_scale: int | float = 100,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ass_style_header(
        *play_resolution,
        profile_name=style_profile,
        font_name=font_name,
        font_scale=font_scale,
    )
    with out_path.open("w", encoding="utf-8-sig", newline="\n") as f:
        f.write(header)
        for en, zh in pairs:
            base = en or zh
            if base is None:
                continue
            zh_text = ass_escape(zh.text) if zh and zh.text else ""
            en_text = ass_escape(en.text) if en and en.text else ""
            if zh_text and en_text:
                text = f"{{\\rChinese}}{zh_text}\\N{{\\rSource}}{en_text}"
                style = "Chinese"
            elif zh_text:
                text = zh_text
                style = "Chinese"
            else:
                text = en_text
                style = "SourceOnly"
            f.write(
                "Dialogue: 0,"
                f"{seconds_to_ass_time(base.start)},{seconds_to_ass_time(base.end)},"
                f"{style},,0,0,0,,{text}\n"
            )


def parse_json_list_response(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("LLM response is not a JSON list")
    return data


def ollama_translate_events(
    events: list[SubtitleEvent],
    model: str,
    cache_path: Path,
    batch_size: int = 5,
    context_lines: int = 30,
    base_url: str = "http://127.0.0.1:11434",
) -> tuple[list[SubtitleEvent], list[SubtitleEvent]]:
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("requests is required for Ollama translation.") from exc
    done_en: list[SubtitleEvent] = []
    done_zh: list[SubtitleEvent] = []
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        for idx, item in enumerate(raw):
            source = events[idx]
            start = float(item.get("start", source.start))
            end = float(item.get("end", source.end))
            en_text = str(item.get("en", item.get("corrected_english", source.text)))
            zh_text = str(item.get("zh", item.get("text", "")))
            done_en.append(SubtitleEvent(start, end, clean_text(en_text)))
            done_zh.append(SubtitleEvent(start, end, clean_text(zh_text)))
    session = requests.Session()

    start_index = len(done_zh)
    start_index -= start_index % batch_size
    done_en = done_en[:start_index]
    done_zh = done_zh[:start_index]

    for idx in range(start_index, len(events), batch_size):
        batch = events[idx : idx + batch_size]
        before_start = max(0, idx - context_lines)
        after_end = min(len(events), idx + len(batch) + context_lines)
        before_text = "\n".join(f"[{i}] {events[i].text}" for i in range(before_start, idx))
        batch_text = "\n".join(f"[{j}] {event.text}" for j, event in enumerate(batch))
        after_text = "\n".join(f"[{i}] {events[i].text}" for i in range(idx + len(batch), after_end))

        prompt = f"""
You will correct and translate only the TARGET LINE(S).

Use the previous and following context to understand names, pronouns, topic continuity, and terminology.
Do not translate the context sections. They are reference only.
No timing labels are provided. Do not invent, change, or discuss timing labels.

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
  {{"index": 0, "corrected_english": "...", "chinese_translation": "..."}}
]

Rules:
- One input line must produce one output object.
- Do not merge two lines.
- Do not split one line into multiple objects.
- Do not output previous or following context lines.
- Do not add explanations, markdown, notes, or extra keys.
- Correct obvious OCR/ASR errors in English before translating.
- Translate into natural Simplified Chinese.
- You MUST translate uppercase descriptive text in parentheses or brackets (e.g. "(SIGHS)" or "[MUSIC]") into Simplified Chinese.
"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
            "options": {"temperature": 0.1},
        }
        
        data = None
        for attempt in range(3):
            try:
                response = session.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=600)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:
                print(f"LLM request failed (attempt {attempt+1}/3): {exc}")
                if attempt < 2:
                    import time
                    time.sleep(2)
                else:
                    print("网络或大模型服务异常，连续重试失败，已中断翻译。再次运行可从断点继续。")
                    import sys
                    sys.exit(1)
                    
        text = data.get("message", {}).get("content", "") if data else ""
        items = parse_json_list_response(text)
        lookup = {}
        for item in items:
            if isinstance(item, dict) and "index" in item:
                try:
                    lookup[int(item["index"])] = item
                except (TypeError, ValueError):
                    pass

        batch_en: list[SubtitleEvent] = []
        batch_zh: list[SubtitleEvent] = []
        for local_idx, event in enumerate(batch):
            item = lookup.get(local_idx, {})
            corrected = clean_text(str(item.get("corrected_english") or event.text))
            translated = clean_text(str(item.get("chinese_translation") or corrected))
            batch_en.append(SubtitleEvent(event.start, event.end, corrected))
            batch_zh.append(SubtitleEvent(event.start, event.end, translated))

        for source, corrected, translated in zip(batch, batch_en, batch_zh):
            if source.start != corrected.start or source.end != corrected.end:
                raise ValueError("English timing changed during translation batch")
            if source.start != translated.start or source.end != translated.end:
                raise ValueError("Chinese timing changed during translation batch")

        done_en.extend(batch_en)
        done_zh.extend(batch_zh)
        write_json_atomic(
            cache_path,
            [
                {"start": en.start, "end": en.end, "en": en.text, "zh": zh.text}
                for en, zh in zip(done_en, done_zh)
            ],
        )
        log(f"Translated {len(done_zh)}/{len(events)}")
        time.sleep(0.05)
    return done_en, done_zh


def print_streams(streams: list[StreamInfo]) -> None:
    for s in streams:
        if s.is_subtitle:
            title = f" | {s.title}" if s.title else ""
            print(f"0:{s.index:02d} {s.lang or '-':8s} {s.codec}{title}")


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "movie"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract/OCR/translate MKV subtitles and write bilingual ASS.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "runtime" / "scratch" / "outputs")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--ocr-scale", type=float, default=2.0)
    parser.add_argument("--crop-pad", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for PGS images per stream.")
    parser.add_argument("--fast-ocr", action="store_true", help="Allow faster OCR settings instead of best-quality defaults.")
    parser.add_argument("--list-streams", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5, help="LLM batch size in subtitle sentence units.")
    parser.add_argument("--context-lines", type=int, default=30, help="Reference this many subtitle lines before and after each target batch.")
    parser.add_argument(
        "--subtitle-style-profile",
        choices=STYLE_PROFILE_NAMES,
        default="adaptive",
        help="ASS display profile: adaptive, mobile, or compact.",
    )
    parser.add_argument("--subtitle-font-name", default="", help="ASS font family name. Defaults to Arial.")
    parser.add_argument("--subtitle-font-scale", type=int, default=100, help="ASS font scale percentage.")
    return parser


def main() -> int:
    configure_output_encoding()
    args = build_arg_parser().parse_args()
    video = args.video.resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    if args.list_streams:
        log(f"Using ffmpeg: {ffmpeg}")
        streams = probe_streams(video, ffmpeg)
        print_streams(streams)
        return 0

    log(
        "subtitle_pipeline.py is now a compatibility entry point; "
        "routing this run through audio_to_subtitle.py."
    )
    forwarded = [
        "audio_to_subtitle.py",
        "--video",
        str(video),
        "--source",
        "auto",
        "--output-root",
        str(args.output.resolve()),
        "--device",
        args.device,
        "--llm-model",
        args.model,
        "--ocr-scale",
        str(args.ocr_scale),
        "--crop-pad",
        str(args.crop_pad),
        "--limit",
        str(args.limit),
        "--batch-size",
        str(args.batch_size),
        "--context-lines",
        str(args.context_lines),
        "--subtitle-style-profile",
        args.subtitle_style_profile,
        "--subtitle-font-scale",
        str(args.subtitle_font_scale),
    ]
    if args.fast_ocr:
        forwarded.append("--fast-ocr")
    if args.subtitle_font_name:
        forwarded.extend(["--subtitle-font-name", args.subtitle_font_name])
    if args.ffmpeg:
        log(
            "The compatibility entry point ignores --ffmpeg; configure ffmpeg on PATH "
            "when using the canonical pipeline."
        )

    from audio_to_subtitle import main as canonical_main

    original_argv = sys.argv
    try:
        sys.argv = forwarded
        canonical_main()
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
