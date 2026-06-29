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
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFilter


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


@dataclass
class StreamInfo:
    index: int
    kind: str
    codec: str
    lang: str = ""
    title: str = ""
    metadata: dict[str, str] | None = None

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
        return any(token in codec for token in TEXT_CODECS)


@dataclass
class SubtitleEvent:
    start: float
    end: float
    text: str


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


def probe_streams(video: Path, ffmpeg: str) -> list[StreamInfo]:
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
    if "forced" in title:
        score -= 10_000
    if "full" in title:
        score += 2_000
    if "sdh" in title:
        score += 500
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
    text = re.sub(r"<[^>]+>", "", text)
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
    return [SubtitleEvent(e.start, e.end, converter.convert(e.text)) for e in events]


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

    def recognize(self, image_path: Path) -> str:
        if hasattr(self.ocr, "predict"):
            result = self.ocr.predict(str(image_path))
        else:
            result = self.ocr.ocr(str(image_path), cls=False)
        return extract_ocr_text(result, self.lang)


def extract_ocr_text(result: Any, lang: str) -> str:
    lines: list[tuple[float, float, str]] = []

    def add_line(box: Any, text: str) -> None:
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
        lines.append((y, x, clean_text(str(text))))

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
                texts = obj.get("rec_texts") or []
                boxes = obj.get("rec_boxes")
                if boxes is None or (hasattr(boxes, "__len__") and len(boxes) == 0):
                    boxes = obj.get("rec_polys")
                if boxes is None:
                    boxes = []
                for idx, text in enumerate(texts):
                    box = boxes[idx] if idx < len(boxes) else None
                    if box is not None and hasattr(box, "tolist"):
                        box = box.tolist()
                    add_line(box, text)
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
                add_line(obj[0], text)
                return
            for item in obj:
                walk(item)

    walk(result)
    if not lines:
        return ""
    lines.sort(key=lambda item: (round(item[0] / 12), item[1]))
    texts = [item[2] for item in lines if item[2]]
    if lang in {"ch", "chinese_cht"}:
        return clean_text("".join(texts))
    return clean_text(" ".join(texts))


def ocr_pgs_events(
    image_events: list[PgsImageEvent],
    lang: str,
    device: str,
    cache_path: Path,
    prefer_accuracy: bool,
) -> list[SubtitleEvent]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(cached) == len(image_events):
            return [SubtitleEvent(float(x["start"]), float(x["end"]), str(x["text"])) for x in cached]

    engine = PaddleOcrEngine(lang=lang, device=device, prefer_accuracy=prefer_accuracy)
    engine.assert_gpu()
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    iterator: Iterable[PgsImageEvent]
    iterator = tqdm(image_events, desc=f"OCR {lang}", unit="line") if tqdm else image_events
    out: list[SubtitleEvent] = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(iterator, 1):
        text = engine.recognize(item.image_path)
        event = SubtitleEvent(item.start, item.end, text)
        out.append(event)
        if idx % 20 == 0:
            write_json_atomic(cache_path, [e.__dict__ for e in out])
    write_json_atomic(cache_path, [e.__dict__ for e in out])
    return out


def read_or_extract_text_events(
    video: Path,
    stream: StreamInfo,
    ffmpeg: str,
    work_dir: Path,
) -> list[SubtitleEvent]:
    srt_path = work_dir / f"stream_{stream.index:02d}.srt"
    if not srt_path.exists():
        extract_subtitle(video, stream.index, ffmpeg, srt_path, "srt")
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
    if not sup_path.exists():
        log(f"Extracting PGS stream 0:{stream.index} -> {sup_path}")
        extract_subtitle(video, stream.index, ffmpeg, sup_path, "copy")
    if image_events_path.exists():
        raw = json.loads(image_events_path.read_text(encoding="utf-8"))
        image_events = [
            PgsImageEvent(float(x["start"]), float(x["end"]), Path(x["image_path"]), str(x["image_hash"]))
            for x in raw
        ]
    else:
        log(f"Rendering PGS images from {sup_path.name}")
        image_events = parse_pgs_to_images(sup_path, image_dir, scale=scale, pad=pad, limit=limit)
        write_json_atomic(
            image_events_path,
            [
                {
                    "start": e.start,
                    "end": e.end,
                    "image_path": str(e.image_path),
                    "image_hash": e.image_hash,
                }
                for e in image_events
            ],
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
    t = clean_text(text).upper()
    if not t:
        return False
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
    letters = [c for c in t if c.isalpha()]
    upper_ratio = sum(1 for c in letters if c == c.upper()) / max(1, len(letters))
    return has_cue_word and upper_ratio > 0.85


def best_overlap(base: SubtitleEvent, candidates: list[tuple[int, SubtitleEvent]]) -> tuple[int, SubtitleEvent] | None:
    best: tuple[float, float, SubtitleEvent] | None = None
    best_idx = -1
    base_center = (base.start + base.end) / 2
    for idx, cand in candidates:
        overlap = max(0.0, min(base.end, cand.end) - max(base.start, cand.start))
        cand_center = (cand.start + cand.end) / 2
        distance = abs(base_center - cand_center)
        if overlap <= 0 and distance > 3.5:
            continue
        score = overlap * 10 - distance
        if best is None or score > best[0]:
            best = (score, distance, cand)
            best_idx = idx
    if not best:
        return None
    score, distance, cand = best
    if score < -2.5 and distance > 2.5:
        return None
    return best_idx, cand


def pair_events(en_events: list[SubtitleEvent], zh_events: list[SubtitleEvent]) -> list[tuple[SubtitleEvent | None, SubtitleEvent | None]]:
    paired: list[tuple[SubtitleEvent | None, SubtitleEvent | None]] = []
    used_zh: set[int] = set()
    start = 0
    for en in en_events:
        while start < len(zh_events) and zh_events[start].end < en.start - 5:
            start += 1
        if is_sdh_sound_cue(en.text):
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
    return paired


def ass_escape(text: str) -> str:
    text = clean_text(text)
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", r"\\N", text)
    return text


def write_bilingual_ass(pairs: list[tuple[SubtitleEvent | None, SubtitleEvent | None]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = """[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Bilingual,Microsoft YaHei,46,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2.2,0.7,2,80,80,58,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with out_path.open("w", encoding="utf-8-sig", newline="\n") as f:
        f.write(header)
        for en, zh in pairs:
            base = en or zh
            if base is None:
                continue
            zh_text = ass_escape(zh.text) if zh and zh.text else ""
            en_text = ass_escape(en.text) if en and en.text else ""
            if zh_text and en_text:
                text = f"{zh_text}\\N{{\\fs34}}{en_text}"
            elif zh_text:
                text = zh_text
            else:
                text = f"{{\\fs38}}{en_text}"
            f.write(
                "Dialogue: 0,"
                f"{seconds_to_ass_time(base.start)},{seconds_to_ass_time(base.end)},"
                f"Bilingual,,0,0,0,,{text}\n"
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
"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        }
        response = session.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        text = data.get("message", {}).get("content", "")
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
    return parser


def main() -> int:
    configure_output_encoding()
    args = build_arg_parser().parse_args()
    video = args.video.resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    log(f"Using ffmpeg: {ffmpeg}")
    streams = probe_streams(video, ffmpeg)
    if args.list_streams:
        print_streams(streams)
        return 0
    en_stream, zh_stream, zh_kind = choose_streams(streams)
    log(f"English stream: 0:{en_stream.index} {en_stream.codec} {en_stream.title}")
    if zh_stream:
        log(f"Chinese stream: 0:{zh_stream.index} {zh_stream.codec} {zh_stream.title} ({zh_kind})")
    else:
        log("Chinese stream: not found; Ollama translation will be used.")

    base_dir = args.output.resolve() / safe_stem(video)
    base_dir.mkdir(parents=True, exist_ok=True)

    en_ocr_lang = "en"
    en_events = get_stream_events(video, en_stream, ffmpeg, base_dir / "english", en_ocr_lang, args)
    en_events = [e for e in en_events if e.text.strip()]
    en_srt = base_dir / f"{safe_stem(video)}.en.srt"
    write_srt(en_events, en_srt)
    log(f"English SRT: {en_srt} ({len(en_events)} lines)")

    if zh_stream:
        if zh_kind == "zh-Hant":
            zh_ocr_lang = "chinese_cht"
        else:
            zh_ocr_lang = "ch"
        zh_events = get_stream_events(video, zh_stream, ffmpeg, base_dir / "chinese", zh_ocr_lang, args)
        zh_events = [e for e in zh_events if e.text.strip()]
        if zh_kind in {"zh-Hant", "zh-yue"}:
            zh_events = convert_traditional_to_simplified(zh_events)
        zh_srt = base_dir / f"{safe_stem(video)}.zh-Hans.srt"
        write_srt(zh_events, zh_srt)
        log(f"Simplified Chinese SRT: {zh_srt} ({len(zh_events)} lines)")
    else:
        zh_cache = base_dir / "ollama_zh_cache.json"
        model_name = args.model
        base_url = "http://127.0.0.1:11434"
        if model_name.startswith("remote:"):
            parts = model_name.split(":", 2)
            if len(parts) == 3:
                base_url = "http://127.0.0.1:11435"
                model_name = parts[2]

        en_events, zh_events = ollama_translate_events(
            en_events,
            model_name,
            zh_cache,
            batch_size=args.batch_size,
            context_lines=args.context_lines,
            base_url=base_url,
        )
        corrected_en_srt = base_dir / f"{safe_stem(video)}.en.corrected.srt"
        write_srt(en_events, corrected_en_srt)
        log(f"Corrected English SRT: {corrected_en_srt} ({len(en_events)} lines)")
        zh_srt = base_dir / f"{safe_stem(video)}.zh-Hans.ollama.srt"
        write_srt(zh_events, zh_srt)
        log(f"Ollama Chinese SRT: {zh_srt} ({len(zh_events)} lines)")

    pairs = pair_events(en_events, zh_events)
    ass_path = base_dir / f"{safe_stem(video)}.bilingual.ass"
    log("Generating bilingual subtitles...")
    write_bilingual_ass(pairs, ass_path)
    matched = sum(1 for en, zh in pairs if en and zh and zh.text.strip())
    zh_total = sum(1 for _, zh in pairs if zh and zh.text.strip())
    log(f"Bilingual ASS: {ass_path}")
    log(f"Matched bilingual lines: {matched}; Chinese lines included: {zh_total}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
