from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE_ASSET_SCHEMA_VERSION = 1
PROCESSING_PLAN_VERSION = 1

TEXT_SUBTITLE_CODECS = {
    "ass",
    "mov_text",
    "ssa",
    "srt",
    "subrip",
    "text",
    "webvtt",
}
BITMAP_SUBTITLE_TOKENS = {
    "hdmv_pgs",
    "pgs",
}
UNKNOWN_LANGUAGES = {"", "auto", "source", "und", "unknown"}


@dataclass(frozen=True)
class SubtitleAsset:
    asset_id: str
    origin: str
    representation: str
    language: str
    script: str
    role: str
    codec: str
    label: str
    text_authority: str
    timing_authority: str
    path: str = ""
    stream_index: int | None = None
    exact_edition: bool = False
    supported: bool = True
    disposition: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_ASSET_SCHEMA_VERSION,
            **asdict(self),
        }


def _stable_hash(payload: Mapping[str, Any], *, prefix: str) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def normalize_language(value: str) -> str:
    language = (value or "").strip().casefold().replace("_", "-")
    if language in UNKNOWN_LANGUAGES:
        return "" if language in {"", "und", "unknown"} else language
    if language in {"en", "eng", "english"} or "english" in language:
        return "en"
    if language in {"ja", "jpn", "jp", "japanese"} or "japanese" in language:
        return "ja"
    if language in {"ko", "kor", "korean"} or "korean" in language:
        return "ko"
    if language in {
        "zh",
        "zho",
        "chi",
        "chs",
        "cht",
        "cmn",
        "cn",
        "chinese",
        "zh-hans",
        "zh-hant",
        "zh-cn",
        "zh-tw",
        "zh-hk",
    } or "chinese" in language:
        return "zh"
    if language in {"fr", "fra", "fre", "french"} or "french" in language:
        return "fr"
    if language in {"de", "deu", "ger", "german"} or "german" in language:
        return "de"
    if language in {"es", "spa", "spanish"} or "spanish" in language:
        return "es"
    return language.split("-", 1)[0]


def infer_language_from_label(label: str) -> str:
    value = (label or "").casefold()
    parts = set(re.split(r"[^0-9a-zA-Z\u3400-\u9fff]+", value))
    if parts & {
        "zh",
        "zho",
        "chi",
        "chs",
        "cht",
        "cmn",
        "cn",
        "sc",
        "tc",
        "chinese",
    } or any(token in value for token in ("zh-hans", "zh-hant", "simplified", "traditional", "中文", "简体", "繁体")):
        return "zh"
    if parts & {"en", "eng", "english"} or "english" in value:
        return "en"
    if parts & {"ja", "jpn", "japanese"} or "japanese" in value:
        return "ja"
    if parts & {"ko", "kor", "korean"} or "korean" in value:
        return "ko"
    return ""


def infer_sidecar_language(path: Path) -> str:
    hinted = infer_language_from_label(path.stem)
    if hinted:
        return hinted
    try:
        raw = path.read_bytes()[:80_000]
    except OSError:
        return ""
    text = ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:20_000]
    if len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", sample)) >= 5:
        return "zh"
    if len(re.findall(r"\b[A-Za-z]{2,}\b", sample)) >= 5:
        return "en"
    return ""


def classify_script(language: str, label: str = "") -> str:
    normalized = (language or "").casefold().replace("_", "-")
    descriptor = f"{normalized} {label or ''}".casefold()
    if any(token in descriptor for token in ("zh-hant", "zh-tw", "zh-hk", "cht", "traditional", "繁体", "繁體")):
        return "traditional"
    if any(token in descriptor for token in ("zh-hans", "zh-cn", "chs", "simplified", "简体", "簡體")):
        return "simplified"
    return "unknown"


def classify_role(
    label: str,
    disposition: Mapping[str, Any] | None = None,
    *,
    coverage_ratio: float | None = None,
    events_per_minute: float | None = None,
) -> str:
    descriptor = (label or "").casefold()
    label_tokens = set(re.split(r"[^a-z0-9]+", descriptor))
    disposition = disposition or {}
    if bool(disposition.get("comment")) or bool(
        disposition.get("visual_impaired")
    ) or any(
        token in descriptor
        for token in (
            "commentary",
            "director comment",
            "audio description",
            "descriptive audio",
            "described audio",
            "导演评论",
            "導演評論",
            "导演解说",
            "導演解說",
            "无障碍音频",
            "無障礙音頻",
            "口述影像",
        )
    ):
        return "commentary"
    if bool(disposition.get("forced")) or any(
        token in descriptor
        for token in (
            "forced",
            "foreign only",
            "signs only",
            "songs & signs",
            "songs and signs",
            "强制",
            "強制",
            "外语对白",
            "外語對白",
        )
    ):
        return "forced"
    if bool(disposition.get("hearing_impaired")) or any(
        token in descriptor
        for token in (
            "sdh",
            "hearing impaired",
            "closed caption",
            "closed captions",
            " cc",
            "听障",
            "聽障",
            "无障碍字幕",
            "無障礙字幕",
        )
    ) or "cc" in label_tokens:
        return "sdh"
    if (
        coverage_ratio is not None
        and events_per_minute is not None
        and coverage_ratio < 0.03
        and events_per_minute < 0.5
    ):
        return "forced"
    if "full" in descriptor or "dialogue" in descriptor:
        return "dialogue"
    return "unknown"


def representation_for(codec: str, origin: str) -> str:
    if origin == "audio":
        return "speech_asr"
    normalized = (codec or "").casefold()
    if any(token in normalized for token in BITMAP_SUBTITLE_TOKENS):
        return "bitmap_ocr"
    codec_tokens = set(re.split(r"[^a-z0-9_]+", normalized))
    if codec_tokens & TEXT_SUBTITLE_CODECS:
        return "authored_text"
    if origin == "sidecar":
        return "authored_text"
    return "unsupported"


def _authority_for(representation: str, *, timing: bool) -> str:
    if representation == "authored_text":
        return "authored"
    if representation == "bitmap_ocr":
        return "authored" if timing else "ocr"
    if representation == "speech_asr":
        return "word_aligned" if timing else "asr"
    return "unknown"


def build_embedded_asset(video_path: Path, stream: Any) -> SubtitleAsset:
    language = normalize_language(str(getattr(stream, "lang", "") or ""))
    codec = str(getattr(stream, "codec", "") or "")
    title = str(getattr(stream, "title", "") or "")
    index = int(getattr(stream, "index"))
    disposition = {
        str(key): int(bool(value))
        for key, value in (getattr(stream, "disposition", None) or {}).items()
    }
    label = f"0:{index} {language or 'und'} {codec} {title}".strip()
    representation = representation_for(codec, "embedded")
    descriptor = {
        "origin": "embedded",
        "video": str(video_path.resolve()),
        "stream_index": index,
        "codec": codec,
        "language": language,
        "title": title,
    }
    metrics: dict[str, Any] = {}
    metadata = getattr(stream, "metadata", None) or {}
    for source_key, target_key in (
        ("NUMBER_OF_FRAMES", "event_count_hint"),
        ("number_of_frames", "event_count_hint"),
        ("nb_frames", "event_count_hint"),
        ("BPS", "bits_per_second"),
        ("bps", "bits_per_second"),
        ("bit_rate", "bits_per_second"),
    ):
        value = metadata.get(source_key)
        if value is None or target_key in metrics:
            continue
        try:
            metrics[target_key] = int(value)
        except (TypeError, ValueError):
            continue
    return SubtitleAsset(
        asset_id=_stable_hash(descriptor, prefix="embedded"),
        origin="embedded",
        representation=representation,
        language=language,
        script=classify_script(str(getattr(stream, "lang", "") or ""), title),
        role=classify_role(title, disposition),
        codec=codec,
        label=label,
        text_authority=_authority_for(representation, timing=False),
        timing_authority=_authority_for(representation, timing=True),
        stream_index=index,
        exact_edition=True,
        supported=representation != "unsupported",
        disposition=disposition,
        metrics=metrics,
    )


def build_sidecar_asset(
    video_path: Path,
    path: Path,
    *,
    language: str = "",
    likely_match: bool = False,
    score: int = 0,
) -> SubtitleAsset:
    resolved = path.expanduser().resolve()
    detected_language = normalize_language(language or infer_sidecar_language(resolved))
    codec = resolved.suffix.lower().lstrip(".")
    representation = representation_for(codec, "sidecar")
    descriptor = {
        "origin": "sidecar",
        "path": str(resolved),
        "size": resolved.stat().st_size if resolved.exists() else None,
        "mtime_ns": resolved.stat().st_mtime_ns if resolved.exists() else None,
        "video": str(video_path.resolve()),
    }
    return SubtitleAsset(
        asset_id=_stable_hash(descriptor, prefix="sidecar"),
        origin="sidecar",
        representation=representation,
        language=detected_language,
        script=classify_script(language or detected_language, resolved.stem),
        role=classify_role(resolved.stem),
        codec=codec,
        label=resolved.name,
        text_authority="authored",
        timing_authority="authored_unverified",
        path=str(resolved),
        exact_edition=bool(likely_match),
        supported=representation != "unsupported",
        metrics={"match_score": int(score)},
    )


def build_audio_asset(
    video_path: Path,
    stream: Any | None = None,
    *,
    stream_index: int | None = None,
    language: str = "",
) -> SubtitleAsset:
    index = int(getattr(stream, "index")) if stream is not None else stream_index
    language = (
        normalize_language(str(getattr(stream, "lang", "") or ""))
        if stream is not None
        else normalize_language(language)
    )
    codec = str(getattr(stream, "codec", "") or "") if stream is not None else "audio"
    title = str(getattr(stream, "title", "") or "") if stream is not None else ""
    disposition = {
        str(key): int(bool(value))
        for key, value in (
            (getattr(stream, "disposition", None) or {}).items()
            if stream is not None
            else []
        )
    }
    role = classify_role(title, disposition)
    if role == "unknown":
        role = "dialogue"
    descriptor = {
        "origin": "audio",
        "video": str(video_path.resolve()),
        "stream_index": index,
        "language": language,
        "codec": codec,
    }
    return SubtitleAsset(
        asset_id=_stable_hash(descriptor, prefix="audio"),
        origin="audio",
        representation="speech_asr",
        language=language,
        script="unknown",
        role=role,
        codec=codec,
        label=(f"0:{index} {language or 'und'} {codec} {title}".strip() if index is not None else "Primary audio"),
        text_authority="asr",
        timing_authority="word_aligned",
        stream_index=index,
        exact_edition=True,
        disposition=disposition,
    )


def _asset_score(asset: SubtitleAsset, lane: str, preferred_language: str) -> tuple[int, str]:
    score = {
        "authored_text": 500,
        "bitmap_ocr": 300,
        "speech_asr": 100,
    }.get(asset.representation, -1000)
    score += {"embedded": 120, "sidecar": 50, "audio": 0}.get(asset.origin, 0)
    score += {"dialogue": 120, "sdh": 100, "unknown": 50, "forced": -700, "commentary": -1000}.get(
        asset.role,
        0,
    )
    if asset.exact_edition:
        score += 40
    score += min(int(asset.metrics.get("match_score") or 0), 100)

    language = normalize_language(asset.language)
    preferred = normalize_language(preferred_language)
    if lane == "chinese":
        if language == "zh":
            score += 400
        else:
            score -= 800
        score += {"simplified": 30, "traditional": 20}.get(asset.script, 0)
    else:
        if language == "zh":
            score -= 900
        elif language:
            score += 300
        if asset.role == "sdh":
            score += 100
        if preferred and preferred not in {"auto", "source"} and language == preferred:
            score += 80
        if language == "en":
            score += 20
    return score, asset.asset_id


def _select_primary(
    assets: Sequence[SubtitleAsset],
    *,
    lane: str,
    preferred_language: str,
) -> SubtitleAsset | None:
    candidates = [
        asset
        for asset in assets
        if asset.supported
        and asset.role not in {"forced", "commentary"}
        and asset.origin != "audio"
    ]
    if lane == "chinese":
        candidates = [asset for asset in candidates if normalize_language(asset.language) == "zh"]
    else:
        candidates = [asset for asset in candidates if normalize_language(asset.language) != "zh"]
        known = [asset for asset in candidates if normalize_language(asset.language)]
        if known:
            candidates = known
    if not candidates:
        return None
    return max(candidates, key=lambda item: _asset_score(item, lane, preferred_language))


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _select_audio_reference(
    assets: Sequence[SubtitleAsset],
    preferred_language: str,
) -> SubtitleAsset | None:
    candidates = [
        asset
        for asset in assets
        if asset.origin == "audio"
        and asset.supported
        and asset.role != "commentary"
    ]
    if not candidates:
        return None
    preferred = normalize_language(preferred_language)
    return max(
        candidates,
        key=lambda item: (
            bool(
                preferred
                and preferred not in {"auto", "source"}
                and normalize_language(item.language) == preferred
            ),
            bool(item.disposition.get("default")),
            bool(normalize_language(item.language)),
            item.role == "dialogue",
            item.stream_index is not None,
            -(item.stream_index or 0),
        ),
    )


def build_processing_plan(
    assets: Sequence[SubtitleAsset],
    *,
    preferred_source_language: str = "auto",
    merge_existing: bool = True,
    synchronize_existing: bool = True,
    allow_audio_source: bool = True,
) -> dict[str, Any]:
    chinese = _select_primary(
        assets,
        lane="chinese",
        preferred_language=preferred_source_language,
    )
    source = _select_primary(
        assets,
        lane="source",
        preferred_language=preferred_source_language,
    )
    existing_assets = [
        asset
        for asset in assets
        if asset.origin in {"embedded", "sidecar"} and asset.supported
    ]
    commentary_audio_assets = [
        asset
        for asset in assets
        if asset.origin == "audio" and asset.role == "commentary"
    ]
    audio_reference = _select_audio_reference(
        assets,
        preferred_source_language,
    )

    if chinese is not None and not merge_existing:
        source = None
    if (
        allow_audio_source
        and chinese is None
        and source is None
        and audio_reference is not None
    ):
        source = audio_reference

    supplements: list[SubtitleAsset] = []
    primary_asset_ids = {
        asset.asset_id
        for asset in (chinese, source)
        if asset is not None
    }
    source_language = normalize_language(source.language) if source is not None else ""
    preferred_language = normalize_language(preferred_source_language)
    forced_candidates: list[SubtitleAsset] = []
    sdh_candidates: list[SubtitleAsset] = []
    for asset in assets:
        if asset.asset_id in primary_asset_ids or not asset.supported:
            continue
        if chinese is None and source is None:
            continue
        if not merge_existing:
            continue
        if asset.origin not in {"embedded", "sidecar"}:
            continue
        language = normalize_language(asset.language)
        if asset.role == "forced" and language != "zh":
            target_language = (
                source_language
                or (
                    preferred_language
                    if preferred_language not in {"", "auto", "source"}
                    else ""
                )
            )
            if target_language and language and language != target_language:
                continue
            forced_candidates.append(asset)
        elif (
            source is not None
            and source.role != "sdh"
            and asset.role == "sdh"
            and language == source_language
        ):
            sdh_candidates.append(asset)
    if forced_candidates:
        supplements.append(
            max(
                forced_candidates,
                key=lambda item: _asset_score(
                    item,
                    "source",
                    preferred_source_language,
                ),
            )
        )
    if sdh_candidates:
        supplements.append(
            max(
                sdh_candidates,
                key=lambda item: _asset_score(
                    item,
                    "source",
                    preferred_source_language,
                ),
            )
        )

    operations: list[str] = []
    local_gpu: list[str] = []
    remote_llm: list[str] = []
    warnings: list[str] = []
    generation_assets = [
        asset
        for asset in (chinese, source, *supplements)
        if asset is not None
    ]
    needs_audio_reference = (
        source is not None
        and source.origin == "audio"
        or (
            synchronize_existing
            and any(
                asset.origin in {"embedded", "sidecar"}
                for asset in generation_assets
            )
        )
    )
    selected: list[SubtitleAsset] = []
    for asset in (
        *generation_assets,
        *(
            (audio_reference,)
            if needs_audio_reference and audio_reference is not None
            else ()
        ),
    ):
        if asset.asset_id not in {item.asset_id for item in selected}:
            selected.append(asset)

    for asset in generation_assets:
        if asset.representation == "bitmap_ocr":
            _append_unique(operations, "extract_bitmap_subtitles")
            _append_unique(operations, "repair_low_confidence_ocr")
            _append_unique(local_gpu, "subtitle_ocr")
        elif asset.representation == "speech_asr":
            _append_unique(operations, "transcribe_audio")
            _append_unique(operations, "repair_asr_risk_windows")
            _append_unique(local_gpu, "speech_recognition")

    if chinese is not None:
        if chinese.script == "traditional":
            _append_unique(operations, "traditional_to_simplified")
        _append_unique(operations, "proofread_existing_chinese")
        _append_unique(remote_llm, "chinese_proofread")
    if source is not None and source.origin == "audio":
        _append_unique(operations, "translate_source")
        _append_unique(remote_llm, "translation")
    elif source is not None and chinese is None:
        _append_unique(operations, "translate_source")
        _append_unique(remote_llm, "translation")
    elif source is not None and chinese is not None:
        _append_unique(operations, "synchronize_existing_tracks")
        _append_unique(operations, "merge_language_lanes")
        _append_unique(operations, "translate_missing_chinese_only")
        _append_unique(remote_llm, "missing_content_review")
    if chinese is not None or source is not None:
        _append_unique(operations, "final_bilingual_quality_review")
        _append_unique(remote_llm, "final_quality_review")

    if existing_assets and all(asset.role in {"forced", "commentary"} for asset in existing_assets):
        warnings.append("只检测到 forced/评论字幕，没有可作为完整对白的已有字幕轨。")
    if (
        synchronize_existing
        and commentary_audio_assets
        and audio_reference is None
        and any(
            asset.origin in {"embedded", "sidecar"}
            for asset in generation_assets
        )
    ):
        warnings.append("只检测到评论或无障碍解说音轨，将保留已有字幕时间，不自动同步。")
    if commentary_audio_assets and chinese is None and source is None:
        warnings.append("只检测到评论或无障碍解说音轨，不能自动用于对白识别。")
    if chinese is None and source is None:
        warnings.append("没有选择到可用的字幕或音频来源。")
    for asset in assets:
        if asset.representation == "unsupported":
            warnings.append(f"不支持的字幕编码：{asset.codec}。")

    timing_candidates = [asset for asset in (source, chinese) if asset is not None]
    timing_reference = (
        max(
            timing_candidates,
            key=lambda item: (
                item.timing_authority == "authored",
                item.origin == "embedded",
                item.representation == "authored_text",
            ),
        )
        if timing_candidates
        else None
    )
    source_modes = {asset.origin for asset in generation_assets}
    existing_source_modes = source_modes & {"sidecar", "embedded"}
    if len(existing_source_modes) > 1:
        recommended_source = "auto"
    elif existing_source_modes:
        recommended_source = next(iter(existing_source_modes))
    else:
        recommended_source = "audio"

    if chinese is not None:
        route = "proofread_existing_chinese"
    elif source is None:
        route = "no_safe_source"
    elif source.origin != "audio":
        route = "translate_existing_source"
    else:
        route = "transcribe_translate"
    plan_base = {
        "version": PROCESSING_PLAN_VERSION,
        "route": route,
        "recommended_source": recommended_source,
        "lanes": {
            "chinese": chinese.asset_id if chinese else None,
            "source": source.asset_id if source else None,
            "supplementary": [asset.asset_id for asset in supplements],
            "timing_reference": timing_reference.asset_id if timing_reference else None,
            "audio_reference": (
                audio_reference.asset_id
                if needs_audio_reference and audio_reference is not None
                else None
            ),
        },
        "operations": operations,
        "compute": {
            "local_gpu": local_gpu,
            "remote_llm": remote_llm,
        },
        "warnings": warnings,
        "selected_assets": [asset.to_dict() for asset in selected],
    }
    return {
        **plan_base,
        "fingerprint": _stable_hash(plan_base, prefix="plan"),
    }


def source_manifest(
    assets: Sequence[SubtitleAsset],
    plan: Mapping[str, Any],
    *,
    video_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "version": SOURCE_ASSET_SCHEMA_VERSION,
        "video": str(video_path.resolve()) if video_path is not None else None,
        "assets": [asset.to_dict() for asset in assets],
        "processing_plan_fingerprint": plan.get("fingerprint"),
    }


def attach_asset_to_segments(
    segments: Iterable[dict[str, Any]],
    asset: SubtitleAsset,
    *,
    lane: str = "source",
) -> None:
    asset_data = asset.to_dict()
    for segment in segments:
        existing_assets = [
            dict(item)
            for item in segment.get("source_assets") or []
            if isinstance(item, dict)
        ]
        if not any(item.get("asset_id") == asset.asset_id for item in existing_assets):
            existing_assets.append(asset_data)
        segment["source_assets"] = existing_assets
        segment.setdefault(f"{lane}_asset_id", asset.asset_id)
        segment.setdefault(f"{lane}_origin", asset.origin)
        segment.setdefault(f"{lane}_representation", asset.representation)
        segment.setdefault(f"{lane}_role", asset.role)
        segment.setdefault(f"{lane}_text_authority", asset.text_authority)
        segment.setdefault(f"{lane}_timing_authority", asset.timing_authority)
        if lane == "source":
            segment.setdefault("text_authority", asset.text_authority)
            segment.setdefault("timing_authority", asset.timing_authority)


def assets_from_segments(segments: Iterable[Mapping[str, Any]]) -> list[SubtitleAsset]:
    assets: dict[str, SubtitleAsset] = {}
    for segment in segments:
        for raw in segment.get("source_assets") or []:
            if not isinstance(raw, Mapping):
                continue
            asset_id = str(raw.get("asset_id") or "")
            if not asset_id or asset_id in assets:
                continue
            asset = asset_from_dict(raw)
            if asset is not None:
                assets[asset_id] = asset
    return list(assets.values())


def asset_from_dict(raw: Mapping[str, Any]) -> SubtitleAsset | None:
    asset_id = str(raw.get("asset_id") or "")
    if not asset_id:
        return None
    try:
        return SubtitleAsset(
            asset_id=asset_id,
            origin=str(raw.get("origin") or "unknown"),
            representation=str(raw.get("representation") or "unsupported"),
            language=str(raw.get("language") or ""),
            script=str(raw.get("script") or "unknown"),
            role=str(raw.get("role") or "unknown"),
            codec=str(raw.get("codec") or ""),
            label=str(raw.get("label") or asset_id),
            text_authority=str(raw.get("text_authority") or "unknown"),
            timing_authority=str(raw.get("timing_authority") or "unknown"),
            path=str(raw.get("path") or ""),
            stream_index=(
                int(raw["stream_index"])
                if raw.get("stream_index") is not None
                else None
            ),
            exact_edition=bool(raw.get("exact_edition")),
            supported=bool(raw.get("supported", True)),
            disposition={
                str(key): int(bool(value))
                for key, value in (raw.get("disposition") or {}).items()
            },
            metrics=dict(raw.get("metrics") or {}),
        )
    except (TypeError, ValueError):
        return None


def is_authored_source(segment: Mapping[str, Any]) -> bool:
    return str(
        segment.get("source_text_authority")
        or segment.get("text_authority")
        or ""
    ) == "authored"


def model_may_hide_segment(segment: Mapping[str, Any]) -> bool:
    authorities = {
        str(segment.get("source_text_authority") or ""),
        str(segment.get("chinese_text_authority") or ""),
        str(segment.get("text_authority") or ""),
    }
    authorities.discard("")
    if "authored" in authorities:
        return False
    return not authorities or bool(authorities & {"asr", "ocr", "unknown"})
