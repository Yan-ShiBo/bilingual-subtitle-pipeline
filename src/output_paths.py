from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


OUTPUT_DIRECTORY_NAME = "1 \u5b57\u5e55"
OUTPUT_ARTIFACT_PATTERNS = (
    "*.segments.source.json",
    "*.segments.checkpoint.json",
    "*.bilingual.ass",
)


def same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return str(first).casefold() == str(second).casefold()


def directory_has_output_artifacts(path: Path) -> bool:
    return path.is_dir() and any(any(path.glob(pattern)) for pattern in OUTPUT_ARTIFACT_PATTERNS)


def candidate_output_roots(video: Path, extra_roots: Iterable[Path] = ()) -> list[Path]:
    candidates = [video.parent]
    for ancestor in list(video.parents)[:4]:
        subtitle_root = ancestor / OUTPUT_DIRECTORY_NAME
        if subtitle_root.is_dir():
            candidates.append(subtitle_root)
    candidates.extend(root for root in extra_roots if root)

    unique: list[Path] = []
    for candidate in candidates:
        if any(same_path(candidate, existing) for existing in unique):
            continue
        unique.append(candidate)
    return unique


def episode_identity(value: str) -> tuple[int | None, int, int | None] | None:
    normalized = re.sub(r"[_\-.]+", " ", value or "")
    match = re.search(r"\bS(\d{1,2})\s*E(\d{1,3})\b", normalized, flags=re.I)
    if match:
        return int(match.group(1)), int(match.group(2)), None
    match = re.search(r"\b(\d{1,3})\s*of\s*(\d{1,3})\b", normalized, flags=re.I)
    if match:
        return None, int(match.group(1)), int(match.group(2))
    match = re.search(r"\b(?:episode|ep)\s*(\d{1,3})\b", normalized, flags=re.I)
    if match:
        return None, int(match.group(1)), None
    return None


def episode_identities_match(
    first: tuple[int | None, int, int | None],
    second: tuple[int | None, int, int | None],
) -> bool:
    first_season, first_episode, first_total = first
    second_season, second_episode, second_total = second
    if first_episode != second_episode:
        return False
    if first_season is not None and second_season is not None and first_season != second_season:
        return False
    if first_total is not None and second_total is not None and first_total != second_total:
        return False
    return True


def normalized_title(value: str) -> str:
    text = re.sub(r"\bS\d{1,2}\s*E\d{1,3}\b", " ", value or "", flags=re.I)
    text = re.sub(r"\b\d{1,3}\s*of\s*\d{1,3}\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:episode|ep)\s*\d{1,3}\b", " ", text, flags=re.I)
    text = re.sub(
        r"\b(?:tvrip|bluray|web[- ]?dl|webrip|remux|xvid|x26[45]|h26[45]|"
        r"2160p|1080p|720p|hdr10?|dovi|dv|aac|dts|truehd|atmos)\b",
        " ",
        text,
        flags=re.I,
    )
    return "".join(character.casefold() for character in text if character.isalnum())


def output_match_score(video: Path, output_dir: Path) -> float:
    video_identity = episode_identity(video.stem)
    output_identity = episode_identity(f"{output_dir.parent.name} {output_dir.name}")
    if (video_identity is None) != (output_identity is None):
        return -1.0
    if video_identity is not None and output_identity is not None and not episode_identities_match(video_identity, output_identity):
        return -1.0

    video_title = normalized_title(video.stem)
    output_titles = [
        normalized_title(output_dir.parent.name),
        normalized_title(output_dir.name),
        normalized_title(f"{output_dir.parent.name} {output_dir.name}"),
    ]
    output_titles = [title for title in output_titles if title]
    if not video_title or not output_titles:
        return -1.0
    return max(SequenceMatcher(None, video_title, title).ratio() for title in output_titles)


def find_existing_output_root(
    video: Path,
    series_name: str,
    movie_name: str,
    extra_roots: Iterable[Path] = (),
) -> Path | None:
    roots = candidate_output_roots(video, extra_roots)
    for root in roots:
        exact_dir = root / series_name / movie_name
        if directory_has_output_artifacts(exact_dir):
            return root

    best: tuple[float, Path] | None = None
    for root in roots:
        if not root.is_dir():
            continue
        artifact_dirs: set[Path] = set()
        for pattern in OUTPUT_ARTIFACT_PATTERNS:
            artifact_dirs.update(path.parent for path in root.glob(f"*/*/{pattern}"))
        for output_dir in artifact_dirs:
            score = output_match_score(video, output_dir)
            threshold = 0.28 if episode_identity(video.stem) is not None else 0.68
            if score >= threshold and (best is None or score > best[0]):
                best = (score, root)
    return best[1] if best else None


def suggested_output_root(video: Path) -> Path:
    for ancestor in list(video.parents)[:4]:
        subtitle_root = ancestor / OUTPUT_DIRECTORY_NAME
        if subtitle_root.is_dir():
            return subtitle_root
    return video.parent / OUTPUT_DIRECTORY_NAME


def resolve_output_root(
    video: Path,
    series_name: str,
    movie_name: str,
    requested: Path | None = None,
    legacy_default: Path | None = None,
) -> tuple[Path, str]:
    if requested is not None and (legacy_default is None or not same_path(requested, legacy_default)):
        return requested, "manual"

    extra_roots = [requested] if requested is not None else []
    existing = find_existing_output_root(video, series_name, movie_name, extra_roots=extra_roots)
    if existing is not None:
        return existing, "existing"
    return suggested_output_root(video), "suggested"
