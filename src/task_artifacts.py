from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_SPECS = (
    ("bilingual_ass", "双语 ASS", ".bilingual.ass"),
    ("bilingual_srt", "双语 SRT", ".bilingual.srt"),
    ("bilingual_font_bundle", "字体内封 MKS", ".bilingual.mks"),
    ("chinese_ass", "中文 ASS", ".zh.ass"),
    ("english_ass", "源文 ASS", ".en.ass"),
    ("chinese_srt", "中文 SRT", ".zh.srt"),
    ("english_srt", "源文 SRT", ".en.srt"),
    ("render_preview", "渲染预览", ".render-preview.png"),
    ("quality_report", "质量报告", ".quality-report.json"),
    ("source_manifest", "来源清单", ".source-manifest.json"),
    ("processing_plan", "处理计划", ".processing-plan.json"),
    ("style_guide", "风格指南", ".style-guide.json"),
    ("terminology_review", "术语审查", ".terminology-review.json"),
    ("terminology_overrides", "术语人工修订", ".terminology-overrides.json"),
    ("final_qa", "独立终审", ".final-qa.json"),
    ("autonomous_review", "夜间复核", ".autonomous-review.json"),
    ("render_validation", "渲染验证", ".render-validation.json"),
    ("subtitle_sync_report", "同步报告", ".subtitle-sync.report.json"),
    ("scene_cut_cache", "镜头切点", ".scene-cuts.json"),
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def task_output_dir(output_root: Path, series_name: str, movie_name: str) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    components: list[str] = []
    for label, raw in (("series", series_name), ("movie", movie_name)):
        value = str(raw or "").strip()
        if (
            not value
            or value in {".", ".."}
            or Path(value).is_absolute()
            or "/" in value
            or "\\" in value
            or "\0" in value
        ):
            raise ValueError(f"Invalid {label} name")
        components.append(value)
    candidate = (root / components[0] / components[1]).resolve(strict=False)
    if not _is_within(candidate, root):
        raise ValueError("Output directory escapes the selected output root")
    return candidate


def collect_task_artifacts(out_dir: Path, movie_name: str) -> list[dict[str, Any]]:
    resolved_dir = out_dir.expanduser().resolve(strict=False)
    report_path = resolved_dir / f"{movie_name}.quality-report.json"
    report = _read_json(report_path)
    report_artifacts = report.get("artifacts") if isinstance(report, dict) else {}
    report_artifacts = report_artifacts if isinstance(report_artifacts, dict) else {}

    output: list[dict[str, Any]] = []
    for key, label, suffix in ARTIFACT_SPECS:
        raw_path = report_artifacts.get(key)
        if raw_path:
            candidate = Path(str(raw_path)).expanduser()
            if not candidate.is_absolute():
                candidate = resolved_dir / candidate
        else:
            candidate = resolved_dir / f"{movie_name}{suffix}"
        candidate = candidate.resolve(strict=False)
        if not _is_within(candidate, resolved_dir) or not candidate.is_file():
            continue
        stat = candidate.stat()
        output.append(
            {
                "key": key,
                "label": label,
                "name": candidate.name,
                "path": str(candidate),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    timezone.utc,
                ).isoformat(),
            }
        )
    return output


def task_snapshot(
    output_root: Path,
    series_name: str,
    movie_name: str,
) -> dict[str, Any]:
    out_dir = task_output_dir(output_root, series_name, movie_name)
    report_path = out_dir / f"{movie_name}.quality-report.json"
    checkpoint_path = out_dir / f"{movie_name}.segments.checkpoint.json"
    run_state_paths = list(out_dir.glob(".*.subtitle-run.json")) if out_dir.is_dir() else []
    report = _read_json(report_path)
    summary = report.get("summary") if isinstance(report, dict) else {}
    artifacts = collect_task_artifacts(out_dir, movie_name)
    timestamps = [item.stat().st_mtime for item in (report_path, checkpoint_path) if item.exists()]
    timestamps.extend(item.stat().st_mtime for item in run_state_paths if item.exists())
    timestamps.extend(
        Path(item["path"]).stat().st_mtime
        for item in artifacts
        if Path(item["path"]).exists()
    )
    modified_at = max(timestamps, default=0.0)
    quality_status = str(report.get("status") or "") if isinstance(report, dict) else ""
    status = "running" if run_state_paths else quality_status or (
        "partial" if checkpoint_path.exists() or artifacts else "empty"
    )
    return {
        "series_name": series_name,
        "movie_name": movie_name,
        "output_root": str(output_root.expanduser().resolve(strict=False)),
        "output_dir": str(out_dir),
        "status": status,
        "quality_status": quality_status,
        "review_items": int((summary or {}).get("review_items") or 0),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "updated_at": (
            datetime.fromtimestamp(modified_at, timezone.utc).isoformat()
            if modified_at
            else ""
        ),
    }


def discover_recent_tasks(output_root: Path, limit: int = 20) -> list[dict[str, Any]]:
    root = output_root.expanduser().resolve(strict=False)
    if not root.is_dir():
        return []
    candidate_dirs: set[Path] = set()
    for pattern in (
        "*/*/*.quality-report.json",
        "*/*/*.segments.checkpoint.json",
        "*/*/*.bilingual.ass",
        "*/*/.*.subtitle-run.json",
    ):
        candidate_dirs.update(path.parent for path in root.glob(pattern) if path.is_file())

    snapshots: list[dict[str, Any]] = []
    for out_dir in candidate_dirs:
        if not _is_within(out_dir.resolve(strict=False), root):
            continue
        try:
            snapshot = task_snapshot(root, out_dir.parent.name, out_dir.name)
        except (OSError, ValueError):
            continue
        if snapshot["status"] != "empty":
            # History only needs a compact summary. The current task endpoint exposes
            # the full artifact manifest when the user loads that task.
            snapshot.pop("artifacts", None)
            snapshots.append(snapshot)
    snapshots.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return snapshots[: max(1, min(100, int(limit)))]


def resolve_known_target(
    output_root: Path,
    series_name: str,
    movie_name: str,
    artifact_key: str,
) -> Path:
    out_dir = task_output_dir(output_root, series_name, movie_name)
    if artifact_key == "output_dir":
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Output directory not found: {out_dir}")
        if task_snapshot(output_root, series_name, movie_name)["status"] == "empty":
            raise FileNotFoundError(
                f"Directory is not a recognized subtitle task output: {out_dir}"
            )
        return out_dir
    for item in collect_task_artifacts(out_dir, movie_name):
        if item["key"] == artifact_key:
            return Path(item["path"])
    raise FileNotFoundError(f"Artifact not found: {artifact_key}")


def open_in_file_manager(target: Path) -> None:
    resolved = target.resolve(strict=True)
    if os.name == "nt":
        command = (
            ["explorer.exe", str(resolved)]
            if resolved.is_dir()
            else ["explorer.exe", f"/select,{resolved}"]
        )
    elif sys.platform == "darwin":
        command = ["open", str(resolved)] if resolved.is_dir() else ["open", "-R", str(resolved)]
    else:
        command = ["xdg-open", str(resolved if resolved.is_dir() else resolved.parent)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
