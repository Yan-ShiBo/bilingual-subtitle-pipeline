from __future__ import annotations

import os
import time
from typing import Any

from process_lock import FileMutex
from subtitle_queue import WORKER_LOCK_PATH, queue_store


IDLE_EXIT_SECONDS = 300
POLL_SECONDS = 2


def _selected_asset(plan: dict[str, Any], lane_name: str) -> dict[str, Any] | None:
    lanes = plan.get("lanes") if isinstance(plan, dict) else {}
    assets = plan.get("selected_assets") if isinstance(plan, dict) else []
    if not isinstance(lanes, dict) or not isinstance(assets, list):
        return None
    asset_id = lanes.get(lane_name)
    return next(
        (
            asset
            for asset in assets
            if isinstance(asset, dict) and asset.get("asset_id") == asset_id
        ),
        None,
    )


def _apply_asset_selection(
    payload: dict[str, Any],
    asset: dict[str, Any] | None,
    *,
    sidecar_field: str,
    stream_field: str,
) -> None:
    if not asset:
        return
    if asset.get("origin") == "sidecar" and asset.get("path"):
        payload[sidecar_field] = str(asset["path"])
    elif asset.get("origin") == "embedded" and asset.get("stream_index") is not None:
        payload[stream_field] = int(asset["stream_index"])


def prepare_payload(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from subtitle_frontend import analyze_input

    payload = dict(item.get("payload") or {})
    payload["path"] = str(item["path"])
    analysis = analyze_input(payload)
    payload["output_root"] = str(analysis["output_root"])
    payload["series_name"] = str(payload.get("series_name") or analysis["series_name"])
    payload["movie_name"] = str(analysis["movie_name"])
    payload["run_mode"] = "resume"

    plan = analysis.get("processing_plan") or {}
    merge_existing = bool(payload.get("merge_existing_subtitles", True))
    chinese = _selected_asset(plan, "chinese")
    source = _selected_asset(plan, "source")
    audio = _selected_asset(plan, "audio_reference")
    if merge_existing:
        _apply_asset_selection(
            payload,
            chinese,
            sidecar_field="chinese_subtitle_file",
            stream_field="chinese_subtitle_stream",
        )
        _apply_asset_selection(
            payload,
            source,
            sidecar_field="english_subtitle_file",
            stream_field="english_subtitle_stream",
        )
    else:
        _apply_asset_selection(
            payload,
            chinese or source,
            sidecar_field="subtitle_file",
            stream_field="subtitle_stream",
        )
    if audio and audio.get("origin") == "audio" and audio.get("stream_index") is not None:
        payload["audio_stream"] = int(audio["stream_index"])
    return payload, analysis


def status_payload(item: dict[str, Any]) -> dict[str, Any] | None:
    prepared = item.get("prepared_payload")
    if not isinstance(prepared, dict):
        return None
    payload = dict(prepared)
    payload["pid"] = int(item.get("pid") or 0)
    return payload


def _progress(status: dict[str, Any]) -> tuple[int, int | None, int]:
    completed = int(status.get("completed_count") or 0)
    total_raw = status.get("total_count")
    total = int(total_raw) if total_raw not in (None, "") else None
    percent = round(completed * 100 / total) if total and total > 0 else 0
    if status.get("outcome") == "success":
        percent = 100
    return completed, total, max(0, min(100, percent))


def _finish_item(item_id: str, status: dict[str, Any]) -> None:
    outcome = str(status.get("outcome") or "failed")
    completed, total, percent = _progress(status)
    if outcome == "success":
        pending = int(status.get("quality_review_pending_count") or 0)
        reported = int(status.get("quality_review_reported_count") or pending)
        reviewed = int(status.get("quality_review_reviewed_sample_count") or 0)
        quality_status = str(status.get("quality_status") or "")
        manual_review_pending = bool(status.get("manual_review_pending"))
        if manual_review_pending:
            stage = "已完成，人工复核修改待重新生成"
        elif pending == 0 and quality_status == "review" and reported > 0 and reviewed:
            stage = f"已完成，质检 {reported} 项（{reviewed} 个代表样本已复核）"
        elif pending == 0 and quality_status == "review" and reported > 0:
            stage = f"已完成，质检报告仍建议复核 {reported} 项"
        elif pending == 0:
            stage = "已完成"
        elif reported > pending:
            stage = f"已完成，质检发现 {reported} 项（{pending} 个代表样本待复核）"
        else:
            stage = f"已完成，待复核 {pending} 项"
        queue_store.update_item(
            item_id,
            status="success",
            stage=stage,
            progress_percent=100,
            completed_count=completed,
            total_count=total,
            quality_status=quality_status,
            review_pending_count=pending,
            quality_reported_count=reported,
            quality_reviewed_sample_count=reviewed,
            manual_review_pending=manual_review_pending,
            quality_review_items=status.get("quality_review_items") or [],
            output_dir=str(status.get("output_dir") or ""),
            error="",
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        return

    cancelled = outcome == "stopped"
    queue_store.update_item(
        item_id,
        status="cancelled" if cancelled else "failed",
        stage="已取消" if cancelled else str(status.get("stage") or "运行失败"),
        progress_percent=percent,
        completed_count=completed,
        total_count=total,
        error=str(status.get("error_summary") or status.get("stderr_tail") or "运行失败")[-1000:],
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


def monitor_item(item: dict[str, Any]) -> None:
    from subtitle_frontend import run_status, stop_processing

    item_id = str(item["id"])
    payload = status_payload(item)
    if payload is None:
        queue_store.update_item(
            item_id,
            status="queued",
            stage="恢复等待",
            pid=0,
            error="worker 在分析阶段退出，已安全重新排队",
        )
        return

    while True:
        latest = queue_store.item(item_id)
        if latest is None:
            return
        if latest.get("cancel_requested"):
            stop_processing(payload)
        status = run_status(payload)
        completed, total, percent = _progress(status)
        queue_store.update_worker(
            pid=os.getpid(),
            status="running",
            active_item_id=item_id,
        )
        queue_store.update_item(
            item_id,
            pid=int(status.get("pid") or payload.get("pid") or 0),
            stage=str(status.get("stage") or "运行中"),
            completed_count=completed,
            total_count=total,
            progress_percent=percent,
            output_dir=str(status.get("output_dir") or latest.get("output_dir") or ""),
        )
        if not status.get("running"):
            _finish_item(item_id, status)
            return
        time.sleep(POLL_SECONDS)


def start_item(item: dict[str, Any]) -> None:
    from subtitle_frontend import start_processing

    item_id = str(item["id"])
    try:
        payload, analysis = prepare_payload(item)
        queue_store.update_item(
            item_id,
            prepared_payload=payload,
            series_name=payload["series_name"],
            movie_name=payload["movie_name"],
            output_root=payload["output_root"],
            output_dir=str(analysis.get("output_dir") or ""),
            total_count=analysis.get("total_count"),
            completed_count=int(analysis.get("completed_count") or 0),
            stage="启动处理",
        )
        started = start_processing(payload)
        queue_store.update_item(
            item_id,
            pid=int(started["pid"]),
            output_dir=str(started.get("output_dir") or ""),
            stage="启动中",
        )
        current = queue_store.item(item_id)
        if current is not None:
            monitor_item(current)
    except Exception as exc:
        queue_store.update_item(
            item_id,
            status="failed",
            stage="启动失败",
            pid=0,
            error=str(exc),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )


def recover_running_items() -> None:
    for item in queue_store.running_items():
        monitor_item(item)


def main() -> None:
    try:
        worker_lock = FileMutex(WORKER_LOCK_PATH, timeout=0.1)
        worker_lock.__enter__()
    except TimeoutError:
        return
    try:
        queue_store.update_worker(
            pid=os.getpid(),
            status="recovering",
            active_item_id="",
        )
        recover_running_items()
        idle_since = time.monotonic()
        while True:
            snapshot = queue_store.snapshot()
            if snapshot.get("paused"):
                queue_store.update_worker(
                    pid=os.getpid(),
                    status="paused",
                    active_item_id="",
                )
                time.sleep(POLL_SECONDS)
                if time.monotonic() - idle_since >= IDLE_EXIT_SECONDS:
                    break
                continue

            item = queue_store.claim_next(os.getpid())
            if item is None:
                queue_store.update_worker(
                    pid=os.getpid(),
                    status="idle",
                    active_item_id="",
                )
                time.sleep(POLL_SECONDS)
                if time.monotonic() - idle_since >= IDLE_EXIT_SECONDS:
                    break
                continue

            idle_since = time.monotonic()
            start_item(item)
            queue_store.update_worker(
                pid=os.getpid(),
                status="idle",
                active_item_id="",
            )
    finally:
        queue_store.update_worker(
            pid=0,
            status="stopped",
            active_item_id="",
        )
        worker_lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
