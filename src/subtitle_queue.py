from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from process_lock import FileMutex


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
QUEUE_DIR = PROJECT_ROOT / "runtime" / "queue"
QUEUE_PATH = QUEUE_DIR / "subtitle-queue.json"
QUEUE_LOCK_PATH = QUEUE_DIR / "subtitle-queue.lock"
WORKER_LOCK_PATH = QUEUE_DIR / "worker.lock"
WORKER_START_LOCK_PATH = QUEUE_DIR / "worker-start.lock"
WORKER_STDOUT_PATH = QUEUE_DIR / "worker.stdout.log"
WORKER_STDERR_PATH = QUEUE_DIR / "worker.stderr.log"
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m2ts", ".ts", ".mov", ".wmv"}
ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"success", "failed", "cancelled"}
PATH_SPECIFIC_FIELDS = {
    "subtitle_file",
    "chinese_subtitle_file",
    "english_subtitle_file",
}
QUEUE_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def natural_sort_key(value: str) -> list[tuple[int, int | str]]:
    parts = re.split(r"(\d+)", str(value).casefold())
    return [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    ]


def _default_state() -> dict[str, Any]:
    return {
        "version": QUEUE_VERSION,
        "paused": True,
        "items": [],
        "worker": {
            "pid": 0,
            "status": "stopped",
            "heartbeat_at": "",
            "active_item_id": "",
        },
        "updated_at": utc_now(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_state()
    if not isinstance(data, dict):
        return _default_state()
    default = _default_state()
    default.update(data)
    if not isinstance(default.get("items"), list):
        default["items"] = []
    if not isinstance(default.get("worker"), dict):
        default["worker"] = _default_state()["worker"]
    return default


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def normalize_episode_payload(base_payload: dict[str, Any], path: Path) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["path"] = str(path)
    payload["movie_name"] = ""
    payload["run_mode"] = "resume"
    payload["restart"] = False
    payload.pop("pid", None)
    for field in PATH_SPECIFIC_FIELDS:
        payload[field] = ""
    return payload


def discover_episode_files(selected: str | Path) -> list[dict[str, Any]]:
    path = Path(selected).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    root = path if path.is_dir() else path.parent
    candidates: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.casefold() not in VIDEO_EXTENSIONS:
            continue
        try:
            relative_parts = candidate.relative_to(root).parts[:-1]
        except ValueError:
            relative_parts = ()
        if any(part.casefold() in {"runtime", "outputs", ".git", "__pycache__"} for part in relative_parts):
            continue
        candidates.append(candidate.resolve())

    candidates = sorted(
        set(candidates),
        key=lambda item: natural_sort_key(str(item.relative_to(root.resolve()))),
    )
    return [
        {
            "path": str(candidate),
            "name": candidate.name,
            "relative_path": str(candidate.relative_to(root.resolve())),
            "size_mb": round(candidate.stat().st_size / 1024 / 1024, 1),
        }
        for candidate in candidates
    ]


class SubtitleQueueStore:
    def __init__(self, path: Path = QUEUE_PATH, lock_path: Path = QUEUE_LOCK_PATH) -> None:
        self.path = path
        self.lock_path = lock_path

    def _mutate(self, callback: Callable[[dict[str, Any]], Any]) -> tuple[dict[str, Any], Any]:
        with FileMutex(self.lock_path, timeout=10):
            state = _read_json(self.path)
            result = callback(state)
            state["version"] = QUEUE_VERSION
            state["updated_at"] = utc_now()
            _write_json_atomic(self.path, state)
            return copy.deepcopy(state), result

    def snapshot(self) -> dict[str, Any]:
        with FileMutex(self.lock_path, timeout=10):
            state = copy.deepcopy(_read_json(self.path))
        items = state.get("items") or []
        counts = {
            status: sum(1 for item in items if item.get("status") == status)
            for status in (*sorted(ACTIVE_STATUSES), *sorted(TERMINAL_STATUSES))
        }
        worker = state.get("worker") or {}
        worker_pid = int(worker.get("pid") or 0)
        worker["running"] = process_is_running(worker_pid)
        state["worker"] = worker
        state["summary"] = {
            "total": len(items),
            **counts,
        }
        return state

    def enqueue(self, paths: list[str], base_payload: dict[str, Any]) -> dict[str, Any]:
        normalized: list[Path] = []
        seen: set[str] = set()
        for raw in paths:
            path = Path(str(raw or "")).expanduser().resolve()
            if not path.is_file() or path.suffix.casefold() not in VIDEO_EXTENSIONS:
                raise ValueError(f"Unsupported episode video: {path}")
            key = os.path.normcase(str(path))
            if key not in seen:
                seen.add(key)
                normalized.append(path)

        def add(state: dict[str, Any]) -> list[str]:
            existing = {
                os.path.normcase(str(Path(str(item.get("path") or "")).expanduser()))
                for item in state["items"]
                if item.get("status") in ACTIVE_STATUSES
            }
            added: list[str] = []
            for path in normalized:
                if os.path.normcase(str(path)) in existing:
                    continue
                item_id = uuid.uuid4().hex
                state["items"].append(
                    {
                        "id": item_id,
                        "path": str(path),
                        "name": path.name,
                        "status": "queued",
                        "stage": "等待处理",
                        "completed_count": 0,
                        "total_count": None,
                        "progress_percent": 0,
                        "attempts": 0,
                        "pid": 0,
                        "cancel_requested": False,
                        "payload": normalize_episode_payload(base_payload, path),
                        "prepared_payload": None,
                        "created_at": utc_now(),
                        "started_at": "",
                        "finished_at": "",
                        "updated_at": utc_now(),
                        "error": "",
                    }
                )
                existing.add(os.path.normcase(str(path)))
                added.append(item_id)
            return added

        state, added_ids = self._mutate(add)
        state["added_ids"] = added_ids
        return state

    def set_paused(self, paused: bool) -> dict[str, Any]:
        state, _ = self._mutate(lambda value: value.__setitem__("paused", bool(paused)))
        return state

    def update_worker(self, **updates: Any) -> dict[str, Any]:
        def apply(state: dict[str, Any]) -> None:
            worker = state.setdefault("worker", {})
            worker.update(updates)
            worker["heartbeat_at"] = utc_now()

        state, _ = self._mutate(apply)
        return state

    def claim_next(self, worker_pid: int) -> dict[str, Any] | None:
        def claim(state: dict[str, Any]) -> dict[str, Any] | None:
            if state.get("paused"):
                return None
            for item in state["items"]:
                if item.get("status") != "queued":
                    continue
                item.update(
                    {
                        "status": "running",
                        "stage": "分析字幕来源",
                        "attempts": int(item.get("attempts") or 0) + 1,
                        "started_at": utc_now(),
                        "finished_at": "",
                        "updated_at": utc_now(),
                        "error": "",
                        "cancel_requested": False,
                    }
                )
                worker = state.setdefault("worker", {})
                worker.update(
                    {
                        "pid": int(worker_pid),
                        "status": "running",
                        "active_item_id": item["id"],
                        "heartbeat_at": utc_now(),
                    }
                )
                return copy.deepcopy(item)
            return None

        _state, item = self._mutate(claim)
        return item

    def update_item(self, item_id: str, **updates: Any) -> dict[str, Any]:
        def apply(state: dict[str, Any]) -> None:
            for item in state["items"]:
                if item.get("id") != item_id:
                    continue
                item.update(updates)
                item["updated_at"] = utc_now()
                return
            raise KeyError(f"Queue item not found: {item_id}")

        state, _ = self._mutate(apply)
        return state

    def running_items(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(item)
            for item in self.snapshot().get("items", [])
            if item.get("status") == "running"
        ]

    def item(self, item_id: str) -> dict[str, Any] | None:
        return next(
            (
                copy.deepcopy(item)
                for item in self.snapshot().get("items", [])
                if item.get("id") == item_id
            ),
            None,
        )

    def action(self, action: str, item_id: str = "", direction: int = 0) -> dict[str, Any]:
        action = str(action or "").strip().casefold()

        def apply(state: dict[str, Any]) -> None:
            items = state["items"]
            index = next((i for i, item in enumerate(items) if item.get("id") == item_id), -1)
            if action == "pause":
                state["paused"] = True
                return
            if action == "resume":
                state["paused"] = False
                return
            if index < 0:
                raise KeyError(f"Queue item not found: {item_id}")
            item = items[index]
            status = str(item.get("status") or "")
            if action == "cancel":
                if status == "running":
                    item["cancel_requested"] = True
                    item["stage"] = "正在终止"
                elif status == "queued":
                    item.update({"status": "cancelled", "stage": "已取消", "finished_at": utc_now()})
                return
            if action == "retry":
                if status == "running":
                    raise ValueError("Cannot retry a running queue item")
                item.update(
                    {
                        "status": "queued",
                        "stage": "等待继续",
                        "pid": 0,
                        "cancel_requested": False,
                        "error": "",
                        "finished_at": "",
                    }
                )
                return
            if action == "remove":
                if status == "running":
                    raise ValueError("Cannot remove a running queue item")
                items.pop(index)
                return
            if action == "move":
                if status == "running":
                    raise ValueError("Cannot move a running queue item")
                target = max(0, min(len(items) - 1, index + int(direction)))
                if target != index:
                    items.insert(target, items.pop(index))
                return
            raise ValueError(f"Unsupported queue action: {action}")

        state, _ = self._mutate(apply)
        return state


queue_store = SubtitleQueueStore()


def ensure_queue_worker() -> dict[str, Any]:
    snapshot = queue_store.snapshot()
    worker = snapshot.get("worker") or {}
    if worker.get("running"):
        return worker

    with FileMutex(WORKER_START_LOCK_PATH, timeout=10):
        snapshot = queue_store.snapshot()
        worker = snapshot.get("worker") or {}
        if worker.get("running"):
            return worker

        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        stdout = WORKER_STDOUT_PATH.open("a", encoding="utf-8")
        stderr = WORKER_STDERR_PATH.open("a", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                [sys.executable, str(APP_DIR / "subtitle_queue_worker.py")],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            stdout.close()
            stderr.close()
        queue_store.update_worker(
            pid=process.pid,
            status="starting",
            active_item_id="",
        )
        return {"pid": process.pid, "running": True, "status": "starting"}
