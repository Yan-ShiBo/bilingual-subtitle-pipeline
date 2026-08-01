from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from process_lock import FileMutex
from remote_ollama_bridge import BRIDGE_SOURCE_SHA256


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
BRIDGE_DIR = PROJECT_ROOT / "runtime" / "remote-bridge"
BRIDGE_STATE_PATH = BRIDGE_DIR / f"bridge-state-{BRIDGE_SOURCE_SHA256[:12]}.json"
BRIDGE_LOCK_PATH = BRIDGE_DIR / "bridge-start.lock"
BRIDGE_STDOUT_PATH = BRIDGE_DIR / "bridge.stdout.log"
BRIDGE_STDERR_PATH = BRIDGE_DIR / "bridge.stderr.log"
BRIDGE_APP = "bilingual-subtitle-remote-bridge"
BRIDGE_PORT_RANGE = range(11436, 11456)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _http_json(url: str, *, method: str = "GET", timeout: float = 2.0) -> dict[str, Any]:
    request = Request(url, data=b"{}" if method == "POST" else None, method=method)
    if method == "POST":
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Bridge returned a non-object response")
    return data


def bridge_status() -> dict[str, Any]:
    state = _read_json(BRIDGE_STATE_PATH)
    port = int(state.get("proxy_port") or 0)
    if port <= 0:
        return {
            "app": BRIDGE_APP,
            "running": False,
            "connected": False,
            "status": "stopped",
            "error": str(state.get("error") or ""),
        }
    try:
        health = _http_json(f"http://127.0.0.1:{port}/bridge/health")
    except Exception as exc:
        return {
            **state,
            "app": BRIDGE_APP,
            "running": False,
            "connected": False,
            "status": "stopped",
            "error": f"Bridge process is unavailable: {exc}",
        }
    if health.get("app") != BRIDGE_APP:
        return {
            "app": BRIDGE_APP,
            "running": False,
            "connected": False,
            "status": "conflict",
            "error": f"Port {port} is not the subtitle remote bridge.",
        }
    if health.get("source_sha256") != BRIDGE_SOURCE_SHA256:
        return {
            **state,
            "app": BRIDGE_APP,
            "running": False,
            "connected": False,
            "status": "stale",
            "error": "Bridge process is running older source code.",
        }
    health["running"] = True
    health["base_url"] = f"http://127.0.0.1:{port}"
    return health


def _port_is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _choose_port() -> int:
    for port in BRIDGE_PORT_RANGE:
        if _port_is_free(port):
            return port
    raise RuntimeError("No free remote bridge port found in range 11436-11455.")


def _start_bridge_process(port: int) -> None:
    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    stdout = BRIDGE_STDOUT_PATH.open("a", encoding="utf-8")
    stderr = BRIDGE_STDERR_PATH.open("a", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(
            [
                sys.executable,
                str(APP_DIR / "remote_ollama_bridge.py"),
                "--port",
                str(port),
                "--status-file",
                str(BRIDGE_STATE_PATH),
            ],
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


def ensure_remote_bridge(
    *,
    wait_for_connection: bool = True,
    timeout: float = 25.0,
    reload_config: bool = False,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, float(timeout))
    with FileMutex(BRIDGE_LOCK_PATH, timeout=min(10.0, timeout)):
        status = bridge_status()
        if not status.get("running"):
            port = _choose_port()
            _start_bridge_process(port)
        elif reload_config:
            try:
                _http_json(
                    f"{status['base_url']}/bridge/reload",
                    method="POST",
                    timeout=2,
                )
                time.sleep(0.25)
            except (OSError, URLError, ValueError):
                pass

    last_status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_status = bridge_status()
        if last_status.get("running") and (
            last_status.get("connected") or not wait_for_connection
        ):
            return last_status
        time.sleep(0.25)
    if last_status.get("running") and not wait_for_connection:
        return last_status
    error = last_status.get("error") or "Remote bridge did not become ready before timeout."
    raise RuntimeError(str(error))


def remote_bridge_base_url(*, allow_reconnecting: bool = True) -> str:
    status = ensure_remote_bridge(
        wait_for_connection=not allow_reconnecting,
        timeout=25,
    )
    base_url = str(status.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("Remote bridge did not publish a local URL.")
    return base_url
