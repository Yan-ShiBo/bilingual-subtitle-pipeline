from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from frontend_settings import settings_store
from ssh_tunnel import SSHTunnelManager


BRIDGE_APP = "bilingual-subtitle-remote-bridge"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
BRIDGE_SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(__file__).with_name("ssh_tunnel.py").resolve(),
    Path(__file__).with_name("frontend_settings.py").resolve(),
    Path(__file__).with_name("process_lock.py").resolve(),
)


def bridge_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(BRIDGE_SOURCE_FILES, key=lambda item: item.name.casefold()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


BRIDGE_SOURCE_SHA256 = bridge_source_fingerprint()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_loopback_host(value: str) -> bool:
    raw = str(value or "").strip().casefold()
    if raw in LOOPBACK_HOSTS:
        return True
    try:
        hostname = urlparse(f"//{raw}").hostname
    except ValueError:
        return False
    return bool(hostname and hostname.casefold() in LOOPBACK_HOSTS)


def remote_config_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        key: config.get(key)
        for key in (
            "host",
            "port",
            "user",
            "name",
            "auth_method",
            "key_filename",
            "remember_password",
        )
    }
    payload["password_available"] = bool(config.get("password"))
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class BridgeRuntime:
    def __init__(self, proxy_port: int, status_path: Path) -> None:
        self.proxy_port = int(proxy_port)
        self.status_path = status_path
        self.tunnel = SSHTunnelManager(local_port=0)
        self.lock = threading.RLock()
        self.status_write_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reload_event = threading.Event()
        self.connected_event = threading.Event()
        self.config: dict[str, Any] = {}
        self.config_fingerprint = ""
        self.last_error = ""
        self.last_attempt_at = ""
        self.last_connected_at = ""
        self.last_probe_at = ""
        self.reconnect_count = 0
        self.models: list[str] = []
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self) -> None:
        self._load_config()
        self.monitor_thread.start()
        self.write_status()

    def stop(self) -> None:
        self.stop_event.set()
        self.reload_event.set()
        with self.lock:
            self.tunnel.disconnect()
            self.connected_event.clear()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)
        self.write_status()

    def request_reload(self) -> None:
        self.reload_event.set()

    def _load_config(self) -> None:
        loaded = settings_store.load()
        remote = loaded.get("remote") if isinstance(loaded, dict) else {}
        self.config = dict(remote) if isinstance(remote, dict) else {}
        self.config_fingerprint = remote_config_fingerprint(self.config)

    def _disconnect(self, error: str = "") -> None:
        with self.lock:
            self.tunnel.disconnect()
            self.connected_event.clear()
            if error:
                self.last_error = error
        self.write_status()

    def _connect_once(self) -> bool:
        with self.lock:
            config = dict(self.config)
        host = str(config.get("host") or "").strip()
        if not host:
            self.last_error = "尚未保存远程服务器配置。"
            self.write_status()
            return False

        self.last_attempt_at = utc_now()
        connected = self.tunnel.connect(
            host,
            int(config.get("port") or 22),
            str(config.get("user") or ""),
            str(config.get("password") or ""),
            str(config.get("name") or "Remote"),
            str(config.get("auth_method") or ""),
            str(config.get("key_filename") or ""),
        )
        if not connected:
            self.last_error = self.tunnel.last_error or "SSH 隧道连接失败。"
            self.connected_event.clear()
            self.write_status()
            return False

        self.connected_event.set()
        self.last_connected_at = utc_now()
        self.last_error = ""
        self.reconnect_count += 1
        self.models = self.tunnel.fetch_models()
        self.write_status()
        return True

    def _probe(self) -> bool:
        status = self.tunnel.status()
        if not status.get("connected") or not status.get("local_port"):
            return False
        request = Request(
            f"http://127.0.0.1:{int(status['local_port'])}/api/version",
            method="GET",
        )
        try:
            with urlopen(request, timeout=8) as response:
                response.read(1024)
            self.last_probe_at = utc_now()
            return True
        except Exception as exc:
            self.last_error = f"远程 Ollama 健康检查失败：{exc}"
            return False

    def _monitor(self) -> None:
        backoff = 1.0
        next_probe = 0.0
        while not self.stop_event.is_set():
            if self.reload_event.is_set():
                self.reload_event.clear()
                self._disconnect()
                self._load_config()
                backoff = 1.0

            if not self.connected_event.is_set():
                if self._connect_once():
                    backoff = 1.0
                    next_probe = time.monotonic() + 10.0
                else:
                    self.stop_event.wait(backoff)
                    backoff = min(30.0, backoff * 2.0)
                continue

            if time.monotonic() >= next_probe:
                if not self._probe():
                    self._disconnect(self.last_error)
                    backoff = 1.0
                    continue
                self.write_status()
                next_probe = time.monotonic() + 10.0
            self.stop_event.wait(1.0)

    def wait_connected(self, timeout: float) -> bool:
        if self.connected_event.is_set() and self.tunnel.status().get("connected"):
            return True
        return self.connected_event.wait(timeout=max(0.0, timeout))

    def proxy_request(
        self,
        method: str,
        path: str,
        body: bytes,
        content_type: str,
    ) -> tuple[int, str, bytes]:
        wait_timeout = float(os.environ.get("SUBTITLE_BRIDGE_WAIT_TIMEOUT", "180"))
        max_attempts = max(1, int(os.environ.get("SUBTITLE_BRIDGE_MAX_PROXY_ATTEMPTS", "3")))
        deadline = time.monotonic() + wait_timeout
        request_attempts = 0
        last_error = "远程服务器尚未连接。"
        while time.monotonic() < deadline and not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if not self.wait_connected(min(5.0, remaining)):
                last_error = self.last_error or last_error
                continue
            tunnel_status = self.tunnel.status()
            local_port = tunnel_status.get("local_port")
            if not local_port:
                self._disconnect("SSH 隧道没有本地端口。")
                continue
            request = Request(
                f"http://127.0.0.1:{int(local_port)}{path}",
                data=body if method != "GET" else None,
                headers={"Content-Type": content_type} if content_type else {},
                method=method,
            )
            try:
                timeout = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "600"))
                with urlopen(request, timeout=timeout) as response:
                    return (
                        int(response.status),
                        str(response.headers.get("Content-Type") or "application/json"),
                        response.read(),
                    )
            except HTTPError as exc:
                return (
                    int(exc.code),
                    str(exc.headers.get("Content-Type") or "application/json"),
                    exc.read(),
                )
            except (URLError, OSError, TimeoutError, socket.timeout) as exc:
                request_attempts += 1
                last_error = str(exc)
                self._disconnect(f"远程请求中断，正在重连：{exc}")
                self.reload_event.set()
                if request_attempts >= max_attempts:
                    break
                deadline = time.monotonic() + wait_timeout
                time.sleep(0.5)

        payload = json.dumps(
            {"error": f"远程服务器在等待期内未恢复：{last_error}"},
            ensure_ascii=False,
        ).encode("utf-8")
        return 503, "application/json; charset=utf-8", payload

    def status(self) -> dict[str, Any]:
        tunnel_status = self.tunnel.status()
        connected = bool(self.connected_event.is_set() and tunnel_status.get("connected"))
        return {
            "app": BRIDGE_APP,
            "source_sha256": BRIDGE_SOURCE_SHA256,
            "pid": os.getpid(),
            "proxy_port": self.proxy_port,
            "connected": connected,
            "status": "connected" if connected else "reconnecting",
            "error": self.last_error or tunnel_status.get("error") or "",
            "name": str(self.config.get("name") or tunnel_status.get("name") or "Remote"),
            "resolved_host": tunnel_status.get("resolved_host") or "",
            "resolved_user": tunnel_status.get("resolved_user") or "",
            "auth_method": str(self.config.get("auth_method") or tunnel_status.get("auth_method") or ""),
            "key_source": tunnel_status.get("key_source") or "",
            "local_tunnel_port": tunnel_status.get("local_port") if connected else None,
            "models": list(self.models),
            "config_fingerprint": self.config_fingerprint,
            "last_attempt_at": self.last_attempt_at,
            "last_connected_at": self.last_connected_at,
            "last_probe_at": self.last_probe_at,
            "reconnect_count": self.reconnect_count,
            "checked_at": utc_now(),
        }

    def write_status(self) -> None:
        with self.status_write_lock:
            payload = self.status()
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.status_path.with_name(
                f".{self.status_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                for attempt in range(5):
                    try:
                        temporary.replace(self.status_path)
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(0.02 * (attempt + 1))
            finally:
                temporary.unlink(missing_ok=True)


class BridgeHandler(BaseHTTPRequestHandler):
    runtime: BridgeRuntime

    def _reject_nonlocal(self) -> bool:
        if is_loopback_host(self.headers.get("Host", "")):
            return False
        self._send_json({"error": "Bridge only accepts localhost requests."}, status=403)
        return True

    def do_GET(self) -> None:
        if self._reject_nonlocal():
            return
        if self.path == "/bridge/health":
            self._send_json(self.runtime.status())
            return
        self._proxy("GET")

    def do_POST(self) -> None:
        if self._reject_nonlocal():
            return
        if self.path == "/bridge/reload":
            self.runtime.request_reload()
            self._send_json({"status": "reloading"})
            return
        self._proxy("POST")

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._send_json({"error": "Request body is too large."}, status=413)
            return
        body = self.rfile.read(length) if length else b""
        status, content_type, response_body = self.runtime.proxy_request(
            method,
            self.path,
            body,
            str(self.headers.get("Content-Type") or "application/json"),
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Resilient local bridge to a remote Ollama server.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()

    runtime = BridgeRuntime(args.port, args.status_file)
    BridgeHandler.runtime = runtime
    server = ThreadingHTTPServer(("127.0.0.1", args.port), BridgeHandler)
    runtime.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        runtime.stop()


if __name__ == "__main__":
    main()
