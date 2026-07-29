import json
import os
import select
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

import paramiko


DEFAULT_SSH_CONFIG_PATH = Path.home() / ".ssh" / "config"


def normalize_auth_method(auth_method: str, password: str) -> str:
    value = str(auth_method or "").strip().lower()
    if value in {"key", "password"}:
        return value
    return "password" if password else "key"


def _expand_key_path(value: str) -> str:
    return str(Path(os.path.expandvars(os.path.expanduser(value))).resolve())


def resolve_ssh_connection(
    host: str,
    port: int = 22,
    user: str = "",
    key_filename: str = "",
    ssh_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    requested_host = str(host or "").strip()
    if not requested_host:
        raise ValueError("SSH 主机不能为空。")

    config_path = ssh_config_path or DEFAULT_SSH_CONFIG_PATH
    config_values: Dict[str, Any] = {}
    if config_path.exists():
        config = paramiko.SSHConfig()
        with config_path.open("r", encoding="utf-8") as handle:
            config.parse(handle)
        config_values = config.lookup(requested_host)

    configured_identities = config_values.get("identityfile") or []
    if isinstance(configured_identities, str):
        configured_identities = [configured_identities]
    identity_files = [key_filename] if str(key_filename or "").strip() else configured_identities
    identity_files = [_expand_key_path(str(path)) for path in identity_files if str(path or "").strip()]

    resolved_port = int(port or config_values.get("port") or 22)
    return {
        "requested_host": requested_host,
        "host": str(config_values.get("hostname") or requested_host),
        "port": resolved_port,
        "user": str(user or config_values.get("user") or "").strip(),
        "identity_files": identity_files,
        "identities_only": str(config_values.get("identitiesonly") or "no").lower() == "yes",
        "server_alive_interval": int(config_values.get("serveraliveinterval") or 30),
        "server_alive_count_max": int(config_values.get("serveralivecountmax") or 3),
    }


class SSHTunnelManager:
    def __init__(self, local_port: int = 11435) -> None:
        self.ssh_client: Optional[paramiko.SSHClient] = None
        self.tunnel_thread: Optional[threading.Thread] = None
        self.local_port = local_port
        self.remote_host = "127.0.0.1"
        self.remote_port = 11434
        self.running = False
        self.last_error = ""
        self.remote_name = ""
        self.auth_method = ""
        self.resolved_host = ""
        self.resolved_user = ""
        self.key_source = ""
        self._server_socket: Optional[socket.socket] = None
        self._tunnel_ready = threading.Event()

    def connect(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        name: str,
        auth_method: str = "",
        key_filename: str = "",
    ) -> bool:
        self.disconnect()
        self.remote_name = str(name or "remote")
        self.auth_method = normalize_auth_method(auth_method, password)
        self._tunnel_ready.clear()

        try:
            connection = resolve_ssh_connection(host, port, user, key_filename)
            self.resolved_host = connection["host"]
            self.resolved_user = connection["user"]
            if not self.resolved_user:
                raise ValueError("SSH 用户名为空；请填写用户名或在 ~/.ssh/config 中配置 User。")

            connect_kwargs: Dict[str, Any] = {
                "hostname": self.resolved_host,
                "port": connection["port"],
                "username": self.resolved_user,
                "timeout": 10,
                "banner_timeout": 10,
                "auth_timeout": 15,
            }
            if self.auth_method == "password":
                if not password:
                    raise ValueError("密码认证已选中，但密码为空。")
                connect_kwargs.update(
                    {
                        "password": password,
                        "allow_agent": False,
                        "look_for_keys": False,
                    }
                )
                self.key_source = ""
            else:
                identity_files = connection["identity_files"]
                missing_files = [path for path in identity_files if not Path(path).is_file()]
                if missing_files:
                    raise FileNotFoundError(f"SSH 密钥文件不存在：{missing_files[0]}")
                connect_kwargs.update(
                    {
                        "password": None,
                        "key_filename": identity_files or None,
                        "allow_agent": not connection["identities_only"],
                        "look_for_keys": not connection["identities_only"],
                    }
                )
                self.key_source = identity_files[0] if identity_files else "SSH agent/default keys"

            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect(**connect_kwargs)
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                client.close()
                raise RuntimeError("SSH 已认证，但连接未保持活动状态。")
            transport.set_keepalive(max(1, connection["server_alive_interval"]))

            self.ssh_client = client
            self.running = True
            self.tunnel_thread = threading.Thread(target=self._forward_local_port, daemon=True)
            self.tunnel_thread.start()
            if not self._tunnel_ready.wait(timeout=5) or not self.running:
                error = self.last_error or "SSH 隧道未能在 5 秒内启动。"
                self.disconnect()
                raise RuntimeError(error)

            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.running = False
            if self.ssh_client:
                self.ssh_client.close()
                self.ssh_client = None
            return False

    def disconnect(self) -> None:
        self.running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
        if self.tunnel_thread and self.tunnel_thread.is_alive():
            self.tunnel_thread.join(timeout=2)
        self.tunnel_thread = None

    def status(self) -> Dict[str, Any]:
        transport = self.ssh_client.get_transport() if self.ssh_client else None
        connected = bool(transport and transport.is_active() and self.running)
        return {
            "connected": connected,
            "error": self.last_error,
            "name": self.remote_name,
            "local_port": self.local_port if connected else None,
            "auth_method": self.auth_method,
            "resolved_host": self.resolved_host,
            "resolved_user": self.resolved_user,
            "key_source": self.key_source if self.auth_method == "key" else "",
        }

    def fetch_models(self) -> list[str]:
        if not self.status()["connected"]:
            return []

        for _attempt in range(5):
            try:
                req = Request(f"http://127.0.0.1:{self.local_port}/api/tags", method="GET")
                with urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    self.last_error = ""
                    return [model["name"] for model in data.get("models", [])]
            except Exception as exc:
                self.last_error = f"Fetch models error: {exc}"
                time.sleep(1)
        return []

    def _forward_local_port(self) -> None:
        transport = self.ssh_client.get_transport() if self.ssh_client else None
        if transport is None:
            self.last_error = "SSH transport is unavailable."
            self.running = False
            self._tunnel_ready.set()
            return

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("127.0.0.1", self.local_port))
            server.listen(100)
            server.settimeout(1.0)
            self._server_socket = server
            self._tunnel_ready.set()
        except Exception as exc:
            server.close()
            self.last_error = f"Local bind error: {exc}"
            self.running = False
            self._tunnel_ready.set()
            return

        try:
            while self.running and transport.is_active():
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                try:
                    channel = transport.open_channel(
                        "direct-tcpip",
                        (self.remote_host, self.remote_port),
                        address,
                    )
                except Exception:
                    client.close()
                    continue
                if channel is None:
                    client.close()
                    continue

                threading.Thread(
                    target=self._handle_connection,
                    args=(client, channel),
                    daemon=True,
                ).start()
        finally:
            try:
                server.close()
            except OSError:
                pass
            self._server_socket = None

    def _handle_connection(self, client_sock: socket.socket, channel: paramiko.Channel) -> None:
        try:
            while self.running:
                readable, _writable, _exceptional = select.select([client_sock, channel], [], [], 1.0)
                if client_sock in readable:
                    data = client_sock.recv(32768)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(32768)
                    if not data:
                        break
                    client_sock.sendall(data)
        finally:
            channel.close()
            client_sock.close()


tunnel_manager = SSHTunnelManager()
