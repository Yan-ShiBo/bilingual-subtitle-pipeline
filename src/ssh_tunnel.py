import threading
import time
import socket
import select
from typing import Optional, Dict, Any
import paramiko
import json
from urllib.request import Request, urlopen

class SSHTunnelManager:
    def __init__(self):
        self.ssh_client: Optional[paramiko.SSHClient] = None
        self.tunnel_thread: Optional[threading.Thread] = None
        self.local_port: int = 11435
        self.remote_host: str = "127.0.0.1"
        self.remote_port: int = 11434
        self.running: bool = False
        self.last_error: str = ""
        self.remote_name: str = ""

    def connect(self, host: str, port: int, user: str, password: str, name: str) -> bool:
        self.disconnect()
        self.remote_name = name
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh_client.connect(host, port, user, password, timeout=10)
            self.running = True
            self.tunnel_thread = threading.Thread(target=self._forward_local_port, daemon=True)
            self.tunnel_thread.start()
            self.last_error = ""
            return True
        except Exception as e:
            self.last_error = str(e)
            self.ssh_client = None
            return False

    def disconnect(self):
        self.running = False
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None

    def status(self) -> Dict[str, Any]:
        return {
            "connected": self.ssh_client is not None and self.ssh_client.get_transport() is not None and self.ssh_client.get_transport().is_active(),
            "error": self.last_error,
            "name": self.remote_name,
            "local_port": self.local_port if self.running else None
        }

    def fetch_models(self) -> list:
        if not self.status()["connected"]:
            return []
        
        for attempt in range(5):
            try:
                req = Request("http://127.0.0.1:11435/api/tags", method="GET")
                with urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return [m["name"] for m in data.get("models", [])]
            except Exception as e:
                self.last_error = f"Fetch models error: {e}"
                time.sleep(1)
        return []

    def _forward_local_port(self):
        class ForwardServer(socket.socket):
            def __init__(self, ssh_transport, local_port, remote_host, remote_port):
                super().__init__(socket.AF_INET, socket.SOCK_STREAM)
                self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.bind(("127.0.0.1", local_port))
                self.listen(100)
                self.ssh_transport = ssh_transport
                self.remote_host = remote_host
                self.remote_port = remote_port
                self.settimeout(1.0)

        if not self.ssh_client or not self.ssh_client.get_transport():
            return

        transport = self.ssh_client.get_transport()
        
        try:
            server = ForwardServer(transport, self.local_port, self.remote_host, self.remote_port)
        except Exception as e:
            self.last_error = f"Local bind error: {e}"
            return

        while self.running and transport.is_active():
            try:
                client, addr = server.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            
            try:
                chan = transport.open_channel("direct-tcpip", (self.remote_host, self.remote_port), addr)
            except Exception as e:
                client.close()
                continue
                
            if chan is None:
                client.close()
                continue

            # Handle the connection in a new thread
            t = threading.Thread(target=self._handle_connection, args=(client, chan), daemon=True)
            t.start()
            
        server.close()

    def _handle_connection(self, client_sock: socket.socket, chan: paramiko.Channel):
        while self.running:
            r, w, x = select.select([client_sock, chan], [], [], 1.0)
            if client_sock in r:
                try:
                    data = client_sock.recv(1024)
                    if not data:
                        break
                    chan.send(data)
                except Exception:
                    break
            if chan in r:
                try:
                    data = chan.recv(1024)
                    if not data:
                        break
                    client_sock.send(data)
                except Exception:
                    break
        chan.close()
        client_sock.close()

tunnel_manager = SSHTunnelManager()
