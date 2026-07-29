import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssh_tunnel import SSHTunnelManager, normalize_auth_method, resolve_ssh_connection  # noqa: E402
from subtitle_frontend import html_page  # noqa: E402


class SshTunnelTests(unittest.TestCase):
    def test_tunnel_falls_back_to_a_free_port_when_preferred_port_is_busy(self) -> None:
        class FakeTransport:
            @staticmethod
            def is_active() -> bool:
                return False

        class FakeClient:
            @staticmethod
            def get_transport():
                return FakeTransport()

        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        preferred_port = int(occupied.getsockname()[1])
        manager = SSHTunnelManager(local_port=preferred_port)
        manager.ssh_client = FakeClient()

        try:
            manager._forward_local_port()
        finally:
            occupied.close()

        self.assertNotEqual(manager.local_port, preferred_port)
        self.assertGreater(manager.local_port, 0)
        self.assertTrue(manager._tunnel_ready.is_set())

    def test_resolve_ssh_connection_uses_openssh_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            key_path = root / "id_test"
            key_path.write_text("placeholder", encoding="utf-8")
            config_path = root / "config"
            config_path.write_text(
                "\n".join(
                    [
                        "Host ai-server 10.12.96.203",
                        "    HostName 10.12.96.203",
                        "    User csynth",
                        f"    IdentityFile {key_path}",
                        "    IdentitiesOnly yes",
                        "    ServerAliveInterval 25",
                    ]
                ),
                encoding="utf-8",
            )

            resolved = resolve_ssh_connection(
                host="10.12.96.203",
                port=22,
                user="",
                ssh_config_path=config_path,
            )

        self.assertEqual(resolved["host"], "10.12.96.203")
        self.assertEqual(resolved["user"], "csynth")
        self.assertEqual(resolved["identity_files"], [str(key_path.resolve())])
        self.assertTrue(resolved["identities_only"])
        self.assertEqual(resolved["server_alive_interval"], 25)

    def test_explicit_key_path_overrides_ssh_config_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_key = root / "config_key"
            explicit_key = root / "explicit_key"
            config_path = root / "config"
            config_path.write_text(
                f"Host ai-server\n    IdentityFile {config_key}\n",
                encoding="utf-8",
            )

            resolved = resolve_ssh_connection(
                host="ai-server",
                key_filename=str(explicit_key),
                ssh_config_path=config_path,
            )

        self.assertEqual(resolved["identity_files"], [str(explicit_key.resolve())])

    def test_legacy_password_payload_remains_compatible(self) -> None:
        self.assertEqual(normalize_auth_method("", "secret"), "password")
        self.assertEqual(normalize_auth_method("", ""), "key")
        self.assertEqual(normalize_auth_method("key", "ignored"), "key")
        self.assertEqual(normalize_auth_method("password", "secret"), "password")

    def test_frontend_defaults_to_key_auth_for_ten_network_server(self) -> None:
        page = html_page()

        self.assertIn('id="remoteHost" value="10.12.96.203"', page)
        self.assertIn('id="remoteUser" value="csynth"', page)
        self.assertIn('<option value="key" selected>SSH 密钥</option>', page)
        self.assertIn('id="remotePasswordOptions" hidden', page)
        self.assertIn("updateRemoteAuthVisibility()", page)
        self.assertIn("auth_method: document.getElementById('remoteAuthMethod').value", page)
        self.assertIn("key_filename: document.getElementById('remoteKeyPath').value", page)
        self.assertIn("select.replaceChildren(new Option('[Local] qwen3:14b'", page)
        self.assertIn("select.add(new Option(`[Remote: ${safeRemoteName}] ${modelName}`", page)
        self.assertNotIn("html += `<option value=\"remote:", page)


if __name__ == "__main__":
    unittest.main()
