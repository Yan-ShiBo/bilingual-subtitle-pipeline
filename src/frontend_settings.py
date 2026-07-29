from __future__ import annotations

import base64
import ctypes
import json
import os
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Any


SETTINGS_VERSION = 1
MAX_FORM_FIELDS = 80
REMOTE_FIELDS = {
    "host",
    "port",
    "user",
    "name",
    "auth_method",
    "key_filename",
    "remember_password",
}


class DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def default_settings_path() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "BilingualSubtitlePipeline" / "frontend-settings.json"
    return Path.home() / ".config" / "bilingual-subtitle-pipeline" / "frontend-settings.json"


def _blob_from_bytes(value: bytes) -> tuple[DataBlob, Any]:
    buffer = ctypes.create_string_buffer(value)
    blob = DataBlob(
        len(value),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    return blob, buffer


def protect_password(password: str) -> str:
    if not password:
        return ""
    if os.name != "nt":
        raise RuntimeError("Password persistence requires Windows DPAPI")

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    input_blob, input_buffer = _blob_from_bytes(password.encode("utf-8"))
    output_blob = DataBlob()
    description = "Bilingual Subtitle Pipeline"
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        description,
        None,
        None,
        None,
        0x1,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    try:
        protected = ctypes.string_at(output_blob.pbData, output_blob.cbData)
        return base64.b64encode(protected).decode("ascii")
    finally:
        kernel32.LocalFree(output_blob.pbData)
        del input_buffer


def unprotect_password(value: str) -> str:
    if not value:
        return ""
    if os.name != "nt":
        return ""

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    protected = base64.b64decode(value)
    input_blob, input_buffer = _blob_from_bytes(protected)
    output_blob = DataBlob()
    description = ctypes.c_wchar_p()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        ctypes.byref(description),
        None,
        None,
        None,
        0x1,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData).decode("utf-8")
    finally:
        if description:
            kernel32.LocalFree(description)
        kernel32.LocalFree(output_blob.pbData)
        del input_buffer


def _safe_scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_form(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, Any] = {}
    for key, item in list(value.items())[:MAX_FORM_FIELDS]:
        name = str(key)[:80]
        scalar = _safe_scalar(item)
        if isinstance(scalar, str):
            scalar = scalar[:32_768]
        output[name] = scalar
    return output


def _sanitize_remote(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    output = {
        key: _safe_scalar(value.get(key))
        for key in REMOTE_FIELDS
        if key in value
    }
    output.pop("password", None)
    return output


class FrontendSettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_settings_path()

    def _read_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": SETTINGS_VERSION, "form": {}, "remote": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": SETTINGS_VERSION, "form": {}, "remote": {}}
        if not isinstance(data, dict):
            return {"version": SETTINGS_VERSION, "form": {}, "remote": {}}
        return data

    def _write_raw(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self) -> dict[str, Any]:
        raw = self._read_raw()
        remote = _sanitize_remote(raw.get("remote"))
        protected_password = str(raw.get("protected_password") or "")
        password = ""
        password_error = ""
        if protected_password:
            try:
                password = unprotect_password(protected_password)
            except Exception as exc:
                password_error = str(exc)
        remote["password"] = password
        remote["remember_password"] = bool(protected_password and password)
        result = {
            "version": SETTINGS_VERSION,
            "form": _sanitize_form(raw.get("form")),
            "remote": remote,
        }
        if password_error:
            result["password_error"] = password_error
        return result

    def save_form(self, form: Any) -> dict[str, Any]:
        raw = self._read_raw()
        raw["version"] = SETTINGS_VERSION
        raw["form"] = _sanitize_form(form)
        self._write_raw(raw)
        return self.load()

    def save_remote(self, remote: Any) -> dict[str, Any]:
        if not isinstance(remote, dict):
            raise ValueError("Remote settings must be an object")
        raw = self._read_raw()
        raw["version"] = SETTINGS_VERSION
        raw["remote"] = _sanitize_remote(remote)
        remember_password = (
            str(remote.get("auth_method") or "").lower() == "password"
            and bool(remote.get("remember_password"))
        )
        password = str(remote.get("password") or "")
        if remember_password and password:
            raw["protected_password"] = protect_password(password)
        else:
            raw.pop("protected_password", None)
        self._write_raw(raw)
        return self.load()


settings_store = FrontendSettingsStore()
