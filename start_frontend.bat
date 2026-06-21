@echo off
setlocal

set URL=http://127.0.0.1:8765/
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }"
if %ERRORLEVEL% EQU 0 (
  start "" "%URL%"
  exit /b 0
)

start "" "%URL%"
python "%~dp0subtitle_frontend.py" --host 127.0.0.1 --port 8765

echo.
echo Frontend stopped. Press any key to close this window.
pause >nul
