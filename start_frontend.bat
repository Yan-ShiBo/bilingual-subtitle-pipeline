@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_frontend.ps1"

echo.
echo Frontend stopped. Press any key to close this window.
pause >nul
