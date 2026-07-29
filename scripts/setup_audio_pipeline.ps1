$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Write-Host "Installing dependencies for Audio-to-Subtitle pipeline..."
python -m pip install --user -r (Join-Path $projectRoot "requirements.txt")

Write-Host "Done! You can now run src\audio_to_subtitle.py or double-click start_frontend.bat"
