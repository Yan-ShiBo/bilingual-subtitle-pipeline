$ErrorActionPreference = "Stop"

Write-Host "Installing dependencies for Audio-to-Subtitle pipeline..."
pip install faster-whisper openai pydub tqdm

Write-Host "Done! You can now run src\audio_to_subtitle.py or double-click start_frontend.bat"
