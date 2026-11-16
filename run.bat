@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

title YellowAI - Local Ollama Pokemon Recharged Yellow Agent

if not exist ".venv\Scripts\activate.bat" (
    echo [Setup] Creating virtual environment .venv...
    python -m venv .venv || exit /b 1
)
call .venv\Scripts\activate.bat

pip install -r requirements.txt --quiet || exit /b 1

:: Ensure local Ollama server is running
powershell -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:11434/api/tags' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] 'ollama' not in PATH. Ensure Ollama is running at http://localhost:11434.
    ) else (
        echo [Ollama] Starting background Ollama server...
        start "" /B ollama serve >nul 2>&1
        timeout /t 3 /nobreak >nul
    )
)

echo [YellowAI] Starting local Ollama agent (auto-launching mGBA if needed)...
python main.py --launch-mgba %*
pause
