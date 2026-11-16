@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m emulator.mgba_launcher %*
) else (
    cd /d "%~dp0mGBA-0.10.5-win64"
    start "" "mGBA.exe" "games\rchrgd yllw v1.8.0.gba"
)

