@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" reset_game.py %*
) else (
    python reset_game.py %*
)
pause
