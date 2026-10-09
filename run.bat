@echo off
setlocal enabledelayedexpansion

title YellowAI - Gemini 3.5 Flash Pokemon Streamer

echo =====================================================================
echo  YellowAI: Gemini 3.5 Flash Pokemon Yellow Recharged Streamer
echo =====================================================================
echo.

:: 1. Check Python installation
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.10+ from https://www.python.org/ and check Add to PATH.
    echo.
    pause
    exit /b 1
)

:: 2. Check or Create Virtual Environment
if not exist ".venv\Scripts\activate.bat" (
    echo [Step 1/3] Creating virtual environment .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo [Step 1/3] Virtual environment .venv found.
)

:: 3. Activate Virtual Environment
call .venv\Scripts\activate.bat

:: 4. Restore / Install Dependencies
echo [Step 2/3] Restoring and verifying required Python packages...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install required packages from requirements.txt.
    pause
    exit /b 1
)
echo Python packages are up to date.

:: 5. Check Environment File (.env)
if not exist ".env" (
    echo [NOTICE] .env file not found. Creating from .env.example...
    copy .env.example .env >nul
    echo [ACTION REQUIRED] Please edit the .env file with your GEMINI_API_KEY before running!
)

:: 6. Launch Main Script with Logger
echo.
echo [Step 3/3] Checking for mGBA and launching YellowAI (logging to logs/latest.log)...
echo.
python main.py %*

set SCRIPT_EXIT_CODE=%ERRORLEVEL%
if %SCRIPT_EXIT_CODE% NEQ 0 (
    echo.
    echo =====================================================================
    echo  YellowAI stopped with code %SCRIPT_EXIT_CODE%.
    echo  Ensure mGBA is running with Pokemon Yellow Recharged before starting.
    echo  Check logs/latest.log for detailed output.
    echo =====================================================================
    echo.
    pause
    exit /b %SCRIPT_EXIT_CODE%
)

echo.
echo YellowAI session closed cleanly.
pause
