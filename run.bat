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

:: 5. Select Backend (if not already specified in args)
set "HAS_BACKEND="
for %%A in (%*) do (
    if /i "%%A"=="--backend" set "HAS_BACKEND=1"
)

set "BACKEND_FLAG="
if not defined HAS_BACKEND (
    echo.
    echo =====================================================================
    echo  Select Intelligence Backend:
    echo  [1] Google Gemini 3.5 Flash [Cloud API]
    echo  [2] Hybrid Mode [Gemma 4 E2B Local + Cloud Gemini Escalation]
    echo  [3] Bionic / LM Studio Local Server [100%% Free / Port 1234]
    echo  [4] Ollama Local VLM [100%% Free / RTX 3060 - Qwen 3.5]
    echo =====================================================================
    set "USER_CHOICE=1"
    set /p "USER_CHOICE=Enter choice [1, 2, 3, or 4, default is 1]: "
    if "!USER_CHOICE!"=="4" (
        set "BACKEND_FLAG=--backend ollama"
        echo [Backend] Selected: Ollama Local VLM
    ) else if "!USER_CHOICE!"=="3" (
        set "BACKEND_FLAG=--backend bionic"
        echo [Backend] Selected: Bionic / LM Studio Local Server
    ) else if "!USER_CHOICE!"=="2" (
        set "BACKEND_FLAG=--backend hybrid"
        echo [Backend] Selected: Hybrid Mode (Local + Cloud Escalation)
    ) else (
        set "BACKEND_FLAG=--backend gemini"
        echo [Backend] Selected: Google Gemini 3.5 Flash
    )
)

:: 6. Check Environment File (.env) for Gemini
if "!BACKEND_FLAG!"=="--backend gemini" (
    if not exist ".env" (
        echo [NOTICE] .env file not found. Creating from .env.example...
        copy .env.example .env >nul
        echo [ACTION REQUIRED] Please edit the .env file with your GEMINI_API_KEY before running!
    )
)

:: 7. Launch Main Script with Logger
echo.
echo [Step 3/3] Checking for mGBA and launching YellowAI...
echo.
python main.py !BACKEND_FLAG! %*

set SCRIPT_EXIT_CODE=%ERRORLEVEL%
if %SCRIPT_EXIT_CODE% NEQ 0 (
    echo.
    echo =====================================================================
    echo  YellowAI stopped with code %SCRIPT_EXIT_CODE%.
    echo  Ensure mGBA is running with Pokemon Yellow Recharged before starting.
    echo  Check logs/ for detailed output.
    echo =====================================================================
    echo.
    pause
    exit /b %SCRIPT_EXIT_CODE%
)

echo.
echo YellowAI session closed cleanly.
pause
