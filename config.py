"""
Central Configuration for YellowAI (Local Ollama VLM Edition).
Runs Pokémon Recharged Yellow via mGBA using only local Ollama (`qwen3.5:9b`).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Directories
STATE_DIR = BASE_DIR / "state"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"
TEMP_SCREENSHOTS_DIR = TEMP_DIR / "screenshots"

# Persistent state files
CONVERSATION_HISTORY_FILE = STATE_DIR / "conversation_history.json"
SIGNS_AND_SPEECH_FILE = STATE_DIR / "signs_and_speech.json"
THINKING_TRACES_FILE = STATE_DIR / "thinking_traces.jsonl"
THINKING_INSIGHTS_FILE = STATE_DIR / "thinking_insights.json"
WEB_KNOWLEDGE_CACHE_FILE = STATE_DIR / "web_knowledge_cache.json"
GOAL_STATE_FILE = STATE_DIR / "goal_state.json"

# Bundled mGBA Emulator & ROM
MGBA_DIR = BASE_DIR / "mGBA-0.10.5-win64"
MGBA_EXE = MGBA_DIR / "mGBA.exe"
DEFAULT_ROM_PATH = MGBA_DIR / "games" / "rchrgd yllw v1.8.0.gba"
DEFAULT_SAVESTATE_PATH = MGBA_DIR / "games" / "rchrgd yllw v1.8.0.ss0"

for _d in (STATE_DIR, LOGS_DIR, TEMP_DIR, TEMP_SCREENSHOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Local Ollama VLM Configuration (RTX 3060 12GB VRAM)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "qwen3.5:9b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))
OLLAMA_THINK_MODE = os.getenv("OLLAMA_THINK_MODE", "auto")  # auto | always | never
OLLAMA_MAX_TOKENS_THINK = int(os.getenv("OLLAMA_MAX_TOKENS_THINK", "1800"))
OLLAMA_MAX_TOKENS_FAST = int(os.getenv("OLLAMA_MAX_TOKENS_FAST", "600"))
OLLAMA_SAVE_THINKING_TRACES = os.getenv("OLLAMA_SAVE_THINKING_TRACES", "true").lower() == "true"

# Multi-Turn Conversation History & Compaction
OLLAMA_HISTORY_MAX_TURNS = int(os.getenv("OLLAMA_HISTORY_MAX_TURNS", "18"))
OLLAMA_HISTORY_KEEP_RECENT = int(os.getenv("OLLAMA_HISTORY_KEEP_RECENT", "6"))
OLLAMA_HISTORY_COMPACT_TOKENS = int(os.getenv("OLLAMA_HISTORY_COMPACT_TOKENS", "10000"))

# Autonomous Web Search & Discovery-First Planning
OLLAMA_ENABLE_WEB_SEARCH = os.getenv("OLLAMA_ENABLE_WEB_SEARCH", "true").lower() == "true"
OLLAMA_AUTO_PLAN_ON_STARTUP = os.getenv("OLLAMA_AUTO_PLAN_ON_STARTUP", "true").lower() == "true"
OLLAMA_DISCOVERY_ON_START = os.getenv("OLLAMA_DISCOVERY_ON_START", "true").lower() == "true"
OLLAMA_MAX_DISCOVERY_TURNS = int(os.getenv("OLLAMA_MAX_DISCOVERY_TURNS", "12"))

# mGBA Window, Resolution & Input Controls
EMULATOR_WINDOW_TITLE_KEYWORDS = ["mGBA", "Recharged Yellow", "Pokemon", "Yellow"]
TARGET_RESOLUTION = (480, 320)  # 2x GBA (240x160)

KEY_MAPPINGS = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "A": "x",
    "B": "z",
    "SELECT": "backspace",
    "START": "enter",
}

INPUT_HOLD_DURATION = 0.14
STEP_SETTLE_DELAY = 0.25
TURN_SETTLE_DELAY = 0.25
MAX_BATCH_ACTIONS = 5

BACKGROUND_INPUT_MODE = os.getenv("BACKGROUND_INPUT_MODE", "postmessage")
MGBA_LUA_HOST = "127.0.0.1"
MGBA_LUA_PORT = 8888
