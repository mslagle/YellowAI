"""
Configuration settings for YellowAI: Gemini 3.5 Flash playing Pokémon Yellow Recharged in mGBA.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.resolve()
STATE_DIR = BASE_DIR / "state"
OVERLAY_DIR = BASE_DIR / "overlay"
DEBUG_CAPTURES_DIR = BASE_DIR / "debug_captures"
LOGS_DIR = BASE_DIR / "logs"

STATE_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# State files & Log capture file
MEMORY_FILE = STATE_DIR / "session_memory.json"
OVERLAY_STATE_FILE = OVERLAY_DIR / "overlay_state.json"
OVERLAY_JS_FILE = OVERLAY_DIR / "overlay_state.js"
LAST_CAPTURE_FILE = LOGS_DIR / "last_capture.png"
PREVIOUS_CAPTURE_FILE = LOGS_DIR / "previous_capture.png"

# Emulator Window Configuration
# mGBA title format is typically "mGBA - [Rom Name]" or "mGBA"
EMULATOR_WINDOW_TITLE_KEYWORDS = ["mGBA", "Recharged", "Yellow", "Pokemon", "Pokémon"]

# Target Capture Resolution for Gemini (clean aspect ratio 3:2 matching GBA)
CAPTURE_TARGET_WIDTH = 480
CAPTURE_TARGET_HEIGHT = 320

# Model Provider Backend ("hybrid", "gemini", "bionic", "lmstudio", or "ollama")
LLM_BACKEND = os.getenv("LLM_BACKEND", "hybrid").lower().strip()

# Hybrid Mode Configuration (used when LLM_BACKEND="hybrid")
HYBRID_LOCAL_BACKEND = os.getenv("HYBRID_LOCAL_BACKEND", "bionic").lower().strip()  # "bionic" or "ollama"
HYBRID_ESCALATE_ON_STUCK_TURNS = int(os.getenv("HYBRID_ESCALATE_ON_STUCK_TURNS", "2"))
HYBRID_ESCALATE_ON_BATTLES = os.getenv("HYBRID_ESCALATE_ON_BATTLES", "false").lower() == "true"
HYBRID_CHECKPOINT_TURNS = int(os.getenv("HYBRID_CHECKPOINT_TURNS", "25"))  # 0 to disable periodic cloud check-in

# Gemini Model Settings (Gemini 3.5 Flash)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash")
# Set thinking_budget=0 for fastest sub-second turn latency on stream
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

# Local Ollama Settings (100% Free / Offline VLM)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "qwen3.5")

# Local Bionic / LM Studio Settings (100% Free / Port 1234)
BIONIC_BASE_URL = os.getenv("BIONIC_BASE_URL", "http://localhost:1234/v1")
BIONIC_MODEL_ID = os.getenv("BIONIC_MODEL_ID", "default")

# Background Input & Window Focus Configuration
# Options: "postmessage" (Win32 background messages, no focus needed),
#          "lua_socket" (mGBA Lua bridge socket, falls back to postmessage),
#          "foreground" (legacy SetForegroundWindow + pydirectinput)
BACKGROUND_INPUT_MODE = os.getenv("BACKGROUND_INPUT_MODE", "postmessage").lower().strip()
LUA_BRIDGE_HOST = os.getenv("LUA_BRIDGE_HOST", "127.0.0.1")
LUA_BRIDGE_PORT = int(os.getenv("LUA_BRIDGE_PORT", "8888"))

# Tier 0 Reflexes & Token Optimization
ENABLE_TIER0_REFLEXES = os.getenv("ENABLE_TIER0_REFLEXES", "true").lower() == "true"
SEND_PREVIOUS_IMAGE_ONLY_WHEN_STUCK = os.getenv("SEND_PREVIOUS_IMAGE_ONLY_WHEN_STUCK", "true").lower() == "true"

# Rolling Dialogue Log Settings
MAX_DIALOGUE_HISTORY = int(os.getenv("MAX_DIALOGUE_HISTORY", "20"))
DIALOGUE_PROMPT_RECENT_COUNT = int(os.getenv("DIALOGUE_PROMPT_RECENT_COUNT", "4"))

# Turn Timing and Delays (in seconds)
BUTTON_HOLD_DURATION = float(os.getenv("BUTTON_HOLD_DURATION", "0.05"))  # ~3 frames hold at 60fps (~16.7ms/frame)
STEP_SETTLE_DELAY = float(os.getenv("STEP_SETTLE_DELAY", "0.25"))        # 0.05s hold + 0.25s settle = 0.30s (> 16-frame 0.267s GBA walk cycle)
TURN_SETTLE_DELAY = float(os.getenv("TURN_SETTLE_DELAY", "0.25"))        # Paced delay between turns (default: 0.25s)
MAX_BATCH_ACTIONS = int(os.getenv("MAX_BATCH_ACTIONS", "6"))       # Maximum steps in a batch action sequence
ANTI_STUCK_CONSECUTIVE_THRESHOLD = 6  # If screen unchanged for N turns, take recovery step

# Last-Ditch Effort / Tactical Rescue Configuration
ENABLE_RESCUE_THINKING = os.getenv("ENABLE_RESCUE_THINKING", "true").lower() == "true"
RESCUE_THINKING_BUDGET = int(os.getenv("RESCUE_THINKING_BUDGET", "2048"))  # Medium-sized thinking budget (2048 tokens)
RESCUE_ENABLE_WEB_SEARCH = os.getenv("RESCUE_ENABLE_WEB_SEARCH", "true").lower() == "true"  # Search internet for clues
RESCUE_STUCK_THRESHOLD = int(os.getenv("RESCUE_STUCK_THRESHOLD", "5"))    # Consecutive blocked turns before rescue
RESCUE_COOLDOWN_TURNS = 6                                                 # Minimum turns between rescue calls

# Battle Strategy & Encounter Thinking Configuration
ENABLE_BATTLE_THINKING = os.getenv("ENABLE_BATTLE_THINKING", "true").lower() == "true"
BATTLE_THINKING_BUDGET = int(os.getenv("BATTLE_THINKING_BUDGET", "1024"))  # Tactical thinking budget for battle start & enemy switches

# Cadence Intervals
CADENCE_BUTTON = 1              # Every turn
CADENCE_THOUGHT = 5             # Every 5 turns -> TTS quote
CADENCE_SHORT_MEMORY = 10       # Every 10 turns -> Short-term memory
CADENCE_MEDIUM_MEMORY = 50      # Every 50 turns -> Medium-term memory
CADENCE_LONG_MEMORY = 100       # Every 100 turns -> Long-term memory

# Default mGBA Keyboard Controls (can be adjusted to match emulator settings)
# In mGBA: Tools -> Settings -> Keybindings
# D-pad: Up, Down, Left, Right
# A: X
# B: Z
# L: A
# R: S
# Start: Enter
# Select: Backspace
KEY_MAPPINGS = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "A": "x",
    "B": "z",
    "START": "enter",
    "SELECT": "backspace",
    "WAIT": None
}

# High-Quality Natural TTS Voices (via Edge Neural TTS)
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AndrewMultilingualNeural")
TTS_SPEECH_RATE = "+10%"

# OBS Stream Overlay Server
OVERLAY_PORT = 8080
ENABLE_OVERLAY_SERVER = True
