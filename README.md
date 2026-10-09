# YellowAI ⚡🎮
**Autonomous Gemini Flash Pokémon Yellow Recharged Streamer**

YellowAI is an AI agent that plays *Pokémon Recharged Yellow* (GBA) live on stream using **Gemini Flash**. It analyzes the game screen every turn, takes actions, maintains a 3-tiered hierarchical memory, and streams real-time commentary through a local neural text-to-speech engine.

---

## 🔁 Turn Cadence & Memory Architecture

| Frequency | Output | Purpose |
| :--- | :--- | :--- |
| **Every Turn** | GBA Button (`A`, `B`, `UP`, etc.) | Advances gameplay frame by frame. |
| **Every 5 Turns** | 1-Sentence Thought Quote | Sent to local neural TTS engine for live stream commentary. |
| **Every 10 Turns** | Short-Term Memory | Tactical focus: immediate room, battle status, NPC dialogue. |
| **Every 50 Turns** | Medium-Term Memory | Operational milestone: current town/route, gym target, party condition. |
| **Every 100 Turns** | Long-Term Memory | Grand strategy: badges won, team lineup, overarching champion path. |

---

## 🛠️ Setup & Installation

### 1. Requirements
Ensure Python 3.10+ is installed and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and insert your Gemini API Key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL_ID=gemini-3.5-flash
GEMINI_THINKING_BUDGET=0
```

### 3. Emulator Controls Setup (mGBA recommended)
Configure your emulator keys to match `config.py` (or edit `config.py` to match your emulator):
- **D-Pad**: Arrow keys (`Up`, `Down`, `Left`, `Right`)
- **A Button**: `x`
- **B Button**: `z`
- **Start**: `Enter`
- **Select**: `Backspace`

---

## 🚀 One-Click Launch (Recommended)

Simply double-click or run:
```bat
run.bat
```
`run.bat` automatically:
1. Creates the Python virtual environment (`.venv`) if not present.
2. Restores and updates all required Python packages from `requirements.txt`.
3. Generates `.env` from template if missing.
4. Checks for an active **mGBA** emulator window.
5. If **mGBA is not detected, it halts immediately** with instructions.
6. Once mGBA is detected, launches the full Gemini 3.5 Flash streamer loop!

---

## 🧪 Testing Subsystems

Before launching the full loop, test each subsystem using the built-in CLI flags:

### Test TTS Audio Commentary
```bash
python main.py --test-tts
```
*Synthesizes and speaks a sample greeting through your default audio device.*

### Test Screen Capture
```bash
python main.py --test-capture
```
*Searches for the emulator window, grabs a frame, and saves it to `debug_captures/test_capture.png`.*

### Test Input Injection
```bash
python main.py --test-input
```
*Focuses the emulator and presses button `A` after 2 seconds.*

---

## 🚀 Running YellowAI

Launch the player:
```bash
python main.py
```

### Useful CLI Options
- `--backend <gemini|ollama>`: Choose intelligence backend (`gemini` for cloud, `ollama` for 100% free local VLM).
- `--ollama-model <MODEL>`: Local model ID (defaults to `qwen2.5-vl:7b`).
- `--max-turns <N>`: Run for a set number of turns (e.g. `python main.py --max-turns 100`).
- `--no-tts`: Run silently without audio commentary.
- `--reset-memory`: Reset stored short, medium, and long-term memory back to Pallet Town defaults.
- `--model <MODEL_ID>`: Override Gemini model (defaults to `gemini-3.5-flash`).

### 🦙 Running 100% Free with Local Models (Ollama)
Pull a Vision-Language Model in Ollama:
```bash
ollama run qwen2.5-vl:7b
```
Then run YellowAI with zero token costs:
```bash
python main.py --backend ollama
```

---

## 📺 OBS Studio Stream Overlay (HUD)

A live animated streamer HUD is included in `overlay/`:
1. In **OBS Studio**, add a new **Browser Source**.
2. Check **Local file** and select `m:\Code\YellowAI\overlay\index.html`.
3. Set width to `480` and height to `360`.
4. As YellowAI plays, it continuously updates `overlay/overlay_state.json`, displaying:
   - Live turn counter & last button pressed.
   - Real-time speech bubble with Gemini's thought.
   - Interactive 3-tier memory drawer (Short / Medium / Long).
