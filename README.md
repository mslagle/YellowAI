# YellowAI — Local Ollama VLM Agent for Pokémon Recharged Yellow

Compact, 100% local vision AI running **Ollama (`qwen3.5:9b`)** against bundled **mGBA 0.10.5** (`rchrgd yllw v1.8.0.gba`). Zero cloud APIs.

## Features
- **Local Vision + Strict GBA Controls**: Sends each turn's screenshot to `qwen3.5:9b` and executes returned GBA button presses (`UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`, `SELECT`, `START`).
- **Temp Screenshot Archive**: Saves every turn's screenshot to `temp/screenshots/run_<id>/turn_NNNNN.png`.
- **Multi-Turn Memory & Compaction**: Persists active turns, compacted episode summaries, and past-run continuity across sessions in `state/conversation_history.json`.
- **Self-Discovery + Goal Dropping**: Learns game progress from signs, NPCs, in-game Start/Trainer Card/Bag/Party/Pokédex checks, and internet search, while permanently tombstoning completed goals in `state/goal_state.json`.
- **Lossless Sign & Speech Archival**: Records all seen signs, dialogue, and menu text untruncated in `state/signs_and_speech.json`.
- **Thinking Persistence**: Archives raw `thinking` traces to `state/thinking_traces.jsonl` and recalls distilled strategic insights from `state/thinking_insights.json`.
- **Free Internet Search**: Uses DuckDuckGo (`web_search` / `fetch_webpage`) via native Ollama tool calling with local caching (`state/web_knowledge_cache.json`).

## Usage
```powershell
# Double-click run.bat, OR run from PowerShell:
.\.venv\Scripts\python.exe main.py --launch-mgba

# Launch mGBA standalone:
.\.venv\Scripts\python.exe -m emulator.mgba_launcher

# Run quick self-test (live Ollama vision + history + web search):
.\.venv\Scripts\python.exe main.py --test-ollama

# Reset saved state (optionally keep distilled insights & web cache):
.\.venv\Scripts\python.exe reset_game.py --clear-logs --keep-wisdom
```
