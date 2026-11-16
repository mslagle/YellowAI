"""
YellowAI: Compact Local Ollama AI for Pokémon Recharged Yellow (mGBA).
"""

import argparse
import sys
import tempfile
from pathlib import Path

from PIL import Image

import config
from agent.ollama_player import OllamaPlayer
from agent.web_search import WebSearchClient
from core.conversation_history import ConversationHistoryManager
from core.dialogue_recorder import DialogueRecorder
from core.goal_tracker import GoalTracker
from core.game_loop import GameLoop
from core.logger import setup_logging
from core.thinking_store import ThinkingStore
from emulator.input_controller import InputController
from emulator.mgba_launcher import launch_mgba
from emulator.screen_capture import ScreenCapture
from reset_game import reset_all_progress


def test_ollama(base_url: str, model_id: str, think_mode: str = "auto", enable_web_search: bool = True) -> None:
    print(f"[Test] Checking Ollama ({model_id} @ {base_url})...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        history = ConversationHistoryManager(storage_path=tmp / "history.json")
        player = OllamaPlayer(
            base_url=base_url,
            model_id=model_id,
            think_mode=think_mode,
            enable_web_search=enable_web_search,
            history_manager=history,
            web_client=WebSearchClient(cache_path=tmp / "web.json"),
            thinking_store=ThinkingStore(traces_path=tmp / "traces.jsonl", insights_path=tmp / "insights.json"),
            goal_tracker=GoalTracker(state_path=tmp / "goal.json"),
            dialogue_recorder=DialogueRecorder(storage_path=tmp / "signs_and_speech.json"),
        )
        if not player.is_server_reachable():
            print(f"[ERROR] Ollama is not reachable at {base_url}. Run 'ollama serve'.")
            sys.exit(1)
        if not player.is_model_available():
            print(f"[ERROR] Model '{model_id}' not found. Run 'ollama pull {model_id}'.")
            sys.exit(1)

        fixture = config.BASE_DIR / "tests" / "fixtures" / "recharged_yellow" / "title_screen.png"
        img = Image.open(fixture).convert("RGB") if fixture.exists() else Image.new("RGB", (240, 160), (255, 220, 40))

        decision = player.decide_action(turn_id=1, image=img, emulator_verified=True, screenshot_path=str(fixture if fixture.exists() else ""))
        print(f"[Test] Buttons={decision.get('flat_actions')} | Reason={decision.get('reasoning')}")
        if decision.get("fallback") or not history.active_turns:
            print("[ERROR] Ollama test returned fallback or failed to persist history.")
            sys.exit(1)

        if enable_web_search:
            resp = player.web_client.search("Pokemon Recharged Yellow", max_results=2)
            items = resp.get("results", [])
            print(f"[Test] Web search returned {len(items)} result(s).")
            for r in items:
                print(f"  - {r.get('title', '')[:70]} ({r.get('url', '')})")

    print("[Test] Local Ollama VLM + history + web search check passed!")


def test_capture() -> None:
    cap = ScreenCapture(window_keywords=config.EMULATOR_WINDOW_TITLE_KEYWORDS)
    hwnd = cap.find_emulator_window()
    if not hwnd:
        print("[ERROR] mGBA window not found.")
        sys.exit(1)
    frame, _ = cap.capture_frame()
    if frame is None:
        print("[ERROR] Failed to capture frame.")
        sys.exit(1)
    out = config.TEMP_DIR / "test_capture.png"
    frame.save(out)
    print(f"[Test] Saved capture to {out}")


def test_input() -> None:
    cap = ScreenCapture(window_keywords=config.EMULATOR_WINDOW_TITLE_KEYWORDS)
    hwnd = cap.find_emulator_window()
    if not hwnd:
        print("[ERROR] mGBA window not found.")
        sys.exit(1)
    ctrl = InputController(key_mappings=config.KEY_MAPPINGS, target_hwnd=hwnd)
    ctrl.focus_hwnd(hwnd)
    ctrl.press_button("A")
    print("[Test] Sent button press 'A' to mGBA.")


def main() -> None:
    parser = argparse.ArgumentParser(description="YellowAI: Compact Local Ollama Pokémon Player")
    parser.add_argument("--model", type=str, default=config.OLLAMA_MODEL_ID, help=f"Ollama model ID (default: {config.OLLAMA_MODEL_ID}).")
    parser.add_argument("--url", type=str, default=config.OLLAMA_BASE_URL, help="Ollama server URL.")
    parser.add_argument("--think-mode", type=str, choices=["auto", "always", "never"], default=config.OLLAMA_THINK_MODE)
    parser.add_argument("--no-web-search", action="store_true", help="Disable DuckDuckGo internet search.")
    parser.add_argument("--no-discovery", action="store_true", help="Skip initial Start-menu/Trainer Card discovery audit.")
    parser.add_argument("--discover-now", action="store_true", help="Force Start-menu/Trainer Card/Bag/Pokédex audit on launch.")
    parser.add_argument("--max-turns", type=int, default=None, help="Stop after N turns.")
    parser.add_argument("--turn-delay", type=float, default=config.TURN_SETTLE_DELAY, help="Seconds between turns.")
    parser.add_argument("--launch-mgba", action="store_true", help="Auto-launch bundled mGBA with Pokémon Recharged Yellow.")
    parser.add_argument("--savestate", type=str, default=None, help="Optional savestate path for --launch-mgba.")
    parser.add_argument("--test-ollama", action="store_true", help="Verify local Ollama vision, buttons, history, and web search.")
    parser.add_argument("--test-capture", action="store_true", help="Capture one test frame from mGBA into temp/.")
    parser.add_argument("--test-input", action="store_true", help="Send test button 'A' to mGBA.")
    parser.add_argument("--reset", action="store_true", help="Reset saved goals, history, dialogue archive, and thinking state.")

    args = parser.parse_args()
    config.OLLAMA_THINK_MODE = args.think_mode
    config.OLLAMA_ENABLE_WEB_SEARCH = config.OLLAMA_ENABLE_WEB_SEARCH and not args.no_web_search
    setup_logging(config.BASE_DIR)

    if args.test_ollama:
        test_ollama(args.url, args.model, args.think_mode, config.OLLAMA_ENABLE_WEB_SEARCH)
        return
    if args.test_capture:
        test_capture()
        return
    if args.test_input:
        test_input()
        return
    if args.reset:
        reset_all_progress()
        return

    checker = ScreenCapture(window_keywords=config.EMULATOR_WINDOW_TITLE_KEYWORDS)
    hwnd = checker.find_emulator_window()
    if not hwnd and args.launch_mgba:
        hwnd = launch_mgba(savestate_path=args.savestate)
    if not hwnd:
        print("[ERROR] mGBA window not detected. Open mGBA or pass --launch-mgba.")
        sys.exit(1)

    loop = GameLoop(
        ollama_base_url=args.url,
        ollama_model_id=args.model,
        turn_settle_delay=args.turn_delay,
        discovery_on_start=(config.OLLAMA_DISCOVERY_ON_START and not args.no_discovery),
        force_discovery=args.discover_now,
    )
    if not loop.run(max_turns=args.max_turns):
        sys.exit(1)


if __name__ == "__main__":
    main()
