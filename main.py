"""
Main entry point for YellowAI.
Run Gemini 3.5 Flash playing Pokémon Yellow Recharged on mGBA with live streaming commentary.
Strictly stops if mGBA is not detected.
"""

import argparse
import sys
import os
import ctypes
from pathlib import Path

# Enable DPI awareness and attach to interactive user desktop before any GUI/audio subsystems initialize
try:
    user32 = ctypes.windll.user32
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

try:
    user32 = ctypes.windll.user32
    h_default = user32.OpenDesktopW('default', 0, False, 0x01FF)
    if h_default:
        user32.SetThreadDesktop(h_default)
except Exception:
    pass

import config
from config import (
    BASE_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL_ID,
    DEBUG_CAPTURES_DIR,
    MEMORY_FILE,
    EMULATOR_WINDOW_TITLE_KEYWORDS,
    TURN_SETTLE_DELAY,
    LAST_CAPTURE_FILE,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_ID,
    BIONIC_BASE_URL,
    BIONIC_MODEL_ID
)
from core.logger import setup_logging
from core.game_loop import GameLoop
from emulator.screen_capture import ScreenCapture
from emulator.input_controller import InputController
from tts.tts_engine import TTSEngine
from core.memory_manager import MemoryManager

def test_tts():
    print("[Test] Testing TTS engine...")
    engine = TTSEngine()
    sample_text = "Hey everyone! I'm Gemini 3.5 Flash, and today we're starting our journey in Pokémon Yellow Recharged!"
    data, rate = engine.synthesize(sample_text)
    if data is not None and len(data) > 0:
        import sounddevice as sd
        print(f"[Test] Synthesized audio successfully ({len(data)} samples @ {rate}Hz). Playing...")
        sd.play(data, rate)
        sd.wait()
        print("[Test] TTS playback complete!")
    else:
        print("[Test] TTS synthesis failed.")
        sys.exit(1)

def test_capture():
    print("[Test] Testing Screen Capture...")
    cap = ScreenCapture()
    hwnd = cap.find_emulator_window()
    if not hwnd:
        print("\n" + "!" * 65)
        print(" [ERROR] Emulator window not detected!")
        print(" Please open mGBA with Pokémon Yellow Recharged running.")
        print(f" Expected window title containing one of: {EMULATOR_WINDOW_TITLE_KEYWORDS}")
        print("!" * 65 + "\n")
        sys.exit(1)

    print(f"[Test] Attached to mGBA: '{cap.target_title}' (HWND: {hwnd})")
    frame, thumb = cap.capture_frame()
    if frame is not None:
        save_path = DEBUG_CAPTURES_DIR / "test_capture.png"
        frame.save(save_path)
        frame.save(LAST_CAPTURE_FILE)
        print(f"[Test] Capture successful! Saved to: {save_path} and {LAST_CAPTURE_FILE}")
    else:
        print("[Test] Failed to capture client frame from emulator window.")
        sys.exit(1)

def test_input():
    print("[Test] Testing Input Injection...")
    from config import KEY_MAPPINGS
    cap = ScreenCapture()
    hwnd = cap.find_emulator_window()
    if not hwnd:
        print("\n[ERROR] mGBA window not found. Open mGBA first!")
        sys.exit(1)

    controller = InputController(key_mappings=KEY_MAPPINGS, target_hwnd=hwnd)
    focused = controller.focus_hwnd(hwnd)
    if not focused:
        print("\n[ERROR] Could not prepare mGBA emulator window for input.")
        sys.exit(1)

    print(f"[Test] Emulator window ready (mode={controller.input_mode}): '{cap.target_title}'")
    print("[Test] Sending background button press 'A' in 2 seconds...")
    import time
    time.sleep(2)
    controller.press_button("A")
    print("[Test] Pressed button 'A'!")

def reset_memory():
    from reset_game import reset_all_progress
    reset_all_progress()

def main():
    from config import HYBRID_LOCAL_BACKEND
    parser = argparse.ArgumentParser(description="YellowAI: Autonomous Pokémon Yellow Recharged Player")
    parser.add_argument("--backend", type=str, choices=["hybrid", "gemini", "ollama", "bionic", "lmstudio"], default=LLM_BACKEND, help="Intelligence backend: 'hybrid' (local + cloud escalation), 'gemini' (cloud), 'ollama' (free local VLM), or 'bionic'/'lmstudio' (local server at port 1234).")
    parser.add_argument("--tts-provider", type=str, choices=["edge", "sapi", "none"], default=getattr(config, "TTS_PROVIDER", "edge"), help="TTS voice engine: 'edge' (neural, auto-falls back to Windows SAPI on error), 'sapi' (offline Windows voice), or 'none'.")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum turns to run before stopping.")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS audio commentary.")
    parser.add_argument("--model", type=str, default=GEMINI_MODEL_ID, help="Gemini model ID to use (when backend=gemini or hybrid escalation).")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL_ID, help="Ollama model ID to use (default: qwen3.5).")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_BASE_URL, help="Ollama base URL (default: http://localhost:11434).")
    parser.add_argument("--bionic-url", type=str, default=BIONIC_BASE_URL, help="Bionic / LM Studio base URL (default: http://localhost:1234/v1).")
    parser.add_argument("--bionic-model", type=str, default=BIONIC_MODEL_ID, help="Bionic / LM Studio model ID (default: 'default', auto-detects loaded model).")
    parser.add_argument("--test-tts", action="store_true", help="Test local TTS audio synthesis and playback.")
    parser.add_argument("--test-capture", action="store_true", help="Capture a single frame and save to debug folder.")
    parser.add_argument("--test-input", action="store_true", help="Send a test button press to the emulator.")
    parser.add_argument("--reset", "--reset-memory", dest="reset_memory", action="store_true", help="Reset all saved memory, quests, dialogue history, and overlay state to starting defaults.")
    parser.add_argument("--turn-delay", type=float, default=TURN_SETTLE_DELAY, help="Delay in seconds between turns (default: 0.25s).")
    parser.add_argument("--serve-overlay", action="store_true", help="Run standalone OBS stream overlay server at http://localhost:8080/.")

    args = parser.parse_args()
    config.TTS_PROVIDER = args.tts_provider
    setup_logging(BASE_DIR)

    if args.serve_overlay:
        from overlay.server import OverlayServer
        from config import OVERLAY_DIR, OVERLAY_PORT
        import time
        server = OverlayServer(OVERLAY_DIR, OVERLAY_PORT)
        server.start()
        print("[Overlay] Press Ctrl+C to stop the overlay server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()
        return

    if args.test_tts:
        test_tts()
        return

    if args.test_capture:
        test_capture()
        return

    if args.test_input:
        test_input()
        return

    if args.reset_memory:
        reset_memory()
        return

    # 1. Strictly verify mGBA emulator is running
    print("[YellowAI] Checking for active mGBA emulator window...")
    checker = ScreenCapture()
    hwnd = checker.find_emulator_window()
    if not hwnd:
        print("\n" + "!" * 65)
        print(" [ERROR] mGBA emulator window not detected!")
        print(" Please open mGBA with Pokémon Yellow Recharged running first.")
        print(f" Window title must contain one of: {EMULATOR_WINDOW_TITLE_KEYWORDS}")
        print(" Stopping YellowAI execution.")
        print("!" * 65 + "\n")
        sys.exit(1)

    backend = args.backend.lower().strip()
    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

    # 2. Check Backend Prerequisites
    local_target = HYBRID_LOCAL_BACKEND if backend == "hybrid" else backend
    if local_target in ["bionic", "lmstudio"]:
        print(f"[YellowAI] Verifying local Bionic / LM Studio service at {args.bionic_url}...")
        from agent.bionic_player import BionicPlayer
        bionic_checker = BionicPlayer(base_url=args.bionic_url, model_id=args.bionic_model)
        if not bionic_checker.is_server_reachable():
            if backend == "hybrid" and api_key:
                print(f"[YellowAI] Local server at {args.bionic_url} not reachable; falling back to Cloud Gemini!")
                backend = "gemini"
            else:
                print("\n" + "!" * 65)
                print(" [ERROR] Local Bionic / LM Studio service is not running or not reachable!")
                print(f" URL: {args.bionic_url}")
                print(" Please launch LM Studio / Bionic and start the Local Server on port 1234.")
                print("!" * 65 + "\n")
                sys.exit(1)
        else:
            # Auto-load model if currently unloaded in LM Studio / Bionic
            bionic_checker.ensure_model_loaded(args.bionic_model if args.bionic_model != "default" else None)
            loaded_models = bionic_checker.get_loaded_models()
            if loaded_models:
                print(f"[YellowAI] Connected to local server. Loaded models: {loaded_models}")
            else:
                print("[YellowAI] Connected to local server. (Ensure a vision model is loaded in LM Studio / Bionic!)")
    elif local_target == "ollama":
        print(f"[YellowAI] Verifying local Ollama service at {args.ollama_url}...")
        from agent.ollama_player import OllamaPlayer
        checker_player = OllamaPlayer(base_url=args.ollama_url, model_id=args.ollama_model)
        if not checker_player.is_server_reachable():
            if backend == "hybrid" and api_key:
                print(f"[YellowAI] Local Ollama at {args.ollama_url} not reachable; falling back to Cloud Gemini!")
                backend = "gemini"
            else:
                print("\n" + "!" * 65)
                print(" [ERROR] Local Ollama service is not running or not reachable!")
                print(f" URL: {args.ollama_url}")
                print(" Please launch Ollama in the background (or run 'ollama serve').")
                print("!" * 65 + "\n")
                sys.exit(1)
        elif not checker_player.is_model_available():
            print("\n" + "!" * 65)
            print(f" [WARNING] Model '{args.ollama_model}' does not appear to be downloaded yet in Ollama.")
            print(f" To pull it, run this command in your terminal:")
            print(f"   ollama run {args.ollama_model}")
            print("!" * 65 + "\n")

    if backend == "gemini" and not api_key:
        print("\n" + "!" * 65)
        print(" [ERROR] GEMINI_API_KEY is not set!")
        print(" Please edit the .env file in m:\\Code\\YellowAI and set your GEMINI_API_KEY.")
        print(" Or run YellowAI with a 100% free local model: python main.py --backend bionic")
        print("!" * 65 + "\n")
        sys.exit(1)

    loop = GameLoop(
        api_key=api_key,
        model_id=args.model,
        backend=backend,
        ollama_base_url=args.ollama_url,
        ollama_model_id=args.ollama_model,
        bionic_base_url=args.bionic_url,
        bionic_model_id=args.bionic_model,
        turn_settle_delay=args.turn_delay,
        enable_tts=not args.no_tts
    )

    success = loop.run(max_turns=args.max_turns)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
