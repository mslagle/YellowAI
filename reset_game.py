"""
Reset utility for YellowAI.
Clears all saved AI progress from memory files, resets the OBS stream overlay state,
and removes cached screen/route capture artifacts so you can start a fresh playthrough.

Usage:
    python reset_game.py                # Reset memory, overlay state, and cached capture frames
    python reset_game.py --clear-logs   # Also delete old .log files and debug_captures/*.png
"""

import argparse
import json
import time
from pathlib import Path

from config import (
    BASE_DIR,
    STATE_DIR,
    MEMORY_FILE,
    OVERLAY_STATE_FILE,
    OVERLAY_JS_FILE,
    LOGS_DIR,
    DEBUG_CAPTURES_DIR,
)
from core.memory_manager import MemoryManager


def reset_all_progress(clear_logs: bool = False) -> None:
    print("=" * 62)
    print(" [YellowAI] Resetting Game Progress & Memory State")
    print("=" * 62)

    # 1. Reset MemoryManager (state/session_memory.json) and any extra JSON files in state/
    if STATE_DIR.exists():
        for json_file in STATE_DIR.glob("*.json"):
            if json_file != MEMORY_FILE:
                try:
                    json_file.unlink()
                    print(f"[Reset] Removed extra state file: {json_file}")
                except Exception as e:
                    print(f"[Reset] Could not remove {json_file}: {e}")

    mgr = MemoryManager(storage_path=MEMORY_FILE)
    mgr.reset()
    print(f"[Reset] Reset session memory to starting state: {MEMORY_FILE}")

    # 2. Reset Overlay State (overlay/overlay_state.json and overlay/overlay_state.js)
    active_q = mgr.priority_quest or (mgr.quests[0] if mgr.quests else "Explore Kanto")
    initial_overlay = {
        "turn": 0,
        "last_action": "NONE",
        "latest_thought": "Ready to start a fresh Pokémon Yellow Recharged adventure!",
        "priority_quest": mgr.priority_quest,
        "active_quest": active_q,
        "quests": list(mgr.quests),
        "completed_milestones": list(mgr.completed_milestones),
        "current_area": mgr.current_area,
        "team": list(mgr.team),
        "supplies": dict(mgr.supplies),
        "badges": mgr.badges,
        "micro_step": mgr.micro_step,
        "short_term_memory": mgr.short_term_memory,
        "medium_term_memory": mgr.medium_term_memory,
        "long_term_memory": mgr.long_term_memory,
        "knowledge_notes": list(mgr.knowledge_notes),
        "dialogue_history": list(mgr.dialogue_history),
        "timestamp": time.time(),
    }
    try:
        OVERLAY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OVERLAY_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_overlay, f, indent=2)
        print(f"[Reset] Reset overlay state: {OVERLAY_STATE_FILE}")
    except Exception as e:
        print(f"[Reset] Could not reset overlay state: {e}")

    if OVERLAY_JS_FILE.exists():
        try:
            OVERLAY_JS_FILE.unlink()
            print(f"[Reset] Removed legacy overlay JS file: {OVERLAY_JS_FILE}")
        except Exception:
            pass

    # 3. Clean up cached capture & route images in logs/ so stale frames aren't reused
    stale_capture_files = [
        LOGS_DIR / "last_capture.png",
        LOGS_DIR / "previous_capture.png",
        LOGS_DIR / "last_poi_annotated.png",
        LOGS_DIR / "last_route_planned.png",
        LOGS_DIR / "last_dialogue_crop.png",
    ]
    removed_captures = 0
    for img_path in stale_capture_files:
        if img_path.exists():
            try:
                img_path.unlink()
                removed_captures += 1
            except Exception:
                pass
    if removed_captures:
        print(f"[Reset] Cleared {removed_captures} cached capture/route image(s) from {LOGS_DIR}")

    # 4. Optionally clear old log files and debug captures
    if clear_logs:
        removed_logs = 0
        if LOGS_DIR.exists():
            for log_file in LOGS_DIR.glob("*.log"):
                try:
                    log_file.unlink()
                    removed_logs += 1
                except Exception:
                    pass
        removed_debug = 0
        if DEBUG_CAPTURES_DIR.exists():
            for dbg_file in DEBUG_CAPTURES_DIR.glob("*.png"):
                try:
                    dbg_file.unlink()
                    removed_debug += 1
                except Exception:
                    pass
        print(f"[Reset] Cleared {removed_logs} log file(s) and {removed_debug} debug capture(s).")

    print("-" * 62)
    print(" [OK] All YellowAI memory and progress files have been reset!")
    print(f"      Starting Area : {mgr.current_area}")
    print(f"      Initial Quest : {mgr.priority_quest}")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(
        description="Reset all YellowAI memory, quests, dialogue history, and overlay progress."
    )
    parser.add_argument(
        "--clear-logs",
        action="store_true",
        help="Also delete old .log files in logs/ and debug screenshots in debug_captures/.",
    )
    args = parser.parse_args()
    reset_all_progress(clear_logs=args.clear_logs)


if __name__ == "__main__":
    main()
