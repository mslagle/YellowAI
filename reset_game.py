"""
Compact Reset Utility for YellowAI (Local Ollama Edition).
Clears saved history, goal state, dialogue archives, thinking traces, and optional temp screenshots.
"""

import argparse
import shutil
import config


def reset_all_progress(clear_logs: bool = False, keep_wisdom: bool = False) -> None:
    preserved = set()
    if keep_wisdom:
        preserved.add(config.THINKING_INSIGHTS_FILE)
        preserved.add(config.WEB_KNOWLEDGE_CACHE_FILE)

    removed_state = 0
    if config.STATE_DIR.exists():
        for path in list(config.STATE_DIR.glob("*.json")) + list(config.STATE_DIR.glob("*.jsonl")):
            if path not in preserved:
                try:
                    path.unlink()
                    removed_state += 1
                except Exception as e:
                    print(f"[Reset] Could not remove {path}: {e}")

    removed_temp = 0
    if clear_logs:
        if config.LOGS_DIR.exists():
            for p in config.LOGS_DIR.glob("*.log"):
                try:
                    p.unlink()
                    removed_temp += 1
                except Exception:
                    pass
        if config.TEMP_SCREENSHOTS_DIR.exists():
            for child in config.TEMP_SCREENSHOTS_DIR.iterdir():
                try:
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
                    removed_temp += 1
                except Exception:
                    pass

    print(f"[Reset] Removed {removed_state} state file(s)" + (f" and {removed_temp} log/temp item(s)" if clear_logs else "") + ".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset YellowAI local Ollama state.")
    parser.add_argument("--clear-logs", action="store_true", help="Also clear logs/ and temp/screenshots/.")
    parser.add_argument("--keep-wisdom", action="store_true", help="Keep distilled thinking_insights.json and web cache.")
    args = parser.parse_args()
    reset_all_progress(clear_logs=args.clear_logs, keep_wisdom=args.keep_wisdom)
