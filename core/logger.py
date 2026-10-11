"""
Logging utility for YellowAI.
Tees all stdout/stderr output to both the console and timestamped log files.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

class TeeLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.log_dir / f"yellowai_{timestamp}.log"
        self.latest_log_path = self.log_dir / "latest.log"

        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr

        self.log_file = open(self.log_file_path, "a", encoding="utf-8", buffering=1)
        self.latest_file = open(self.latest_log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        try:
            self.terminal_stdout.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal_stdout, "encoding", "ascii") or "ascii"
            safe_msg = message.encode(encoding, errors="replace").decode(encoding)
            self.terminal_stdout.write(safe_msg)
        self.terminal_stdout.flush()
        try:
            self.log_file.write(message)
            self.log_file.flush()
            self.latest_file.write(message)
            self.latest_file.flush()
        except Exception:
            pass

    def write_file_only(self, message):
        """Writes message strictly to log files, keeping it out of the console terminal."""
        try:
            self.log_file.write(message)
            self.log_file.flush()
            self.latest_file.write(message)
            self.latest_file.flush()
        except Exception:
            pass

    def flush(self):
        self.terminal_stdout.flush()
        try:
            self.log_file.flush()
            self.latest_file.flush()
        except Exception:
            pass

_global_tee: TeeLogger = None

def setup_logging(base_dir: Path) -> TeeLogger:
    """Initializes dual logging to console and logs/ directory."""
    global _global_tee
    log_dir = base_dir / "logs"
    tee = TeeLogger(log_dir)
    sys.stdout = tee
    sys.stderr = tee
    _global_tee = tee
    print(f"[Logger] Session logging initialized -> {tee.log_file_path}")
    return tee

def log_file_only(message: str):
    """Logs message to log files only without writing to console stdout."""
    global _global_tee
    msg = message if message.endswith("\n") else f"{message}\n"
    if _global_tee is not None:
        _global_tee.write_file_only(msg)
    else:
        # Fallback if logger hasn't been set up yet
        pass
