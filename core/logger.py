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
        self.terminal_stdout.write(message)
        self.terminal_stdout.flush()
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

def setup_logging(base_dir: Path) -> TeeLogger:
    """Initializes dual logging to console and logs/ directory."""
    log_dir = base_dir / "logs"
    tee = TeeLogger(log_dir)
    sys.stdout = tee
    sys.stderr = tee
    print(f"[Logger] Session logging initialized -> {tee.log_file_path}")
    return tee
