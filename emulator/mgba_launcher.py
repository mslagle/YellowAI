"""
mGBA Emulator Launcher & Process Manager for YellowAI.
Launches the bundled mGBA-0.10.5-win64 emulator with Pokémon Yellow Recharged,
optionally loads a savestate, and waits until the mGBA window handle (HWND) is live.

CLI Usage:
    python -m emulator.mgba_launcher
    python -m emulator.mgba_launcher --scale 3 --savestate "mGBA-0.10.5-win64/games/rchrgd yllw v1.8.0.ss0"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import config
from emulator.screen_capture import ScreenCapture


def find_mgba_executable() -> Optional[Path]:
    """Locates mGBA.exe in the workspace."""
    if config.MGBA_EXE.exists():
        return config.MGBA_EXE
    for candidate in config.BASE_DIR.glob("mGBA*/mGBA.exe"):
        return candidate
    return None


def find_rom_file(rom_path: Optional[str] = None) -> Optional[Path]:
    """Locates the Pokémon Yellow Recharged ROM file."""
    if rom_path:
        p = Path(rom_path)
        if p.exists():
            return p
    if config.DEFAULT_ROM_PATH.exists():
        return config.DEFAULT_ROM_PATH
    for ext in ("*.gba", "*.gbc", "*.gb"):
        matches = list((config.MGBA_DIR / "games").glob(ext))
        if matches:
            return matches[0]
    return None


def launch_mgba(
    rom_path: Optional[str] = None,
    savestate_path: Optional[str] = None,
    scale: int = 2,
    wait_timeout: float = 15.0,
) -> Optional[int]:
    """
    Ensures mGBA is running with Pokémon Yellow Recharged loaded.
    If already running, returns the existing HWND; otherwise launches mGBA and waits for HWND.
    """
    cap = ScreenCapture(window_keywords=config.EMULATOR_WINDOW_TITLE_KEYWORDS)
    existing_hwnd = cap.find_emulator_window()
    if existing_hwnd:
        print(f"[mGBA Launcher] mGBA already running: '{cap.target_title}' (HWND: {existing_hwnd})")
        return existing_hwnd

    mgba_exe = find_mgba_executable()
    if not mgba_exe:
        print(f"[mGBA Launcher] ERROR: mGBA.exe not found in {config.MGBA_DIR}")
        return None

    rom = find_rom_file(rom_path)
    if not rom:
        print(f"[mGBA Launcher] ERROR: Pokémon Yellow Recharged ROM not found in {config.MGBA_DIR / 'games'}")
        return None

    cmd = [str(mgba_exe)]
    if 1 <= scale <= 8:
        cmd.append(f"-{scale}")
    if savestate_path:
        ss = Path(savestate_path)
        if ss.exists():
            cmd.extend(["-t", str(ss)])
    cmd.append(str(rom))

    print(f"[mGBA Launcher] Starting mGBA: {' '.join(cmd)}")
    creation_flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        cmd,
        cwd=str(mgba_exe.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        hwnd = cap.find_emulator_window()
        if hwnd:
            print(f"[mGBA Launcher] Ready: '{cap.target_title}' (HWND: {hwnd})")
            return hwnd
        time.sleep(0.4)

    print("[mGBA Launcher] Warning: Launched mGBA, but window handle was not detected within timeout.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch mGBA with Pokémon Yellow Recharged")
    parser.add_argument("--rom", type=str, default=None, help="Path to .gba ROM file")
    parser.add_argument("--savestate", type=str, default=None, help="Optional .ss0 savestate to load on boot")
    parser.add_argument("--scale", type=int, default=2, help="Viewport scale multiplier (1-8, default: 2)")
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for mGBA window")
    args = parser.parse_args()

    hwnd = launch_mgba(
        rom_path=args.rom,
        savestate_path=args.savestate,
        scale=args.scale,
        wait_timeout=args.timeout,
    )
    if not hwnd:
        sys.exit(1)


if __name__ == "__main__":
    main()
