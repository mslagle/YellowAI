"""
Input controller for YellowAI.
Injects keyboard inputs into mGBA using DirectInput and Win32 focus controls.
Automatically detects when the mGBA emulator window is not in focus, un-minimizes it,
and grabs foreground control before injecting any button inputs.
"""

import time
import ctypes
from typing import Dict, Optional
import pydirectinput

pydirectinput.FAILSAFE = False


class InputController:
    def __init__(
        self,
        key_mappings: Dict[str, Optional[str]],
        hold_duration: float = 0.12,
        target_hwnd: Optional[int] = None
    ):
        self.key_mappings = key_mappings
        self.hold_duration = hold_duration
        self.target_hwnd = target_hwnd

    def set_target_hwnd(self, hwnd: int):
        """Sets or updates the target emulator window handle (HWND)."""
        self.target_hwnd = hwnd

    def _ensure_desktop(self):
        """Attaches the thread to the interactive user desktop 'default' on WinSta0."""
        try:
            user32 = ctypes.windll.user32
            h_default = user32.OpenDesktopW('default', 0, False, 0x01FF)
            if h_default:
                user32.SetThreadDesktop(h_default)
        except Exception:
            pass

    def is_in_focus(self, hwnd: Optional[int] = None) -> bool:
        """Checks if the emulator window is currently the active foreground window."""
        target = hwnd or self.target_hwnd
        if not target:
            return False

        self._ensure_desktop()
        try:
            user32 = ctypes.windll.user32
            fg = user32.GetForegroundWindow()
            is_iconic = bool(user32.IsIconic(target))
            return fg == target and not is_iconic
        except Exception:
            return False

    def ensure_focus(self, hwnd: Optional[int] = None) -> bool:
        """
        Automatically grabs foreground control of the emulator window if it is not in focus.
        Un-minimizes if minimized (SW_RESTORE) and uses the Win32 AttachThreadInput + Alt key release
        technique to reliably bypass Windows background focus-stealing restrictions.
        """
        target = hwnd or self.target_hwnd
        if not target:
            return False

        self._ensure_desktop()

        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            fg_hwnd = user32.GetForegroundWindow()
            is_iconic = bool(user32.IsIconic(target))

            # Fast path: already focused and visible
            if fg_hwnd == target and not is_iconic:
                return True

            fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
            cur_thread = kernel32.GetCurrentThreadId()
            target_thread = user32.GetWindowThreadProcessId(target, None)

            # 1. Simulate Alt key release to unlock Windows SetForegroundWindow permissions
            user32.keybd_event(0x12, 0, 0, 0)  # VK_MENU down
            user32.keybd_event(0x12, 0, 2, 0)  # VK_MENU up (KEYEVENTF_KEYUP)

            # 2. Attach thread input to foreground thread and target thread
            if fg_thread and fg_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, fg_thread, True)
            if target_thread and target_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, target_thread, True)

            # 3. Restore window if minimized and bring directly to top
            user32.ShowWindow(target, 9)  # SW_RESTORE
            user32.BringWindowToTop(target)
            user32.SetForegroundWindow(target)
            user32.SetFocus(target)

            # 4. Detach thread input
            if fg_thread and fg_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, fg_thread, False)
            if target_thread and target_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, target_thread, False)

            time.sleep(0.06)
            in_focus = user32.GetForegroundWindow() == target
            if in_focus:
                print(f"[InputController] Automatically grabbed focus of mGBA (HWND: {target})")
            return in_focus

        except Exception as e:
            print(f"[InputController] Error grabbing focus: {e}")
            return False

    def focus_hwnd(self, hwnd: int) -> bool:
        """Brings the specific emulator window to the foreground (alias for ensure_focus)."""
        return self.ensure_focus(hwnd)

    def press_button(self, button: str, hold_override: Optional[float] = None) -> bool:
        """
        Presses and releases the requested GBA button.
        Automatically verifies that mGBA is in focus before sending the keystroke.
        Returns True if a key was pressed, False if WAIT or invalid.
        """
        button = button.upper().strip()
        if button == "WAIT" or button not in self.key_mappings:
            return False

        key = self.key_mappings.get(button)
        if not key:
            return False

        # Automatically ensure mGBA is focused before injecting input
        if self.target_hwnd:
            self.ensure_focus(self.target_hwnd)

        duration = hold_override if hold_override is not None else self.hold_duration

        try:
            pydirectinput.keyDown(key)
            time.sleep(duration)
            pydirectinput.keyUp(key)
            return True
        except Exception as e:
            print(f"[InputController] Error sending key {key} for button {button}: {e}")
            return False

    def emergency_menu_escape(self, presses: int = 4, try_move_away: bool = True):
        """
        Attempts to break out of stuck dialogues, menus, or unclosable inspection signs (like mailboxes).
        Tries pressing B, START, and walking away (UP or away from obstacle).
        """
        print(f"[InputController] Triggering Emergency Escape ({presses}x B + START/UP escape)...")
        if self.target_hwnd:
            self.ensure_focus(self.target_hwnd)

        b_key = self.key_mappings.get("B", "z")
        start_key = self.key_mappings.get("START", "enter")
        up_key = self.key_mappings.get("UP", "up")

        # 1. First try pressing B to close standard menus/dialogs
        if b_key:
            for _ in range(presses):
                try:
                    pydirectinput.keyDown(b_key)
                    time.sleep(0.08)
                    pydirectinput.keyUp(b_key)
                    time.sleep(0.12)
                except Exception:
                    pass

        if try_move_away:
            # 2. Try START to toggle/close stuck dialogs
            try:
                if start_key:
                    pydirectinput.keyDown(start_key)
                    time.sleep(0.08)
                    pydirectinput.keyUp(start_key)
                    time.sleep(0.15)
                    # Close start menu if opened
                    if b_key:
                        pydirectinput.keyDown(b_key)
                        time.sleep(0.08)
                        pydirectinput.keyUp(b_key)
                        time.sleep(0.12)
            except Exception:
                pass

            # 3. Step away (UP to back out of south-facing obstacles like mailboxes)
            try:
                if up_key:
                    pydirectinput.keyDown(up_key)
                    time.sleep(0.15)
                    pydirectinput.keyUp(up_key)
                    time.sleep(0.12)
            except Exception:
                pass
