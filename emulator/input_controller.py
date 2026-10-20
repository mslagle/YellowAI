"""
Input controller for YellowAI.
Supports 100% background input injection into mGBA without stealing Windows focus:
- "postmessage": Sends Win32 WM_KEYDOWN / WM_KEYUP messages directly to the mGBA HWND (and child Qt widgets).
- "lua_socket": Sends button commands over TCP to mGBA's yellowai_bridge.lua (falls back to postmessage if disconnected).
- "foreground": Legacy mode that grabs foreground window focus and uses pydirectinput.
"""

import socket
import time
import ctypes
from ctypes import wintypes
from typing import Dict, List, Optional

try:
    import pydirectinput
    pydirectinput.FAILSAFE = False
    pydirectinput.PAUSE = 0.0
except Exception:
    pydirectinput = None

import config

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_ACTIVATE = 0x0006
WA_ACTIVE = 1
SW_SHOWNOACTIVATE = 4
MAPVK_VK_TO_VSC = 0

# Virtual Key Code lookup for mGBA key bindings
VK_CODE_MAP: Dict[str, int] = {
    "up": 0x26,         # VK_UP
    "down": 0x28,       # VK_DOWN
    "left": 0x25,       # VK_LEFT
    "right": 0x27,      # VK_RIGHT
    "enter": 0x0D,      # VK_RETURN
    "return": 0x0D,
    "backspace": 0x08,  # VK_BACK
    "space": 0x20,      # VK_SPACE
    "shift": 0x10,      # VK_SHIFT
    "ctrl": 0x11,       # VK_CONTROL
    "tab": 0x09,        # VK_TAB
    "esc": 0x1B,        # VK_ESCAPE
    "escape": 0x1B,
}

# Arrow keys and navigation keys require the extended-key bit (bit 24) in lParam
EXTENDED_VKS = {0x25, 0x26, 0x27, 0x28}


def _key_name_to_vk(key_name: str) -> Optional[int]:
    """Resolves a key name string (e.g. 'up', 'x', 'z', 'enter') to a Win32 Virtual-Key code."""
    if not key_name:
        return None
    k = key_name.lower().strip()
    if k in VK_CODE_MAP:
        return VK_CODE_MAP[k]
    if len(k) == 1 and k.isalnum():
        return ord(k.upper())
    return None


class InputController:
    def __init__(
        self,
        key_mappings: Dict[str, Optional[str]],
        hold_duration: float = 0.05,
        target_hwnd: Optional[int] = None,
        input_mode: Optional[str] = None,
    ):
        self.key_mappings = key_mappings
        self.hold_duration = hold_duration
        self.target_hwnd = target_hwnd
        self.input_mode = (input_mode or getattr(config, "BACKGROUND_INPUT_MODE", "postmessage")).lower().strip()
        self._activated_hwnds: set = set()
        self._lua_sock: Optional[socket.socket] = None
        self._lua_last_retry: float = 0.0

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

    def _get_target_hwnds(self, hwnd: int) -> List[int]:
        """Returns the main HWND plus any direct child windows (e.g. Qt QOpenGLWidget / viewport)."""
        targets = [hwnd]
        try:
            user32 = ctypes.windll.user32
            WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            def enum_child_cb(child_hwnd, lparam):
                if child_hwnd and child_hwnd not in targets:
                    targets.append(child_hwnd)
                return True

            user32.EnumChildWindows(hwnd, WNDENUMPROC(enum_child_cb), 0)
        except Exception:
            pass
        return targets

    def _prepare_background_window(self, hwnd: int):
        """
        Ensures the mGBA window is un-minimized without stealing focus, and sends a synthetic
        WM_ACTIVATE(WA_ACTIVE) once so Qt processes background key messages reliably.
        """
        try:
            user32 = ctypes.windll.user32
            if bool(user32.IsIconic(hwnd)):
                user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
            if hwnd not in self._activated_hwnds:
                for h in self._get_target_hwnds(hwnd):
                    user32.PostMessageW(h, WM_ACTIVATE, WA_ACTIVE, 0)
                self._activated_hwnds.add(hwnd)
        except Exception:
            pass

    @staticmethod
    def _build_key_lparam(vk: int, is_keyup: bool) -> int:
        """Constructs the Win32 WM_KEYDOWN / WM_KEYUP lParam bitfield including hardware scan code."""
        try:
            scan_code = ctypes.windll.user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) & 0xFF
        except Exception:
            scan_code = 0
        lparam = 1 | (scan_code << 16)
        if vk in EXTENDED_VKS:
            lparam |= (1 << 24)
        if is_keyup:
            lparam |= (1 << 30) | (1 << 31)
        return lparam

    def _post_key_event(self, hwnd: int, key_name: str, is_keyup: bool) -> bool:
        """Sends WM_KEYDOWN or WM_KEYUP to the mGBA window and its child widgets without stealing focus."""
        vk = _key_name_to_vk(key_name)
        if vk is None or not hwnd:
            return False

        self._ensure_desktop()
        self._prepare_background_window(hwnd)

        user32 = ctypes.windll.user32
        msg = WM_KEYUP if is_keyup else WM_KEYDOWN
        lparam = self._build_key_lparam(vk, is_keyup)

        sent = False
        for target in self._get_target_hwnds(hwnd):
            if user32.PostMessageW(target, msg, vk, lparam):
                sent = True
        return sent

    def _send_lua_command(self, cmd: str) -> bool:
        """Sends a command to yellowai_bridge.lua over TCP socket if connected."""
        now = time.time()
        if self._lua_sock is None:
            if now - self._lua_last_retry < 5.0:
                return False
            self._lua_last_retry = now
            try:
                host = getattr(config, "LUA_BRIDGE_HOST", "127.0.0.1")
                port = getattr(config, "LUA_BRIDGE_PORT", 8888)
                sock = socket.create_connection((host, port), timeout=0.15)
                sock.settimeout(0.15)
                self._lua_sock = sock
            except Exception:
                self._lua_sock = None
                return False

        try:
            self._lua_sock.sendall((cmd.strip() + "\n").encode("ascii"))
            return True
        except Exception:
            try:
                self._lua_sock.close()
            except Exception:
                pass
            self._lua_sock = None
            return False

    def is_in_focus(self, hwnd: Optional[int] = None) -> bool:
        """
        In background modes ('postmessage' / 'lua_socket'), returns True as long as the window exists
        and is not minimized, since foreground focus is not required.
        """
        target = hwnd or self.target_hwnd
        if not target:
            return False

        self._ensure_desktop()
        try:
            user32 = ctypes.windll.user32
            if self.input_mode != "foreground":
                return bool(user32.IsWindow(target)) and not bool(user32.IsIconic(target))
            fg = user32.GetForegroundWindow()
            is_iconic = bool(user32.IsIconic(target))
            return fg == target and not is_iconic
        except Exception:
            return False

    def ensure_focus(self, hwnd: Optional[int] = None) -> bool:
        """
        In background modes ('postmessage' / 'lua_socket'), prepares the window in the background
        without stealing foreground focus.
        In 'foreground' mode, brings the emulator window to the foreground.
        """
        target = hwnd or self.target_hwnd
        if not target:
            return False

        self._ensure_desktop()

        if self.input_mode != "foreground":
            self._prepare_background_window(target)
            return True

        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            fg_hwnd = user32.GetForegroundWindow()
            is_iconic = bool(user32.IsIconic(target))

            if fg_hwnd == target and not is_iconic:
                return True

            fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
            cur_thread = kernel32.GetCurrentThreadId()
            target_thread = user32.GetWindowThreadProcessId(target, None)

            user32.keybd_event(0x12, 0, 0, 0)
            user32.keybd_event(0x12, 0, 2, 0)

            if fg_thread and fg_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, fg_thread, True)
            if target_thread and target_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, target_thread, True)

            user32.ShowWindow(target, 9)
            user32.BringWindowToTop(target)
            user32.SetForegroundWindow(target)
            user32.SetFocus(target)

            if fg_thread and fg_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, fg_thread, False)
            if target_thread and target_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, target_thread, False)

            time.sleep(0.06)
            return user32.GetForegroundWindow() == target
        except Exception:
            return False

    def focus_hwnd(self, hwnd: int) -> bool:
        """Prepares the specific emulator window for input (without stealing focus in background mode)."""
        return self.ensure_focus(hwnd)

    def _press_key_internal(self, button: str, key: str, duration: float) -> bool:
        """Dispatches a single button press via Lua socket, background PostMessageW, or foreground DirectInput."""
        if self.input_mode == "lua_socket":
            if self._send_lua_command(f"DOWN:{button}"):
                time.sleep(duration)
                self._send_lua_command(f"UP:{button}")
                return True

        if self.input_mode in ("postmessage", "lua_socket") and self.target_hwnd:
            ok_down = self._post_key_event(self.target_hwnd, key, is_keyup=False)
            time.sleep(duration)
            ok_up = self._post_key_event(self.target_hwnd, key, is_keyup=True)
            return ok_down or ok_up

        # Fallback or explicit "foreground" mode
        if self.target_hwnd and self.input_mode == "foreground":
            self.ensure_focus(self.target_hwnd)
        if pydirectinput is not None:
            pydirectinput.keyDown(key)
            time.sleep(duration)
            pydirectinput.keyUp(key)
            return True
        return False

    def press_button(self, button: str, hold_override: Optional[float] = None) -> bool:
        """
        Presses and releases the requested GBA button.
        Uses background PostMessageW or Lua socket by default so mGBA never needs foreground focus.
        Returns True if a key was pressed, False if WAIT or invalid.
        """
        button = button.upper().strip()
        if button == "WAIT" or button not in self.key_mappings:
            return False

        key = self.key_mappings.get(button)
        if not key:
            return False

        duration = hold_override if hold_override is not None else self.hold_duration

        try:
            return self._press_key_internal(button, key, duration)
        except Exception as e:
            print(f"[InputController] Error sending key {key} for button {button}: {e}")
            return False

    def emergency_menu_escape(self, presses: int = 4, try_move_away: bool = True):
        """
        Attempts to break out of stuck dialogues, menus, or unclosable inspection signs (like mailboxes).
        Tries pressing B, START, and walking away (UP or away from obstacle) using background-safe input.
        """
        print(f"[InputController] Triggering Emergency Escape ({presses}x B + START/UP escape)...")

        for _ in range(presses):
            self.press_button("B", hold_override=0.08)
            time.sleep(0.12)

        if try_move_away:
            self.press_button("START", hold_override=0.08)
            time.sleep(0.15)
            self.press_button("B", hold_override=0.08)
            time.sleep(0.12)
            self.press_button("UP", hold_override=0.15)
            time.sleep(0.12)

