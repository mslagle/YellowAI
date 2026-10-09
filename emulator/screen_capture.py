"""
Screen capture module for YellowAI (Optimized for mGBA).
Locates mGBA using Win32 Desktop enumeration, Process ID matching, and PrintWindow / BitBlt APIs.
Accurately crops emulator menu bars for pixel-perfect GBA gameplay frames (3:2 aspect ratio).
Supports Per-Monitor DPI awareness, dynamic emulator window scaling (1x, 2x, 3x, etc.),
and dedicated worker thread desktop isolation handling.
"""

import ctypes
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np

# Set DPI awareness early so Windows doesn't virtualize window dimensions
try:
    user32 = ctypes.windll.user32
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', wintypes.DWORD),
        ('biWidth', wintypes.LONG),
        ('biHeight', wintypes.LONG),
        ('biPlanes', wintypes.WORD),
        ('biBitCount', wintypes.WORD),
        ('biCompression', wintypes.DWORD),
        ('biSizeImage', wintypes.DWORD),
        ('biXPelsPerMeter', wintypes.LONG),
        ('biYPelsPerMeter', wintypes.LONG),
        ('biClrUsed', wintypes.DWORD),
        ('biClrImportant', wintypes.DWORD)
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', wintypes.DWORD * 3)]


def _create_bmi(width: int, height: int) -> BITMAPINFO:
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height  # top-down DIB
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    return bmi


class ScreenCapture:
    def __init__(self, target_width: int = 480, target_height: int = 320, window_keywords: list = None):
        self.target_width = target_width
        self.target_height = target_height
        self.window_keywords = window_keywords or ["mgba", "recharged", "yellow", "pokemon", "pokémon", "fire"]
        self.target_hwnd: Optional[int] = None
        self.target_title: str = ""
        self.target_pid: Optional[int] = None
        # Dedicated worker thread guarantees clean desktop attachment without ERROR_BUSY (170)
        self._executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _attach_desktop():
        """Attaches the current worker thread to the interactive user desktop 'default' on WinSta0."""
        try:
            u32 = ctypes.windll.user32
            h_default = u32.OpenDesktopW('default', 0, False, 0x01FF)
            if h_default:
                u32.SetThreadDesktop(h_default)
            return h_default
        except Exception:
            return None

    def _find_window_worker(self) -> Optional[Tuple[int, int, str]]:
        self._attach_desktop()
        user32 = ctypes.windll.user32

        found_windows = []
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def get_proc_image_name(pid_val):
            try:
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                h_proc = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid_val)
                if h_proc:
                    buf = ctypes.create_unicode_buffer(1024)
                    size = wintypes.DWORD(1024)
                    if ctypes.windll.kernel32.QueryFullProcessImageNameW(h_proc, 0, buf, ctypes.byref(size)):
                        ctypes.windll.kernel32.CloseHandle(h_proc)
                        return buf.value.lower()
                    ctypes.windll.kernel32.CloseHandle(h_proc)
            except Exception:
                pass
            return ""

        def enum_cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True

            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value.strip()

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top

            if w > 150 and h > 150:
                proc_img = get_proc_image_name(pid.value)
                # Strictly exclude IDEs, chat agents, and browsers
                if any(x in proc_img for x in ["antigravity", "code.exe", "cursor.exe", "chrome.exe", "firefox.exe", "msedge.exe"]):
                    return True

                title_lower = title.lower()
                if "mgba" in proc_img or "mgba" in title_lower:
                    found_windows.append((hwnd, pid.value, title, 100))
                else:
                    for kw in self.window_keywords:
                        if kw in title_lower and not any(x in proc_img for x in ["electron", "browser", "editor"]):
                            found_windows.append((hwnd, pid.value, title, 50))
                            break
            return True

        h_default = user32.OpenDesktopW('default', 0, False, 0x01FF)
        if h_default:
            user32.EnumDesktopWindows(h_default, WNDENUMPROC(enum_cb), 0)
        else:
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

        if found_windows:
            found_windows.sort(key=lambda x: x[3], reverse=True)
            best_hwnd, best_pid, best_title, _ = found_windows[0]
            return best_hwnd, best_pid, best_title

        return None

    def find_emulator_window(self) -> Optional[int]:
        """
        Searches for an open mGBA window across the interactive desktop.
        Matches by process name ('mGBA.exe') and/or window title.
        Returns the window handle (HWND) or None.
        """
        future = self._executor.submit(self._find_window_worker)
        res = future.result()
        if res:
            self.target_hwnd, self.target_pid, self.target_title = res
            print(f"[ScreenCapture] Successfully found and attached to mGBA: '{self.target_title}' (HWND: {self.target_hwnd})")
            return self.target_hwnd

        self.target_hwnd = None
        self.target_title = ""
        self.target_pid = None
        return None

    def is_window_alive(self) -> bool:
        """Strict check: verifies the emulator window exists and is visible."""
        if not self.target_hwnd:
            self.find_emulator_window()
            if not self.target_hwnd:
                return False

        try:
            user32 = ctypes.windll.user32
            is_valid = bool(user32.IsWindow(self.target_hwnd))
            is_visible = bool(user32.IsWindowVisible(self.target_hwnd))
            return is_valid and is_visible
        except Exception:
            return False

    def _capture_frame_worker(self) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        self._attach_desktop()
        hwnd = self.target_hwnd
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        c_rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(c_rect))
        cw = c_rect.right - c_rect.left
        ch = c_rect.bottom - c_rect.top

        if cw <= 50 or ch <= 50:
            return None, None

        # 1. Capture client area using PrintWindow with PW_CLIENTONLY
        hwnd_dc = user32.GetWindowDC(hwnd)
        mfc_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        save_bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, cw, ch)
        gdi32.SelectObject(mfc_dc, save_bitmap)

        pw_success = user32.PrintWindow(hwnd, mfc_dc, PW_CLIENTONLY)

        bmi = _create_bmi(cw, ch)
        buffer = ctypes.create_string_buffer(cw * ch * 4)
        gdi32.GetDIBits(mfc_dc, save_bitmap, 0, ch, buffer, ctypes.byref(bmi), 0)

        gdi32.DeleteObject(save_bitmap)
        gdi32.DeleteDC(mfc_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)

        client_img = Image.frombuffer('RGBA', (cw, ch), buffer, 'raw', 'BGRA', 0, 1).convert('RGB')

        # Fallback to desktop BitBlt if PrintWindow returned black or failed
        arr_check = np.array(client_img)
        if not pw_success or arr_check.max() < 10:
            pt = wintypes.POINT(0, 0)
            user32.ClientToScreen(hwnd, ctypes.byref(pt))

            sdc = user32.GetDC(0)
            mdc = gdi32.CreateCompatibleDC(sdc)
            bmp = gdi32.CreateCompatibleBitmap(sdc, cw, ch)
            gdi32.SelectObject(mdc, bmp)
            gdi32.BitBlt(mdc, 0, 0, cw, ch, sdc, pt.x, pt.y, SRCCOPY)
            bmi_fallback = _create_bmi(cw, ch)
            gdi32.GetDIBits(mdc, bmp, 0, ch, buffer, ctypes.byref(bmi_fallback), 0)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mdc)
            user32.ReleaseDC(0, sdc)
            client_img = Image.frombuffer('RGBA', (cw, ch), buffer, 'raw', 'BGRA', 0, 1).convert('RGB')

        # 2. Dynamically determine Qt menu bar height
        menu_h = 0
        scale = round(cw / 240)
        expected_h = scale * 160
        if scale >= 1 and abs(cw - scale * 240) <= 4 and ch >= expected_h:
            menu_h = ch - expected_h
        else:
            # Scan for menu bar transition in rows 15..55
            arr = np.array(client_img)
            scan_limit = min(55, ch - 50)
            for y in range(15, scan_limit):
                if arr[y, :, :].mean() > 200 and arr[y + 1, :, :].mean() <= 200:
                    menu_h = y + 1
                    break

        # 3. Compute 3:2 GBA canvas viewport within available game area
        avail_w = cw
        avail_h = max(10, ch - menu_h)

        if avail_w / avail_h > 1.5:
            # Pillarboxed (black bars left and right)
            game_h = avail_h
            game_w = int(round(avail_h * 1.5))
            game_x = (avail_w - game_w) // 2
            game_y = menu_h
        else:
            # Letterboxed (black bars top and bottom)
            game_w = avail_w
            game_h = int(round(avail_w / 1.5))
            game_x = 0
            game_y = menu_h + (avail_h - game_h) // 2

        gameplay_img = client_img.crop((game_x, game_y, game_x + game_w, game_y + game_h))

        # Standardized resize to target dimensions (480x320)
        resized_img = gameplay_img.resize((self.target_width, self.target_height), Image.Resampling.BILINEAR)
        thumb = np.array(resized_img.convert("L").resize((32, 32), Image.Resampling.NEAREST))
        return resized_img, thumb

    def capture_frame(self) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """
        Captures the live mGBA gameplay frame.
        Captures client area directly, strips Qt menu bar, maintains strict 3:2 GBA aspect ratio.
        Returns (PIL.Image resized to target_width x target_height, numpy thumbnail).
        """
        if not self.is_window_alive():
            return None, None

        try:
            future = self._executor.submit(self._capture_frame_worker)
            return future.result()
        except Exception as e:
            print(f"[ScreenCapture] Capture error: {e}")
            return None, None
