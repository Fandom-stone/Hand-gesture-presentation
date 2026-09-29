"""
Keystroke delivery, plus a helper for checking which window received it.

pyautogui sends keys to whichever window currently has focus, so a gesture
can be recognised perfectly and still do nothing if the camera preview -- or
any other window -- is in front. `foreground_window_title()` makes that
visible instead of leaving it as a mystery.
"""

from __future__ import annotations

import sys

_IS_WINDOWS = sys.platform == "win32"


def press_keys(keys) -> None:
    """Press a single key, or a chord such as ("ctrl", "f5")."""
    import pyautogui

    if isinstance(keys, str):
        pyautogui.press(keys)
    else:
        pyautogui.hotkey(*keys)


def foreground_window_title() -> str:
    """Title of the window that currently has keyboard focus.

    Returns "" when it cannot be determined (non-Windows, or the call fails).
    """
    if not _IS_WINDOWS:
        return ""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:  # noqa: BLE001 - diagnostics only, never fatal
        return ""
