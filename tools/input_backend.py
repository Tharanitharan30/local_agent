import os
import time
from typing import Dict, List, Optional, Tuple, Any

import config

# Standard X11 / XKB Keysym mappings
KEYSYM_MAP: Dict[str, int] = {
    # Whitespace & Navigation
    "enter": 0xFF0D,
    "return": 0xFF0D,
    "tab": 0xFF09,
    "space": 0x0020,
    "backspace": 0xFF08,
    "delete": 0xFFFF,
    "del": 0xFFFF,
    "escape": 0xFF1B,
    "esc": 0xFF1B,
    "home": 0xFF50,
    "end": 0xFF57,
    "left": 0xFF51,
    "up": 0xFF52,
    "right": 0xFF53,
    "down": 0xFF54,
    "pageup": 0xFF55,
    "page_up": 0xFF55,
    "pagedown": 0xFF56,
    "page_down": 0xFF56,
    # Modifiers
    "ctrl": 0xFFE3,
    "control": 0xFFE3,
    "ctrl_l": 0xFFE3,
    "ctrl_r": 0xFFE4,
    "alt": 0xFFE9,
    "alt_l": 0xFFE9,
    "alt_r": 0xFFEA,
    "shift": 0xFFE1,
    "shift_l": 0xFFE1,
    "shift_r": 0xFFE2,
    "super": 0xFFEB,
    "win": 0xFFEB,
    "meta": 0xFFEB,
    "insert": 0xFF63,
    "capslock": 0xFFE5,
    "caps_lock": 0xFFE5,
}

# Add Function keys F1 - F12 (0xFFBE - 0xFFC9)
for _i in range(1, 13):
    KEYSYM_MAP[f"f{_i}"] = 0xFFBE + (_i - 1)

# Linux evdev button codes
BUTTON_CODES: Dict[str, int] = {
    "left": 0x110,   # BTN_LEFT / 272
    "right": 0x111,  # BTN_RIGHT / 273
    "middle": 0x112, # BTN_MIDDLE / 274
}


class DesktopInputBackend:
    """
    Unified Linux Desktop Input Controller.
    Primary backend: FreeDesktop XDG Desktop Portal (org.freedesktop.portal.RemoteDesktop) over D-Bus via jeepney.
    This provides native, non-root Wayland pointer and keyboard injection on GNOME, KDE, and modern compositors.
    Fallback backend: pyautogui / pynput for Xwayland / X11.
    """

    def __init__(self):
        self._conn = None
        self._session_path: Optional[str] = None
        self._portal_available = False
        self._screen_width = 1920
        self._screen_height = 1080
        self._init_screen_size()
        self._init_portal()

    def _init_screen_size(self) -> None:
        """Query screen dimensions via pyautogui or fallback to 1920x1080."""
        try:
            import pyautogui
            size = pyautogui.size()
            self._screen_width = int(size.width)
            self._screen_height = int(size.height)
        except Exception:
            self._screen_width = 1920
            self._screen_height = 1080

    def _init_portal(self) -> bool:
        """Initializes and authenticates the FreeDesktop RemoteDesktop session."""
        try:
            import socket
            from jeepney import new_method_call, DBusAddress
            from jeepney.io.blocking import open_dbus_connection

            conn = open_dbus_connection(bus="SESSION")
            if hasattr(conn, "sock") and conn.sock:
                conn.sock.settimeout(1.0)

            # Listen for Response signal
            add_match_msg = new_method_call(
                DBusAddress("/org/freedesktop/DBus", "org.freedesktop.DBus", "org.freedesktop.DBus"),
                "AddMatch",
                "s",
                ("type='signal',interface='org.freedesktop.portal.Request',member='Response'",),
            )
            conn.send_and_get_reply(add_match_msg)

            # 1. CreateSession
            token = f"zia_{int(time.time() * 1000)}"
            msg = new_method_call(
                DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                "CreateSession",
                "a{sv}",
                ({"session_handle_token": ("s", token)},),
            )
            reply = conn.send_and_get_reply(msg)
            if not reply.body:
                conn.close()
                return False

            req_path = reply.body[0]
            session_path = None
            timeout_end = time.time() + 2.0
            while time.time() < timeout_end:
                try:
                    sig = conn.receive()
                    if sig.header.message_type.name == "signal" and sig.header.fields.get(1) == req_path:
                        code, results = sig.body
                        if code == 0 and "session_handle" in results:
                            session_path = results["session_handle"][1]
                        break
                except (TimeoutError, socket.timeout):
                    continue

            if not session_path:
                conn.close()
                return False

            # 2. SelectDevices (Pointer = 1, Keyboard = 2, Touchscreen = 4 => 7)
            dev_token = f"dev_{int(time.time() * 1000)}"
            msg2 = new_method_call(
                DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                "SelectDevices",
                "oa{sv}",
                (session_path, {"handle_token": ("s", dev_token), "types": ("u", 7)}),
            )
            reply2 = conn.send_and_get_reply(msg2)
            req2_path = reply2.body[0]
            dev_selected = False
            timeout_end = time.time() + 2.0
            while time.time() < timeout_end:
                try:
                    sig = conn.receive()
                    if sig.header.message_type.name == "signal" and sig.header.fields.get(1) == req2_path:
                        code, results = sig.body
                        dev_selected = (code == 0)
                        break
                except (TimeoutError, socket.timeout):
                    continue

            if not dev_selected:
                conn.close()
                return False

            # 3. Start Session
            start_token = f"start_{int(time.time() * 1000)}"
            msg3 = new_method_call(
                DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                "Start",
                "osa{sv}",
                (session_path, "", {"handle_token": ("s", start_token)}),
            )
            reply3 = conn.send_and_get_reply(msg3)
            req3_path = reply3.body[0]
            started = False
            timeout_end = time.time() + 2.0
            while time.time() < timeout_end:
                try:
                    sig = conn.receive()
                    if sig.header.message_type.name == "signal" and sig.header.fields.get(1) == req3_path:
                        code, results = sig.body
                        started = (code == 0)
                        break
                except (TimeoutError, socket.timeout):
                    continue

            if started:
                self._conn = conn
                self._session_path = session_path
                self._portal_available = True
                return True

            conn.close()
            return False

        except Exception:
            return False

            conn.close()
            return False

        except Exception as e:
            self._portal_available = False
            self._conn = None
            self._session_path = None
            return False

    def get_screen_size(self) -> Tuple[int, int]:
        """Returns (width, height) of active display."""
        return self._screen_width, self._screen_height

    def validate_coordinates(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """Validates that (x, y) coordinates reside within the active display bounds."""
        if x < 0 or x >= self._screen_width or y < 0 or y >= self._screen_height:
            return (
                False,
                f"Coordinates ({x}, {y}) are outside the active screen bounds "
                f"[0, {self._screen_width}) x [0, {self._screen_height}).",
            )
        return True, None

    def move(self, x: int, y: int) -> bool:
        """Moves cursor to absolute screen coordinates (x, y)."""
        valid, err = self.validate_coordinates(x, y)
        if not valid:
            raise ValueError(err)

        if self._portal_available and self._conn and self._session_path:
            try:
                from jeepney import new_method_call, DBusAddress
                msg = new_method_call(
                    DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                    "NotifyPointerMotionAbsolute",
                    "oa{sv}udd",
                    (self._session_path, {}, 0, float(x), float(y)),
                )
                self._conn.send_message(msg)
                time.sleep(config.DEFAULT_MOUSE_DELAY)
                return True
            except Exception:
                pass

        # Fallback to pyautogui
        try:
            import pyautogui
            pyautogui.moveTo(x, y)
            time.sleep(config.DEFAULT_MOUSE_DELAY)
            return True
        except Exception as e:
            raise RuntimeError(f"Mouse move failed: {str(e)}")

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Moves to (x, y) and performs single or multiple clicks with specified mouse button."""
        self.move(x, y)
        btn_norm = button.lower().strip()
        btn_code = BUTTON_CODES.get(btn_norm, 0x110)

        if self._portal_available and self._conn and self._session_path:
            try:
                from jeepney import new_method_call, DBusAddress
                for i in range(clicks):
                    # Button down
                    down_msg = new_method_call(
                        DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                        "NotifyPointerButton",
                        "oa{sv}iu",
                        (self._session_path, {}, btn_code, 1),
                    )
                    self._conn.send_message(down_msg)
                    time.sleep(0.04)

                    # Button up
                    up_msg = new_method_call(
                        DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                        "NotifyPointerButton",
                        "oa{sv}iu",
                        (self._session_path, {}, btn_code, 0),
                    )
                    self._conn.send_message(up_msg)
                    if i < clicks - 1:
                        time.sleep(0.08)
                time.sleep(config.DEFAULT_MOUSE_DELAY)
                return True
            except Exception:
                pass

        # Fallback to pyautogui
        try:
            import pyautogui
            pyautogui.click(x=x, y=y, button=btn_norm, clicks=clicks)
            time.sleep(config.DEFAULT_MOUSE_DELAY)
            return True
        except Exception as e:
            raise RuntimeError(f"Mouse click failed: {str(e)}")

    def scroll(self, steps: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """
        Scrolls vertically by discrete steps.
        Positive steps scroll down; negative steps scroll up.
        """
        if x is not None and y is not None:
            self.move(x, y)

        if self._portal_available and self._conn and self._session_path:
            try:
                from jeepney import new_method_call, DBusAddress
                msg = new_method_call(
                    DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                    "NotifyPointerAxisDiscrete",
                    "oa{sv}ui",
                    (self._session_path, {}, 0, int(steps)),
                )
                self._conn.send_message(msg)
                time.sleep(config.DEFAULT_MOUSE_DELAY)
                return True
            except Exception:
                pass

        try:
            import pyautogui
            # In pyautogui, positive scroll moves up, so invert steps
            pyautogui.scroll(-steps)
            time.sleep(config.DEFAULT_MOUSE_DELAY)
            return True
        except Exception as e:
            raise RuntimeError(f"Mouse scroll failed: {str(e)}")

    def _char_to_keysym(self, char: str) -> int:
        """Translates a character to an X11 keysym integer."""
        if char == "\n":
            return KEYSYM_MAP["enter"]
        if char == "\t":
            return KEYSYM_MAP["tab"]
        if char == "\b":
            return KEYSYM_MAP["backspace"]
        return ord(char)

    def type_text(self, text: str, delay: float = config.DEFAULT_TYPING_DELAY) -> bool:
        """Types raw text string character by character."""
        if not text:
            return True

        if self._portal_available and self._conn and self._session_path:
            try:
                from jeepney import new_method_call, DBusAddress
                for ch in text:
                    sym = self._char_to_keysym(ch)
                    down_msg = new_method_call(
                        DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                        "NotifyKeyboardKeysym",
                        "oa{sv}iu",
                        (self._session_path, {}, sym, 1),
                    )
                    self._conn.send_message(down_msg)
                    if delay > 0:
                        time.sleep(delay / 2)
                    up_msg = new_method_call(
                        DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                        "NotifyKeyboardKeysym",
                        "oa{sv}iu",
                        (self._session_path, {}, sym, 0),
                    )
                    self._conn.send_message(up_msg)
                    if delay > 0:
                        time.sleep(delay / 2)
                return True
            except Exception:
                pass

        # Fallback to pyautogui
        try:
            import pyautogui
            pyautogui.write(text, interval=delay)
            return True
        except Exception as e:
            raise RuntimeError(f"Keyboard type failed: {str(e)}")

    def _resolve_keysym(self, key_name: str) -> int:
        """Resolves a named key or single character to keysym."""
        normalized = key_name.strip().lower()
        if normalized in KEYSYM_MAP:
            return KEYSYM_MAP[normalized]
        if len(key_name) == 1:
            return ord(key_name)
        supported = sorted(list(KEYSYM_MAP.keys()))
        raise ValueError(f"Unknown key '{key_name}'. Supported special keys: {supported}")

    def press_key(self, key: str) -> bool:
        """Presses and releases a single key by name (e.g. 'ENTER', 'TAB', 'ESCAPE')."""
        sym = self._resolve_keysym(key)

        if self._portal_available and self._conn and self._session_path:
            try:
                from jeepney import new_method_call, DBusAddress
                down_msg = new_method_call(
                    DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                    "NotifyKeyboardKeysym",
                    "oa{sv}iu",
                    (self._session_path, {}, sym, 1),
                )
                self._conn.send_message(down_msg)
                time.sleep(0.04)
                up_msg = new_method_call(
                    DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                    "NotifyKeyboardKeysym",
                    "oa{sv}iu",
                    (self._session_path, {}, sym, 0),
                )
                self._conn.send_message(up_msg)
                time.sleep(config.DEFAULT_TYPING_DELAY)
                return True
            except Exception:
                pass

        try:
            import pyautogui
            pyautogui.press(key.lower())
            return True
        except Exception as e:
            raise RuntimeError(f"Key press failed: {str(e)}")

    def hotkey(self, keys: List[str]) -> bool:
        """
        Executes a key combination shortcut (e.g. ['CTRL', 'L'] or ['ALT', 'TAB']).
        Keys are pressed in sequence and released in reverse order.
        """
        if not keys:
            return True

        syms = [self._resolve_keysym(k) for k in keys]

        if self._portal_available and self._conn and self._session_path:
            try:
                from jeepney import new_method_call, DBusAddress
                # Press in sequence
                for sym in syms:
                    down_msg = new_method_call(
                        DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                        "NotifyKeyboardKeysym",
                        "oa{sv}iu",
                        (self._session_path, {}, sym, 1),
                    )
                    self._conn.send_message(down_msg)
                    time.sleep(0.02)

                time.sleep(0.05)

                # Release in reverse order
                for sym in reversed(syms):
                    up_msg = new_method_call(
                        DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.RemoteDesktop"),
                        "NotifyKeyboardKeysym",
                        "oa{sv}iu",
                        (self._session_path, {}, sym, 0),
                    )
                    self._conn.send_message(up_msg)
                    time.sleep(0.02)

                time.sleep(config.DEFAULT_TYPING_DELAY)
                return True
            except Exception:
                pass

        try:
            import pyautogui
            lower_keys = [k.lower() for k in keys]
            pyautogui.hotkey(*lower_keys)
            return True
        except Exception as e:
            raise RuntimeError(f"Hotkey execution failed: {str(e)}")

    def close(self) -> None:
        """Closes the active D-Bus session if open."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._session_path = None
            self._portal_available = False


_GLOBAL_INPUT_BACKEND: Optional[DesktopInputBackend] = None


def get_input_backend() -> DesktopInputBackend:
    """Returns singleton instance of DesktopInputBackend."""
    global _GLOBAL_INPUT_BACKEND
    if _GLOBAL_INPUT_BACKEND is None:
        _GLOBAL_INPUT_BACKEND = DesktopInputBackend()
    return _GLOBAL_INPUT_BACKEND
