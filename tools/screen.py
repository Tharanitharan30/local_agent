import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from PIL import Image

import config
from tools.base import BaseTool


def capture_display_portal(timeout: int = config.SCREEN_TIMEOUT) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
    """
    Captures the primary Wayland screen using the FreeDesktop XDG Desktop Portal over D-Bus via jeepney.
    This works reliably across GNOME Shell, KDE Plasma, and Wayland desktop environments without root.
    Loads the screenshot directly into a PIL Image and immediately unlinks the temporary file.
    """
    try:
        from jeepney import new_method_call, DBusAddress
        from jeepney.io.blocking import open_dbus_connection
    except ImportError:
        return False, None, "jeepney library is not installed for D-Bus portal screen capture."

    temp_path: Optional[Path] = None
    try:
        conn = open_dbus_connection(bus="SESSION")
        if hasattr(conn, "sock") and conn.sock:
            conn.sock.settimeout(1.0)

        # Listen for Response signal from portal Request interface
        add_match_msg = new_method_call(
            DBusAddress("/org/freedesktop/DBus", "org.freedesktop.DBus", "org.freedesktop.DBus"),
            "AddMatch",
            "s",
            ("type='signal',interface='org.freedesktop.portal.Request',member='Response'",),
        )
        conn.send_and_get_reply(add_match_msg)

        # Call Screenshot method non-interactively
        shot_msg = new_method_call(
            DBusAddress("/org/freedesktop/portal/desktop", "org.freedesktop.portal.Desktop", "org.freedesktop.portal.Screenshot"),
            "Screenshot",
            "sa{sv}",
            ("", {"interactive": ("b", False)}),
        )
        reply = conn.send_and_get_reply(shot_msg)
        if not reply.body:
            conn.close()
            return False, None, "No response handle returned by XDG Desktop Portal."

        req_path = reply.body[0]
        start_time = time.time()

        # Wait for the Response signal
        img: Optional[Image.Image] = None
        err_msg: Optional[str] = None

        while time.time() - start_time < timeout:
            try:
                msg = conn.receive()
                if (
                    msg.header.message_type.name == "signal"
                    and msg.header.fields.get(1) == req_path
                ):
                    res_code, res_dict = msg.body
                    if res_code == 0 and "uri" in res_dict:
                        uri = res_dict["uri"][1]
                        file_path_str = urllib.parse.unquote(urllib.parse.urlparse(uri).path)
                        temp_path = Path(file_path_str)
                        if temp_path.exists():
                            with Image.open(temp_path) as opened_img:
                                img = opened_img.copy()
                        else:
                            err_msg = f"Portal returned file path that does not exist: {file_path_str}"
                    else:
                        err_msg = f"Portal screenshot request denied or cancelled (code {res_code})."
                    break
            except Exception:
                continue

        conn.close()

        if img is not None:
            return True, img, None
        return False, None, err_msg or f"Portal screenshot timed out after {timeout} seconds."

    except Exception as e:
        return False, None, f"Portal capture error: {str(e)}"
    finally:
        # Privacy guarantee: Always clean up temporary files created by the portal immediately
        if temp_path is not None and temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass


def capture_display_grim() -> Tuple[bool, Optional[Image.Image], Optional[str]]:
    """
    Fallback screen capture using grim utility (for wlroots/sway Wayland environments).
    """
    grim_path = shutil.which("grim")
    if not grim_path:
        return False, None, "grim utility not found on PATH."

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_name = tmp_file.name

    try:
        proc = subprocess.run(
            [grim_path, tmp_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.SCREEN_TIMEOUT,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            return False, None, f"grim capture failed: {err}"

        with Image.open(tmp_name) as opened_img:
            img = opened_img.copy()
        return True, img, None
    except Exception as e:
        return False, None, f"grim execution error: {str(e)}"
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def capture_screen_image() -> Tuple[bool, Optional[Image.Image], Optional[str]]:
    """
    Tries Wayland XDG Desktop Portal first, falls back to grim.
    Returns (success, PIL_Image, error_message).
    Ensures zero disk residue after capture.
    """
    # 1. Primary Wayland Portal capture
    success, img, err = capture_display_portal()
    if success and img is not None:
        return True, img, None

    # 2. Fallback: grim
    grim_success, grim_img, grim_err = capture_display_grim()
    if grim_success and grim_img is not None:
        return True, grim_img, None

    return False, None, f"Screen capture failed: {err or grim_err}"


def preprocess_screenshot(
    image: Image.Image,
    max_width: int = config.SCREEN_MAX_WIDTH,
    max_height: int = config.SCREEN_MAX_HEIGHT,
) -> Image.Image:
    """
    Preprocesses screenshot to prevent oversized tensors and excessive VRAM consumption:
    - Preserves aspect ratio.
    - Downscales using high-quality Lanczos resampling if image exceeds max dimensions.
    - Converts to standard RGB format.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class ScreenTool(BaseTool):
    """
    Screen observation tool: Captures the primary display, downsamples appropriately
    to preserve GPU memory, feeds the image into the local vision model,
    and returns a structured visual description.
    Read-only: Does NOT implement mouse or keyboard interactions.
    """

    name = "screen.capture"
    description = (
        "Captures the user's current screen and uses a lightweight local vision model "
        "to describe visible open windows, active applications, UI state, terminal outputs, "
        "or error messages. Read-only."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "display": {
                "type": "string",
                "description": "The display to capture ('primary').",
                "default": "primary",
            },
            "query": {
                "type": "string",
                "description": (
                    "Optional specific question or visual focus (e.g. 'What error is shown?', "
                    "'What application is open in the foreground?')."
                ),
                "default": "",
            },
        },
    }

    def __init__(self, vision_model: Optional[Any] = None):
        super().__init__()
        self.aliases = ["screen", "screen_capture", "capture_screen"]
        self.vision_model = vision_model

    def _get_vision_model(self) -> Any:
        if self.vision_model is not None:
            return self.vision_model
        # Lazy import and instantiation to avoid loading vision model into memory if unused
        from model.vision import get_vision_model
        self.vision_model = get_vision_model()
        return self.vision_model

    def execute(self, display: str = "primary", query: str = "", **kwargs: Any) -> Dict[str, Any]:
        """
        Executes screen capture and visual analysis.
        Returns structured dictionary with dimensions and visual description.
        """
        # Step 1: Capture screenshot in memory
        success, img, err = capture_screen_image()
        if not success or img is None:
            return {
                "success": False,
                "display": display,
                "width": 0,
                "height": 0,
                "error": f"Failed to capture screen: {err}",
            }

        orig_width, orig_height = img.size

        # Step 2: Preprocess screenshot (resize with aspect ratio preserved)
        preprocessed = preprocess_screenshot(img)

        # Step 3: Run local vision inference
        try:
            v_model = self._get_vision_model()
            visual_description = v_model.describe_screen(preprocessed, query=query)
        except Exception as e:
            return {
                "success": False,
                "display": display,
                "width": orig_width,
                "height": orig_height,
                "error": f"Vision model inference error: {str(e)}",
            }
        finally:
            # Explicit cleanup of PIL images
            del img
            del preprocessed

        return {
            "success": True,
            "display": display,
            "width": orig_width,
            "height": orig_height,
            "description": visual_description,
        }
