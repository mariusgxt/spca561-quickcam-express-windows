"""Userspace Windows driver for SPCA561A webcams (Logitech QuickCam Express)."""
__version__ = "0.1.0"

from .camera import SPCA561Camera, CameraNotFound  # noqa: F401
from .decode import decode_spca561, debayer  # noqa: F401
