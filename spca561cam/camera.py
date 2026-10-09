"""Userspace driver for SPCA561A (rev 012a) webcams such as the
Logitech QuickCam Express (046d:0928).

Register sequences follow the Linux gspca spca561 driver
(drivers/media/usb/gspca/spca561.c, GPL-2.0-or-later).
"""
import ctypes
import queue
import threading
import time

import libusb_package
import usb1

# Cameras handled by the "rev 012a" code path in the Linux driver.
SUPPORTED_IDS = [
    (0x046d, 0x0928),  # Logitech QuickCam Express
    (0x046d, 0x0929),  # Labtec WebCam Elch 2
    (0x046d, 0x092a), (0x046d, 0x092b), (0x046d, 0x092c),
    (0x046d, 0x092d), (0x046d, 0x092e), (0x046d, 0x092f),
    (0x041e, 0x403b),  # Creative
]

ISO_EP = 0x81
ISO_ALT = 7          # 1023-byte packets, full bandwidth
ISO_PACKET_SIZE = 1023
ISO_PACKETS_PER_TRANSFER = 32
ISO_TRANSFERS = 4

# (width, height) -> (mode register value, compressed?)
MODES = {
    (352, 288): (0, True),
    (320, 240): (1, True),
    (176, 144): (2, False),
    (160, 120): (3, False),
}

EXPOSURE_MAX = 2047 + 325
RAW_HEADER_LEN = 20

_DATA1 = [
    (0x29, 0x8118), (0x08, 0x8114), (0x0e, 0x8112), (0x00, 0x8102),
    (0x92, 0x8804), (0x04, 0x8802),
]
_SENSOR_MAP1 = [
    (0x8320, 0x3304), (0x8303, 0x0125), (0x8304, 0x0169), (0x8328, 0x000b),
    (0x833c, 0x0001), (0x832f, 0x1904), (0x8307, 0x00aa), (0x8301, 0x0003),
    (0x8302, 0x000e),
]
_DATA2 = [
    (0x21, 0x8118), (0x10, 0x8500), (0x07, 0x8601), (0x07, 0x8602),
    (0x04, 0x8501), (0x07, 0x8201), (0x08, 0x8200), (0x01, 0x8200),
    (0x90, 0x8604), (0x00, 0x8605), (0xb0, 0x8603),
    (0x07, 0x8601), (0x07, 0x8602), (0x00, 0x8610), (0x00, 0x8611),
    (0x00, 0x8612), (0x00, 0x8613), (0x43, 0x8614), (0x40, 0x8615),
    (0x71, 0x8616), (0x40, 0x8617),
    (0x0c, 0x8620), (0xc8, 0x8631), (0xc8, 0x8634), (0x23, 0x8635),
    (0x1f, 0x8636), (0xdd, 0x8637), (0xe1, 0x8638), (0x1d, 0x8639),
    (0x21, 0x863a), (0xe3, 0x863b), (0xdf, 0x863c), (0xf0, 0x8505),
    (0x32, 0x850a), (0x29, 0x8118),
]
_SENSOR_MAP2 = [(0x8339, 0x0000), (0x8307, 0x00aa)]

_libusb_loaded = False


def _load_libusb():
    global _libusb_loaded
    if not _libusb_loaded:
        usb1.loadLibrary(ctypes.CDLL(str(libusb_package.get_library_path())))
        _libusb_loaded = True


class CameraNotFound(RuntimeError):
    pass


class SPCA561Camera:
    """Streams raw frames from an SPCA561 rev 012a camera.

    read() returns (frame_bytes, compressed, snapshot_button_pressed).
    """

    def __init__(self, width=352, height=288):
        if (width, height) not in MODES:
            raise ValueError(f"unsupported resolution {width}x{height}, "
                             f"choose one of {sorted(MODES)}")
        self.width, self.height = width, height
        self.mode, self.compressed = MODES[(width, height)]
        _load_libusb()
        self._ctx = usb1.USBContext()
        self._ctx.open()
        self._dev = None
        for vid, pid in SUPPORTED_IDS:
            self._dev = self._ctx.openByVendorIDAndProductID(vid, pid)
            if self._dev is not None:
                break
        if self._dev is None:
            self._ctx.close()
            raise CameraNotFound(
                "No SPCA561 camera found. Is it plugged in and bound to WinUSB (Zadig)?")
        self._dev.claimInterface(0)
        self._frames = queue.Queue(maxsize=4)
        self._cur = None
        self._cur_compressed = False
        self._cur_snap = False
        self._running = False
        self._thread = None
        self._transfers = []
        self.dropped = 0

    # -- register access --------------------------------------------------
    def _reg_w_val(self, index, value):
        self._dev.controlWrite(0x40, 0, value & 0xff, index, b"", timeout=500)

    def _reg_w_buf(self, index, data):
        self._dev.controlWrite(0x40, 0, 0, index, bytes(data), timeout=500)

    def _reg_r(self, index, length=1):
        return self._dev.controlRead(0xC0, 0, 0, index, length, timeout=500)

    def _write_vector(self, data):
        for value, index in data:
            self._reg_w_val(index, value)

    def _sensor_mapwrite(self, data):
        for index, value in data:
            self._reg_w_buf(index, (value & 0xff, value >> 8))

    # -- controls (rev 012a) ---------------------------------------------
    def set_brightness(self, val):
        """-128..127, default 0."""
        for i in range(4):
            self._reg_w_val(0x8610 + i, val & 0xff)

    def set_white(self, white):
        """Hue / white balance, 1..127, default 64."""
        red = 0x20 + white * 3 // 8
        blue = 0x90 - white * 5 // 8
        self._reg_w_val(0x8614, red)
        self._reg_w_val(0x8616, blue)

    def set_exposure(self, val):
        """1..EXPOSURE_MAX, default 700. Higher values lower the frame rate."""
        val = max(1, min(EXPOSURE_MAX, int(val)))
        table = (0, 450, 550, 625, EXPOSURE_MAX)
        expo = 0
        for i in range(len(table) - 1):
            if val <= table[i + 1]:
                expo = val - table[i]
                if i:
                    expo += 300
                expo |= i << 11
                break
        self._reg_w_buf(0x8309, (expo & 0xff, expo >> 8))

    def set_gain(self, val):
        """0..255, default 63."""
        val = max(0, min(255, int(val)))
        if val < 64:
            g = val
        elif val < 128:
            g = (val // 2) | 0x40
        else:
            g = (val // 4) | 0xc0
        self._reg_w_buf(0x8335, (g, 0))

    # -- streaming --------------------------------------------------------
    def start(self):
        self._write_vector(_DATA1)
        self._sensor_mapwrite(_SENSOR_MAP1)
        self._write_vector(_DATA2)
        self._sensor_mapwrite(_SENSOR_MAP2)

        self._dev.setInterfaceAltSetting(0, ISO_ALT)

        if self.mode <= 1:
            self._reg_w_val(0x8500, 0x10 | self.mode)
        else:
            self._reg_w_val(0x8500, self.mode)
        self._reg_w_buf(0x8307, (0xaa, 0x00))
        self._reg_w_val(0x8700, 0x8a)
        self._reg_w_val(0x8112, 0x1e | 0x20)
        self._reg_w_val(0x850b, 0x03)
        reg8391 = (0x92, 0x30, 0x20, 0x00, 0x0c, 0x00, 0x00, 0x00)
        self._reg_w_buf(0x8391, reg8391)
        self._reg_w_buf(0x8390, reg8391)
        self._reg_w_val(0x8114, 0x00)  # LED on

        self.set_white(0x40)
        self.set_brightness(0)
        self.set_exposure(700)
        self.set_gain(63)

        self._running = True
        for _ in range(ISO_TRANSFERS):
            t = self._dev.getTransfer(iso_packets=ISO_PACKETS_PER_TRANSFER)
            t.setIsochronous(ISO_EP, ISO_PACKET_SIZE * ISO_PACKETS_PER_TRANSFER,
                             callback=self._on_transfer, timeout=1000)
            t.submit()
            self._transfers.append(t)
        self._thread = threading.Thread(target=self._event_loop, daemon=True)
        self._thread.start()

    def _event_loop(self):
        while self._running or any(t.isSubmitted() for t in self._transfers):
            try:
                self._ctx.handleEventsTimeout(0.1)
            except usb1.USBErrorInterrupted:
                pass

    def _on_transfer(self, transfer):
        status = transfer.getStatus()
        if status == usb1.TRANSFER_COMPLETED:
            for pstatus, data in transfer.iterISO():
                if pstatus == usb1.TRANSFER_COMPLETED and len(data):
                    self._scan_packet(data)
        elif status in (usb1.TRANSFER_NO_DEVICE, usb1.TRANSFER_CANCELLED):
            return
        if self._running:
            try:
                transfer.submit()
            except usb1.USBError:
                self._running = False

    def _scan_packet(self, data):
        seq = data[0]
        if seq == 0:  # start of frame
            self._finish_frame()
            if len(data) < 3:
                return
            self._cur_snap = bool(data[1] & 0x20)
            self._cur_compressed = bool(data[2] & 0x10)
            payload = data[1:]
            if not self._cur_compressed:
                payload = payload[RAW_HEADER_LEN:]
            self._cur = bytearray(payload)
        elif seq == 0xff:  # empty packet
            return
        elif self._cur is not None:
            self._cur += data[1:]

    def _finish_frame(self):
        if self._cur is None:
            return
        frame = (bytes(self._cur), self._cur_compressed, self._cur_snap)
        self._cur = None
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            self.dropped += 1
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(frame)

    def read(self, timeout=2.0):
        """Return the next (frame_bytes, compressed, snapshot) tuple."""
        return self._frames.get(timeout=timeout)

    def stop(self):
        if not self._running and not self._transfers:
            return
        self._running = False
        for t in self._transfers:
            try:
                if t.isSubmitted():
                    t.cancel()
            except usb1.USBError:
                pass
        if self._thread:
            self._thread.join(timeout=2)
        self._transfers = []
        try:
            self._reg_w_val(0x8112, 0x0e)
            self._reg_w_val(0x8114, 0x08)  # LED off
            self._dev.setInterfaceAltSetting(0, 0)
        except usb1.USBError:
            pass

    def close(self):
        self.stop()
        try:
            self._dev.releaseInterface(0)
            self._dev.close()
        finally:
            self._ctx.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
