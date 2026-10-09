"""Turn an SPCA561 camera into a normal Windows webcam via a virtual camera."""
import argparse
import queue
import sys
import time

import cv2
import numpy as np

from .camera import EXPOSURE_MAX, MODES, CameraNotFound, SPCA561Camera
from .decode import debayer, decode_spca561


class AutoExposure:
    """Simple software AE loop: exposure first (keeps the frame rate), then
    gain, then long exposures that lower the frame rate."""

    FAST_EXPOSURE_LIMIT = 600

    def __init__(self, cam, target=115, tolerance=12):
        self.cam = cam
        self.target = target
        self.tol = tolerance
        self.exposure = 700
        self.gain = 63
        self.countdown = 0

    def update(self, bayer):
        self.countdown -= 1
        if self.countdown > 0:
            return
        self.countdown = 3  # let the sensor settle for a few frames
        mean = float(bayer.mean())
        if abs(mean - self.target) <= self.tol:
            return
        step = 1.0 + min(0.5, abs(mean - self.target) / 255)
        if mean < self.target:
            if self.exposure < self.FAST_EXPOSURE_LIMIT:
                self.exposure = min(self.FAST_EXPOSURE_LIMIT, self.exposure * step + 1)
            elif self.gain < 255:
                self.gain = min(255, self.gain + 12)
            else:
                self.exposure = min(EXPOSURE_MAX, self.exposure * step)
        else:
            if self.exposure > self.FAST_EXPOSURE_LIMIT:
                self.exposure = max(self.FAST_EXPOSURE_LIMIT, self.exposure / step)
            elif self.gain > 32:
                self.gain = max(32, self.gain - 12)
            else:
                self.exposure = max(1, self.exposure / step)
        self.cam.set_exposure(self.exposure)
        self.cam.set_gain(self.gain)


class GrayWorldWB:
    """Smoothed gray-world white balance in software."""

    def __init__(self, smoothing=0.9):
        self.gains = np.ones(3, dtype=np.float32)
        self.smoothing = smoothing

    def apply(self, bgr):
        means = bgr.reshape(-1, 3).mean(axis=0) + 1e-3
        target = means.mean() / means
        self.gains = self.smoothing * self.gains + (1 - self.smoothing) * target
        lut = [np.clip(np.arange(256) * g, 0, 255).astype(np.uint8) for g in self.gains]
        return cv2.merge([cv2.LUT(c, l) for c, l in zip(cv2.split(bgr), lut)])


def frames(cam, auto_exposure=True, white_balance=True):
    """Yield decoded BGR frames from the camera."""
    ae = AutoExposure(cam) if auto_exposure else None
    wb = GrayWorldWB() if white_balance else None
    while True:
        try:
            data, compressed, _snap = cam.read()
        except queue.Empty:
            raise RuntimeError("camera stopped sending frames") from None
        if compressed:
            bayer = decode_spca561(data, cam.width, cam.height)
            if bayer is None:
                continue  # corrupt frame (dropped USB packet)
        else:
            need = cam.width * cam.height
            if len(data) < need:
                continue
            bayer = np.frombuffer(data[:need], np.uint8).reshape(cam.height, cam.width)
        if ae:
            ae.update(bayer)
        bgr = debayer(bayer)
        if wb:
            bgr = wb.apply(bgr)
        yield bgr


def parse_size(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="spca561cam",
        description="Use an SPCA561 webcam (e.g. Logitech QuickCam Express) on Windows "
                    "through the OBS virtual camera.")
    p.add_argument("--resolution", default="352x288",
                   choices=[f"{w}x{h}" for w, h in sorted(MODES, reverse=True)],
                   help="sensor resolution (default: 352x288)")
    p.add_argument("--output-size", type=parse_size, default=(640, 480), metavar="WxH",
                   help="size of the virtual camera image (default: 640x480; many apps "
                        "dislike tiny webcams)")
    p.add_argument("--fps", type=int, default=15, help="virtual camera frame rate (default: 15)")
    p.add_argument("--mirror", action="store_true", help="mirror the image horizontally")
    p.add_argument("--no-auto-exposure", action="store_true")
    p.add_argument("--no-white-balance", action="store_true")
    p.add_argument("--preview", action="store_true",
                   help="only show a preview window, no virtual camera")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    w, h = parse_size(args.resolution)
    try:
        cam = SPCA561Camera(w, h)
    except CameraNotFound as e:
        print(e, file=sys.stderr)
        return 1

    with cam:
        stream = frames(cam, not args.no_auto_exposure, not args.no_white_balance)
        out_w, out_h = args.output_size

        def prepare(bgr):
            if args.mirror:
                bgr = cv2.flip(bgr, 1)
            if (out_w, out_h) != (bgr.shape[1], bgr.shape[0]):
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            return bgr

        if args.preview:
            print("Preview running - press q or Esc in the window to quit.")
            for bgr in stream:
                cv2.imshow("SPCA561 preview", prepare(bgr))
                if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                    break
            cv2.destroyAllWindows()
            return 0

        import pyvirtualcam
        try:
            vcam = pyvirtualcam.Camera(out_w, out_h, args.fps, backend="obs",
                                       fmt=pyvirtualcam.PixelFormat.BGR)
        except RuntimeError as e:
            print(f"Could not open the OBS virtual camera: {e}\n"
                  "Install OBS Studio (26 or newer) - it provides the 'OBS Virtual Camera' device.",
                  file=sys.stderr)
            return 1
        with vcam:
            print(f"Streaming {w}x{h} -> '{vcam.device}' at {out_w}x{out_h}. "
                  "Select 'OBS Virtual Camera' in your app. Ctrl+C to stop.")
            last = time.time()
            n = 0
            try:
                for bgr in stream:
                    vcam.send(prepare(bgr))
                    n += 1
                    if time.time() - last >= 10:
                        print(f"  {n / (time.time() - last):.1f} fps from camera")
                        n, last = 0, time.time()
            except KeyboardInterrupt:
                pass
    return 0
