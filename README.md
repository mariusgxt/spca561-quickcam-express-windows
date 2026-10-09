# spca561-quickcam-express-windows

Use a **Logitech QuickCam Express** (USB `046d:0928`, around 2000) or another
webcam built on the **Sunplus SPCA561A** chip as a normal webcam on **Windows 10/11 (64-bit)**.

Logitech never shipped a 64-bit driver for these cameras. This project is a small
userspace driver in Python. It talks to the camera via WinUSB/libusb, decodes the
camera's compressed Bayer stream and feeds the picture into the **OBS Virtual Camera**,
so Zoom, Teams, Discord, browsers and so on can use it.

## Supported cameras

The cameras the Linux `gspca_spca561` driver treats as "rev 012a":

| USB ID      | Camera                               | Tested |
|-------------|--------------------------------------|--------|
| `046d:0928` | Logitech QuickCam Express            | yes    |
| `046d:0929` | Labtec WebCam Elch 2                 | no     |
| `046d:092a`-`092f` | other Logitech/Labtec SPCA561 models | no |
| `041e:403b` | Creative                             | no     |

The "rev 072a" cameras (Creative, Micro Innovation and others) need a different init
sequence and are not supported yet.

## Setup

1. **Install [Python](https://www.python.org/downloads/) 3.9 or newer** and
   **[OBS Studio](https://obsproject.com/)** (version 26 or newer). OBS provides the
   "OBS Virtual Camera" device. You do not need to keep OBS open.
2. **Bind the camera to WinUSB** with [Zadig](https://zadig.akeo.ie/):
   - Plug in the camera and start Zadig.
   - Select the camera from the list. It is usually called "Camera", USB ID `046D 0928`.
     If it is missing, enable *Options → List All Devices*.
   - Choose **WinUSB** as the target driver and click **Install Driver**.
3. **Install this package:**
   ```bash
   pip install -r requirements.txt
   ```
   or `pip install .` if you want the `spca561cam` command.

## Usage

Double-click `start-webcam.bat`, or run:

```bash
python -m spca561cam
```

Then pick **"OBS Virtual Camera"** as your webcam in the app you use. Keep the window
open while you use the camera. Press Ctrl+C or close the window to stop.

Options:

| Option | Meaning |
|--------|---------|
| `--preview` | only show a preview window, no virtual camera |
| `--resolution 352x288` | sensor mode: `352x288`, `320x240` (compressed), `176x144`, `160x120` (raw) |
| `--output-size 640x480` | size of the virtual camera image (default `640x480`) |
| `--mirror` | mirror horizontally |
| `--no-auto-exposure` | disable the software auto exposure |
| `--no-white-balance` | disable the software gray-world white balance |

Expect about 9-10 frames per second at 352×288. That is what the camera manages over
USB 1.1.

### Using it from Python

```python
from spca561cam import SPCA561Camera
from spca561cam.app import frames

with SPCA561Camera(352, 288) as cam:
    for bgr in frames(cam):   # numpy arrays, OpenCV BGR order
        ...
```

## How it works

- `spca561cam/camera.py` sends the register init sequence for the SPCA561 rev 012a.
  This sequence comes from the Linux gspca driver. The module then reads the
  isochronous endpoint `0x81` (alternate setting 7, 1023-byte packets) through libusb
  and puts the packets back together into frames.
- `spca561cam/decode.py` ports the SPCA561 decompressor from libv4lconvert. It is
  compiled with numba and needs about 2-3 ms per frame. The module also demosaics the
  GBRG Bayer image.
- `spca561cam/app.py` adds software auto exposure and white balance, then sends the
  frames to the OBS Virtual Camera through `pyvirtualcam`.

## Going back to the old state

In Device Manager, right-click the camera, choose *Uninstall device* and tick
"Delete the driver software". Then unplug the camera and plug it back in.

## Credits & license

- Register sequences: Linux `drivers/media/usb/gspca/spca561.c` by Michel Xhaard,
  Jean-François Moine and Hans de Goede (GPL-2.0-or-later).
- Decompressor: `libv4lconvert/spca561-decompress.c` by Andrzej Szombierski (LGPL-2.1-or-later).

This project is licensed under the **GNU General Public License v2.0 or later**, see [LICENSE](LICENSE).
It is not affiliated with Logitech.
