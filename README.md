# ESP32-S3 Video Streaming & Gesture Recognition

> **COE 454 — Homework 1** · KFUPM

Real-time video streaming from an **ESP32-S3 + OV2640** camera to a PC over Wi-Fi (TCP), with server-side frame interpolation via RAFT optical flow and a custom CNN for hand-gesture (digit 1–5) recognition.

---

## Architecture

```
┌──────────────────┐        TCP :4444        ┌──────────────────────────┐
│   ESP32-S3       │ ─────────────────────── │   PC  (cloud.py)         │
│   OV2640 Camera  │   FRM | len | JPEG      │   Tkinter GUI            │
│   MicroPython    │ ◄── START / STOP / CLOSE│   OpenCV + RAFT interp.  │
└──────────────────┘                         └──────────────────────────┘
                                                       │
                                               ┌───────┴────────┐
                                               │  ml.py (train) │
                                               │  TinyGestureNet │
                                               └────────────────┘
```

## Project Structure

```
.
├── esp32/               # MicroPython firmware for ESP32-S3
│   ├── main.py          # Wi-Fi, TCP client, camera capture loop
│   └── lib/
│       ├── Arducam.py   # Ported Arducam driver (SPI/I2C)
│       └── OV2640_reg.py
├── pc/
│   └── cloud.py         # TCP server, Tkinter GUI, RAFT interpolation
├── ml/
│   ├── ml.py            # Dataset merging, training (TinyGestureNet)
│   └── label_map.json   # Class ↔ index mapping {1..5}
├── REPORT.md            # Detailed protocol specification
└── .gitignore
```

## Protocol

| Direction | Format | Description |
|---|---|---|
| **ESP32 → PC** | `FRM` (3 B) + `length` (4 B, big-endian) + JPEG payload | Framed binary video stream |
| **PC → ESP32** | ASCII command + `\n` | `START`, `STOP`, `CLOSE` |

The server auto-detects a **fallback raw-JPEG mode** (SOI/EOI scanning) if the ESP32 sends bare JPEGs instead of framed packets.

### Timing

- **Streaming**: frames sent back-to-back; GC every 20 frames
- **Stopped**: last frame re-sent every 400 ms (freeze-frame keepalive)
- **Error recovery**: automatic reconnect on any socket failure

## Hardware

| Component | Details |
|---|---|
| MCU | ESP32-S3-DevKitC-1 (MicroPython) |
| Camera | OV2640 — SPI + I2C, QVGA (320 × 240) |
| Network | Wi-Fi (mobile hotspot), TCP port 4444 |

## Server-Side Processing

**Frame interpolation** smooths playback when the native frame rate is low:

| Method | Engine | Notes |
|---|---|---|
| `ml_raft` (default) | RAFT-Small (PyTorch) | Bidirectional optical flow, 6 update iterations |
| `cv_flow` (fallback) | OpenCV Farneback | Used when PyTorch/RAFT unavailable |
| `none` | — | Pass-through, no interpolation |

Frames are upscaled **1.3×** (cubic interpolation) before display. Interpolation is skipped for gaps > 200 ms or dropped frames.

## ML — Hand Gesture Recognition

A lightweight **TinyGestureNet** CNN trained to classify digits **1–5** from hand-gesture images.

### Model Architecture

```
Conv2d(3→16, 3×3) → BN → ReLU → MaxPool
Conv2d(16→32, 3×3) → BN → ReLU → MaxPool
Conv2d(32→64, 3×3) → BN → ReLU → MaxPool
Conv2d(64→96, 3×3) → BN → ReLU → AdaptiveAvgPool(1×1)
Flatten → Dropout(0.5) → Linear(96 → 5)
```

- **Input**: 64 × 64 RGB
- **Optimizer**: AdamW (lr=1e-3, weight decay=1e-4)
- **Scheduler**: Cosine annealing (60 epochs)
- **Augmentation**: random affine, color jitter
- **Loss**: Cross-entropy with inverse-frequency class weights

### Training

```bash
cd ml
python ml.py                          # uses default dataset paths
python ml.py --epochs 100 --lr 5e-4   # custom hyperparameters
```

Datasets are merged from multiple sources, deduplicated, and stratified-split (80/20). The best checkpoint is saved to `ml/best_model.pth`.

## Getting Started

### ESP32 (Client)

1. **Flash MicroPython** onto the ESP32-S3:
   ```bash
   esptool.py --chip esp32s3 --port /dev/cu.usbmodem101 erase_flash
   esptool.py --chip esp32s3 --port /dev/cu.usbmodem101 write_flash -z 0x0 ESP32_GENERIC_S3.bin
   ```
   > Download firmware from [micropython.org/download/ESP32_GENERIC_S3](https://micropython.org/download/ESP32_GENERIC_S3). Hold **BOOT** while pressing **RST** to enter download mode.

2. **Upload** `esp32/main.py` and `esp32/lib/` to the board:
   ```bash
   mpremote connect /dev/cu.usbmodem101 cp -r esp32/ :
   ```

3. **Configure** Wi-Fi credentials and server IP in `esp32/main.py`:
   ```python
   SSID = "YOUR_SSID"
   PASSWORD = "YOUR_PASSWORD"
   SERVER_IP = "YOUR_SERVER_IP"
   ```

4. **Reset** the board — it connects automatically.

### PC (Server)

1. **Install dependencies**:
   ```bash
   pip install opencv-python numpy pillow
   pip install torch torchvision   # optional, for RAFT interpolation
   ```

2. **Run the server**:
   ```bash
   python pc/cloud.py
   ```

3. The GUI listens on port **4444**. Once the ESP32 connects, use **Start** / **Stop** to control the stream.

## License

MIT
