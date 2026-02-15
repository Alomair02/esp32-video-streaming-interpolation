# COE 454 – HW1: ESP32 Video Streaming Protocol Report

## 1. Overview

This project implements a real-time video streaming system between an **ESP32-S3** microcontroller (equipped with an OV2640 camera over SPI/I2C) and a **PC-based cloud server**. Communication occurs over a TCP socket on a shared Wi-Fi network (mobile hotspot). The server provides a Tkinter GUI for playback and control, and optionally performs frame interpolation using RAFT optical flow to enhance stream quality.

## 2. Protocol Specification

### 2.1 Transport

| Property | Value |
|---|---|
| Transport | TCP (reliable, ordered) |
| Port | 4444 |
| Network | Wi-Fi (mobile hotspot) |
| Roles | ESP32 = client, PC = server |

### 2.2 Message Types

The protocol defines two message directions with distinct formats:

#### Server → ESP32: Commands

Plain ASCII strings terminated by a newline character (`\n`).

| Command | Meaning |
|---|---|
| `START\n` | Begin capturing and streaming frames |
| `STOP\n` | Pause streaming (ESP32 resends last frame periodically) |
| `CLOSE\n` | Gracefully close the TCP connection |

#### ESP32 → Server: Video Frames (Framed Protocol)

Each frame is sent as a binary packet:

```
+-------+------------+-----------------+
| Magic | Length     | JPEG Payload    |
| 3 B   | 4 B (BE)  | Variable        |
+-------+------------+-----------------+
  "FRM"   uint32       raw JPEG bytes
```

- **Magic** (3 bytes): The ASCII literal `FRM` (`0x46 0x52 0x4D`), used for synchronization.
- **Length** (4 bytes): Big-endian unsigned 32-bit integer indicating the payload size in bytes (max 256 KB).
- **Payload**: A complete JPEG image (SOI `0xFFD8` … EOI `0xFFD9`).

The server also supports a **fallback raw-JPEG mode** (auto-detected): if the first bytes received are a JPEG SOI marker instead of `FRM`, the server switches to extracting frames by scanning for SOI/EOI boundaries.

### 2.3 Timing & Sequencing

```
    ESP32                              Server (PC)
      |                                    |
      |--- TCP connect (port 4444) ------->|  (1) Connection setup
      |                                    |
      |            [idle — awaiting cmd]   |
      |                                    |
      |<------------ "START\n" -----------|  (2) User presses Start
      |                                    |
      |--- FRM | len | JPEG #1 ---------->|  \
      |--- FRM | len | JPEG #2 ---------->|   } (3) Continuous streaming
      |--- ...                             |  /    (capture → transmit loop)
      |                                    |
      |<------------ "STOP\n" ------------|  (4) User presses Stop
      |                                    |
      |--- FRM | len | last frame ------->|  (5) Freeze-frame resend
      |        (every ~400 ms)             |      keeps display alive
      |                                    |
      |<------------ "CLOSE\n" -----------|  (6) Teardown (optional)
      |--- [TCP FIN] -------------------->|
```

- **Idle state**: ESP32 sleeps ~20 ms per loop iteration to save resources.
- **Streaming state**: Frames are captured and sent back-to-back; garbage collection runs every 20 frames.
- **Freeze-frame**: When stopped, the last captured frame is retransmitted every 400 ms so the display remains populated.
- **Reconnect**: On any socket error, the ESP32 automatically reconnects to the server and resumes its previous state (streaming or idle).

## 3. Cloud-Side Processing (Extra)

The server optionally performs **frame interpolation** to smooth playback when the native ESP32 frame rate is low:

- **RAFT-Small** (PyTorch): Computes bidirectional optical flow between consecutive frames and synthesizes intermediate frames via warping and blending.
- **Farneback fallback**: If PyTorch/RAFT is unavailable, OpenCV's Farneback dense optical flow is used.
- **Upscaling**: Frames are enlarged by a configurable factor (default 1.3×) using cubic interpolation.

Interpolation is skipped when frames are too far apart in time (>200 ms) or sequence (dropped frames), avoiding visual artifacts.

## 4. Hardware & Software

| Component | Details |
|---|---|
| MCU | ESP32-S3 (MicroPython) |
| Camera | OV2640 (SPI + I2C), QVGA resolution |
| Server | Python 3, Tkinter, OpenCV, PyTorch (optional) |
| Connection | Mobile hotspot Wi-Fi, TCP port 4444 |

## 5. Challenges & Solutions

| Challenge | Solution |
|---|---|
| **No compatible camera library** — No existing MicroPython library supported the OV2640 via SPI/I2C on ESP32-S3. Arducam libraries targeted Raspberry Pi Pico (CircuitPython). | Ported and modified the Arducam CircuitPython library to work with MicroPython on ESP32-S3, adapting SPI initialization and pin mappings. Multiple SPI configurations are tried automatically at startup. |
| **Interpolation latency** — RAFT optical flow adds noticeable processing delay on the server side. | Tuned the balance between source resolution (QVGA), upscale factor, RAFT iteration count, and number of interpolation steps. A single synthetic frame per pair with 6 RAFT updates provides a good quality-vs-latency trade-off. |

## 6. How to Run

**ESP32 (Client):**
1. Flash MicroPython firmware onto the ESP32-S3-DevKitC-1:
   ```bash
   # Erase flash first
   esptool.py --chip esp32s3 --port /dev/cu.usbmodem101 erase_flash
   # Flash the standard MicroPython firmware (.bin from micropython.org)
   esptool.py --chip esp32s3 --port /dev/cu.usbmodem101 write_flash -z 0x0 ESP32_GENERIC_S3-20xxxxxx-vX.X.X.bin
   ```
   > Download the firmware from [micropython.org/download/ESP32_GENERIC_S3](https://micropython.org/download/ESP32_GENERIC_S3). Use the generic ESP32-S3 build. Hold the **BOOT** button while pressing **RST** to enter download mode if the port is not detected.
2. Upload `esp32/main.py` and `esp32/lib/` to the board (e.g., via `mpremote`).
3. Edit `SSID`, `PASSWORD`, and `SERVER_IP` in `main.py` to match your network.
4. Reset the board — it connects to Wi-Fi and the server automatically.

**PC (Server):**
1. Install dependencies: `pip install opencv-python numpy pillow` (optionally `torch torchvision` for RAFT interpolation).
2. Run: `python pc/cloud.py`
3. The GUI opens and listens on port 4444. Once the ESP32 connects, use the **Start**/**Stop** buttons to control the stream.