# i did some frame interpolation here just to improve the
# quality of the video stream when upscaling
# you can think of it as "cloud processing"
import socket
import threading
import time
import tkinter as tk
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import torch
except Exception:
    torch = None

try:
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
except Exception:
    Raft_Small_Weights = None
    raft_small = None



PORT = 4444


# The following constants are only for interpolation
RECV_CHUNK = 4096
FRAME_MAGIC = b"FRM"
FRAME_HEADER_LEN = 7
FRAME_MAX_BYTES = 1024 * 1024
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

INTERP_MODE = "ml_raft"   # "ml_raft", "cv_flow", "none"
INTERP_STEPS = 1          # 1 means 2x output fps (1 synthetic frame between real frames)
RAFT_UPDATES = 6          # fewer updates for lower latency
UPSCALE_FACTOR = 1.3      # >1.0 to enlarge in processing stage
MAX_INTERP_ID_GAP = 1     # skip interpolation if we dropped too many source frames
MAX_INTERP_DT_S = 0.20    # skip interpolation across larger time gaps

DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 960
DISPLAY_QUEUE_MAX = 4
UI_REFRESH_MS = 16

# Thsi class is for intermediate frame interpolation on the server side
# Just to show some "cloud processing" capability, and smooth out the video when resolution is increased.
# It is not optimized for performance at all + the esp cannot run it on edge.
class FrameInterpolator:
    def __init__(self, mode, steps, upscale_factor):
        self.mode = mode
        self.steps = max(0, int(steps))
        self.upscale_factor = float(upscale_factor)

        self.raft = None
        self.raft_transform = None
        self.device = "cpu"
        self.info = ""

        if self.mode == "ml_raft":
            self._init_raft()
        elif self.mode == "cv_flow":
            self.info = "CV interpolation (Farneback)"
        else:
            self.info = "Interpolation disabled"

    def _init_raft(self):
        if torch is None or raft_small is None or Raft_Small_Weights is None:
            self.mode = "cv_flow"
            self.info = "Torch/RAFT unavailable, using CV interpolation"
            return
        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            weights = Raft_Small_Weights.DEFAULT
            self.raft_transform = weights.transforms()
            self.raft = raft_small(weights=weights, progress=False).to(self.device).eval()
            self.info = f"ML interpolation (RAFT-small on {self.device})"
        except Exception as e:
            self.mode = "cv_flow"
            self.raft = None
            self.raft_transform = None
            self.info = f"RAFT init failed ({repr(e)}), using CV interpolation"

    @staticmethod
    def decode_jpeg(jpeg_bytes):
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _warp_rgb(rgb, flow):
        h, w = rgb.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x - flow[..., 0]).astype(np.float32)
        map_y = (grid_y - flow[..., 1]).astype(np.float32)
        return cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    def _interp_with_flows(self, prev_rgb, curr_rgb, flow12, flow21):
        mids = []
        for i in range(self.steps):
            t = (i + 1) / (self.steps + 1)
            warp_prev = self._warp_rgb(prev_rgb, t * flow12)
            warp_curr = self._warp_rgb(curr_rgb, (1.0 - t) * flow21)
            mid = cv2.addWeighted(warp_prev, 1.0 - t, warp_curr, t, 0.0)
            mids.append(mid)
        return mids

    def _interpolate_cv(self, prev_rgb, curr_rgb):
        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        flow12 = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow21 = cv2.calcOpticalFlowFarneback(
            curr_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return self._interp_with_flows(prev_rgb, curr_rgb, flow12, flow21)

    def _interpolate_raft(self, prev_rgb, curr_rgb):
        if self.raft is None or self.raft_transform is None:
            return self._interpolate_cv(prev_rgb, curr_rgb)

        h, w = prev_rgb.shape[:2]
        if h < 8 or w < 8:
            return []

        # RAFT requires H and W divisible by 8.
        h8 = h - (h % 8)
        w8 = w - (w % 8)
        if h8 != h or w8 != w:
            prev_rgb = prev_rgb[:h8, :w8]
            curr_rgb = curr_rgb[:h8, :w8]

        try:
            img1 = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0)
            img2 = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0)
            img1, img2 = self.raft_transform(img1, img2)
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            with torch.inference_mode():
                flow12 = self.raft(img1, img2, num_flow_updates=RAFT_UPDATES)[-1]
                flow21 = self.raft(img2, img1, num_flow_updates=RAFT_UPDATES)[-1]

            flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()
            flow21 = flow21[0].permute(1, 2, 0).detach().cpu().numpy()
            return self._interp_with_flows(prev_rgb, curr_rgb, flow12, flow21)
        except Exception:
            return self._interpolate_cv(prev_rgb, curr_rgb)

    def interpolate(self, prev_rgb, curr_rgb):
        if self.steps <= 0 or prev_rgb is None:
            return []
        if self.mode == "ml_raft":
            return self._interpolate_raft(prev_rgb, curr_rgb)
        if self.mode == "cv_flow":
            return self._interpolate_cv(prev_rgb, curr_rgb)
        return []

    def upscale(self, rgb):
        if self.upscale_factor <= 1.0:
            return rgb
        h, w = rgb.shape[:2]
        new_w = max(1, int(w * self.upscale_factor))
        new_h = max(1, int(h * self.upscale_factor))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


# This is the application running the server side logic, and receiving the video stream from the esp32.
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("COE454 HW1 - Cloud Server")

        self.running = True

        self.conn = None
        self.conn_lock = threading.Lock()
        self.peer = None

        self.raw_lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_jpeg_id = 0
        self.latest_jpeg_ts = 0.0
        self.reset_processing = False
        self.stream_protocol = None
        self.rx_frame_count = 0

        self.display_lock = threading.Lock()
        self.display_queue = deque(maxlen=DISPLAY_QUEUE_MAX)
        self.tk_img = None

        self.interpolator = FrameInterpolator(INTERP_MODE, INTERP_STEPS, UPSCALE_FACTOR)

        self.status = tk.StringVar(value=f"Waiting for ESP32... | {self.interpolator.info}")
        tk.Label(root, textvariable=self.status, padx=10, pady=8).pack()

        self.video = tk.Label(root, text="Waiting for frames...", bg="black", fg="white")
        self.video.pack(padx=10, pady=8)

        btns = tk.Frame(root)
        btns.pack(pady=8)
        self.start_btn = tk.Button(btns, text="Start", width=12, command=lambda: self.send_cmd("START"))
        self.stop_btn = tk.Button(btns, text="Stop", width=12, command=lambda: self.send_cmd("STOP"))
        self.start_btn.grid(row=0, column=0, padx=8)
        self.stop_btn.grid(row=0, column=1, padx=8)

        tk.Button(root, text="Exit", width=12, command=self.on_quit).pack(pady=8)

        self.server_thread = threading.Thread(target=self.server_loop, daemon=True)
        self.server_thread.start()

        self.proc_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.proc_thread.start()

        self.update_buttons()
        self.refresh_video()

    def update_buttons(self):
        connected = self.is_connected()
        state = tk.NORMAL if connected else tk.DISABLED
        self.start_btn.config(state=state)
        self.stop_btn.config(state=state)
        self.root.after(250, self.update_buttons)

    def is_connected(self):
        with self.conn_lock:
            return self.conn is not None

    def mark_processing_reset(self):
        with self.raw_lock:
            self.reset_processing = True

    # Server loop
    def server_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", PORT))
        srv.listen(1)

        while self.running:
            self.set_status(f"Listening on port {PORT}... waiting for ESP32 | {self.interpolator.info}")
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            with self.conn_lock:
                if self.conn:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                self.conn = conn
                self.peer = addr

            self.stream_protocol = None
            self.rx_frame_count = 0
            self.mark_processing_reset()
            self.set_status(f"Connected: {addr[0]}:{addr[1]} | {self.interpolator.info}")

            try:
                self.recv_frame_stream(conn)
            except Exception as e:
                self.set_status(f"Disconnected. Waiting for ESP32... ({repr(e)}) | {self.interpolator.info}")
            finally:
                self.mark_processing_reset()
                with self.conn_lock:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    if self.conn is conn:
                        self.conn = None
                        self.peer = None

    def recv_frame_stream(self, conn):
        rx = bytearray()
        while self.running:
            data = conn.recv(RECV_CHUNK)
            if not data:
                raise ConnectionError("ESP32 disconnected")
            rx.extend(data)

            if self.stream_protocol is None:
                self.stream_protocol = self.detect_stream_protocol(rx)
                if self.stream_protocol is not None:
                    self.set_status(
                        f"Connected: {self.peer[0]}:{self.peer[1]} | {self.interpolator.info} | protocol={self.stream_protocol}"
                    )

            if self.stream_protocol == "framed":
                frames = self.extract_frame_packets(rx)
            elif self.stream_protocol == "raw_jpeg":
                frames = self.extract_raw_jpegs(rx)
            else:
                frames = []

            for frame in frames:
                with self.raw_lock:
                    self.latest_jpeg = frame
                    self.latest_jpeg_id += 1
                    self.latest_jpeg_ts = time.perf_counter()
                self.rx_frame_count += 1
                if self.rx_frame_count % 30 == 0 and self.peer is not None:
                    self.set_status(
                        f"Connected: {self.peer[0]}:{self.peer[1]} | {self.interpolator.info} | "
                        f"protocol={self.stream_protocol} | frames={self.rx_frame_count}"
                    )

    @staticmethod
    def detect_stream_protocol(rx):
        if rx.startswith(FRAME_MAGIC):
            return "framed"
        if rx.startswith(JPEG_SOI):
            return "raw_jpeg"

        i_magic = rx.find(FRAME_MAGIC)
        i_jpeg = rx.find(JPEG_SOI)
        if i_magic >= 0 and (i_jpeg < 0 or i_magic <= i_jpeg):
            if i_magic > 0:
                del rx[:i_magic]
            if rx.startswith(FRAME_MAGIC):
                return "framed"
        if i_jpeg >= 0:
            if i_jpeg > 0:
                del rx[:i_jpeg]
            if rx.startswith(JPEG_SOI):
                return "raw_jpeg"

        # Keep the buffer bounded while protocol is unknown.
        if len(rx) > 4096:
            del rx[:-4]
        return None

    @staticmethod
    def extract_frame_packets(rx):
        frames = []
        while True:
            start = rx.find(FRAME_MAGIC)
            if start < 0:
                # Keep only enough bytes for a partial magic prefix.
                keep = len(FRAME_MAGIC) - 1
                if len(rx) > keep:
                    del rx[:-keep]
                break

            if start > 0:
                del rx[:start]

            if len(rx) < FRAME_HEADER_LEN:
                break

            frame_len = int.from_bytes(rx[len(FRAME_MAGIC):FRAME_HEADER_LEN], "big")
            if frame_len <= 0 or frame_len > FRAME_MAX_BYTES:
                # Bad header, drop one byte and resync.
                del rx[0]
                continue

            full_len = FRAME_HEADER_LEN + frame_len
            if len(rx) < full_len:
                break

            frames.append(bytes(rx[FRAME_HEADER_LEN:full_len]))
            del rx[:full_len]
        return frames

    @staticmethod
    def extract_raw_jpegs(rx):
        frames = []
        while True:
            start = rx.find(JPEG_SOI)
            if start < 0:
                if len(rx) > 1:
                    del rx[:-1]
                break
            if start > 0:
                del rx[:start]
            end = rx.find(JPEG_EOI, 2)
            if end < 0:
                break
            frames.append(bytes(rx[:end + 2]))
            del rx[:end + 2]
        return frames

    def process_loop(self):
        last_jpeg_id = 0
        last_jpeg_ts = 0.0
        prev_rgb = None
        while self.running:
            reset = False
            jpeg = None
            jpeg_id = 0
            jpeg_ts = 0.0

            with self.raw_lock:
                if self.reset_processing:
                    self.reset_processing = False
                    reset = True
                if self.latest_jpeg_id != last_jpeg_id:
                    jpeg = self.latest_jpeg
                    jpeg_id = self.latest_jpeg_id
                    jpeg_ts = self.latest_jpeg_ts

            if reset:
                prev_rgb = None
                last_jpeg_id = 0
                last_jpeg_ts = 0.0
                with self.display_lock:
                    self.display_queue.clear()

            if jpeg is None:
                time.sleep(0.002)
                continue

            curr_rgb = self.interpolator.decode_jpeg(jpeg)
            if curr_rgb is None:
                last_jpeg_id = jpeg_id
                last_jpeg_ts = jpeg_ts
                continue

            id_gap = 1 if last_jpeg_id == 0 else max(1, jpeg_id - last_jpeg_id)
            dt = 0.0 if last_jpeg_ts == 0.0 else max(0.0, jpeg_ts - last_jpeg_ts)
            allow_interp = (
                prev_rgb is not None
                and id_gap <= MAX_INTERP_ID_GAP
                and dt <= MAX_INTERP_DT_S
            )

            out_frames = []
            if allow_interp:
                out_frames.extend(self.interpolator.interpolate(prev_rgb, curr_rgb))
            out_frames.append(curr_rgb)
            prev_rgb = curr_rgb
            last_jpeg_id = jpeg_id
            last_jpeg_ts = jpeg_ts

            with self.display_lock:
                for fr in out_frames:
                    self.display_queue.append(self.interpolator.upscale(fr))

    def refresh_video(self):
        frame = None
        with self.display_lock:
            if self.display_queue:
                frame = self.display_queue.popleft()

        if frame is not None:
            img = Image.fromarray(frame)
            w, h = img.size
            scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
            if scale < 1.0:
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                if hasattr(Image, "Resampling"):
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    img = img.resize(new_size, Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.video.config(image=self.tk_img, text="")

        self.root.after(UI_REFRESH_MS, self.refresh_video)

    def send_cmd(self, cmd: str):
        payload = (cmd.strip().upper() + "\n").encode()
        with self.conn_lock:
            conn = self.conn
        if not conn:
            self.set_status("No ESP32 connected!")
            return
        try:
            conn.sendall(payload)
            self.set_status(f"Sent command: {cmd} | {self.interpolator.info}")
        except Exception as e:
            self.set_status(f"Failed to send: {repr(e)}")
            with self.conn_lock:
                try:
                    conn.close()
                except Exception:
                    pass
                if self.conn is conn:
                    self.conn = None

    def set_status(self, msg: str):
        self.root.after(0, lambda: self.status.set(msg))

    def on_quit(self):
        self.running = False
        with self.conn_lock:
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
