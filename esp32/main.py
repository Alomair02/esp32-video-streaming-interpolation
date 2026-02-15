import gc
import network
import socket
import time
from machine import Pin, I2C, SPI
import uselect

try:
    from machine import SoftSPI
except ImportError:
    SoftSPI = None

import neopixel
from Arducam import Arducam

SSID = "YOUR_SSID"
PASSWORD = "YOUR_PASSWORD"
SERVER_IP = "YOUR_SERVER_IP"
SERVER_PORT = 4444

CMD_BUF_SIZE = 64
FREEZE_RESEND_MS = 400
MAX_CMD_BUFFER = 1024
JPEG_SIZE_LIMIT = 256 * 1024
CAMERA_FRAMESIZE = "QVGA"
FRAME_MAGIC = b"FRM"

# NeoPixel RGB LED on GPIO 48
np = neopixel.NeoPixel(Pin(48, Pin.OUT), 1)


def led_red():
    np[0] = (50, 0, 0)
    np.write()


def led_green():
    np[0] = (0, 50, 0)
    np.write()


def connect_wifi():
    w = network.WLAN(network.STA_IF)
    w.active(False)
    time.sleep(1)
    w.active(True)
    time.sleep(2)

    try:
        w.disconnect()
    except Exception:
        pass
    time.sleep(1)

    try:
        w.scan()
        print("Available WiFi networks:", [ssid.decode() for ssid, *_ in w.scan()])
    except Exception:
        pass

    print("WiFi: connecting to", SSID)
    w.connect(SSID, PASSWORD)

    for _ in range(30):
        if w.isconnected():
            print("WiFi: connected, IP =", w.ifconfig()[0])
            return True
        print("WiFi: status", w.status())
        time.sleep(1)

    print("WiFi: failed")
    return False


def connect_server():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((SERVER_IP, SERVER_PORT))
            print("Socket: connected to", SERVER_IP, SERVER_PORT)
            return s
        except Exception as e:
            print("Socket: connect failed:", repr(e))
            time.sleep(2)


def build_spi_candidates():
    cands = [
        ("SPI(1) 1MHz mode0", lambda: SPI(1, baudrate=1_000_000, polarity=0, phase=0,
                                          sck=Pin(12), mosi=Pin(10), miso=Pin(11))),
        ("SPI(2) 1MHz mode0", lambda: SPI(2, baudrate=1_000_000, polarity=0, phase=0,
                                          sck=Pin(12), mosi=Pin(10), miso=Pin(11))),
        ("SPI(1) 2MHz mode0", lambda: SPI(1, baudrate=2_000_000, polarity=0, phase=0,
                                          sck=Pin(12), mosi=Pin(10), miso=Pin(11))),
        ("SPI(2) 2MHz mode0", lambda: SPI(2, baudrate=2_000_000, polarity=0, phase=0,
                                          sck=Pin(12), mosi=Pin(10), miso=Pin(11))),
    ]
    if SoftSPI is not None:
        cands.append(
            ("SoftSPI 1MHz mode0", lambda: SoftSPI(baudrate=1_000_000, polarity=0, phase=0,
                                                   sck=Pin(12), mosi=Pin(10), miso=Pin(11)))
        )
    return cands


def init_camera():
    i2c = I2C(0, scl=Pin(14), sda=Pin(13), freq=100_000)
    cs_pin = Pin(9, Pin.OUT, value=1)

    cam = None
    for label, builder in build_spi_candidates():
        try:
            spi = builder()
        except Exception as e:
            print("SPI setup failed for", label, ":", repr(e))
            continue

        print("Trying", label)
        test_cam = Arducam(spi, cs_pin, i2c)
        if test_cam.Spi_Test(3):
            print("Using", label)
            cam = test_cam
            break

    if cam is None:
        raise RuntimeError("SPI test failed on all candidates; check CS/MISO/MOSI/SCK wiring and 3.3V/GND.")

    cam.Camera_Detection()
    cam.init()
    cam.set_jpeg()
    cam.set_framesize(CAMERA_FRAMESIZE)
    cam.set_max_jpeg_size(JPEG_SIZE_LIMIT)
    return cam


def is_no_data_error(exc):
    if not exc.args:
        return False
    code = exc.args[0]
    return code in (11, 110, 116, 118, 119)


def make_poller(sock):
    p = uselect.poll()
    p.register(sock, uselect.POLLIN)
    return p


def read_commands(sock, pending, poller):
    if not poller.poll(0):
        return pending, []

    try:
        data = sock.recv(CMD_BUF_SIZE)
    except OSError as e:
        if is_no_data_error(e):
            return pending, []
        raise

    if not data:
        raise ConnectionError("server closed")

    pending += data
    cmds = []
    while True:
        nl = pending.find(b"\n")
        if nl < 0:
            break
        line = pending[:nl]
        pending = pending[nl + 1:]
        cmd = line.decode("utf-8", "ignore").strip().upper()
        if cmd:
            cmds.append(cmd)

    if len(pending) > MAX_CMD_BUFFER:
        pending = pending[-MAX_CMD_BUFFER:]

    return pending, cmds


def send_jpeg(sock, jpg):
    header = FRAME_MAGIC + len(jpg).to_bytes(4, "big")
    sock.sendall(header)
    sock.sendall(jpg)


def main():
    print("BOOT")
    led_red()  # stopped by default

    cam = init_camera()

    if not connect_wifi():
        return

    sock = connect_server()
    pending = b""
    poller = make_poller(sock)
    streaming = False
    last_frame = b""
    last_freeze_send = time.ticks_ms()
    frame_count = 0

    while True:
        try:
            pending, cmds = read_commands(sock, pending, poller)
            for cmd in cmds:
                print("CMD:", cmd)
                if cmd == "START":
                    streaming = True
                    led_green()
                elif cmd == "STOP":
                    streaming = False
                    led_red()
                    if last_frame:
                        send_jpeg(sock, last_frame)
                        last_freeze_send = time.ticks_ms()
                elif cmd == "CLOSE":
                    print("Socket: closing connection as requested")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    return

            if streaming:
                cam.capture()
                jpg = cam.read_jpeg()
                if jpg:
                    last_frame = jpg
                    send_jpeg(sock, jpg)
                    frame_count += 1
                    if frame_count % 20 == 0:
                        gc.collect()
                else:
                    time.sleep_ms(5)
            else:
                if last_frame:
                    now = time.ticks_ms()
                    if time.ticks_diff(now, last_freeze_send) >= FREEZE_RESEND_MS:
                        send_jpeg(sock, last_frame)
                        last_freeze_send = now
                    else:
                        time.sleep_ms(20)
                else:
                    time.sleep_ms(20)

        except Exception as e:
            print("Socket/stream error:", repr(e))
            try:
                sock.close()
            except Exception:
                pass
            sock = connect_server()
            pending = b""
            poller = make_poller(sock)
            if streaming:
                led_green()
            else:
                led_red()


main()
