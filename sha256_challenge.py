"""
Bit-Pooling Higher-Lower Search & Circuit Camera Lab
(Tkinter / OpenCV Framework with Visual Bit Registers, Serial-Hex Receive & Forced Serial Read)
Adapted from the RP-style optoelectronic decision circuit framework.
"""

import asyncio
import hashlib
import queue
import random
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


# ==============================================================================
# SERIAL RECEIVER (serial port -> hex input)
# ==============================================================================
START_CMD = b"HEADER\n"       # sketch has no "start"; HEADER makes it reply HEADER:<DATA string>
DIFF_CMD_FMT = "diff {n}\n"   # sent by the 'Send diff n' button; edit to match the sketch

# Sizes / padding, matching sha256_challenge.ino
CHALLENGE_BYTES = 32                    # 256-bit challenge -> 64 hex chars
NONCE_BYTES = 8                         # 64-bit nonce      -> 16 hex chars
HASH_BYTES = 32                         # SHA-256 digest    -> 64 hex chars
BIT_DEPTH = CHALLENGE_BYTES * 8         # 256 bits
MAX_HEADER = 159                        # sketch: MAX_INPUT 160 incl. terminator
HEX_DIGITS = CHALLENGE_BYTES * 2        # 64


def pad_hex(value: int, nbytes: int) -> str:
    """Fixed-width, zero-padded, UPPERCASE hex with no prefix (as the Arduino sends/expects)."""
    return f"{value:0{nbytes * 2}X}"


def sha_pad(msg: bytes) -> bytes:
    """SHA-256 padding: msg + 0x80 + zeros + 64-bit big-endian bit length (multiple of 64 bytes)."""
    out = bytearray(msg) + b"\x80"
    while len(out) % 64 != 56:
        out.append(0)
    return bytes(out) + (len(msg) * 8).to_bytes(8, "big")


def meets_difficulty(digest: bytes, difficulty: int) -> bool:
    """True if the digest has at least `difficulty` leading zero bits (as checkDifficulty())."""
    return (int.from_bytes(digest, "big") >> (256 - difficulty)) == 0


def build_pow_line(header: str, nonce: int, digest: bytes) -> str:
    """
    Exactly the line the sketch prints when it finds a solution:
      POW:<nonce>:<hash>:<padded message>:<padded nonce>:<padded hash>:<header>
      nonce            decimal
      hash             64 lowercase hex
      padded message   SHA-256 padding of  header + decimal nonce   (ASCII)
      padded nonce     SHA-256 padding of  the decimal nonce text alone
      padded hash      SHA-256 padding of  the 32 raw hash bytes
      header           the DATA string (last, because it may contain colons)
    """
    ntext = str(nonce).encode("ascii")
    msg = header.encode("utf-8") + ntext
    return ":".join(["POW", str(nonce), digest.hex(),
                     sha_pad(msg).hex(), sha_pad(ntext).hex(), sha_pad(digest).hex(),
                     header])


def mine_pow(header: str, difficulty: int, stop=None, limit: int = 1 << 32):
    """
    Same search as the sketch: SHA256(header + decimal nonce), nonce counting up from 0.
    Returns (nonce, digest, hashes); nonce is None if cancelled or the nonce space ran out.
    """
    prefix = hashlib.sha256(header.encode("utf-8"))
    for nonce in range(limit):
        if stop is not None and nonce % 20000 == 0 and stop.is_set():
            return None, None, nonce
        h = prefix.copy()
        h.update(str(nonce).encode("ascii"))
        digest = h.digest()
        if meets_difficulty(digest, difficulty):
            return nonce, digest, nonce + 1
    return None, None, limit


class SerialReceiver:
    """Serial channel. Receives hex; transmits only the START and DIFF commands."""
    def __init__(self):
        self._port = None
        self.port_name = None
        self.baud = 115200

    @staticmethod
    def list_ports():
        if serial is None:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    @property
    def is_open(self):
        return self._port is not None

    def open(self, name, baud=115200):
        self.close()
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        self._port = serial.Serial(name, baud, timeout=0)
        self.port_name = name
        self.baud = baud

    def close(self):
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None
        self.port_name = None

    def send_start(self):
        """Send the START command so the device begins sending hex."""
        if self._port is None:
            return False
        try:
            self._port.write(START_CMD)
            return True
        except Exception:
            self.close()
            return False

    def send_diff(self, n: int):
        """Send the difficulty command, e.g. 'diff 12\\n'."""
        if self._port is None:
            return False
        try:
            self._port.write(DIFF_CMD_FMT.format(n=n).encode("ascii"))
            return True
        except Exception:
            self.close()
            return False

    def poll(self) -> bytes:
        """Return any bytes that arrived since the last call."""
        if self._port is None:
            return b""
        try:
            n = self._port.in_waiting
            return self._port.read(n) if n else b""
        except Exception:
            self.close()
            return b""


# ==============================================================================
# MESSAGE EXTRACTOR (serial text lines -> fields)
# ==============================================================================
class MessageExtractor:
    """
    Reassembles newline-terminated serial lines from sha256_challenge.ino:
        POW:<nonce>:<64-hex hash>:<padded message>:<padded nonce>:<padded hash>:<header>
        HEADER:<DATA string>
        [TERMINATED] / [IDLE] / [ERROR] ... (any line starting with '[')
    feed() returns a list of dicts, each with a "kind" key.
    """
    POW_RE = re.compile(r"^POW:(\d{1,10}):([0-9A-Fa-f]{64}):([0-9A-Fa-f]+):"
                        r"([0-9A-Fa-f]+):([0-9A-Fa-f]+):(.*)$")
    MAX_LINE = 1024                                 # longest sketch line is ~880 chars

    def __init__(self):
        self._buf = bytearray()
        self._overflow = False

    def reset(self):
        self._buf.clear()
        self._overflow = False

    def feed(self, data: bytes):
        out = []
        for b in data:
            if b == 0x0A:                                   # '\n' ends a message
                if not self._overflow:
                    msg = self._parse(bytes(self._buf).decode("utf-8", "replace").strip())
                    if msg is not None:
                        out.append(msg)
                self._buf.clear()
                self._overflow = False
            elif len(self._buf) < self.MAX_LINE:
                self._buf.append(b)
            else:
                self._overflow = True                       # discard the rest of a runaway line
        return out

    def _parse(self, line: str):
        if not line:
            return None
        m = self.POW_RE.match(line)
        if m:
            nonce = int(m.group(1))
            padded = (m.group(3), m.group(4), m.group(5))
            if nonce >= 1 << 32 or any(len(f) % 128 for f in padded):
                return dict(kind="BAD", text=line)          # padded fields are whole 64-byte blocks
            return dict(kind="POW", nonce=nonce, hash_hex=m.group(2).lower(),
                        header=m.group(6), line=line)
        if line.startswith("HEADER:"):
            return dict(kind="HEADER", header=line[7:])
        if line.startswith("POW:"):
            return dict(kind="BAD", text=line)              # right prefix, wrong format
        if line.startswith("["):
            return dict(kind="STATUS", text=line)
        return dict(kind="TEXT", text=line)


# ==============================================================================
# BIT-POOLING HIGHER-LOWER ORACLE CLASS
# ==============================================================================
class BitPoolingOracle:
    """
    Encapsulates the secret target and uses bit-pooling across successive
    frames/queries to accumulate bits MSB-first, mimicking the optoelectronic circuit.
    """
    def __init__(self, bit_depth=BIT_DEPTH, target=None):
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1
        self._target = (int(np.clip(target, 0, self.max_val))
                        if target is not None else random.randint(0, self.max_val))

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.pooled_bits = 0
        self.pooled_count = 0
        self.is_done = False

    def reveal(self):
        return self._target if self.is_done else 0

    def query_bit(self, trial_val):
        """Is the secret target >= the trial value?"""
        if self.is_done:
            return 0
        return 1 if self._target >= trial_val else 0

    def pool_step(self, val):
        """Applies hardware-style bit pooling (kernel, stride, operator) to a value."""
        bits = [(val >> i) & 1 for i in range(self.bit_depth)]

        pooled_out = []
        for i in range(0, len(bits), self.stride):
            block = bits[i: i + self.kernel_size]
            if not block:
                break

            if self.pool_op == "POPCOUNT":
                res_bit = 1 if (sum(block) / len(block)) >= self.schmitt_trigger else 0
            elif self.pool_op == "AND":
                res_bit = 1 if all(b == 1 for b in block) else 0
            elif self.pool_op == "OR":
                res_bit = 1 if any(b == 1 for b in block) else 0
            elif self.pool_op == "XOR":
                res_bit = sum(block) % 2
            elif self.pool_op == "MAJORITY":
                res_bit = 1 if sum(block) > (len(block) / 2.0) else 0
            else:
                res_bit = block[0]

            pooled_out.append(res_bit)

        out_val = 0
        for idx, b in enumerate(pooled_out):
            out_val |= (b << idx)
        return out_val

    @property
    def target(self):
        """Raises PermissionError if accessed before search completion."""
        if not self.is_done:
            raise PermissionError("SECURITY VIOLATION: Oracle target is classified until fully pooled!")
        return self._target


# ==============================================================================
# BIT-POOLING SEARCH ENGINE
# ==============================================================================
class BitPoolingSearchEngine:
    def __init__(self, bit_depth=BIT_DEPTH):
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        # Hardware & pooling settings (persist across resets)
        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.oracle = None
        self.reset()

    def reset(self, new_target=None):
        self.oracle = BitPoolingOracle(self.bit_depth, new_target)
        self.oracle.kernel_size = self.kernel_size
        self.oracle.stride = self.stride
        self.oracle.pool_op = self.pool_op
        self.oracle.schmitt_trigger = self.schmitt_trigger

        self.current_guess = 0
        self.step_count = 0
        self.history = [self.current_guess]
        self.converged = False
        self.last_status = "READY"
        self._update_registers()

    def step(self):
        """Advances the bit-pooling search by evaluating one bit per step."""
        if self.converged or self.step_count >= self.bit_depth:
            return

        self.step_count += 1

        shift = self.bit_depth - self.step_count
        trial_val = self.oracle.pooled_bits | (1 << shift)

        bit = self.oracle.query_bit(trial_val)
        self.oracle.pooled_bits |= (bit << shift)
        self.current_guess = self.oracle.pooled_bits

        self.history.append(self.current_guess)

        if self.step_count >= self.bit_depth:
            if self.current_guess == self.oracle._target:
                self.oracle.is_done = True
                self.converged = True
                self.last_status = f"LOCKED & REVEALED AT {self.current_guess}"
            else:
                self.last_status = "POOL MISMATCH"

        self._update_registers()

    def _update_registers(self):
        self.guess_reg = self.current_guess
        try:
            self.target_reg = self.oracle.target
        except PermissionError:
            self.target_reg = 0
        self.pooled_register = self.oracle.pool_step(self.current_guess)


# ==============================================================================
# CAMERA SOURCE HANDLING
# ==============================================================================
class Source:
    def __init__(self, index: int):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            raise RuntimeError(f"Cannot open camera {index}.")
        self.cap = cap
        self.rgb = None

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cap.release()


def ppm_photo(rgb: np.ndarray) -> tk.PhotoImage:
    h, w, _ = rgb.shape
    data = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


# ==============================================================================
# BACKGROUND SIMULATION THREAD
# ==============================================================================
class SimThread(threading.Thread):
    def __init__(self, cam_index, ctl, out, engine):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.ctl = ctl
        self.out = out
        self.engine = engine

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out, engine = self.ctl, self.out, self.engine
        try:
            src = Source(self.cam_index)
        except RuntimeError as e:
            out.put(("error", str(e)))
            out.put(("done", None))
            return
        out.put(("info", f"USB camera {self.cam_index} opened"))

        t_prev = time.perf_counter()
        idx = 0
        prev_even = False
        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    out.put(("info", "camera stopped delivering frames"))
                    break
                now = time.perf_counter()
                dt = min(max(now - t_prev, 1e-3), 0.25)
                t_prev = now

                small = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA)
                motion_detected = idx > 5 and np.std(small) > 5.0
                x = int(np.std(small) * 10) if motion_detected else 0

                # Two sequential even motion triggers advance the bit-pooling step
                even_now = x > 0 and x % 2 == 0
                even_state = even_now and prev_even
                prev_even = even_now

                if even_state and ctl.get("auto_step", True) and not engine.converged:
                    engine.step()

                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                pw, ph = 320, 240
                h_img, w_img = rgb.shape[:2]
                preview = rgb[np.arange(ph) * h_img // ph][:, np.arange(pw) * w_img // pw].copy()

                if ctl.get("ir_heatmap", False):
                    preview[:, :, 0] = np.clip(preview[:, :, 0].astype(int) + 80, 0, 255)
                    preview[:, :, 2] = np.clip(preview[:, :, 2].astype(int) - 50, 0, 255)

                # Target exposed on convergence, or once a forced serial read loaded it
                exposed_target = (engine.oracle._target
                                  if (engine.converged or ctl.get("serial_loaded", False)) else 0)

                out.put(("frame", dict(
                    preview=preview, idx=idx, dt=dt, target=exposed_target,
                    guess=engine.current_guess, pooled=engine.pooled_register,
                    status=engine.last_status, steps=engine.step_count,
                    history=list(engine.history), converged=engine.converged
                )))
                idx += 1
                await asyncio.sleep(0.03)
        finally:
            src.close()
            out.put(("done", None))


# ==============================================================================
# MAIN GUI APPLICATION
# ==============================================================================
class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Bit-Pooling Oracle Search Lab — Serial-Hex Input")
        self.q = queue.Queue()
        self.ctl = dict(stop=False, ir_heatmap=False, auto_step=True, serial_loaded=False)
        self.engine = BitPoolingSearchEngine(bit_depth=BIT_DEPTH)
        self.serial = SerialReceiver()
        self.extractor = MessageExtractor()
        self.rx_header = ""        # header / DATA string (the "hex" in the RX Hex box)
        self.rx_buf = b""          # header as 32 bytes, only if it is 64 hex chars
        self.rx_difficulty = None
        self.rx_line = ""          # converted POW line for rx_header (device's or ours)
        self.convert_mode = False  # RX Hex box shows the converted POW line
        self.mining_conv = False   # a local Convert search is running
        self.mine_stop = threading.Event()
        self.thread = None
        self.photo = None

        self._build_ui()
        self._draw_current()
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.after(15, self._poll)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill="both", expand=True)

        ctrl_frame = ttk.LabelFrame(main_frame, text="Laboratory Controls")
        ctrl_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(ctrl_frame, text="Start Lab", command=self.start_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Stop Lab", command=self.stop_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Step Bit Pool", command=self.step_search).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Reset Pool", command=self.reset_search).pack(side="left", padx=5, pady=5)

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="IR Thermal Heatmap", variable=self.v_ir,
                        command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get())).pack(side="left", padx=10)

        # Serial-hex frame (receives hex; sends START and DIFF commands)
        target_frame = ttk.LabelFrame(main_frame, text="Serial-Hex Input")
        target_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(target_frame, text="Port:").pack(side="left", padx=(5, 2), pady=5)
        self.port_combo = ttk.Combobox(target_frame, values=SerialReceiver.list_ports(), width=10)
        self.port_combo.pack(side="left", padx=2, pady=5)
        ttk.Button(target_frame, text="⟳", width=3, command=self._refresh_ports).pack(side="left", padx=1, pady=5)

        ttk.Label(target_frame, text="Baud:").pack(side="left", padx=(8, 2), pady=5)
        self.baud_combo = ttk.Combobox(target_frame, values=[9600, 19200, 38400, 57600, 115200, 230400],
                                       width=7)
        self.baud_combo.set("115200")
        self.baud_combo.pack(side="left", padx=2, pady=5)

        self.btn_connect = ttk.Button(target_frame, text="Connect", command=self.toggle_connect)
        self.btn_connect.pack(side="left", padx=5, pady=5)

        ttk.Label(target_frame, text="RX Hex:").pack(side="left", padx=(10, 2), pady=5)
        self.v_hex = tk.StringVar(value="")
        self.e_hex = ttk.Entry(target_frame, width=24, textvariable=self.v_hex)
        self.e_hex.pack(side="left", padx=2, pady=5)
        self.btn_convert = ttk.Button(target_frame, text="Convert", command=self.convert_rx)
        self.btn_convert.pack(side="left", padx=2, pady=5)
        ttk.Button(target_frame, text="Copy", command=self.copy_conv).pack(side="left", padx=2, pady=5)

        ttk.Button(target_frame, text="Randomize Target", command=self.randomize_target).pack(side="left", padx=5, pady=5)

        self.v_force_serial = tk.BooleanVar(value=False)
        self.chk_force_serial = ttk.Checkbutton(
            target_frame, text="Force Serial Read", variable=self.v_force_serial,
            command=self._on_force_serial)
        self.chk_force_serial.pack(side="left", padx=10, pady=5)

        ttk.Label(target_frame, text="diff n:").pack(side="left", padx=(10, 2), pady=5)
        self.diff_spin = ttk.Spinbox(target_frame, from_=0, to=BIT_DEPTH, increment=1, width=5)
        self.diff_spin.set(8)
        self.diff_spin.pack(side="left", padx=2, pady=5)
        ttk.Button(target_frame, text="Send diff n", command=self.send_diff).pack(side="left", padx=5, pady=5)

        # Viewport and registers
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        cam_frame = ttk.LabelFrame(display_frame, text="Optoelectronic Circuit Camera Viewport")
        cam_frame.pack(side="left", padx=5, pady=5)
        self.cam_canvas = tk.Canvas(cam_frame, width=320, height=240, bg="black")
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_img_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        right_panel = ttk.Frame(display_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        reg_frame = ttk.LabelFrame(right_panel, text="Bit-Pooling Hardware Registers")
        reg_frame.pack(fill="x", padx=5, pady=5)

        self.reg_canvas = tk.Canvas(reg_frame, width=480, height=140, bg="#121414", highlightthickness=0)
        self.reg_canvas.pack(padx=5, pady=5)

        # Selectable / copyable text view of the registers (hex + binary)
        dump_frame = ttk.LabelFrame(reg_frame, text="Register Dump (select / copy)")
        dump_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.txt_dump = tk.Text(dump_frame, width=58, height=7, wrap="char",
                                font=("Consolas", 9), bg="#121414", fg="#d2e0f0",
                                insertbackground="#d2e0f0", selectbackground="#38445c",
                                relief="flat", undo=False)
        self.txt_dump.pack(fill="x", padx=5, pady=5)
        self._dump_text = None
        self.txt_dump.bind("<Key>", self._dump_key)
        self.txt_dump.bind("<Button-1>", lambda e: self.txt_dump.focus_set())

        self.dump_menu = tk.Menu(self.window, tearoff=0)
        self.dump_menu.add_command(label="Copy", command=self._copy_selection)
        self.dump_menu.add_command(label="Select All", command=self._select_all_dump)
        for seq in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):
            self.txt_dump.bind(seq, self._dump_popup)

        dump_btns = ttk.Frame(dump_frame)
        dump_btns.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Button(dump_btns, text="Copy Hex", command=lambda: self._copy_registers("hex")).pack(side="left", padx=2)
        ttk.Button(dump_btns, text="Copy Binary", command=lambda: self._copy_registers("bin")).pack(side="left", padx=2)
        ttk.Button(dump_btns, text="Copy All", command=lambda: self._copy_registers("all")).pack(side="left", padx=2)

        self.lbl_status = ttk.Label(reg_frame, text="Status: IDLE", font=("Consolas", 11), foreground="#00e5ff")
        self.lbl_status.pack(anchor="w", padx=10, pady=(5, 0))
        self.lbl_serial = ttk.Label(reg_frame, text="Serial: disconnected | RX: (none)",
                                    font=("Consolas", 10), foreground="#88a0c0")
        self.lbl_serial.pack(anchor="w", padx=10, pady=(0, 5))

        set_frame = ttk.LabelFrame(right_panel, text="Pooling & Algorithm Settings")
        set_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(set_frame, text="Pooling Operator:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.op_combo = ttk.Combobox(set_frame, values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"], state="readonly")
        self.op_combo.set("POPCOUNT")
        self.op_combo.grid(row=0, column=1, padx=5, pady=5)
        self.op_combo.bind("<<ComboboxSelected>>", self._update_pool_op)

        ttk.Label(set_frame, text="Kernel Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.kernel_spin = ttk.Spinbox(set_frame, from_=2, to=8, increment=2, width=5, command=self._update_kernel)
        self.kernel_spin.set(4)
        self.kernel_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)

    # ---------------------------------------------------------------- controls
    def start_lab(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.ctl["stop"] = False
        self.thread = SimThread(0, self.ctl, self.q, self.engine)
        self.thread.start()

    def stop_lab(self):
        self.ctl["stop"] = True

    def step_search(self):
        self.engine.step()
        self._draw_current()

    def reset_search(self):
        self._load_target(None)
        self._clear_rx()

    def randomize_target(self):
        self._load_target(None)
        self._clear_rx()

    def _load_target(self, value):
        self.engine.reset(value)
        self.ctl["serial_loaded"] = False
        self._draw_current()

    # ---------------------------------------------------------------- serial hex
    def _refresh_ports(self):
        self.port_combo["values"] = SerialReceiver.list_ports()

    def toggle_connect(self):
        if self.serial.is_open:
            self.serial.close()
            self.btn_connect.config(text="Connect")
            self._refresh_serial_label()
            return
        name = self.port_combo.get().strip()
        if not name:
            messagebox.showerror("Serial Error", "Select a serial port first.")
            return
        try:
            self.serial.open(name, int(self.baud_combo.get()))
        except Exception as e:
            messagebox.showerror("Serial Error", str(e))
            return
        self.btn_connect.config(text="Disconnect")
        self._refresh_serial_label()

    def _clear_rx(self):
        self.mine_stop.set()                    # cancel any running Convert search
        self.mining_conv = False
        self.rx_header = ""
        self.rx_buf = b""
        self.rx_difficulty = None
        self.rx_line = ""
        self.convert_mode = False
        self.extractor.reset()
        self._show_rx()
        self._refresh_serial_label()

    def _on_force_serial(self):
        if self.v_force_serial.get():
            if self.serial.send_start():
                self._refresh_serial_label("TX: start")
            self._apply_rx()
        else:
            self.ctl["serial_loaded"] = False
            self._draw_current()

    def send_diff(self):
        """Send 'diff <n>' to the device using the value in the diff spinbox."""
        try:
            n = int(self.diff_spin.get())
        except ValueError:
            messagebox.showerror("Serial Error", "Difficulty must be an integer.")
            return
        if not 0 <= n <= BIT_DEPTH:
            messagebox.showerror("Serial Error", f"Difficulty must be 0-{BIT_DEPTH}.")
            return
        if not self.serial.is_open:
            messagebox.showerror("Serial Error", "Connect to a serial port first.")
            return
        if self.serial.send_diff(n):
            self._refresh_serial_label(f"TX: diff {n}")
        else:
            self.btn_connect.config(text="Connect")
            self._refresh_serial_label("(send failed)")

    def _set_header(self, text: str):
        self.rx_header = text
        self.rx_buf = (bytes.fromhex(text)
                       if re.fullmatch(rf"[0-9A-Fa-f]{{{HEX_DIGITS}}}", text) else b"")

    def _show_rx(self):
        """RX Hex box: the converted POW line in convert mode, otherwise the raw header."""
        if self.convert_mode and self.rx_line:
            self.v_hex.set(self.rx_line)
        else:
            self.v_hex.set(self.rx_header)
        self.e_hex.icursor(0)
        self.e_hex.xview_moveto(0)
        self.btn_convert.config(
            text="Cancel" if self.mining_conv else ("Show Raw" if self.convert_mode else "Convert"))

    def convert_rx(self):
        """Convert the RX Hex text into the sketch's POW line (searches the nonce from 0)."""
        if self.mining_conv:                    # pressed while searching: cancel
            self.mine_stop.set()
            self._refresh_serial_label("(cancelling...)")
            return
        if self.convert_mode:                   # pressed while converted: back to raw header
            self.convert_mode = False
            self._show_rx()
            self._refresh_serial_label("(showing raw header)")
            return

        header = self.v_hex.get().strip() or self.rx_header
        if header.upper().startswith("POW:"):
            messagebox.showinfo("Convert", "RX Hex already holds a POW line.")
            return
        if not header:
            messagebox.showerror("Convert", "RX Hex is empty: paste a header, or use Force Serial Read.")
            return
        if len(header) > MAX_HEADER:
            messagebox.showerror("Convert", f"Header is {len(header)} characters; the sketch allows {MAX_HEADER}.")
            return
        try:
            diff = int(self.diff_spin.get())
        except ValueError:
            messagebox.showerror("Convert", "Difficulty must be an integer.")
            return
        if not 0 <= diff <= BIT_DEPTH:
            messagebox.showerror("Convert", f"Difficulty must be 0-{BIT_DEPTH}.")
            return
        if diff > 26 and not messagebox.askokcancel(
                "Convert", f"Difficulty {diff} may take a very long time in Python. Continue?"):
            return

        self.mine_stop = threading.Event()
        self.mining_conv = True
        self._set_header(header)
        self._show_rx()                         # button becomes Cancel
        self._refresh_serial_label(f"(searching nonce, diff {diff}...)")
        threading.Thread(target=self._mine_worker, args=(header, diff, self.mine_stop),
                         daemon=True).start()

    def _mine_worker(self, header, diff, stop):
        res = mine_pow(header, diff, stop)
        self.q.put(("mined", dict(header=header, result=res, stop=stop)))

    def _on_mined(self, d):
        if d["stop"] is not self.mine_stop:     # result of an older, replaced search
            return
        self.mining_conv = False
        nonce, digest, hashes = d["result"]
        if nonce is None:
            self._show_rx()
            self._refresh_serial_label("(convert cancelled)")
            return
        self._set_header(d["header"])
        self.rx_line = build_pow_line(d["header"], nonce, digest)
        self.convert_mode = True
        self._show_rx()
        self._refresh_serial_label(f"(converted: nonce={nonce}, {hashes} hashes)")

    def copy_conv(self):
        text = self.v_hex.get()
        if not text:
            return
        self.window.clipboard_clear()
        self.window.clipboard_append(text)
        self._refresh_serial_label("(copied RX Hex)")

    def _apply_rx(self):
        """Load the received hex value as the oracle target and unmask TARGET."""
        if not self.rx_buf:
            return
        value = min(int.from_bytes(self.rx_buf, "big"), self.engine.max_val)
        self.engine.reset(value)
        self.ctl["serial_loaded"] = True
        self._refresh_serial_label(f"-> {value}")
        self._draw_current()

    def _refresh_serial_label(self, note=""):
        conn = f"{self.serial.port_name} @{self.serial.baud}" if self.serial.is_open else "disconnected"
        if self.rx_header:
            h = self.rx_header
            rx = h if len(h) <= 20 else f"{h[:8]}...{h[-8:]}"
            if self.rx_difficulty is not None:
                rx += f" d={self.rx_difficulty}"
        else:
            rx = "(none)"
        self.lbl_serial.config(text=f"Serial: {conn} | RX: {rx} {note}".rstrip())

    # ---------------------------------------------------------------- settings
    def _update_pool_op(self, _event):
        op = self.op_combo.get()
        self.engine.pool_op = op
        self.engine.oracle.pool_op = op
        self._draw_current()

    def _update_kernel(self):
        try:
            k = int(self.kernel_spin.get())
        except ValueError:
            return
        self.engine.kernel_size = k
        self.engine.oracle.kernel_size = k
        self._draw_current()

    # ---------------------------------------------------------------- drawing
    def _draw_current(self):
        e = self.engine
        e._update_registers()
        exposed = e.converged or self.ctl.get("serial_loaded", False)
        self._draw_bit_rows(e.oracle._target if exposed else 0,
                            e.current_guess, e.pooled_register, exposed)
        self.lbl_status.config(text=f"Status: {e.last_status} | Bit Step: {e.step_count}/{e.bit_depth}")

    def _draw_bit_rows(self, target_val, guess_val, pooled_val, exposed):
        self._update_dump(target_val, guess_val, pooled_val, exposed)
        self.reg_canvas.delete("all")
        bit_depth = self.engine.bit_depth

        rows = [
            ("TARGET", target_val if exposed else 0, "#00e676", not exposed),
            ("GUESS ", guess_val, "#ffb300", False),
            ("POOLED", pooled_val, "#b388ff", False),
        ]

        y_offset = 10
        for label, val, color, hidden in rows:
            self.reg_canvas.create_text(10, y_offset + 10, text=label, fill="#88a0c0",
                                        anchor="w", font=("Consolas", 10, "bold"))
            bx = 90
            for b in range(bit_depth - 1, -1, -1):
                bit_state = (val >> b) & 1
                self.reg_canvas.create_rectangle(
                    bx, y_offset, bx + 22, y_offset + 22,
                    fill=color if (bit_state and not hidden) else "#1e2430",
                    outline="#38445c")
                txt_val = "?" if hidden else str(bit_state)
                txt_color = "#000000" if (bit_state and not hidden) else "#708090"
                self.reg_canvas.create_text(bx + 11, y_offset + 11, text=txt_val,
                                            fill=txt_color, font=("Consolas", 9, "bold"))
                bx += 26

            num_str = "[HIDDEN]" if hidden else f"[0x{val:03X}]"
            self.reg_canvas.create_text(bx + 5, y_offset + 11, text=num_str,
                                        fill="#d2e0f0", anchor="w", font=("Consolas", 9))
            y_offset += 40

    # ---------------------------------------------------------------- register dump (copy / paste)
    def _register_strings(self, target_val, guess_val, pooled_val, exposed):
        """Return {name: (hex, bin)} for each register; target is masked until exposed."""
        bd = self.engine.bit_depth
        digits = (bd + 3) // 4
        out = {}
        for name, val, hidden in (("TARGET", target_val, not exposed),
                                  ("GUESS", guess_val, False),
                                  ("POOLED", pooled_val, False)):
            if hidden:
                out[name] = ("[HIDDEN]", "[HIDDEN]")
            else:
                out[name] = (f"{val:0{digits}X}", f"{val:0{bd}b}")
        self._last_regs = out
        return out

    def _update_dump(self, target_val, guess_val, pooled_val, exposed):
        regs = self._register_strings(target_val, guess_val, pooled_val, exposed)
        lines = []
        for name, (h, b) in regs.items():
            lines.append(f"{name} HEX: {h}")
            lines.append(f"{name} BIN: {b}")
        text = "\n".join(lines)
        if text == self._dump_text:          # unchanged: keep the user's selection
            return
        self._dump_text = text
        self.txt_dump.delete("1.0", "end")
        self.txt_dump.insert("1.0", text)

    def _dump_key(self, event):
        """Read-only: allow copy, select-all and navigation; block everything else."""
        ctrl = (event.state & 0x4) != 0
        if ctrl and event.keysym.lower() in ("c", "a", "insert"):
            if event.keysym.lower() == "a":
                self._select_all_dump()
                return "break"
            return None
        if event.keysym in ("Left", "Right", "Up", "Down", "Home", "End",
                            "Prior", "Next", "Shift_L", "Shift_R", "Control_L", "Control_R"):
            return None
        return "break"

    def _select_all_dump(self):
        self.txt_dump.tag_add("sel", "1.0", "end-1c")
        self.txt_dump.focus_set()

    def _dump_popup(self, event):
        self.txt_dump.focus_set()
        self.dump_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _copy_selection(self):
        try:
            text = self.txt_dump.get("sel.first", "sel.last")
        except tk.TclError:
            return
        self.window.clipboard_clear()
        self.window.clipboard_append(text)

    def _copy_registers(self, mode):
        regs = getattr(self, "_last_regs", None)
        if not regs:
            return
        lines = []
        for name, (h, b) in regs.items():
            if mode in ("hex", "all"):
                lines.append(f"{name} HEX: {h}" if mode == "all" else f"{name}: {h}")
            if mode in ("bin", "all"):
                lines.append(f"{name} BIN: {b}" if mode == "all" else f"{name}: {b}")
        self.window.clipboard_clear()
        self.window.clipboard_append("\n".join(lines))
        self._refresh_serial_label(f"(copied {mode})")

    # ---------------------------------------------------------------- polling
    def _poll(self):
        # Receive serial hex; always shown, only applied while Force Serial Read is on
        new = self.serial.poll()
        if new:
            for msg in self.extractor.feed(new):
                kind = msg["kind"]
                if kind == "POW":
                    self._set_header(msg["header"])
                    self.rx_line = msg["line"]
                    self._show_rx()
                    self._refresh_serial_label(f"POW nonce={msg['nonce']}")
                    if self.v_force_serial.get():
                        self._apply_rx()
                elif kind == "HEADER":
                    if msg["header"] != self.rx_header:
                        self.rx_line = ""
                        self.convert_mode = False
                    self._set_header(msg["header"])
                    self._show_rx()
                    self._refresh_serial_label()
                    if self.v_force_serial.get():
                        self._apply_rx()
                elif kind in ("STATUS", "BAD"):
                    self._refresh_serial_label(f"[{kind}: {msg['text'][:24]}]")
        elif self.btn_connect.cget("text") == "Disconnect" and not self.serial.is_open:
            self.btn_connect.config(text="Connect")
            self._refresh_serial_label("(port lost)")

        try:
            for _ in range(100):
                ev = self.q.get_nowait()
                if ev[0] == "frame":
                    d = ev[1]
                    self.photo = ppm_photo(d["preview"])
                    self.cam_canvas.itemconfig(self.cam_img_id, image=self.photo)
                    exposed = d["converged"] or self.ctl.get("serial_loaded", False)
                    self._draw_bit_rows(d["target"], d["guess"], d["pooled"], exposed)
                    self.lbl_status.config(
                        text=f"Status: {d['status']} | Bit Step: {d['steps']}/{self.engine.bit_depth}")
                elif ev[0] == "mined":
                    self._on_mined(ev[1])
                elif ev[0] == "error":
                    messagebox.showerror("Camera Error", ev[1])
        except queue.Empty:
            pass
        self.window.after(30, self._poll)

    def _close(self):
        self.ctl["stop"] = True
        self.serial.close()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("1260x700")
    root.mainloop()
