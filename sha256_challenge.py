"""
Bit-Pooling Higher-Lower Search & Circuit Camera Lab
(Tkinter / OpenCV Framework with Visual Bit Registers, Auto-Revealing Oracle
 and a Serial Peripheral linked to the TARGET register)

The serial peripheral is bound to the oracle with the SAME gating logic as the
TARGET register: nothing about the target leaves the oracle until it is locked.

  oracle  --events-->  serial TX     EVT RESET
                                     EVT STEP n/N GUESS=0x... BIT=b
                                     EVT LOCK TARGET=<dec>     (only after lock = auto-reveal)
  serial RX --commands--> oracle     SET <dec|0xhex>   load a new hidden target
                                     RESET             new random target
                                     STEP              answer one bit
                                     GET               replies "EVT GET TARGET=n" or "EVT GET LOCKED"

Registers match the padded SHA-256 block of sha256_challenge.ino: every register is 512 bits
(one 64-byte block = message + 0x80 + zeros + 64-bit bit length), shown as 128 hex characters.
Hex values on the wire are 128 characters. SET accepts a number (0x... hex or decimal) or plain
text (up to 55 bytes), which is padded into a block exactly like sha256_pad().

Port "LOOPBACK" is a built-in virtual device (every line sent comes back), so the link
works with no hardware. Real ports need:  pip install pyserial
"""

import asyncio
import hashlib
import queue
import secrets
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


# ==============================================================================
# PADDED SHA-256 BLOCK (same layout as sha256_challenge.ino)
# ==============================================================================
BLOCK_BYTES = 64
BLOCK_BITS = BLOCK_BYTES * 8          # 512-bit hardware registers
BLOCK_HEX = BLOCK_BITS // 4           # 128 hex characters
POW_DATA = "arduino-sha256"


def sha256_pad(msg: bytes) -> bytes:
    """Minimal SHA-256 padding: msg, 0x80, zeros, 64-bit big-endian bit length."""
    total = ((len(msg) + 9 + 63) // 64) * 64
    return msg + b"\x80" + b"\x00" * (total - len(msg) - 9) + (len(msg) * 8).to_bytes(8, "big")


def sha256_unpad(padded: bytes) -> bytes:
    """Inverse of sha256_pad. Raises ValueError if padding is not minimal/valid."""
    if len(padded) < 64 or len(padded) % 64:
        raise ValueError("not a whole number of 64-byte blocks")
    length = int.from_bytes(padded[-8:], "big") // 8
    if length > len(padded) - 9 or sha256_pad(padded[:length]) != padded:
        raise ValueError("invalid padding")
    return padded[:length]


def padded_block_int(msg: bytes) -> int:
    """Message -> single padded 512-bit register value."""
    if len(msg) > 55:
        raise ValueError("message must be at most 55 bytes to fit one 512-bit block")
    return int.from_bytes(sha256_pad(msg), "big")


def random_padded_target() -> int:
    """Default hidden target: padded block of POW_DATA + a random nonce (like the miner's message)."""
    return padded_block_int((POW_DATA + str(secrets.randbelow(1 << 32))).encode("ascii"))


def decode_block(value: int):
    """(message_text, sha256_hex) of a padded block register, or None if the padding is invalid."""
    try:
        msg = sha256_unpad(value.to_bytes(BLOCK_BYTES, "big"))
    except (ValueError, OverflowError):
        return None
    return msg.decode("ascii", errors="replace"), hashlib.sha256(msg).hexdigest()


# ==============================================================================
# BIT-POOLING HIGHER-LOWER ORACLE CLASS
# ==============================================================================
class BitPoolingOracle:
    """
    Encapsulates the secret target and uses bit-pooling across successive
    queries to accumulate bits MSB-first, mimicking the optoelectronic circuit.
    """
    def __init__(self, bit_depth=BLOCK_BITS, target=None):
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1
        if target is None:
            target = random_padded_target() if bit_depth == BLOCK_BITS else secrets.randbits(bit_depth)
        self._target = int(target) & self.max_val

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
        """Is the secret target >= trial_val?  Returns 1 or 0."""
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
                res = sum(block) / len(block)
                res_bit = 1 if res >= self.schmitt_trigger else 0
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
# BIT-POOLING SEARCH ENGINE  (thread-safe, publishes events to listeners)
# ==============================================================================
class BitPoolingSearchEngine:
    def __init__(self, bit_depth=BLOCK_BITS):
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1
        self.lock = threading.RLock()          # camera thread, GUI and serial all drive the engine
        self.listeners = []                    # callables f(event_name, **data)

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.oracle = BitPoolingOracle(self.bit_depth)
        self.current_guess = 0
        self.step_count = 0
        self.history = [0]
        self.converged = False
        self.last_status = "READY"
        self.pooled_register = 0
        self._update_registers()

    def _emit(self, name, **data):
        for f in list(self.listeners):
            try:
                f(name, **data)
            except Exception:
                pass

    def reset(self, new_target=None):
        with self.lock:
            self.oracle = BitPoolingOracle(self.bit_depth, new_target)
            self.oracle.pool_op = self.pool_op
            self.oracle.kernel_size = self.kernel_size
            self.current_guess = 0
            self.step_count = 0
            self.history = [0]
            self.converged = False
            self.last_status = "READY"
            self._update_registers()
            self._emit("reset")

    def step(self):
        """Answers one higher/lower query (one bit, MSB first)."""
        with self.lock:
            if self.converged or self.step_count >= self.bit_depth:
                return
            self.step_count += 1
            shift = self.bit_depth - self.step_count
            trial_val = self.oracle.pooled_bits | (1 << shift)
            bit = self.oracle.query_bit(trial_val)

            self.oracle.pooled_bits |= (bit << shift)
            self.current_guess = self.oracle.pooled_bits
            self.history.append(self.current_guess)
            self.last_status = f"POOLING {self.step_count}/{self.bit_depth}"
            self._update_registers()
            self._emit("step", step=self.step_count, total=self.bit_depth,
                       guess=self.current_guess, bit=bit)

            if self.step_count >= self.bit_depth:
                if self.current_guess == self.oracle._target:
                    self.oracle.is_done = True          # auto-reveal: unlocks the target
                    self.converged = True
                    self.last_status = f"LOCKED & REVEALED AT {self.current_guess}"
                    self._update_registers()
                    self._emit("lock", target=self.oracle._target)   # <- feeds the serial peripheral
                else:
                    self.last_status = "POOL MISMATCH"

    def set_pool_op(self, op):
        with self.lock:
            self.pool_op = op
            self.oracle.pool_op = op

    def set_kernel(self, k):
        with self.lock:
            self.kernel_size = k
            self.oracle.kernel_size = k

    def _update_registers(self):
        self.guess_reg = self.current_guess
        try:
            self.target_reg = self.oracle.target
        except PermissionError:
            self.target_reg = 0
        self.pooled_register = self.oracle.pool_step(self.current_guess)

    def snapshot(self, force_reveal=False):
        with self.lock:
            return dict(
                target=self.oracle._target if (self.converged or force_reveal) else 0,
                guess=self.current_guess,
                pooled=self.pooled_register,
                status=self.last_status,
                steps=self.step_count,
                converged=self.converged,
            )


# ==============================================================================
# SERIAL PERIPHERAL  (real pyserial port, or built-in LOOPBACK virtual device)
# ==============================================================================
class SerialPeripheral:
    LOOPBACK = "LOOPBACK"

    def __init__(self, on_command):
        self.on_command = on_command          # f(cmd, arg) -> called from the RX thread; must only enqueue
        self.log = queue.Queue()              # ("TX"|"RX"|"SYS", text) for the GUI
        self.txq = queue.Queue()
        self._lb = queue.Queue()              # loopback wire
        self._stop = threading.Event()
        self._port = None
        self.name = None
        self.connected = False
        self.reg = None                       # the peripheral's mirrored TARGET register (None = no data)

    @staticmethod
    def available_ports():
        names = [SerialPeripheral.LOOPBACK]
        try:
            from serial.tools import list_ports
            names += [p.device for p in list_ports.comports()]
        except Exception:
            pass
        return names

    def connect(self, name, baud=115200):
        self.disconnect()
        port = None
        if name != self.LOOPBACK:
            try:
                import serial
            except ImportError:
                raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
            port = serial.Serial(name, baud, timeout=0.1, write_timeout=1.0)
        stop = threading.Event()
        self._stop, self._port, self.name = stop, port, name
        self.reg = None
        self.connected = True
        for target in (self._tx_loop, self._rx_loop):
            threading.Thread(target=target, args=(stop, port), daemon=True).start()
        self.log.put(("SYS", f"connected: {name}" + (f" @ {baud}" if port else " (virtual)")))

    def disconnect(self):
        if not self.connected:
            return
        self._stop.set()
        try:
            if self._port is not None:
                self._port.close()
        except Exception:
            pass
        self._port, self.connected = None, False
        self.log.put(("SYS", f"disconnected: {self.name}"))

    # ---- called by the engine link / GUI (any thread, non-blocking) ----------
    def send(self, line):
        if not self.connected:
            return
        if line.startswith("EVT LOCK TARGET="):          # the linked register follows what is sent
            self.reg = int(line.split("=", 1)[1], 16)
        elif line.startswith("EVT RESET"):
            self.reg = None
        self.txq.put(line)

    # ---- worker threads --------------------------------------------------------
    def _tx_loop(self, stop, port):
        while not stop.is_set():
            try:
                line = self.txq.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if port is not None:
                    port.write((line + "\n").encode("ascii", "replace"))
                else:
                    self._lb.put(line)                   # virtual device echoes every line
                self.log.put(("TX", line))
            except Exception as e:
                self.log.put(("SYS", f"write error: {e}"))

    def _rx_loop(self, stop, port):
        while not stop.is_set():
            try:
                if port is not None:
                    raw = port.readline()
                    line = raw.decode("ascii", "replace").strip() if raw else ""
                else:
                    line = self._lb.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                if not stop.is_set():
                    self.log.put(("SYS", f"read error: {e}"))
                    self.connected = False
                return
            if line:
                self.log.put(("RX", line))
                self._parse(line)

    def _parse(self, line):
        up = line.strip()
        if up.startswith("EVT LOCK TARGET="):
            try:
                self.reg = int(up.split("=", 1)[1], 16)
            except ValueError:
                pass
        elif up.startswith("EVT RESET"):
            self.reg = None
        elif up.startswith("EVT"):
            pass                                          # our own events echoed back
        else:
            parts = up.split(None, 1)
            self.on_command(parts[0].upper(), parts[1] if len(parts) > 1 else "")


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
# BACKGROUND SIMULATION THREAD  (camera -> preview + motion triggers)
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

        idx = 0
        prev_even = False
        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    out.put(("info", "camera stopped delivering frames"))
                    break

                small = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA)
                motion_detected = idx > 5 and np.std(small) > 5.0
                x = int(np.std(small) * 10) if motion_detected else 0

                # Two sequential even motion triggers advance the bit-pooling step
                even_now = x > 0 and x % 2 == 0
                even_state = even_now and prev_even
                prev_even = even_now

                if even_state and ctl.get("auto_step", True):
                    engine.step()

                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                pw, ph = 320, 240
                h_img, w_img = rgb.shape[:2]
                preview = rgb[np.arange(ph) * h_img // ph][:, np.arange(pw) * w_img // pw].copy()

                if ctl.get("ir_heatmap", False):
                    preview[:, :, 0] = np.clip(preview[:, :, 0].astype(int) + 80, 0, 255)
                    preview[:, :, 2] = np.clip(preview[:, :, 2].astype(int) - 50, 0, 255)

                out.put(("frame", dict(preview=preview, idx=idx)))     # preview only
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
        self.window.title("Bit-Pooling Oracle Search Lab — Target linked to Serial Peripheral")
        self.q = queue.Queue()
        self.cmdq = queue.Queue()                      # serial RX commands -> Tk thread
        self.ctl = dict(stop=False, ir_heatmap=False, auto_step=True, force_reveal=False)
        self.engine = BitPoolingSearchEngine(bit_depth=BLOCK_BITS)
        self.thread = None
        self._reg_key = None

        # --- the link: oracle events -> serial TX, serial RX commands -> oracle ---
        self.serial = SerialPeripheral(on_command=lambda c, a: self.cmdq.put((c, a)))
        self.engine.listeners.append(self._engine_event)

        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.after(15, self._poll)
        self.window.after(50, self._tick)

    # ---- oracle -> serial (runs on whichever thread drove the engine; only enqueues) ----
    def _engine_event(self, name, **d):
        s = self.serial
        if name == "reset":
            s.send("EVT RESET")
        elif name == "step":
            s.send(f"EVT STEP {d['step']}/{d['total']} GUESS={d['guess']:0{BLOCK_HEX}X} BIT={d['bit']}")
        elif name == "lock":
            s.send(f"EVT LOCK TARGET={d['target']:0{BLOCK_HEX}X}")   # target leaves the oracle only once locked

    # ---- serial -> oracle (Tk thread) ----
    def _handle_command(self, cmd, arg):
        e, s = self.engine, self.serial
        if cmd == "SET":
            arg = arg.strip()
            try:
                val = int(arg, 0)                                  # number: decimal or 0x hex
            except ValueError:
                try:
                    val = padded_block_int(arg.encode("ascii"))    # text: padded into one block
                except (ValueError, UnicodeEncodeError) as ex:
                    s.send(f"EVT ERR {ex}")
                    return
            if 0 <= val <= e.max_val:
                e.reset(val)
            else:
                s.send(f"EVT ERR range 0..2^{e.bit_depth}-1")
        elif cmd == "RESET":
            e.reset()
        elif cmd == "STEP":
            e.step()
        elif cmd == "GET":
            snap = e.snapshot(False)
            s.send(f"EVT GET TARGET={snap['target']:0{BLOCK_HEX}X}" if snap["converged"] else "EVT GET LOCKED")
        else:
            s.send(f"EVT ERR unknown-command {cmd}")

    def _build_ui(self):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill="both", expand=True)

        ctrl_frame = ttk.LabelFrame(main_frame, text="Laboratory Controls")
        ctrl_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(ctrl_frame, text="Start Lab", command=self.start_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Stop Lab", command=self.stop_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Step Bit Pool", command=self.step_search).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Run Full Oracle", command=self.run_oracle).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Reset Pool", command=self.reset_search).pack(side="left", padx=5, pady=5)

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="IR Thermal Heatmap", variable=self.v_ir,
                        command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get())).pack(side="left", padx=10)

        target_frame = ttk.LabelFrame(main_frame, text="Hidden Target Management (Auto-Revealing Oracle)")
        target_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(target_frame, text="Target (text, 0x hex or decimal; padded to 512 bits):").pack(side="left", padx=5, pady=5)
        self.v_custom_target = tk.StringVar(value="[SECURE]")
        ttk.Entry(target_frame, width=28, textvariable=self.v_custom_target).pack(side="left", padx=5, pady=5)
        ttk.Button(target_frame, text="Set Custom", command=self.set_custom_target).pack(side="left", padx=2, pady=5)
        ttk.Button(target_frame, text="Randomize Target", command=self.randomize_target).pack(side="left", padx=5, pady=5)

        self.v_reveal = tk.BooleanVar(value=False)
        ttk.Checkbutton(target_frame, text="Force Reveal Target", variable=self.v_reveal,
                        command=lambda: self.ctl.update(force_reveal=self.v_reveal.get())).pack(side="left", padx=15, pady=5)

        # Serial peripheral linked to the TARGET register
        ser_frame = ttk.LabelFrame(main_frame, text="Serial Peripheral  (linked to TARGET: locked target is sent automatically)")
        ser_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        row = ttk.Frame(ser_frame)
        row.pack(fill="x", padx=5, pady=5)
        ttk.Label(row, text="Port:").pack(side="left")
        self.port_combo = ttk.Combobox(row, values=SerialPeripheral.available_ports(), width=14)
        self.port_combo.set(SerialPeripheral.LOOPBACK)
        self.port_combo.pack(side="left", padx=4)
        ttk.Button(row, text="Rescan", command=lambda: self.port_combo.config(
            values=SerialPeripheral.available_ports())).pack(side="left")
        ttk.Label(row, text="Baud:").pack(side="left", padx=(10, 2))
        self.v_baud = tk.StringVar(value="115200")
        ttk.Entry(row, width=8, textvariable=self.v_baud).pack(side="left")
        self.btn_conn = ttk.Button(row, text="Connect", command=self.toggle_serial)
        self.btn_conn.pack(side="left", padx=10)
        self.lbl_serial = ttk.Label(row, text="Serial: disconnected", font=("Consolas", 9))
        self.lbl_serial.pack(side="left", padx=5)

        row2 = ttk.Frame(ser_frame)
        row2.pack(fill="x", padx=5)
        ttk.Label(row2, text="Send line:").pack(side="left")
        self.v_tx = tk.StringVar(value="diff 25")
        e = ttk.Entry(row2, width=30, textvariable=self.v_tx)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self.send_line())
        ttk.Button(row2, text="Send", command=self.send_line).pack(side="left")
        ttk.Label(row2, text="(commands: TERMINATOR | STOP | DIFF | DIFF 12 | HEADER | DATA hello | SHA256 hello | STATUS | RESET | HELP)", foreground="#708090").pack(side="left", padx=10)

        wrap = ttk.Frame(ser_frame)
        wrap.pack(fill="both", expand=True, padx=5, pady=5)
        self.txt_serial = tk.Text(wrap, height=7, width=100, font=("Consolas", 9), state="disabled", wrap="none")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.txt_serial.yview)
        self.txt_serial.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt_serial.pack(side="left", fill="both", expand=True)

        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        cam_frame = ttk.LabelFrame(display_frame, text="Optoelectronic Circuit Camera Viewport")
        cam_frame.pack(side="left", anchor="n", padx=5, pady=5)
        self.cam_canvas = tk.Canvas(cam_frame, width=320, height=240, bg="black")
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_img_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        right_panel = ttk.Frame(display_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        reg_frame = ttk.LabelFrame(right_panel, text="Bit-Pooling Hardware Registers")
        reg_frame.pack(fill="x", padx=5, pady=5)
        self.reg_canvas = tk.Canvas(reg_frame, width=700, height=330, bg="#121414", highlightthickness=0)
        self.reg_canvas.pack(padx=5, pady=5)
        self.lbl_status = ttk.Label(reg_frame, text="Status: IDLE", font=("Consolas", 11), foreground="#00e5ff")
        self.lbl_status.pack(anchor="w", padx=10, pady=(5, 0))
        self.lbl_decoded = ttk.Label(reg_frame, text="", font=("Consolas", 9), foreground="#d2e0f0")
        self.lbl_decoded.pack(anchor="w", padx=10, pady=(0, 5))

        set_frame = ttk.LabelFrame(right_panel, text="Pooling & Algorithm Settings")
        set_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(set_frame, text="Pooling Operator:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.op_combo = ttk.Combobox(set_frame, values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"], state="readonly")
        self.op_combo.set("POPCOUNT")
        self.op_combo.grid(row=0, column=1, padx=5, pady=5)
        self.op_combo.bind("<<ComboboxSelected>>", lambda ev: self.engine.set_pool_op(self.op_combo.get()))

        ttk.Label(set_frame, text="Kernel Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.kernel_spin = ttk.Spinbox(set_frame, from_=2, to=8, increment=2, width=5, command=self._update_kernel)
        self.kernel_spin.set(4)
        self.kernel_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)

    # ---- controls ------------------------------------------------------------
    def start_lab(self):
        self.ctl["stop"] = False
        if self.thread is None or not self.thread.is_alive():
            self.thread = SimThread(0, self.ctl, self.q, self.engine)
            self.thread.start()

    def stop_lab(self):
        self.ctl["stop"] = True

    def step_search(self):
        self.engine.step()

    def run_oracle(self):
        """Answers all 512 bits (8 per 40 ms, ~2.5 s) so registers and the serial stream animate with no camera."""
        gen = id(self.engine.oracle)
        if getattr(self, "_run_gen", None) == gen:
            return
        self._run_gen = gen

        def go():
            if id(self.engine.oracle) != gen or self.engine.converged:
                self._run_gen = None
                return
            for _ in range(8):
                self.engine.step()
            self.window.after(40, go)
        go()

    def reset_search(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")

    def set_custom_target(self):
        raw = self.v_custom_target.get().strip()
        try:
            try:
                val = int(raw, 0)
            except ValueError:
                val = padded_block_int(raw.encode("ascii"))           # plain text -> padded block
            if not 0 <= val <= self.engine.max_val:
                raise ValueError(f"Target must fit in {self.engine.bit_depth} bits.")
        except (ValueError, UnicodeEncodeError) as ex:
            messagebox.showerror("Invalid Target", str(ex))
            return
        self.engine.reset(val)
        messagebox.showinfo("Target Set", "New custom target loaded into the 512-bit register.")

    def randomize_target(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")

    def _update_kernel(self):
        try:
            self.engine.set_kernel(int(self.kernel_spin.get()))
        except ValueError:
            pass

    # ---- serial controls -------------------------------------------------------
    def toggle_serial(self):
        if self.serial.connected:
            self.serial.disconnect()
            self.btn_conn.config(text="Connect")
            return
        try:
            self.serial.connect(self.port_combo.get().strip(), int(self.v_baud.get()))
        except Exception as ex:
            messagebox.showerror("Serial Error", str(ex))
            return
        self.btn_conn.config(text="Disconnect")
        snap = self.engine.snapshot(False)
        if snap["converged"]:                                     # already locked: link catches up
            self.serial.send(f"EVT LOCK TARGET={snap['target']:0{BLOCK_HEX}X}")

    def send_line(self):
        line = self.v_tx.get().strip()
        if line and self.serial.connected:
            self.serial.send(line)

    # ---- drawing ---------------------------------------------------------------
    def _draw_bit_rows(self, target_val, guess_val, pooled_val, serial_val, revealed):
        """512-bit registers as 128 hex characters, wrapped at 64 per line (like the .ino)."""
        self.reg_canvas.delete("all")
        depth = self.engine.bit_depth
        width = depth // 4
        is_exposed = revealed or self.engine.converged

        rows = [
            ("TARGET", f"{target_val:0{width}X}" if is_exposed else "[HIDDEN]", "#00e676"),
            ("GUESS ", f"{guess_val:0{width}X}", "#ffb300"),
            ("POOLED", f"{pooled_val:0{width}X}", "#b388ff"),
            ("SERIAL", f"{serial_val:0{width}X}" if serial_val is not None else "[NO DATA]", "#40c4ff"),
        ]

        y = 16
        for label, text, colour in rows:
            self.reg_canvas.create_text(8, y, text=label, fill="#88a0c0", anchor="w",
                                        font=("Consolas", 10, "bold"))
            for i in range(0, len(text), 64):
                self.reg_canvas.create_text(78, y, text=text[i:i + 64], fill=colour, anchor="w",
                                            font=("Consolas", 10))
                y += 18
            y += 17

        self.reg_canvas.create_text(8, y, anchor="w", fill="#d2e0f0", font=("Consolas", 10),
                                    text=f"{depth}-bit registers | {self.engine.step_count}/{depth} reconstruction steps")

    # ---- UI loops --------------------------------------------------------------
    def _poll(self):
        """Camera thread -> preview image only."""
        latest = None
        try:
            for _ in range(100):
                ev = self.q.get_nowait()
                if ev[0] == "frame":
                    latest = ev[1]
                elif ev[0] == "error":
                    messagebox.showerror("Camera Error", ev[1])
        except queue.Empty:
            pass
        if latest is not None:
            self.photo = ppm_photo(latest["preview"])
            self.cam_canvas.itemconfig(self.cam_img_id, image=self.photo)
        self.window.after(30, self._poll)

    def _tick(self):
        """~20 Hz: serial commands -> oracle, oracle/serial state -> screen. No camera needed."""
        try:
            for _ in range(50):                                    # serial RX commands
                try:
                    cmd, arg = self.cmdq.get_nowait()
                except queue.Empty:
                    break
                self._handle_command(cmd, arg)

            lines = []                                             # serial log
            try:
                while True:
                    lines.append(self.serial.log.get_nowait())
            except queue.Empty:
                pass
            if lines:
                self.txt_serial.config(state="normal")
                for kind, text in lines:
                    self.txt_serial.insert("end", f"{kind:>3}  {text}\n")
                n = int(self.txt_serial.index("end-1c").split(".")[0])
                if n > 300:
                    self.txt_serial.delete("1.0", f"{n - 300}.0")
                self.txt_serial.see("end")
                self.txt_serial.config(state="disabled")

            d = self.engine.snapshot(self.v_reveal.get())
            sreg = self.serial.reg
            key = (d["target"], d["guess"], d["pooled"], d["converged"], self.v_reveal.get(), sreg)
            if key != self._reg_key:
                self._reg_key = key
                self._draw_bit_rows(d["target"], d["guess"], d["pooled"], sreg, self.v_reveal.get())

            if d["converged"]:
                dec = decode_block(d["target"])
                self.lbl_decoded.config(text=f"message {dec[0]!r}  sha256 {dec[1]}" if dec
                                        else "message: <not a padded block>")
            else:
                self.lbl_decoded.config(text="")

            link = f"{self.serial.name}" if self.serial.connected else "off"
            self.lbl_status.config(text=f"Status: {d['status']} | Bit Step: {d['steps']}/{self.engine.bit_depth}"
                                        f" | Serial: {link}")
            self.lbl_serial.config(text=f"Serial: {'connected ' + str(self.serial.name) if self.serial.connected else 'disconnected'}")
            if not self.serial.connected and self.btn_conn.cget("text") != "Connect":
                self.btn_conn.config(text="Connect")
        finally:
            self.window.after(50, self._tick)

    def _close(self):
        self.ctl["stop"] = True
        self.serial.disconnect()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("1000x980")
    root.mainloop()
