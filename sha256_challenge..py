"""
Bit-Pooling Oracle Lock Box Lab
Python + Arduino SHA-256 proof-of-work miner.

Protocol:
    PC -> Arduino:  DIFF <0-256>\\n                 (mining starts on receipt)
    Arduino -> PC:  POW:<nonce>:<64 hex hash>\\n    (the only unprompted output)

Pressing "Send Difficulty" sends DIFF to the Arduino, which starts mining
SHA256(DATA + nonce) straight away and replies with just the proof of work.

At the same moment, the camera bit-pooling oracle starts working in the
background on its own. It does not read anything from the Arduino: it
searches for the same proof of work itself, loads the resulting hash into
the bit-pooling register, and reconstructs it bit by bit. The oracle's
result and the Arduino's result are each stored with their elapsed time
and compared once both exist.

Install dependencies:
    pip install pyserial opencv-python numpy
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
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None
    serial_tools = None
else:
    serial_tools = serial.tools

try:
    import cv2
except ImportError:
    cv2 = None


PASSCODE_BITS = 256
PASSCODE_BYTES = PASSCODE_BITS // 8
PASSCODE_HEX_CHARS = PASSCODE_BYTES * 2


# ==============================================================================
# 256-BIT BIT-POOLING ORACLE
# ==============================================================================

class BitPoolingOracle:
    """
    Stores a hidden 256-bit target and answers higher-or-equal queries.

    The search engine reconstructs the target one bit at a time, MSB first.
    """

    def __init__(self, bit_depth: int = PASSCODE_BITS, target=None):
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer.")

        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        if target is None:
            target = secrets.randbits(bit_depth)

        self._target = int(target) & self.max_val

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.pooled_bits = 0
        self.pooled_count = 0
        self.is_done = False

    def query_bit(self, trial_val: int) -> int:
        """Returns 1 if the hidden target is greater than or equal to trial_val."""
        if self.is_done:
            return 0

        return 1 if self._target >= trial_val else 0

    def pool_step(self, val: int) -> int:
        """Applies the selected pooling operator to the current guess."""
        bits = [(val >> i) & 1 for i in range(self.bit_depth)]
        pooled_out = []

        for i in range(0, len(bits), self.stride):
            block = bits[i:i + self.kernel_size]

            if not block:
                break

            if self.pool_op == "POPCOUNT":
                result = sum(block) / len(block)
                output_bit = 1 if result >= self.schmitt_trigger else 0
            elif self.pool_op == "AND":
                output_bit = 1 if all(b == 1 for b in block) else 0
            elif self.pool_op == "OR":
                output_bit = 1 if any(b == 1 for b in block) else 0
            elif self.pool_op == "XOR":
                output_bit = sum(block) % 2
            elif self.pool_op == "MAJORITY":
                output_bit = 1 if sum(block) > (len(block) / 2.0) else 0
            else:
                output_bit = block[0]

            pooled_out.append(output_bit)

        output_value = 0
        for index, bit in enumerate(pooled_out):
            output_value |= bit << index

        return output_value

    @property
    def target(self) -> int:
        """The target remains hidden until the full bit depth is reconstructed."""
        if not self.is_done:
            raise PermissionError(
                "SECURITY VIOLATION: Oracle target is classified until fully pooled."
            )

        return self._target


# ==============================================================================
# 256-BIT BIT-POOLING SEARCH ENGINE
# ==============================================================================

class BitPoolingSearchEngine:
    def __init__(self, bit_depth: int = PASSCODE_BITS):
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer.")

        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        # The camera thread, the GUI and the background oracle all touch the
        # engine, so reset/step are serialised.
        self._lock = threading.RLock()

        self.oracle = BitPoolingOracle(self.bit_depth)
        self.current_guess = 0
        self.step_count = 0
        self.history = [0]
        self.converged = False
        self.last_status = "INITIALIZED"

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.guess_reg = 0
        self.target_reg = 0
        self.pooled_register = 0

        self._update_registers()

    def _reset(self, new_target=None):
        self.oracle = BitPoolingOracle(self.bit_depth, new_target)
        self.current_guess = 0
        self.step_count = 0
        self.history = [0]
        self.converged = False
        self.last_status = "READY"
        self._update_registers()

    def _step(self):
        """Advances the bit-pooling search by one bit."""
        if self.converged or self.step_count >= self.bit_depth:
            return

        self.step_count += 1

        shift = self.bit_depth - self.step_count
        trial_value = self.oracle.pooled_bits | (1 << shift)

        bit = self.oracle.query_bit(trial_value)

        if bit:
            self.oracle.pooled_bits |= 1 << shift

        self.current_guess = self.oracle.pooled_bits
        self.history.append(self.current_guess)
        self._update_registers()

        if self.step_count >= self.bit_depth:
            self.oracle.is_done = True
            self.converged = True

            if self.current_guess == self.oracle._target:
                self.last_status = "LOCKED & REVEALED"
            else:
                self.last_status = "RECONSTRUCTION MISMATCH"

    def reset(self, new_target=None):
        with self._lock:
            self._reset(new_target)

    def step(self):
        """Advances the bit-pooling search by one bit."""
        with self._lock:
            self._step()

    def _update_registers(self):
        self.guess_reg = self.current_guess

        try:
            self.target_reg = self.oracle.target
        except PermissionError:
            self.target_reg = 0

        self.pooled_register = self.oracle.pool_step(self.current_guess)


# ==============================================================================
# SHA-256 PROOF OF WORK
# ==============================================================================

POW_DATA = "arduino-sha256"  # must match inputData in sha256_challenge.ino
MIN_DIFF = 0
MAX_DIFF = 256


class MiningSession:
    """
    One difficulty send. Holds what the background oracle found and what the
    Arduino returned, each with its elapsed time since the difficulty was
    sent, and publishes a snapshot to the GUI whenever either one arrives.
    """

    def __init__(self, diff: int, data: str, out):
        self.diff = diff
        self.data = data
        self.out = out
        self.cancel = threading.Event()
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()
        self.predicted = None
        self.arduino = None

    def _record(self, nonce: int, hash_int: int) -> dict:
        return {
            "nonce": nonce,
            "hash": hash_int,
            "elapsed": time.perf_counter() - self.t0,
        }

    def set_predicted(self, nonce: int, hash_int: int):
        if self.cancel.is_set():
            return

        with self._lock:
            self.predicted = self._record(nonce, hash_int)

        self.out.put(("pow_update", self.snapshot()))

    def set_arduino(self, nonce: int, hash_int: int):
        if self.cancel.is_set():
            return

        with self._lock:
            self.arduino = self._record(nonce, hash_int)

        self.out.put(("pow_update", self.snapshot()))

    def snapshot(self) -> dict:
        with self._lock:
            predicted = dict(self.predicted) if self.predicted else None
            arduino = dict(self.arduino) if self.arduino else None

        match = None

        if predicted is not None and arduino is not None:
            match = (
                predicted["nonce"] == arduino["nonce"]
                and predicted["hash"] == arduino["hash"]
            )

        return {
            "diff": self.diff,
            "predicted": predicted,
            "arduino": arduino,
            "match": match,
        }


class BackgroundPredictor(threading.Thread):
    """
    The camera oracle's background worker. It runs independently of the
    Arduino link: it searches for the same proof of work (SHA256(DATA + nonce),
    nonce counting up from 0), loads the hash into the bit-pooling engine as
    the hidden target, and lets the engine reconstruct it one bit at a time.
    """

    def __init__(self, session: MiningSession, engine: BitPoolingSearchEngine):
        super().__init__(daemon=True)
        self.session = session
        self.engine = engine

    def run(self):
        session = self.session
        base = hashlib.sha256(session.data.encode("ascii"))
        limit = 1 << (256 - session.diff)  # leading-zero test: value < limit

        nonce = 0
        found = None

        while nonce <= 0xFFFFFFFF:
            if (nonce & 0xFFF) == 0 and session.cancel.is_set():
                return

            h = base.copy()
            h.update(str(nonce).encode("ascii"))
            value = int.from_bytes(h.digest(), "big")

            if value < limit:
                found = value
                break

            nonce += 1

        if found is None or session.cancel.is_set():
            return

        self.engine.reset(found)

        while not session.cancel.is_set() and not self.engine.converged:
            self.engine.step()

        if session.cancel.is_set():
            return

        session.set_predicted(nonce, self.engine.current_guess)


# ==============================================================================
# ARDUINO MINER LINK
# ==============================================================================

class ArduinoMiner:
    """
    Serial link to the Arduino SHA-256 miner.

        PC -> Arduino: DIFF <bits>\\n
        Arduino -> PC: POW:<nonce>:<64 hex characters>\\n

    Owned by a single thread, so no locking is needed.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.2,
    ):
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. Run: pip install pyserial"
            )

        self.ser = serial.Serial(port, baudrate, timeout=timeout)

        # Most boards reset when the port opens; give the sketch time to boot.
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    @staticmethod
    def list_ports():
        if serial_tools is None:
            return []

        return [
            device.device
            for device in serial_tools.list_ports.comports()
        ]

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send_diff(self, diff: int):
        self.ser.write(f"DIFF {int(diff)}\n".encode("ascii"))
        self.ser.flush()

    def read_pow(self):
        """
        Reads one line. Returns (nonce, hash_int) for a valid POW line,
        otherwise None.
        """
        raw = self.ser.readline()

        if not raw:
            return None

        line = raw.decode("ascii", errors="ignore").strip()

        if not line.startswith("POW:"):
            return None

        parts = line.split(":")

        if len(parts) != 3 or len(parts[2]) != 64:
            return None

        try:
            return int(parts[1]), int(parts[2], 16)
        except ValueError:
            return None


class ArduinoThread(threading.Thread):
    def __init__(self, port, ctl, out, engine):
        super().__init__(daemon=True)
        self.port = port
        self.ctl = ctl
        self.out = out
        self.engine = engine
        self.commands = queue.Queue()
        self.session = None

    def request_diff(self, diff: int):
        self.commands.put(diff)

    def _start_session(self, miner: ArduinoMiner, diff: int):
        if self.session is not None:
            self.session.cancel.set()

        session = MiningSession(diff, POW_DATA, self.out)
        self.session = session

        # Arduino and oracle are started together and then run on their own.
        miner.send_diff(diff)
        BackgroundPredictor(session, self.engine).start()

        self.out.put(("pow_update", session.snapshot()))

    def run(self):
        miner = None

        try:
            miner = ArduinoMiner(self.port)
            self.out.put((
                "arduino",
                "Connected. Set a difficulty and press Send Difficulty.",
            ))

            while not self.ctl["arduino_stop"]:
                try:
                    while True:
                        self._start_session(miner, self.commands.get_nowait())
                except queue.Empty:
                    pass

                result = miner.read_pow()

                if result is not None and self.session is not None:
                    self.session.set_arduino(*result)

        except Exception as e:
            self.out.put((
                "arduino_error",
                f"{type(e).__name__}: {e}",
            ))

        finally:
            if self.session is not None:
                self.session.cancel.set()

            if miner is not None:
                miner.close()

            self.out.put(("arduino", "Disconnected."))


# ==============================================================================
# CAMERA SOURCE
# ==============================================================================

class Source:
    def __init__(self, index: int):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not installed. Run: pip install opencv-python"
            )

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
    data = (
        b"P6 %d %d 255\n" % (w, h)
        + np.ascontiguousarray(rgb, np.uint8).tobytes()
    )
    return tk.PhotoImage(data=data, format="PPM")


# ==============================================================================
# CAMERA THREAD
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

        out.put(("info", f"USB camera {self.cam_index} opened."))
        out.put(("camera_started", None))

        previous_time = time.perf_counter()
        frame_index = 0
        previous_even = False

        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)

                if gray is None:
                    out.put((
                        "info",
                        "Camera stopped delivering frames.",
                    ))
                    break

                now = time.perf_counter()
                dt = min(max(now - previous_time, 1e-3), 0.25)
                previous_time = now

                small = cv2.resize(
                    gray, (128, 96), interpolation=cv2.INTER_AREA
                )
                motion_detected = (
                    frame_index > 5 and np.std(small) > 5.0
                )
                motion_value = (
                    int(np.std(small) * 10)
                    if motion_detected
                    else 0
                )

                even_now = motion_value > 0 and motion_value % 2 == 0
                even_state = even_now and previous_even
                previous_even = even_now

                if (
                    even_state
                    and ctl.get("auto_step", True)
                    and not engine.converged
                ):
                    engine.step()

                rgb = src.rgb

                if rgb is None:
                    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                if rgb.ndim == 2:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
                elif rgb.shape[2] == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

                preview = cv2.resize(
                    rgb,
                    (320, 240),
                    interpolation=cv2.INTER_AREA,
                ).copy()

                if ctl.get("ir_heatmap", False):
                    preview[:, :, 0] = np.clip(
                        preview[:, :, 0].astype(int) + 80,
                        0,
                        255,
                    )
                    preview[:, :, 2] = np.clip(
                        preview[:, :, 2].astype(int) - 50,
                        0,
                        255,
                    )

                exposed_target = (
                    engine.oracle._target
                    if (
                        engine.converged
                        or ctl.get("force_reveal", False)
                    )
                    else 0
                )

                out.put((
                    "frame",
                    {
                        "preview": preview,
                        "idx": frame_index,
                        "dt": dt,
                        "target": exposed_target,
                        "guess": engine.current_guess,
                        "pooled": engine.pooled_register,
                        "status": engine.last_status,
                        "steps": engine.step_count,
                        "history": list(engine.history),
                        "converged": engine.converged,
                    },
                ))

                frame_index += 1
                await asyncio.sleep(0.03)

        finally:
            src.close()
            out.put(("done", None))


# ==============================================================================
# MAIN GUI
# ==============================================================================

class App:
    def __init__(self, window):
        self.window = window
        self.window.title(
            "Bit-Pooling Oracle Lock Box Lab — 256-Bit Passcode"
        )

        self.q = queue.Queue()
        self.ctl = {
            "stop": False,
            "ir_heatmap": False,
            "auto_step": True,
            "force_reveal": False,
            "arduino_stop": False,
        }

        self.engine = BitPoolingSearchEngine(
            bit_depth=PASSCODE_BITS
        )
        self.camera_thread = None
        self.camera_ready = False
        self.pending_diff = None
        self.arduino_thread = None

        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.after(15, self._poll)

    def _build_ui(self):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Laboratory controls
        control_frame = ttk.LabelFrame(
            main_frame, text="Laboratory Controls"
        )
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Start Camera",
            command=self.start_camera,
        ).pack(side="left", padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Stop Camera",
            command=self.stop_camera,
        ).pack(side="left", padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Step Bit Pool",
            command=self.step_search,
        ).pack(side="left", padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Reset Pool",
            command=self.reset_search,
        ).pack(side="left", padx=5, pady=5)

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="IR Thermal Heatmap",
            variable=self.v_ir,
            command=lambda: self.ctl.update(
                ir_heatmap=self.v_ir.get()
            ),
        ).pack(side="left", padx=10)

        # Arduino controls
        arduino_frame = ttk.LabelFrame(
            main_frame,
            text="Arduino Miner — SHA-256 Proof of Work",
        )
        arduino_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(
            arduino_frame,
            text="Serial port:",
        ).pack(side="left", padx=5, pady=5)

        self.v_arduino_port = tk.StringVar(value="COM3")
        self.e_arduino_port = ttk.Entry(
            arduino_frame,
            width=14,
            textvariable=self.v_arduino_port,
        )
        self.e_arduino_port.pack(side="left", padx=2, pady=5)

        ttk.Button(
            arduino_frame,
            text="List Ports",
            command=self.list_arduino_ports,
        ).pack(side="left", padx=2, pady=5)

        ttk.Button(
            arduino_frame,
            text="Connect Arduino",
            command=self.connect_arduino,
        ).pack(side="left", padx=10, pady=5)

        ttk.Button(
            arduino_frame,
            text="Disconnect",
            command=self.disconnect_arduino,
        ).pack(side="left", padx=2, pady=5)

        ttk.Label(
            arduino_frame,
            text="Difficulty (bits):",
        ).pack(side="left", padx=(25, 5), pady=5)

        self.v_diff = tk.StringVar(value="12")
        ttk.Spinbox(
            arduino_frame,
            from_=MIN_DIFF,
            to=MAX_DIFF,
            width=5,
            textvariable=self.v_diff,
        ).pack(side="left", padx=2, pady=5)

        ttk.Button(
            arduino_frame,
            text="Send Difficulty",
            command=self.send_difficulty,
        ).pack(side="left", padx=5, pady=5)

        # Proof-of-work results
        pow_frame = ttk.LabelFrame(
            main_frame,
            text="Proof of Work",
        )
        pow_frame.pack(fill="x", padx=5, pady=5)

        self.v_link = tk.StringVar(value="Arduino: not connected")
        self.v_pow_diff = tk.StringVar(value="")
        self.v_pow_oracle = tk.StringVar(value="ORACLE   -")
        self.v_pow_arduino = tk.StringVar(value="ARDUINO  -")
        self.v_pow_match = tk.StringVar(value="")

        for var in (
            self.v_link,
            self.v_pow_diff,
            self.v_pow_oracle,
            self.v_pow_arduino,
            self.v_pow_match,
        ):
            ttk.Label(
                pow_frame,
                textvariable=var,
                font=("Consolas", 10),
            ).pack(anchor="w", padx=10, pady=1)

        # Local target controls
        target_frame = ttk.LabelFrame(
            main_frame,
            text="Local 256-Bit Oracle Target (hexadecimal)",
        )
        target_frame.pack(fill="x", padx=5, pady=5)

        self.target_label = ttk.Label(
            target_frame,
            text="Target (64 hex characters):",
        )
        self.target_label.pack(side="left", padx=5, pady=5)

        self.v_custom_target = tk.StringVar(
            value="[SECURE]"
        )
        self.e_custom_target = ttk.Entry(
            target_frame,
            width=72,
            textvariable=self.v_custom_target,
        )
        self.e_custom_target.pack(
            side="left", padx=5, pady=5
        )

        ttk.Button(
            target_frame,
            text="Set Custom",
            command=self.set_custom_target,
        ).pack(side="left", padx=2, pady=5)

        ttk.Button(
            target_frame,
            text="Randomize Target",
            command=self.randomize_target,
        ).pack(side="left", padx=5, pady=5)

        self.v_reveal = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            target_frame,
            text="Force Reveal Local Target",
            variable=self.v_reveal,
            command=lambda: self.ctl.update(
                force_reveal=self.v_reveal.get()
            ),
        ).pack(side="left", padx=15, pady=5)

        # Camera and register display
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(
            fill="both", expand=True, padx=5, pady=5
        )

        camera_frame = ttk.LabelFrame(
            display_frame,
            text="Optoelectronic Circuit Camera Viewport",
        )
        camera_frame.pack(side="left", padx=5, pady=5)

        self.cam_canvas = tk.Canvas(
            camera_frame,
            width=320,
            height=240,
            bg="black",
        )
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_image_id = self.cam_canvas.create_image(
            0, 0, anchor="nw"
        )

        right_panel = ttk.Frame(display_frame)
        right_panel.pack(
            side="right",
            fill="both",
            expand=True,
            padx=5,
            pady=5,
        )

        register_frame = ttk.LabelFrame(
            right_panel,
            text="256-Bit Passcode Registers",
        )
        register_frame.pack(
            fill="both", expand=True, padx=5, pady=5
        )

        self.register_canvas = tk.Canvas(
            register_frame,
            width=760,
            height=190,
            bg="#121414",
            highlightthickness=0,
        )
        self.register_canvas.pack(
            padx=5, pady=5
        )

        self.status_label = ttk.Label(
            register_frame,
            text="Status: IDLE",
            font=("Consolas", 11),
            foreground="#00e5ff",
        )
        self.status_label.pack(
            anchor="w", padx=10, pady=5
        )

        # Pooling settings
        settings_frame = ttk.LabelFrame(
            right_panel,
            text="Pooling & Algorithm Settings",
        )
        settings_frame.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5,
        )

        ttk.Label(
            settings_frame,
            text="Pooling Operator:",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=5,
            pady=5,
        )

        self.operator_combo = ttk.Combobox(
            settings_frame,
            values=[
                "POPCOUNT",
                "AND",
                "OR",
                "XOR",
                "MAJORITY",
            ],
            state="readonly",
        )
        self.operator_combo.set("POPCOUNT")
        self.operator_combo.grid(
            row=0, column=1, padx=5, pady=5
        )
        self.operator_combo.bind(
            "<<ComboboxSelected>>",
            self._update_pool_operator,
        )

        ttk.Label(
            settings_frame,
            text="Kernel Size:",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=5,
            pady=5,
        )

        self.kernel_spin = ttk.Spinbox(
            settings_frame,
            from_=2,
            to=8,
            increment=2,
            width=5,
            command=self._update_kernel,
        )
        self.kernel_spin.set(4)
        self.kernel_spin.grid(
            row=1, column=1, padx=5, pady=5
        )

    def start_camera(self):
        if (
            self.camera_thread is not None
            and self.camera_thread.is_alive()
        ):
            messagebox.showwarning(
                "Already Running",
                "The camera lab is already running.",
            )
            return

        self.ctl["stop"] = False
        self.camera_thread = SimThread(
            0, self.ctl, self.q, self.engine
        )
        self.camera_thread.start()

    def stop_camera(self):
        self.ctl["stop"] = True

    def step_search(self):
        self.engine.step()

    def reset_search(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")

    def list_arduino_ports(self):
        try:
            ports = ArduinoMiner.list_ports()
        except Exception as e:
            messagebox.showerror("Serial Error", str(e))
            return

        if not ports:
            messagebox.showwarning(
                "No Serial Ports",
                "No serial ports were found.",
            )
            return

        self.v_arduino_port.set(ports[0])
        messagebox.showinfo(
            "Serial Ports", "\n".join(ports)
        )

    def connect_arduino(self):
        if (
            self.arduino_thread is not None
            and self.arduino_thread.is_alive()
        ):
            messagebox.showwarning(
                "Already Connected",
                "The lock box is already connected.",
            )
            return

        port = self.v_arduino_port.get().strip()

        if not port:
            messagebox.showerror(
                "Missing Port",
                "Enter a serial port such as COM3.",
            )
            return

        self.ctl["arduino_stop"] = False
        self.arduino_thread = ArduinoThread(
            port,
            self.ctl,
            self.q,
            self.engine,
        )
        self.arduino_thread.start()

        if self.camera_ready and self.pending_diff is not None:
            self.arduino_thread.request_diff(self.pending_diff)
            self.pending_diff = None

    def disconnect_arduino(self):
        if self.arduino_thread is not None:
            self.ctl["arduino_stop"] = True

    def _read_diff(self):
        try:
            diff = int(self.v_diff.get())
        except ValueError:
            return None

        return diff if MIN_DIFF <= diff <= MAX_DIFF else None

    def _arduino_alive(self):
        return (
            self.arduino_thread is not None
            and self.arduino_thread.is_alive()
        )

    def send_difficulty(self):
        if not self._arduino_alive():
            messagebox.showwarning(
                "Not Connected",
                "Connect the Arduino first.",
            )
            return

        diff = self._read_diff()

        if diff is None:
            messagebox.showerror(
                "Invalid Difficulty",
                f"Enter a whole number from {MIN_DIFF} to {MAX_DIFF}.",
            )
            return

        if not self.camera_ready:
            # Mining begins once the webcam is running.
            self.pending_diff = diff
            self.v_link.set(
                f"Arduino: difficulty {diff} queued, "
                "mining starts when the webcam starts."
            )
            return

        self.arduino_thread.request_diff(diff)

    def _on_camera_started(self):
        self.camera_ready = True

        diff = self.pending_diff

        if diff is None:
            diff = self._read_diff()

        if diff is None:
            return

        if self._arduino_alive():
            self.pending_diff = None
            self.arduino_thread.request_diff(diff)
        else:
            # Connect Arduino later; mining starts as soon as it is up.
            self.pending_diff = diff
            self.v_link.set(
                f"Webcam running. Difficulty {diff} will start mining "
                "when the Arduino connects."
            )

    def _show_pow(self, snap):
        def line(label, item):
            if item is None:
                return f"{label}  ..."

            return (
                f"{label}  nonce {item['nonce']}  "
                f"hash {item['hash']:064x}  "
                f"({item['elapsed']:.3f} s)"
            )

        self.v_pow_diff.set(f"Difficulty: {snap['diff']} bits")
        self.v_pow_oracle.set(line("ORACLE  ", snap["predicted"]))
        self.v_pow_arduino.set(line("ARDUINO ", snap["arduino"]))

        if snap["match"] is None:
            self.v_pow_match.set("")
        elif snap["match"]:
            self.v_pow_match.set("MATCH: same nonce and hash")
        else:
            self.v_pow_match.set("MISMATCH: nonce or hash differs")

    def set_custom_target(self):
        text = self.v_custom_target.get().strip()

        if text.startswith("0x") or text.startswith("0X"):
            text = text[2:]

        try:
            value = int(text, 16)
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Enter a 64-character hexadecimal value.",
            )
            return

        if 0 <= value <= self.engine.max_val:
            self.engine.reset(value)
            messagebox.showinfo(
                "Target Set",
                "New 256-bit target configured for bit pooling.",
            )
        else:
            messagebox.showerror(
                "Invalid Range",
                "Target must be a 256-bit hexadecimal value.",
            )

    def randomize_target(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")

    def _update_pool_operator(self, event):
        operator = self.operator_combo.get()
        self.engine.pool_op = operator
        self.engine.oracle.pool_op = operator

    def _update_kernel(self):
        try:
            kernel = int(self.kernel_spin.get())
            self.engine.kernel_size = kernel
            self.engine.oracle.kernel_size = kernel
        except ValueError:
            pass

    def _draw_hex_rows(
        self,
        target_value,
        guess_value,
        pooled_value,
        revealed,
    ):
        self.register_canvas.delete("all")

        is_exposed = revealed or self.engine.converged
        active_target = target_value if is_exposed else 0

        rows = [
            (
                "TARGET",
                active_target,
                "#00e676",
                not is_exposed,
            ),
            ("GUESS ", guess_value, "#ffb300", False),
            (
                "POOLED",
                pooled_value,
                "#b388ff",
                False,
            ),
        ]

        y_offset = 20

        for label, value, colour, hidden in rows:
            self.register_canvas.create_text(
                10,
                y_offset,
                text=label,
                fill="#88a0c0",
                anchor="w",
                font=("Consolas", 10, "bold"),
            )

            hex_text = (
                "[HIDDEN]"
                if hidden
                else f"{value:064X}"
            )

            self.register_canvas.create_text(
                90,
                y_offset,
                text=hex_text,
                fill=colour,
                anchor="w",
                font=("Consolas", 10),
            )

            y_offset += 35

        self.register_canvas.create_text(
            10,
            y_offset + 10,
            text=(
                f"256-bit passcode | "
                f"{self.engine.step_count}/256 reconstruction steps"
            ),
            fill="#d2e0f0",
            anchor="w",
            font=("Consolas", 10),
        )

    def _poll(self):
        try:
            for _ in range(100):
                event = self.q.get_nowait()

                if event[0] == "frame":
                    data = event[1]
                    self.photo = ppm_photo(data["preview"])
                    self.cam_canvas.itemconfig(
                        self.cam_image_id,
                        image=self.photo,
                    )

                    self._draw_hex_rows(
                        data["target"],
                        data["guess"],
                        data["pooled"],
                        self.v_reveal.get(),
                    )

                    self.status_label.config(
                        text=(
                            f"Status: {data['status']} | "
                            f"Bit Step: {data['steps']}/256"
                        )
                    )

                elif event[0] == "arduino":
                    self.v_link.set(f"Arduino: {event[1]}")

                elif event[0] == "pow_update":
                    self._show_pow(event[1])

                elif event[0] == "arduino_error":
                    messagebox.showerror(
                        "Arduino Error", event[1]
                    )

                elif event[0] == "camera_started":
                    self._on_camera_started()

                elif event[0] == "done":
                    self.camera_ready = False

                elif event[0] == "error":
                    messagebox.showerror(
                        "Camera Error", event[1]
                    )

        except queue.Empty:
            pass

        self.window.after(30, self._poll)

    def _close(self):
        self.ctl["stop"] = True
        self.ctl["arduino_stop"] = True
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("1240x780")
    root.mainloop()
