"""
Bit-Pooling Oracle Camera Lab + Arduino SHA-256 Proof-of-Work Lock Box

This program retains the original:
    - BitPoolingOracle
    - BitPoolingSearchEngine
    - Camera source
    - Camera bit-pooling simulation thread
    - Visual bit registers
    - Pooling operator and kernel controls

It adds an independent Arduino SHA-256 proof-of-work lock-box client.

Arduino proof-of-work protocol:
    Arduino -> PC: POW:<64-hex challenge>:<difficulty>\\n
    PC -> Arduino: SOLUTION:<16-hex nonce>:<64-hex SHA-256 hash>\\n
    Arduino -> PC: DISARMED\\n or REJECTED\\n

The PC searches for a nonce such that:
    SHA256(challenge_bytes || nonce_bytes)
has at least the requested number of leading zero bits.

Install:
    pip install pyserial opencv-python numpy
"""

import asyncio
import hashlib
import queue
import random
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


# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================

CHALLENGE_BYTES = 32
NONCE_BYTES = 8
HASH_BYTES = 32

CHALLENGE_HEX_CHARS = CHALLENGE_BYTES * 2
NONCE_HEX_CHARS = NONCE_BYTES * 2
HASH_HEX_CHARS = HASH_BYTES * 2


# ==============================================================================
# SAFE RESPONSE DELAY
# ==============================================================================

class SafeDelay:
    """
    Thread-safe, validated delay before the oracle or proof-of-work solver
    responds to an Arduino challenge.

    The Arduino issues a new challenge every five seconds, so the maximum
    response delay is kept below five seconds.
    """

    MIN_DELAY_S = 0.0
    MAX_DELAY_S = 4.0
    DEFAULT_DELAY_S = 0.10

    def __init__(self, delay_s: float = DEFAULT_DELAY_S):
        self._lock = threading.Lock()
        self._delay_s = self._clamp(delay_s)

    @staticmethod
    def _clamp(value) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = SafeDelay.DEFAULT_DELAY_S

        if value != value:  # Reject NaN.
            value = SafeDelay.DEFAULT_DELAY_S

        return min(
            max(value, SafeDelay.MIN_DELAY_S),
            SafeDelay.MAX_DELAY_S,
        )

    def get(self) -> float:
        with self._lock:
            return self._delay_s

    def set(self, value) -> float:
        with self._lock:
            self._delay_s = self._clamp(value)
            return self._delay_s


# ==============================================================================
# ORIGINAL BIT-POOLING ORACLE
# ==============================================================================

class BitPoolingOracle:
    """
    Encapsulates the secret target and uses bit-pooling across successive
    frames/queries to accumulate bits MSB-first, mimicking the optoelectronic
    circuit.
    """

    def __init__(self, bit_depth: int = 10, target=None):
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer.")

        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        if target is None:
            target = random.randint(0, self.max_val)

        self._target = int(
            np.clip(int(target), 0, self.max_val)
        )

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.pooled_bits = 0
        self.pooled_count = 0
        self.is_done = False

    def reveal(self) -> int:
        if not self.is_done:
            return 0

        return self._target

    def query_bit(self, trial_val: int) -> int:
        """
        Oracle evaluation: determines whether the secret target is greater
        than or equal to the pooled trial value.
        """
        if self.is_done:
            return 0

        return 1 if self._target >= trial_val else 0

    def pool_step(self, val: int) -> int:
        """Applies hardware-style bit pooling."""
        bits = [
            (val >> i) & 1
            for i in range(self.bit_depth)
        ]

        pooled_out = []

        for i in range(0, len(bits), self.stride):
            block = bits[i:i + self.kernel_size]

            if not block:
                break

            if self.pool_op == "POPCOUNT":
                result = sum(block) / len(block)
                output_bit = (
                    1
                    if result >= self.schmitt_trigger
                    else 0
                )
            elif self.pool_op == "AND":
                output_bit = (
                    1
                    if all(b == 1 for b in block)
                    else 0
                )
            elif self.pool_op == "OR":
                output_bit = (
                    1
                    if any(b == 1 for b in block)
                    else 0
                )
            elif self.pool_op == "XOR":
                output_bit = sum(block) % 2
            elif self.pool_op == "MAJORITY":
                output_bit = (
                    1
                    if sum(block) > (len(block) / 2.0)
                    else 0
                )
            else:
                output_bit = block[0]

            pooled_out.append(output_bit)

        output_value = 0

        for index, bit in enumerate(pooled_out):
            output_value |= bit << index

        return output_value

    @property
    def target(self) -> int:
        """The target is hidden until the full bit depth is reconstructed."""
        if not self.is_done:
            raise PermissionError(
                "SECURITY VIOLATION: Oracle target is classified "
                "until fully pooled."
            )

        return self._target


# ==============================================================================
# ORIGINAL BIT-POOLING SEARCH ENGINE
# ==============================================================================

class BitPoolingSearchEngine:
    def __init__(self, bit_depth: int = 10):
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError(
                "bit_depth must be a positive integer."
            )

        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

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

    def reset(self, new_target=None):
        self.oracle = BitPoolingOracle(
            self.bit_depth, new_target
        )
        self.current_guess = 0
        self.step_count = 0
        self.history = [0]
        self.converged = False
        self.last_status = "READY"
        self._update_registers()

    def step(self):
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
                self.last_status = (
                    f"LOCKED & REVEALED AT "
                    f"{self.current_guess}"
                )
            else:
                self.last_status = (
                    "RECONSTRUCTION MISMATCH"
                )

    def _update_registers(self):
        self.guess_reg = self.current_guess

        try:
            self.target_reg = self.oracle.target
        except PermissionError:
            self.target_reg = 0

        self.pooled_register = self.oracle.pool_step(
            self.current_guess
        )


# ==============================================================================
# SHA-256 PROOF-OF-WORK SOLVER
# ==============================================================================

def hash_meets_difficulty(
    hash_digest: bytes,
    difficulty_bits: int,
) -> bool:
    """
    Returns True when hash_digest has at least difficulty_bits leading zeroes.
    """
    full_zero_bytes = difficulty_bits // 8
    remaining_bits = difficulty_bits % 8

    if any(hash_digest[:full_zero_bytes]):
        return False

    if remaining_bits == 0:
        return True

    mask = (0xFF << (8 - remaining_bits)) & 0xFF

    return (
        hash_digest[full_zero_bytes] & mask
    ) == 0


def solve_proof_of_work(
    challenge_bytes: bytes,
    difficulty_bits: int,
    stop_event: threading.Event,
    progress_queue: queue.Queue = None,
):
    """
    Finds a nonce such that SHA256(challenge || nonce) meets the difficulty.

    The nonce is an 8-byte big-endian counter.
    """
    if not 0 <= difficulty_bits <= 256:
        raise ValueError(
            "Difficulty must be between 0 and 256."
        )

    nonce = 0
    start_time = time.perf_counter()

    while not stop_event.is_set():
        nonce_bytes = nonce.to_bytes(
            NONCE_BYTES, "big"
        )

        digest = hashlib.sha256(
            challenge_bytes + nonce_bytes
        ).digest()

        if hash_meets_difficulty(
            digest, difficulty_bits
        ):
            elapsed = (
                time.perf_counter() - start_time
            )

            return {
                "nonce": nonce,
                "nonce_hex": f"{nonce:016X}",
                "hash_hex": digest.hex().upper(),
                "attempts": nonce + 1,
                "elapsed_s": elapsed,
            }

        nonce += 1

        if (
            progress_queue is not None
            and nonce % 100000 == 0
        ):
            progress_queue.put((
                "pow_progress",
                {
                    "attempts": nonce,
                    "elapsed_s": (
                        time.perf_counter()
                        - start_time
                    ),
                },
            ))

    return None


# ==============================================================================
# ARDUINO LOCK BOX LINK
# ==============================================================================

class ArduinoLockBox:
    """
    Serial link to the Arduino SHA-256 proof-of-work lock box.

    Expected protocol:
        Arduino -> PC: POW:<64-hex challenge>:<difficulty>\\n
        PC -> Arduino: SOLUTION:<16-hex nonce>:<64-hex hash>\\n
        Arduino -> PC: DISARMED\\n or REJECTED\\n
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 1.0,
    ):
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. "
                "Run: pip install pyserial"
            )

        self.ser = serial.Serial(
            port, baudrate, timeout=timeout
        )
        self._lock = threading.Lock()

    @staticmethod
    def list_ports():
        if serial_tools is None:
            return []

        return [
            device.device
            for device in serial_tools.list_ports.comports()
        ]

    def close(self):
        with self._lock:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def read_challenge(self):
        """
        Reads one proof-of-work challenge.

        Returns:
            tuple: (challenge_bytes, difficulty_bits), or None.
        """
        with self._lock:
            while True:
                raw = self.ser.readline()

                if not raw:
                    return None

                line = raw.decode(
                    "ascii", errors="ignore"
                ).strip()

                if not line.startswith("POW:"):
                    continue

                try:
                    challenge_text, difficulty_text = line[
                        4:
                    ].split(":", 1)

                    if (
                        len(challenge_text)
                        != CHALLENGE_HEX_CHARS
                    ):
                        continue

                    challenge_bytes = bytes.fromhex(
                        challenge_text
                    )
                    difficulty = int(
                        difficulty_text
                    )

                    if not 0 <= difficulty <= 256:
                        continue

                    return (
                        challenge_bytes,
                        difficulty,
                    )

                except (ValueError, IndexError):
                    continue

    def send_solution(
        self, nonce: int, hash_digest: bytes
    ) -> str:
        """Sends the nonce and SHA-256 hash to the Arduino."""
        nonce_hex = f"{nonce:016X}"
        hash_hex = hash_digest.hex().upper()

        message = (
            f"SOLUTION:{nonce_hex}:{hash_hex}\n"
        )

        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write(
                message.encode("ascii")
            )
            self.ser.flush()

            response = self.ser.readline().decode(
                "ascii", errors="ignore"
            ).strip()

        return response.upper()


def handle_arduino_challenge(
    lock_box: ArduinoLockBox,
    challenge_bytes: bytes,
    difficulty_bits: int,
    stop_event: threading.Event,
    safe_delay: SafeDelay,
    progress_queue: queue.Queue,
):
    """
    Receives one Arduino proof-of-work challenge, solves it, and returns the
    nonce and hash for validation.
    """
    delay = safe_delay.get()

    if delay > 0:
        time.sleep(delay)

    result = solve_proof_of_work(
        challenge_bytes,
        difficulty_bits,
        stop_event,
        progress_queue,
    )

    if result is None:
        return {
            "challenge_hex": challenge_bytes.hex().upper(),
            "difficulty": difficulty_bits,
            "nonce": None,
            "nonce_hex": None,
            "hash_hex": None,
            "response": "STOPPED",
            "disarmed": False,
        }

    response = lock_box.send_solution(
        result["nonce"],
        bytes.fromhex(result["hash_hex"]),
    )

    return {
        "challenge_hex": challenge_bytes.hex().upper(),
        "difficulty": difficulty_bits,
        "nonce": result["nonce"],
        "nonce_hex": result["nonce_hex"],
        "hash_hex": result["hash_hex"],
        "attempts": result["attempts"],
        "elapsed_s": result["elapsed_s"],
        "response": response,
        "disarmed": response == "DISARMED",
    }


class ArduinoThread(threading.Thread):
    def __init__(
        self,
        port,
        ctl,
        out,
        safe_delay,
        difficulty_override=None,
    ):
        super().__init__(daemon=True)
        self.port = port
        self.ctl = ctl
        self.out = out
        self.safe_delay = safe_delay
        self.difficulty_override = difficulty_override
        self.stop_event = threading.Event()

    def run(self):
        lock_box = None

        try:
            lock_box = ArduinoLockBox(self.port)
            self.out.put((
                "arduino",
                "Connected to SHA-256 lock box.",
            ))

            while not self.ctl["stop"]:
                challenge = lock_box.read_challenge()

                if challenge is None:
                    continue

                (
                    challenge_bytes,
                    arduino_difficulty,
                ) = challenge

                difficulty = (
                    self.difficulty_override
                    if self.difficulty_override
                    is not None
                    else arduino_difficulty
                )

                result = handle_arduino_challenge(
                    lock_box,
                    challenge_bytes,
                    difficulty,
                    self.stop_event,
                    self.safe_delay,
                    self.out,
                )

                self.out.put(
                    ("pow_result", result)
                )

        except Exception as e:
            self.out.put((
                "arduino_error",
                f"{type(e).__name__}: {e}",
            ))

        finally:
            if lock_box is not None:
                lock_box.close()

            self.out.put((
                "arduino",
                "Lock-box connection closed.",
            ))


# ==============================================================================
# ORIGINAL CAMERA SOURCE
# ==============================================================================

class Source:
    def __init__(self, index: int):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not installed. "
                "Run: pip install opencv-python"
            )

        backend = (
            cv2.CAP_DSHOW
            if sys.platform == "win32"
            else cv2.CAP_ANY
        )

        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not (
            cap.isOpened() and cap.read()[0]
        ):
            cap.release()
            raise RuntimeError(
                f"Cannot open camera {index}."
            )

        self.cap = cap
        self.rgb = None

    def read(self):
        ok, bgr = self.cap.read()

        if not ok:
            return None

        self.rgb = cv2.cvtColor(
            bgr, cv2.COLOR_BGR2RGB
        )
        return cv2.cvtColor(
            bgr, cv2.COLOR_BGR2GRAY
        )

    def close(self):
        self.cap.release()


def ppm_photo(rgb: np.ndarray) -> tk.PhotoImage:
    h, w, _ = rgb.shape
    data = (
        b"P6 %d %d 255\n" % (w, h)
        + np.ascontiguousarray(
            rgb, np.uint8
        ).tobytes()
    )
    return tk.PhotoImage(
        data=data, format="PPM"
    )


# ==============================================================================
# ORIGINAL CAMERA BIT-POOLING THREAD
# ==============================================================================

class SimThread(threading.Thread):
    def __init__(
        self, cam_index, ctl, out, engine
    ):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.ctl = ctl
        self.out = out
        self.engine = engine

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out, engine = (
            self.ctl,
            self.out,
            self.engine,
        )

        try:
            src = Source(self.cam_index)
        except RuntimeError as e:
            out.put(("error", str(e)))
            out.put(("done", None))
            return

        out.put((
            "info",
            f"USB camera {self.cam_index} opened.",
        ))

        previous_time = time.perf_counter()
        frame_index = 0
        previous_even = False

        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(
                    src.read
                )

                if gray is None:
                    out.put((
                        "info",
                        "Camera stopped delivering frames.",
                    ))
                    break

                now = time.perf_counter()
                dt = min(
                    max(
                        now - previous_time,
                        1e-3,
                    ),
                    0.25,
                )
                previous_time = now

                small = cv2.resize(
                    gray,
                    (128, 96),
                    interpolation=cv2.INTER_AREA,
                )

                motion_detected = (
                    frame_index > 5
                    and np.std(small) > 5.0
                )

                motion_value = (
                    int(np.std(small) * 10)
                    if motion_detected
                    else 0
                )

                even_now = (
                    motion_value > 0
                    and motion_value % 2 == 0
                )

                even_state = (
                    even_now and previous_even
                )
                previous_even = even_now

                if (
                    even_state
                    and ctl.get("auto_step", True)
                    and not engine.converged
                ):
                    engine.step()

                rgb = src.rgb

                if rgb is None:
                    rgb = cv2.cvtColor(
                        gray,
                        cv2.COLOR_GRAY2RGB,
                    )

                if rgb.ndim == 2:
                    rgb = cv2.cvtColor(
                        rgb,
                        cv2.COLOR_GRAY2RGB,
                    )
                elif rgb.shape[2] == 4:
                    rgb = cv2.cvtColor(
                        rgb,
                        cv2.COLOR_RGBA2RGB,
                    )

                preview = cv2.resize(
                    rgb,
                    (320, 240),
                    interpolation=cv2.INTER_AREA,
                ).copy()

                if ctl.get("ir_heatmap", False):
                    preview[:, :, 0] = np.clip(
                        preview[:, :, 0].astype(int)
                        + 80,
                        0,
                        255,
                    )
                    preview[:, :, 2] = np.clip(
                        preview[:, :, 2].astype(int)
                        - 50,
                        0,
                        255,
                    )

                exposed_target = (
                    engine.oracle._target
                    if (
                        engine.converged
                        or ctl.get(
                            "force_reveal", False
                        )
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
                        "history": list(
                            engine.history
                        ),
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
            "Bit-Pooling Oracle Camera Lab + "
            "SHA-256 Proof-of-Work Lock Box"
        )

        self.q = queue.Queue()
        self.safe_delay = SafeDelay(
            SafeDelay.DEFAULT_DELAY_S
        )

        self.ctl = {
            "stop": False,
            "ir_heatmap": False,
            "auto_step": True,
            "force_reveal": False,
            "safe_delay": self.safe_delay,
        }

        # Original 10-bit camera bit-pooling oracle remains active.
        self.engine = BitPoolingSearchEngine(
            bit_depth=10
        )

        self.camera_thread = None
        self.arduino_thread = None

        self._build_ui()
        self.window.protocol(
            "WM_DELETE_WINDOW", self._close
        )
        self.window.after(15, self._poll)

    def _build_ui(self):
        main_frame = ttk.Frame(
            self.window, padding=10
        )
        main_frame.pack(
            fill="both", expand=True
        )

        # Original camera controls
        control_frame = ttk.LabelFrame(
            main_frame,
            text="Laboratory Controls",
        )
        control_frame.pack(
            fill="x", padx=5, pady=5
        )

        ttk.Button(
            control_frame,
            text="Start Camera",
            command=self.start_camera,
        ).pack(
            side="left", padx=5, pady=5
        )

        ttk.Button(
            control_frame,
            text="Stop Camera",
            command=self.stop_camera,
        ).pack(
            side="left", padx=5, pady=5
        )

        ttk.Button(
            control_frame,
            text="Step Bit Pool",
            command=self.step_search,
        ).pack(
            side="left", padx=5, pady=5
        )

        ttk.Button(
            control_frame,
            text="Reset Pool",
            command=self.reset_search,
        ).pack(
            side="left", padx=5, pady=5
        )

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="IR Thermal Heatmap",
            variable=self.v_ir,
            command=lambda: self.ctl.update(
                ir_heatmap=self.v_ir.get()
            ),
        ).pack(
            side="left", padx=10
        )

        # Original local target controls
        target_frame = ttk.LabelFrame(
            main_frame,
            text=(
                "Hidden Target Management "
                "(Auto-Revealing Oracle)"
            ),
        )
        target_frame.pack(
            fill="x", padx=5, pady=5
        )

        self.target_label = ttk.Label(
            target_frame,
            text=(
                f"Target (0-{self.engine.max_val}):"
            ),
        )
        self.target_label.pack(
            side="left", padx=5, pady=5
        )

        self.v_custom_target = tk.StringVar(
            value="[SECURE]"
        )
        self.e_custom_target = ttk.Entry(
            target_frame,
            width=10,
            textvariable=self.v_custom_target,
        )
        self.e_custom_target.pack(
            side="left", padx=5, pady=5
        )

        ttk.Button(
            target_frame,
            text="Set Custom",
            command=self.set_custom_target,
        ).pack(
            side="left", padx=2, pady=5
        )

        ttk.Button(
            target_frame,
            text="Randomize Target",
            command=self.randomize_target,
        ).pack(
            side="left", padx=5, pady=5
        )

        self.v_reveal = tk.BooleanVar(
            value=False
        )
        ttk.Checkbutton(
            target_frame,
            text="Force Reveal Target",
            variable=self.v_reveal,
            command=lambda: self.ctl.update(
                force_reveal=self.v_reveal.get()
            ),
        ).pack(
            side="left", padx=15, pady=5
        )

        # Arduino SHA-256 controls
        arduino_frame = ttk.LabelFrame(
            main_frame,
            text=(
                "Arduino SHA-256 Proof-of-Work "
                "Lock Box"
            ),
        )
        arduino_frame.pack(
            fill="x", padx=5, pady=5
        )

        ttk.Label(
            arduino_frame,
            text="Serial port:",
        ).pack(
            side="left", padx=5, pady=5
        )

        self.v_arduino_port = tk.StringVar(
            value="COM3"
        )
        self.e_arduino_port = ttk.Entry(
            arduino_frame,
            width=14,
            textvariable=self.v_arduino_port,
        )
        self.e_arduino_port.pack(
            side="left", padx=2, pady=5
        )

        ttk.Button(
            arduino_frame,
            text="List Ports",
            command=self.list_arduino_ports,
        ).pack(
            side="left", padx=2, pady=5
        )

        ttk.Button(
            arduino_frame,
            text="Connect Lock Box",
            command=self.connect_arduino,
        ).pack(
            side="left", padx=10, pady=5
        )

        ttk.Button(
            arduino_frame,
            text="Disconnect",
            command=self.disconnect_arduino,
        ).pack(
            side="left", padx=2, pady=5
        )

        ttk.Label(
            arduino_frame,
            text="Leading zero bits:",
        ).pack(
            side="left", padx=(15, 2)
        )

        self.v_difficulty = tk.StringVar(
            value="20"
        )
        self.e_difficulty = ttk.Entry(
            arduino_frame,
            width=8,
            textvariable=self.v_difficulty,
        )
        self.e_difficulty.pack(
            side="left", padx=2, pady=5
        )

        ttk.Label(
            arduino_frame,
            text="Response delay (s):",
        ).pack(
            side="left", padx=(15, 2)
        )

        self.v_delay = tk.StringVar(
            value=(
                f"{SafeDelay.DEFAULT_DELAY_S:.3f}"
            )
        )
        self.e_delay = ttk.Entry(
            arduino_frame,
            width=10,
            textvariable=self.v_delay,
        )
        self.e_delay.pack(
            side="left", padx=2, pady=5
        )

        ttk.Button(
            arduino_frame,
            text="Apply Settings",
            command=self.apply_settings,
        ).pack(
            side="left", padx=10, pady=5
        )

        # Camera viewport
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5,
        )

        camera_frame = ttk.LabelFrame(
            display_frame,
            text=(
                "Optoelectronic Circuit "
                "Camera Viewport"
            ),
        )
        camera_frame.pack(
            side="left", padx=5, pady=5
        )

        self.cam_canvas = tk.Canvas(
            camera_frame,
            width=320,
            height=240,
            bg="black",
        )
        self.cam_canvas.pack(
            padx=5, pady=5
        )
        self.cam_image_id = (
            self.cam_canvas.create_image(
                0, 0, anchor="nw"
            )
        )

        # Original bit-pooling register display
        right_panel = ttk.Frame(
            display_frame
        )
        right_panel.pack(
            side="right",
            fill="both",
            expand=True,
            padx=5,
            pady=5,
        )

        register_frame = ttk.LabelFrame(
            right_panel,
            text=(
                "Bit-Pooling Hardware "
                "Registers"
            ),
        )
        register_frame.pack(
            fill="x", padx=5, pady=5
        )

        self.register_canvas = tk.Canvas(
            register_frame,
            width=520,
            height=170,
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

        # Proof-of-work result panel
        pow_frame = ttk.LabelFrame(
            right_panel,
            text="SHA-256 Proof-of-Work Result",
        )
        pow_frame.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5,
        )

        self.pow_status_label = ttk.Label(
            pow_frame,
            text="Proof-of-Work: IDLE",
            font=("Consolas", 11, "bold"),
            foreground="#00e5ff",
            wraplength=700,
            justify="left",
        )
        self.pow_status_label.pack(
            anchor="w", padx=10, pady=5
        )

        self.pow_challenge_label = ttk.Label(
            pow_frame,
            text="Challenge: —",
            font=("Consolas", 9),
            foreground="#d2e0f0",
            wraplength=700,
            justify="left",
        )
        self.pow_challenge_label.pack(
            anchor="w", padx=10, pady=2
        )

        self.pow_nonce_label = ttk.Label(
            pow_frame,
            text="Nonce: —",
            font=("Consolas", 9),
            foreground="#ffb300",
            wraplength=700,
            justify="left",
        )
        self.pow_nonce_label.pack(
            anchor="w", padx=10, pady=2
        )

        self.pow_hash_label = ttk.Label(
            pow_frame,
            text="SHA-256 Hash: —",
            font=("Consolas", 9),
            foreground="#00e676",
            wraplength=700,
            justify="left",
        )
        self.pow_hash_label.pack(
            anchor="w", padx=10, pady=2
        )

        self.pow_stats_label = ttk.Label(
            pow_frame,
            text="Attempts: — | Time: —",
            font=("Consolas", 9),
            foreground="#b388ff",
        )
        self.pow_stats_label.pack(
            anchor="w", padx=10, pady=5
        )

        # Original pooling settings
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

    def apply_settings(self):
        try:
            difficulty = int(
                self.v_difficulty.get()
            )

            if not 0 <= difficulty <= 256:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid Difficulty",
                "Enter a difficulty between 0 and 256.",
            )
            return

        delay = self.safe_delay.set(
            self.v_delay.get()
        )
        self.v_delay.set(f"{delay:.3f}")

        messagebox.showinfo(
            "Settings Applied",
            f"Difficulty: {difficulty} leading zero bits\n"
            f"Response delay: {delay:.3f} seconds",
        )

    def list_arduino_ports(self):
        try:
            ports = (
                ArduinoLockBox.list_ports()
            )
        except Exception as e:
            messagebox.showerror(
                "Serial Error", str(e)
            )
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

        port = (
            self.v_arduino_port.get().strip()
        )

        if not port:
            messagebox.showerror(
                "Missing Port",
                "Enter a serial port such as COM3.",
            )
            return

        try:
            difficulty = int(
                self.v_difficulty.get()
            )

            if not 0 <= difficulty <= 256:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid Difficulty",
                "Enter a difficulty between 0 and 256.",
            )
            return

        self.ctl["stop"] = False
        self.arduino_thread = ArduinoThread(
            port,
            self.ctl,
            self.q,
            self.safe_delay,
            difficulty_override=difficulty,
        )
        self.arduino_thread.start()

    def disconnect_arduino(self):
        if self.arduino_thread is not None:
            self.ctl["stop"] = True
            self.arduino_thread.stop_event.set()

    def set_custom_target(self):
        try:
            value = int(
                self.v_custom_target.get()
            )
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter a valid integer.",
            )
            return

        if 0 <= value <= self.engine.max_val:
            self.engine.reset(value)
            messagebox.showinfo(
                "Target Set",
                "New custom target configured "
                "for bit pooling.",
            )
        else:
            messagebox.showerror(
                "Invalid Range",
                f"Target must be between 0 and "
                f"{self.engine.max_val}.",
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
            kernel = int(
                self.kernel_spin.get()
            )
            self.engine.kernel_size = kernel
            self.engine.oracle.kernel_size = (
                kernel
            )
        except ValueError:
            pass

    def _draw_bit_rows(
        self,
        target_value,
        guess_value,
        pooled_value,
        revealed,
    ):
        self.register_canvas.delete("all")

        bit_depth = self.engine.bit_depth
        is_exposed = (
            revealed or self.engine.converged
        )
        active_target = (
            target_value if is_exposed else 0
        )

        rows = [
            (
                "TARGET",
                active_target,
                "#00e676",
                not is_exposed,
            ),
            (
                "GUESS ",
                guess_value,
                "#ffb300",
                False,
            ),
            (
                "POOLED",
                pooled_value,
                "#b388ff",
                False,
            ),
        ]

        y_offset = 10

        for label, value, colour, hidden in rows:
            self.register_canvas.create_text(
                10,
                y_offset + 10,
                text=label,
                fill="#88a0c0",
                anchor="w",
                font=(
                    "Consolas", 10, "bold"
                ),
            )

            x = 90

            for bit_index in range(
                bit_depth - 1, -1, -1
            ):
                bit_state = (
                    value >> bit_index
                ) & 1

                self.register_canvas.create_rectangle(
                    x,
                    y_offset,
                    x + 18,
                    y_offset + 18,
                    fill=(
                        colour
                        if bit_state and not hidden
                        else "#1e2430"
                    ),
                    outline="#38445c",
                )

                text = (
                    "?" if hidden else str(bit_state)
                )
                text_colour = (
                    "#000000"
                    if bit_state and not hidden
                    else "#708090"
                )

                self.register_canvas.create_text(
                    x + 9,
                    y_offset + 9,
                    text=text,
                    fill=text_colour,
                    font=(
                        "Consolas", 8, "bold"
                    ),
                )

                x += 22

            number_text = (
                "[HIDDEN]"
                if hidden
                else f"[{value}]"
            )

            self.register_canvas.create_text(
                x + 5,
                y_offset + 9,
                text=number_text,
                fill="#d2e0f0",
                anchor="w",
                font=("Consolas", 9),
            )

            y_offset += 48

    def _poll(self):
        try:
            for _ in range(100):
                event = self.q.get_nowait()

                if event[0] == "frame":
                    data = event[1]
                    self.photo = ppm_photo(
                        data["preview"]
                    )
                    self.cam_canvas.itemconfig(
                        self.cam_image_id,
                        image=self.photo,
                    )

                    self._draw_bit_rows(
                        data["target"],
                        data["guess"],
                        data["pooled"],
                        self.v_reveal.get(),
                    )

                    delay = (
                        self.safe_delay.get()
                    )
                    self.status_label.config(
                        text=(
                            f"Status: {data['status']} | "
                            f"Bit Step: "
                            f"{data['steps']}/"
                            f"{self.engine.bit_depth} | "
                            f"Delay: {delay:.3f}s"
                        )
                    )

                elif event[0] == "arduino":
                    self.pow_status_label.config(
                        text=(
                            f"Proof-of-Work: {event[1]}"
                        )
                    )

                elif event[0] == "pow_progress":
                    data = event[1]

                    self.pow_stats_label.config(
                        text=(
                            f"Searching... Attempts: "
                            f"{data['attempts']:,} | "
                            f"Time: "
                            f"{data['elapsed_s']:.2f}s"
                        )
                    )

                elif event[0] == "pow_result":
                    result = event[1]

                    self.pow_challenge_label.config(
                        text=(
                            "Challenge: "
                            f"{result['challenge_hex']}"
                        )
                    )

                    if result["nonce"] is None:
                        self.pow_nonce_label.config(
                            text="Nonce: STOPPED"
                        )
                        self.pow_hash_label.config(
                            text="SHA-256 Hash: —"
                        )
                        self.pow_stats_label.config(
                            text="Search stopped."
                        )
                        self.pow_status_label.config(
                            text=(
                                "Proof-of-Work: "
                                "SEARCH STOPPED"
                            ),
                            foreground="#ffb300",
                        )
                    else:
                        self.pow_nonce_label.config(
                            text=(
                                "Nonce: "
                                f"{result['nonce_hex']} "
                                f"({result['nonce']})"
                            )
                        )

                        self.pow_hash_label.config(
                            text=(
                                "SHA-256 Hash: "
                                f"{result['hash_hex']}"
                            )
                        )

                        self.pow_stats_label.config(
                            text=(
                                f"Attempts: "
                                f"{result['attempts']:,} | "
                                f"Time: "
                                f"{result['elapsed_s']:.3f}s"
                            )
                        )

                        if result["disarmed"]:
                            self.pow_status_label.config(
                                text=(
                                    "Proof-of-Work: "
                                    "DISARMED | Valid proof"
                                ),
                                foreground="#00e676",
                            )
                        else:
                            self.pow_status_label.config(
                                text=(
                                    "Proof-of-Work: REJECTED | "
                                    f"{result['response']}"
                                ),
                                foreground="#ff5252",
                            )

                elif event[0] == "arduino_error":
                    messagebox.showerror(
                        "Arduino Error",
                        event[1],
                    )

                elif event[0] == "error":
                    messagebox.showerror(
                        "Camera Error", event[1]
                    )

        except queue.Empty:
            pass

        self.window.after(30, self._poll)

    def _close(self):
        self.ctl["stop"] = True

        if self.arduino_thread is not None:
            self.arduino_thread.stop_event.set()

        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("1240x780")
    root.mainloop()
