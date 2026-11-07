"""
Bit-Pooling Oracle Lock Box Lab
Python + Arduino SHA-256 proof-of-work miner.

Protocol:
    PC -> Arduino:  DIFF <0-256>\\n                 (mining starts on receipt)
    Arduino -> PC:  POW:<nonce>:<64 hex hash>:<padded message hex>:<padded nonce hex>:<padded hash hex>:<header>\\n
                    (the only unprompted output)

Pressing "Send Difficulty" sends DIFF to the Arduino, which starts mining
SHA256(DATA + nonce) straight away and replies with just the proof of work.

The Arduino is the only thing that mines. The Python side never searches for
a proof of work; when the Arduino reports one, Python shows the header and
nonce and recomputes that single hash to verify it. The oracle's own guess
is verified the same way as soon as it is made, using the header the
Arduino reports for its DATA string (one hash of the guessed nonce).

At the same moment the difficulty is sent, the oracle starts a background
guess of both the nonce and the hash on its own. It reads nothing from the
Arduino and does no searching, so all it can use is what the protocol states:
a valid hash begins with DIFF zero bits, and the first valid nonce is on
the order of 2^DIFF. When the Arduino's proof of work arrives, the two are
stored with their elapsed times and compared (nonce, exact hash, and how many
of the 256 hash bits agree).

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


def sha256_pad(message: bytes) -> bytes:
    """
    SHA-256 padding: message, 0x80, zero bytes, then the 64-bit big-endian
    bit length, making a whole number of 64-byte blocks. For the Arduino's
    proof of work the message is header + decimal nonce, so the nonce is
    carried inside the padded message.
    """
    zeros = (55 - len(message)) % 64
    return (
        message
        + b"\x80"
        + b"\x00" * zeros
        + (len(message) * 8).to_bytes(8, "big")
    )


def format_blocks(data: bytes) -> str:
    """Hex of a padded message, one 64-byte block per space-separated group."""
    return " ".join(
        data[i:i + 64].hex() for i in range(0, len(data), 64)
    )


def padded_agreement(a: bytes, b: bytes):
    """(agreeing bits, total bits, identical) for two padded messages."""
    n = min(len(a), len(b))
    diff = int.from_bytes(a[:n], "big") ^ int.from_bytes(b[:n], "big")
    total = max(len(a), len(b)) * 8
    return n * 8 - bin(diff).count("1"), total, a == b


def pow_blocks(header: bytes, nonce: int):
    """
    The three padded fields of a proof of work, each the SHA-256 padding of
    its own bytes: (header + nonce, nonce text alone, the 32 hash bytes).
    """
    nonce_text = str(nonce).encode("ascii")
    message = header + nonce_text
    digest = hashlib.sha256(message).digest()

    return (
        sha256_pad(message),
        sha256_pad(nonce_text),
        sha256_pad(digest),
    )


def pow_whole(header: bytes, nonce: int) -> bytes:
    """The whole padded record: message blocks, nonce block, hash block."""
    return b"".join(pow_blocks(header, nonce))


# ==============================================================================
# 256-BIT BIT-POOLING ORACLE
# ==============================================================================

class BitPoolingOracle:
    """
    SHA-256 padded-record oracle. The hidden passcode is the whole padded
    proof-of-work record: the padded message (header + nonce), the padded
    nonce and the padded hash, each in SHA-256 padding (bytes, 0x80, zeros,
    64-bit length), concatenated in that order. With the Arduino connected,
    the header and nonce come from its proof of work, so the passcode is the
    padded fields the Arduino reports. The bit depth is 512 per block, so it
    follows the header length.

    The oracle answers higher-or-equal queries; the search engine
    reconstructs the whole record one bit at a time, MSB first.
    """

    def __init__(self, header=None, nonce=None):
        if header is None:
            header = secrets.token_bytes(16)

        if isinstance(header, str):
            header = header.encode("utf-8")

        if nonce is None:
            nonce = secrets.randbelow(1 << 32)

        # Only header and nonce are stored; the padded record (and the hash
        # inside it) is recomputed per query.
        self.header = bytes(header)
        self.nonce = int(nonce)

        self.bit_depth = len(pow_whole(self.header, self.nonce)) * 8
        self.max_val = (1 << self.bit_depth) - 1

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.pooled_bits = 0
        self.pooled_count = 0
        self.is_done = False

    def _padded_value(self) -> int:
        return int.from_bytes(pow_whole(self.header, self.nonce), "big")

    @property
    def _target(self) -> int:
        return self._padded_value()

    def query_bit(self, trial_val: int) -> int:
        """Returns 1 if the padded record is greater than or equal to trial_val."""
        if self.is_done:
            return 0

        return 1 if self._padded_value() >= trial_val else 0

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
    def __init__(self):
        # The camera thread, the GUI and the background oracle all touch the
        # engine, so reset/step are serialised.
        self._lock = threading.RLock()

        self.oracle = BitPoolingOracle()
        self.bit_depth = self.oracle.bit_depth
        self.max_val = self.oracle.max_val

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

    def _reset(self, header=None, nonce=None):
        self.oracle = BitPoolingOracle(header, nonce)
        self.bit_depth = self.oracle.bit_depth
        self.max_val = self.oracle.max_val

        # Keep the pooling settings chosen in the GUI across re-arming.
        self.oracle.pool_op = self.pool_op
        self.oracle.kernel_size = self.kernel_size
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

    def reset(self, header=None, nonce=None):
        """Re-arms the oracle with the padded record of header + nonce."""
        with self._lock:
            self._reset(header, nonce)

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
# PROOF-OF-WORK SESSION
# ==============================================================================

MIN_DIFF = 0
MAX_DIFF = 256


def verify_pow(
    header: bytes,
    nonce: int,
    claimed_hash: int,
    diff: int,
    padded=None,
    padded_nonce=None,
    padded_hash=None,
) -> dict:
    """
    Verifies a reported proof of work with a single SHA-256 computation of
    header + decimal nonce. Any padded field the Arduino sent (message,
    nonce, hash) must equal the SHA-256 padding of the matching bytes. This
    checks a result; it does not search for one.
    """
    message = header + str(nonce).encode("ascii")
    computed = int.from_bytes(hashlib.sha256(message).digest(), "big")
    zero_bits = PASSCODE_BITS - computed.bit_length()
    expected = pow_blocks(header, nonce)
    hash_ok = computed == claimed_hash
    diff_ok = zero_bits >= diff
    padded_ok = all(
        got is None or got == want
        for got, want in zip((padded, padded_nonce, padded_hash), expected)
    )

    return {
        "message": message,
        "padded": expected[0],
        "whole": b"".join(expected),
        "computed": computed,
        "zero_bits": zero_bits,
        "hash_ok": hash_ok,
        "diff_ok": diff_ok,
        "padded_ok": padded_ok,
        "ok": hash_ok and diff_ok and padded_ok,
    }


class MiningSession:
    """
    One difficulty send. Holds the oracle's background guess and the
    Arduino's proof of work, each with its elapsed time since the difficulty
    was sent, and publishes a snapshot to the GUI whenever either arrives.
    """

    def __init__(self, diff: int, out, header=None):
        self.diff = diff
        self.out = out
        self.header = header
        self.cancel = threading.Event()
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()
        self.predicted = None
        self.arduino = None

    def _record(self, hash_int: int, nonce=None) -> dict:
        return {
            "nonce": nonce,
            "hash": hash_int,
            "elapsed": time.perf_counter() - self.t0,
        }

    def set_predicted(self, hash_int: int, nonce: int):
        if self.cancel.is_set():
            return

        record = self._record(hash_int, nonce)
        record["verify"] = self._verify_guess(record, self.header)

        with self._lock:
            self.predicted = record

        self.out.put(("pow_update", self.snapshot()))

    def _verify_guess(self, record: dict, header):
        """One SHA-256 of header + guessed nonce, or None without a header."""
        if header is None:
            return None

        return verify_pow(header, record["nonce"], record["hash"], self.diff)

    def set_arduino(
        self,
        nonce: int,
        hash_int: int,
        padded: bytes = b"",
        padded_nonce: bytes = b"",
        padded_hash: bytes = b"",
        header: bytes = b"",
    ):
        if self.cancel.is_set():
            return

        record = self._record(hash_int, nonce)
        record["header"] = header
        record["padded"] = padded
        record["padded_nonce"] = padded_nonce
        record["padded_hash"] = padded_hash
        record["whole"] = padded + padded_nonce + padded_hash
        record["verify"] = verify_pow(
            header, nonce, hash_int, self.diff,
            padded, padded_nonce, padded_hash,
        )

        with self._lock:
            self.arduino = record

            # No header was available when the oracle guessed: verify the
            # guess now, against the header the Arduino reports.
            if self.predicted is not None and self.predicted["verify"] is None:
                self.predicted["verify"] = self._verify_guess(
                    self.predicted, header
                )

        self.out.put(("pow_update", self.snapshot()))

    def snapshot(self) -> dict:
        with self._lock:
            predicted = dict(self.predicted) if self.predicted else None
            arduino = dict(self.arduino) if self.arduino else None

        match = None
        nonce_match = None
        bits_agree = None

        if predicted is not None and arduino is not None:
            difference = predicted["hash"] ^ arduino["hash"]
            nonce_match = predicted["nonce"] == arduino["nonce"]
            match = nonce_match and difference == 0
            bits_agree = PASSCODE_BITS - bin(difference).count("1")

        # The oracle's predicted padded record: the padding of header plus
        # its guessed nonce, of the guessed nonce alone, and of its guessed
        # hash. No hashing and no Arduino data beyond the header, which the
        # protocol exposes.
        predicted_padded = None
        padded_agree = None
        header = self.header

        if header is None and arduino is not None:
            header = arduino["header"]

        if predicted is not None and header is not None:
            guess_nonce = str(predicted["nonce"]).encode("ascii")
            predicted_padded = (
                sha256_pad(header + guess_nonce)
                + sha256_pad(guess_nonce)
                + sha256_pad(predicted["hash"].to_bytes(32, "big"))
            )

        if predicted_padded is not None and arduino is not None:
            padded_agree = padded_agreement(
                predicted_padded, arduino["whole"]
            )

        return {
            "diff": self.diff,
            "predicted": predicted,
            "arduino": arduino,
            "match": match,
            "nonce_match": nonce_match,
            "bits_agree": bits_agree,
            "predicted_padded": predicted_padded,
            "padded_agree": padded_agree,
        }


def blind_guess(diff: int):
    """
    Returns (nonce, hash) guessed without hashing anything and without
    reading the Arduino.

    The hash has its DIFF leading bits set to zero (the protocol rule) and
    random bits elsewhere. The first valid nonce is geometrically distributed
    with success chance 2^-DIFF per try, so no nonce is much likelier than
    nonce 0, and the best any blind guess can do is about 1 in 2^DIFF. The
    nonce is drawn from the small range [0, 2^(DIFF-4)), which reaches
    about 97% of that best case. Chance of an exact hash hit is
    1 in 2^(256-DIFF).
    """
    free_bits = PASSCODE_BITS - diff
    hash_guess = secrets.randbits(free_bits) if free_bits > 0 else 0
    nonce_guess = secrets.randbelow(1 << min(max(diff - 4, 0), 32))
    return nonce_guess, hash_guess


class BackgroundPredictor(threading.Thread):
    """
    The oracle's background worker. It runs independently of the Arduino and
    does not mine: the Arduino is the only thing that hashes.
    """

    def __init__(self, session: MiningSession):
        super().__init__(daemon=True)
        self.session = session

    def run(self):
        session = self.session

        if session.cancel.is_set():
            return

        nonce, hash_guess = blind_guess(session.diff)
        session.set_predicted(hash_guess, nonce)


# ==============================================================================
# ARDUINO MINER LINK
# ==============================================================================

class ArduinoMiner:
    """
    Serial link to the Arduino SHA-256 miner.

        PC -> Arduino: DIFF <bits>\\n
        Arduino -> PC: POW:<nonce>:<64 hex>:<padded message hex>:<padded nonce hex>:<padded hash hex>:<header>\\n

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

    def query_header(self, timeout: float = 1.0):
        """Asks the Arduino for its header (DATA string). None if no reply."""
        self.ser.reset_input_buffer()
        self.ser.write(b"HEADER\n")
        self.ser.flush()

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            line = self.ser.readline().rstrip(b"\r\n")

            if line.startswith(b"HEADER:"):
                return line[len(b"HEADER:"):]

        return None

    def send_diff(self, diff: int):
        self.ser.write(f"DIFF {int(diff)}\n".encode("ascii"))
        self.ser.flush()

    def read_pow(self):
        """
        Reads one line. Returns (nonce, hash_int, padded_message,
        padded_nonce, padded_hash, header) for a valid POW line, otherwise
        None. The header is everything after the sixth colon, so it may
        itself contain colons.
        """
        raw = self.ser.readline()

        if not raw:
            return None

        line = raw.rstrip(b"\r\n")

        if not line.startswith(b"POW:"):
            return None

        parts = line.split(b":", 6)

        if len(parts) != 7 or len(parts[2]) != 64:
            return None

        try:
            return (
                int(parts[1]),
                int(parts[2], 16),
                bytes.fromhex(parts[3].decode("ascii")),
                bytes.fromhex(parts[4].decode("ascii")),
                bytes.fromhex(parts[5].decode("ascii")),
                parts[6],
            )
        except ValueError:
            return None


class ArduinoThread(threading.Thread):
    def __init__(self, port, ctl, out):
        super().__init__(daemon=True)
        self.port = port
        self.ctl = ctl
        self.out = out
        self.commands = queue.Queue()
        self.session = None

    def request_diff(self, diff: int):
        self.commands.put(diff)

    def _start_session(self, miner: ArduinoMiner, diff: int):
        if self.session is not None:
            self.session.cancel.set()

        header = miner.query_header()
        session = MiningSession(diff, self.out, header)
        self.session = session

        # Arduino and oracle are started together and then run on their own.
        miner.send_diff(diff)
        BackgroundPredictor(session).start()

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
            "Bit-Pooling Oracle Lock Box Lab — SHA-256 Padded Message"
        )

        self.q = queue.Queue()
        self.ctl = {
            "stop": False,
            "ir_heatmap": False,
            "auto_step": True,
            "force_reveal": False,
            "arduino_stop": False,
        }

        self.engine = BitPoolingSearchEngine()
        self.camera_thread = None
        self.camera_ready = False
        self.pending_diff = None
        self.arduino_thread = None
        self._oracle_key = None

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
        self.v_pow_header = tk.StringVar(value="HEADER   -")
        self.v_pow_nonce = tk.StringVar(value="NONCE    -")
        self.v_pow_hash = tk.StringVar(value="HASH     -")
        self.v_pow_padded = tk.StringVar(value="PAD MSG  -")
        self.v_pow_padded_nonce = tk.StringVar(value="PAD NONCE -")
        self.v_pow_padded_hash = tk.StringVar(value="PAD HASH -")
        self.v_pow_verify = tk.StringVar(value="VERIFY   -")
        self.v_pow_oracle = tk.StringVar(value="ORACLE   -")
        self.v_pow_oracle_padded = tk.StringVar(value="ORACLE PADDED   -")
        self.v_pow_oracle_verify = tk.StringVar(value="ORACLE VERIFY   -")
        self.v_pow_match = tk.StringVar(value="")

        for var in (
            self.v_link,
            self.v_pow_diff,
            self.v_pow_header,
            self.v_pow_nonce,
            self.v_pow_hash,
            self.v_pow_padded,
            self.v_pow_padded_nonce,
            self.v_pow_padded_hash,
            self.v_pow_verify,
            self.v_pow_oracle,
            self.v_pow_oracle_padded,
            self.v_pow_oracle_verify,
            self.v_pow_match,
        ):
            ttk.Label(
                pow_frame,
                textvariable=var,
                wraplength=1180,
                font=("Consolas", 10),
            ).pack(anchor="w", padx=10, pady=1)

        # Local target controls
        target_frame = ttk.LabelFrame(
            main_frame,
            text="SHA-256 Oracle (passcode = padded message | nonce | hash)",
        )
        target_frame.pack(fill="x", padx=5, pady=5)

        self.target_label = ttk.Label(
            target_frame,
            text="Oracle header (nonce 0):",
        )
        self.target_label.pack(side="left", padx=5, pady=5)

        self.v_custom_target = tk.StringVar(
            value="[SECURE]"
        )
        self.e_custom_target = ttk.Entry(
            target_frame,
            width=60,
            textvariable=self.v_custom_target,
        )
        self.e_custom_target.pack(
            side="left", padx=5, pady=5
        )

        ttk.Button(
            target_frame,
            text="Hash Message",
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
            text="Force Reveal Passcode",
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
            text="Padded-Message Passcode Registers",
        )
        register_frame.pack(
            fill="both", expand=True, padx=5, pady=5
        )

        self.register_canvas = tk.Canvas(
            register_frame,
            width=760,
            height=330,
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
        predicted, arduino = snap["predicted"], snap["arduino"]

        # A verified Arduino proof of work arms the SHA-256 oracle: the
        # passcode becomes SHA256(header + nonce), i.e. the PoW hash.
        if arduino is not None and arduino["verify"]["ok"]:
            key = (arduino["elapsed"], arduino["nonce"])

            if key != self._oracle_key:
                self._oracle_key = key
                self.engine.reset(arduino["header"], arduino["nonce"])
                self.v_custom_target.set("[SECURE]")

        def odds(bits):
            return f"1 in {2 ** bits:,}" if bits <= 20 else f"1 in 2^{bits}"

        self.v_pow_diff.set(
            f"Difficulty: {snap['diff']} bits   |   blind-guess odds: "
            f"nonce {odds(snap['diff'])}, "
            f"hash {odds(PASSCODE_BITS - snap['diff'])}"
        )

        if arduino is None:
            self.v_pow_header.set("HEADER   ...")
            self.v_pow_nonce.set("NONCE    ...")
            self.v_pow_hash.set("HASH     ...")
            self.v_pow_padded.set("PAD MSG  ...")
            self.v_pow_padded_nonce.set("PAD NONCE ...")
            self.v_pow_padded_hash.set("PAD HASH ...")
            self.v_pow_verify.set("VERIFY   ...")
        else:
            check = arduino["verify"]
            message = check["message"].decode("utf-8", errors="replace")

            self.v_pow_header.set(
                "HEADER   "
                + arduino["header"].decode("utf-8", errors="replace")
            )
            self.v_pow_nonce.set(f"NONCE    {arduino['nonce']}")
            self.v_pow_hash.set(
                f"HASH     {arduino['hash']:064x}  "
                f"({arduino['elapsed']:.3f} s)"
            )

            self.v_pow_padded.set(
                f"PAD MSG  {format_blocks(arduino['padded'])}"
            )
            self.v_pow_padded_nonce.set(
                f"PAD NONCE {format_blocks(arduino['padded_nonce'])}"
            )
            self.v_pow_padded_hash.set(
                f"PAD HASH {format_blocks(arduino['padded_hash'])}"
            )

            if check["ok"]:
                verdict = (
                    f"OK  SHA256(\"{message}\") equals the reported hash, "
                    f"{check['zero_bits']} leading zero bits "
                    f">= {snap['diff']} required, padded message, nonce and "
                    "hash are valid SHA-256 padding"
                )
            elif not check["hash_ok"]:
                verdict = (
                    f"FAILED  SHA256(\"{message}\") is "
                    f"{check['computed']:064x}, not the reported hash"
                )
            elif not check["padded_ok"]:
                verdict = (
                    "FAILED  a reported padded field is not the SHA-256 "
                    "padding of the matching message, nonce or hash"
                )
            else:
                verdict = (
                    f"FAILED  only {check['zero_bits']} leading zero bits, "
                    f"{snap['diff']} required"
                )

            self.v_pow_verify.set("VERIFY   " + verdict)

        if snap["predicted_padded"] is None:
            self.v_pow_oracle_padded.set(
                "ORACLE PADDED   ..."
                if predicted is None
                else "ORACLE PADDED   waiting for the header"
            )
        else:
            self.v_pow_oracle_padded.set(
                "ORACLE PADDED   (message | nonce | hash) "
                + format_blocks(snap["predicted_padded"])
            )

        if predicted is None:
            self.v_pow_oracle.set("ORACLE   ...")
        else:
            self.v_pow_oracle.set(
                f"ORACLE   guess nonce {predicted['nonce']}  "
                f"hash {predicted['hash']:064x}  "
                f"({predicted['elapsed']:.3f} s)"
            )

        guess_check = predicted["verify"] if predicted is not None else None

        if predicted is None:
            self.v_pow_oracle_verify.set("ORACLE VERIFY   ...")
        elif guess_check is None:
            self.v_pow_oracle_verify.set(
                "ORACLE VERIFY   waiting for the header"
            )
        else:
            message = guess_check["message"].decode("utf-8", errors="replace")
            detail = (
                f"\n                 SHA256(\"{message}\") = "
                f"{guess_check['computed']:064x}  "
                f"({guess_check['zero_bits']} leading zero bits, "
                f"{snap['diff']} required)"
            )

            if guess_check["ok"]:
                verdict = "OK  the guessed nonce and hash are a valid proof of work"
            elif not guess_check["hash_ok"]:
                verdict = "FAILED  the guessed hash is not SHA256(header + guessed nonce)"
            else:
                verdict = "FAILED  the guessed hash lacks the required leading zero bits"

            self.v_pow_oracle_verify.set(
                "ORACLE VERIFY   " + verdict + detail
            )

        if snap["match"] is None:
            self.v_pow_match.set("")
        elif snap["match"]:
            self.v_pow_match.set(
                "MATCH: oracle nonce and hash equal the Arduino's"
            )
        else:
            nonce_text = "nonce equal" if snap["nonce_match"] else "nonce differs"
            self.v_pow_match.set(
                f"NO MATCH: {nonce_text}, {snap['bits_agree']}/256 hash bits "
                f"agree (the {snap['diff']}-bit zero prefix is shared by "
                "definition)"
            )

        if snap["padded_agree"] is not None:
            agree, total, same = snap["padded_agree"]
            self.v_pow_match.set(
                self.v_pow_match.get()
                + "\n"
                + (
                    "PADDED MATCH: oracle's whole padded record equals the "
                    "Arduino's"
                    if same
                    else f"PADDED NO MATCH: {agree}/{total} padded-record "
                    "bits agree"
                )
            )

    def set_custom_target(self):
        text = self.v_custom_target.get()

        if not text or text == "[SECURE]":
            messagebox.showerror(
                "Invalid Input",
                "Enter the header the oracle should pad (nonce 0).",
            )
            return

        self.engine.reset(text.encode("utf-8"), 0)
        self.v_custom_target.set("[SECURE]")
        messagebox.showinfo(
            "Oracle Armed",
            "Passcode is now the padded message, nonce and hash of that "
            "header with nonce 0 (hidden until pooled).",
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
        depth = self.engine.bit_depth
        width = depth // 4

        # Grow the canvas so a long header's rows are never clipped.
        row_lines = 1 if not is_exposed else -(-width // 64)
        row_lines_full = -(-width // 64)
        needed = 14 + (row_lines + 2 * row_lines_full) * 15 + 3 * 8 + 30
        self.register_canvas.config(height=max(330, needed))

        rows = [
            ("TARGET", active_target, "#00e676", not is_exposed),
            ("GUESS ", guess_value, "#ffb300", False),
            ("POOLED", pooled_value, "#b388ff", False),
        ]

        y_offset = 14

        for label, value, colour, hidden in rows:
            self.register_canvas.create_text(
                10,
                y_offset,
                text=label,
                fill="#88a0c0",
                anchor="w",
                font=("Consolas", 10, "bold"),
            )

            hex_text = "[HIDDEN]" if hidden else f"{value:0{width}X}"

            # One 64-byte block (128 hex characters) per two lines.
            for start in range(0, len(hex_text), 64):
                self.register_canvas.create_text(
                    90,
                    y_offset,
                    text=hex_text[start:start + 64],
                    fill=colour,
                    anchor="w",
                    font=("Consolas", 10),
                )
                y_offset += 15

            y_offset += 8

        self.register_canvas.create_text(
            10,
            y_offset + 4,
            text=(
                f"{depth}-bit padded record | "
                f"{self.engine.step_count}/{depth} reconstruction steps"
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
                            f"Bit Step: {data['steps']}/{self.engine.bit_depth}"
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
    root.geometry("1240x880")
    root.mainloop()
