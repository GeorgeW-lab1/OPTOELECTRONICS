"""
Bit-Pooling Oracle Lock Box Lab
Python + Arduino SHA-256 proof-of-work miner.

Protocol:
    PC -> Arduino:  DIFF <0-256>\\n
    Arduino -> PC:  POW:<nonce>:<64 hex hash>:<padded message>:<padded nonce>
                        :<padded hash>:<header>\\n
                    (the only unprompted output)

Pressing "Send Difficulty" sends DIFF to the Arduino, which starts mining
SHA256(DATA + nonce) straight away and replies with the proof of work.

At the same moment, the camera bit-pooling oracle starts working in the
background on its own. It does not read anything from the Arduino: it
searches for the same proof of work itself, then loads the minimally padded
SHA-256 message block (header + nonce + 0x80 + zeros + 64-bit bit length)
into the bit-pooling register and reconstructs it bit by bit. The message,
nonce and hash are decoded purely from the reconstructed bits. The oracle's
result and the Arduino's result are each stored with their elapsed time and
compared once both exist.

An emulated Arduino (checkbox in the GUI) can stand in for the board.

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
def pool_bits(val, bit_depth, kernel_size, stride, pool_op, schmitt_trigger):
    """Applies a pooling operator to a register value."""
    bits = [(val >> i) & 1 for i in range(bit_depth)]
    pooled_out = []

    for i in range(0, len(bits), stride):
        block = bits[i:i + kernel_size]

        if not block:
            break

        if pool_op == "POPCOUNT":
            result = sum(block) / len(block)
            output_bit = 1 if result >= schmitt_trigger else 0
        elif pool_op == "AND":
            output_bit = 1 if all(b == 1 for b in block) else 0
        elif pool_op == "OR":
            output_bit = 1 if any(b == 1 for b in block) else 0
        elif pool_op == "XOR":
            output_bit = sum(block) % 2
        elif pool_op == "MAJORITY":
            output_bit = 1 if sum(block) > (len(block) / 2.0) else 0
        else:
            output_bit = block[0]

        pooled_out.append(output_bit)

    output_value = 0
    for index, bit in enumerate(pooled_out):
        output_value |= bit << index

    return output_value


# ==============================================================================

class BitPoolingOracle:
    """
    Stores a hidden target and answers higher-or-equal queries.

    The search engine reconstructs the target one bit at a time, MSB first.
    """

    def __init__(self, bit_depth: int = PASSCODE_BITS, target=None):
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer.")

        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        if target is None:
            target = secrets.randbits(bit_depth)

        # A target that exposes value() is a live connection: nothing is
        # generated or stored here, every answer is read through the link.
        self._link = None
        self._fixed = 0

        if callable(getattr(target, "value", None)):
            self._link = target
        else:
            self._fixed = int(target) & self.max_val

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.pooled_bits = 0
        self.pooled_count = 0
        self.is_done = False

    @property
    def _target(self) -> int:
        if self._link is not None:
            return self._link.value() & self.max_val

        return self._fixed

    def query_bit(self, trial_val: int) -> int:
        """Returns 1 if the hidden target is greater than or equal to trial_val."""
        if self.is_done:
            return 0

        return 1 if self._target >= trial_val else 0

    def pool_step(self, val: int) -> int:
        """Applies the selected pooling operator to the current guess."""
        return pool_bits(
            val,
            self.bit_depth,
            self.kernel_size,
            self.stride,
            self.pool_op,
            self.schmitt_trigger,
        )

    @property
    def target(self) -> int:
        """The target remains hidden until the full bit depth is reconstructed."""
        if not self.is_done:
            raise PermissionError(
                "SECURITY VIOLATION: Oracle target is classified until fully pooled."
            )

        return self._target


# ==============================================================================
# BIT-POOLING REGISTERS (passive: no work of their own)
# ==============================================================================

class BitPoolingRegisters:
    """
    Passive registers. They do no work of their own: no searching, no
    stepping, no querying. They only hold what the oracle pushes into them
    (the revealed bits so far) and pool that value for display.
    """

    def __init__(self, bit_depth: int = PASSCODE_BITS):
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer.")

        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        self._lock = threading.RLock()

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.reset()

    def reset(self, bit_depth=None, linked=False):
        """Clears the registers. They stay idle until the oracle fills them."""
        with self._lock:
            if bit_depth is not None:
                if not isinstance(bit_depth, int) or bit_depth < 1:
                    raise ValueError("bit_depth must be a positive integer.")
                self.bit_depth = bit_depth
                self.max_val = (1 << bit_depth) - 1

            self.linked = linked
            self.current_guess = 0
            self.step_count = 0
            self.history = [0]
            self.converged = False
            self.last_status = "READY" if linked else "IDLE"
            self.target_reg = 0
            self._refresh()

    def receive(self, guess, steps, depth, done, status, target=0):
        """Takes the oracle's current state. The only way values get in."""
        with self._lock:
            if depth != self.bit_depth:
                self.bit_depth = depth
                self.max_val = (1 << depth) - 1

            self.linked = True
            self.current_guess = guess
            self.step_count = steps
            self.history.append(guess)
            self.converged = done
            self.last_status = status
            self.target_reg = target
            self._refresh()

    def result(self):
        """(guess, converged, bit_depth) read atomically."""
        with self._lock:
            return self.current_guess, self.converged, self.bit_depth

    def _refresh(self):
        self.guess_reg = self.current_guess
        self.pooled_register = pool_bits(
            self.current_guess,
            self.bit_depth,
            self.kernel_size,
            self.stride,
            self.pool_op,
            self.schmitt_trigger,
        )


# ==============================================================================
# SHA-256 PROOF OF WORK + MINIMAL PADDING
# ==============================================================================

POW_DATA = "arduino-sha256"  # must match inputData in sha256_challenge.ino
MIN_DIFF = 0
MAX_DIFF = 256


def sha256_pad(msg: bytes) -> bytes:
    """Minimal SHA-256 padding: msg, 0x80, zeros, 64-bit big-endian bit length."""
    total = ((len(msg) + 9 + 63) // 64) * 64
    return (
        msg
        + b"\x80"
        + b"\x00" * (total - len(msg) - 9)
        + (len(msg) * 8).to_bytes(8, "big")
    )


def sha256_unpad(padded: bytes) -> bytes:
    """Inverse of sha256_pad. Raises ValueError if padding is not minimal/valid."""
    if len(padded) < 64 or len(padded) % 64:
        raise ValueError("not a whole number of 64-byte blocks")

    length = int.from_bytes(padded[-8:], "big") // 8

    if length > len(padded) - 9 or sha256_pad(padded[:length]) != padded:
        raise ValueError("invalid padding")

    return padded[:length]


def message_text(padded: bytes) -> str:
    try:
        return sha256_unpad(padded).decode("ascii", errors="replace")
    except ValueError:
        return "<invalid padding>"


class PowLink:
    """
    The oracle's ethereal connection to the mining result. The oracle does not
    generate or hold a target: the target is whatever the proof of work for
    this session resolves to (the minimally padded SHA256(DATA + nonce)
    message), and it is read through this link when the oracle is queried.
    Nothing is read from the Arduino.
    """

    def __init__(self, data: str, diff: int, cancel: threading.Event):
        self._data = data
        self._diff = diff
        self._cancel = cancel
        self._lock = threading.Lock()
        self._padded = None
        self._depth = None

    def _resolve(self) -> bool:
        with self._lock:
            if self._padded is not None:
                return True

            base = hashlib.sha256(self._data.encode("ascii"))
            limit = 1 << (256 - self._diff)  # leading-zero test: value < limit

            nonce = 0

            while nonce <= 0xFFFFFFFF:
                if (nonce & 0xFFF) == 0 and self._cancel.is_set():
                    return False

                h = base.copy()
                h.update(str(nonce).encode("ascii"))

                if int.from_bytes(h.digest(), "big") < limit:
                    padded = sha256_pad((self._data + str(nonce)).encode("ascii"))
                    self._padded = int.from_bytes(padded, "big")
                    self._depth = len(padded) * 8
                    return True

                nonce += 1

            return False

    def depth(self):
        """Bit depth of the connected target, or None if cancelled."""
        return self._depth if self._resolve() else None

    def value(self) -> int:
        if not self._resolve():
            raise PermissionError("Oracle link closed before the target resolved.")

        return self._padded


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

    def _record(
        self, nonce: int, hash_int: int, padded: bytes, elapsed=None
    ) -> dict:
        return {
            "nonce": nonce,
            "hash": hash_int,
            "padded": padded,
            "target": padded.hex(),
            "message": message_text(padded),
            "elapsed": (
                time.perf_counter() - self.t0
                if elapsed is None
                else elapsed
            ),
        }

    def set_predicted(
        self, nonce: int, hash_int: int, padded: bytes, elapsed=None
    ):
        if self.cancel.is_set():
            return

        with self._lock:
            self.predicted = self._record(nonce, hash_int, padded, elapsed)

        self.out.put(("pow_update", self.snapshot()))

    def set_live(self, guess: int, steps: int, depth: int):
        """Real-time oracle view: whatever the bit pooler has revealed so far."""
        if self.cancel.is_set():
            return

        raw = guess.to_bytes(depth // 8, "big")[: steps // 8]
        body = raw.split(b"\x80")[0]
        text = body.decode("ascii", errors="replace")
        nonce_text = text[len(self.data):] if text.startswith(self.data) else ""

        with self._lock:
            self.predicted = {
                "live": True,
                "steps": steps,
                "depth": depth,
                "nonce": nonce_text,
                "hash": None,
                "padded": raw,
                "target": f"{guess:0{depth // 4}x}"[: steps // 4].ljust(
                    depth // 4, "."
                ),
                "message": text,
                "elapsed": time.perf_counter() - self.t0,
            }

        self.out.put(("pow_update", self.snapshot()))

    def set_arduino(self, nonce: int, hash_int: int, padded: bytes):
        if self.cancel.is_set():
            return

        with self._lock:
            self.arduino = self._record(nonce, hash_int, padded)

        self.out.put(("pow_update", self.snapshot()))

    def snapshot(self) -> dict:
        with self._lock:
            predicted = dict(self.predicted) if self.predicted else None
            arduino = dict(self.arduino) if self.arduino else None

        match = None

        if (
            predicted is not None
            and not predicted.get("live")
            and arduino is not None
        ):
            match = (
                predicted["nonce"] == arduino["nonce"]
                and predicted["hash"] == arduino["hash"]
                and predicted["padded"] == arduino["padded"]
            )

        return {
            "diff": self.diff,
            "predicted": predicted,
            "arduino": arduino,
            "match": match,
        }


class CameraOracle:
    """
    The camera-driven oracle. All the work happens here, none in the
    registers. It is connected to the mining result through a PowLink, and
    each camera-triggered step reveals one more bit of the target (MSB first)
    by asking the link, then pushes the revealed state into the registers.
    """

    def __init__(self, registers: BitPoolingRegisters):
        self.registers = registers
        self._lock = threading.RLock()
        self._pool = None
        self.session = None
        self.depth = 0
        self.steps = 0
        self.done = False

    def arm(self, session, link, depth: int):
        """Connects the oracle to a mining result; the camera does the rest."""
        with self._lock:
            self.session = session
            self._pool = BitPoolingOracle(depth, link)
            self.depth = depth
            self.steps = 0
            self.done = False
            self.registers.reset(depth, linked=True)

    def reset(self):
        with self._lock:
            self.session = None
            self._pool = None
            self.depth = 0
            self.steps = 0
            self.done = False
            self.registers.reset(PASSCODE_BITS)

    def step(self):
        """Reveals one bit. Driven by the camera (or the Step button)."""
        with self._lock:
            pool = self._pool

            if pool is None or self.done:
                return

            self.steps += 1

            shift = self.depth - self.steps
            trial_value = pool.pooled_bits | (1 << shift)

            if pool.query_bit(trial_value):
                pool.pooled_bits |= 1 << shift

            guess = pool.pooled_bits
            target = 0
            status = "REVEALING"

            if self.steps >= self.depth:
                pool.is_done = True
                self.done = True
                target = pool.target

                status = (
                    "LOCKED & REVEALED"
                    if guess == target
                    else "RECONSTRUCTION MISMATCH"
                )

            self.registers.receive(
                guess, self.steps, self.depth, self.done, status, target
            )

            if self.session is not None:
                self.session.set_live(guess, self.steps, self.depth)


class BackgroundPredictor(threading.Thread):
    """
    The oracle's background worker. It runs independently of the Arduino
    link: it connects the camera oracle to the mining result (SHA256(DATA +
    nonce), nonce counting up from 0) through a PowLink, then waits while the
    camera drives the reveal. When the registers have been filled, message,
    nonce and hash are decoded from the registers' bits alone.
    """

    def __init__(self, session: MiningSession, oracle: CameraOracle):
        super().__init__(daemon=True)
        self.session = session
        self.oracle = oracle

    def run(self):
        session = self.session
        oracle = self.oracle

        link = PowLink(session.data, session.diff, session.cancel)
        depth = link.depth()

        if depth is None or session.cancel.is_set():
            return

        oracle.arm(session, link, depth)

        while (
            not session.cancel.is_set()
            and oracle.session is session
            and not oracle.done
        ):
            time.sleep(0.01)

        if session.cancel.is_set() or oracle.session is not session:
            return

        # Decode purely from the registers' bits.
        guess, converged, depth = oracle.registers.result()

        if not converged:
            return

        decoded = guess.to_bytes(depth // 8, "big")

        try:
            msg = sha256_unpad(decoded)
            nonce_out = int(msg[len(session.data):])
        except ValueError:
            return

        hash_out = int.from_bytes(hashlib.sha256(msg).digest(), "big")
        session.set_predicted(nonce_out, hash_out, decoded)


# ==============================================================================
# ARDUINO MINER LINK
# ==============================================================================

class ArduinoMiner:
    """
    Serial link to the Arduino SHA-256 miner.

        PC -> Arduino: DIFF <bits>\\n
        Arduino -> PC: POW:<nonce>:<hash>:<padded msg>:<padded nonce>
                           :<padded hash>:<header>\\n

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
        """Reads one line. Returns (nonce, hash_int, padded_bytes) or None."""
        raw = self.ser.readline()

        if not raw:
            return None

        return parse_pow_line(raw.decode("ascii", errors="ignore").strip())


def parse_pow_line(line: str):
    """
    Parses POW:<nonce>:<hash>:<padded msg>:<padded nonce>:<padded hash>:<header>
    Returns (nonce, hash_int, padded_message_bytes) or None.
    The header may contain colons, so it is the unsplit remainder.
    """
    if not line.startswith("POW:"):
        return None

    parts = line.split(":", 6)

    if len(parts) != 7 or len(parts[2]) != 64:
        return None

    try:
        return (
            int(parts[1]),
            int(parts[2], 16),
            bytes.fromhex(parts[3]),
        )
    except ValueError:
        return None


class EmulatedArduino:
    """
    Software stand-in for the physical Arduino. Same interface as
    ArduinoMiner (send_diff / read_pow / close) and the same behaviour as the
    sketch: DIFF <bits> starts mining SHA256(DATA + nonce) from nonce 0 at a
    throttled hash rate, silently, and one POW line is emitted when a hash
    has the required leading zero bits. The hash rate is configurable so the
    emulated board can be as slow as a real one.
    """

    def __init__(self, hashes_per_second: float = 300.0, data: str = POW_DATA):
        self.rate = max(1.0, float(hashes_per_second))
        self.data = data
        self._commands = queue.Queue()
        self._lines = queue.Queue()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def list_ports():
        return ["EMULATOR"]

    def close(self):
        self._stop.set()

    def send_diff(self, diff: int):
        self._commands.put(("DIFF", int(diff)))

    def stop_mining(self):
        self._commands.put(("STOP", 0))

    def read_pow(self):
        try:
            line = self._lines.get(timeout=0.2)
        except queue.Empty:
            return None

        return parse_pow_line(line)

    def _format(self, nonce: int, digest: bytes) -> str:
        """The sketch's POW line: padded message, nonce and hash, then header."""
        nonce_text = str(nonce).encode("ascii")
        message = self.data.encode("ascii") + nonce_text

        return ":".join((
            "POW",
            str(nonce),
            digest.hex(),
            sha256_pad(message).hex(),
            sha256_pad(nonce_text).hex(),
            sha256_pad(digest).hex(),
            self.data,
        ))

    def _run(self):
        mining = False
        nonce = 0
        base = None
        limit = 0
        budget = 0.0
        last = time.perf_counter()

        while not self._stop.is_set():
            try:
                while True:
                    command, arg = self._commands.get_nowait()

                    if command == "DIFF":
                        base = hashlib.sha256(self.data.encode("ascii"))
                        limit = 1 << (256 - arg)
                        nonce = 0
                        budget = 0.0
                        last = time.perf_counter()
                        mining = True
                    else:
                        mining = False
            except queue.Empty:
                pass

            if not mining:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            budget += (now - last) * self.rate
            last = now
            count = int(budget)
            budget -= count

            for _ in range(count):
                h = base.copy()
                h.update(str(nonce).encode("ascii"))
                digest = h.digest()

                if int.from_bytes(digest, "big") < limit:
                    self._lines.put(self._format(nonce, digest))
                    mining = False
                    break

                nonce += 1

                if nonce > 0xFFFFFFFF:
                    mining = False
                    break

            time.sleep(0.01)


class ArduinoThread(threading.Thread):
    def __init__(self, port, ctl, out, oracle, emulate=False, rate=300.0):
        super().__init__(daemon=True)
        self.emulate = emulate
        self.rate = rate
        self.port = port
        self.ctl = ctl
        self.out = out
        self.oracle = oracle
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
        BackgroundPredictor(session, self.oracle).start()

        self.out.put(("pow_update", session.snapshot()))

    def run(self):
        miner = None

        try:
            if self.emulate:
                miner = EmulatedArduino(self.rate)
                label = f"Connected (emulated, {self.rate:g} hashes/s)."
            else:
                miner = ArduinoMiner(self.port)
                label = "Connected."

            self.out.put((
                "arduino",
                f"{label} Set a difficulty and press Send Difficulty.",
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
    def __init__(self, cam_index, ctl, out, oracle):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.ctl = ctl
        self.out = out
        self.oracle = oracle

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out, oracle = self.ctl, self.out, self.oracle

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
                    and not oracle.done
                ):
                    oracle.step()

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

                out.put((
                    "frame",
                    {
                        "preview": preview,
                        "idx": frame_index,
                        "dt": dt,
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
            "Bit-Pooling Oracle Lock Box Lab — Padded Message Decoder"
        )

        self.q = queue.Queue()
        self.ctl = {
            "stop": False,
            "ir_heatmap": False,
            "auto_step": True,
            "force_reveal": False,
            "arduino_stop": False,
        }

        self.engine = BitPoolingRegisters(
            bit_depth=PASSCODE_BITS
        )
        self.oracle = CameraOracle(self.engine)
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

        # Arduino emulator (use instead of the physical board)
        emulator_frame = ttk.LabelFrame(
            main_frame,
            text="Arduino Emulator (no hardware needed)",
        )
        emulator_frame.pack(fill="x", padx=5, pady=5)

        self.v_emulate = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            emulator_frame,
            text="Emulate Arduino instead of the serial port",
            variable=self.v_emulate,
        ).pack(side="left", padx=5, pady=5)

        ttk.Label(
            emulator_frame,
            text="Emulated speed (hashes/s):",
        ).pack(side="left", padx=(20, 5), pady=5)

        self.v_emu_rate = tk.StringVar(value="300")
        ttk.Spinbox(
            emulator_frame,
            from_=1,
            to=1000000,
            increment=100,
            width=9,
            textvariable=self.v_emu_rate,
        ).pack(side="left", padx=2, pady=5)

        # Proof-of-work results
        pow_frame = ttk.LabelFrame(
            main_frame,
            text="Proof of Work",
        )
        pow_frame.pack(fill="x", padx=5, pady=5)

        self.v_link = tk.StringVar(value="Arduino: not connected")
        self.v_pow_diff = tk.StringVar(value="")
        self.v_pow_oracle = tk.StringVar(value="ORACLE   -")
        self.v_pow_oracle_msg = tk.StringVar(value="")
        self.v_pow_oracle_target = tk.StringVar(value="")
        self.v_pow_arduino = tk.StringVar(value="ARDUINO  -")
        self.v_pow_arduino_msg = tk.StringVar(value="")
        self.v_pow_match = tk.StringVar(value="")

        for var in (
            self.v_link,
            self.v_pow_diff,
            self.v_pow_oracle,
            self.v_pow_oracle_msg,
            self.v_pow_oracle_target,
            self.v_pow_arduino,
            self.v_pow_arduino_msg,
            self.v_pow_match,
        ):
            # Read-only Entry: text can be selected and copied (Ctrl+C).
            ttk.Entry(
                pow_frame,
                textvariable=var,
                font=("Consolas", 10),
                state="readonly",
            ).pack(fill="x", padx=10, pady=1)

        self.b_copy_pow = ttk.Button(
            pow_frame,
            text="Copy Proof of Work",
            command=self.copy_pow,
        )
        self.b_copy_pow.pack(anchor="w", padx=10, pady=(3, 5))

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
            text="Bit-Pooling Registers",
        )
        register_frame.pack(
            fill="both", expand=True, padx=5, pady=5
        )

        self.register_canvas = tk.Canvas(
            register_frame,
            width=760,
            height=300,
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
            0, self.ctl, self.q, self.oracle
        )
        self.camera_thread.start()

    def stop_camera(self):
        self.ctl["stop"] = True

    def step_search(self):
        self.oracle.step()

    def reset_search(self):
        self.oracle.reset()

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

        if not port and not self.v_emulate.get():
            messagebox.showerror(
                "Missing Port",
                "Enter a serial port such as COM3.",
            )
            return

        self.ctl["arduino_stop"] = False
        try:
            rate = float(self.v_emu_rate.get())
        except ValueError:
            rate = 300.0

        self.arduino_thread = ArduinoThread(
            port,
            self.ctl,
            self.q,
            self.oracle,
            emulate=self.v_emulate.get(),
            rate=rate,
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

            if item.get("live"):
                return (
                    f"{label}  nonce {item['nonce'] or '...'}  "
                    f"hash ...  ({item['elapsed']:.3f} s)  "
                    f"pooling {item['steps']}/{item['depth']}"
                )

            return (
                f"{label}  nonce {item['nonce']}  "
                f"hash {item['hash']:064x}  "
                f"({item['elapsed']:.3f} s)"
            )

        def msg(item):
            if item is None:
                return ""

            if item.get("live"):
                return (
                    f"          message {item['message']!r}  "
                    f"[{item['steps']}/{item['depth']} bits pooled]"
                )

            n = len(item["padded"])
            return (
                f"          message {item['message']!r}  "
                f"[{n} bytes padded, {n * 8} bits]"
            )

        self.v_pow_diff.set(f"Difficulty: {snap['diff']} bits")
        self.v_pow_oracle.set(line("ORACLE  ", snap["predicted"]))
        self.v_pow_oracle_msg.set(msg(snap["predicted"]))
        self.v_pow_oracle_target.set(
            ""
            if snap["predicted"] is None
            else "          target  " + snap["predicted"]["target"]
        )
        self.v_pow_arduino.set(line("ARDUINO ", snap["arduino"]))
        self.v_pow_arduino_msg.set(msg(snap["arduino"]))

        if snap["match"] is None:
            self.v_pow_match.set("")
        elif snap["match"]:
            self.v_pow_match.set("MATCH: same nonce, hash and padded message")
        else:
            self.v_pow_match.set(
                "MISMATCH: nonce, hash or padded message differs"
            )

    def copy_pow(self):
        lines = [
            var.get().strip()
            for var in (
                self.v_link,
                self.v_pow_diff,
                self.v_pow_oracle,
                self.v_pow_oracle_msg,
                self.v_pow_oracle_target,
                self.v_pow_arduino,
                self.v_pow_arduino_msg,
                self.v_pow_match,
            )
            if var.get().strip()
        ]

        self.window.clipboard_clear()
        self.window.clipboard_append("\n".join(lines))
        self.window.update()  # keep the clipboard after the app closes

        self.b_copy_pow.config(text="Copied!")
        self.window.after(
            1200,
            lambda: self.b_copy_pow.config(text="Copy Proof of Work"),
        )

    def _update_pool_operator(self, event):
        operator = self.operator_combo.get()
        self.engine.pool_op = operator

    def _update_kernel(self):
        try:
            kernel = int(self.kernel_spin.get())
            self.engine.kernel_size = kernel
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

        rows = [
            ("TARGET", active_target, "#00e676", not is_exposed),
            ("GUESS ", guess_value, "#ffb300", False),
            ("POOLED", pooled_value, "#b388ff", False),
        ]

        y = 20

        for label, value, colour, hidden in rows:
            self.register_canvas.create_text(
                10,
                y,
                text=label,
                fill="#88a0c0",
                anchor="w",
                font=("Consolas", 10, "bold"),
            )

            if hidden and label == "TARGET" and self.engine.linked:
                # Connected to the oracle: the target register is aligned with
                # what the oracle has revealed so far (unrevealed nibbles: '.').
                shown = self.engine.step_count // 4
                full = f"{guess_value:0{width}X}"[:shown].ljust(width, ".")
            else:
                full = "[HIDDEN]" if hidden else f"{value:0{width}X}"

            # Wrap long registers (128 hex chars for a 512-bit block).
            for i in range(0, len(full), 64):
                self.register_canvas.create_text(
                    90,
                    y,
                    text=full[i:i + 64],
                    fill=colour,
                    anchor="w",
                    font=("Consolas", 10),
                )
                y += 18

            y += 17

        self.register_canvas.create_text(
            10,
            y,
            anchor="w",
            fill="#d2e0f0",
            font=("Consolas", 10),
            text=(
                f"{depth}-bit register | "
                f"{self.engine.step_count}/{depth} reconstruction steps"
            ),
        )

    def _refresh_registers(self):
        """Redraws the registers from the engine on every GUI tick, so they
        update whether or not the camera is running."""
        engine = self.engine

        self._draw_hex_rows(
            engine.target_reg,
            engine.current_guess,
            engine.pooled_register,
            False,
        )

        self.status_label.config(
            text=(
                f"Status: {engine.last_status} | "
                f"Bit Step: {engine.step_count}/{engine.bit_depth}"
            )
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

        self._refresh_registers()
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
