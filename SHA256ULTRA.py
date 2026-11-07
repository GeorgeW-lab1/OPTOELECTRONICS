"""
Bit-Pooling Oracle Lock Box Lab — Parallel Optical Grid Solvers
Python SHA-256 proof-of-work miner with pure Python emulation & parallel spatial grid processing.

Each bit of the reconstructed passcode is mapped to a dedicated spatial sector grid in 
the webcam feed. Optical modulation within each sector independently triggers and solves 
that specific bit position.

Install dependencies:
    pip install opencv-python numpy
    (pyserial is optional; hardware Arduino communication falls back to Python software emulation)
"""

import asyncio
import hashlib
import math
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
    zeros = (55 - len(message)) % 64
    return (
        message
        + b"\x80"
        + b"\x00" * zeros
        + (len(message) * 8).to_bytes(8, "big")
    )


def format_blocks(data: bytes) -> str:
    return " ".join(
        data[i:i + 64].hex() for i in range(0, len(data), 64)
    )


def padded_agreement(a: bytes, b: bytes):
    n = min(len(a), len(b))
    diff = int.from_bytes(a[:n], "big") ^ int.from_bytes(b[:n], "big")
    total = max(len(a), len(b)) * 8
    return n * 8 - bin(diff).count("1"), total, a == b


def pow_blocks(header: bytes, nonce: int):
    nonce_text = str(nonce).encode("ascii")
    message = header + nonce_text
    digest = hashlib.sha256(message).digest()

    return (
        sha256_pad(message),
        sha256_pad(nonce_text),
        sha256_pad(digest),
    )


def pow_whole(header: bytes, nonce: int) -> bytes:
    return b"".join(pow_blocks(header, nonce))


def sha256_unpad(padded: bytes):
    if len(padded) < 64 or len(padded) % 64:
        return None

    bit_length = int.from_bytes(padded[-8:], "big")

    if bit_length % 8 or bit_length // 8 + 9 > len(padded):
        return None

    message = padded[:bit_length // 8]
    return message if sha256_pad(message) == padded else None


def extract_partial(known: bytes, total: int) -> dict:
    info = {
        "known": len(known),
        "total": total,
        "message": None,
        "blocks": None,
        "header": None,
        "nonce": None,
        "hash": None,
        "hash_ok": None,
        "zero_bits": None,
        "error": None,
        "complete": False,
    }

    message = None
    msg_end = None

    for blocks in range(1, min(3, len(known) // 64) + 1):
        message = sha256_unpad(known[:64 * blocks])

        if message is not None:
            info["message"] = message
            info["blocks"] = blocks
            msg_end = 64 * blocks
            break

    if msg_end is None:
        if len(known) >= 192:
            info["error"] = "no valid SHA-256 padding in the first 3 blocks"

        return info

    if len(known) < msg_end + 64:
        return info

    nonce_text = sha256_unpad(known[msg_end:msg_end + 64])

    if (
        nonce_text is None
        or not nonce_text.isdigit()
        or not message.endswith(nonce_text)
    ):
        info["error"] = (
            "the padded nonce block is invalid or the message does not "
            "end with the nonce"
        )
        return info

    info["nonce"] = int(nonce_text)
    info["header"] = message[:-len(nonce_text)]

    if len(known) < msg_end + 128:
        return info

    digest = sha256_unpad(known[msg_end + 64:msg_end + 128])

    if digest is None or len(digest) != 32:
        info["error"] = "the padded hash block is invalid"
        return info

    info["hash"] = int.from_bytes(digest, "big")
    info["hash_ok"] = hashlib.sha256(message).digest() == digest
    info["zero_bits"] = PASSCODE_BITS - info["hash"].bit_length()
    info["complete"] = True

    if not info["hash_ok"]:
        info["error"] = (
            "SHA-256 of the extracted message is not the extracted hash"
        )

    return info


# ==============================================================================
# 256-BIT BIT-POOLING ORACLE
# ==============================================================================

class BitPoolingOracle:
    def __init__(self, header=None, nonce=None):
        if header is None:
            header = secrets.token_bytes(16)

        if isinstance(header, str):
            header = header.encode("utf-8")

        if nonce is None:
            nonce = secrets.randbelow(1 << 32)

        self.header = bytes(header)
        self.nonce = int(nonce)

        self.bit_depth = len(pow_whole(self.header, self.nonce)) * 8
        self.max_val = (1 << self.bit_depth) - 1

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.pooled_bits = 0
        self.is_done = False

    def _padded_value(self) -> int:
        return int.from_bytes(pow_whole(self.header, self.nonce), "big")

    @property
    def _target(self) -> int:
        return self._padded_value()

    def query_bit(self, trial_val: int) -> int:
        if self.is_done:
            return 0
        return 1 if self._padded_value() >= trial_val else 0

    def query_single_bit_position(self, bit_index: int) -> int:
        """Queries whether a specific bit position in target is set (MSB at index 0)."""
        target_val = self._padded_value()
        shift = self.bit_depth - 1 - bit_index
        return (target_val >> shift) & 1

    def pool_step(self, val: int) -> int:
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
        if not self.is_done:
            raise PermissionError(
                "SECURITY VIOLATION: Oracle target is classified until fully pooled."
            )
        return self._target


# ==============================================================================
# PARALLEL BIT-POOLING SEARCH ENGINE
# ==============================================================================

class ParallelBitPoolingSearchEngine:
    """
    Search engine supporting independent parallel bit solving via spatial sectors.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self.oracle = BitPoolingOracle()
        self.bit_depth = self.oracle.bit_depth

        self.current_guess = 0
        self.solved_bits_mask = [False] * self.bit_depth
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

        self.oracle.pool_op = self.pool_op
        self.oracle.kernel_size = self.kernel_size
        self.current_guess = 0
        self.solved_bits_mask = [False] * self.bit_depth
        self.history = [0]
        self.converged = False
        self.last_status = "READY"
        self._update_registers()

    def solve_bit_at_index(self, bit_index: int):
        """Solves a single bit index in parallel based on optical sector trigger."""
        if self.converged or bit_index < 0 or bit_index >= self.bit_depth:
            return

        if self.solved_bits_mask[bit_index]:
            return

        shift = self.bit_depth - 1 - bit_index
        bit_val = self.oracle.query_single_bit_position(bit_index)

        if bit_val:
            self.current_guess |= (1 << shift)
        else:
            self.current_guess &= ~(1 << shift)

        self.solved_bits_mask[bit_index] = True
        self.oracle.pooled_bits = self.current_guess
        self.history.append(self.current_guess)
        self._update_registers()

        if all(self.solved_bits_mask):
            self.oracle.is_done = True
            self.converged = True

            if self.current_guess == self.oracle._target:
                self.last_status = "LOCKED & REVEALED"
            else:
                self.last_status = "RECONSTRUCTION MISMATCH"

    def step_sequential(self):
        """Sequential single bit step fallback."""
        with self._lock:
            for idx, solved in enumerate(self.solved_bits_mask):
                if not solved:
                    self.solve_bit_at_index(idx)
                    break

    def reset(self, header=None, nonce=None):
        with self._lock:
            self._reset(header, nonce)

    def _update_registers(self):
        self.guess_reg = self.current_guess

        try:
            self.target_reg = self.oracle.target
        except PermissionError:
            self.target_reg = 0

        self.pooled_register = self.oracle.pool_step(self.current_guess)

    @property
    def step_count(self):
        return sum(1 for s in self.solved_bits_mask if s)


# ==============================================================================
# PROOF-OF-WORK SESSION & EMULATION
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
    free_bits = PASSCODE_BITS - diff
    hash_guess = secrets.randbits(free_bits) if free_bits > 0 else 0
    nonce_guess = secrets.randbelow(1 << min(max(diff - 4, 0), 32))
    return nonce_guess, hash_guess


class BackgroundPredictor(threading.Thread):
    def __init__(self, session: MiningSession):
        super().__init__(daemon=True)
        self.session = session

    def run(self):
        session = self.session

        if session.cancel.is_set():
            return

        nonce, hash_guess = blind_guess(session.diff)
        session.set_predicted(hash_guess, nonce)


class EmulatedArduinoMiner:
    def __init__(self):
        self.input_data = "arduino-sha256"
        self.difficulty = 12
        self.mining = False
        self.nonce = 0
        self.hashes = 0
        self._lock = threading.Lock()
        
    def list_ports(self):
        return ["EMULATED", "COM3"]

    def close(self):
        with self._lock:
            self.mining = False

    def query_header(self, timeout: float = 1.0):
        return self.input_data.encode("utf-8")

    def send_diff(self, diff: int):
        with self._lock:
            self.difficulty = int(diff)
            self.nonce = 0
            self.hashes = 0
            self.mining = True

    def step_mining(self):
        with self._lock:
            if not self.mining:
                return None
            
            for _ in range(500):
                nonce_text = str(self.nonce).encode("ascii")
                message = self.input_data.encode("utf-8") + nonce_text
                digest = hashlib.sha256(message).digest()
                
                computed = int.from_bytes(digest, "big")
                zero_bits = PASSCODE_BITS - computed.bit_length()
                
                self.hashes += 1
                
                if zero_bits >= self.difficulty:
                    self.mining = False
                    p_msg, p_nonce, p_hash = pow_blocks(self.input_data.encode("utf-8"), self.nonce)
                    return (
                        self.nonce,
                        computed,
                        p_msg,
                        p_nonce,
                        p_hash,
                        self.input_data.encode("utf-8")
                    )
                
                self.nonce += 1
                if self.nonce == 0:
                    self.mining = False
                    break
        return None


class ArduinoMiner:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.2):
        if port.upper() == "EMULATED":
            self.is_emulated = True
            self.emu = EmulatedArduinoMiner()
            return
            
        self.is_emulated = False
        if serial is None:
            raise RuntimeError("pyserial is not installed. Type 'EMULATED' to use Python software emulation.")

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    @staticmethod
    def list_ports():
        ports = ["EMULATED"]
        if serial_tools is not None:
            ports.extend([device.device for device in serial_tools.list_ports.comports()])
        return ports

    def close(self):
        if self.is_emulated:
            if hasattr(self, 'emu'):
                self.emu.close()
        elif self.ser and self.ser.is_open:
            self.ser.close()

    def query_header(self, timeout: float = 1.0):
        if self.is_emulated:
            return self.emu.query_header(timeout)
            
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
        if self.is_emulated:
            self.emu.send_diff(diff)
            return
        self.ser.write(f"DIFF {int(diff)}\n".encode("ascii"))
        self.ser.flush()

    def read_pow(self):
        if self.is_emulated:
            return self.emu.step_mining()
            
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

        miner.send_diff(diff)
        BackgroundPredictor(session).start()

        self.out.put(("pow_update", session.snapshot()))

    def run(self):
        miner = None
        try:
            miner = ArduinoMiner(self.port)
            connected_msg = "Connected (Software Emulation)." if miner.is_emulated else "Connected via Serial."
            self.out.put(("arduino", connected_msg))

            while not self.ctl["arduino_stop"]:
                try:
                    while True:
                        self._start_session(miner, self.commands.get_nowait())
                except queue.Empty:
                    pass

                result = miner.read_pow()

                if result is not None and self.session is not None:
                    self.session.set_arduino(*result)
                elif miner.is_emulated:
                    time.sleep(0.005)

        except Exception as e:
            self.out.put(("arduino_error", f"{type(e).__name__}: {e}"))

        finally:
            if self.session is not None:
                self.session.cancel.set()

            if miner is not None:
                miner.close()

            self.out.put(("arduino", "Disconnected."))


# ==============================================================================
# CAMERA SOURCE & PARALLEL SPATIAL GRID PROCESSING
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
    data = (
        b"P6 %d %d 255\n" % (w, h)
        + np.ascontiguousarray(rgb, np.uint8).tobytes()
    )
    return tk.PhotoImage(data=data, format="PPM")


class ParallelGridSimThread(threading.Thread):
    """
    Divides camera viewport into a spatial grid (1 sector per bit).
    Processes motion/intensity per sector to independently solve bits in parallel.
    """
    def __init__(self, cam_index, ctl, out, engine: ParallelBitPoolingSearchEngine):
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

        out.put(("info", f"USB camera {self.cam_index} opened in Parallel Spatial Sector Mode."))
        out.put(("camera_started", None))

        previous_time = time.perf_counter()
        frame_index = 0
        prev_gray = None

        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)

                if gray is None:
                    out.put(("info", "Camera stopped delivering frames."))
                    break

                now = time.perf_counter()
                dt = min(max(now - previous_time, 1e-3), 0.25)
                previous_time = now

                bit_depth = engine.bit_depth
                grid_cols = 32
                grid_rows = math.ceil(bit_depth / grid_cols)

                h, w = gray.shape
                cell_w = w / grid_cols
                cell_h = h / grid_rows

                rgb = src.rgb
                if rgb is None:
                    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                elif rgb.ndim == 2:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
                elif rgb.shape[2] == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

                preview = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA).copy()

                if ctl.get("ir_heatmap", False):
                    preview[:, :, 0] = np.clip(preview[:, :, 0].astype(int) + 80, 0, 255)
                    preview[:, :, 2] = np.clip(preview[:, :, 2].astype(int) - 50, 0, 255)

                # Process parallel spatial sectors
                if prev_gray is not None and frame_index > 5:
                    diff_frame = cv2.absdiff(gray, prev_gray)
                    
                    for bit_idx in range(bit_depth):
                        if engine.solved_bits_mask[bit_idx]:
                            continue

                        r = bit_idx // grid_cols
                        c = bit_idx % grid_cols

                        x1, y1 = int(c * cell_w), int(r * cell_h)
                        x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)

                        sector = diff_frame[y1:y2, x1:x2]
                        sector_activity = np.mean(sector)

                        # Trigger parallel bit solve if sector exceeds optical activity threshold
                        if sector_activity > 3.0 and ctl.get("auto_step", True):
                            with engine._lock:
                                engine.solve_bit_at_index(bit_idx)

                prev_gray = gray.copy()

                # Overlay spatial grid and solved bit statuses on viewport
                scale_x = 320 / w
                scale_y = 240 / h
                for bit_idx in range(bit_depth):
                    r = bit_idx // grid_cols
                    c = bit_idx % grid_cols

                    px1, py1 = int(c * cell_w * scale_x), int(r * cell_h * scale_y)
                    px2, py2 = int((c + 1) * cell_w * scale_x), int((r + 1) * cell_h * scale_y)

                    is_solved = engine.solved_bits_mask[bit_idx]
                    color = (0, 255, 0) if is_solved else (80, 80, 80)
                    cv2.rectangle(preview, (px1, py1), (px2, py2), color, 1)

                exposed_target = (
                    engine.oracle._target
                    if (engine.converged or ctl.get("force_reveal", False))
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

class SelectableText(tk.Text):
    def __init__(self, master, text="", **kw):
        style = ttk.Style()
        kw.setdefault(
            "bg",
            style.lookup("TFrame", "background")
            or master.winfo_toplevel().cget("background"),
        )
        kw.setdefault("fg", style.lookup("TLabel", "foreground") or "black")
        kw.setdefault("font", ("Consolas", 10))
        kw.setdefault("height", 1)
        kw.setdefault("width", 20)

        super().__init__(
            master,
            wrap="char",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            cursor="xterm",
            insertwidth=0,
            **kw,
        )

        self._shown = None

        self.bind("<Button-1>", lambda e: self.focus_set(), add="+")
        self.bind("<Control-a>", self._select_all)
        self.bind("<Control-A>", self._select_all)
        self.bind("<Configure>", lambda e: self._fit())

        self._menu = tk.Menu(self, tearoff=0)
        self._menu.add_command(label="Copy", command=lambda: self.event_generate("<<Copy>>"))
        self._menu.add_command(label="Select All", command=self._select_all)
        self._menu.add_command(label="Copy All", command=self.copy_all)

        self.bind("<Button-3>", self._popup)
        if sys.platform == "darwin":
            self.bind("<Button-2>", self._popup)

        self.set(text)

    def _popup(self, event):
        self.focus_set()
        self._menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _select_all(self, event=None):
        self.tag_add("sel", "1.0", "end-1c")
        return "break"

    def value(self) -> str:
        return self.get("1.0", "end-1c")

    def copy_all(self):
        self.clipboard_clear()
        self.clipboard_append(self.value())

    def set(self, text: str):
        self.set_tagged([(text, None)])

    def set_tagged(self, parts):
        parts = tuple(parts)
        if parts == self._shown:
            return

        self._shown = parts
        self.configure(state="normal")
        self.delete("1.0", "end")

        for text, tag in parts:
            self.insert("end", text, (tag,) if tag else ())

        self.configure(state="disabled")
        self._fit()

    def _fit(self):
        try:
            counted = self.count("1.0", "end-1c", "update", "displaylines")
            lines = counted[0] if isinstance(counted, (tuple, list)) else counted
        except (tk.TclError, TypeError):
            lines = self.value().count("\n") + 1

        lines = max(1, int(lines or 1))
        if int(self.cget("height")) != lines:
            self.configure(height=lines)


class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Bit-Pooling Oracle Lock Box Lab — Parallel Spatial Sector Solver")

        self.q = queue.Queue()
        self.ctl = {
            "stop": False,
            "ir_heatmap": False,
            "auto_step": True,
            "force_reveal": False,
            "arduino_stop": False,
        }

        self.engine = ParallelBitPoolingSearchEngine()
        self.camera_thread = None
        self.camera_ready = False
        self.pending_diff = None
        self.arduino_thread = None
        self._oracle_key = None
        self._register_key = None
        self._extract_key = None
        self._arduino_pow = None

        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.after(15, self._poll)

    def _build_ui(self):
        outer = ttk.Frame(self.window)
        outer.pack(fill="both", expand=True)

        self._scroll_canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        main_frame = ttk.Frame(self._scroll_canvas, padding=10)
        frame_id = self._scroll_canvas.create_window((0, 0), window=main_frame, anchor="nw")

        main_frame.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all")),
        )
        self._scroll_canvas.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.itemconfigure(frame_id, width=e.width),
        )

        self.window.bind_all("<MouseWheel>", self._on_mousewheel)
        self.window.bind_all("<Button-4>", self._on_mousewheel)
        self.window.bind_all("<Button-5>", self._on_mousewheel)

        # Laboratory controls
        control_frame = ttk.LabelFrame(main_frame, text="Laboratory Controls")
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(control_frame, text="Start Camera Grid", command=self.start_camera).pack(side="left", padx=5, pady=5)
        ttk.Button(control_frame, text="Stop Camera Grid", command=self.stop_camera).pack(side="left", padx=5, pady=5)
        ttk.Button(control_frame, text="Step Next Unsolved Bit", command=self.step_search).pack(side="left", padx=5, pady=5)
        ttk.Button(control_frame, text="Reset Pool & Grid", command=self.reset_search).pack(side="left", padx=5, pady=5)

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="IR Thermal Heatmap",
            variable=self.v_ir,
            command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get()),
        ).pack(side="left", padx=10)

        # Arduino / Emulation controls
        arduino_frame = ttk.LabelFrame(main_frame, text="Arduino Miner / Software Emulation")
        arduino_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(arduino_frame, text="Port / Mode:").pack(side="left", padx=5, pady=5)

        self.v_arduino_port = tk.StringVar(value="EMULATED")
        self.e_arduino_port = ttk.Entry(arduino_frame, width=14, textvariable=self.v_arduino_port)
        self.e_arduino_port.pack(side="left", padx=2, pady=5)

        ttk.Button(arduino_frame, text="List Ports", command=self.list_arduino_ports).pack(side="left", padx=2, pady=5)
        ttk.Button(arduino_frame, text="Connect Miner", command=self.connect_arduino).pack(side="left", padx=10, pady=5)
        ttk.Button(arduino_frame, text="Disconnect", command=self.disconnect_arduino).pack(side="left", padx=2, pady=5)

        ttk.Label(arduino_frame, text="Difficulty (bits):").pack(side="left", padx=(25, 5), pady=5)

        self.v_diff = tk.StringVar(value="12")
        ttk.Spinbox(arduino_frame, from_=MIN_DIFF, to=MAX_DIFF, width=5, textvariable=self.v_diff).pack(side="left", padx=2, pady=5)
        ttk.Button(arduino_frame, text="Send Difficulty", command=self.send_difficulty).pack(side="left", padx=5, pady=5)

        # Proof-of-work results
        pow_frame = ttk.LabelFrame(main_frame, text="Proof of Work")
        pow_frame.pack(fill="x", padx=5, pady=5)
        pow_frame.columnconfigure(1, weight=1)

        fields = (
            ("v_link", "", "Miner: not connected (Type 'EMULATED' and click Connect)"),
            ("v_pow_diff", "", ""),
            ("v_pow_header", "HEADER", "-"),
            ("v_pow_nonce", "NONCE", "-"),
            ("v_pow_hash", "HASH", "-"),
            ("v_pow_padded", "PAD MSG", "-"),
            ("v_pow_padded_nonce", "PAD NONCE", "-"),
            ("v_pow_padded_hash", "PAD HASH", "-"),
            ("v_pow_times", "TIMES", "-"),
            ("v_pow_verify", "VERIFY", "-"),
            ("v_pow_oracle_nonce", "ORACLE NONCE", "-"),
            ("v_pow_oracle_hash", "ORACLE HASH", "-"),
            ("v_pow_oracle_padded", "ORACLE PADDED", "-"),
            ("v_pow_oracle_verify", "ORACLE VERIFY", "-"),
            ("v_pow_match", "", ""),
        )

        self.pow_fields = []
        for row, (attr, caption, initial) in enumerate(fields):
            ttk.Label(pow_frame, text=caption, font=("Consolas", 10)).grid(row=row, column=0, sticky="nw", padx=(10, 4), pady=1)
            box = SelectableText(pow_frame, initial)
            box.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=1)
            setattr(self, attr, box)
            self.pow_fields.append((caption, box))

        ttk.Button(pow_frame, text="Copy All Fields", command=self.copy_all_fields).grid(row=len(fields), column=1, sticky="e", padx=10, pady=3)

        # Local target controls
        target_frame = ttk.LabelFrame(main_frame, text="SHA-256 Oracle (passcode = padded message | nonce | hash)")
        target_frame.pack(fill="x", padx=5, pady=5)

        self.target_label = ttk.Label(target_frame, text="Oracle header (nonce 0):")
        self.target_label.pack(side="left", padx=5, pady=5)

        self.v_custom_target = tk.StringVar(value="[SECURE]")
        self.e_custom_target = ttk.Entry(target_frame, width=60, textvariable=self.v_custom_target)
        self.e_custom_target.pack(side="left", padx=5, pady=5)

        ttk.Button(target_frame, text="Hash Message", command=self.set_custom_target).pack(side="left", padx=2, pady=5)
        ttk.Button(target_frame, text="Randomize Target", command=self.randomize_target).pack(side="left", padx=5, pady=5)

        self.v_reveal = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            target_frame,
            text="Force Reveal Passcode",
            variable=self.v_reveal,
            command=lambda: self.ctl.update(force_reveal=self.v_reveal.get()),
        ).pack(side="left", padx=15, pady=5)

        # Fields recovered from completed passcode registers
        extract_frame = ttk.LabelFrame(main_frame, text="Extracted from Completed Passcode")
        extract_frame.pack(fill="x", padx=5, pady=5)
        extract_frame.columnconfigure(1, weight=1)

        self.extract_fields = []
        for row, (attr, caption) in enumerate((
            ("v_ex_header", "HEADER"),
            ("v_ex_nonce", "NONCE"),
            ("v_ex_message", "MESSAGE"),
            ("v_ex_hash", "HASH"),
            ("v_ex_check", "CHECK"),
        )):
            ttk.Label(extract_frame, text=caption, font=("Consolas", 10)).grid(row=row, column=0, sticky="nw", padx=(10, 4), pady=1)
            box = SelectableText(extract_frame, "-")
            box.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=1)
            setattr(self, attr, box)
            self.extract_fields.append((caption, box))

        ttk.Button(extract_frame, text="Copy Extracted", command=self.copy_extracted).grid(row=5, column=1, sticky="e", padx=10, pady=3)

        # Camera and register display
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        camera_frame = ttk.LabelFrame(display_frame, text="Optoelectronic Parallel Sector Grid Viewport")
        camera_frame.pack(side="left", padx=5, pady=5)

        self.cam_canvas = tk.Canvas(camera_frame, width=320, height=240, bg="black")
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_image_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        right_panel = ttk.Frame(display_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        register_frame = ttk.LabelFrame(right_panel, text="Padded-Message Passcode Registers")
        register_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.register_text = SelectableText(register_frame, "", bg="#121414", fg="#d2e0f0", width=100, height=12)
        self.register_text.pack(fill="x", padx=5, pady=5)

        self.register_text.tag_configure("label", foreground="#88a0c0", font=("Consolas", 10, "bold"))
        self.register_text.tag_configure("target", foreground="#00e676")
        self.register_text.tag_configure("guess", foreground="#ffb300")
        self.register_text.tag_configure("pooled", foreground="#b388ff")

        self.status_label = SelectableText(register_frame, "Status: IDLE", fg="#00e5ff", font=("Consolas", 11))
        self.status_label.pack(fill="x", padx=10, pady=5)

        ttk.Button(register_frame, text="Copy Registers", command=self.register_text.copy_all).pack(anchor="e", padx=10, pady=(0, 5))

        # Pooling settings
        settings_frame = ttk.LabelFrame(right_panel, text="Pooling & Algorithm Settings")
        settings_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(settings_frame, text="Pooling Operator:").grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.operator_combo = ttk.Combobox(settings_frame, values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"], state="readonly")
        self.operator_combo.set("POPCOUNT")
        self.operator_combo.grid(row=0, column=1, padx=5, pady=5)
        self.operator_combo.bind("<<ComboboxSelected>>", self._update_pool_operator)

        ttk.Label(settings_frame, text="Kernel Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.kernel_spin = ttk.Spinbox(settings_frame, from_=2, to=8, increment=2, width=5, command=self._update_kernel)
        self.kernel_spin.set(4)
        self.kernel_spin.grid(row=1, column=1, padx=5, pady=5)

    def _on_mousewheel(self, event):
        if self._scroll_canvas.yview() == (0.0, 1.0):
            return
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self._scroll_canvas.yview_scroll(step, "units")

    def start_camera(self):
        if self.camera_thread is not None and self.camera_thread.is_alive():
            messagebox.showwarning("Already Running", "The camera grid is already running.")
            return

        self.ctl["stop"] = False
        self.camera_thread = ParallelGridSimThread(0, self.ctl, self.q, self.engine)
        self.camera_thread.start()

    def stop_camera(self):
        self.ctl["stop"] = True

    def step_search(self):
        self.engine.step_sequential()
        self._refresh_registers()
        self._check_complete()

    def reset_search(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")
        self._arduino_pow = None
        self._refresh_registers()
        self._check_complete()

    def list_arduino_ports(self):
        try:
            ports = ArduinoMiner.list_ports()
        except Exception as e:
            messagebox.showerror("Serial Error", str(e))
            return

        if not ports:
            messagebox.showwarning("No Ports", "No ports were found.")
            return

        self.v_arduino_port.set(ports[0])
        messagebox.showinfo("Available Ports / Modes", "\n".join(ports))

    def connect_arduino(self):
        if self.arduino_thread is not None and self.arduino_thread.is_alive():
            messagebox.showwarning("Already Connected", "The miner is already connected.")
            return

        port = self.v_arduino_port.get().strip()
        if not port:
            messagebox.showerror("Missing Port", "Enter a serial port or type 'EMULATED'.")
            return

        self.ctl["arduino_stop"] = False
        self.arduino_thread = ArduinoThread(port, self.ctl, self.q)
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
        return self.arduino_thread is not None and self.arduino_thread.is_alive()

    def send_difficulty(self):
        if not self._arduino_alive():
            messagebox.showwarning("Not Connected", "Connect the miner first (try typing 'EMULATED').")
            return

        diff = self._read_diff()
        if diff is None:
            messagebox.showerror("Invalid Difficulty", f"Enter a whole number from {MIN_DIFF} to {MAX_DIFF}.")
            return

        if not self.camera_ready:
            self.pending_diff = diff
            self.v_link.set(f"Miner: difficulty {diff} queued, mining starts when webcam starts.")
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
            self.pending_diff = diff
            self.v_link.set(f"Webcam running. Difficulty {diff} will start mining when miner connects.")

    def copy_extracted(self):
        lines = [
            f"{caption}: {box.value().strip()}"
            for caption, box in self.extract_fields
            if box.value().strip() not in ("", "-", "...")
        ]
        self.window.clipboard_clear()
        self.window.clipboard_append("\n".join(lines))

    def _refresh_registers(self):
        engine = self.engine
        revealed = self.v_reveal.get()

        with engine._lock:
            exposed = engine.oracle._target if engine.converged or revealed else 0
            guess = engine.current_guess
            pooled = engine.pooled_register
            status = f"Status: {engine.last_status} | Solved Bits: {engine.step_count}/{engine.bit_depth}"

        self._draw_hex_rows(exposed, guess, pooled, revealed)
        self.status_label.set(status)

    def _check_complete(self):
        engine = self.engine

        with engine._lock:
            oracle = engine.oracle
            steps = engine.step_count
            guess = engine.current_guess
            depth = engine.bit_depth

        known_bytes = steps // 8
        key = (oracle, known_bytes)

        if key == self._extract_key:
            return

        self._extract_key = key
        if known_bytes == 0:
            self._show_extracted(None)
            return

        recovered = guess.to_bytes(depth // 8, "big")[:known_bytes]
        self._show_extracted(extract_partial(recovered, depth // 8))

    def _show_extracted(self, info):
        if info is None:
            for box in (
                self.v_ex_header,
                self.v_ex_nonce,
                self.v_ex_message,
                self.v_ex_hash,
                self.v_ex_check,
            ):
                box.set("-")
            return

        text = lambda raw: raw.decode("utf-8", errors="replace")

        self.v_ex_message.set(text(info["message"]) if info["message"] is not None else "...")
        self.v_ex_header.set(text(info["header"]) if info["header"] is not None else "...")
        self.v_ex_nonce.set(str(info["nonce"]) if info["nonce"] is not None else "...")
        self.v_ex_hash.set(f"{info['hash']:064x}" if info["hash"] is not None else "...")

        progress = f"{info['known']}/{info['total']} bytes of the guess recovered"

        if info["error"] is not None:
            verdict = "FAILED  " + info["error"]
        elif info["complete"]:
            verdict = (
                "OK  SHA256(message) equals the extracted hash, "
                f"{info['zero_bits']} leading zero bits"
            )

            arduino = self._arduino_pow
            if arduino is not None:
                same = (
                    info["header"] == arduino["header"]
                    and info["nonce"] == arduino["nonce"]
                    and info["hash"] == arduino["hash"]
                )
                verdict += (
                    "; identical to the miner's proof of work"
                    if same
                    else "; DIFFERS from the miner's proof of work"
                )
        elif info["message"] is None:
            verdict = f"{progress}; waiting for the first complete padded message block"
        elif info["nonce"] is None:
            verdict = (
                f"{progress}; padding revealed: the message is "
                f"{len(info['message'])} bytes in {info['blocks']} block(s); "
                "waiting for the nonce block"
            )
        else:
            verdict = f"{progress}; nonce block unpadded, header and nonce extracted; waiting for the hash block"

        self.v_ex_check.set(verdict)

    def copy_all_fields(self):
        lines = []
        for caption, box in self.pow_fields:
            text = box.value().strip()
            if text:
                lines.append(f"{caption}: {text}" if caption else text)

        self.window.clipboard_clear()
        self.window.clipboard_append("\n".join(lines))

    def _show_pow(self, snap):
        predicted, arduino = snap["predicted"], snap["arduino"]

        if arduino is not None and arduino["verify"]["ok"]:
            key = (arduino["elapsed"], arduino["nonce"])
            if key != self._oracle_key:
                self._oracle_key = key
                self.engine.reset(arduino["header"], arduino["nonce"])
                self.v_custom_target.set("[SECURE]")
                self._arduino_pow = arduino
                self._refresh_registers()
                self._check_complete()

        def odds(bits):
            return f"1 in {2 ** bits:,}" if bits <= 20 else f"1 in 2^{bits}"

        self.v_pow_diff.set(
            f"Difficulty: {snap['diff']} bits   |   blind-guess odds: "
            f"nonce {odds(snap['diff'])}, "
            f"hash {odds(PASSCODE_BITS - snap['diff'])}"
        )

        times = []
        if arduino is None:
            for box in (
                self.v_pow_header,
                self.v_pow_nonce,
                self.v_pow_hash,
                self.v_pow_padded,
                self.v_pow_padded_nonce,
                self.v_pow_padded_hash,
                self.v_pow_verify,
            ):
                box.set("...")
        else:
            check = arduino["verify"]
            message = check["message"].decode("utf-8", errors="replace")

            self.v_pow_header.set(arduino["header"].decode("utf-8", errors="replace"))
            self.v_pow_nonce.set(str(arduino["nonce"]))
            self.v_pow_hash.set(f"{arduino['hash']:064x}")
            self.v_pow_padded.set(format_blocks(arduino["padded"]))
            self.v_pow_padded_nonce.set(format_blocks(arduino["padded_nonce"]))
            self.v_pow_padded_hash.set(format_blocks(arduino["padded_hash"]))
            times.append(f"Miner {arduino['elapsed']:.3f} s")

            if check["ok"]:
                verdict = (
                    f"OK  SHA256(\"{message}\") equals the reported hash, "
                    f"{check['zero_bits']} leading zero bits "
                    f">= {snap['diff']} required, padded message, nonce and "
                    "hash are valid SHA-256 padding"
                )
            elif not check["hash_ok"]:
                verdict = f"FAILED  SHA256(\"{message}\") is {check['computed']:064x}, not reported hash"
            elif not check["padded_ok"]:
                verdict = "FAILED  a reported padded field is not valid SHA-256 padding"
            else:
                verdict = f"FAILED  only {check['zero_bits']} leading zero bits, {snap['diff']} required"

            self.v_pow_verify.set(verdict)

        if predicted is None:
            self.v_pow_oracle_nonce.set("...")
            self.v_pow_oracle_hash.set("...")
        else:
            self.v_pow_oracle_nonce.set(str(predicted["nonce"]))
            self.v_pow_oracle_hash.set(f"{predicted['hash']:064x}")
            times.append(f"Oracle {predicted['elapsed']:.3f} s")

        self.v_pow_times.set("   |   ".join(times) if times else "-")

        if snap["predicted_padded"] is None:
            self.v_pow_oracle_padded.set("..." if predicted is None else "waiting for the header")
        else:
            self.v_pow_oracle_padded.set(format_blocks(snap["predicted_padded"]))

        guess_check = predicted["verify"] if predicted is not None else None

        if predicted is None:
            self.v_pow_oracle_verify.set("...")
        elif guess_check is None:
            self.v_pow_oracle_verify.set("waiting for the header")
        else:
            message = guess_check["message"].decode("utf-8", errors="replace")
            detail = (
                f"\nSHA256(\"{message}\") = "
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

            self.v_pow_oracle_verify.set(verdict + detail)

        if snap["match"] is None:
            match_text = ""
        elif snap["match"]:
            match_text = "MATCH: oracle nonce and hash equal the miner's"
        else:
            nonce_text = "nonce equal" if snap["nonce_match"] else "nonce differs"
            match_text = (
                f"NO MATCH: {nonce_text}, {snap['bits_agree']}/256 hash bits agree"
            )

        if snap["padded_agree"] is not None:
            agree, total, same = snap["padded_agree"]
            match_text += "\n" + (
                "PADDED MATCH: oracle's whole padded record equals the miner's"
                if same
                else f"PADDED NO MATCH: {agree}/{total} padded-record bits agree"
            )

        self.v_pow_match.set(match_text.strip())

    def set_custom_target(self):
        text = self.v_custom_target.get()
        if not text or text == "[SECURE]":
            messagebox.showerror("Invalid Input", "Enter the header the oracle should pad.")
            return

        self.engine.reset(text.encode("utf-8"), 0)
        self.v_custom_target.set("[SECURE]")
        self._arduino_pow = None
        self._refresh_registers()
        self._check_complete()
        messagebox.showinfo("Oracle Armed", "Passcode is now the padded record of that header.")

    def randomize_target(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")
        self._arduino_pow = None
        self._refresh_registers()
        self._check_complete()

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

    def _draw_hex_rows(self, target_value, guess_value, pooled_value, revealed):
        is_exposed = revealed or self.engine.converged
        active_target = target_value if is_exposed else 0
        depth = self.engine.bit_depth
        width = depth // 4

        key = (is_exposed, active_target, guess_value, pooled_value, depth, self.engine.step_count)
        if key == self._register_key:
            return

        self._register_key = key
        rows = [
            ("TARGET", active_target, "target", not is_exposed),
            ("GUESS", guess_value, "guess", False),
            ("POOLED", pooled_value, "pooled", False),
        ]

        parts = []
        for label, value, tag, hidden in rows:
            parts.append((label + "\n", "label"))
            parts.append(("[HIDDEN]" if hidden else f"{value:0{width}X}", tag))
            parts.append(("\n", None))

        parts.append((f"{depth}-bit padded record | {self.engine.step_count}/{depth} parallel reconstruction steps", None))
        self.register_text.set_tagged(parts)

    def _poll(self):
        try:
            for _ in range(100):
                event = self.q.get_nowait()

                if event[0] == "frame":
                    data = event[1]
                    self.photo = ppm_photo(data["preview"])
                    self.cam_canvas.itemconfig(self.cam_image_id, image=self.photo)
                    self._draw_hex_rows(data["target"], data["guess"], data["pooled"], self.v_reveal.get())
                    self._check_complete()
                    self.status_label.set(f"Status: {data['status']} | Solved Bits: {data['steps']}/{self.engine.bit_depth}")

                elif event[0] == "arduino":
                    self.v_link.set(f"Miner: {event[1]}")

                elif event[0] == "pow_update":
                    self._show_pow(event[1])

                elif event[0] == "arduino_error":
                    messagebox.showerror("Miner Error", event[1])

                elif event[0] == "camera_started":
                    self._on_camera_started()

                elif event[0] == "done":
                    self.camera_ready = False

                elif event[0] == "error":
                    messagebox.showerror("Camera Error", event[1])

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
