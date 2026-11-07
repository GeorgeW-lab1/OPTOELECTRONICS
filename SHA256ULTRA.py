
"""
Bit-Pooling Oracle Lock Box Lab
Python + Emulated Arduino SHA-256 Proof-of-Work Miner.

Features:
    - Minimal SHA-256 message padding.
    - Strict MSB-first higher/lower bit reconstruction.
    - 256-bit SHA-256 proof-of-work verification.
    - Emulated Arduino nonce miner.
    - Camera-driven bit-sector activation.
    - Bit-pooling registers.
    - Hidden target register until convergence.
    - Automatic message extraction.
    - Exact padding verification.
    - Tkinter GUI with OpenCV camera.

Install:
    pip install opencv-python numpy

Python:
    3.10+
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


# =============================================================================
# CONFIGURATION
# =============================================================================

PASSCODE_BITS = 256

MIN_DIFF = 0
MAX_DIFF = 256

DEFAULT_HEADER = "arduino-sha256"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

GRID_COLUMNS = 32

CAMERA_THRESHOLD = 3.0


# =============================================================================
# MINIMAL SHA-256 PADDING
# =============================================================================

def sha256_pad_minimal(message: bytes) -> bytes:
    """
    Construct the minimum valid SHA-256 padded message.

    SHA-256 processes blocks of 64 bytes.

    Padding:
        1. Original message.
        2. One 0x80 byte.
        3. Minimum number of zero bytes.
        4. 64-bit big-endian message bit length.
    """

    if not isinstance(message, bytes):
        raise TypeError("Message must be bytes.")

    msg_len = len(message)

    if msg_len * 8 >= (1 << 64):
        raise ValueError("Message exceeds SHA-256 length encoding.")

    total_len = ((msg_len + 9 + 63) // 64) * 64

    zero_count = total_len - msg_len - 9

    return (
        message
        + b"\x80"
        + b"\x00" * zero_count
        + (msg_len * 8).to_bytes(8, "big")
    )


def sha256_unpad_minimal(padded: bytes):
    """
    Validate and remove minimal SHA-256 padding.

    Returns:
        Original message bytes if valid.
        None if invalid.
    """

    if not isinstance(padded, bytes):
        return None

    if len(padded) < 64:
        return None

    if len(padded) % 64 != 0:
        return None

    bit_length = int.from_bytes(padded[-8:], "big")

    if bit_length % 8 != 0:
        return None

    byte_len = bit_length // 8

    if byte_len + 9 > len(padded):
        return None

    # Mandatory SHA-256 delimiter.
    if padded[byte_len] != 0x80:
        return None

    # All remaining padding bytes must be zero.
    if any(padded[byte_len + 1:-8]):
        return None

    message = padded[:byte_len]

    # Reconstruct and compare the complete padded block.
    if sha256_pad_minimal(message) != padded:
        return None

    return message


def get_padded_message(header: bytes, nonce: int) -> bytes:
    return sha256_pad_minimal(
        header + str(nonce).encode("ascii")
    )


# =============================================================================
# MINIMAL MESSAGE ORACLE
# =============================================================================

class MinimalMessageOracle:

    def __init__(self, header=None, nonce=None):

        if header is None:
            header = DEFAULT_HEADER

        if isinstance(header, str):
            header = header.encode("utf-8")

        if nonce is None:
            nonce = secrets.randbelow(1 << 32)

        self.header = bytes(header)
        self.nonce = int(nonce)

        self.padded_message = get_padded_message(
            self.header,
            self.nonce
        )

        self.bit_depth = len(self.padded_message) * 8

        self.max_val = (1 << self.bit_depth) - 1

        self.kernel_size = 4
        self.stride = 2

        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.pooled_bits = 0

        self.is_done = False

    def _padded_value(self):
        return int.from_bytes(
            self.padded_message,
            "big"
        )

    @property
    def _target(self):
        return self._padded_value()

    def query_bit(self, trial_val: int) -> int:
        """
        Higher/lower oracle.

        Returns:
            1 if target >= trial.
            0 otherwise.

        Reconstruction MUST proceed MSB-first.
        """

        if self.is_done:
            return 0

        return int(self._target >= trial_val)

    def pool_step(self, val: int) -> int:
        """
        Bit-pooling register simulation.

        This register is a visualization of pooled bits.
        It does not replace the higher/lower reconstruction.
        """

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

                output_bit = int(
                    result >= self.schmitt_trigger
                )

            elif self.pool_op == "AND":

                output_bit = int(all(block))

            elif self.pool_op == "OR":

                output_bit = int(any(block))

            elif self.pool_op == "XOR":

                output_bit = sum(block) % 2

            elif self.pool_op == "MAJORITY":

                output_bit = int(
                    sum(block) > len(block) / 2
                )

            else:

                output_bit = block[0]

            pooled_out.append(output_bit)

        output_value = 0

        for index, bit in enumerate(pooled_out):
            output_value |= bit << index

        return output_value

    @property
    def target(self):

        if not self.is_done:
            raise PermissionError(
                "SECURITY VIOLATION: Target is classified."
            )

        return self._target


# =============================================================================
# PARALLEL SEARCH ENGINE
# =============================================================================

class ParallelMinimalSearchEngine:

    def __init__(self):

        self._lock = threading.RLock()

        self.oracle = MinimalMessageOracle()

        self.bit_depth = self.oracle.bit_depth
        self.max_val = self.oracle.max_val

        self.current_guess = 0
        self.solved_bits_mask = 0

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

    def reset(self, header=None, nonce=None):

        with self._lock:

            self.oracle = MinimalMessageOracle(
                header,
                nonce
            )

            self.bit_depth = self.oracle.bit_depth
            self.max_val = self.oracle.max_val

            self.oracle.pool_op = self.pool_op
            self.oracle.kernel_size = self.kernel_size
            self.oracle.stride = self.stride

            self.current_guess = 0
            self.solved_bits_mask = 0

            self.history = [0]

            self.converged = False

            self.last_status = "READY"

            self._update_registers()

    def solve_bit_at_index(self, bit_index: int):

        with self._lock:

            if self.converged:
                return

            if not 0 <= bit_index < self.bit_depth:
                return

            # CRITICAL:
            # Only solve the next unsolved bit.
            #
            # The higher/lower oracle is a binary
            # search over the complete padded integer.
            #
            # Therefore reconstruction must be MSB-first.

            solved_count = self.solved_bits_mask.bit_count()

            if bit_index != solved_count:
                return

            shift = self.bit_depth - 1 - bit_index

            # Test whether this bit is one.
            trial_value = (
                self.current_guess | (1 << shift)
            )

            bit = self.oracle.query_bit(trial_value)

            if bit:

                self.current_guess |= (1 << shift)

            self.solved_bits_mask |= (1 << bit_index)

            self.oracle.pooled_bits = self.current_guess

            self.history.append(self.current_guess)

            self._update_registers()

            solved_count = self.solved_bits_mask.bit_count()

            self.last_status = (
                f"PARALLEL SOLVING (MINIMAL): "
                f"{solved_count}/{self.bit_depth} sectors active"
            )

            # All sectors reconstructed.
            if solved_count == self.bit_depth:

                self.oracle.is_done = True

                self.converged = True

                if self.current_guess == self.oracle._target:

                    self.last_status = (
                        "MINIMAL PADDING RECONSTRUCTED & VERIFIED"
                    )

                else:

                    self.last_status = (
                        "RECONSTRUCTION MISMATCH"
                    )

    def step(self):

        with self._lock:

            if self.converged:
                return

            next_index = self.solved_bits_mask.bit_count()

            if next_index < self.bit_depth:

                self.solve_bit_at_index(next_index)

    def _update_registers(self):

        self.guess_reg = self.current_guess

        try:

            self.target_reg = self.oracle.target

        except PermissionError:

            self.target_reg = 0

        self.pooled_register = self.oracle.pool_step(
            self.current_guess
        )

    def extract_message(self):

        with self._lock:

            steps = self.solved_bits_mask.bit_count()

            total_bytes = self.bit_depth // 8

            if steps < self.bit_depth:

                known_bytes = steps // 8

                return (
                    f"Reconstructing... "
                    f"({known_bytes}/{total_bytes} bytes)"
                )

            guessed_bytes = self.current_guess.to_bytes(
                total_bytes,
                "big"
            )

            unpadded = sha256_unpad_minimal(
                guessed_bytes
            )

            if unpadded is None:

                return "INVALID MINIMAL SHA-256 PADDING"

            # Additional integrity verification.
            expected = get_padded_message(
                self.oracle.header,
                self.oracle.nonce
            )

            if guessed_bytes != expected:

                return "PADDING INTEGRITY FAILURE"

            try:

                return unpadded.decode("utf-8")

            except UnicodeDecodeError:

                return unpadded.decode(
                    "utf-8",
                    errors="replace"
                )


# =============================================================================
# EMULATED ARDUINO SHA-256 MINER
# =============================================================================

class EmulatedArduinoMiner:

    def __init__(self):

        self.input_data = DEFAULT_HEADER

        self.difficulty = 12

        self.mining = False

        self.nonce = 0

        self.hashes = 0

        self.last_hash = None

        self._lock = threading.Lock()

    def close(self):

        with self._lock:
            self.mining = False

    def query_header(self):

        return self.input_data.encode("utf-8")

    def send_diff(self, diff: int):

        if not MIN_DIFF <= diff <= MAX_DIFF:
            raise ValueError("Difficulty outside allowed range.")

        with self._lock:

            self.difficulty = int(diff)

            self.nonce = 0

            self.hashes = 0

            self.last_hash = None

            self.mining = True

    def step_mining(self):

        with self._lock:

            if not self.mining:
                return None

            for _ in range(500):

                nonce_text = str(self.nonce).encode("ascii")

                message = (
                    self.input_data.encode("utf-8")
                    + nonce_text
                )

                digest = hashlib.sha256(
                    message
                ).digest()

                computed = int.from_bytes(
                    digest,
                    "big"
                )

                zero_bits = (
                    PASSCODE_BITS - computed.bit_length()
                )

                self.hashes += 1

                self.last_hash = digest.hex()

                if zero_bits >= self.difficulty:

                    self.mining = False

                    padded = sha256_pad_minimal(message)

                    return {
                        "nonce": self.nonce,
                        "hash": computed,
                        "digest": digest.hex(),
                        "padded_msg": padded,
                        "header": self.input_data.encode("utf-8"),
                        "hashes": self.hashes,
                        "difficulty": self.difficulty
                    }

                self.nonce += 1

                if self.nonce > 0xFFFFFFFFFFFFFFFF:

                    self.mining = False

                    break

        return None


# =============================================================================
# ARDUINO COMMUNICATION THREAD
# =============================================================================

class ArduinoThread(threading.Thread):

    def __init__(self, ctl, out):

        super().__init__(daemon=True)

        self.ctl = ctl
        self.out = out

        self.commands = queue.Queue()

        self.miner = EmulatedArduinoMiner()

    def request_diff(self, diff: int):

        self.commands.put(diff)

    def run(self):

        self.out.put((
            "arduino",
            "Connected (Software Emulation)."
        ))

        while not self.ctl["arduino_stop"]:

            try:

                while True:

                    diff = self.commands.get_nowait()

                    self.miner.send_diff(diff)

                    self.out.put((
                        "miner_started",
                        (
                            self.miner.query_header(),
                            diff
                        )
                    ))

            except queue.Empty:
                pass

            result = self.miner.step_mining()

            if result is not None:

                self.out.put((
                    "pow_found",
                    result
                ))

            else:

                time.sleep(0.005)

        self.miner.close()

        self.out.put((
            "arduino",
            "Disconnected."
        ))


# =============================================================================
# CAMERA
# =============================================================================

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

        cap = cv2.VideoCapture(
            index,
            backend
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        if not cap.isOpened():

            cap.release()

            raise RuntimeError(
                f"Cannot open camera {index}."
            )

        ok, frame = cap.read()

        if not ok:

            cap.release()

            raise RuntimeError(
                f"Cannot read camera {index}."
            )

        self.cap = cap

        self.rgb = None

    def read(self):

        ok, bgr = self.cap.read()

        if not ok:
            return None

        self.rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

        return cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        )

    def close(self):

        self.cap.release()


def ppm_photo(rgb: np.ndarray):

    h, w, _ = rgb.shape

    data = (
        b"P6 %d %d 255\n" % (w, h)
        + np.ascontiguousarray(
            rgb,
            dtype=np.uint8
        ).tobytes()
    )

    return tk.PhotoImage(
        data=data,
        format="PPM"
    )


# =============================================================================
# CAMERA GRID SIMULATOR
# =============================================================================

class ParallelMinimalGridSimThread(threading.Thread):

    def __init__(
        self,
        cam_index,
        ctl,
        out,
        engine
    ):

        super().__init__(daemon=True)

        self.cam_index = cam_index

        self.ctl = ctl

        self.out = out

        self.engine = engine

    def run(self):

        asyncio.run(self._run())

    async def _run(self):

        ctl = self.ctl

        out = self.out

        engine = self.engine

        try:

            src = Source(self.cam_index)

        except RuntimeError as e:

            out.put(("error", str(e)))

            out.put(("done", None))

            return

        out.put(("camera_started", None))

        try:

            while not ctl["stop"]:

                gray = await asyncio.to_thread(
                    src.read
                )

                if gray is None:
                    break

                h, w = gray.shape

                bit_depth = engine.bit_depth

                grid_cols = GRID_COLUMNS

                grid_rows = (
                    bit_depth + grid_cols - 1
                ) // grid_cols

                cell_h = max(1, h // grid_rows)

                cell_w = max(1, w // grid_cols)

                # -----------------------------------------------------
                # CAMERA ACTIVATION
                # -----------------------------------------------------

                if (
                    ctl.get("auto_step", True)
                    and not engine.converged
                ):

                    # Only the next unsolved sector can activate.
                    # This preserves MSB-first ordering.

                    next_index = (
                        engine.solved_bits_mask.bit_count()
                    )

                    if next_index < bit_depth:

                        r = next_index // grid_cols

                        c = next_index % grid_cols

                        y1 = r * cell_h

                        y2 = min(
                            (r + 1) * cell_h,
                            h
                        )

                        x1 = c * cell_w

                        x2 = min(
                            (c + 1) * cell_w,
                            w
                        )

                        cell = gray[y1:y2, x1:x2]

                        if (
                            cell.size
                            and np.std(cell) > CAMERA_THRESHOLD
                        ):

                            engine.step()

                # -----------------------------------------------------
                # CAMERA DISPLAY
                # -----------------------------------------------------

                rgb = (
                    src.rgb
                    if src.rgb is not None
                    else cv2.cvtColor(
                        gray,
                        cv2.COLOR_GRAY2RGB
                    )
                )

                preview = cv2.resize(
                    rgb,
                    (320, 240),
                    interpolation=cv2.INTER_AREA
                ).copy()

                prev_h, prev_w, _ = preview.shape

                p_cell_h = prev_h / grid_rows

                p_cell_w = prev_w / grid_cols

                solved_mask = engine.solved_bits_mask

                for r in range(grid_rows):

                    for c in range(grid_cols):

                        bit_index = (
                            r * grid_cols + c
                        )

                        if bit_index >= bit_depth:
                            continue

                        solved = (
                            solved_mask & (1 << bit_index)
                        ) != 0

                        x_start = int(c * p_cell_w)

                        y_start = int(r * p_cell_h)

                        x_end = int((c + 1) * p_cell_w)

                        y_end = int((r + 1) * p_cell_h)

                        color = (
                            (0, 255, 0)
                            if solved
                            else (60, 60, 60)
                        )

                        cv2.rectangle(
                            preview,
                            (x_start, y_start),
                            (x_end, y_end),
                            color,
                            1
                        )

                with engine._lock:

                    data = {
                        "preview": preview,
                        "target": (
                            engine.oracle._target
                            if (
                                engine.converged
                                or ctl.get("force_reveal", False)
                            )
                            else 0
                        ),
                        "guess": engine.current_guess,
                        "pooled": engine.pooled_register,
                        "status": engine.last_status,
                        "steps": engine.solved_bits_mask.bit_count(),
                        "total_bits": engine.bit_depth,
                        "message": engine.extract_message()
                    }

                out.put(("frame", data))

                await asyncio.sleep(0.03)

        finally:

            src.close()

            out.put(("done", None))


# =============================================================================
# SELECTABLE TEXT
# =============================================================================

class SelectableText(tk.Text):

    def __init__(
        self,
        master,
        text="",
        **kw
    ):

        style = ttk.Style()

        kw.setdefault(
            "bg",
            style.lookup("TFrame", "background")
            or master.winfo_toplevel().cget("background")
        )

        kw.setdefault(
            "fg",
            style.lookup("TLabel", "foreground")
            or "black"
        )

        kw.setdefault(
            "font",
            ("Consolas", 10)
        )

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
            **kw
        )

        self._shown = None

        self.bind(
            "<Button-1>",
            lambda e: self.focus_set(),
            add="+"
        )

        self.set(text)

    def value(self):

        return self.get(
            "1.0",
            "end-1c"
        )

    def set(self, text):

        self.set_tagged([
            (str(text), None)
        ])

    def set_tagged(self, parts):

        parts = tuple(parts)

        if parts == self._shown:
            return

        self._shown = parts

        self.configure(state="normal")

        self.delete("1.0", "end")

        for text, tag in parts:

            self.insert(
                "end",
                text,
                (tag,) if tag else ()
            )

        self.configure(state="disabled")

        self._fit()

    def _fit(self):

        try:

            counted = self.count(
                "1.0",
                "end-1c",
                "update",
                "displaylines"
            )

            lines = (
                counted[0]
                if isinstance(counted, (tuple, list))
                else counted
            )

        except (tk.TclError, TypeError):

            lines = self.value().count("\n") + 1

        lines = max(1, int(lines or 1))

        if int(self.cget("height")) != lines:

            self.configure(height=lines)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class App:

    def __init__(self, window):

        self.window = window

        self.window.title(
            "Minimal Padding Bit-Pooling Oracle — SHA-256"
        )

        self.q = queue.Queue()

        self.ctl = {
            "stop": False,
            "auto_step": True,
            "force_reveal": False,
            "arduino_stop": False
        }

        self.engine = ParallelMinimalSearchEngine()

        self.camera_thread = None

        self.arduino_thread = None

        self._register_key = None

        self.photo = None

        self._build_ui()

        self.window.protocol(
            "WM_DELETE_WINDOW",
            self._close
        )

        self.window.after(
            15,
            self._poll
        )

    # -------------------------------------------------------------------------
    # GUI
    # -------------------------------------------------------------------------

    def _build_ui(self):

        main_frame = ttk.Frame(
            self.window,
            padding=10
        )

        main_frame.pack(
            fill="both",
            expand=True
        )

        # CONTROLS

        control_frame = ttk.LabelFrame(
            main_frame,
            text="Controls"
        )

        control_frame.pack(
            fill="x",
            padx=5,
            pady=5
        )

        ttk.Button(
            control_frame,
            text="Start Camera Grid",
            command=self.start_camera
        ).pack(
            side="left",
            padx=5,
            pady=5
        )

        ttk.Button(
            control_frame,
            text="Stop Camera",
            command=self.stop_camera
        ).pack(
            side="left",
            padx=5,
            pady=5
        )

        ttk.Button(
            control_frame,
            text="Reset Engine",
            command=self.reset_search
        ).pack(
            side="left",
            padx=5,
            pady=5
        )

        ttk.Button(
            control_frame,
            text="Connect Emulated Miner",
            command=self.connect_arduino
        ).pack(
            side="left",
            padx=15,
            pady=5
        )

        ttk.Label(
            control_frame,
            text="Diff:"
        ).pack(
            side="left",
            padx=2
        )

        self.v_diff = tk.StringVar(
            value="12"
        )

        ttk.Spinbox(
            control_frame,
            from_=MIN_DIFF,
            to=MAX_DIFF,
            width=4,
            textvariable=self.v_diff
        ).pack(
            side="left",
            padx=2
        )

        ttk.Button(
            control_frame,
            text="Send Difficulty",
            command=self.send_difficulty
        ).pack(
            side="left",
            padx=5
        )

        # MESSAGE OUTPUT

        msg_frame = ttk.LabelFrame(
            main_frame,
            text="Extracted Message Output (Minimal Padding)"
        )

        msg_frame.pack(
            fill="x",
            padx=5,
            pady=5
        )

        self.v_msg_output = SelectableText(
            msg_frame,
            "Reconstructing message...",
            font=("Consolas", 14, "bold"),
            fg="#00e676"
        )

        self.v_msg_output.pack(
            fill="x",
            padx=10,
            pady=5
        )

        # DISPLAY

        display_frame = ttk.Frame(
            main_frame
        )

        display_frame.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        # CAMERA VIEWPORT

        camera_frame = ttk.LabelFrame(
            display_frame,
            text="Camera Viewport (Minimal Bit Sectors)"
        )

        camera_frame.pack(
            side="left",
            padx=5,
            pady=5
        )

        self.cam_canvas = tk.Canvas(
            camera_frame,
            width=320,
            height=240,
            bg="black"
        )

        self.cam_canvas.pack(
            padx=5,
            pady=5
        )

        self.cam_image_id = self.cam_canvas.create_image(
            0,
            0,
            anchor="nw"
        )

        # RIGHT PANEL

        right_panel = ttk.Frame(
            display_frame
        )

        right_panel.pack(
            side="right",
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        register_frame = ttk.LabelFrame(
            right_panel,
            text="Minimal Padded Message Registers"
        )

        register_frame.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        self.register_text = SelectableText(
            register_frame,
            "",
            bg="#121414",
            fg="#d2e0f0",
            width=80,
            height=10
        )

        self.register_text.pack(
            fill="x",
            padx=5,
            pady=5
        )

        self.register_text.tag_configure(
            "label",
            foreground="#88a0c0",
            font=("Consolas", 10, "bold")
        )

        self.register_text.tag_configure(
            "target",
            foreground="#00e676"
        )

        self.register_text.tag_configure(
            "guess",
            foreground="#ffb300"
        )

        self.status_label = SelectableText(
            register_frame,
            "Status: IDLE",
            fg="#00e5ff",
            font=("Consolas", 11)
        )

        self.status_label.pack(
            fill="x",
            padx=10,
            pady=5
        )

    # -------------------------------------------------------------------------
    # CAMERA CONTROL
    # -------------------------------------------------------------------------

    def start_camera(self):

        if (
            self.camera_thread
            and self.camera_thread.is_alive()
        ):
            return

        self.ctl["stop"] = False

        self.camera_thread = ParallelMinimalGridSimThread(
            0,
            self.ctl,
            self.q,
            self.engine
        )

        self.camera_thread.start()

    def stop_camera(self):

        self.ctl["stop"] = True

    # -------------------------------------------------------------------------
    # ENGINE CONTROL
    # -------------------------------------------------------------------------

    def reset_search(self):

        self.engine.reset()

        self._register_key = None

        self._refresh_registers()

        self.v_msg_output.set(
            "Reconstructing message..."
        )

    # -------------------------------------------------------------------------
    # MINER CONTROL
    # -------------------------------------------------------------------------

    def connect_arduino(self):

        if (
            self.arduino_thread
            and self.arduino_thread.is_alive()
        ):
            return

        self.ctl["arduino_stop"] = False

        self.arduino_thread = ArduinoThread(
            self.ctl,
            self.q
        )

        self.arduino_thread.start()

    def send_difficulty(self):

        if not (
            self.arduino_thread
            and self.arduino_thread.is_alive()
        ):

            messagebox.showwarning(
                "Miner",
                "Connect the emulated miner first."
            )

            return

        try:

            diff = int(self.v_diff.get())

            if not MIN_DIFF <= diff <= MAX_DIFF:

                raise ValueError

            self.arduino_thread.request_diff(
                diff
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Difficulty",
                f"Enter a difficulty between "
                f"{MIN_DIFF} and {MAX_DIFF}."
            )

    # -------------------------------------------------------------------------
    # REGISTER DISPLAY
    # -------------------------------------------------------------------------

    def _refresh_registers(self):

        engine = self.engine

        revealed = self.ctl.get(
            "force_reveal",
            False
        )

        with engine._lock:

            exposed = (
                engine.oracle._target
                if engine.converged or revealed
                else 0
            )

            guess = engine.current_guess

            pooled = engine.pooled_register

        self._draw_hex_rows(
            exposed,
            guess,
            pooled,
            revealed
        )

    def _draw_hex_rows(
        self,
        target_value,
        guess_value,
        pooled_value,
        revealed
    ):

        is_exposed = (
            revealed
            or self.engine.converged
        )

        active_target = (
            target_value
            if is_exposed
            else 0
        )

        depth = self.engine.bit_depth

        width = depth // 4

        key = (
            is_exposed,
            active_target,
            guess_value,
            pooled_value,
            self.engine.solved_bits_mask
        )

        if key == self._register_key:
            return

        self._register_key = key

        rows = [
            (
                "TARGET MINIMAL PADDING",
                active_target,
                "target",
                not is_exposed
            ),
            (
                "GUESS MINIMAL PADDING",
                guess_value,
                "guess",
                False
            ),
            (
                "POOLED REGISTER",
                pooled_value,
                "guess",
                False
            )
        ]

        parts = []

        for label, value, tag, hidden in rows:

            parts.append((
                label + "\n",
                "label"
            ))

            parts.append((
                "[HIDDEN]"
                if hidden
                else f"{value:0{width}X}",
                tag
            ))

            parts.append(("\n", None))

        solved = self.engine.solved_bits_mask.bit_count()

        parts.append((
            f"{depth}-bit Minimal Padded Block | "
            f"{solved}/{depth} Sectors Active",
            None
        ))

        self.register_text.set_tagged(
            parts
        )

    # -------------------------------------------------------------------------
    # EVENT POLLING
    # -------------------------------------------------------------------------

    def _poll(self):

        try:

            for _ in range(100):

                event = self.q.get_nowait()

                event_type = event[0]

                data = event[1]

                if event_type == "frame":

                    self.photo = ppm_photo(
                        data["preview"]
                    )

                    self.cam_canvas.itemconfig(
                        self.cam_image_id,
                        image=self.photo
                    )

                    self._refresh_registers()

                    self.status_label.set(
                        f"Status: {data['status']} | "
                        f"Active: {data['steps']}/"
                        f"{data['total_bits']}"
                    )

                    self.v_msg_output.set(
                        data["message"]
                    )

                elif event_type == "pow_found":

                    pow_data = data

                    # Independently verify the SHA-256 result.
                    message = (
                        pow_data["header"]
                        + str(pow_data["nonce"]).encode("ascii")
                    )

                    actual_digest = hashlib.sha256(
                        message
                    ).hexdigest()

                    if actual_digest != pow_data["digest"]:

                        self.status_label.set(
                            "Status: SHA-256 VERIFICATION FAILURE"
                        )

                        continue

                    # Verify proof-of-work difficulty.
                    digest_int = int(
                        actual_digest,
                        16
                    )

                    zero_bits = (
                        PASSCODE_BITS
                        - digest_int.bit_length()
                    )

                    if zero_bits < pow_data["difficulty"]:

                        self.status_label.set(
                            "Status: INVALID PROOF OF WORK"
                        )

                        continue

                    # Verify the padded message.
                    if (
                        sha256_unpad_minimal(
                            pow_data["padded_msg"]
                        ) != message
                    ):

                        self.status_label.set(
                            "Status: INVALID MINIMAL PADDING"
                        )

                        continue

                    # Reset oracle using the actual found nonce.
                    self.engine.reset(
                        pow_data["header"],
                        pow_data["nonce"]
                    )

                    self._register_key = None

                    self._refresh_registers()

                    self.status_label.set(
                        "Status: POW FOUND | "
                        f"Nonce: {pow_data['nonce']} | "
                        f"Difficulty: {pow_data['difficulty']} | "
                        f"Hashes: {pow_data['hashes']}"
                    )

                    self.v_msg_output.set(
                        "PoW verified. Reconstructing padded message..."
                    )

                elif event_type == "miner_started":

                    header, diff = data

                    # The nonce is initially zero.
                    self.engine.reset(
                        header,
                        0
                    )

                    self._register_key = None

                    self._refresh_registers()

                    self.status_label.set(
                        f"Status: MINER STARTED | Difficulty: {diff}"
                    )

                elif event_type == "arduino":

                    self.status_label.set(
                        f"Status: {data}"
                    )

                elif event_type == "camera_started":

                    self.status_label.set(
                        "Status: CAMERA CONNECTED"
                    )

                elif event_type == "error":

                    messagebox.showerror(
                        "Camera Error",
                        data
                    )

                    self.status_label.set(
                        f"Status: {data}"
                    )

                elif event_type == "done":

                    self.status_label.set(
                        "Status: CAMERA STOPPED"
                    )

        except queue.Empty:
            pass

        if self.window.winfo_exists():

            self.window.after(
                30,
                self._poll
            )

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------

    def _close(self):

        self.ctl["stop"] = True

        self.ctl["arduino_stop"] = True

        self.window.destroy()


# =============================================================================
# SELF-TEST
# =============================================================================

def run_self_tests():

    print("\nMINIMAL SHA-256 PADDING SELF-TEST")

    test_messages = [
        b"",
        b"a",
        b"hello",
        b"arduino-sha256",
        b"arduino-sha2560",
        b"A" * 55,
        b"A" * 56,
        b"A" * 63,
        b"A" * 64,
        b"A" * 100
    ]

    for message in test_messages:

        padded = sha256_pad_minimal(message)

        recovered = sha256_unpad_minimal(
            padded
        )

        assert recovered == message

        assert len(padded) % 64 == 0

        assert hashlib.sha256(message).digest() == hashlib.sha256(
            recovered
        ).digest()

        print(
            f"[PASS] {len(message):4d} bytes -> "
            f"{len(padded):4d} padded bytes"
        )

    print("\nHIGHER/LOWER ORACLE SELF-TEST")

    engine = ParallelMinimalSearchEngine()

    engine.reset(
        b"arduino-sha256",
        12345
    )

    for _ in range(engine.bit_depth):

        engine.step()

    assert engine.converged

    recovered_bytes = engine.current_guess.to_bytes(
        engine.bit_depth // 8,
        "big"
    )

    recovered = sha256_unpad_minimal(
        recovered_bytes
    )

    expected = b"arduino-sha25612345"

    assert recovered == expected

    print("[PASS] MSB-first reconstruction")

    print("[PASS] Exact padded-block recovery")

    print("[PASS] Padding validation")

    print("[PASS] Extracted message:", recovered.decode())

    print("\nALL SELF-TESTS PASSED.\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    if "--test" in sys.argv:

        run_self_tests()

    else:

        root = tk.Tk()

        app = App(root)

        root.geometry("1100x700")

        root.mainloop()
