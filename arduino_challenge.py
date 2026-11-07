"""
Bit-Pooling Oracle Lock Box Lab
Python + Arduino challenge-response system.

Protocol:
    Arduino -> PC:  CHALLENGE:<0-1023>\n
    PC -> Arduino:  GUESS:<0-1023>\n
    Arduino -> PC:  DISARMED\n or REJECTED\n

The Arduino emits a new random challenge every five seconds. The Python
bit-pooling oracle reconstructs that challenge and returns it. The Arduino
disarms only when the returned value matches the still-current challenge.
"""

import asyncio
import queue
import random
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
# SAFE TIME DELAY
# ==============================================================================

class SafeDelay:
    """
    Thread-safe, validated delay before the oracle begins responding to a
    new Arduino challenge.

    Valid range: 0 to 10 seconds. This must remain shorter than the Arduino's
    five-second challenge window.
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

        return min(max(value, SafeDelay.MIN_DELAY_S), SafeDelay.MAX_DELAY_S)

    def get(self) -> float:
        with self._lock:
            return self._delay_s

    def set(self, value) -> float:
        with self._lock:
            self._delay_s = self._clamp(value)
            return self._delay_s


# ==============================================================================
# BIT-POOLING ORACLE
# ==============================================================================

class BitPoolingOracle:
    """
    Stores a hidden target and answers successive higher-or-equal queries.

    The search engine reconstructs the target one bit at a time, MSB first.
    """

    def __init__(self, bit_depth: int = 10, target=None):
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer.")

        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        if target is None:
            target = random.randint(0, self.max_val)

        self._target = int(np.clip(int(target), 0, self.max_val))

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
# BIT-POOLING SEARCH ENGINE
# ==============================================================================

class BitPoolingSearchEngine:
    def __init__(self, bit_depth: int = 10):
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer.")

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
        self.oracle = BitPoolingOracle(self.bit_depth, new_target)
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
                self.last_status = f"LOCKED & REVEALED AT {self.current_guess}"
            else:
                self.last_status = "RECONSTRUCTION MISMATCH"

    def _update_registers(self):
        self.guess_reg = self.current_guess

        try:
            self.target_reg = self.oracle.target
        except PermissionError:
            self.target_reg = 0

        self.pooled_register = self.oracle.pool_step(self.current_guess)


# ==============================================================================
# ARDUINO LOCK BOX LINK
# ==============================================================================

class ArduinoLockBox:
    """
    Serial link to an Arduino lock box.

    Expected protocol:
        Arduino -> PC: CHALLENGE:<0-1023>\\n
        PC -> Arduino: GUESS:<0-1023>\\n
        Arduino -> PC: DISARMED\\n or REJECTED\\n
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. Run: pip install pyserial"
            )

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self._lock = threading.Lock()

    @staticmethod
    def list_ports():
        if serial_tools is None:
            return []

        return [device.device for device in serial_tools.list_ports.comports()]

    def close(self):
        with self._lock:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def read_challenge(self):
        """
        Reads one CHALLENGE line.

        Returns:
            int: challenge value, or None if no complete line arrived.
        """
        with self._lock:
            while True:
                raw = self.ser.readline()

                if not raw:
                    return None

                line = raw.decode("ascii", errors="ignore").strip()

                if line.startswith("CHALLENGE:"):
                    try:
                        return int(line.split(":", 1)[1])
                    except ValueError:
                        continue

    def send_guess(self, guess: int) -> str:
        """Sends the oracle guess and reads the Arduino response."""
        message = f"GUESS:{int(guess)}\n"

        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write(message.encode("ascii"))
            self.ser.flush()

            response = self.ser.readline().decode(
                "ascii", errors="ignore"
            ).strip()

        return response.upper()


def handle_arduino_challenge(
    lock_box: ArduinoLockBox,
    engine: BitPoolingSearchEngine,
    challenge: int,
    delay_s: float = 0.0,
):
    """
    Receives one Arduino challenge, lets the oracle reconstruct it, and sends
    the result back for validation.
    """
    engine.reset(challenge)

    if delay_s > 0:
        time.sleep(delay_s)

    while not engine.converged and engine.step_count < engine.bit_depth:
        engine.step()

    guess = engine.current_guess
    response = lock_box.send_guess(guess)

    return {
        "challenge": challenge,
        "guess": guess,
        "response": response,
        "disarmed": response == "DISARMED",
    }


class ArduinoThread(threading.Thread):
    def __init__(self, port, ctl, out, engine, safe_delay):
        super().__init__(daemon=True)
        self.port = port
        self.ctl = ctl
        self.out = out
        self.engine = engine
        self.safe_delay = safe_delay

    def run(self):
        lock_box = None

        try:
            lock_box = ArduinoLockBox(self.port)
            self.out.put(("arduino", "Connected to lock box."))

            while not self.ctl["stop"]:
                challenge = lock_box.read_challenge()

                if challenge is None:
                    continue

                if not 0 <= challenge <= self.engine.max_val:
                    self.out.put((
                        "arduino",
                        f"Ignored out-of-range challenge: {challenge}",
                    ))
                    continue

                result = handle_arduino_challenge(
                    lock_box,
                    self.engine,
                    challenge,
                    self.safe_delay.get(),
                )

                self.out.put(("arduino_result", result))

        except Exception as e:
            self.out.put((
                "arduino_error",
                f"{type(e).__name__}: {e}",
            ))

        finally:
            if lock_box is not None:
                lock_box.close()

            self.out.put(("arduino", "Lock-box connection closed."))


# ==============================================================================
# CAMERA SOURCE
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

        t_prev = time.perf_counter()
        frame_index = 0
        previous_even = False

        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)

                if gray is None:
                    out.put(("info", "Camera stopped delivering frames."))
                    break

                now = time.perf_counter()
                dt = min(max(now - t_prev, 1e-3), 0.25)
                t_prev = now

                small = cv2.resize(
                    gray, (128, 96), interpolation=cv2.INTER_AREA
                )
                motion_detected = frame_index > 5 and np.std(small) > 5.0
                motion_value = (
                    int(np.std(small) * 10) if motion_detected else 0
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
                    interpolation=cv2.INTER_AREA
                ).copy()

                if ctl.get("ir_heatmap", False):
                    preview[:, :, 0] = np.clip(
                        preview[:, :, 0].astype(int) + 80, 0, 255
                    )
                    preview[:, :, 2] = np.clip(
                        preview[:, :, 2].astype(int) - 50, 0, 255
                    )

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

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Bit-Pooling Oracle Lock Box Lab")

        self.q = queue.Queue()
        self.safe_delay = SafeDelay(SafeDelay.DEFAULT_DELAY_S)

        self.ctl = {
            "stop": False,
            "ir_heatmap": False,
            "auto_step": True,
            "force_reveal": False,
            "safe_delay": self.safe_delay,
        }

        self.engine = BitPoolingSearchEngine(bit_depth=10)
        self.camera_thread = None
        self.arduino_thread = None

        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.after(15, self._poll)

    def _build_ui(self):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Laboratory controls
        control_frame = ttk.LabelFrame(main_frame, text="Laboratory Controls")
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            control_frame, text="Start Camera", command=self.start_camera
        ).pack(side="left", padx=5, pady=5)

        ttk.Button(
            control_frame, text="Stop Camera", command=self.stop_camera
        ).pack(side="left", padx=5, pady=5)

        ttk.Button(
            control_frame, text="Step Bit Pool", command=self.step_search
        ).pack(side="left", padx=5, pady=5)

        ttk.Button(
            control_frame, text="Reset Pool", command=self.reset_search
        ).pack(side="left", padx=5, pady=5)

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="IR Thermal Heatmap",
            variable=self.v_ir,
            command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get()),
        ).pack(side="left", padx=10)

        # Arduino lock-box controls
        arduino_frame = ttk.LabelFrame(
            main_frame, text="Arduino Lock Box — Challenge Response"
        )
        arduino_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(arduino_frame, text="Serial port:").pack(
            side="left", padx=5, pady=5
        )

        self.v_arduino_port = tk.StringVar(value="COM3")
        self.e_arduino_port = ttk.Entry(
            arduino_frame, width=12, textvariable=self.v_arduino_port
        )
        self.e_arduino_port.pack(side="left", padx=2, pady=5)

        ttk.Button(
            arduino_frame, text="List Ports", command=self.list_arduino_ports
        ).pack(side="left", padx=2, pady=5)

        ttk.Button(
            arduino_frame, text="Connect Lock Box", command=self.connect_arduino
        ).pack(side="left", padx=10, pady=5)

        ttk.Button(
            arduino_frame, text="Disconnect", command=self.disconnect_arduino
        ).pack(side="left", padx=2, pady=5)

        # Safe delay
        delay_frame = ttk.LabelFrame(
            main_frame, text="Oracle Response Delay (0-4 seconds)"
        )
        delay_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(delay_frame, text="Delay before response (s):").pack(
            side="left", padx=5, pady=5
        )

        self.v_delay = tk.StringVar(value=f"{SafeDelay.DEFAULT_DELAY_S:.3f}")
        self.e_delay = ttk.Entry(
            delay_frame, width=10, textvariable=self.v_delay
        )
        self.e_delay.pack(side="left", padx=2, pady=5)

        ttk.Button(
            delay_frame, text="Apply Delay", command=self.apply_delay
        ).pack(side="left", padx=2, pady=5)

        ttk.Label(
            delay_frame,
            text="Keep below 5 s so the response arrives before the next challenge.",
            foreground="#88a0c0",
        ).pack(side="left", padx=10)

        # Hidden target controls
        target_frame = ttk.LabelFrame(
            main_frame, text="Local Oracle Target (manual testing)"
        )
        target_frame.pack(fill="x", padx=5, pady=5)

        self.target_label = ttk.Label(
            target_frame, text=f"Target (0-{self.engine.max_val}):"
        )
        self.target_label.pack(side="left", padx=5, pady=5)

        self.v_custom_target = tk.StringVar(value="[SECURE]")
        self.e_custom_target = ttk.Entry(
            target_frame, width=10, textvariable=self.v_custom_target
        )
        self.e_custom_target.pack(side="left", padx=5, pady=5)

        ttk.Button(
            target_frame, text="Set Custom", command=self.set_custom_target
        ).pack(side="left", padx=2, pady=5)

        ttk.Button(
            target_frame, text="Randomize Target", command=self.randomize_target
        ).pack(side="left", padx=5, pady=5)

        self.v_reveal = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            target_frame,
            text="Force Reveal Local Target",
            variable=self.v_reveal,
            command=lambda: self.ctl.update(force_reveal=self.v_reveal.get()),
        ).pack(side="left", padx=15, pady=5)

        # Camera and registers
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        camera_frame = ttk.LabelFrame(
            display_frame, text="Optoelectronic Circuit Camera Viewport"
        )
        camera_frame.pack(side="left", padx=5, pady=5)

        self.cam_canvas = tk.Canvas(
            camera_frame, width=320, height=240, bg="black"
        )
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_image_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        right_panel = ttk.Frame(display_frame)
        right_panel.pack(
            side="right", fill="both", expand=True, padx=5, pady=5
        )

        register_frame = ttk.LabelFrame(
            right_panel, text="Bit-Pooling Hardware Registers"
        )
        register_frame.pack(fill="x", padx=5, pady=5)

        self.register_canvas = tk.Canvas(
            register_frame,
            width=520,
            height=170,
            bg="#121414",
            highlightthickness=0,
        )
        self.register_canvas.pack(padx=5, pady=5)

        self.status_label = ttk.Label(
            register_frame,
            text="Status: IDLE",
            font=("Consolas", 11),
            foreground="#00e5ff",
        )
        self.status_label.pack(anchor="w", padx=10, pady=5)

        # Pooling settings
        settings_frame = ttk.LabelFrame(
            right_panel, text="Pooling & Algorithm Settings"
        )
        settings_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(settings_frame, text="Pooling Operator:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )

        self.operator_combo = ttk.Combobox(
            settings_frame,
            values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"],
            state="readonly",
        )
        self.operator_combo.set("POPCOUNT")
        self.operator_combo.grid(row=0, column=1, padx=5, pady=5)
        self.operator_combo.bind(
            "<<ComboboxSelected>>", self._update_pool_operator
        )

        ttk.Label(settings_frame, text="Kernel Size:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
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
        self.kernel_spin.grid(row=1, column=1, padx=5, pady=5)

    def start_camera(self):
        if self.camera_thread is not None and self.camera_thread.is_alive():
            messagebox.showwarning(
                "Already Running", "The camera lab is already running."
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

    def apply_delay(self):
        delay = self.safe_delay.set(self.v_delay.get())
        self.v_delay.set(f"{delay:.3f}")

        messagebox.showinfo(
            "Delay Applied",
            f"Oracle response delay set to {delay:.3f} seconds.",
        )

    def list_arduino_ports(self):
        try:
            ports = ArduinoLockBox.list_ports()
        except Exception as e:
            messagebox.showerror("Serial Error", str(e))
            return

        if not ports:
            messagebox.showwarning(
                "No Serial Ports", "No serial ports were found."
            )
            return

        self.v_arduino_port.set(ports[0])
        messagebox.showinfo("Serial Ports", "\n".join(ports))

    def connect_arduino(self):
        if (
            self.arduino_thread is not None
            and self.arduino_thread.is_alive()
        ):
            messagebox.showwarning(
                "Already Connected", "The lock box is already connected."
            )
            return

        port = self.v_arduino_port.get().strip()

        if not port:
            messagebox.showerror(
                "Missing Port", "Enter a serial port such as COM3."
            )
            return

        self.ctl["stop"] = False
        self.arduino_thread = ArduinoThread(
            port, self.ctl, self.q, self.engine, self.safe_delay
        )
        self.arduino_thread.start()

    def disconnect_arduino(self):
        if self.arduino_thread is not None:
            self.ctl["stop"] = True

    def set_custom_target(self):
        try:
            value = int(self.v_custom_target.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Please enter a valid integer."
            )
            return

        if 0 <= value <= self.engine.max_val:
            self.engine.reset(value)
            messagebox.showinfo(
                "Target Set",
                "New local target configured for bit pooling.",
            )
        else:
            messagebox.showerror(
                "Invalid Range",
                f"Target must be between 0 and {self.engine.max_val}.",
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

    def _draw_bit_rows(self, target_value, guess_value, pooled_value, revealed):
        self.register_canvas.delete("all")

        bit_depth = self.engine.bit_depth
        is_exposed = revealed or self.engine.converged
        active_target = target_value if is_exposed else 0

        rows = [
            ("TARGET", active_target, "#00e676", not is_exposed),
            ("GUESS ", guess_value, "#ffb300", False),
            ("POOLED", pooled_value, "#b388ff", False),
        ]

        y_offset = 10

        for label, value, colour, hidden in rows:
            self.register_canvas.create_text(
                10,
                y_offset + 10,
                text=label,
                fill="#88a0c0",
                anchor="w",
                font=("Consolas", 10, "bold"),
            )

            x = 90

            for bit_index in range(bit_depth - 1, -1, -1):
                bit_state = (value >> bit_index) & 1

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

                text = "?" if hidden else str(bit_state)
                text_colour = (
                    "#000000" if bit_state and not hidden else "#708090"
                )

                self.register_canvas.create_text(
                    x + 9,
                    y_offset + 9,
                    text=text,
                    fill=text_colour,
                    font=("Consolas", 8, "bold"),
                )

                x += 22

            number_text = "[HIDDEN]" if hidden else f"[{value}]"
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
                    self.photo = ppm_photo(data["preview"])
                    self.cam_canvas.itemconfig(
                        self.cam_image_id, image=self.photo
                    )

                    self._draw_bit_rows(
                        data["target"],
                        data["guess"],
                        data["pooled"],
                        self.v_reveal.get(),
                    )

                    delay = self.safe_delay.get()
                    self.status_label.config(
                        text=(
                            f"Status: {data['status']} | "
                            f"Bit Step: {data['steps']}/"
                            f"{self.engine.bit_depth} | "
                            f"Delay: {delay:.3f}s"
                        )
                    )

                elif event[0] == "arduino":
                    self.status_label.config(text=f"Lock Box: {event[1]}")

                elif event[0] == "arduino_result":
                    result = event[1]

                    if result["disarmed"]:
                        status = (
                            f"DISARMED | Challenge {result['challenge']} | "
                            f"Oracle guess {result['guess']}"
                        )
                    else:
                        status = (
                            f"REJECTED | Challenge {result['challenge']} | "
                            f"Oracle guess {result['guess']} | "
                            f"{result['response']}"
                        )

                    self.status_label.config(text=status)

                elif event[0] == "arduino_error":
                    messagebox.showerror("Arduino Error", event[1])

                elif event[0] == "error":
                    messagebox.showerror("Camera Error", event[1])

        except queue.Empty:
            pass

        self.window.after(30, self._poll)

    def _close(self):
        self.ctl["stop"] = True
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("1000x760")
    root.mainloop()
