"""
Bit-Pooling Higher-Lower Search & OpenCV Camera Lab
Command-Line Edition

Features:
    - 512-bit higher/lower oracle search
    - OpenCV live camera
    - Motion-triggered bit search
    - Arduino serial target input
    - POPCOUNT, AND, OR, XOR, MAJORITY pooling
    - Adjustable kernel and stride
    - Copy-ready register export
    - Target protection until convergence or manual reveal

Install:
    pip install numpy opencv-python pyserial

Run:
    python bit_pooling_cli.py
"""

import sys
import time
import random
import threading

import numpy as np
import cv2

try:
    import serial
except ImportError:
    serial = None


# =============================================================================
# CONFIGURATION
# =============================================================================

BIT_DEPTH = 512
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# =============================================================================
# ORACLE
# =============================================================================

class BitPoolingOracle:

    def __init__(self, bit_depth=512, target=None):
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        self._target = (
            max(0, min(int(target), self.max_val))
            if target is not None
            else random.randint(0, self.max_val)
        )

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.pooled_bits = 0
        self.is_done = False

    def query_bit(self, trial):
        if self.is_done:
            return 0

        return int(self._target >= trial)

    @property
    def target(self):
        if not self.is_done:
            raise PermissionError("Target is protected.")

        return self._target

    def pool_step(self, value):

        bits = [
            (value >> i) & 1
            for i in range(self.bit_depth - 1, -1, -1)
        ]

        pooled = []

        for i in range(0, len(bits), self.stride):

            block = bits[i:i + self.kernel_size]

            if not block:
                break

            ones = sum(block)
            count = len(block)

            if self.pool_op == "POPCOUNT":
                result = int(ones / count >= self.schmitt_trigger)

            elif self.pool_op == "AND":
                result = int(ones == count)

            elif self.pool_op == "OR":
                result = int(ones > 0)

            elif self.pool_op == "XOR":
                result = ones % 2

            elif self.pool_op == "MAJORITY":
                result = int(ones > count / 2)

            else:
                result = block[0]

            pooled.append(result)

        output = 0

        for bit in pooled:
            output = (output << 1) | bit

        return output


# =============================================================================
# SEARCH ENGINE
# =============================================================================

class BitPoolingSearchEngine:

    def __init__(self, bit_depth=512):

        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"

        self.lock = threading.RLock()

        self.reset()

    def reset(self, target=None):

        with self.lock:

            self.oracle = BitPoolingOracle(
                self.bit_depth,
                target
            )

            self.oracle.kernel_size = self.kernel_size
            self.oracle.stride = self.stride
            self.oracle.pool_op = self.pool_op

            self.current_guess = 0
            self.step_count = 0
            self.history = [0]

            self.guess_reg = 0
            self.pooled_register = 0

            self.converged = False
            self.last_status = "READY"

            self.update_registers()

    def update_registers(self):

        self.guess_reg = self.current_guess

        self.pooled_register = self.oracle.pool_step(
            self.current_guess
        )

    def set_pool_settings(self, operation=None, kernel=None, stride=None):

        with self.lock:

            if operation is not None:
                self.pool_op = operation

            if kernel is not None:
                self.kernel_size = max(1, int(kernel))

            if stride is not None:
                self.stride = max(1, int(stride))

            self.oracle.pool_op = self.pool_op
            self.oracle.kernel_size = self.kernel_size
            self.oracle.stride = self.stride

            self.update_registers()

    def step(self):

        with self.lock:

            if self.converged:
                return False

            if self.step_count >= self.bit_depth:
                return False

            shift = self.bit_depth - 1 - self.step_count

            trial = self.current_guess | (1 << shift)

            bit = self.oracle.query_bit(trial)

            if bit:
                self.current_guess |= (1 << shift)

            self.oracle.pooled_bits = self.current_guess

            self.step_count += 1

            self.history.append(self.current_guess)

            if self.step_count >= self.bit_depth:

                if self.current_guess == self.oracle._target:

                    self.oracle.is_done = True
                    self.converged = True
                    self.last_status = "LOCKED & REVEALED"

                else:
                    self.last_status = "POOL MISMATCH"

            self.update_registers()

            return True

    def run(self, verbose=True):

        while not self.converged:

            self.step()

            if verbose:
                print(
                    f"\rSearch: {self.step_count}/{self.bit_depth}",
                    end="",
                    flush=True
                )

        if verbose:
            print()

        return self.current_guess

    def snapshot(self):

        with self.lock:

            return {
                "guess": self.guess_reg,
                "pooled": self.pooled_register,
                "target": self.oracle._target,
                "steps": self.step_count,
                "status": self.last_status,
                "converged": self.converged
            }


# =============================================================================
# APPLICATION
# =============================================================================

class BitPoolingCLI:

    def __init__(self):

        self.engine = BitPoolingSearchEngine(BIT_DEPTH)

        self.force_reveal = False

        self.serial_port = None
        self.serial_buffer = bytearray()

        self.auto_step = True

        self.motion_threshold = 5.0
        self.previous_even = False

        self.camera_active = False

        self.frame_count = 0
        self.fps = 0.0
        self.fps_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # FORMATTING
    # -------------------------------------------------------------------------

    def binary(self, value):
        return format(value, f"0{BIT_DEPTH}b")

    def hexadecimal(self, value):
        return format(value, "x")

    def parse_integer(self, value):

        value = value.strip()

        if value.lower().startswith(("0x", "0b")):
            return int(value, 0)

        return int(value, 10)

    # -------------------------------------------------------------------------
    # REGISTERS
    # -------------------------------------------------------------------------

    def show_register(self, name, value, hidden=False):

        print(f"\n{name} REGISTER")

        if hidden:
            print("[HIDDEN]")
            return

        print("DECIMAL:", value)
        print("HEX:    ", "0x" + self.hexadecimal(value))
        print("BINARY: ", self.binary(value))

    def show_registers(self):

        state = self.engine.snapshot()

        print("\n" + "=" * 70)
        print("BIT-POOLING REGISTERS")
        print("=" * 70)

        self.show_register(
            "TARGET",
            state["target"],
            not (state["converged"] or self.force_reveal)
        )

        self.show_register("GUESS", state["guess"])
        self.show_register("POOLED", state["pooled"])

        print("\nSEARCH")
        print("Steps:", state["steps"], "/", BIT_DEPTH)
        print("Status:", state["status"])
        print("Operator:", self.engine.pool_op)
        print("Kernel:", self.engine.kernel_size)
        print("Stride:", self.engine.stride)

        print("=" * 70)

    def export_register(self, name):

        state = self.engine.snapshot()

        if name == "TARGET":

            if not (state["converged"] or self.force_reveal):
                print("Target is protected.")
                return

            value = state["target"]

        elif name == "GUESS":
            value = state["guess"]

        elif name == "POOLED":
            value = state["pooled"]

        else:
            return

        print("\n" + self.binary(value))

    def export_all(self):

        state = self.engine.snapshot()

        target = (
            self.binary(state["target"])
            if state["converged"] or self.force_reveal
            else "[HIDDEN]"
        )

        print("\nTARGET:", target)
        print("GUESS: ", self.binary(state["guess"]))
        print("POOLED:", self.binary(state["pooled"]))

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------

    def step_search(self):

        if not self.engine.step():
            print("Search already completed.")
            return

        state = self.engine.snapshot()

        print(
            f"Step {state['steps']}/{BIT_DEPTH} | "
            f"Guess: 0x{state['guess']:x} | "
            f"Pooled: 0x{state['pooled']:x}"
        )

        if state["converged"]:
            print("Target reconstructed successfully.")

    def run_search(self):

        print("Running search...")

        start = time.perf_counter()

        self.engine.run()

        elapsed = time.perf_counter() - start

        print("Search complete.")
        print("Queries:", self.engine.step_count)
        print(f"Elapsed: {elapsed:.6f} seconds")

        self.show_registers()

    def reset_search(self):

        self.engine.reset()
        self.force_reveal = False

        print("New random target generated.")

    def load_target(self, value):

        if not 0 <= value <= self.engine.max_val:
            print("Target outside 512-bit range.")
            return

        self.engine.reset(value)
        self.force_reveal = False

        print("Target loaded.")

    # -------------------------------------------------------------------------
    # POOL SETTINGS
    # -------------------------------------------------------------------------

    def configure_pooling(self):

        print("\nPOOL OPERATORS")
        print("1. POPCOUNT")
        print("2. AND")
        print("3. OR")
        print("4. XOR")
        print("5. MAJORITY")

        choices = {
            "1": "POPCOUNT",
            "2": "AND",
            "3": "OR",
            "4": "XOR",
            "5": "MAJORITY"
        }

        choice = input("Operator: ").strip()

        if choice not in choices:
            print("Invalid selection.")
            return

        try:

            kernel = int(input("Kernel size [4]: ") or "4")
            stride = int(input("Stride [2]: ") or "2")

            if kernel < 1 or stride < 1:
                raise ValueError

            self.engine.set_pool_settings(
                operation=choices[choice],
                kernel=kernel,
                stride=stride
            )

            print("Pooling settings updated.")

        except ValueError:
            print("Invalid kernel or stride.")

    # -------------------------------------------------------------------------
    # SERIAL
    # -------------------------------------------------------------------------

    def connect_serial(self):

        if serial is None:
            print("Install pyserial: pip install pyserial")
            return

        port_name = input("COM port [COM3]: ").strip() or "COM3"
        baud = int(input("Baud [115200]: ") or "115200")

        self.disconnect_serial()

        try:

            self.serial_port = serial.Serial(
                port_name,
                baud,
                timeout=0
            )

            self.serial_buffer.clear()

            print(f"Connected to {port_name}.")

        except Exception as exc:

            self.serial_port = None
            print("Serial error:", exc)

    def disconnect_serial(self):

        if self.serial_port is not None:

            try:
                self.serial_port.close()
            except Exception:
                pass

            self.serial_port = None

        self.serial_buffer.clear()

        print("Serial disconnected.")

    def process_serial(self):

        port = self.serial_port

        if port is None or not port.is_open:
            return

        try:

            waiting = port.in_waiting

            if waiting:
                self.serial_buffer.extend(port.read(waiting))

            while b"\n" in self.serial_buffer:

                raw, _, remainder = self.serial_buffer.partition(b"\n")
                self.serial_buffer = bytearray(remainder)

                line = raw.decode(
                    "ascii",
                    errors="ignore"
                ).strip()

                if not line:
                    continue

                try:

                    value = self.parse_integer(line)

                    if 0 <= value <= self.engine.max_val:

                        self.engine.reset(value)
                        self.force_reveal = False

                        print(
                            f"\n[SERIAL] New target loaded: {line}"
                        )

                    else:
                        print("\n[SERIAL] Target outside 512-bit range.")

                except ValueError:
                    print(f"\n[SERIAL] Ignored: {line}")

        except Exception as exc:

            print("\n[SERIAL ERROR]", exc)
            self.disconnect_serial()

    # -------------------------------------------------------------------------
    # OPENCV CAMERA
    # -------------------------------------------------------------------------

    def draw_camera_overlay(self, frame, motion_value, motion_detected):

        state = self.engine.snapshot()

        height, width = frame.shape[:2]

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 135),
            (15, 15, 15),
            -1
        )

        lines = [
            "BIT-POOLING CAMERA LAB",
            f"STEP: {state['steps']}/{BIT_DEPTH}",
            f"OPERATOR: {self.engine.pool_op}",
            f"KERNEL: {self.engine.kernel_size} STRIDE: {self.engine.stride}",
            f"MOTION: {motion_value:.2f}",
            f"AUTO STEP: {'ON' if self.auto_step else 'OFF'}",
            f"STATUS: {state['status']}"
        ]

        for i, line in enumerate(lines):

            cv2.putText(
                frame,
                line,
                (12, 20 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 230, 255),
                1,
                cv2.LINE_AA
            )

        if motion_detected:

            cv2.putText(
                frame,
                "MOTION DETECTED",
                (12, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

    def camera_loop(self):

        cap = cv2.VideoCapture(CAMERA_INDEX)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():

            print("Could not open camera.")
            return

        print("\nOpenCV camera started.")
        print("Camera controls:")
        print("  Q = quit camera")
        print("  SPACE = manually step")
        print("  A = toggle automatic motion stepping")
        print("  R = reset random target")
        print("  V = toggle target reveal")
        print("  S = show registers in terminal")
        print("  E = export all registers in terminal")

        previous_even = False
        previous_time = time.perf_counter()

        self.camera_active = True

        try:

            while self.camera_active:

                ok, frame = cap.read()

                if not ok:
                    print("Camera frame unavailable.")
                    break

                self.process_serial()

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                small = cv2.resize(
                    gray,
                    (128, 96),
                    interpolation=cv2.INTER_AREA
                )

                motion_value = float(np.std(small))

                motion_detected = (
                    self.frame_count > 5
                    and motion_value > self.motion_threshold
                )

                # Preserve the original two-consecutive-even-trigger rule.
                motion_signal = (
                    int(motion_value * 10)
                    if motion_detected
                    else 0
                )

                even_now = (
                    motion_signal > 0
                    and motion_signal % 2 == 0
                )

                even_state = even_now and previous_even
                previous_even = even_now

                if even_state and self.auto_step:
                    self.engine.step()

                self.draw_camera_overlay(
                    frame,
                    motion_value,
                    motion_detected
                )

                now = time.perf_counter()
                elapsed = now - previous_time
                previous_time = now

                if elapsed > 0:
                    self.fps = 1.0 / elapsed

                cv2.putText(
                    frame,
                    f"FPS: {self.fps:.1f}",
                    (frame.shape[1] - 120, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                cv2.imshow(
                    "Bit-Pooling Higher-Lower Camera Lab",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == 32:
                    self.step_search()

                elif key == ord("a"):
                    self.auto_step = not self.auto_step
                    print("Auto stepping:", self.auto_step)

                elif key == ord("r"):
                    self.reset_search()

                elif key == ord("v"):
                    self.force_reveal = not self.force_reveal
                    print("Target reveal:", self.force_reveal)

                elif key == ord("s"):
                    self.show_registers()

                elif key == ord("e"):
                    self.export_all()

                self.frame_count += 1

        finally:

            self.camera_active = False

            cap.release()
            cv2.destroyAllWindows()

            print("Camera stopped.")

    # -------------------------------------------------------------------------
    # MENU
    # -------------------------------------------------------------------------

    def menu(self):

        print("\n" + "=" * 65)
        print("BIT-POOLING HIGHER-LOWER SEARCH")
        print("=" * 65)

        print("1. Start OpenCV camera")
        print("2. Step search")
        print("3. Run complete search")
        print("4. Show registers")
        print("5. Configure pooling")
        print("6. Load manual target")
        print("7. Reset random target")
        print("8. Toggle target reveal")
        print("9. Export TARGET")
        print("10. Export GUESS")
        print("11. Export POOLED")
        print("12. Export ALL")
        print("13. Connect Arduino serial")
        print("14. Read serial target")
        print("15. Disconnect serial")
        print("0. Exit")

        print("=" * 65)

    def run(self):

        print("512-bit Bit-Pooling Search Initialized.")
        print("A random oracle target has been generated.")

        try:

            while True:

                self.menu()

                choice = input("Select: ").strip()

                if choice == "1":
                    self.camera_loop()

                elif choice == "2":
                    self.step_search()

                elif choice == "3":
                    self.run_search()

                elif choice == "4":
                    self.show_registers()

                elif choice == "5":
                    self.configure_pooling()

                elif choice == "6":

                    raw = input("Target: ")

                    try:
                        self.load_target(
                            self.parse_integer(raw)
                        )
                    except ValueError:
                        print("Invalid target.")

                elif choice == "7":
                    self.reset_search()

                elif choice == "8":

                    self.force_reveal = not self.force_reveal

                    print(
                        "Target revealed."
                        if self.force_reveal
                        else "Target hidden."
                    )

                elif choice == "9":
                    self.export_register("TARGET")

                elif choice == "10":
                    self.export_register("GUESS")

                elif choice == "11":
                    self.export_register("POOLED")

                elif choice == "12":
                    self.export_all()

                elif choice == "13":
                    self.connect_serial()

                elif choice == "14":
                    if self.serial_port is None:
                        print("Connect serial first.")
                    else:
                        print(
                            "Start the camera to continuously read serial, "
                            "or send data while the camera is active."
                        )

                elif choice == "15":
                    self.disconnect_serial()

                elif choice == "0":
                    break

                else:
                    print("Invalid selection.")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")

        finally:

            self.camera_active = False
            self.disconnect_serial()
            cv2.destroyAllWindows()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    if cv2 is None:
        print("OpenCV is required: pip install opencv-python")
        sys.exit(1)

    app = BitPoolingCLI()
    app.run()
