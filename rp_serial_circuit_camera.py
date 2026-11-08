
"""
Bit-Pooling Higher-Lower Search & OpenCV Camera Lab
Isolated Oracle / Target Edition

Python 3.10+

Install:
    pip install numpy opencv-python pyserial

Run:
    python bit_pooling_cli.py

Isolation:
    - Oracle owns the target.
    - Search engine cannot directly access the target.
    - Hardware registers receive only the reconstructed guess
      and its pooled representation.
    - Oracle returns only higher/lower answers during search.
    - Target reveal is locked until all 512 bits are reconstructed.
    - Manual reveal cannot bypass the completion lock.
"""

import sys
import time
import random
import threading
import queue

import numpy as np
import cv2

try:
    import serial
except ImportError:
    serial = None


# ============================================================
# CONFIGURATION
# ============================================================

BIT_DEPTH = 512
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

POOL_OPERATORS = (
    "POPCOUNT",
    "AND",
    "OR",
    "XOR",
    "MAJORITY"
)


# ============================================================
# ISOLATED ORACLE
# ============================================================

class IsolatedOracle:
    """
    The oracle privately owns the target.

    During search, the only available operation is query_bit().
    The target cannot be revealed until finish_search() is called
    after the required number of queries.
    """

    def __init__(self, bit_depth=512, target=None):

        self.__bit_depth = bit_depth
        self.__max_value = (1 << bit_depth) - 1

        if target is None:
            self.__target = random.randint(0, self.__max_value)
        else:
            self.__target = max(
                0,
                min(int(target), self.__max_value)
            )

        self.__query_count = 0
        self.__complete = False

        self.__lock = threading.RLock()

    def query_bit(self, trial_value):

        with self.__lock:

            if self.__complete:
                raise RuntimeError("Oracle search is already complete.")

            self.__query_count += 1

            return int(self.__target >= trial_value)

    def finish_search(self):

        with self.__lock:

            if self.__query_count < self.__bit_depth:
                raise PermissionError(
                    "Oracle cannot unlock before all bits are reconstructed."
                )

            self.__complete = True

    def reveal(self):

        with self.__lock:

            if not self.__complete:
                raise PermissionError(
                    "Target remains isolated until reconstruction completes."
                )

            return self.__target

    def verify(self, candidate):

        with self.__lock:

            if not self.__complete:
                raise PermissionError(
                    "Verification unavailable before reconstruction completes."
                )

            return candidate == self.__target

    @property
    def query_count(self):
        return self.__query_count

    @property
    def is_complete(self):
        return self.__complete


# ============================================================
# HARDWARE-STYLE BIT POOLING
# ============================================================

class BitPoolingHardware:

    def __init__(self, bit_depth=512):

        self.bit_depth = bit_depth

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.guess_register = 0
        self.pooled_register = 0

    def configure(self, operation=None, kernel=None, stride=None):

        if operation is not None:

            operation = operation.upper()

            if operation not in POOL_OPERATORS:
                raise ValueError("Unknown pooling operator.")

            self.pool_op = operation

        if kernel is not None:
            self.kernel_size = max(1, int(kernel))

        if stride is not None:
            self.stride = max(1, int(stride))

    def receive_guess(self, guess):

        self.guess_register = guess

        self.pooled_register = self.pool(guess)

    def pool(self, value):

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
                result = int(
                    ones / count >= self.schmitt_trigger
                )

            elif self.pool_op == "AND":
                result = int(ones == count)

            elif self.pool_op == "OR":
                result = int(ones > 0)

            elif self.pool_op == "XOR":
                result = ones % 2

            elif self.pool_op == "MAJORITY":
                result = int(ones > count / 2)

            else:
                result = 0

            pooled.append(result)

        output = 0

        for bit in pooled:
            output = (output << 1) | bit

        return output


# ============================================================
# SEARCH ENGINE
# ============================================================

class BitPoolingSearchEngine:

    def __init__(self, bit_depth=512):

        self.bit_depth = bit_depth
        self.max_value = (1 << bit_depth) - 1

        self.lock = threading.RLock()

        self.hardware = BitPoolingHardware(bit_depth)

        self.reset()

    def reset(self, target=None):

        with self.lock:

            self.oracle = IsolatedOracle(
                self.bit_depth,
                target
            )

            self.current_guess = 0
            self.step_count = 0

            self.history = [0]

            self.converged = False
            self.verified = False

            self.last_status = "READY"

            # Hardware starts with empty registers.
            self.hardware.receive_guess(0)

    def configure_pooling(self, operation=None, kernel=None, stride=None):

        with self.lock:

            self.hardware.configure(
                operation,
                kernel,
                stride
            )

            # Only the current reconstructed guess enters hardware.
            self.hardware.receive_guess(self.current_guess)

    def step(self):

        with self.lock:

            if self.converged:
                return False

            if self.step_count >= self.bit_depth:
                return False

            shift = self.bit_depth - 1 - self.step_count

            trial_value = (
                self.current_guess | (1 << shift)
            )

            # The oracle returns one bit only.
            # No target value is transferred to the engine.
            answer = self.oracle.query_bit(trial_value)

            if answer:
                self.current_guess |= (1 << shift)

            self.step_count += 1

            self.history.append(self.current_guess)

            # Hardware receives the reconstructed guess only.
            self.hardware.receive_guess(
                self.current_guess
            )

            if self.step_count == self.bit_depth:

                # Oracle unlocks only after the complete reconstruction.
                self.oracle.finish_search()

                self.converged = True

                # Verification returns a boolean, not the target.
                self.verified = self.oracle.verify(
                    self.current_guess
                )

                self.last_status = (
                    "RECONSTRUCTION VERIFIED"
                    if self.verified
                    else "RECONSTRUCTION MISMATCH"
                )

            return True

    def run(self, verbose=True):

        while not self.converged:

            self.step()

            if verbose:

                print(
                    f"\rReconstruction: "
                    f"{self.step_count}/{self.bit_depth}",
                    end="",
                    flush=True
                )

        if verbose:
            print()

        return self.current_guess

    def snapshot(self):

        with self.lock:

            return {
                "guess": self.hardware.guess_register,
                "pooled": self.hardware.pooled_register,
                "steps": self.step_count,
                "status": self.last_status,
                "converged": self.converged,
                "verified": self.verified,
                "target_unlocked": self.oracle.is_complete
            }

    def reveal_target(self):

        # Oracle itself enforces the completion lock.
        return self.oracle.reveal()


# ============================================================
# COMMAND-LINE APPLICATION
# ============================================================

class BitPoolingCLI:

    def __init__(self):

        self.engine = BitPoolingSearchEngine(BIT_DEPTH)

        self.auto_step = True
        self.force_reveal = False

        self.motion_threshold = 5.0
        self.previous_even = False

        self.serial_port = None
        self.serial_thread = None
        self.serial_stop = threading.Event()
        self.serial_queue = queue.Queue()

        self.camera_active = False

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    def binary(self, value):

        return format(value, f"0{BIT_DEPTH}b")

    def parse_integer(self, value):

        value = value.strip()

        if value.lower().startswith(("0x", "0b")):
            return int(value, 0)

        return int(value, 10)

    # --------------------------------------------------------
    # REGISTER DISPLAY
    # --------------------------------------------------------

    def show_register(self, name, value):

        print(f"\n{name} REGISTER")
        print("DECIMAL:", value)
        print("HEX:    ", hex(value))
        print("BINARY: ", self.binary(value))

    def show_registers(self):

        state = self.engine.snapshot()

        print("\n" + "=" * 70)
        print("BIT-POOLING HARDWARE REGISTERS")
        print("=" * 70)

        self.show_register(
            "GUESS",
            state["guess"]
        )

        self.show_register(
            "POOLED",
            state["pooled"]
        )

        print("\nSEARCH STATE")
        print("Steps:", state["steps"], "/", BIT_DEPTH)
        print("Status:", state["status"])
        print("Verified:", state["verified"])

        print("\nORACLE ISOLATION")
        print(
            "Target:",
            "UNLOCKED" if state["target_unlocked"] else "ISOLATED"
        )

        print("=" * 70)

    def show_target(self):

        state = self.engine.snapshot()

        if not state["target_unlocked"]:
            print("Target is isolated. Reconstruction is incomplete.")
            return

        target = self.engine.reveal_target()

        self.show_register("REVEALED TARGET", target)

    def export_register(self, name):

        state = self.engine.snapshot()

        if name == "GUESS":
            value = state["guess"]

        elif name == "POOLED":
            value = state["pooled"]

        elif name == "TARGET":

            if not state["target_unlocked"]:
                print("Target remains isolated.")
                return

            value = self.engine.reveal_target()

        else:
            print("Unknown register.")
            return

        print("\n" + self.binary(value))

    def export_all(self):

        state = self.engine.snapshot()

        print("\nGUESS:")
        print(self.binary(state["guess"]))

        print("\nPOOLED:")
        print(self.binary(state["pooled"]))

        if state["target_unlocked"]:

            print("\nTARGET:")
            print(self.binary(self.engine.reveal_target()))

        else:
            print("\nTARGET: [ISOLATED]")

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

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

            print("\nAll 512 bits reconstructed.")

            if state["verified"]:
                print("Oracle verification successful.")
                print("Target isolation has now been lifted.")

            else:
                print("Verification failed.")

    def run_search(self):

        print("Reconstructing 512-bit value...")

        start = time.perf_counter()

        self.engine.run()

        elapsed = time.perf_counter() - start

        state = self.engine.snapshot()

        print("\nReconstruction complete.")
        print("Steps:", state["steps"])
        print("Verified:", state["verified"])
        print(f"Elapsed: {elapsed:.6f} seconds")

        self.show_registers()

    def reset_search(self):

        self.engine.reset()

        print("New random target generated.")
        print("Oracle target isolated.")
        print("Hardware registers cleared.")

    def load_target(self, value):

        if not 0 <= value <= self.engine.max_value:
            print("Target outside 512-bit range.")
            return

        self.engine.reset(value)

        print("Serial/manual target loaded.")
        print("Oracle target isolated.")
        print("Hardware registers cleared.")

    # --------------------------------------------------------
    # POOL CONFIGURATION
    # --------------------------------------------------------

    def configure_pooling(self):

        print("\nPOOL OPERATORS")

        for index, operation in enumerate(POOL_OPERATORS, 1):
            print(f"{index}. {operation}")

        try:

            choice = int(input("Operator: "))

            if not 1 <= choice <= len(POOL_OPERATORS):
                raise ValueError

            operation = POOL_OPERATORS[choice - 1]

            kernel = int(input("Kernel size [4]: ") or "4")
            stride = int(input("Stride [2]: ") or "2")

            self.engine.configure_pooling(
                operation,
                kernel,
                stride
            )

            print("Pooling settings updated.")

        except ValueError:
            print("Invalid settings.")

    # --------------------------------------------------------
    # SERIAL INPUT
    # --------------------------------------------------------

    def connect_serial(self):

        if serial is None:
            print("Install pyserial: pip install pyserial")
            return

        port_name = input("COM port [COM3]: ").strip() or "COM3"

        try:
            baud = int(input("Baud [115200]: ") or "115200")

            self.disconnect_serial()

            self.serial_port = serial.Serial(
                port_name,
                baud,
                timeout=0.2
            )

            self.serial_stop.clear()

            self.serial_thread = threading.Thread(
                target=self.serial_reader,
                daemon=True
            )

            self.serial_thread.start()

            print(f"Connected to {port_name}.")
            print("Waiting for newline-terminated target values.")

        except Exception as exc:

            self.serial_port = None
            print("Serial connection error:", exc)

    def serial_reader(self):

        while not self.serial_stop.is_set():

            port = self.serial_port

            if port is None:
                break

            try:

                line = port.readline()

                if line:

                    value = line.decode(
                        "ascii",
                        errors="ignore"
                    ).strip()

                    if value:
                        self.serial_queue.put(value)

            except Exception as exc:

                self.serial_queue.put(
                    ("ERROR", str(exc))
                )

                break

    def process_serial_queue(self):

        while True:

            try:
                item = self.serial_queue.get_nowait()

            except queue.Empty:
                break

            if isinstance(item, tuple):

                if item[0] == "ERROR":
                    print("\n[SERIAL ERROR]", item[1])

                continue

            try:

                value = self.parse_integer(item)

                if 0 <= value <= self.engine.max_value:

                    self.load_target(value)

                    print(
                        "[SERIAL] New isolated target accepted."
                    )

                else:
                    print("[SERIAL] Target outside 512-bit range.")

            except ValueError:
                print("[SERIAL] Ignored invalid input:", item)

    def disconnect_serial(self):

        self.serial_stop.set()

        if self.serial_port is not None:

            try:
                self.serial_port.close()
            except Exception:
                pass

        self.serial_port = None

        print("Serial disconnected.")

    # --------------------------------------------------------
    # OPENCV
    # --------------------------------------------------------

    def draw_overlay(self, frame, motion_value, motion_detected):

        state = self.engine.snapshot()

        height, width = frame.shape[:2]

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 155),
            (15, 15, 15),
            -1
        )

        target_status = (
            "UNLOCKED"
            if state["target_unlocked"]
            else "ISOLATED"
        )

        lines = [
            "BIT-POOLING HIGHER-LOWER CAMERA LAB",
            f"RECONSTRUCTION: {state['steps']}/{BIT_DEPTH}",
            f"POOL: {self.engine.hardware.pool_op}",
            f"KERNEL: {self.engine.hardware.kernel_size}",
            f"STRIDE: {self.engine.hardware.stride}",
            f"MOTION: {motion_value:.2f}",
            f"AUTO STEP: {'ON' if self.auto_step else 'OFF'}",
            f"ORACLE TARGET: {target_status}"
        ]

        for i, line in enumerate(lines):

            cv2.putText(
                frame,
                line,
                (12, 20 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
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
            cap.release()
            return

        print("\nOpenCV camera active.")

        print("""
Camera controls:
    Q       Quit camera
    SPACE   Advance one search step
    A       Toggle automatic motion stepping
    R       Reset random target
    S       Show hardware registers
    E       Export registers
    V       Attempt target reveal (only after completion)
""")

        previous_even = False
        self.camera_active = True

        try:

            while self.camera_active:

                ok, frame = cap.read()

                if not ok:
                    print("Camera frame unavailable.")
                    break

                self.process_serial_queue()

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
                    motion_value > self.motion_threshold
                )

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

                self.draw_overlay(
                    frame,
                    motion_value,
                    motion_detected
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
                    print("Automatic stepping:", self.auto_step)

                elif key == ord("r"):
                    self.reset_search()

                elif key == ord("s"):
                    self.show_registers()

                elif key == ord("e"):
                    self.export_all()

                elif key == ord("v"):
                    self.show_target()

        finally:

            self.camera_active = False

            cap.release()
            cv2.destroyAllWindows()

            print("Camera stopped.")

    # --------------------------------------------------------
    # TERMINAL MENU
    # --------------------------------------------------------

    def menu(self):

        print("""
========================================================
BIT-POOLING HIGHER-LOWER SEARCH
========================================================
1. Start OpenCV camera
2. Advance one search step
3. Run complete reconstruction
4. Show hardware registers
5. Configure pooling
6. Load manual target
7. Reset random target
8. Reveal target (completion required)
9. Export GUESS register
10. Export POOLED register
11. Export TARGET register
12. Export all registers
13. Connect Arduino serial
14. Disconnect Arduino serial
0. Exit
========================================================
""")

    def run(self):

        print("512-bit search initialized.")
        print("Oracle target is isolated.")
        print("Hardware registers start empty.")

        try:

            while True:

                self.process_serial_queue()
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

                    raw = input("Target decimal/hex/binary: ")

                    try:
                        value = self.parse_integer(raw)
                        self.load_target(value)

                    except ValueError:
                        print("Invalid target.")

                elif choice == "7":
                    self.reset_search()

                elif choice == "8":
                    self.show_target()

                elif choice == "9":
                    self.export_register("GUESS")

                elif choice == "10":
                    self.export_register("POOLED")

                elif choice == "11":
                    self.export_register("TARGET")

                elif choice == "12":
                    self.export_all()

                elif choice == "13":
                    self.connect_serial()

                elif choice == "14":
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app = BitPoolingCLI()
    app.run()
