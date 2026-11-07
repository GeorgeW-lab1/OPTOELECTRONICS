
"""
BIT-POOLING SHA-256 ORACLE LOCK BOX
Python 3.10+

Features:
    - SHA-256 minimal message padding
    - Higher/lower bit-pooling oracle
    - MSB-first reconstruction
    - Emulated proof-of-work miner
    - Independent SHA-256 verification
    - Full reconstructed-message verification
    - Tkinter GUI
    - Padding and reconstruction self-tests

Run:
    python minimal_padding_bitpool_oracle.py

Run tests:
    python minimal_padding_bitpool_oracle.py --test
"""

import hashlib
import queue
import threading
import time
import sys
import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# SHA-256 PADDING
# ============================================================

def sha256_pad_minimal(message: bytes) -> bytes:
    """
    Apply standard SHA-256 Merkle-Damgard padding.

    Output is always a multiple of 64 bytes.
    The final 8 bytes contain the original message length
    in bits, encoded big-endian.
    """

    if not isinstance(message, bytes):
        raise TypeError("Message must be bytes")

    bit_length = len(message) * 8

    if bit_length >= (1 << 64):
        raise ValueError("Message is too long for SHA-256 padding")

    zero_count = (56 - (len(message) + 1) % 64) % 64

    padded = (
        message
        + b"\x80"
        + (b"\x00" * zero_count)
        + bit_length.to_bytes(8, "big")
    )

    if len(padded) % 64 != 0:
        raise AssertionError("Internal SHA-256 padding error")

    return padded


def sha256_unpad_minimal(padded: bytes):
    """
    Strictly validate SHA-256 padding and recover
    the original message.

    Returns:
        bytes: recovered message
        None: invalid padding
    """

    if not isinstance(padded, bytes):
        return None

    if not padded:
        return None

    if len(padded) % 64 != 0:
        return None

    if len(padded) < 64:
        return None

    bit_length = int.from_bytes(padded[-8:], "big")

    if bit_length % 8 != 0:
        return None

    message_length = bit_length // 8

    if message_length > len(padded) - 9:
        return None

    expected_padding_length = (
        (56 - (message_length + 1) % 64) % 64
    )

    expected_total_length = (
        message_length
        + 1
        + expected_padding_length
        + 8
    )

    if expected_total_length != len(padded):
        return None

    if padded[message_length] != 0x80:
        return None

    padding_start = message_length + 1
    padding_end = len(padded) - 8

    if any(padded[padding_start:padding_end]):
        return None

    message = padded[:message_length]

    # Exact round-trip validation.
    if sha256_pad_minimal(message) != padded:
        return None

    return message


def leading_zero_bits(digest: bytes) -> int:
    """Count leading zero bits in a digest."""

    count = 0

    for byte in digest:
        if byte == 0:
            count += 8
            continue

        count += 8 - byte.bit_length()
        break

    return count


# ============================================================
# HASH VERIFICATION
# ============================================================

def verify_pow_result(
    header: bytes,
    nonce: int,
    expected_digest: bytes,
    difficulty: int,
    expected_padded=None
):
    """
    Independently verify:
        1. Nonce and header input.
        2. SHA-256 digest.
        3. Proof-of-work difficulty.
        4. SHA-256 padding.
        5. Optional expected padded message.
    """

    try:
        if not isinstance(header, bytes):
            return False, "Invalid header type"

        if not isinstance(nonce, int) or nonce < 0:
            return False, "Invalid nonce"

        if not isinstance(expected_digest, bytes):
            return False, "Invalid digest type"

        if len(expected_digest) != 32:
            return False, "Digest must be 32 bytes"

        if not 0 <= difficulty <= 256:
            return False, "Invalid difficulty"

        message = header + str(nonce).encode("ascii")

        calculated_digest = hashlib.sha256(message).digest()

        if calculated_digest != expected_digest:
            return False, "SHA-256 digest mismatch"

        zeros = leading_zero_bits(calculated_digest)

        if zeros < difficulty:
            return False, "Insufficient proof-of-work difficulty"

        padded = sha256_pad_minimal(message)

        recovered = sha256_unpad_minimal(padded)

        if recovered is None:
            return False, "Generated SHA-256 padding is invalid"

        if recovered != message:
            return False, "Padding round-trip mismatch"

        if expected_padded is not None:
            if padded != expected_padded:
                return False, "Miner padding mismatch"

        if hashlib.sha256(recovered).digest() != calculated_digest:
            return False, "Recovered-message hash mismatch"

        return True, {
            "message": recovered,
            "nonce": nonce,
            "digest": calculated_digest.hex(),
            "leading_zero_bits": zeros,
            "difficulty": difficulty,
            "padded_message": padded,
            "verified": True
        }

    except Exception as exc:
        return False, str(exc)


def verify_oracle_reconstruction(
    reconstructed_integer: int,
    bit_depth: int,
    expected_message: bytes,
    expected_digest: bytes,
    difficulty: int,
    expected_padded: bytes
):
    """
    Verify the actual reconstructed oracle output.

    Crucially, this converts the integer into exactly the
    original number of bytes. It never discards leading zeros.
    """

    if bit_depth <= 0 or bit_depth % 512 != 0:
        return False, "Invalid oracle bit depth"

    byte_length = bit_depth // 8

    try:
        reconstructed = int(reconstructed_integer).to_bytes(
            byte_length,
            "big"
        )
    except (OverflowError, ValueError):
        return False, "Oracle integer exceeds its bit depth"

    if len(reconstructed) != byte_length:
        return False, "Oracle byte length mismatch"

    if reconstructed != expected_padded:
        return False, "Oracle reconstruction differs from target"

    recovered = sha256_unpad_minimal(reconstructed)

    if recovered is None:
        return False, "Invalid minimal padding in reconstructed data"

    if recovered != expected_message:
        return False, "Recovered message differs from expected message"

    digest = hashlib.sha256(recovered).digest()

    if digest != expected_digest:
        return False, "Reconstructed message SHA-256 mismatch"

    zeros = leading_zero_bits(digest)

    if zeros < difficulty:
        return False, "Reconstructed hash fails proof-of-work"

    return True, {
        "message": recovered,
        "padded": reconstructed,
        "digest": digest.hex(),
        "leading_zero_bits": zeros,
        "difficulty": difficulty,
        "verified": True
    }


# ============================================================
# HIGHER / LOWER BIT ORACLE
# ============================================================

class MinimalMessageOracle:

    def __init__(self, padded_message: bytes):

        if not isinstance(padded_message, bytes):
            raise TypeError("Oracle target must be bytes")

        if len(padded_message) == 0 or len(padded_message) % 64:
            raise ValueError("Oracle target must contain complete SHA-256 blocks")

        self.target_bytes = padded_message

        self.bit_depth = len(padded_message) * 8

        self.target_integer = int.from_bytes(
            padded_message,
            "big"
        )

        self.query_count = 0

    def query_bit(self, trial_value: int) -> bool:
        """
        Higher/lower comparison.

        Returns True if the hidden target is greater than
        or equal to the trial value.
        """

        self.query_count += 1

        return self.target_integer >= trial_value


# ============================================================
# BIT-POOLING RECONSTRUCTION ENGINE
# ============================================================

class ParallelMinimalSearchEngine:

    def __init__(self):

        self.oracle = None

        self.current_guess = 0

        self.solved_bits_mask = 0

        self.bit_depth = 0

        self.solved_bits = 0

        self.converged = False

        self.lock = threading.RLock()

    def reset(self, message: bytes):

        padded = sha256_pad_minimal(message)

        with self.lock:

            self.oracle = MinimalMessageOracle(padded)

            self.bit_depth = len(padded) * 8

            self.current_guess = 0

            self.solved_bits_mask = 0

            self.solved_bits = 0

            self.converged = False

        return padded

    def reset_with_padded_message(self, padded: bytes):

        recovered = sha256_unpad_minimal(padded)

        if recovered is None:
            raise ValueError("Cannot initialise oracle with invalid padding")

        with self.lock:

            self.oracle = MinimalMessageOracle(padded)

            self.bit_depth = len(padded) * 8

            self.current_guess = 0

            self.solved_bits_mask = 0

            self.solved_bits = 0

            self.converged = False

    def step(self):

        with self.lock:

            if self.oracle is None:
                return None

            if self.converged:
                return None

            index = self.solved_bits

            if index >= self.bit_depth:
                self.converged = True
                return None

            # MSB-first reconstruction.
            shift = self.bit_depth - 1 - index

            trial_value = (
                self.current_guess
                | (1 << shift)
            )

            # Higher/lower query determines the next bit.
            if self.oracle.query_bit(trial_value):
                self.current_guess = trial_value

            self.solved_bits_mask |= 1 << shift

            self.solved_bits += 1

            if self.solved_bits >= self.bit_depth:
                self.converged = True

            return {
                "index": index,
                "guess": self.current_guess,
                "solved_bits": self.solved_bits,
                "total_bits": self.bit_depth,
                "converged": self.converged
            }

    def solve_all(self):

        while not self.converged:
            self.step()

        return self.current_guess

    def get_reconstructed_bytes(self):

        with self.lock:

            if not self.converged:
                return None

            return self.current_guess.to_bytes(
                self.bit_depth // 8,
                "big"
            )


# ============================================================
# EMULATED ARDUINO PROOF-OF-WORK MINER
# ============================================================

class EmulatedArduinoMiner:

    def __init__(self, event_queue):

        self.events = event_queue

        self.stop_event = threading.Event()

        self.thread = None

        self.hash_count = 0

    def start(self, header, difficulty):

        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._mine,
            args=(header, difficulty),
            daemon=True
        )

        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _mine(self, header, difficulty):

        nonce = 0

        started = time.time()

        while not self.stop_event.is_set():

            message = header + str(nonce).encode("ascii")

            digest = hashlib.sha256(message).digest()

            self.hash_count += 1

            zeros = leading_zero_bits(digest)

            if zeros >= difficulty:

                padded = sha256_pad_minimal(message)

                self.events.put({
                    "type": "pow_found",
                    "header": header,
                    "nonce": nonce,
                    "digest": digest,
                    "padded_msg": padded,
                    "difficulty": difficulty,
                    "hash_count": self.hash_count,
                    "elapsed": time.time() - started
                })

                return

            if nonce % 10000 == 0:

                elapsed = max(time.time() - started, 0.000001)

                self.events.put({
                    "type": "progress",
                    "nonce": nonce,
                    "hash_count": self.hash_count,
                    "hash_rate": self.hash_count / elapsed
                })

            nonce += 1

        self.events.put({
            "type": "stopped"
        })


# ============================================================
# SELF TESTS
# ============================================================

def run_self_tests():

    print("Running SHA-256 padding tests...")

    test_lengths = [
        0, 1, 2, 14, 15, 55, 56, 57,
        63, 64, 65, 100, 119, 120,
        127, 128, 129, 500, 1000
    ]

    for length in test_lengths:

        message = bytes(
            (i % 256 for i in range(length))
        )

        padded = sha256_pad_minimal(message)

        recovered = sha256_unpad_minimal(padded)

        assert recovered == message

        assert len(padded) % 64 == 0

        assert hashlib.sha256(message).digest() == hashlib.sha256(recovered).digest()

        print(
            f"  PASS: {length} bytes -> {len(padded)} padded bytes"
        )

    print("\nTesting bit-pooling reconstruction...")

    messages = [
        b"",
        b"hello",
        b"arduino-sha256",
        b"bit-pooling oracle test"
    ]

    for message in messages:

        engine = ParallelMinimalSearchEngine()

        padded = engine.reset(message)

        reconstructed = engine.solve_all()

        recovered_bytes = reconstructed.to_bytes(
            engine.bit_depth // 8,
            "big"
        )

        assert recovered_bytes == padded

        assert sha256_unpad_minimal(recovered_bytes) == message

        print(
            f"  PASS: {len(message)} message bytes, "
            f"{engine.bit_depth} reconstructed bits"
        )

    print("\nTesting proof-of-work verification...")

    header = b"arduino-sha256"

    difficulty = 8

    nonce = 0

    while True:

        message = header + str(nonce).encode("ascii")

        digest = hashlib.sha256(message).digest()

        if leading_zero_bits(digest) >= difficulty:
            break

        nonce += 1

    padded = sha256_pad_minimal(message)

    valid, result = verify_pow_result(
        header,
        nonce,
        digest,
        difficulty,
        padded
    )

    assert valid, result

    engine = ParallelMinimalSearchEngine()

    engine.reset_with_padded_message(padded)

    engine.solve_all()

    valid, result = verify_oracle_reconstruction(
        engine.current_guess,
        engine.bit_depth,
        message,
        digest,
        difficulty,
        padded
    )

    assert valid, result

    print("  PASS: proof-of-work hash verified")
    print("  PASS: oracle reconstruction verified")
    print("  PASS: reconstructed padding verified")
    print("\nALL TESTS PASSED")


# ============================================================
# TKINTER GUI
# ============================================================

class BitPoolingOracleApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Bit-Pooling SHA-256 Oracle Lock Box")

        self.root.geometry("1050x780")

        self.root.minsize(800, 650)

        self.events = queue.Queue()

        self.engine = ParallelMinimalSearchEngine()

        self.miner = EmulatedArduinoMiner(self.events)

        self.pow_data = None

        self.pow_verification = None

        self.oracle_verified = False

        self.auto_running = False

        self.auto_job = None

        self._build_ui()

        self.root.after(100, self._poll)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # --------------------------------------------------------

    def _build_ui(self):

        main = ttk.Frame(self.root, padding=12)

        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="BIT-POOLING SHA-256 ORACLE",
            font=("Segoe UI", 17, "bold")
        ).pack(pady=(0, 12))

        # Header
        header_frame = ttk.LabelFrame(
            main,
            text="Proof-of-work input",
            padding=10
        )

        header_frame.pack(fill="x", pady=5)

        ttk.Label(
            header_frame,
            text="Header:"
        ).grid(row=0, column=0, sticky="w")

        self.header_var = tk.StringVar(
            value="arduino-sha256"
        )

        ttk.Entry(
            header_frame,
            textvariable=self.header_var,
            width=55
        ).grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(
            header_frame,
            text="Difficulty:"
        ).grid(row=1, column=0, sticky="w", pady=6)

        self.difficulty_var = tk.StringVar(value="12")

        ttk.Entry(
            header_frame,
            textvariable=self.difficulty_var,
            width=10
        ).grid(row=1, column=1, sticky="w", padx=6, pady=6)

        header_frame.columnconfigure(1, weight=1)

        buttons = ttk.Frame(header_frame)

        buttons.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=5
        )

        ttk.Button(
            buttons,
            text="Start Mining",
            command=self.start_mining
        ).pack(side="left", padx=3)

        ttk.Button(
            buttons,
            text="Stop Mining",
            command=self.stop_mining
        ).pack(side="left", padx=3)

        # Oracle controls
        oracle_frame = ttk.LabelFrame(
            main,
            text="Bit-pooling oracle",
            padding=10
        )

        oracle_frame.pack(fill="x", pady=6)

        self.status_var = tk.StringVar(
            value="Status: Waiting for proof-of-work"
        )

        ttk.Label(
            oracle_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")

        self.progress = ttk.Progressbar(
            oracle_frame,
            maximum=100,
            mode="determinate"
        )

        self.progress.pack(fill="x", pady=8)

        self.progress_var = tk.StringVar(
            value="0 / 0 bits"
        )

        ttk.Label(
            oracle_frame,
            textvariable=self.progress_var
        ).pack(anchor="w")

        control_buttons = ttk.Frame(oracle_frame)

        control_buttons.pack(fill="x", pady=8)

        ttk.Button(
            control_buttons,
            text="Step One Bit",
            command=self.step_once
        ).pack(side="left", padx=3)

        ttk.Button(
            control_buttons,
            text="Solve Automatically",
            command=self.start_auto
        ).pack(side="left", padx=3)

        ttk.Button(
            control_buttons,
            text="Stop",
            command=self.stop_auto
        ).pack(side="left", padx=3)

        # Output
        output_frame = ttk.LabelFrame(
            main,
            text="Verification and reconstruction",
            padding=10
        )

        output_frame.pack(fill="both", expand=True, pady=6)

        self.output = tk.Text(
            output_frame,
            height=22,
            wrap="word",
            font=("Consolas", 10)
        )

        self.output.pack(
            fill="both",
            expand=True
        )

        self.output.insert(
            "end",
            "Waiting for a proof-of-work challenge...\n"
        )

        self.output.configure(state="disabled")

        # Bottom
        ttk.Label(
            main,
            text=(
                "The oracle reconstructs the padded input message. "
                "SHA-256 is then independently recomputed and verified."
            ),
            foreground="gray"
        ).pack(pady=5)

    # --------------------------------------------------------

    def log(self, message):

        self.output.configure(state="normal")

        self.output.insert("end", str(message) + "\n")

        self.output.see("end")

        self.output.configure(state="disabled")

    # --------------------------------------------------------

    def start_mining(self):

        try:

            header = self.header_var.get().encode("utf-8")

            difficulty = int(self.difficulty_var.get())

            if not 0 <= difficulty <= 256:
                raise ValueError("Difficulty must be between 0 and 256")

            if not header:
                raise ValueError("Header cannot be empty")

        except Exception as exc:

            messagebox.showerror("Invalid input", str(exc))

            return

        self.stop_auto()

        self.miner.hash_count = 0

        self.pow_data = None

        self.pow_verification = None

        self.oracle_verified = False

        self.engine = ParallelMinimalSearchEngine()

        self.progress["value"] = 0

        self.progress_var.set("Waiting for nonce")

        self.status_var.set("Status: Mining")

        self.log("\n========================================")
        self.log("STARTING PROOF-OF-WORK MINER")
        self.log(f"Header: {header!r}")
        self.log(f"Difficulty: {difficulty}")
        self.log("========================================")

        self.miner.start(header, difficulty)

    # --------------------------------------------------------

    def stop_mining(self):

        self.miner.stop()

        self.status_var.set("Status: Mining stopped")

    # --------------------------------------------------------

    def handle_pow_found(self, data):

        header = data["header"]

        nonce = data["nonce"]

        digest = data["digest"]

        difficulty = data["difficulty"]

        padded = data["padded_msg"]

        valid, result = verify_pow_result(
            header,
            nonce,
            digest,
            difficulty,
            padded
        )

        if not valid:

            self.status_var.set(
                "Status: MINER HASH VERIFICATION FAILED"
            )

            self.log("REJECTED: " + str(result))

            return

        # Also check that the miner's integer hash agrees
        # with the independently calculated digest.
        if int.from_bytes(digest, "big") != int(result["digest"], 16):

            self.status_var.set("Status: HASH INTEGER MISMATCH")

            self.log("REJECTED: Hash integer mismatch")

            return

        self.pow_data = data

        self.pow_verification = result

        self.oracle_verified = False

        self.engine = ParallelMinimalSearchEngine()

        self.engine.reset_with_padded_message(padded)

        self.status_var.set(
            "Status: Hash verified. Oracle ready."
        )

        self.progress["value"] = 0

        self.progress_var.set(
            f"0 / {self.engine.bit_depth} bits"
        )

        self.log("\nPROOF OF WORK FOUND")
        self.log(f"Nonce: {nonce}")
        self.log(f"SHA-256: {result['digest']}")
        self.log(f"Leading zero bits: {result['leading_zero_bits']}")
        self.log(f"Required difficulty: {difficulty}")
        self.log(f"Hashes tested: {data['hash_count']}")
        self.log(f"Mining time: {data['elapsed']:.3f} seconds")
        self.log(f"Original message: {result['message']!r}")
        self.log(f"Padded message bytes: {len(padded)}")
        self.log(f"Oracle reconstruction depth: {self.engine.bit_depth}")
        self.log("MINER HASH VERIFICATION: PASS")
        self.log("MINER PADDING VERIFICATION: PASS")
        self.log("\nOracle is ready for bit-by-bit reconstruction.")

    # --------------------------------------------------------

    def step_once(self):

        result = self.engine.step()

        if result is None:

            if self.engine.converged:
                self.finish_oracle_verification()

            return

        self.update_progress(result)

        if result["converged"]:
            self.finish_oracle_verification()

    # --------------------------------------------------------

    def update_progress(self, result):

        total = result["total_bits"]

        solved = result["solved_bits"]

        percentage = (
            100 * solved / total
            if total
            else 0
        )

        self.progress["value"] = percentage

        self.progress_var.set(
            f"{solved} / {total} bits ({percentage:.2f}%)"
        )

        self.status_var.set(
            f"Status: Reconstructing bit {solved} of {total}"
        )

    # --------------------------------------------------------

    def start_auto(self):

        if self.engine.oracle is None:

            self.status_var.set("Status: No oracle target")

            return

        if self.engine.converged:

            self.finish_oracle_verification()

            return

        if self.auto_running:
            return

        self.auto_running = True

        self._auto_step()

    # --------------------------------------------------------

    def _auto_step(self):

        if not self.auto_running:
            return

        result = self.engine.step()

        if result is None:

            self.stop_auto()

            if self.engine.converged:
                self.finish_oracle_verification()

            return

        self.update_progress(result)

        if result["converged"]:

            self.stop_auto()

            self.finish_oracle_verification()

            return

        self.auto_job = self.root.after(
            1,
            self._auto_step
        )

    # --------------------------------------------------------

    def stop_auto(self):

        self.auto_running = False

        if self.auto_job is not None:

            try:
                self.root.after_cancel(self.auto_job)
            except tk.TclError:
                pass

            self.auto_job = None

    # --------------------------------------------------------

    def finish_oracle_verification(self):

        if self.pow_data is None:
            return

        if self.oracle_verified:
            return

        padded = self.engine.get_reconstructed_bytes()

        if padded is None:

            self.status_var.set(
                "Status: Oracle reconstruction incomplete"
            )

            return

        data = self.pow_data

        expected_message = (
            data["header"]
            + str(data["nonce"]).encode("ascii")
        )

        valid, result = verify_oracle_reconstruction(
            self.engine.current_guess,
            self.engine.bit_depth,
            expected_message,
            data["digest"],
            data["difficulty"],
            data["padded_msg"]
        )

        if not valid:

            self.status_var.set(
                "Status: ORACLE VERIFICATION FAILED"
            )

            self.log("\nORACLE VERIFICATION FAILED")
            self.log(str(result))

            return

        self.oracle_verified = True

        self.status_var.set(
            "Status: FULL SHA-256 + ORACLE VERIFIED"
        )

        self.log("\n========================================")
        self.log("FULL ORACLE RECONSTRUCTION VERIFIED")
        self.log("========================================")
        self.log(f"Recovered message: {result['message']!r}")
        self.log(f"Reconstructed bytes: {len(result['padded'])}")
        self.log(f"Reconstructed bits: {self.engine.bit_depth}")
        self.log(f"Recovered SHA-256: {result['digest']}")
        self.log(f"Leading zero bits: {result['leading_zero_bits']}")
        self.log(f"Difficulty: {result['difficulty']}")
        self.log(f"Oracle comparisons: {self.engine.oracle.query_count}")
        self.log("Padding validation: PASS")
        self.log("Exact padded-byte comparison: PASS")
        self.log("Recovered message comparison: PASS")
        self.log("Independent SHA-256 verification: PASS")
        self.log("Proof-of-work difficulty verification: PASS")
        self.log("\nALL VERIFICATION CHECKS PASSED")

    # --------------------------------------------------------

    def _poll(self):

        try:

            while True:

                event = self.events.get_nowait()

                event_type = event.get("type")

                if event_type == "pow_found":

                    self.handle_pow_found(event)

                elif event_type == "progress":

                    self.status_var.set(
                        "Status: Mining | "
                        f"Nonce {event['nonce']} | "
                        f"{event['hash_rate']:.0f} hashes/s"
                    )

                elif event_type == "stopped":

                    self.status_var.set("Status: Mining stopped")

        except queue.Empty:
            pass

        self.root.after(100, self._poll)

    # --------------------------------------------------------

    def close(self):

        self.stop_auto()

        self.miner.stop()

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():

    if "--test" in sys.argv:

        run_self_tests()

        return

    root = tk.Tk()

    app = BitPoolingOracleApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
