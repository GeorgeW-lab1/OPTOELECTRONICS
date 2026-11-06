import hashlib
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import random
import json
import tkinter as tk

from tkinter import ttk, messagebox

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 640
HEIGHT = 480

MASK32 = 0xFFFFFFFF

GRID_OPTIONS = {
    "2x2": 2,
    "3x3": 3,
    "4x4": 4,
    "8x8": 8,
}


# ============================================================
# CAMERA ENGINE
# ONE PHYSICAL CAMERA
# ============================================================

class CameraEngine:

    def __init__(self, device=0):

        self.device = device
        self.frame = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.thread = None
        self.running = False
        self.error = None
        self.fps = 0.0

    def start(self):

        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.error = None

        self.thread = threading.Thread(
            target=self.capture_loop,
            daemon=True
        )

        self.thread.start()

    def capture_loop(self):

        cap = cv2.VideoCapture(self.device)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        if not cap.isOpened():

            self.error = "Unable to open camera."
            cap.release()
            return

        self.running = True

        count = 0
        start = time.perf_counter()

        try:

            while not self.stop_event.is_set():

                ok, frame = cap.read()

                if not ok:
                    time.sleep(0.02)
                    continue

                frame = cv2.resize(
                    frame,
                    (WIDTH, HEIGHT),
                    interpolation=cv2.INTER_AREA
                )

                with self.lock:
                    self.frame = frame

                count += 1

                elapsed = time.perf_counter() - start

                if elapsed >= 1.0:

                    self.fps = count / elapsed

                    count = 0
                    start = time.perf_counter()

        finally:

            cap.release()
            self.running = False

    def snapshot(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):

        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        self.running = False


# ============================================================
# BIT ORACLE
# ============================================================

def calculate_cell(args):

    frame, x1, y1, x2, y2, index = args

    cell = frame[y1:y2, x1:x2]

    raw = np.ascontiguousarray(cell).tobytes()

    digest = hashlib.sha256(raw).digest()

    bit = digest[-1] & 1

    return {
        "index": index,
        "bit": bit,
        "hash": digest.hex(),
        "size": len(raw),
    }


def calculate_grid(frame, grid_size):

    height, width = frame.shape[:2]

    jobs = []

    index = 0

    for row in range(grid_size):

        for col in range(grid_size):

            x1 = col * width // grid_size
            x2 = (col + 1) * width // grid_size

            y1 = row * height // grid_size
            y2 = (row + 1) * height // grid_size

            jobs.append((
                frame,
                x1,
                y1,
                x2,
                y2,
                index
            ))

            index += 1

    results = [
        calculate_cell(job)
        for job in jobs
    ]

    return results


# ============================================================
# COMBINED CAMERA ORACLE
# ============================================================

def build_challenge(frame, results):

    bit_string = "".join(
        str(item["bit"]) for item in results
    )

    digest_data = b"".join(
        bytes.fromhex(item["hash"])
        for item in results
    )

    frame_digest = hashlib.sha256(
        np.ascontiguousarray(frame).tobytes()
    ).digest()

    challenge = hashlib.sha256(
        b"CAMERA_BIT_ORACLE_V1"
        + frame_digest
        + digest_data
        + bit_string.encode()
    ).digest()

    return {
        "bits": bit_string,
        "frame_hash": frame_digest.hex(),
        "challenge": challenge,
        "grid_hash": hashlib.sha256(
            digest_data
        ).hexdigest(),
        "cells": len(results),
    }


# ============================================================
# PARALLEL POW WORKER
# ============================================================

def pow_worker(
    worker_id,
    worker_count,
    challenge,
    difficulty,
    start_nonce,
    stop_event,
    result_queue,
    counts,
    bests
):

    base = hashlib.sha256(challenge)

    target = 1 << (256 - difficulty)

    nonce_space = 1 << 32

    attempts_limit = (
        (nonce_space - 1 - worker_id) // worker_count
    ) + 1

    count = 0
    best = 0

    for k in range(attempts_limit):

        if k % 4096 == 0 and stop_event.is_set():
            return

        nonce = (
            start_nonce + worker_id + k * worker_count
        ) & MASK32

        h = base.copy()

        h.update(nonce.to_bytes(4, "little"))

        digest = h.digest()

        value = int.from_bytes(digest, "big")

        zero_bits = 256 - value.bit_length()

        best = max(best, zero_bits)

        count += 1

        if value < target:

            counts[worker_id] = count
            bests[worker_id] = best

            result_queue.put((
                "HIT",
                nonce,
                digest.hex()
            ))

            return

        if count % 4096 == 0:

            counts[worker_id] = count
            bests[worker_id] = best

    counts[worker_id] = count
    bests[worker_id] = best

    result_queue.put(("DONE", worker_id))


# ============================================================
# PARALLEL POW ENGINE
# ============================================================

class PoWEngine:

    def __init__(self):

        self.processes = []

        self.running = False
        self.converged = False

        self.solution = None

        self.stop_event = None
        self.result_queue = None

        self.counts = None
        self.bests = None

        self.workers = 1

        self.scanned = 0
        self.best_bits = 0

        self.hashrate = 0
        self.elapsed = 0

        self.start_time = None

        self.status = "READY"

        self.challenge = None
        self.difficulty = 0
        self.start_nonce = 0

        self.done_workers = set()

    def start(self, challenge, difficulty, workers):

        self.stop()

        if not 1 <= difficulty <= 32:
            raise ValueError("Difficulty must be 1-32.")

        workers = max(1, workers)

        self.solution = None
        self.converged = False

        self.scanned = 0
        self.best_bits = 0
        self.hashrate = 0
        self.elapsed = 0

        self.challenge = challenge
        self.difficulty = difficulty
        self.workers = workers

        self.start_nonce = random.randrange(1 << 32)

        self.done_workers = set()

        ctx = mp.get_context("spawn")

        self.stop_event = ctx.Event()
        self.result_queue = ctx.Queue()

        self.counts = ctx.RawArray("Q", workers)
        self.bests = ctx.RawArray("i", workers)

        self.processes = []

        for i in range(workers):

            process = ctx.Process(
                target=pow_worker,
                daemon=True,
                args=(
                    i,
                    workers,
                    challenge,
                    difficulty,
                    self.start_nonce,
                    self.stop_event,
                    self.result_queue,
                    self.counts,
                    self.bests
                )
            )

            process.start()

            self.processes.append(process)

        self.start_time = time.perf_counter()

        self.running = True
        self.status = "SCANNING"

    def poll(self):

        if not self.running:
            return

        self.scanned = sum(self.counts)

        self.best_bits = max(
            [0] + list(self.bests)
        )

        self.elapsed = (
            time.perf_counter() - self.start_time
        )

        if self.elapsed > 0:
            self.hashrate = self.scanned / self.elapsed

        try:

            while True:

                message = self.result_queue.get_nowait()

                if message[0] == "HIT":

                    if self.solution is None:

                        _, nonce, digest = message

                        check = hashlib.sha256(
                            self.challenge
                            + nonce.to_bytes(4, "little")
                        ).hexdigest()

                        if check != digest:
                            continue

                        zero_bits = (
                            256
                            - int(digest, 16).bit_length()
                        )

                        if zero_bits < self.difficulty:
                            continue

                        self.solution = {
                            "nonce": nonce,
                            "nonce_hex": hex(nonce),
                            "hash": digest,
                            "zero_bits": zero_bits,
                            "difficulty": self.difficulty,
                            "challenge": self.challenge.hex(),
                        }

                        self.converged = True
                        self.status = "LOCKED & REVEALED"

                        self.stop()

                        return

                elif message[0] == "DONE":

                    self.done_workers.add(message[1])

        except queue.Empty:
            pass

        if len(self.done_workers) >= self.workers:

            self.converged = True
            self.status = "NONCE SPACE EXHAUSTED"

            self.stop()

    def stop(self):

        if self.stop_event is not None:
            self.stop_event.set()

        for process in self.processes:

            if process.is_alive():

                process.join(timeout=0.5)

                if process.is_alive():

                    process.terminate()
                    process.join(timeout=0.5)

        self.processes = []

        self.running = False

        if self.status == "SCANNING":
            self.status = "STOPPED"

    def registers(self):

        nonce = (
            self.solution["nonce"]
            if self.solution is not None
            else None
        )

        return {
            "NONCE": nonce,
            "CURSOR": self.scanned & MASK32,
            "HASH32": self.best_bits & MASK32,
            "POOLED": (
                self.scanned ^ self.best_bits
            ) & MASK32
        }


# ============================================================
# TKINTER APPLICATION
# ============================================================

class CameraBitOracleApp:

    def __init__(self, root):

        self.root = root

        root.title(
            "SHA-256 Camera Bit Oracle Lab"
        )

        root.geometry("1150x900")
        root.minsize(800, 600)
        root.configure(bg="#10151c")

        self.camera = CameraEngine(0)
        self.engine = PoWEngine()

        self.grid_size = 4

        self.current_results = []
        self.current_challenge = None

        self.photo = None
        self.running = True

        self.last_result_key = None

        self._build_ui()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.camera.start()

        self.root.after(80, self.refresh)

    # ========================================================
    # UI
    # ========================================================

    def _build_ui(self):

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TFrame",
            background="#10151c"
        )

        style.configure(
            "TLabel",
            background="#10151c",
            foreground="#dce7f5"
        )

        style.configure(
            "TLabelframe",
            background="#10151c",
            foreground="#00d9ff"
        )

        style.configure(
            "TLabelframe.Label",
            background="#10151c",
            foreground="#00d9ff"
        )

        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        self.scroll_canvas = tk.Canvas(
            outer,
            bg="#10151c",
            highlightthickness=0
        )

        scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=self.scroll_canvas.yview
        )

        self.scroll_canvas.configure(
            yscrollcommand=scrollbar.set
        )

        scrollbar.pack(side="right", fill="y")

        self.scroll_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        main = ttk.Frame(
            self.scroll_canvas,
            padding=8
        )

        self.main_window = self.scroll_canvas.create_window(
            (0, 0),
            window=main,
            anchor="nw"
        )

        main.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            )
        )

        self.scroll_canvas.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.itemconfigure(
                self.main_window,
                width=e.width
            )
        )

        self.scroll_canvas.bind_all(
            "<MouseWheel>",
            self.mousewheel
        )

        self.scroll_canvas.bind_all(
            "<Button-4>",
            lambda e: self.scroll_canvas.yview_scroll(-3, "units")
        )

        self.scroll_canvas.bind_all(
            "<Button-5>",
            lambda e: self.scroll_canvas.yview_scroll(3, "units")
        )

        # CAMERA CONTROLS

        controls = ttk.LabelFrame(
            main,
            text="CAMERA CONTROLS",
            padding=8
        )

        controls.pack(
            fill="x",
            pady=5
        )

        ttk.Label(
            controls,
            text="Grid size:"
        ).pack(side="left", padx=5)

        self.grid_combo = ttk.Combobox(
            controls,
            values=list(GRID_OPTIONS.keys()),
            state="readonly",
            width=8
        )

        self.grid_combo.set("4x4")

        self.grid_combo.pack(
            side="left",
            padx=5
        )

        self.grid_combo.bind(
            "<<ComboboxSelected>>",
            self.change_grid
        )

        ttk.Button(
            controls,
            text="Start Camera",
            command=self.camera.start
        ).pack(side="left", padx=5)

        ttk.Button(
            controls,
            text="Stop Camera",
            command=self.camera.stop
        ).pack(side="left", padx=5)

        self.camera_label = ttk.Label(
            controls,
            text="CAMERA STARTING"
        )

        self.camera_label.pack(
            side="right",
            padx=10
        )

        # CAMERA GRID

        camera_box = ttk.LabelFrame(
            main,
            text="NORMAL CAMERA IMAGE / BIT ORACLES",
            padding=5
        )

        camera_box.pack(
            fill="both",
            expand=True,
            pady=5
        )

        self.canvas = tk.Canvas(
            camera_box,
            width=640,
            height=480,
            bg="#05080c",
            highlightthickness=0
        )

        self.canvas.pack(
            padx=5,
            pady=5
        )

        # ORACLE CONTROLS

        oracle_box = ttk.LabelFrame(
            main,
            text="PARALLEL PROOF OF WORK",
            padding=8
        )

        oracle_box.pack(
            fill="x",
            pady=5
        )

        ttk.Label(
            oracle_box,
            text="Difficulty:"
        ).pack(side="left", padx=4)

        self.diff = ttk.Spinbox(
            oracle_box,
            from_=1,
            to=32,
            width=5
        )

        self.diff.set(20)

        self.diff.pack(
            side="left",
            padx=5
        )

        ttk.Label(
            oracle_box,
            text="Workers:"
        ).pack(side="left", padx=4)

        self.worker_count = ttk.Spinbox(
            oracle_box,
            from_=1,
            to=os.cpu_count() or 2,
            width=5
        )

        self.worker_count.set(
            max(1, (os.cpu_count() or 2) - 1)
        )

        self.worker_count.pack(
            side="left",
            padx=5
        )

        ttk.Button(
            oracle_box,
            text="SOLVE",
            command=self.solve
        ).pack(side="left", padx=6)

        ttk.Button(
            oracle_box,
            text="STOP",
            command=self.stop
        ).pack(side="left", padx=5)

        ttk.Button(
            oracle_box,
            text="RESET",
            command=self.reset
        ).pack(side="left", padx=5)

        # STATUS

        self.status_label = ttk.Label(
            main,
            text="READY",
            font=("Consolas", 10)
        )

        self.status_label.pack(
            fill="x",
            pady=5
        )

        # BINARY REGISTER

        binary_box = ttk.LabelFrame(
            main,
            text="COMBINED CAMERA BINARY REGISTER",
            padding=8
        )

        binary_box.pack(
            fill="x",
            pady=5
        )

        self.binary_label = ttk.Label(
            binary_box,
            text="Waiting for camera frame...",
            font=("Consolas", 12),
            wraplength=1000
        )

        self.binary_label.pack(
            fill="x"
        )

        # HARDWARE REGISTERS

        register_box = ttk.LabelFrame(
            main,
            text="32-BIT HARDWARE REGISTERS",
            padding=5
        )

        register_box.pack(
            fill="x",
            pady=5
        )

        self.reg_canvas = tk.Canvas(
            register_box,
            height=125,
            bg="#080d13",
            highlightthickness=0
        )

        self.reg_canvas.pack(
            fill="x"
        )

        # RESULT

        result_box = ttk.LabelFrame(
            main,
            text="SOLVED ORACLE",
            padding=5
        )

        result_box.pack(
            fill="both",
            pady=5
        )

        self.result_text = tk.Text(
            result_box,
            height=8,
            bg="#080d13",
            fg="#00e5ff",
            font=("Consolas", 9)
        )

        self.result_text.pack(
            fill="both",
            expand=True
        )

        self.draw_registers()

    # ========================================================
    # SCROLL WHEEL
    # ========================================================

    def mousewheel(self, event):

        if event.delta:

            self.scroll_canvas.yview_scroll(
                int(-event.delta / 120),
                "units"
            )

    # ========================================================
    # GRID CONFIGURATION
    # ========================================================

    def change_grid(self, event=None):

        self.grid_size = GRID_OPTIONS[
            self.grid_combo.get()
        ]

        self.current_results = []
        self.current_challenge = None

    # ========================================================
    # CAMERA DISPLAY
    # ========================================================

    def update_camera_grid(self):

        frame = self.camera.snapshot()

        if frame is None:
            return

        results = calculate_grid(
            frame,
            self.grid_size
        )

        challenge = build_challenge(
            frame,
            results
        )

        self.current_results = results
        self.current_challenge = challenge

        display = frame.copy()

        height, width = display.shape[:2]

        for row in range(self.grid_size):

            for col in range(self.grid_size):

                index = (
                    row * self.grid_size + col
                )

                x1 = col * width // self.grid_size
                x2 = (col + 1) * width // self.grid_size

                y1 = row * height // self.grid_size
                y2 = (row + 1) * height // self.grid_size

                bit = results[index]["bit"]

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 255),
                    1
                )

                cv2.circle(
                    display,
                    (x1 + 20, y1 + 20),
                    13,
                    (0, 220, 0) if bit else (0, 100, 255),
                    -1
                )

                cv2.putText(
                    display,
                    str(bit),
                    (x1 + 16, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                cv2.putText(
                    display,
                    f"BIT {index}",
                    (x1 + 38, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1
                )

                short_hash = results[index]["hash"][:8]

                cv2.putText(
                    display,
                    short_hash,
                    (x1 + 7, y2 - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1
                )

        rgb = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        ppm = (
            b"P6 640 480 255\n"
            + np.ascontiguousarray(rgb).tobytes()
        )

        self.photo = tk.PhotoImage(
            data=ppm,
            format="PPM"
        )

        self.canvas.delete("all")

        self.canvas.create_image(
            0,
            0,
            image=self.photo,
            anchor="nw"
        )

        self.binary_label.configure(
            text=(
                f"{challenge['bits']}\n"
                f"CELLS: {challenge['cells']} | "
                f"GRID HASH: {challenge['grid_hash'][:32]}"
            )
        )

    # ========================================================
    # SOLVER
    # ========================================================

    def solve(self):

        if self.engine.running:
            return

        if self.current_challenge is None:

            messagebox.showerror(
                "Camera",
                "Waiting for a valid camera frame."
            )

            return

        try:

            difficulty = int(self.diff.get())
            workers = int(self.worker_count.get())

            if not 1 <= difficulty <= 32:
                raise ValueError(
                    "Difficulty must be between 1 and 32."
                )

            if workers < 1:
                raise ValueError(
                    "Workers must be at least 1."
                )

            if difficulty > 25:

                if not messagebox.askyesno(
                    "Heavy computation",
                    f"Expected work: approximately "
                    f"{2**difficulty:,} hashes.\n\n"
                    "Continue?"
                ):
                    return

            frozen_challenge = self.current_challenge[
                "challenge"
            ]

            self.engine.start(
                frozen_challenge,
                difficulty,
                workers
            )

            self.last_result_key = None

            self.result_text.delete(
                "1.0",
                "end"
            )

        except Exception as e:

            messagebox.showerror(
                "Solver Error",
                str(e)
            )

    def stop(self):

        self.engine.stop()

    def reset(self):

        self.engine.stop()

        self.engine.solution = None
        self.engine.scanned = 0
        self.engine.best_bits = 0

        self.engine.elapsed = 0
        self.engine.hashrate = 0

        self.engine.converged = False
        self.engine.status = "READY"

        self.last_result_key = None

        self.result_text.delete(
            "1.0",
            "end"
        )

        self.draw_registers()

    # ========================================================
    # REGISTER DRAWING
    # ========================================================

    def draw_registers(self):

        canvas = self.reg_canvas

        canvas.delete("all")

        registers = self.engine.registers()

        rows = [
            ("NONCE", registers["NONCE"], "#00e676"),
            ("CURSOR", registers["CURSOR"], "#ffb300"),
            ("HASH32", registers["HASH32"], "#40c4ff"),
            ("POOLED", registers["POOLED"], "#b388ff")
        ]

        for row, (name, value, color) in enumerate(rows):

            y = 7 + row * 29

            canvas.create_text(
                8,
                y + 8,
                text=name,
                fill="#a0b8d0",
                anchor="w",
                font=("Consolas", 9, "bold")
            )

            hidden = (
                name == "NONCE"
                and value is None
            )

            val = value or 0

            x = 85

            for bit in range(31, -1, -1):

                bit_value = (val >> bit) & 1

                canvas.create_rectangle(
                    x,
                    y,
                    x + 14,
                    y + 17,
                    fill=(
                        color
                        if bit_value and not hidden
                        else "#1e2834"
                    ),
                    outline="#34475a"
                )

                canvas.create_text(
                    x + 7,
                    y + 8,
                    text=(
                        "?" if hidden
                        else str(bit_value)
                    ),
                    fill="#ffffff",
                    font=("Consolas", 7)
                )

                x += 15

            canvas.create_text(
                x + 5,
                y + 8,
                text=(
                    "[HIDDEN]"
                    if hidden
                    else f"[{value}]"
                ),
                fill="#dce7f5",
                anchor="w",
                font=("Consolas", 8)
            )

    # ========================================================
    # REFRESH LOOP
    # ========================================================

    def refresh(self):

        if not self.running:
            return

        if self.camera.running:

            self.update_camera_grid()

            self.camera_label.configure(
                text=f"CAMERA FPS: {self.camera.fps:.1f}"
            )

        else:

            self.camera_label.configure(
                text=(
                    self.camera.error
                    or "CAMERA STOPPED"
                )
            )

        if self.engine.running:
            self.engine.poll()

        e = self.engine

        difficulty = (
            e.difficulty if e.difficulty else 0
        )

        self.status_label.configure(
            text=(
                f"{e.status} | "
                f"WORKERS: {e.workers} | "
                f"HASHES: {e.scanned:,} | "
                f"RATE: {e.hashrate:,.0f} H/s | "
                f"BEST: {e.best_bits}/{difficulty} | "
                f"TIME: {e.elapsed:.1f}s"
            )
        )

        self.draw_registers()

        if e.solution is not None:

            result_key = e.solution["hash"]

            if result_key != self.last_result_key:

                self.result_text.delete(
                    "1.0",
                    "end"
                )

                self.result_text.insert(
                    "end",
                    json.dumps(
                        e.solution,
                        indent=4
                    )
                )

                self.last_result_key = result_key

        self.root.after(
            100,
            self.refresh
        )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def close(self):

        self.running = False

        self.engine.stop()
        self.camera.stop()

        self.root.destroy()


# ============================================================
# SELF TEST
# ============================================================

def selftest():

    print("CAMERA BIT ORACLE SELF TEST")

    frame = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    for y in range(HEIGHT):
        for x in range(WIDTH):

            frame[y, x] = (
                x % 256,
                y % 256,
                (x + y) % 256
            )

    for name, grid_size in GRID_OPTIONS.items():

        results = calculate_grid(
            frame,
            grid_size
        )

        challenge = build_challenge(
            frame,
            results
        )

        assert len(results) == grid_size ** 2

        assert len(challenge["bits"]) == grid_size ** 2

        assert all(
            item["bit"] in (0, 1)
            for item in results
        )

        print(
            name,
            "cells:",
            len(results),
            "bits:",
            challenge["bits"]
        )

    challenge = hashlib.sha256(
        b"camera oracle test"
    ).digest()

    difficulty = 8

    target = 1 << (256 - difficulty)

    found = False

    for nonce in range(100000):

        digest = hashlib.sha256(
            challenge + nonce.to_bytes(4, "little")
        ).digest()

        if int.from_bytes(digest, "big") < target:

            found = True

            print(
                "PoW verified:",
                nonce,
                digest.hex()
            )

            break

    assert found

    print("ALL SELF TESTS PASSED")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    if "--selftest" in sys.argv:

        selftest()

    else:

        root = tk.Tk()

        app = CameraBitOracleApp(root)

        root.mainloop()
