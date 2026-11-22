import bisect
import csv
import math
import queue
import random
import re
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    from PIL import Image, ImageTk
    HAS_WEBCAM_LIBS = True
except ImportError:
    HAS_WEBCAM_LIBS = False


"""
END-TO-END PHOTONIC + MODAL LOGIC TIMING SIMULATOR
(SELECT -> DRIVER A, 256-BIT RANDOM BINARY QUEUE BETWEEN OR -> 74HC74,
 NO-FLIP PROBE COMPARING QUEUE VS. AND BEFORE 150 ms DELAY)

Photonic & logic chain
----------------------
Laser (P_peak mW, extinction ratio ER dB, RIN) or USB Webcam ROI A/B
  -> Beam splitter / coupler (split A/B, insertion loss L_ins dB)
  -> Photodetectors A & B (responsivity R A/W)
  -> TIA + RC node (tau = 50 ms, 0..5 V rail, V > 0.7 V comparator -> b)
  -> Logic core:
       * Driver A -> a (optionally gated by SELECT / Q feedback)
       * pre_delay_and = a AND b (BEFORE 150 ms delay line) -> delayed_and
       * Y = a OR b; D = delayed_and OR Y
       * 256-bit Random Binary FIFO Queue between OR and 74HC74 -> d_queued
       * 74HC74 D-FF clocked at 10 Hz (100 ms) sampling d_queued: SELECT = Q, Observe = NOT Q
       * Running-AND accumulator (with optional startup-delay masking)
       * Noise-free (p = 0) probe comparing 256-bit queue output against pre-delay AND

Subset-sum add-on
-----------------
N random integers (b bits each) plus a random target T (or a set loaded from file).
The candidate subset is a bit register; the error-match signal is
E = sum(selected) - T (numeric error) and popcount(sum XOR T) (bit-level DIFF count,
giving a match rate). Engines: Swap search (greedy + add/remove/1-1 swap moves found
by bisect, with random kicks), Anneal, and Random queue (blind baseline). Hardness
depends on density n/b: dense (n >> b) is easy, while n ~ b is the genuinely hard
regime where these heuristics only approximate. Heuristic search, not a
polynomial-time solver.
"""


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)  # allow printing/parsing very large subset-sum integers


def parse_subset_file(path: str):
    """Read a subset-sum set from text/CSV. Returns (values, target_or_None, zeros_skipped).

    Accepted: integers separated by whitespace/commas/semicolons, any number of lines.
    Target (optional): a line "target: N" / "T=N", or the export format "# target" + next line.
    Lines starting with # or // are comments.
    """
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()
    values: list[int] = []
    target = None
    zeros = 0
    next_is_target = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if re.fullmatch(r"#\s*target\s*", low):
            next_is_target = True
            continue
        m = re.match(r"^[#/]*\s*(?:target|sum|t)\s*[:=]\s*(\d+)\s*$", low)
        if m:
            target = int(m.group(1))
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        for tok in re.split(r"[,\s;]+", line):
            if not tok:
                continue
            if not re.fullmatch(r"[+-]?\d+", tok):
                raise ValueError(f"Not an integer: {tok[:30]!r}")
            val = int(tok)
            if next_is_target:
                target, next_is_target = val, False
                continue
            if val < 0:
                raise ValueError("Negative numbers are not supported (positive integers only).")
            if val == 0:
                zeros += 1
                continue
            values.append(val)
    if target is not None and target < 0:
        raise ValueError("Target must be >= 0.")
    if len(values) < 2:
        raise ValueError("Need at least 2 positive integers in the file.")
    return values, target, zeros


def mw_to_dbm(power_mw: float) -> float:
    return 10.0 * math.log10(max(power_mw, 1e-12))


def db_to_linear(db_val: float) -> float:
    return 10.0 ** (db_val / 10.0)


class WebcamWorker:
    """Captures latest frames from a USB webcam outside the Tkinter UI thread."""

    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self, timeout: float = 1.5):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def _put_latest(self, item):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            pass

    def _capture_loop(self):
        capture = None
        try:
            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                self.error = (
                    f"Cannot open USB camera {self.camera_index}. "
                    "Check connection/permissions or close other camera apps."
                )
                return

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.error = "USB camera stopped returning frames."
                    break
                self._put_latest((time.perf_counter(), frame))
        except Exception as exc:
            self.error = str(exc)
        finally:
            if capture is not None:
                capture.release()

    def latest_frame(self):
        latest = None
        while True:
            try:
                latest = self.frames.get_nowait()
            except queue.Empty:
                return latest


class PhotonicBenchModel:
    """End-to-end photonic + logic discrete-step simulation engine."""

    DT = 0.005
    RC_TAU = 0.050
    SUPPLY = 5.0
    THRESHOLD = 0.7
    AND_DELAY = 0.150
    CLOCK_PERIOD = 0.100
    INPUT_PERIOD = 0.100

    def __init__(
        self,
        probe_duration: float = 2.0,
        seed: int = 7,
        mode: str = "Random",
        ignore_startup: bool = False,
        manual_x: bool = False,
        feed_select_to_driver_a: bool = True,
        d_queue_length: int = 256,
        laser_power_mw: float = 10.0,
        extinction_db: float = 20.0,
        split_ratio_a: float = 0.5,
        insertion_loss_db: float = 1.5,
        responsivity_a_per_w: float = 0.85,
        tia_gain_v_per_ma: float = 2.5,
        delay_loss_db: float = 0.5,
        optical_noise_std: float = 0.01,
    ):
        self.probe_steps = max(1, int(round(probe_duration / self.DT)))
        self.total_steps = 2 * self.probe_steps
        self.probe_duration = self.probe_steps * self.DT
        self.total_duration = self.total_steps * self.DT

        self.flip_probability = 0.0
        self.seed = seed
        self.mode = mode
        self.ignore_startup = ignore_startup
        self.manual_x = manual_x
        self.feed_select_to_driver_a = feed_select_to_driver_a
        self.d_queue_length = max(1, int(d_queue_length))

        self.laser_power_mw = laser_power_mw
        self.extinction_db = extinction_db
        self.split_ratio_a = split_ratio_a
        self.insertion_loss_db = insertion_loss_db
        self.responsivity_a_per_w = responsivity_a_per_w
        self.tia_gain_v_per_ma = tia_gain_v_per_ma
        self.delay_loss_db = delay_loss_db
        self.optical_noise_std = optical_noise_std

        self.delay_steps = max(1, int(round(self.AND_DELAY / self.DT)))
        self.clock_steps = max(1, int(round(self.CLOCK_PERIOD / self.DT)))
        self.input_steps = max(1, int(round(self.INPUT_PERIOD / self.DT)))

        self.reset()

    def set_queue_length(self, new_length: int):
        new_length = max(1, int(new_length))
        self.d_queue_length = new_length
        while len(self.d_bit_queue) > new_length:
            self.d_bit_queue.popleft()
        while len(self.d_bit_queue) < new_length:
            self.d_bit_queue.appendleft(self.queue_rng.randrange(2))
        self.d_queued = self.d_bit_queue[0]

    def load_queue_bits(self, bits):
        """Load a bit list (e.g. a subset selection) into the random queue, zero padded."""
        bits = [int(b) & 1 for b in bits][: self.d_queue_length]
        bits += [0] * (self.d_queue_length - len(bits))
        self.d_bit_queue = deque(bits)
        self.d_queued = self.d_bit_queue[0]

    def reset(self):
        self.input_rng = random.Random(self.seed)
        self.photonic_rng = random.Random(self.seed + 2)
        self.queue_rng = random.Random(self.seed + 3)

        self.step_index = 0
        self.time = 0.0

        self.x = 0
        self.random_a_bit = 1
        self.optical_tx_mw = 0.0
        self.optical_a_mw = 0.0
        self.optical_b_mw = 0.0
        self.photocurrent_a_ua = 0.0
        self.photocurrent_b_ua = 0.0
        self.optical_delayed_mw = 0.0
        self.optical_phase_rad = 0.0

        self.driver_a_raw = 0
        self.driver_a_select_gate = 1
        self.a = 0
        self.b = 0
        self.rc_voltage = 0.0

        # AND BEFORE the 150 ms delay line
        self.pre_delay_and = 0
        self.truth = 0
        self.delayed_and = 0
        self.y = 0
        self.d = 0

        # Pre-fill the queue with d_queue_length (default 256) random binary bits
        self.d_bit_queue: deque[int] = deque(
            self.queue_rng.randrange(2) for _ in range(self.d_queue_length)
        )
        self.d_queued = self.d_bit_queue[0]

        self.post_queue_and = 0
        self.probe_count = 0
        self.probe_matches = 0
        self.probe_mismatches = 0
        self.last_match: int | None = None
        self.last_lag_steps = max(0, self.d_queue_length - 1)
        self.lag_steps_sum = 0

        self.clock = 0
        self.q = 0
        self.not_q = 1

        self.accumulator = 1
        self.accumulator_samples = 0

        self.and_history: list[int] = []
        self.optical_and_history: list[float] = []
        self.records: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.step_index >= self.total_steps

    def _update_inputs(self, cam_a: int | None = None, cam_b: int | None = None):
        if self.mode == "USB Webcam" and cam_a is not None and cam_b is not None:
            if self.step_index % self.input_steps == 0:
                self.random_a_bit = self.input_rng.randrange(2)
            self.x = int(cam_b)
            self.driver_a_raw = int(cam_a) & self.random_a_bit
            return

        if self.step_index % self.input_steps != 0:
            return
        update_num = self.step_index // self.input_steps
        if self.mode == "Random":
            self.x = self.input_rng.randrange(2)
            self.driver_a_raw = self.input_rng.randrange(2)
        elif self.mode == "Square wave":
            self.x = update_num % 2
            self.driver_a_raw = self.input_rng.randrange(2)
        elif self.mode == "Manual":
            self.x = int(bool(self.manual_x))
            self.driver_a_raw = self.input_rng.randrange(2)
        else:  # "Direct (x -> a, b)"
            self.x = int(bool(self.manual_x))
            self.driver_a_raw = self.x

    def step(
        self,
        cam_a: int | None = None,
        cam_b: int | None = None,
        brightness_a: float = 0.0,
        brightness_b: float = 0.0,
        threshold_px: float = 130.0,
    ) -> bool:
        if self.finished:
            return False

        self._update_inputs(cam_a=cam_a, cam_b=cam_b)

        self.driver_a_select_gate = self.q if self.feed_select_to_driver_a else 1
        self.a = self.driver_a_raw & self.driver_a_select_gate

        er_linear = db_to_linear(self.extinction_db)
        low_mw = self.laser_power_mw / max(er_linear, 1.0)
        base_tx_mw = self.laser_power_mw if self.x else low_mw
        rin_factor = 1.0 + self.photonic_rng.gauss(0.0, self.optical_noise_std)
        self.optical_tx_mw = max(0.0, base_tx_mw * rin_factor)

        self.optical_phase_rad = (
            self.optical_phase_rad + 2.0 * math.pi * 1.5 * self.DT
        ) % (2.0 * math.pi)

        loss_lin = db_to_linear(-self.insertion_loss_db)
        split_a = min(max(self.split_ratio_a, 0.0), 1.0)
        split_b = 1.0 - split_a

        self.optical_a_mw = self.optical_tx_mw * split_a * loss_lin
        self.optical_b_mw = self.optical_tx_mw * split_b * loss_lin

        self.photocurrent_a_ua = max(0.0, self.optical_a_mw * self.responsivity_a_per_w * 1000.0)
        self.photocurrent_b_ua = max(0.0, self.optical_b_mw * self.responsivity_a_per_w * 1000.0)

        if self.mode == "Photonic TIA":
            drive_v_target = min(
                self.SUPPLY,
                (self.photocurrent_b_ua / 1000.0) * self.tia_gain_v_per_ma,
            )
        else:
            drive_v_target = self.SUPPLY * self.x

        decay = math.exp(-self.DT / self.RC_TAU)
        self.rc_voltage = drive_v_target + (self.rc_voltage - drive_v_target) * decay
        self.b = int(self.rc_voltage > self.THRESHOLD)

        # AND BEFORE the 150 ms delay line
        self.pre_delay_and = self.a & self.b
        self.truth = self.pre_delay_and
        self.and_history.append(self.pre_delay_and)

        opt_and_mw = self.optical_tx_mw if self.pre_delay_and else 0.0
        self.optical_and_history.append(opt_and_mw)

        delayed_idx = self.step_index - self.delay_steps
        if delayed_idx >= 0:
            self.delayed_and = self.and_history[delayed_idx]
            self.optical_delayed_mw = (
                self.optical_and_history[delayed_idx] * db_to_linear(-self.delay_loss_db)
            )
        else:
            self.delayed_and = 0
            self.optical_delayed_mw = 0.0

        self.y = self.a | self.b
        self.d = self.delayed_and | self.y

        # Shift current D into the 256-bit random binary FIFO and pop oldest bit
        self.d_bit_queue.append(int(self.d))
        if len(self.d_bit_queue) > self.d_queue_length:
            self.d_queued = self.d_bit_queue.popleft()
        else:
            self.d_queued = self.d_bit_queue[0]
        effective_queue_steps = max(0, len(self.d_bit_queue) - 1)

        self.post_queue_and = self.d_queued & self.b
        self.last_lag_steps = effective_queue_steps

        completed = self.step_index + 1
        self.time = completed * self.DT

        phase = completed % self.clock_steps
        self.clock = int(phase < self.clock_steps // 2)
        rising_edge = phase == 0

        if rising_edge:
            self.q = self.d_queued
            self.not_q = 1 - self.q
            if not self.ignore_startup or delayed_idx >= 0:
                self.accumulator &= self.delayed_and
                self.accumulator_samples += 1

        # Compare queue output (D_queued) directly against AND BEFORE delay (pre_delay_and), no flip noise
        probe_active = completed <= self.probe_steps
        match_cell: int | str = ""
        if probe_active:
            matched = int(self.d_queued == self.pre_delay_and) #not equal or equal still represents 1/4th success in this case, post processing could determine 3/4th correctness
            self.last_match = matched
            self.probe_count += 1
            self.probe_matches += matched
            self.probe_mismatches += 1 - matched
            self.lag_steps_sum += self.last_lag_steps
            match_cell = matched

        self.records.append({
            "time_s": round(self.time, 6),
            "mode": self.mode,
            "optical_x": self.x,
            "tx_power_mw": round(self.optical_tx_mw, 5),
            "tx_power_dbm": round(mw_to_dbm(self.optical_tx_mw), 3),
            "rx_a_mw": round(self.optical_a_mw, 5),
            "rx_b_mw": round(self.optical_b_mw, 5),
            "iph_a_ua": round(self.photocurrent_a_ua, 3),
            "iph_b_ua": round(self.photocurrent_b_ua, 3),
            "delayed_opt_mw": round(self.optical_delayed_mw, 5),
            "brightness_A": round(brightness_a, 2),
            "brightness_B": round(brightness_b, 2),
            "brightness_threshold": round(threshold_px, 2),
            "driver_a_raw": self.driver_a_raw,
            "driver_a_select_feedback": int(self.feed_select_to_driver_a),
            "driver_a_select_gate": self.driver_a_select_gate,
            "a": self.a,
            "rc_voltage_V": round(self.rc_voltage, 6),
            "b": self.b,
            "pre_delay_and": self.pre_delay_and,
            "delayed_and": self.delayed_and,
            "Y_or": self.y,
            "D_raw_or": self.d,
            "d_queue_len": len(self.d_bit_queue),
            "queue_lag_steps": self.last_lag_steps,
            "queue_lag_ms": round(self.last_lag_steps * self.DT * 1000.0, 2),
            "D_queued": self.d_queued,
            "post_queue_and": self.post_queue_and,
            "probe_active": int(probe_active),
            "queue_vs_pre_delay_and_match": match_cell,
            "clock": self.clock,
            "rising_edge": int(rising_edge),
            "Q_SELECT": self.q,
            "not_Q_Observe": self.not_q,
            "running_and": self.accumulator,
            "accumulator_samples": self.accumulator_samples,
        })

        self.step_index = completed
        return True


class MatchSubsetSumSolver:
    """Subset-sum search driven by the error between the subset sum and the target."""

    ENGINES = ("Swap search", "Anneal", "Random queue")
    TARGET_MODES = ("Planted subset", "Random in [0,total]")

    def __init__(self, n: int, bits: int = 32, target_mode: str = "Planted subset",
                 engine: str = "Swap search", max_steps: int = 20_000, seed: int = 7,
                 values=None, target=None):
        if values is not None:                       # loaded set from file
            values = [int(v) for v in values]
            if len(values) < 2 or min(values) < 1:
                raise ValueError("Need at least 2 positive integers.")
            n = len(values)
            bits = max(2, max(v.bit_length() for v in values))
        if n < 2:
            raise ValueError("Need at least 2 numbers.")
        if values is None and not (2 <= bits <= 20_000):
            raise ValueError("Bits per number must be in [2, 20000].")
        self.n = int(n)
        self.bits = int(bits)
        self.engine = engine
        self.max_steps = int(max_steps)

        prob_rng = random.Random(seed + 301)
        self.rng = random.Random(seed + 401)
        self.values = values if values is not None else [
            prob_rng.randrange(1, 1 << self.bits) for _ in range(self.n)
        ]
        self.total = sum(self.values)
        if target is not None:
            self.target = int(target)
        elif target_mode == "Planted subset":
            self.target = sum(v for v in self.values if prob_rng.getrandbits(1))
        else:
            self.target = prob_rng.randrange(0, self.total + 1)
        self.density = self.n / self.bits

        self.sel = [0] * self.n
        self.cur = 0

    # --- error matching signals ---
    @property
    def error(self) -> int:
        """Signed numeric error: subset sum minus target."""
        return self.cur - self.target

    def bit_diffs(self, value: int | None = None) -> tuple[int, int]:
        v = self.cur if value is None else value
        width = max(v.bit_length(), self.target.bit_length(), 1)
        return bin(v ^ self.target).count("1"), width

    # --- helpers ---
    def _greedy_init(self):
        self.sel = [0] * self.n
        self.cur = 0
        for i in sorted(range(self.n), key=lambda k: -self.values[k]):
            if self.cur + self.values[i] <= self.target:
                self.sel[i] = 1
                self.cur += self.values[i]

    def _rebuild(self):
        self.in_list = sorted((self.values[i], i) for i in range(self.n) if self.sel[i])
        self.out_list = sorted((self.values[i], i) for i in range(self.n) if not self.sel[i])

    def _toggle(self, i: int):
        v = self.values[i]
        if self.sel[i]:
            del self.in_list[bisect.bisect_left(self.in_list, (v, i))]
            bisect.insort(self.out_list, (v, i))
            self.sel[i] = 0
            self.cur -= v
        else:
            del self.out_list[bisect.bisect_left(self.out_list, (v, i))]
            bisect.insort(self.in_list, (v, i))
            self.sel[i] = 1
            self.cur += v

    @staticmethod
    def _near(lst, want):
        """Entries of a sorted (value, idx) list closest to the wanted value."""
        pos = bisect.bisect_left(lst, (want,))
        for p in (pos - 1, pos):
            if 0 <= p < len(lst):
                yield lst[p]

    def _best_move(self, sample_cap: int = 3000):
        """Best add / remove / 1-1 swap by resulting |error|. Returns (abs_err, toggles)."""
        e = self.error
        best = (abs(e), None)
        for v, i in self._near(self.out_list, -e):                 # add one
            if abs(e + v) < best[0]:
                best = (abs(e + v), (i,))
        for v, i in self._near(self.in_list, e):                   # remove one
            if abs(e - v) < best[0]:
                best = (abs(e - v), (i,))
        in_list = self.in_list
        if len(in_list) <= sample_cap:
            picks = in_list
        else:
            picks = [in_list[self.rng.randrange(len(in_list))] for _ in range(sample_cap)]
        for a, ia in picks:                                        # swap in/out
            for b, ib in self._near(self.out_list, a - e):
                ne = abs(e - a + b)
                if ne < best[0]:
                    best = (ne, (ia, ib))
        return best

    def run(self, stop_event=None, progress=None) -> dict:
        t0 = time.perf_counter()
        step = 0
        stopped = lambda: stop_event is not None and stop_event.is_set()

        if self.engine == "Random queue":
            limit = min(self.max_steps, max(50, 2_000_000 // self.n))
            best_abs, best_sel = None, [0] * self.n
            while step < limit and best_abs != 0 and not stopped():
                self.sel = [self.rng.getrandbits(1) for _ in range(self.n)]
                self.cur = sum(v for v, s in zip(self.values, self.sel) if s)
                if best_abs is None or abs(self.error) < best_abs:
                    best_abs, best_sel = abs(self.error), list(self.sel)
                step += 1
                if progress is not None and step % 20 == 0:
                    progress(step, best_abs, time.perf_counter() - t0)

        elif self.engine == "Anneal":
            limit = max(self.max_steps, 1) * 100
            self.sel = [self.rng.getrandbits(1) for _ in range(self.n)]
            self.cur = sum(v for v, s in zip(self.values, self.sel) if s)
            best_abs, best_sel = abs(self.error), list(self.sel)
            t_hi = max(1, (1 << self.bits) // 2)
            while step < limit and best_abs != 0 and not stopped():
                i = self.rng.randrange(self.n)
                v = self.values[i]
                new_e = self.error + (-v if self.sel[i] else v)
                delta = abs(new_e) - abs(self.error)
                temp = max(1, t_hi * (limit - step) // limit // 8 + 1)
                accept = delta <= 0 or (
                    delta < 40 * temp and self.rng.random() < math.exp(-(delta * 1000 // temp) / 1000.0)
                )
                if accept:
                    self.sel[i] ^= 1
                    self.cur += -v if self.sel[i] == 0 else v
                    if abs(self.error) < best_abs:
                        best_abs, best_sel = abs(self.error), list(self.sel)
                step += 1
                if progress is not None and step % 50_000 == 0:
                    progress(step, best_abs, time.perf_counter() - t0)

        else:  # Swap search
            limit = self.max_steps
            self._greedy_init()
            self._rebuild()
            best_abs, best_sel = abs(self.error), list(self.sel)
            stale = 0
            while step < limit and best_abs != 0 and not stopped():
                new_abs, move = self._best_move()
                if move is not None:
                    for i in move:
                        self._toggle(i)
                    stale = 0
                else:                                   # local minimum: random kick
                    stale += 1
                    if stale % 20 == 0:                 # restart from best state
                        self.sel = list(best_sel)
                        self.cur = sum(v for v, s in zip(self.values, self.sel) if s)
                        self._rebuild()
                    for _ in range(1 + self.rng.randrange(3)):
                        self._toggle(self.rng.randrange(self.n))
                if abs(self.error) < best_abs:
                    best_abs, best_sel = abs(self.error), list(self.sel)
                step += 1
                if progress is not None and step % 10 == 0:
                    progress(step, best_abs, time.perf_counter() - t0)

        best_sum = sum(v for v, s in zip(self.values, best_sel) if s)
        diffs, width = self.bit_diffs(best_sum)
        return {
            "solved": best_sum == self.target,
            "abs_error": abs(best_sum - self.target),
            "rel_error": (abs(best_sum - self.target) / self.target) if self.target else 0.0,
            "bit_diffs": diffs,
            "bit_width": width,
            "match_rate": 1.0 - diffs / width,
            "steps": step,
            "elapsed": time.perf_counter() - t0,
            "assignment": best_sel,
            "chosen": sum(best_sel),
            "n": self.n,
            "bits": self.bits,
            "density": self.density,
        }

    def to_text(self) -> str:
        return "\n".join([f"# target", str(self.target), "# values (one per line)"] + [str(v) for v in self.values]) + "\n"


class PhotonicCircuitSimulator(tk.Tk):
    ROI_A = (0.58, 0.25, 0.92, 0.75)  # Right ROI
    ROI_B = (0.08, 0.25, 0.42, 0.75)  # Left ROI

    def __init__(self):
        super().__init__()
        self.title("Photonic & Modal Logic Simulator — 256-Bit Random Queue vs. Pre-Delay AND (No Flip Noise)")
        self.geometry("1380x940")
        self.minsize(960, 620)

        self.after_id: str | None = None
        self.cam_after_id: str | None = None
        self.resize_after_id: str | None = None
        self.running = False
        self.model: PhotonicBenchModel | None = None

        self.webcam: WebcamWorker | None = None
        self.latest_capture_time: float | None = None
        self.preview_image = None
        self.brightness_a = 0.0
        self.brightness_b = 0.0
        self.detector_a = 0
        self.detector_b = 0

        self.duration_var = tk.StringVar(value="2.0")
        self.speed_var = tk.StringVar(value="1.0")
        self.seed_var = tk.StringVar(value="7")
        self.queue_len_var = tk.StringVar(value="256")
        self.mode_var = tk.StringVar(value="Random")
        self.manual_x_var = tk.BooleanVar(value=False)
        self.ignore_startup_var = tk.BooleanVar(value=False)
        self.feedback_driver_a_var = tk.BooleanVar(value=True)

        self.cam_index_var = tk.StringVar(value="0")
        self.cam_thresh_var = tk.DoubleVar(value=130.0)

        self.laser_mw_var = tk.StringVar(value="10.0")
        self.er_db_var = tk.StringVar(value="20.0")
        self.split_a_var = tk.StringVar(value="0.5")
        self.loss_db_var = tk.StringVar(value="1.5")
        self.resp_var = tk.StringVar(value="0.85")
        self.tia_gain_var = tk.StringVar(value="2.5")
        self.rin_std_var = tk.StringVar(value="0.01")

        self.ss_n_var = tk.StringVar(value="10000")
        self.ss_bits_var = tk.StringVar(value="32")
        self.ss_target_var = tk.StringVar(value="Planted subset")
        self.ss_engine_var = tk.StringVar(value="Swap search")
        self.ss_steps_var = tk.StringVar(value="20000")
        self.ss_status_var = tk.StringVar(value="Subset-sum: idle")
        self.ss_solver: MatchSubsetSumSolver | None = None
        self.ss_result: dict | None = None
        self.ss_thread: threading.Thread | None = None
        self.ss_stop = threading.Event()
        self.ss_events: queue.Queue = queue.Queue()
        self.ss_after_id: str | None = None
        self.ss_loaded = None    # (values, target) from a set file

        self.status_var = tk.StringVar()
        self.photonic_var = tk.StringVar()
        self.stats_var = tk.StringVar()
        self.cam_status_var = tk.StringVar(
            value="USB Cam: off" if HAS_WEBCAM_LIBS else "USB Cam: cv2/PIL unavailable"
        )

        self._build_ui()
        self.reset_simulation()
        self.protocol("WM_DELETE_WINDOW", self.close)
        if HAS_WEBCAM_LIBS:
            self._poll_webcam()

    def _build_scrollable_shell(self) -> ttk.Frame:
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        self.scroll_canvas = tk.Canvas(outer, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.v_scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        content_frame = ttk.Frame(self.scroll_canvas)
        self._scroll_window_id = self.scroll_canvas.create_window(
            (0, 0), window=content_frame, anchor="nw"
        )

        content_frame.bind("<Configure>", self._on_content_configure)
        self.scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)
        self.scroll_canvas.bind("<Enter>", self._bind_mousewheel)
        self.scroll_canvas.bind("<Leave>", self._unbind_mousewheel)

        return content_frame

    def _on_content_configure(self, _event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_scroll_canvas_configure(self, event):
        self.scroll_canvas.itemconfigure(self._scroll_window_id, width=event.width)

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            self.scroll_canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.scroll_canvas.yview_scroll(3, "units")
        elif event.delta:
            steps = int(-event.delta / 120) or (-1 if event.delta > 0 else 1)
            self.scroll_canvas.yview_scroll(steps * 3, "units")

    def _bind_mousewheel(self, _event=None):
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    def _build_ui(self):
        root = self._build_scrollable_shell()

        logic_frame = ttk.LabelFrame(
            root, text="Logic & Timing Controls (Flip noise = 0.0, Queue vs. Pre-Delay AND)", padding=7
        )
        logic_frame.pack(fill="x", padx=10, pady=(8, 4))

        top_fields = [
            ("Probe dur (s)", self.duration_var, 7),
            ("Speed", self.speed_var, 7),
            ("Seed", self.seed_var, 7),
            ("1/0 Queue len", self.queue_len_var, 7),
        ]
        for col, (lbl, var, w) in enumerate(top_fields):
            ttk.Label(logic_frame, text=lbl).grid(row=0, column=col * 2, padx=4, sticky="w")
            entry = ttk.Entry(logic_frame, textvariable=var, width=w)
            entry.grid(row=0, column=col * 2 + 1, padx=4)
            if var is self.queue_len_var:
                entry.bind("<FocusOut>", self._sync_live_vars)
                entry.bind("<Return>", self._sync_live_vars)

        ttk.Label(logic_frame, text="Flip noise").grid(row=0, column=8, padx=(10, 4), sticky="w")
        ttk.Label(logic_frame, text="0.0 (OFF)", font=("Arial", 9, "bold")).grid(
            row=0, column=9, padx=4, sticky="w"
        )

        ttk.Label(logic_frame, text="Input mode").grid(row=1, column=0, padx=4, pady=5, sticky="w")
        mode_box = ttk.Combobox(
            logic_frame,
            textvariable=self.mode_var,
            values=["Random", "Square wave", "Manual", "Direct (x -> a, b)", "Photonic TIA", "USB Webcam"],
            state="readonly",
            width=16,
        )
        mode_box.grid(row=1, column=1, padx=4, pady=5, sticky="w")
        mode_box.bind("<<ComboboxSelected>>", self._sync_live_vars)

        ttk.Checkbutton(
            logic_frame, text="Manual x HIGH", variable=self.manual_x_var, command=self._sync_live_vars
        ).grid(row=1, column=2, columnspan=2, padx=6, sticky="w")

        ttk.Checkbutton(
            logic_frame,
            text="Ignore startup in accum",
            variable=self.ignore_startup_var,
            command=self._sync_live_vars,
        ).grid(row=1, column=4, columnspan=2, padx=6, sticky="w")

        ttk.Checkbutton(
            logic_frame,
            text="Feed SELECT back into Driver A",
            variable=self.feedback_driver_a_var,
            command=self._sync_live_vars,
        ).grid(row=1, column=6, columnspan=4, padx=6, sticky="w")

        cam_frame = ttk.LabelFrame(
            logic_frame, text="USB Webcam ROI Input (Left = B -> x, Right = A -> Driver A)", padding=5
        )
        cam_frame.grid(row=2, column=0, columnspan=10, sticky="ew", pady=(4, 0))

        ttk.Label(cam_frame, text="USB Cam idx").grid(row=0, column=0, padx=4)
        ttk.Entry(cam_frame, textvariable=self.cam_index_var, width=5).grid(row=0, column=1, padx=4)

        self.start_cam_btn = ttk.Button(
            cam_frame,
            text="Start USB Cam",
            command=self.start_webcam,
            state="normal" if HAS_WEBCAM_LIBS else "disabled",
        )
        self.start_cam_btn.grid(row=0, column=2, padx=4)

        self.stop_cam_btn = ttk.Button(
            cam_frame,
            text="Stop USB Cam",
            command=self.stop_webcam,
            state="normal" if HAS_WEBCAM_LIBS else "disabled",
        )
        self.stop_cam_btn.grid(row=0, column=3, padx=4)

        ttk.Label(cam_frame, text="ROI Thresh").grid(row=0, column=4, padx=(10, 4))
        ttk.Scale(
            cam_frame, from_=0, to=255, variable=self.cam_thresh_var, orient="horizontal", length=160
        ).grid(row=0, column=5, padx=4)

        ttk.Label(cam_frame, textvariable=self.cam_status_var).grid(row=0, column=6, padx=10, sticky="w")

        phot_frame = ttk.LabelFrame(root, text="Photonic Front-End Parameters (Reset to apply)", padding=7)
        phot_frame.pack(fill="x", padx=10, pady=(0, 4))

        phot_fields = [
            ("Laser P (mW)", self.laser_mw_var, 7),
            ("ER (dB)", self.er_db_var, 7),
            ("Split A", self.split_a_var, 7),
            ("Coupler loss (dB)", self.loss_db_var, 7),
            ("Resp (A/W)", self.resp_var, 7),
            ("TIA (V/mA)", self.tia_gain_var, 7),
            ("RIN sigma", self.rin_std_var, 7),
        ]
        for col, (lbl, var, w) in enumerate(phot_fields):
            r = 0 if col < 4 else 1
            c_idx = (col % 4) * 2
            ttk.Label(phot_frame, text=lbl).grid(row=r, column=c_idx, padx=4, pady=2, sticky="w")
            ttk.Entry(phot_frame, textvariable=var, width=w).grid(row=r, column=c_idx + 1, padx=4, pady=2)

        ss_frame = ttk.LabelFrame(
            root, text="NP-Hard Solver — Subset Sum (error = subset sum - target, bit DIFF = popcount(sum XOR target))",
            padding=7,
        )
        ss_frame.pack(fill="x", padx=10, pady=(0, 4))

        for col, (lbl, var, w) in enumerate([
            ("N numbers", self.ss_n_var, 8),
            ("Bits / number", self.ss_bits_var, 6),
            ("Max steps", self.ss_steps_var, 9),
        ]):
            ttk.Label(ss_frame, text=lbl).grid(row=0, column=col * 2, padx=4, sticky="w")
            ttk.Entry(ss_frame, textvariable=var, width=w).grid(row=0, column=col * 2 + 1, padx=4)
        ttk.Label(ss_frame, text="Target").grid(row=0, column=6, padx=4, sticky="w")
        ttk.Combobox(
            ss_frame, textvariable=self.ss_target_var, values=list(MatchSubsetSumSolver.TARGET_MODES),
            state="readonly", width=20,
        ).grid(row=0, column=7, padx=4)
        ttk.Label(ss_frame, text="Engine").grid(row=0, column=8, padx=4, sticky="w")
        ttk.Combobox(
            ss_frame, textvariable=self.ss_engine_var, values=list(MatchSubsetSumSolver.ENGINES),
            state="readonly", width=13,
        ).grid(row=0, column=9, padx=4)

        self.ss_solve_btn = ttk.Button(ss_frame, text="Solve", command=self.start_ss_solver)
        self.ss_solve_btn.grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(ss_frame, text="Stop", command=self.ss_stop.set).grid(row=1, column=1, padx=4)
        ttk.Button(ss_frame, text="Load subset into queue", command=self.load_ss_into_queue).grid(
            row=1, column=2, columnspan=2, padx=4
        )
        ttk.Button(ss_frame, text="Export instance", command=self.export_ss_instance).grid(
            row=1, column=4, columnspan=2, padx=4
        )
        ttk.Button(ss_frame, text="Load set from file", command=self.load_ss_from_file).grid(
            row=1, column=6, columnspan=2, padx=4
        )
        ttk.Button(ss_frame, text="Random instance", command=self.clear_ss_loaded).grid(
            row=1, column=8, columnspan=2, padx=4
        )
        ttk.Label(ss_frame, textvariable=self.ss_status_var).grid(
            row=2, column=0, columnspan=10, padx=4, sticky="w"
        )

        btns = ttk.Frame(root, padding=(10, 2))
        btns.pack(fill="x")

        self.run_btn = ttk.Button(btns, text="Run", command=self.toggle_running)
        self.run_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Step 5 ms", command=self.single_step).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset", command=self.reset_simulation).pack(side="left", padx=4)
        ttk.Button(btns, text="Run to end", command=self.run_to_end).pack(side="left", padx=4)
        ttk.Button(btns, text="Export CSV", command=self.export_csv).pack(side="left", padx=4)

        ttk.Label(root, textvariable=self.status_var, padding=(12, 2)).pack(fill="x")
        ttk.Label(root, textvariable=self.photonic_var, padding=(12, 2)).pack(fill="x")

        middle = ttk.Frame(root)
        middle.pack(fill="both", expand=True, padx=10, pady=4)

        self.canvas = tk.Canvas(
            middle, background="white", height=520, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._schedule_redraw)

        self.cam_panel = ttk.LabelFrame(middle, text="USB Webcam Feed", padding=4, width=320)
        self.cam_preview = ttk.Label(self.cam_panel, text="USB cam stopped", anchor="center")
        self.cam_preview.pack(fill="both", expand=True)
        if HAS_WEBCAM_LIBS:
            self.cam_panel.pack(side="right", fill="y", padx=(8, 0))

        ttk.Label(root, textvariable=self.stats_var, padding=(12, 2)).pack(fill="x")

        self.waveform = tk.Canvas(
            root, background="#f7fafc", height=180, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        self.waveform.pack(fill="x", padx=10, pady=(0, 10))
        self.waveform.bind("<Configure>", self._schedule_redraw)

    # ---------------------------------------------------------
    # USB Webcam Controls
    # ---------------------------------------------------------

    def start_webcam(self):
        if not HAS_WEBCAM_LIBS:
            return
        if self.webcam is not None and self.webcam.is_alive:
            messagebox.showinfo("USB Webcam", "Stop the active USB camera first.")
            return
        try:
            idx = int(self.cam_index_var.get())
            if idx < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid camera index", "Enter a non-negative integer (e.g., 0 or 1).")
            return

        self.latest_capture_time = None
        self.detector_a = 0
        self.detector_b = 0
        self.webcam = WebcamWorker(idx)
        self.webcam.start()
        self.cam_status_var.set(f"Opening USB cam {idx}...")

    def stop_webcam(self):
        if self.webcam is not None:
            self.webcam.stop()
            self.webcam = None
        self.latest_capture_time = None
        self.detector_a = 0
        self.detector_b = 0
        self.cam_status_var.set("USB Cam: stopped")
        self.cam_preview.configure(image="", text="USB cam stopped")
        self.redraw()

    def _webcam_fresh(self) -> bool:
        return (
            self.webcam is not None
            and self.webcam.is_alive
            and self.latest_capture_time is not None
            and (time.perf_counter() - self.latest_capture_time) < 1.0
        )

    def _process_webcam_frame(self, capture_time: float, frame):
        self.latest_capture_time = capture_time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        def slice_roi(box):
            x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
            return x1, y1, x2, y2, gray[y1:y2, x1:x2]

        ax1, ay1, ax2, ay2, reg_a = slice_roi(self.ROI_A)
        bx1, by1, bx2, by2, reg_b = slice_roi(self.ROI_B)

        self.brightness_a = float(reg_a.mean()) if reg_a.size else 0.0
        self.brightness_b = float(reg_b.mean()) if reg_b.size else 0.0

        thresh = float(self.cam_thresh_var.get())
        self.detector_a = int(self.brightness_a >= thresh)
        self.detector_b = int(self.brightness_b >= thresh)

        vis = frame.copy()
        for label, (x1, y1, x2, y2), state, bright in [
            ("A", (ax1, ay1, ax2, ay2), self.detector_a, self.brightness_a),
            ("B", (bx1, by1, bx2, by2), self.detector_b, self.brightness_b),
        ]:
            col = (40, 210, 40) if state else (50, 80, 230)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                vis, f"{label}: {bright:.0f}->{state}", (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2
            )

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pw = max(200, self.cam_preview.winfo_width())
        ph = max(160, self.cam_preview.winfo_height())
        pil_img.thumbnail((pw, ph))
        self.preview_image = ImageTk.PhotoImage(pil_img)
        self.cam_preview.configure(image=self.preview_image, text="")
        self.cam_status_var.set(f"USB Cam live (thresh {thresh:.0f})")

    def _poll_webcam(self):
        if self.webcam is not None:
            latest = self.webcam.latest_frame()
            if latest is not None:
                self._process_webcam_frame(*latest)
            if not self.webcam.is_alive:
                err = self.webcam.error
                self.webcam = None
                self.latest_capture_time = None
                self.detector_a = 0
                self.detector_b = 0
                self.cam_status_var.set(f"USB Cam error: {err}" if err else "USB Cam: stopped")
        self.cam_after_id = self.after(30, self._poll_webcam)

    # ---------------------------------------------------------
    # NP-hard solver: subset sum
    # ---------------------------------------------------------

    def start_ss_solver(self):
        if self.ss_thread is not None and self.ss_thread.is_alive():
            messagebox.showinfo("Subset sum", "Solver already running. Press Stop first.")
            return
        try:
            steps = int(self.ss_steps_var.get())
            seed = int(self.seed_var.get())
            if not (1 <= steps <= 50_000_000):
                raise ValueError("Max steps must be in [1, 50,000,000].")
            if self.ss_loaded is not None:
                values, target = self.ss_loaded
                self.ss_solver = MatchSubsetSumSolver(
                    0, 0, self.ss_target_var.get(), self.ss_engine_var.get(), steps, seed,
                    values=values, target=target,
                )
            else:
                n = int(self.ss_n_var.get())
                bits = int(self.ss_bits_var.get())
                if not (2 <= n <= 200_000):
                    raise ValueError("N must be in [2, 200000].")
                if not (2 <= bits <= 20_000):
                    raise ValueError("Bits per number must be in [2, 20000].")
                self.ss_solver = MatchSubsetSumSolver(
                    n, bits, self.ss_target_var.get(), self.ss_engine_var.get(), steps, seed
                )
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid subset-sum settings", str(exc))
            return

        self.ss_result = None
        self.ss_stop.clear()
        while not self.ss_events.empty():
            self.ss_events.get_nowait()
        solver = self.ss_solver

        def worker():
            def progress(step, best_abs, elapsed):
                self.ss_events.put(("progress", step, best_abs, elapsed))
            try:
                self.ss_events.put(("done", solver.run(self.ss_stop, progress)))
            except Exception as exc:
                self.ss_events.put(("error", str(exc)))

        self.ss_thread = threading.Thread(target=worker, daemon=True)
        self.ss_thread.start()
        self.ss_solve_btn.configure(state="disabled")
        self.ss_status_var.set(
            f"Subset-sum: {solver.engine}, N={solver.n}, {solver.bits}-bit numbers "
            f"(density n/b = {solver.density:.2f}), target has {solver.target.bit_length()} bits ..."
        )
        self._poll_ss()

    @staticmethod
    def _fmt_big(x: int) -> str:
        s = str(x)
        return s if len(s) <= 18 else f"{s[:8]}..e{len(s) - 1}"

    def _poll_ss(self):
        self.ss_after_id = None
        finished = False
        while True:
            try:
                ev = self.ss_events.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "progress":
                _, step, best_abs, elapsed = ev
                self.ss_status_var.set(
                    f"Subset-sum running: step {step}, best |error| = {self._fmt_big(best_abs)}, {elapsed:.1f}s"
                )
            elif ev[0] == "done":
                res = ev[1]
                self.ss_result = res
                finished = True
                verdict = "EXACT (error 0, all bits match)" if res["solved"] else "closest found (not exact)"
                self.ss_status_var.set(
                    f"Subset-sum: {verdict} | N={res['n']} b={res['bits']} density={res['density']:.2f} | "
                    f"|error|={self._fmt_big(res['abs_error'])} (rel {res['rel_error']:.2e}) | "
                    f"bit match {res['match_rate']:.1%} | {res['chosen']} chosen | "
                    f"{res['steps']} steps, {res['elapsed']:.2f}s"
                )
            elif ev[0] == "error":
                finished = True
                self.ss_status_var.set(f"Subset-sum error: {ev[1]}")
        if finished or (self.ss_thread is not None and not self.ss_thread.is_alive() and self.ss_events.empty()):
            self.ss_solve_btn.configure(state="normal")
            return
        self.ss_after_id = self.after(100, self._poll_ss)

    def load_ss_into_queue(self):
        if self.model is None or self.ss_result is None:
            messagebox.showinfo("Subset sum", "Run the solver first.")
            return
        bits = self.ss_result["assignment"]
        if len(bits) > self.model.d_queue_length:
            messagebox.showinfo(
                "Subset sum",
                f"The queue holds {self.model.d_queue_length} bits; loading the first "
                f"{self.model.d_queue_length} of {len(bits)} selection bits.",
            )
        self.model.load_queue_bits(bits)
        self.redraw()

    def export_ss_instance(self):
        if self.ss_solver is None:
            messagebox.showinfo("Subset sum", "Run the solver first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export subset-sum instance", defaultextension=".txt", filetypes=[("Text", "*.txt")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.ss_solver.to_text())
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))

    def load_ss_from_file(self):
        path = filedialog.askopenfilename(
            title="Load subset-sum set (integers, optional target)",
            filetypes=[("Text / CSV", "*.txt *.csv *.dat"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            values, target, zeros = parse_subset_file(path)
            probe = MatchSubsetSumSolver(
                0, 0, self.ss_target_var.get(), self.ss_engine_var.get(), 1,
                int(self.seed_var.get()), values=values, target=target,
            )
        except (ValueError, OSError, tk.TclError) as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.ss_loaded = (values, probe.target)   # keep the (given or generated) target fixed
        self.ss_solver = probe
        self.ss_n_var.set(str(probe.n))
        self.ss_bits_var.set(str(probe.bits))
        src_t = "from file" if target is not None else f"generated ({self.ss_target_var.get()})"
        extra = f", {zeros} zero(s) skipped" if zeros else ""
        self.ss_status_var.set(
            f"Subset-sum: loaded {probe.n} numbers (max {probe.bits} bits, density {probe.density:.2f}){extra}; "
            f"target {src_t}: {self._fmt_big(probe.target)}. Press Solve."
        )

    def clear_ss_loaded(self):
        self.ss_loaded = None
        self.ss_status_var.set("Subset-sum: using random generated instance")

    # ---------------------------------------------------------
    # Simulation Controls
    # ---------------------------------------------------------

    def _parse_config(self):
        duration = float(self.duration_var.get())
        speed = float(self.speed_var.get())
        seed = int(self.seed_var.get())
        q_len = int(self.queue_len_var.get())
        laser_mw = float(self.laser_mw_var.get())
        er_db = float(self.er_db_var.get())
        split_a = float(self.split_a_var.get())
        loss_db = float(self.loss_db_var.get())
        resp = float(self.resp_var.get())
        tia_gain = float(self.tia_gain_var.get())
        rin_std = float(self.rin_std_var.get())

        if not math.isfinite(duration) or not (0 < duration <= 120):
            raise ValueError("Probe duration must be > 0 and <= 120 s.")
        if not math.isfinite(speed) or not (0.01 <= speed <= 100):
            raise ValueError("Playback speed must be in [0.01, 100].")
        if not (1 <= q_len <= 4096):
            raise ValueError("1/0 Queue length must be between 1 and 4096 bits.")
        if not math.isfinite(laser_mw) or laser_mw < 0:
            raise ValueError("Laser power must be >= 0 mW.")
        if not (0 <= split_a <= 1):
            raise ValueError("Split ratio A must be in [0, 1].")
        if not math.isfinite(resp) or resp <= 0:
            raise ValueError("Responsivity must be > 0 A/W.")
        if not math.isfinite(tia_gain) or tia_gain < 0:
            raise ValueError("TIA gain must be >= 0 V/mA.")
        if not math.isfinite(rin_std) or rin_std < 0:
            raise ValueError("RIN sigma must be >= 0.")

        return duration, speed, seed, q_len, laser_mw, er_db, split_a, loss_db, resp, tia_gain, rin_std

    def _sync_live_vars(self, _event=None):
        if self.model is not None:
            self.model.mode = self.mode_var.get()
            self.model.manual_x = bool(self.manual_x_var.get())
            self.model.ignore_startup = bool(self.ignore_startup_var.get())
            self.model.feed_select_to_driver_a = bool(self.feedback_driver_a_var.get())
            try:
                q_len = int(self.queue_len_var.get())
                if 1 <= q_len <= 4096:
                    self.model.set_queue_length(q_len)
            except (ValueError, tk.TclError):
                pass
        self.redraw()

    def _step_once(self) -> bool:
        assert self.model is not None
        self._sync_live_vars()
        fresh = self._webcam_fresh()
        return self.model.step(
            cam_a=self.detector_a if fresh else None,
            cam_b=self.detector_b if fresh else None,
            brightness_a=self.brightness_a,
            brightness_b=self.brightness_b,
            threshold_px=float(self.cam_thresh_var.get()),
        )

    def pause(self):
        self.running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.run_btn.configure(text="Run")

    def reset_simulation(self):
        self.pause()
        try:
            duration, speed, seed, q_len, laser_mw, er_db, split_a, loss_db, resp, tia_gain, rin_std = (
                self._parse_config()
            )
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.playback_speed = speed
        self.model = PhotonicBenchModel(
            probe_duration=duration,
            seed=seed,
            mode=self.mode_var.get(),
            ignore_startup=self.ignore_startup_var.get(),
            manual_x=self.manual_x_var.get(),
            feed_select_to_driver_a=self.feedback_driver_a_var.get(),
            d_queue_length=q_len,
            laser_power_mw=laser_mw,
            extinction_db=er_db,
            split_ratio_a=split_a,
            insertion_loss_db=loss_db,
            responsivity_a_per_w=resp,
            tia_gain_v_per_ma=tia_gain,
            optical_noise_std=rin_std,
        )
        self.redraw()

    def toggle_running(self):
        if self.model is None:
            return
        if self.running:
            self.pause()
            self.redraw()
            return
        if self.model.finished:
            self.status_var.set("Simulation finished. Press Reset to run again.")
            return
        if self.mode_var.get() == "USB Webcam" and not self._webcam_fresh():
            messagebox.showwarning("USB Camera not ready", "Start the USB webcam first or choose another input mode.")
            return
        try:
            _, speed, *_ = self._parse_config()
            self.playback_speed = speed
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.running = True
        self.run_btn.configure(text="Pause")
        self._tick()

    def _tick(self):
        self.after_id = None
        if not self.running or self.model is None:
            return
        try:
            self.playback_speed = float(self.speed_var.get())
        except (ValueError, tk.TclError):
            pass

        self._step_once()
        self.redraw()

        if self.model.finished:
            self.pause()
            self.redraw()
            return

        delay_ms = max(1, round(1000 * self.model.DT / max(0.01, self.playback_speed)))
        self.after_id = self.after(delay_ms, self._tick)

    def single_step(self):
        if self.model is None:
            return
        self.pause()
        self._step_once()
        self.redraw()

    def run_to_end(self):
        if self.model is None:
            return
        self.pause()
        while not self.model.finished:
            self._step_once()
        self.redraw()

    def _schedule_redraw(self, _event=None):
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        self.resize_after_id = self.after(25, self.redraw)

    def redraw(self):
        self.resize_after_id = None
        if self.model is None:
            return
        self._draw_circuit()
        self._draw_waveforms()
        self._update_status()

    def _section(self, x, y, w, h, title):
        c = self.canvas
        c.create_rectangle(x, y, x + w, y + h, outline="#d4dde5", dash=(4, 3))
        c.create_text(x + 12, y + 13, text=title, anchor="w", font=("Arial", 10, "bold"), fill="#34495e")

    def _box(self, x, y, w, h, text, value=None):
        fill = "#f5f7fa" if value is None else ("#d8f3e6" if value else "#edf0f4")
        self.canvas.create_rectangle(x, y, x + w, y + h, fill=fill, outline="#aebdc9")
        self.canvas.create_text(x + w / 2, y + h / 2, text=text, font=("Arial", 10), fill="#203040", justify="center")

    def _wire(self, points, value=None, dash=None):
        color = "#12805c" if value else "#82919e"
        kw = {"arrow": tk.LAST, "fill": color, "width": 2}
        if dash is not None:
            kw["dash"] = dash
        self.canvas.create_line(*points, **kw)

    def _gate(self, x, y, name, value):
        c = self.canvas
        fill = "#d8f3e6" if value else "#edf0f4"
        c.create_polygon(
            x, y, x + 62, y, x + 85, y + 23, x + 62, y + 46, x, y + 46,
            fill=fill, outline="#aebdc9"
        )
        c.create_text(x + 37, y + 23, text=name, font=("Arial", 10, "bold"), fill="#203040")

    def _format_queue_preview(self, m: PhotonicBenchModel) -> str:
        bits = "".join(str(bit) for bit in m.d_bit_queue)
        if len(bits) > 12:
            bits = bits[:6] + ".." + bits[-4:]
        return f"[{bits}] ({len(m.d_bit_queue)}/{m.d_queue_length})"

    def _draw_circuit(self):
        m = self.model
        assert m is not None
        c = self.canvas
        c.delete("all")

        self._section(10, 10, 1140, 125, f"PHOTONIC BENCH — mode: {m.mode}")
        self._section(10, 145, 1140, 115, "PARALLEL DRIVERS (SELECT -> Driver A) / TIA + RC THRESHOLD")
        self._section(10, 270, 1140, 200, "MAIN CIRCUIT — pre-delay AND -> 150 ms delay + 256-Bit Random Queue -> 74HC74")
        self._section(10, 480, 1140, 100, "NO-FLIP PROBE — compares 256-bit queue out (D_q) vs. AND BEFORE 150 ms delay")

        self._box(880, 40, 220, 45, f"Laser ({m.laser_power_mw:g} mW)\nTx = {m.optical_tx_mw:.2f} mW", m.x)
        self._box(590, 40, 220, 45, f"Coupler / x={m.x}\nphi={m.optical_phase_rad:.2f} rad", m.x)
        self._wire([880, 62, 810, 62], m.x)

        self._box(
            40, 80, 220, 40,
            f"PD A / Cam A ({self.brightness_a:.0f})\n{m.optical_a_mw:.2f} mW | {m.photocurrent_a_ua:.1f} uA",
            m.driver_a_raw,
        )
        self._box(
            310, 80, 220, 40,
            f"PD B / Cam B ({self.brightness_b:.0f})\n{m.optical_b_mw:.2f} mW | {m.photocurrent_b_ua:.1f} uA",
            m.x,
        )
        self._wire([700, 85, 700, 100, 530, 100], m.x)
        self._wire([590, 62, 150, 62, 150, 80], m.x)

        fb_str = f"Q={m.q}" if m.feed_select_to_driver_a else "OFF"
        self._box(
            40, 182, 225, 56,
            f"Driver A (SEL fb: {fb_str})\nraw={m.driver_a_raw}, sel_gate={m.driver_a_select_gate} -> a={m.a}",
            m.a,
        )
        self._box(295, 182, 190, 56, f"Driver B / x\nx = {m.x}", m.x)
        self._box(515, 182, 210, 56, f"RC (tau={m.RC_TAU*1000:.0f}ms)\nVc = {m.rc_voltage:.3f} V")
        self._box(760, 182, 260, 56, f"Comparator > {m.THRESHOLD} V\nb = {m.b}", m.b)

        self._wire([150, 120, 150, 182], m.driver_a_raw)
        self._wire([420, 120, 420, 182], m.x)
        self._wire([485, 210, 515, 210], m.x)
        self._wire([725, 210, 760, 210], m.b)

        self._wire(
            [1065, 370, 1065, 276, 22, 276, 22, 210, 40, 210],
            m.q if m.feed_select_to_driver_a else 0,
            dash=None if m.feed_select_to_driver_a else (5, 4),
        )

        # Pre-delay AND gate
        c.create_text(35, 328, text=f"a={m.a}\nb={m.b}", anchor="w", font=("Arial", 11))
        self._gate(100, 308, "AND", m.pre_delay_and)
        self._wire([68, 323, 100, 323], m.a)
        self._wire([68, 340, 100, 340], m.b)

        self._box(
            225, 308, 185, 46,
            f"Delay {m.AND_DELAY*1000:.0f} ms ({m.optical_delayed_mw:.2f}mW)\nout = {m.delayed_and}",
            m.delayed_and,
        )
        self._wire([185, 331, 225, 331], m.pre_delay_and)

        self._box(450, 308, 180, 46, f"Running AND\nval = {m.accumulator}", m.accumulator)
        self._wire([410, 323, 450, 323], m.delayed_and)

        c.create_text(35, 412, text=f"a={m.a}\nb={m.b}", anchor="w", font=("Arial", 11))
        self._gate(100, 390, "OR", m.y)
        self._wire([68, 405, 100, 405], m.a)
        self._wire([68, 422, 100, 422], m.b)

        self._gate(245, 390, "OR", m.d)
        self._wire([185, 413, 245, 413], m.y)
        self._wire([410, 339, 430, 339, 430, 400, 245, 400], m.delayed_and)

        self._box(
            355, 382, 195, 62,
            f"Rnd Queue (in D={m.d})\n{self._format_queue_preview(m)}\nout={m.d_queued} (lag {m.last_lag_steps}st)",
            m.d_queued,
        )
        self._wire([330, 413, 355, 413], m.d)

        self._box(585, 378, 175, 70, f"74HC74 D-FF\nDq={m.d_queued}  Q={m.q}\nClk: {1/m.CLOCK_PERIOD:g} Hz", m.q)
        self._wire([550, 413, 585, 413], m.d_queued)

        self._box(1000, 370, 130, 30, f"SELECT = {m.q}", m.q)
        self._box(1000, 420, 130, 30, f"Observe = {m.not_q}", m.not_q)
        self._wire([760, 395, 1000, 385], m.q)
        self._wire([760, 425, 1000, 435], m.not_q)

        # No-flip probe comparing 256-bit queue output against AND BEFORE delay
        active = m.step_index < m.probe_steps
        p_state = "ACTIVE" if active else "STOPPED"
        self._box(
            55, 515, 225, 45,
            f"Pre-Delay AND = {m.pre_delay_and}\nQueue out D_q = {m.d_queued}",
            m.pre_delay_and,
        )
        self._box(
            310, 515, 170, 45,
            f"Flip noise: 0.0\nlag = {m.last_lag_steps} steps",
        )
        self._wire([280, 538, 310, 538], m.pre_delay_and)

        match_str = "-" if m.last_match is None else ("MATCH" if m.last_match else "DIFF")
        self._box(
            510, 515, 310, 45,
            f"Probe ({p_state}): D_q={m.d_queued} vs pre_AND={m.pre_delay_and}\n"
            f"state = {match_str} | post_Q_AND={m.post_queue_and}",
        )
        self._wire([480, 538, 510, 538])

        self._box(
            850, 515, 255, 45,
            f"Samples={m.probe_count} | Match={m.probe_matches} | Diff={m.probe_mismatches}",
        )
        self._wire([820, 538, 850, 538])

        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        c.scale("all", 0, 0, w / 1160, h / 590)

    def _draw_waveforms(self):
        m = self.model
        assert m is not None
        p = self.waveform
        p.delete("all")

        w = max(600, p.winfo_width())
        left = 115
        right = w - 25
        plot_w = max(10, right - left)

        signals = [
            ("pre_del AND", "pre_delay_and", "#1976a3"),
            ("delayed AND", "delayed_and", "#2c7a7b"),
            ("D (OR out)", "D_raw_or", "#b36a20"),
            ("D_queued", "D_queued", "#7753a6"),
            ("Q / SELECT", "Q_SELECT", "#bd4258"),
        ]

        stride = max(1, len(m.records) // 1800)
        visible = m.records[::stride]
        if m.records and (not visible or visible[-1] is not m.records[-1]):
            visible.append(m.records[-1])

        total_dur = max(m.DT, m.total_duration)

        for idx, (label, key, color) in enumerate(signals):
            base_y = 24 + idx * 28
            p.create_text(8, base_y, text=label, anchor="w", fill="#34495e")
            p.create_line(left, base_y + 7, right, base_y + 7, fill="#dce4eb")

            points = [left, base_y + 7]
            prev_y = base_y + 7
            for rec in visible:
                x = left + plot_w * (rec["time_s"] / total_dur)
                y = base_y + 7 - 14 * rec[key]
                points.extend([x, prev_y, x, y])
                prev_y = y

            if len(points) >= 4:
                p.create_line(*points, fill=color, width=2)

        probe_x = left + plot_w * (m.probe_duration / total_dur)
        p.create_line(probe_x, 8, probe_x, 152, fill="#888888", dash=(4, 3))
        p.create_text(left, 167, text="0 s", anchor="w")
        p.create_text(probe_x, 167, text=f"Probe stops: {m.probe_duration:g} s")
        p.create_text(right, 167, text=f"{m.total_duration:g} s", anchor="e")

    def _update_status(self):
        m = self.model
        assert m is not None

        state = "Finished" if m.finished else ("Running" if self.running else "Paused")
        fb_str = f"ON ({m.driver_a_select_gate})" if m.feed_select_to_driver_a else "OFF"
        self.status_var.set(
            f"{state}  |  Time: {m.time:.3f} / {m.total_duration:g} s  |  "
            f"dt: {m.DT * 1000:g} ms  |  Mode: {m.mode}  |  SELECT->DriverA: {fb_str}  |  FlipNoise: 0.0"
        )
        self.photonic_var.set(
            f"Tx: {m.optical_tx_mw:.2f} mW ({mw_to_dbm(m.optical_tx_mw):.1f} dBm)  |  "
            f"PD-A: {m.photocurrent_a_ua:.1f} uA (CamA={self.brightness_a:.0f})  |  "
            f"PD-B: {m.photocurrent_b_ua:.1f} uA (CamB={self.brightness_b:.0f})"
        )

        match_pct = (m.probe_matches / m.probe_count) if m.probe_count else 0.0
        avg_lag_ms = (
            (m.lag_steps_sum / m.probe_count) * m.DT * 1000.0
            if m.probe_count else 0.0
        )
        self.stats_var.set(
            f"pre_delay_AND={m.pre_delay_and}, D_queued={m.d_queued}, del_AND={m.delayed_and}, Q={m.q}  |  "
            f"Queue: {len(m.d_bit_queue)}/{m.d_queue_length} (avg lag {avg_lag_ms:.1f} ms)  |  "
            f"Queue==PreDelayAND: {m.probe_matches}/{m.probe_count} ({match_pct:.1%}), "
            f"Diffs: {m.probe_mismatches}"
        )

    def export_csv(self):
        if self.model is None or not self.model.records:
            messagebox.showinfo("No data", "Run the simulation or step first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export photonic simulation data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self.model.records[0].keys()))
                writer.writeheader()
                writer.writerows(self.model.records)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return

        messagebox.showinfo("Export complete", f"Exported {len(self.model.records)} samples.")

    def close(self):
        self.pause()
        self.ss_stop.set()
        if self.ss_after_id is not None:
            self.after_cancel(self.ss_after_id)
        self._unbind_mousewheel()
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        if self.cam_after_id is not None:
            self.after_cancel(self.cam_after_id)
        if self.webcam is not None:
            self.webcam.stop()
        self.destroy()


if __name__ == "__main__":
    app = PhotonicCircuitSimulator()
    app.mainloop()
