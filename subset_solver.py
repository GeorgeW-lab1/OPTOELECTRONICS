"""
SUBSET-SUM SOLVER (error-matching search)

Standalone tool: N random integers (b bits each) plus a target, or a set loaded from
a file. The candidate subset is a bit register; the error-match signal is
  E = sum(selected) - target          (numeric error)
  popcount(sum XOR target)            (bit-level DIFF count -> match rate)

Engines
  Swap search  : greedy start, then add / remove / 1-1 swap moves found by bisect,
                 with random kicks and restarts when stuck
  Anneal       : simulated annealing on single-bit flips
  Random queue : blind random subsets (baseline)

Subset sum is NP-hard. This is heuristic search: it finds exact answers on dense
instances (n >> bits per number) and only gets close when n ~ bits. No webcam or
photonic simulation is involved.

File format for "Load set from file": positive integers separated by whitespace,
commas or semicolons; optional target line ("target: N", "T=N", or "# target" followed
by the number); lines starting with # or // are comments.
"""

import bisect
import math
import queue
import random
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


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


def parse_target_text(text: str):
    """Parse a custom subset-sum target. Blank -> None. Accepts 1234, 1,234, 1_234, 0xFF."""
    s = text.strip().replace(",", "").replace("_", "").replace(" ", "")
    if not s:
        return None
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", s):
        return int(s, 16)
    if re.fullmatch(r"\d+", s):
        return int(s)
    raise ValueError("Custom target must be a non-negative integer (e.g. 123456 or 0xFF).")


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


class SubsetSumApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Subset-Sum Solver — error-matching search")
        self.geometry("900x560")
        self.minsize(720, 440)

        self.n_var = tk.StringVar(value="10000")
        self.bits_var = tk.StringVar(value="32")
        self.steps_var = tk.StringVar(value="20000")
        self.seed_var = tk.StringVar(value="7")
        self.target_mode_var = tk.StringVar(value="Planted subset")
        self.engine_var = tk.StringVar(value="Swap search")
        self.custom_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle. Set parameters and press Solve.")

        self.solver: MatchSubsetSumSolver | None = None
        self.result: dict | None = None
        self.loaded = None          # (values, target) from a set file
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self.after_id: str | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------- UI ----------------
    def _build_ui(self):
        frame = ttk.LabelFrame(self, text="Instance", padding=8)
        frame.pack(fill="x", padx=10, pady=(10, 4))

        fields = [
            ("N numbers", self.n_var, 8),
            ("Bits / number", self.bits_var, 6),
            ("Max steps", self.steps_var, 9),
            ("Seed", self.seed_var, 6),
        ]
        for col, (label, var, width) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=0, column=col * 2, padx=4, sticky="w")
            ttk.Entry(frame, textvariable=var, width=width).grid(row=0, column=col * 2 + 1, padx=4)

        ttk.Label(frame, text="Target mode").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Combobox(
            frame, textvariable=self.target_mode_var, values=list(MatchSubsetSumSolver.TARGET_MODES),
            state="readonly", width=20,
        ).grid(row=1, column=1, columnspan=2, padx=4, sticky="w")
        ttk.Label(frame, text="Engine").grid(row=1, column=3, padx=4, sticky="w")
        ttk.Combobox(
            frame, textvariable=self.engine_var, values=list(MatchSubsetSumSolver.ENGINES),
            state="readonly", width=14,
        ).grid(row=1, column=4, columnspan=2, padx=4, sticky="w")

        ttk.Label(frame, text="Custom target (blank = use mode / file)").grid(
            row=2, column=0, columnspan=3, padx=4, sticky="w"
        )
        ttk.Entry(frame, textvariable=self.custom_var, width=52).grid(
            row=2, column=3, columnspan=5, padx=4, pady=2, sticky="w"
        )

        btns = ttk.Frame(self, padding=(10, 2))
        btns.pack(fill="x")
        self.solve_btn = ttk.Button(btns, text="Solve", command=self.start)
        self.solve_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Stop", command=self.stop_event.set).pack(side="left", padx=4)
        ttk.Button(btns, text="Load set from file", command=self.load_from_file).pack(side="left", padx=4)
        ttk.Button(btns, text="Random instance", command=self.clear_loaded).pack(side="left", padx=4)
        ttk.Button(btns, text="Export instance", command=self.export_instance).pack(side="left", padx=4)
        ttk.Button(btns, text="Export solution", command=self.export_solution).pack(side="left", padx=4)

        ttk.Label(self, textvariable=self.status_var, wraplength=860, justify="left", padding=(12, 6)).pack(fill="x")

        log_frame = ttk.LabelFrame(self, text="Run log", padding=4)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    def _log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    @staticmethod
    def _fmt_big(x: int) -> str:
        s = str(x)
        return s if len(s) <= 18 else f"{s[:8]}..e{len(s) - 1}"

    # ---------------- solving ----------------
    def _build_solver(self) -> MatchSubsetSumSolver:
        steps = int(self.steps_var.get())
        seed = int(self.seed_var.get())
        if not (1 <= steps <= 50_000_000):
            raise ValueError("Max steps must be in [1, 50,000,000].")
        custom = parse_target_text(self.custom_var.get())
        if self.loaded is not None:
            values, target = self.loaded
            return MatchSubsetSumSolver(
                0, 0, self.target_mode_var.get(), self.engine_var.get(), steps, seed,
                values=values, target=target if custom is None else custom,
            )
        n = int(self.n_var.get())
        bits = int(self.bits_var.get())
        if not (2 <= n <= 200_000):
            raise ValueError("N must be in [2, 200000].")
        if not (2 <= bits <= 20_000):
            raise ValueError("Bits per number must be in [2, 20000].")
        return MatchSubsetSumSolver(
            n, bits, self.target_mode_var.get(), self.engine_var.get(), steps, seed, target=custom
        )

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            messagebox.showinfo("Subset sum", "Solver already running. Press Stop first.")
            return
        try:
            self.solver = self._build_solver()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.result = None
        self.stop_event.clear()
        while not self.events.empty():
            self.events.get_nowait()
        solver = self.solver

        def worker():
            def progress(step, best_abs, elapsed):
                self.events.put(("progress", step, best_abs, elapsed))
            try:
                self.events.put(("done", solver.run(self.stop_event, progress)))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        self.solve_btn.configure(state="disabled")
        warn = " | WARNING: target exceeds the sum of all numbers (unreachable)" if solver.target > solver.total else ""
        msg = (
            f"{solver.engine}, N={solver.n}, {solver.bits}-bit numbers (density n/b = {solver.density:.2f}), "
            f"target {self._fmt_big(solver.target)} ({solver.target.bit_length()} bits){warn}"
        )
        self.status_var.set("Running: " + msg)
        self._log("START  " + msg)
        self._poll()

    def _poll(self):
        self.after_id = None
        finished = False
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "progress":
                _, step, best_abs, elapsed = ev
                self.status_var.set(
                    f"Running: step {step}, best |error| = {self._fmt_big(best_abs)}, {elapsed:.1f}s"
                )
            elif ev[0] == "done":
                res = ev[1]
                self.result = res
                finished = True
                verdict = "EXACT (error 0, all bits match)" if res["solved"] else "closest found (not exact)"
                line = (
                    f"{verdict} | N={res['n']} b={res['bits']} density={res['density']:.2f} | "
                    f"|error|={self._fmt_big(res['abs_error'])} (rel {res['rel_error']:.2e}) | "
                    f"bit match {res['match_rate']:.1%} | {res['chosen']} chosen | "
                    f"{res['steps']} steps, {res['elapsed']:.2f}s"
                )
                self.status_var.set(line)
                self._log("RESULT " + line)
            elif ev[0] == "error":
                finished = True
                self.status_var.set(f"Error: {ev[1]}")
                self._log("ERROR  " + ev[1])
        if finished or (self.thread is not None and not self.thread.is_alive() and self.events.empty()):
            self.solve_btn.configure(state="normal")
            return
        self.after_id = self.after(100, self._poll)

    # ---------------- files ----------------
    def load_from_file(self):
        path = filedialog.askopenfilename(
            title="Load subset-sum set (integers, optional target)",
            filetypes=[("Text / CSV", "*.txt *.csv *.dat"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            values, target, zeros = parse_subset_file(path)
            custom = parse_target_text(self.custom_var.get())
            probe = MatchSubsetSumSolver(
                0, 0, self.target_mode_var.get(), self.engine_var.get(), 1,
                int(self.seed_var.get()), values=values, target=custom if custom is not None else target,
            )
        except (ValueError, OSError, tk.TclError) as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.loaded = (values, probe.target)   # keep the (given or generated) target fixed
        self.solver = probe
        self.n_var.set(str(probe.n))
        self.bits_var.set(str(probe.bits))
        src_t = (
            "custom" if custom is not None
            else "from file" if target is not None
            else f"generated ({self.target_mode_var.get()})"
        )
        extra = f", {zeros} zero(s) skipped" if zeros else ""
        msg = (
            f"Loaded {probe.n} numbers (max {probe.bits} bits, density {probe.density:.2f}){extra}; "
            f"target {src_t}: {self._fmt_big(probe.target)}. Press Solve."
        )
        self.status_var.set(msg)
        self._log("LOAD   " + msg)

    def clear_loaded(self):
        self.loaded = None
        self.status_var.set("Using a random generated instance.")
        self._log("LOAD   switched to random generated instance")

    def export_instance(self):
        if self.solver is None:
            messagebox.showinfo("Subset sum", "Run or load an instance first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export subset-sum instance", defaultextension=".txt", filetypes=[("Text", "*.txt")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.solver.to_text())
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_solution(self):
        if self.result is None or self.solver is None:
            messagebox.showinfo("Subset sum", "Run the solver first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export chosen subset", defaultextension=".txt", filetypes=[("Text", "*.txt")]
        )
        if not path:
            return
        res = self.result
        chosen = [v for v, s in zip(self.solver.values, res["assignment"]) if s]
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"# target\n{self.solver.target}\n")
                fh.write(f"# subset sum\n{sum(chosen)}\n")
                fh.write(f"# error\n{sum(chosen) - self.solver.target}\n")
                fh.write(f"# chosen numbers ({len(chosen)})\n")
                fh.write("\n".join(str(v) for v in chosen) + "\n")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))

    def close(self):
        self.stop_event.set()
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.destroy()


if __name__ == "__main__":
    SubsetSumApp().mainloop()
