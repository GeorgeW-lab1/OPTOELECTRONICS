"""
Bit-Pooling Factorisation Lab - N only, factors unknown, MSB-first, A and B both
oracle-decided (balanced P and Q), predicted pair ordering, deferred check.
At each depth the 4 candidate bit pairs for (A, B) are ranked by a prediction score
(distance of the cell-midpoint product from N). The best-ranked pair is proposed
first; the oracle confirms it with an interval test that uses only N, or rejects it
and the next best is tried. Exhausted nodes backtrack (DFS, O(1) work per step).
The expensive check N % a == 0 runs ONCE per fully pooled leaf. Factors are unsealed
only if it verifies.
"""

import asyncio
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def fmt_big(x):
    h = hex(x)[2:]
    if len(h) > 24:
        h = h[:10] + "…" + h[-10:]
    return f"0x{h} ({x.bit_length()} bits)"


def parse_int(s):
    s = s.strip().replace("_", "")
    if not s:
        raise ValueError("Enter N.")
    return int(s, 0) if s.lower().startswith(("0x", "0b", "0o")) else int(s)


# ==============================================================================
# ORACLE: knows only N
# ==============================================================================
class BitPoolingOracle:
    def __init__(self, n):
        if n < 4:
            raise ValueError("N must be at least 4.")
        self.n = n
        self.is_done = False
        self.__d = 0
        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

    def query_bit(self, A, B, r):
        """Oracle bit: 1 if prefixes A, B (r bits still unpooled in each) can still
        give a*b == N, with a <= b. Pure interval test, uses only N."""
        if A > B:
            return 0
        m = (1 << r) - 1
        alo, blo = A << r, B << r
        return 1 if alo * blo <= self.n <= (alo + m) * (blo + m) else 0

    def predict(self, A, B, r):
        """Prediction score (lower = better): distance of the cell-midpoint product from N."""
        m = (1 << r) - 1
        am, bm = (A << r) + (m >> 1), (B << r) + (m >> 1)
        return abs(am * bm - self.n)

    def submit(self, d):
        """The single expensive check. Unseals the factors only if N % d == 0."""
        if 1 < d < self.n and self.n % d == 0:
            self.__d = d
            self.is_done = True
        return self.is_done

    @property
    def factors(self):
        if not self.is_done:
            raise PermissionError("SECURITY VIOLATION: factors sealed until a solution is verified!")
        other = self.n // self.__d
        return min(self.__d, other), max(self.__d, other)

    def pool_step(self, val, nbits):
        bits = [(val >> i) & 1 for i in range(nbits)]
        out = []
        for i in range(0, len(bits), self.stride):
            block = bits[i:i + self.kernel_size]
            if not block:
                break
            if self.pool_op == "POPCOUNT":
                r = 1 if sum(block) / len(block) >= self.schmitt_trigger else 0
            elif self.pool_op == "AND":
                r = 1 if all(block) else 0
            elif self.pool_op == "OR":
                r = 1 if any(block) else 0
            elif self.pool_op == "XOR":
                r = sum(block) % 2
            elif self.pool_op == "MAJORITY":
                r = 1 if sum(block) > len(block) / 2.0 else 0
            else:
                r = block[0]
            out.append(r)
        v = 0
        for i, b in enumerate(out):
            v |= b << i
        return v, len(out)


# ==============================================================================
# ENGINE: MSB-first, A and B oracle-decided, predicted order, DFS, deferred check
# ==============================================================================
class SearchEngine:
    WINDOW = 32

    def __init__(self):
        self.lock = threading.RLock()
        self.kernel_size = 4
        self.pool_op = "POPCOUNT"
        self.oracle = None
        self.running = False
        self.L = 2
        self._clear()
        self.last_status = "NO PROBLEM LOADED"

    def _clear(self):
        self.A = self.B = 0                 # prefixes pooled so far (MSB first)
        self.depth = 0                      # bits pooled in each register
        self.opt = [0] * (self.L + 2)       # index into the predicted order, per depth
        self.order = [None] * (self.L + 2)  # predicted order of the 4 bit pairs, per depth
        self.checks = 0                     # deferred checks performed
        self.converged = False
        self.finished = False
        self.p = self.q = 0

    def _rank(self):
        """Rank the 4 (A bit, B bit) pairs at the current depth, best prediction first."""
        o, r = self.oracle, self.L - self.depth - 1
        sc = []
        for i in range(4):
            A2, B2 = (self.A << 1) | (i >> 1), (self.B << 1) | (i & 1)
            sc.append((o.predict(A2, B2, r) if A2 <= B2 else float("inf"), i))
        sc.sort()
        return [i for _, i in sc]

    def load(self, n):
        with self.lock:
            self.running = False
            self.oracle = BitPoolingOracle(n)
            self._apply()
            self.L = n.bit_length() // 2 + 1        # balanced: both factors fit in L bits
            self._clear()
            self.order[0] = self._rank()
            self.last_status = "READY"

    def restart(self):
        with self.lock:
            if self.oracle:
                self.load(self.oracle.n)

    def set_pool(self, op=None, kernel=None):
        with self.lock:
            if op:
                self.pool_op = op
            if kernel:
                self.kernel_size = kernel
            self._apply()

    def _apply(self):
        if self.oracle:
            self.oracle.pool_op = self.pool_op
            self.oracle.kernel_size = self.kernel_size

    def _accept(self, d):
        if self.oracle.submit(d):
            self.converged = self.finished = True
            self.p, self.q = self.oracle.factors
            self.last_status = "VERIFIED - factors unsealed"
            return True
        return False

    def _backtrack(self):
        if self.depth == 0:
            self.finished, self.running = True, False
            self.last_status = "NO BALANCED FACTOR PAIR FOUND - N is prime"
            return
        self.depth -= 1
        self.A >>= 1
        self.B >>= 1

    def step(self):
        """Pool one bit of A and B (MSB first). O(1) work, no candidate loops."""
        with self.lock:
            o = self.oracle
            if o is None or self.finished:
                return
            if o.n % 2 == 0:
                if not self._accept(2):
                    self.finished = True
                return

            if self.depth == self.L:                       # both registers full: deferred check
                self.checks += 1
                if self._accept(self.A):
                    return
                self.last_status = "CHECK FAILED - backtrack"
                self._backtrack()
                return

            i = self.opt[self.depth]
            if i >= 4:                                     # all predictions rejected
                self.last_status = "BACKTRACKING"
                self._backtrack()
                return
            self.opt[self.depth] = i + 1
            combo = self.order[self.depth][i]              # i-th best predicted bit pair
            A2 = (self.A << 1) | (combo >> 1)
            B2 = (self.B << 1) | (combo & 1)
            if o.query_bit(A2, B2, self.L - self.depth - 1):   # oracle confirms
                self.A, self.B = A2, B2
                self.depth += 1
                self.opt[self.depth] = 0
                if self.depth < self.L:
                    self.order[self.depth] = self._rank()
                self.last_status = "POOLING A and B (predicted, oracle-confirmed), check deferred"
            else:
                self.last_status = "ORACLE REJECTED predicted pair - trying next best"

    def solve_all(self):
        if self.running or self.oracle is None:
            return
        self.running = True

        def work():
            while self.running and not self.finished:
                self.step()
            self.running = False
        threading.Thread(target=work, daemon=True).start()

    def stop(self):
        self.running = False

    def snapshot(self):
        W, o = self.WINDOW, self.oracle
        z = (0, 1, 0)
        s = dict(W=W, n=z, a=z, b=z, p_win=0, pool_bits=0,
                 n_txt="—", a_txt="—", b_txt="—", p_txt="",
                 status=self.last_status, level=0, front=0, maxl=0)
        if o is None:
            return s
        with self.lock:
            def top(x, width, pooled):
                shown = min(W, width)                       # MSB-aligned window
                return (x >> (width - shown), shown, min(pooled, shown))

            L, d = self.L, self.depth
            s["level"], s["maxl"], s["front"] = d, L, self.checks
            s["n"] = top(o.n, o.n.bit_length(), W)
            s["n_txt"] = f"N top {W} bits   {fmt_big(o.n)}"
            s["a"] = top(self.A << (L - d), L, d)
            s["b"] = top(self.B << (L - d), L, d)
            r = L - d
            a_pred = (self.A << r) + (((1 << r) - 1) >> 1)
            b_pred = o.n // a_pred if a_pred else 0
            s["a_txt"] = f"A predicted ≈ {fmt_big(a_pred) if a_pred else '—'}"
            s["b_txt"] = f"B predicted ≈ {fmt_big(b_pred) if b_pred else '—'}"
            s["p_win"], s["pool_bits"] = o.pool_step(s["a"][0], s["a"][1])
            s["p_txt"] = f"pooled A ({o.pool_op}, k={o.kernel_size})"
            if self.converged:
                s["a"] = top(self.p, L, L)
                s["b"] = top(self.q, L, L)
                s["a_txt"], s["b_txt"] = "P = " + fmt_big(self.p), "Q = " + fmt_big(self.q)
        return s


# ==============================================================================
# CAMERA
# ==============================================================================
class Source:
    def __init__(self, index):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            raise RuntimeError(f"Cannot open camera {index}.")
        self.cap, self.rgb = cap, None

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cap.release()


def ppm_photo(rgb):
    h, w, _ = rgb.shape
    data = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


class SimThread(threading.Thread):
    def __init__(self, cam_index, ctl, out, app):
        super().__init__(daemon=True)
        self.cam_index, self.ctl, self.out, self.app = cam_index, ctl, out, app

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out = self.ctl, self.out
        try:
            src = Source(self.cam_index)
        except RuntimeError as e:
            out.put(("error", str(e)))
            return
        idx, prev_even = 0, False
        try:
            while not ctl["stop"]:
                engine = self.app.engine
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    break
                small = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA)
                x = int(np.std(small) * 10) if (idx > 5 and np.std(small) > 5.0) else 0
                even_now = x > 0 and x % 2 == 0
                if even_now and prev_even and ctl.get("auto_step", True) and not engine.running:
                    engine.step()
                prev_even = even_now

                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                h, w = rgb.shape[:2]
                prev = rgb[np.arange(240) * h // 240][:, np.arange(320) * w // 320].copy()
                if ctl.get("ir_heatmap", False):
                    prev[:, :, 0] = np.clip(prev[:, :, 0].astype(int) + 80, 0, 255)
                    prev[:, :, 2] = np.clip(prev[:, :, 2].astype(int) - 50, 0, 255)
                out.put(("frame", dict(preview=prev)))
                idx += 1
                await asyncio.sleep(0.03)
        finally:
            src.close()


# ==============================================================================
# GUI
# ==============================================================================
class App:
    def __init__(self, window):
        self.window = window
        window.title("Bit-Pooling Factorisation Lab - MSB-first, predicted pair, deferred check")
        self.q = queue.Queue()
        self.ctl = dict(stop=False, ir_heatmap=False, auto_step=True)
        self.thread = None
        self.engine = SearchEngine()
        self._build_ui()
        window.protocol("WM_DELETE_WINDOW", self._close)
        window.after(15, self._poll)
        window.after(100, self._refresh)

    def _build_ui(self):
        mf = ttk.Frame(self.window, padding=10)
        mf.pack(fill="both", expand=True)

        cf = ttk.LabelFrame(mf, text="Laboratory Controls")
        cf.pack(fill="x", padx=5, pady=5)
        for text, cmd in [("Start Camera", self.start_lab), ("Stop Camera", self.stop_lab),
                          ("Step Bit Pool", self._step), ("Step x8", self._step8),
                          ("Solve All", self.engine.solve_all), ("Stop Solve", self.engine.stop),
                          ("Reset", self.engine.restart)]:
            ttk.Button(cf, text=text, command=cmd).pack(side="left", padx=5, pady=5)
        self.v_ir = tk.BooleanVar()
        ttk.Checkbutton(cf, text="IR Thermal Heatmap", variable=self.v_ir,
                        command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get())).pack(side="left", padx=10)

        pf = ttk.LabelFrame(mf, text="Problem: N = P x Q  (P, Q unknown)")
        pf.pack(fill="x", padx=5, pady=5)
        ttk.Label(pf, text="N:").pack(side="left", padx=(5, 2), pady=5)
        self.v_n = tk.StringVar()
        ttk.Entry(pf, width=60, textvariable=self.v_n).pack(side="left", padx=2, pady=5)
        ttk.Button(pf, text="Load", command=self.load_n).pack(side="left", padx=8, pady=5)

        df = ttk.Frame(mf)
        df.pack(fill="both", expand=True, padx=5, pady=5)
        camf = ttk.LabelFrame(df, text="Optoelectronic Circuit Camera Viewport")
        camf.pack(side="left", padx=5, pady=5, anchor="n")
        self.cam_canvas = tk.Canvas(camf, width=320, height=240, bg="black")
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_img_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        rp = ttk.Frame(df)
        rp.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        rf = ttk.LabelFrame(rp, text="Registers (top 32-bit windows)")
        rf.pack(fill="x", padx=5, pady=5)
        self.reg_canvas = tk.Canvas(rf, width=640, height=230, bg="#121414", highlightthickness=0)
        self.reg_canvas.pack(padx=5, pady=5)
        self.lbl_status = ttk.Label(rf, text="Status: IDLE", font=("Consolas", 11), foreground="#00e5ff")
        self.lbl_status.pack(anchor="w", padx=10, pady=5)

        sf = ttk.LabelFrame(rp, text="Pooling Settings")
        sf.pack(fill="both", expand=True, padx=5, pady=5)
        ttk.Label(sf, text="Pooling Operator:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.op_combo = ttk.Combobox(sf, values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"], state="readonly")
        self.op_combo.set("POPCOUNT")
        self.op_combo.grid(row=0, column=1, padx=5, pady=5)
        self.op_combo.bind("<<ComboboxSelected>>", lambda e: self.engine.set_pool(op=self.op_combo.get()))
        ttk.Label(sf, text="Kernel Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.kernel_spin = ttk.Spinbox(sf, from_=2, to=8, increment=2, width=5, command=self._kernel)
        self.kernel_spin.set(4)
        self.kernel_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)

    def _step(self):
        if not self.engine.running:
            self.engine.step()

    def _step8(self):
        for _ in range(8):
            self._step()

    def start_lab(self):
        if self.thread and self.thread.is_alive():
            return
        self.ctl["stop"] = False
        self.thread = SimThread(0, self.ctl, self.q, self)
        self.thread.start()

    def stop_lab(self):
        self.ctl["stop"] = True

    def load_n(self):
        try:
            self.engine.load(parse_int(self.v_n.get()))
        except ValueError as e:
            messagebox.showerror("Invalid N", str(e))

    def _kernel(self):
        try:
            self.engine.set_pool(kernel=int(self.kernel_spin.get()))
        except ValueError:
            pass

    def _draw(self, s):
        c = self.reg_canvas
        c.delete("all")
        rows = [("N hi  ", s["n"], "#00e676", s["n_txt"]),
                ("A     ", s["a"], "#ffb300", s["a_txt"]),
                ("B     ", s["b"], "#ff7043", s["b_txt"]),
                ("POOLED", (s["p_win"], s["pool_bits"], s["pool_bits"]), "#b388ff", s["p_txt"])]
        y = 6
        for label, (val, nbits, pooled), color, txt in rows:
            c.create_text(6, y + 10, text=label, fill="#88a0c0", anchor="w", font=("Consolas", 10, "bold"))
            bx = 70
            for i in range(nbits):                          # i = 0 is the MSB, drawn leftmost
                bit = (val >> (nbits - 1 - i)) & 1
                done = i < pooled
                lit = bit and done
                c.create_rectangle(bx, y, bx + 16, y + 22, fill=color if lit else "#1e2430", outline="#38445c")
                c.create_text(bx + 8, y + 11, text=str(bit) if done else "·",
                              fill="#000000" if lit else "#708090", font=("Consolas", 8, "bold"))
                bx += 17
            c.create_text(70, y + 33, text=txt, fill="#d2e0f0", anchor="w", font=("Consolas", 9))
            y += 55

    def _refresh(self):
        s = self.engine.snapshot()
        self._draw(s)
        self.lbl_status.config(
            text=f"Status: {s['status']} | bits pooled: {s['level']}/{s['maxl']} | checks: {s['front']:,}")
        self.window.after(100, self._refresh)

    def _poll(self):
        try:
            for _ in range(100):
                ev = self.q.get_nowait()
                if ev[0] == "frame":
                    self.photo = ppm_photo(ev[1]["preview"])
                    self.cam_canvas.itemconfig(self.cam_img_id, image=self.photo)
                elif ev[0] == "error":
                    messagebox.showerror("Camera Error", ev[1])
        except queue.Empty:
            pass
        self.window.after(30, self._poll)

    def _close(self):
        self.ctl["stop"] = True
        self.engine.stop()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.geometry("1080x700")
    root.mainloop()
