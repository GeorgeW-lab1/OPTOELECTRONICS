"""
SHA-256 Proof-of-Work Oracle Lab (Tkinter / optional OpenCV camera viewport)

The oracle no longer does higher/lower binary search (SHA-256 has no monotonic
signal to search on). Instead it is a PoW oracle:

  INPUT algorithms  : how (data, nonce) become the bytes that get hashed
  OUTPUT algorithms : how the solved nonce / hash / data are reported
  Solver            : scans the nonce space in batches; the winning nonce stays
                      masked ("?") in the registers until it is found, then
                      auto-reveals.

Expected work is ~2**difficulty hashes. Pure Python does roughly 0.5-1 M hash/s,
so difficulty 20-24 bits is practical; 32+ bits is not.

CLI self-test:  python pow_oracle_lab.py --selftest
"""

import hashlib
import json
import queue
import random
import sys
import threading
import time

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:  # headless self-test still works
    tk = ttk = messagebox = None

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ==============================================================================
# INPUT / OUTPUT ALGORITHMS
# ==============================================================================
def _sha256(b):
    return hashlib.sha256(b).digest()


def _dsha256(b):
    return _sha256(_sha256(b))


# Each input algorithm: (build_bytes(data, nonce), hash_fn, reverse_digest_for_difficulty)
INPUT_ALGOS = {
    "TEXT + DEC NONCE": (lambda d, n: d + str(n).encode(), _sha256, False),
    "TEXT + HEX NONCE": (lambda d, n: d + format(n, "x").encode(), _sha256, False),
    "RAW + 32b LE NONCE": (lambda d, n: d + n.to_bytes(4, "little"), _sha256, False),
    "RAW + 32b BE NONCE": (lambda d, n: d + n.to_bytes(4, "big"), _sha256, False),
    "TEXT:NONCE (hashcash)": (lambda d, n: d + b":" + str(n).encode(), _sha256, False),
    # data = 76-byte header as hex (first 80-byte hex also accepted; nonce slot replaced)
    "BTC HEADER (dSHA256)": (lambda d, n: d[:76] + n.to_bytes(4, "little"), _dsha256, True),
}


def parse_data(text, algo):
    """Turn the data field into bytes. BTC header mode expects hex."""
    if algo.startswith("BTC"):
        t = text.strip()
        if t.lower().startswith("0x"):
            t = t[2:]
        t = "".join(t.split())
        try:
            raw = bytes.fromhex(t)
            if len(raw) >= 76:
                return raw  # real hex header (76 bytes, or 80 with nonce slot)
        except ValueError:
            pass
        # Not a valid hex header: derive a deterministic 76-byte header from the text
        # version(4) | prev_hash(32) | merkle_root(32) | time(4) | bits(4)
        seed = text.encode()
        h1 = _sha256(seed)
        h2 = _sha256(h1)
        return (b"\x00\x00\x00\x20" + h1 + h2 + h1[:4] + h2[:4])
    return text.encode()


def out_nonce_dec(r): return str(r["nonce"])
def out_nonce_hex(r): return "0x%08x" % r["nonce"]
def out_hash_hex(r): return r["hash"]
def out_full_hex(r): return r["input"].hex()
def out_json(r):
    return json.dumps({k: (v.hex() if isinstance(v, bytes) else v) for k, v in r.items()}, indent=2)
def out_btc_header(r): return r["input"].hex() if len(r["input"]) == 80 else "(not a BTC header input)"


OUTPUT_ALGOS = {
    "NONCE (decimal)": out_nonce_dec,
    "NONCE (hex)": out_nonce_hex,
    "HASH (hex)": out_hash_hex,
    "FULL HASHED INPUT (hex)": out_full_hex,
    "BTC HEADER (hex)": out_btc_header,
    "JSON REPORT": out_json,
}


# ==============================================================================
# POW ORACLE
# ==============================================================================
class PoWOracle:
    """Holds the puzzle. The winning nonce is classified until found."""

    def __init__(self, data=b"hello", input_algo="TEXT + DEC NONCE", difficulty=20, nonce_bits=32):
        self.data = data
        self.input_algo = input_algo
        self.difficulty = difficulty          # required leading zero bits
        self.nonce_bits = nonce_bits
        self.nonce_space = 1 << nonce_bits
        self.is_done = False
        self._solution = None                 # result dict once solved

        # pooling config (applied to hash bits for the POOLED register)
        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

    def _digest(self, nonce):
        build, hfn, rev = INPUT_ALGOS[self.input_algo]
        buf = build(self.data, nonce)
        d = hfn(buf)
        return buf, (d[::-1] if rev else d)

    def check(self, nonce):
        """Oracle query: returns (meets_difficulty, digest_bytes, hashed_input)."""
        buf, d = self._digest(nonce)
        ok = (int.from_bytes(d, "big") >> (256 - self.difficulty)) == 0 if self.difficulty else True
        return ok, d, buf

    def pool_step(self, val, width):
        bits = [(val >> i) & 1 for i in range(width)]
        out = []
        for i in range(0, len(bits), self.stride):
            block = bits[i:i + self.kernel_size]
            if not block:
                break
            op = self.pool_op
            if op == "POPCOUNT":
                b = 1 if sum(block) / len(block) >= self.schmitt_trigger else 0
            elif op == "AND":
                b = 1 if all(block) else 0
            elif op == "OR":
                b = 1 if any(block) else 0
            elif op == "XOR":
                b = sum(block) % 2
            elif op == "MAJORITY":
                b = 1 if sum(block) > len(block) / 2.0 else 0
            else:
                b = block[0]
            out.append(b)
        v = 0
        for i, b in enumerate(out):
            v |= b << i
        return v

    @property
    def solution(self):
        if not self.is_done:
            raise PermissionError("Oracle solution is classified until a valid nonce is found!")
        return self._solution


# ==============================================================================
# POW SEARCH ENGINE
# ==============================================================================
class PoWSearchEngine:
    REG_BITS = 32

    def __init__(self):
        self.oracle = PoWOracle()
        self.batch = 5000
        self.reset()

    def configure(self, data, input_algo, difficulty, start=None):
        self.oracle = PoWOracle(data, input_algo, difficulty)
        self.reset(start)

    def reset(self, start=None):
        o = self.oracle
        o.is_done, o._solution = False, None
        self.cursor = start if start is not None else random.randrange(o.nonce_space)
        self.start_nonce = self.cursor
        self.scanned = 0
        self.best_zero_bits = 0
        self.last_digest = bytes(32)
        self.converged = False
        self.last_status = "READY"
        self.t0 = None
        self.elapsed = 0.0

    def step(self):
        """Scan one batch of nonces; stop early on a hit."""
        if self.converged:
            return
        o = self.oracle
        if self.t0 is None:
            self.t0 = time.perf_counter()
            self.last_status = "SCANNING"
        for _ in range(self.batch):
            if self.scanned >= o.nonce_space:
                self.last_status = "NONCE SPACE EXHAUSTED"
                self.converged = True
                break
            n = self.cursor
            ok, d, buf = o.check(n)
            self.last_digest = d
            self.scanned += 1
            z = 256 - int.from_bytes(d, "big").bit_length()
            if z > self.best_zero_bits:
                self.best_zero_bits = z
            if ok:
                o._solution = dict(nonce=n, hash=d.hex(), input=buf, zero_bits=z,
                                   difficulty=o.difficulty, attempts=self.scanned,
                                   input_algo=o.input_algo)
                o.is_done = True
                self.converged = True
                self.last_status = f"LOCKED & REVEALED nonce={n}"
                break
            self.cursor = (n + 1) % o.nonce_space
        self.elapsed = time.perf_counter() - self.t0

    @property
    def hashrate(self):
        return self.scanned / self.elapsed if self.elapsed > 0 else 0.0

    # register views (32-bit)
    def regs(self):
        o = self.oracle
        nonce = o.solution["nonce"] if o.is_done else None   # masked while searching
        cur = self.cursor
        hash32 = int.from_bytes(self.last_digest[:4], "big")
        pooled = o.pool_step(hash32, self.REG_BITS)
        return nonce, cur, hash32, pooled


# ==============================================================================
# CAMERA (optional viewport)
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
        self.cap = cap

    def read_rgb(self):
        ok, bgr = self.cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def close(self):
        self.cap.release()


def ppm_photo(rgb):
    h, w, _ = rgb.shape
    data = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


class CamThread(threading.Thread):
    def __init__(self, index, ctl, out):
        super().__init__(daemon=True)
        self.index, self.ctl, self.out = index, ctl, out

    def run(self):
        try:
            src = Source(self.index)
        except RuntimeError as e:
            self.out.put(("error", str(e)))
            return
        try:
            while not self.ctl["cam_stop"]:
                rgb = src.read_rgb()
                if rgb is None:
                    break
                small = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
                if self.ctl.get("ir_heatmap"):
                    small[:, :, 0] = np.clip(small[:, :, 0].astype(int) + 80, 0, 255)
                    small[:, :, 2] = np.clip(small[:, :, 2].astype(int) - 50, 0, 255)
                self.out.put(("cam", small))
                time.sleep(0.03)
        finally:
            src.close()


# ==============================================================================
# SOLVER THREAD
# ==============================================================================
class SolverThread(threading.Thread):
    def __init__(self, engine, ctl, out):
        super().__init__(daemon=True)
        self.engine, self.ctl, self.out = engine, ctl, out

    def run(self):
        e = self.engine
        last = 0.0
        while not self.ctl["stop"] and not e.converged:
            e.step()
            now = time.perf_counter()
            if now - last > 0.05:
                self.out.put(("progress", None))
                last = now
        self.out.put(("progress", None))
        self.out.put(("done", None))


# ==============================================================================
# GUI
# ==============================================================================
class App:
    def __init__(self, window):
        self.window = window
        window.title("SHA-256 PoW Oracle Lab - Auto-Revealing Registers")
        self.q = queue.Queue()
        self.ctl = dict(stop=False, cam_stop=True, ir_heatmap=False)
        self.engine = PoWSearchEngine()
        self.solver = None
        self.cam_thread = None
        self._build_ui()
        window.protocol("WM_DELETE_WINDOW", self._close)
        window.after(30, self._poll)

    def _build_ui(self):
        m = ttk.Frame(self.window, padding=10)
        m.pack(fill="both", expand=True)

        # --- Input panel
        inp = ttk.LabelFrame(m, text="INPUT: puzzle data + algorithm")
        inp.pack(fill="x", padx=5, pady=5)
        ttk.Label(inp, text="Data:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.v_data = tk.StringVar(value="hello world")
        ttk.Entry(inp, textvariable=self.v_data, width=60).grid(row=0, column=1, columnspan=5, sticky="we", padx=5)

        ttk.Label(inp, text="Input algo:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.c_in = ttk.Combobox(inp, values=list(INPUT_ALGOS), state="readonly", width=24)
        self.c_in.set("TEXT + DEC NONCE")
        self.c_in.grid(row=1, column=1, padx=5)

        ttk.Label(inp, text="Difficulty (zero bits):").grid(row=1, column=2, padx=5)
        self.sp_diff = ttk.Spinbox(inp, from_=1, to=40, width=5)
        self.sp_diff.set(20)
        self.sp_diff.grid(row=1, column=3, padx=5)

        ttk.Label(inp, text="Start nonce (blank=random):").grid(row=1, column=4, padx=5)
        self.v_start = tk.StringVar()
        ttk.Entry(inp, textvariable=self.v_start, width=12).grid(row=1, column=5, padx=5)

        # --- Output panel
        out = ttk.LabelFrame(m, text="OUTPUT: result format")
        out.pack(fill="x", padx=5, pady=5)
        ttk.Label(out, text="Output algo:").pack(side="left", padx=5, pady=5)
        self.c_out = ttk.Combobox(out, values=list(OUTPUT_ALGOS), state="readonly", width=26)
        self.c_out.set("JSON REPORT")
        self.c_out.pack(side="left", padx=5)
        self.c_out.bind("<<ComboboxSelected>>", lambda e: self._render_output())

        # --- Controls
        ctl = ttk.LabelFrame(m, text="Laboratory Controls")
        ctl.pack(fill="x", padx=5, pady=5)
        ttk.Button(ctl, text="Solve PoW", command=self.start_solve).pack(side="left", padx=5, pady=5)
        ttk.Button(ctl, text="Stop", command=self.stop_solve).pack(side="left", padx=5)
        ttk.Button(ctl, text="Step Batch", command=self.step_once).pack(side="left", padx=5)
        ttk.Button(ctl, text="Reset", command=self.reset).pack(side="left", padx=5)
        ttk.Button(ctl, text="Start Camera", command=self.start_cam).pack(side="left", padx=15)
        ttk.Button(ctl, text="Stop Camera", command=lambda: self.ctl.update(cam_stop=True)).pack(side="left")
        self.v_ir = tk.BooleanVar()
        ttk.Checkbutton(ctl, text="IR Heatmap", variable=self.v_ir,
                        command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get())).pack(side="left", padx=10)

        # --- Display
        disp = ttk.Frame(m)
        disp.pack(fill="both", expand=True, padx=5, pady=5)
        cam = ttk.LabelFrame(disp, text="Camera Viewport")
        cam.pack(side="left", padx=5)
        self.cam_canvas = tk.Canvas(cam, width=320, height=240, bg="black")
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_img_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        right = ttk.Frame(disp)
        right.pack(side="right", fill="both", expand=True, padx=5)
        rf = ttk.LabelFrame(right, text="Hardware Registers (32-bit)")
        rf.pack(fill="x")
        self.reg_canvas = tk.Canvas(rf, width=620, height=130, bg="#121414", highlightthickness=0)
        self.reg_canvas.pack(padx=5, pady=5)
        self.lbl_status = ttk.Label(rf, text="Status: IDLE", font=("Consolas", 10), foreground="#00e5ff")
        self.lbl_status.pack(anchor="w", padx=10, pady=3)

        of = ttk.LabelFrame(right, text="Result")
        of.pack(fill="both", expand=True, pady=5)
        self.txt = tk.Text(of, height=8, width=70, font=("Consolas", 9))
        self.txt.pack(fill="both", expand=True, padx=5, pady=5)

        # pooling settings
        pf = ttk.LabelFrame(m, text="Pooling Settings (POOLED register)")
        pf.pack(fill="x", padx=5, pady=5)
        ttk.Label(pf, text="Operator:").pack(side="left", padx=5)
        self.op_combo = ttk.Combobox(pf, values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"],
                                     state="readonly", width=12)
        self.op_combo.set("POPCOUNT")
        self.op_combo.pack(side="left", padx=5)
        self.op_combo.bind("<<ComboboxSelected>>", lambda e: setattr(self.engine.oracle, "pool_op", self.op_combo.get()))
        ttk.Label(pf, text="Kernel:").pack(side="left", padx=5)
        self.kernel_spin = ttk.Spinbox(pf, from_=2, to=8, increment=2, width=4, command=self._kernel)
        self.kernel_spin.set(4)
        self.kernel_spin.pack(side="left")

    # ---- actions
    def _kernel(self):
        try:
            self.engine.oracle.kernel_size = int(self.kernel_spin.get())
        except ValueError:
            pass

    def _configure(self):
        algo = self.c_in.get()
        try:
            data = parse_data(self.v_data.get(), algo)
            diff = int(self.sp_diff.get())
            start = int(self.v_start.get()) if self.v_start.get().strip() else None
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return False
        if diff > 28 and not messagebox.askyesno(
                "Heavy work", f"{diff} bits needs ~{2**diff:,} hashes; pure Python may take very long. Continue?"):
            return False
        self.engine.configure(data, algo, diff, start)
        self.engine.oracle.pool_op = self.op_combo.get()
        self.engine.oracle.kernel_size = int(self.kernel_spin.get())
        return True

    def start_solve(self):
        if self.solver and self.solver.is_alive():
            return
        if not self._configure():
            return
        self.txt.delete("1.0", "end")
        self.ctl["stop"] = False
        self.solver = SolverThread(self.engine, self.ctl, self.q)
        self.solver.start()

    def stop_solve(self):
        self.ctl["stop"] = True

    def step_once(self):
        if self.solver and self.solver.is_alive():
            return
        if self.engine.t0 is None and not self._configure():
            return
        self.engine.step()
        self._refresh()

    def reset(self):
        self.ctl["stop"] = True
        self.engine.reset()
        self.txt.delete("1.0", "end")
        self._refresh()

    def start_cam(self):
        if self.cam_thread and self.cam_thread.is_alive():
            return
        self.ctl["cam_stop"] = False
        self.cam_thread = CamThread(0, self.ctl, self.q)
        self.cam_thread.start()

    # ---- display
    def _render_output(self):
        if not self.engine.oracle.is_done:
            return
        self.txt.delete("1.0", "end")
        self.txt.insert("end", OUTPUT_ALGOS[self.c_out.get()](self.engine.oracle.solution))

    def _draw_row(self, y, label, val, color, hidden):
        self.reg_canvas.create_text(8, y + 9, text=label, fill="#88a0c0", anchor="w", font=("Consolas", 9, "bold"))
        bx, w = 70, 15
        for b in range(31, -1, -1):
            bit = ((val or 0) >> b) & 1
            self.reg_canvas.create_rectangle(bx, y, bx + w, y + 18,
                                             fill=color if (bit and not hidden) else "#1e2430", outline="#38445c")
            self.reg_canvas.create_text(bx + w / 2, y + 9, text="?" if hidden else str(bit),
                                        fill="#000" if (bit and not hidden) else "#708090", font=("Consolas", 7, "bold"))
            bx += w + 1
        self.reg_canvas.create_text(bx + 4, y + 9, text="[HIDDEN]" if hidden else f"[{val}]",
                                    fill="#d2e0f0", anchor="w", font=("Consolas", 8))

    def _refresh(self):
        e = self.engine
        nonce, cur, h32, pooled = e.regs()
        self.reg_canvas.delete("all")
        hidden = nonce is None
        self._draw_row(8, "NONCE*", nonce, "#00e676", hidden)    # winning nonce, auto-reveals
        self._draw_row(36, "CURSOR", cur, "#ffb300", False)      # nonce being tried
        self._draw_row(64, "HASH32", h32, "#40c4ff", False)      # first 32 bits of last digest
        self._draw_row(92, "POOLED", pooled, "#b388ff", False)
        self.lbl_status.config(
            text=f"{e.last_status} | tried {e.scanned:,} | {e.hashrate:,.0f} H/s | "
                 f"best {e.best_zero_bits}/{e.oracle.difficulty} bits")
        if e.converged and e.oracle.is_done:
            self._render_output()

    def _poll(self):
        try:
            for _ in range(100):
                kind, payload = self.q.get_nowait()
                if kind == "progress":
                    self._refresh()
                elif kind == "cam":
                    self.photo = ppm_photo(payload)
                    self.cam_canvas.itemconfig(self.cam_img_id, image=self.photo)
                elif kind == "error":
                    messagebox.showerror("Camera Error", payload)
        except queue.Empty:
            pass
        self.window.after(30, self._poll)

    def _close(self):
        self.ctl["stop"] = True
        self.ctl["cam_stop"] = True
        self.window.destroy()


# ==============================================================================
def selftest():
    for algo, data, diff in [
        ("TEXT + DEC NONCE", b"hello", 16),
        ("TEXT:NONCE (hashcash)", b"1:16:resource", 16),
        ("RAW + 32b LE NONCE", b"abc", 14),
        ("BTC HEADER (dSHA256)", bytes(76), 12),
    ]:
        eng = PoWSearchEngine()
        eng.configure(data, algo, diff, start=0)
        eng.batch = 100000
        while not eng.converged:
            eng.step()
        sol = eng.oracle.solution
        ok, _, _ = eng.oracle.check(sol["nonce"])
        print(f"{algo:24s} diff={diff} nonce={sol['nonce']:<10d} hash={sol['hash'][:20]}... verified={ok}")
        assert ok
    print(OUTPUT_ALGOS["JSON REPORT"](sol))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        root = tk.Tk()
        App(root)
        root.geometry("1000x760")
        root.mainloop()
