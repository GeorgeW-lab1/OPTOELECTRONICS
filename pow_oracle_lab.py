"""
SHA-256 Proof-of-Work Oracle Lab (Tkinter / optional OpenCV camera viewport)

The oracle is a PoW oracle:
  INPUT algorithms  : how (data, nonce) become the bytes that get hashed
  OUTPUT algorithms : how the solved nonce / hash / data are reported
  Solver            : parallel worker PROCESSES scan the nonce space (GUI stays
                      responsive). The winning nonce stays masked ("?") in the
                      registers until found, then auto-reveals.

Expected work is ~2**difficulty hashes. Each worker does roughly 1-2 M hash/s.
Raw/BTC modes use a 32-bit nonce space (difficulty 32 has ~63% chance of a
solution existing); text modes use a 64-bit space.

BTC block submission: after solving with "BTC HEADER (dSHA256)" (or with any 80-byte
header pasted by hand), press "Submit BTC Block..." to splice the header onto your
transaction data and send it to a Bitcoin Core node with the `submitblock` RPC.
A node only accepts a block whose hash meets the nBits target inside the header
(mainnet is far out of reach for this tool; regtest, nBits 0x207fffff, works with
difficulty 1).

CLI self-test:  python pow_oracle_lab.py --selftest
"""

import base64
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

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

MASK32 = 0xFFFFFFFF


# ==============================================================================
# INPUT / OUTPUT ALGORITHMS
# ==============================================================================
def _sha256(b):
    return hashlib.sha256(b).digest()


def _dsha256(b):
    return _sha256(_sha256(b))


# name: (prefix_fn(data)->bytes, suffix_fn(nonce)->bytes, double_sha, reverse_digest)
# hashed input = prefix + suffix. The prefix is hashed once; only the suffix per nonce.
INPUT_ALGOS = {
    "TEXT + DEC NONCE": (lambda d: d, lambda n: str(n).encode(), False, False),
    "TEXT + HEX NONCE": (lambda d: d, lambda n: format(n, "x").encode(), False, False),
    "RAW + 32b LE NONCE": (lambda d: d, lambda n: n.to_bytes(4, "little"), False, False),
    "RAW + 32b BE NONCE": (lambda d: d, lambda n: n.to_bytes(4, "big"), False, False),
    "TEXT:NONCE (hashcash)": (lambda d: d + b":", lambda n: str(n).encode(), False, False),
    # 76-byte header prefix + 4-byte LE nonce, double SHA-256, byte-reversed for difficulty
    "BTC HEADER (dSHA256)": (lambda d: d[:76], lambda n: n.to_bytes(4, "little"), True, True),
}


def nonce_bits_for(algo):
    return 32 if algo.startswith(("RAW", "BTC")) else 64


def parse_data(text, algo):
    """Turn the data field into bytes. BTC header mode accepts hex or plain text."""
    if algo.startswith("BTC"):
        t = text.strip()
        if t.lower().startswith("0x"):
            t = t[2:]
        t = "".join(t.split())
        try:
            raw = bytes.fromhex(t)
            if len(raw) >= 76:
                return raw
        except ValueError:
            pass
        # Not a valid hex header: derive a deterministic 76-byte header from the text
        seed = text.encode()
        h1 = _sha256(seed)
        h2 = _sha256(h1)
        return b"\x00\x00\x00\x20" + h1 + h2 + h1[:4] + h2[:4]
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
# BITCOIN BLOCK SUBMISSION (manual, via Bitcoin Core JSON-RPC)
# ==============================================================================
def clean_hex(text):
    """Hex text (optional 0x prefix, any whitespace) -> bytes. Raises ValueError."""
    t = "".join(text.split())
    if t.lower().startswith("0x"):
        t = t[2:]
    return bytes.fromhex(t)


def bits_to_target(bits):
    """Decode the compact nBits encoding into the 256-bit target (0 if invalid/negative)."""
    exp, mant = bits >> 24, bits & 0x007FFFFF
    if bits & 0x00800000:
        return 0
    return mant >> (8 * (3 - exp)) if exp <= 3 else mant << (8 * (exp - 3))


def header_info(header):
    """Decode an 80-byte block header and check its hash against its own nBits target."""
    if len(header) != 80:
        raise ValueError(f"Block header must be exactly 80 bytes (got {len(header)}).")
    bits = int.from_bytes(header[72:76], "little")
    target = bits_to_target(bits)
    block_hash = _dsha256(header)[::-1].hex()
    return dict(
        version=int.from_bytes(header[0:4], "little"),
        prev_hash=header[4:36][::-1].hex(),
        merkle_root=header[36:68][::-1].hex(),
        time=int.from_bytes(header[68:72], "little"),
        bits=bits,
        nonce=int.from_bytes(header[76:80], "little"),
        hash=block_hash,
        meets_target=target > 0 and int(block_hash, 16) <= target,
    )


def assemble_block(header, body_hex):
    """header (80 bytes) + transaction data -> full serialized block bytes.

    `body_hex` is the tx-count varint followed by all serialized transactions. As a
    convenience, a complete block (whose first 76 header bytes match `header`) is also
    accepted: its header is dropped and replaced by the solved one.
    """
    if len(header) != 80:
        raise ValueError(f"Block header must be exactly 80 bytes (got {len(header)}).")
    try:
        raw = clean_hex(body_hex)
    except ValueError:
        raise ValueError("Transaction data is not valid hex.")
    if len(raw) > 80 and raw[:76] == header[:76]:
        raw = raw[80:]
    if not raw:
        raise ValueError("Transaction data is empty. Paste the tx-count varint + transactions "
                         "(or a full block hex) in the transactions box.")
    if raw[0] == 0:
        raise ValueError("Transaction count is 0; a block needs at least the coinbase transaction.")
    return header + raw


def btc_rpc(url, user, password, method, params, timeout=30):
    """Minimal Bitcoin Core JSON-RPC call (HTTP basic auth). Returns the 'result' field."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise RuntimeError("Node URL must start with http:// or https://")
    payload = json.dumps({"jsonrpc": "1.0", "id": "pow_oracle_lab",
                          "method": method, "params": params}).encode()
    req = urllib.request.Request(url, payload, {"Content-Type": "application/json"})
    if user or password:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError("RPC authentication failed (HTTP 401) - check user/password.")
        body = e.read()                      # Bitcoin Core reports RPC errors with HTTP 4xx/5xx + JSON
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach node at {url}: {e.reason}")
    try:
        resp = json.loads(body)
    except ValueError:
        raise RuntimeError("Node returned a non-JSON response.")
    if resp.get("error"):
        err = resp["error"]
        raise RuntimeError(f"RPC error {err.get('code')}: {err.get('message')}")
    return resp.get("result")


_REJECT_NOTES = {
    "duplicate": "the node already has this block",
    "inconclusive": "block stored but not on the best chain (side branch / orphan for now)",
    "high-hash": "block hash does not meet the target in nBits",
    "bad-prevblk": "previous block is unknown to the node (stale or wrong chain)",
    "bad-txnmrklroot": "merkle root in the header does not match the transactions",
    "bad-cb-height": "coinbase does not encode the correct block height",
    "bad-diffbits": "nBits does not match the difficulty the node expects",
    "time-too-old": "header timestamp is too old",
    "time-too-new": "header timestamp is too far in the future",
}


def describe_submit_result(mode, res):
    """Human-readable outcome of submitblock (mode='submit') or proposal validation."""
    if res is None:
        return ("VALID: the node accepts this block (proposal mode does not check proof-of-work)."
                if mode == "validate" else "ACCEPTED: the node accepted the block and will relay it.")
    note = f" - {_REJECT_NOTES[res]}" if res in _REJECT_NOTES else ""
    return f"REJECTED: {res}{note}"


# ==============================================================================
# POW ORACLE
# ==============================================================================
class PoWOracle:
    """Holds the puzzle. The winning nonce is classified until found."""

    def __init__(self, data=b"hello", input_algo="TEXT + DEC NONCE", difficulty=20):
        self.data = data
        self.input_algo = input_algo
        self.difficulty = difficulty
        self.nonce_bits = nonce_bits_for(input_algo)
        self.nonce_space = 1 << self.nonce_bits
        self.is_done = False
        self._solution = None

        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

    def _digest(self, nonce):
        prefix_fn, suffix_fn, double, rev = INPUT_ALGOS[self.input_algo]
        buf = prefix_fn(self.data) + suffix_fn(nonce)
        d = _dsha256(buf) if double else _sha256(buf)
        return buf, (d[::-1] if rev else d)

    def check(self, nonce):
        """Oracle query: (meets_difficulty, digest_bytes, hashed_input)."""
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
# PARALLEL WORKER (top-level so it can be pickled / spawned on Windows)
# ==============================================================================
def _worker(wid, nw, data, algo, diff, start, space, stop, resq, counts, bests, lastn, lasth):
    prefix_fn, suffix_fn, double, rev = INPUT_ALGOS[algo]
    base = hashlib.sha256(prefix_fn(data))
    sha = hashlib.sha256
    from_bytes = int.from_bytes
    order = "little" if rev else "big"          # little == big of the reversed digest
    target = 1 << (256 - diff) if diff else 1 << 256
    best = 1 << 256
    CH = 20000
    k, done = wid, 0                            # worker i tests indices i, i+nw, i+2nw, ...
    val, n = best, start
    while k < space:
        if stop.is_set():
            return
        end = min(k + CH * nw, space)
        for kk in range(k, end, nw):
            n = (start + kk) % space
            h = base.copy()
            h.update(suffix_fn(n))
            d = h.digest()
            if double:
                d = sha(d).digest()
            val = from_bytes(d, order)
            if val < best:
                best = val
                if val < target:
                    counts[wid] = done + (kk - k) // nw + 1
                    bests[wid] = 256 - val.bit_length()
                    resq.put(("hit", n))
                    return
        done += len(range(k, end, nw))
        k = end - ((end - k - 1) % nw) + nw - 1 if False else k + len(range(k, end, nw)) * nw
        counts[wid] = done
        bests[wid] = 256 - best.bit_length()
        if wid == 0:
            lastn.value = n
            lasth.value = val >> 224
    resq.put(("done", wid))


# ==============================================================================
# POW SEARCH ENGINE
# ==============================================================================
class PoWSearchEngine:
    REG_BITS = 32

    def __init__(self):
        self.oracle = PoWOracle()
        self.batch = 5000
        self.procs = []
        self.stop_evt = None
        self.running = False
        self.reset()

    def configure(self, data, input_algo, difficulty, start=None):
        self.stop_mp()
        self.oracle = PoWOracle(data, input_algo, difficulty)
        self.reset(start)

    def reset(self, start=None):
        self.stop_mp()
        o = self.oracle
        o.is_done, o._solution = False, None
        self.cursor = start if start is not None else random.randrange(1 << 32)
        self.start_nonce = self.cursor
        self.scanned = 0
        self.best_zero_bits = 0
        self.last_digest = bytes(32)
        self.converged = False
        self.last_status = "READY"
        self.t0 = None
        self.elapsed = 0.0
        self.nworkers = 1

    # ---- single-process stepping (Step Batch button, self-test)
    def step(self):
        if self.converged or self.running:
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
            self.best_zero_bits = max(self.best_zero_bits, z)
            if ok:
                self._set_solution(n, d, buf, z)
                break
            self.cursor = (n + 1) % o.nonce_space
        self.elapsed = time.perf_counter() - self.t0

    def _set_solution(self, n, d, buf, z):
        o = self.oracle
        o._solution = dict(nonce=n, hash=d.hex(), input=buf, zero_bits=z, difficulty=o.difficulty,
                           attempts=self.scanned, input_algo=o.input_algo)
        o.is_done = True
        self.converged = True
        self.last_status = f"LOCKED & REVEALED nonce={n}"

    # ---- multi-process solving
    def start_mp(self, nworkers):
        if self.running or self.converged:
            return
        o = self.oracle
        ctx = mp.get_context("spawn")
        self.nworkers = nworkers
        self.stop_evt = ctx.Event()
        self.counts = ctx.RawArray("Q", nworkers)
        self.bests = ctx.RawArray("i", nworkers)
        self.lastn = ctx.RawValue("Q", self.cursor)
        self.lasth = ctx.RawValue("Q", 0)
        self.resq = ctx.Queue()
        self.done_workers = 0
        self.procs = []
        for i in range(nworkers):
            p = ctx.Process(target=_worker, daemon=True, args=(
                i, nworkers, o.data, o.input_algo, o.difficulty, self.start_nonce, o.nonce_space,
                self.stop_evt, self.resq, self.counts, self.bests, self.lastn, self.lasth))
            p.start()
            self.procs.append(p)
        self.t0 = time.perf_counter()
        self.running = True
        self.last_status = f"SCANNING ({nworkers} workers)"

    def poll(self):
        """Pull progress from the workers (call from the GUI loop; non-blocking)."""
        if not self.running:
            return
        self.scanned = sum(self.counts)
        self.best_zero_bits = max([0] + list(self.bests))
        self.cursor = self.lastn.value
        self.last_digest = self.lasth.value.to_bytes(4, "big") + bytes(28)
        self.elapsed = time.perf_counter() - self.t0
        try:
            while True:
                kind, val = self.resq.get_nowait()
                if kind == "hit" and not self.converged:
                    ok, d, buf = self.oracle.check(val)
                    z = 256 - int.from_bytes(d, "big").bit_length()
                    self.scanned = sum(self.counts)
                    self.last_digest = d
                    self._set_solution(val, d, buf, z)
                    self.stop_mp(keep_status=True)
                elif kind == "done":
                    self.done_workers += 1
        except queue.Empty:
            pass
        if self.running and self.done_workers >= self.nworkers:
            self.converged = True
            self.last_status = "NONCE SPACE EXHAUSTED"
            self.stop_mp(keep_status=True)

    def stop_mp(self, keep_status=False):
        if self.stop_evt is not None:
            self.stop_evt.set()
        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        was = self.running
        self.procs = []
        self.running = False
        if was and not keep_status and not self.converged:
            self.last_status = "STOPPED"

    @property
    def hashrate(self):
        return self.scanned / self.elapsed if self.elapsed > 0 else 0.0

    def regs(self):
        o = self.oracle
        nonce = (o.solution["nonce"] & MASK32) if o.is_done else None   # masked while searching
        cur = self.cursor & MASK32
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
                time.sleep(0.05)
        finally:
            src.close()


# ==============================================================================
# GUI
# ==============================================================================
class App:
    def __init__(self, window):
        self.window = window
        window.title("SHA-256 PoW Oracle Lab - Auto-Revealing Registers")
        self.q = queue.Queue()
        self.ctl = dict(cam_stop=True, ir_heatmap=False)
        self.engine = PoWSearchEngine()
        self.cam_thread = None
        self._last_draw = 0.0
        self._rendered = False
        self._btc_busy = False
        self._submit_win = None
        self._build_ui()
        window.protocol("WM_DELETE_WINDOW", self._close)
        window.after(50, self._poll)

    def _build_ui(self):
        m = ttk.Frame(self.window, padding=10)
        m.pack(fill="both", expand=True)

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
        self.sp_diff = ttk.Spinbox(inp, from_=1, to=48, width=5)
        self.sp_diff.set(20)
        self.sp_diff.grid(row=1, column=3, padx=5)
        ttk.Label(inp, text="Start nonce (blank=random):").grid(row=1, column=4, padx=5)
        self.v_start = tk.StringVar()
        ttk.Entry(inp, textvariable=self.v_start, width=12).grid(row=1, column=5, padx=5)

        cpus = os.cpu_count() or 2
        ttk.Label(inp, text="Workers:").grid(row=2, column=0, sticky="w", padx=5, pady=3)
        self.sp_workers = ttk.Spinbox(inp, from_=1, to=cpus, width=5)
        self.sp_workers.set(max(1, cpus - 1))
        self.sp_workers.grid(row=2, column=1, sticky="w", padx=5)

        out = ttk.LabelFrame(m, text="OUTPUT: result format")
        out.pack(fill="x", padx=5, pady=5)
        ttk.Label(out, text="Output algo:").pack(side="left", padx=5, pady=5)
        self.c_out = ttk.Combobox(out, values=list(OUTPUT_ALGOS), state="readonly", width=26)
        self.c_out.set("JSON REPORT")
        self.c_out.pack(side="left", padx=5)
        self.c_out.bind("<<ComboboxSelected>>", lambda e: self._render_output())

        ctl = ttk.LabelFrame(m, text="Laboratory Controls")
        ctl.pack(fill="x", padx=5, pady=5)
        ttk.Button(ctl, text="Solve PoW", command=self.start_solve).pack(side="left", padx=5, pady=5)
        ttk.Button(ctl, text="Stop", command=self.stop_solve).pack(side="left", padx=5)
        ttk.Button(ctl, text="Step Batch", command=self.step_once).pack(side="left", padx=5)
        ttk.Button(ctl, text="Reset", command=self.reset).pack(side="left", padx=5)
        ttk.Button(ctl, text="Submit BTC Block...", command=self.open_submit_dialog).pack(side="left", padx=5)
        ttk.Button(ctl, text="Start Camera", command=self.start_cam).pack(side="left", padx=15)
        ttk.Button(ctl, text="Stop Camera", command=lambda: self.ctl.update(cam_stop=True)).pack(side="left")
        self.v_ir = tk.BooleanVar()
        ttk.Checkbutton(ctl, text="IR Heatmap", variable=self.v_ir,
                        command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get())).pack(side="left", padx=10)

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
        if diff > 28:
            note = ""
            if nonce_bits_for(algo) == 32 and diff >= 32:
                note = ("\n\nThis algorithm has a 32-bit nonce space, so a solution may not exist "
                        f"(~{100 * (1 - 2.718281828 ** (-(2 ** 32) / 2 ** diff)):.0f}% chance).")
            if not messagebox.askyesno("Heavy work", f"{diff} bits needs ~{2 ** diff:,} hashes on average. "
                                                     f"Continue?{note}"):
                return False
        self._rendered = False
        self.engine.configure(data, algo, diff, start)
        self.engine.oracle.pool_op = self.op_combo.get()
        self.engine.oracle.kernel_size = int(self.kernel_spin.get())
        return True

    def start_solve(self):
        if self.engine.running:
            return
        if not self._configure():
            return
        self.txt.delete("1.0", "end")
        try:
            nw = max(1, int(self.sp_workers.get()))
        except ValueError:
            nw = 1
        self.engine.start_mp(nw)

    def stop_solve(self):
        self.engine.stop_mp()
        self._refresh()

    def step_once(self):
        if self.engine.running:
            return
        if self.engine.t0 is None and not self._configure():
            return
        self.engine.step()
        self._refresh()

    def reset(self):
        self.engine.reset()
        self._rendered = False
        self.txt.delete("1.0", "end")
        self._refresh()

    def start_cam(self):
        if self.cam_thread and self.cam_thread.is_alive():
            return
        self.ctl["cam_stop"] = False
        self.cam_thread = CamThread(0, self.ctl, self.q)
        self.cam_thread.start()

    # ---- BTC block submission
    def _btc_log(self, msg):
        w = getattr(self, "btc_log", None)
        if w is None or not self._submit_win or not self._submit_win.winfo_exists():
            return
        w.insert("end", msg + "\n")
        w.see("end")

    def open_submit_dialog(self):
        if self._submit_win is not None and self._submit_win.winfo_exists():
            self._submit_win.lift()
            return
        win = self._submit_win = tk.Toplevel(self.window)
        win.title("Submit BTC Block to the Blockchain")
        f = ttk.Frame(win, padding=10)
        f.pack(fill="both", expand=True)

        node = ttk.LabelFrame(f, text="Bitcoin Core node (JSON-RPC)")
        node.pack(fill="x", pady=4)
        self.v_rpc_url = tk.StringVar(value="http://127.0.0.1:8332")
        self.v_rpc_user = tk.StringVar()
        self.v_rpc_pass = tk.StringVar()
        ttk.Label(node, text="URL:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(node, textvariable=self.v_rpc_url, width=48).grid(row=0, column=1, columnspan=3, sticky="we", padx=5)
        ttk.Label(node, text="User:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(node, textvariable=self.v_rpc_user, width=18).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(node, text="Password:").grid(row=1, column=2, sticky="e", padx=5)
        ttk.Entry(node, textvariable=self.v_rpc_pass, show="*", width=18).grid(row=1, column=3, sticky="w", padx=5)

        hf = ttk.LabelFrame(f, text="Block header (80 bytes hex) - blank = use the solved header")
        hf.pack(fill="x", pady=4)
        self.v_btc_header = tk.StringVar()
        ttk.Entry(hf, textvariable=self.v_btc_header).pack(fill="x", padx=5, pady=5)

        bf = ttk.LabelFrame(f, text="Transactions (hex): tx-count varint + txs, or a full block hex")
        bf.pack(fill="both", expand=True, pady=4)
        self.btc_body = tk.Text(bf, height=6, width=90, font=("Consolas", 9))
        self.btc_body.pack(fill="both", expand=True, padx=5, pady=5)

        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Check Header", command=self._btc_check).pack(side="left", padx=3)
        ttk.Button(row, text="Build Block Hex", command=self._btc_build).pack(side="left", padx=3)
        ttk.Button(row, text="Copy Block Hex", command=self._btc_copy).pack(side="left", padx=3)
        ttk.Button(row, text="Validate (no submit)", command=lambda: self._btc_send("validate")).pack(side="left", padx=12)
        ttk.Button(row, text="SUBMIT BLOCK", command=lambda: self._btc_send("submit")).pack(side="left", padx=3)

        lf = ttk.LabelFrame(f, text="Log")
        lf.pack(fill="both", expand=True, pady=4)
        self.btc_log = tk.Text(lf, height=10, width=90, font=("Consolas", 9))
        self.btc_log.pack(fill="both", expand=True, padx=5, pady=5)
        self._btc_log("Solve a BTC HEADER first (or paste a header), paste the transactions, "
                      "then Validate / Submit.")

    def _btc_header(self):
        """The 80-byte header to submit: manual override if given, else the solved one."""
        override = self.v_btc_header.get().strip()
        if override:
            try:
                header = clean_hex(override)
            except ValueError:
                raise ValueError("Header override is not valid hex.")
            if len(header) != 80:
                raise ValueError(f"Header must be exactly 80 bytes (got {len(header)}).")
            return header
        o = self.engine.oracle
        if not o.is_done or not o.input_algo.startswith("BTC"):
            raise ValueError("No solved BTC header yet. Solve with 'BTC HEADER (dSHA256)' "
                             "or paste an 80-byte header above.")
        return o.solution["input"]

    def _btc_collect(self):
        """Return (header, block_hex), or None after showing what is wrong."""
        try:
            header = self._btc_header()
            block = assemble_block(header, self.btc_body.get("1.0", "end"))
        except ValueError as e:
            messagebox.showerror("Cannot build block", str(e), parent=self._submit_win)
            return None
        return header, block.hex()

    def _btc_check(self):
        try:
            info = header_info(self._btc_header())
        except ValueError as e:
            messagebox.showerror("Header", str(e), parent=self._submit_win)
            return
        self._btc_log(
            f"version={info['version']:#x} time={info['time']} nBits={info['bits']:#010x} nonce={info['nonce']}\n"
            f"prev  ={info['prev_hash']}\nmerkle={info['merkle_root']}\nhash  ={info['hash']}\n"
            f"meets own nBits target: {'YES' if info['meets_target'] else 'NO (a node will reject it: high-hash)'}")

    def _btc_build(self):
        got = self._btc_collect()
        if got:
            self._btc_log(f"Block hex ({len(got[1]) // 2} bytes):\n{got[1]}")

    def _btc_copy(self):
        got = self._btc_collect()
        if got:
            self._submit_win.clipboard_clear()
            self._submit_win.clipboard_append(got[1])
            self._btc_log(f"Copied block hex ({len(got[1]) // 2} bytes) to clipboard.")

    def _btc_send(self, mode):
        """mode 'validate' = getblocktemplate proposal (nothing is broadcast); 'submit' = submitblock."""
        if self._btc_busy:
            return
        got = self._btc_collect()
        if not got:
            return
        header, block_hex = got
        info = header_info(header)
        url = self.v_rpc_url.get().strip()
        win = self._submit_win
        if mode == "submit":
            if not info["meets_target"] and not messagebox.askyesno(
                    "Header below target",
                    "This header's hash does NOT meet its own nBits target, so the node will "
                    "reject it. Send anyway?", parent=win):
                return
            if not messagebox.askyesno(
                    "Submit block",
                    f"Submit block\n{info['hash']}\nto {url}?\n\nIf valid, the node relays it to the "
                    f"network and this cannot be undone.", parent=win):
                return
        user, pw = self.v_rpc_user.get(), self.v_rpc_pass.get()
        self._btc_busy = True
        self._btc_log(f"[{mode}] sending {len(block_hex) // 2}-byte block {info['hash']} to {url} ...")

        def work():
            try:
                if mode == "submit":
                    res = btc_rpc(url, user, pw, "submitblock", [block_hex])
                else:
                    res = btc_rpc(url, user, pw, "getblocktemplate",
                                  [{"mode": "proposal", "data": block_hex, "rules": ["segwit"]}])
                msg = describe_submit_result(mode, res)
            except Exception as e:
                msg = f"FAILED: {e}"
            self.q.put(("btc", msg))

        threading.Thread(target=work, daemon=True).start()

    # ---- display
    def _render_output(self):
        if not self.engine.oracle.is_done:
            return
        self.txt.delete("1.0", "end")
        self.txt.insert("end", OUTPUT_ALGOS[self.c_out.get()](self.engine.oracle.solution))
        self._rendered = True

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
        self._draw_row(8, "NONCE*", nonce, "#00e676", nonce is None)
        self._draw_row(36, "CURSOR", cur, "#ffb300", False)
        self._draw_row(64, "HASH32", h32, "#40c4ff", False)
        self._draw_row(92, "POOLED", pooled, "#b388ff", False)
        eta = ""
        if e.running and e.hashrate > 0 and not e.converged:
            rem = max(0.0, 2 ** e.oracle.difficulty - e.scanned) / e.hashrate
            eta = f" | ~{rem:,.0f}s expected remaining"
        self.lbl_status.config(
            text=f"{e.last_status} | tried {e.scanned:,} | {e.hashrate:,.0f} H/s | "
                 f"best {e.best_zero_bits}/{e.oracle.difficulty} bits{eta}")
        if e.oracle.is_done and not self._rendered:
            self._render_output()

    def _poll(self):
        try:
            for _ in range(20):
                kind, payload = self.q.get_nowait()
                if kind == "cam":
                    self.photo = ppm_photo(payload)
                    self.cam_canvas.itemconfig(self.cam_img_id, image=self.photo)
                elif kind == "btc":
                    self._btc_busy = False
                    self._btc_log(payload)
                elif kind == "error":
                    messagebox.showerror("Camera Error", payload)
        except queue.Empty:
            pass
        was_running = self.engine.running
        if was_running:
            self.engine.poll()
            now = time.perf_counter()
            if now - self._last_draw > 0.25 or not self.engine.running:
                self._last_draw = now
                self._refresh()
        self.window.after(60, self._poll)

    def _close(self):
        self.ctl["cam_stop"] = True
        self.engine.stop_mp()
        self.window.destroy()


# ==============================================================================
def selftest_btc_submit():
    import http.server
    genesis = bytes.fromhex(
        "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b2"
        "7ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c")
    info = header_info(genesis)
    assert info["hash"] == "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
    assert info["meets_target"] and info["nonce"] == 2083236893
    assert bits_to_target(0x1d00ffff) == 0xFFFF << 208
    tampered = genesis[:76] + (1).to_bytes(4, "little")
    assert not header_info(tampered)["meets_target"]

    body = bytes.fromhex("01" + "ab" * 20)                       # fake tx data, 1 tx
    assert assemble_block(genesis, body.hex()) == genesis + body
    assert assemble_block(genesis, "0x" + (genesis + body).hex()) == genesis + body   # full-block form
    other = b"\x02" + genesis[1:]                                  # different prefix: pasted data is a body
    assert assemble_block(other, (genesis + body).hex()) == other + genesis + body
    assert assemble_block(tampered, (genesis + body).hex()) == tampered + body   # same prefix: header swapped
    for bad in ("", "zz", "00"):
        try:
            assemble_block(genesis, bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(req=req, auth=self.headers.get("Authorization"))
            res = "high-hash" if req["method"] == "getblocktemplate" else None
            out = json.dumps({"result": res, "error": None, "id": req["id"]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    blk = (genesis + body).hex()
    res = btc_rpc(url, "user", "pw", "submitblock", [blk])
    assert res is None and seen["req"] == {"jsonrpc": "1.0", "id": "pow_oracle_lab",
                                           "method": "submitblock", "params": [blk]}
    assert seen["auth"] == "Basic " + base64.b64encode(b"user:pw").decode()
    assert describe_submit_result("submit", res).startswith("ACCEPTED")
    res = btc_rpc(url, "user", "pw", "getblocktemplate", [{"mode": "proposal", "data": blk, "rules": ["segwit"]}])
    assert "REJECTED: high-hash" in describe_submit_result("validate", res)
    srv.shutdown()
    try:
        btc_rpc("file:///etc/passwd", "", "", "submitblock", [blk])
        raise AssertionError("non-http URL must be refused")
    except RuntimeError:
        pass
    print("BTC submit: header decode, block assembly, submitblock/proposal RPC (mock node) ... ok")


def selftest():
    selftest_btc_submit()
    for algo, data, diff in [
        ("TEXT + DEC NONCE", b"hello", 16),
        ("TEXT:NONCE (hashcash)", b"1:16:resource", 16),
        ("RAW + 32b LE NONCE", b"abc", 14),
        ("BTC HEADER (dSHA256)", parse_data("hello world", "BTC HEADER (dSHA256)"), 12),
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

    # multi-process path
    eng = PoWSearchEngine()
    eng.configure(b"hello", "TEXT + DEC NONCE", 20, start=0)
    eng.start_mp(2)
    while eng.running:
        eng.poll()
        time.sleep(0.05)
    sol = eng.oracle.solution
    print(f"MP 2 workers diff=20 nonce={sol['nonce']} hash={sol['hash'][:20]}... "
          f"verified={eng.oracle.check(sol['nonce'])[0]} rate={eng.hashrate:,.0f} H/s")


if __name__ == "__main__":
    mp.freeze_support()
    if "--selftest" in sys.argv:
        selftest()
    else:
        root = tk.Tk()
        App(root)
        root.geometry("1000x780")
        root.mainloop()
