"""
Bit-Pooling Higher-Lower Search & Circuit Camera Lab
(Tkinter / OpenCV Framework with Visual Bit Registers)
Adapted from the RP-style optoelectronic decision circuit framework.

The secret is now the live BTC/USDT price from Binance, held by an isolated
oracle process (see btc_oracle.py). Every N seconds (configurable):

  1. the oracle fetches a fresh price (new epoch),
  2. the engine asks it two higher/lower questions against the previous lock
     -> prediction HIGH / LOW / FLAT,
  3. HIGH injects Shift+W, LOW injects Shift+S,
  4. the engine bit-pools (MSB first) to lock the new price as the next reference.

Requires: numpy, opencv-python (camera viewport), pynput (keystrokes).
"""

import argparse
import asyncio
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pynput.keyboard import Controller as _KbController, Key as _Key
except Exception:  # not installed, or no display server
    _KbController = _Key = None

"""
btc_oracle.py - isolated live-price oracle.

The secret (the BTC price from Binance, as an integer number of price units)
lives ONLY inside a separate OS process. Nothing else in the program can read
it; the rest of the code talks to it through OracleClient and may only ask
higher/lower questions:

    refresh()            fetch a fresh price from Binance -> starts a new epoch
    compare(trial, e)    1 if secret >= trial else 0
    verify(guess, e)     True if guess == secret (also unlocks reveal for epoch e)
    reveal(e)            the secret, but only after verify() succeeded in epoch e

Every query carries the epoch it was meant for, so an answer can never mix
two different prices.
"""

import json
import multiprocessing as mp
import random
import threading
import urllib.request

# Tried in order. data-api.binance.vision serves public market data and is
# reachable in regions where api.binance.com is blocked.
ENDPOINTS = ["https://api.binance.com", "https://data-api.binance.vision"]


class OracleError(RuntimeError):
    pass


class StaleEpochError(OracleError):
    pass


# ------------------------------------------------------------------------------
# Runs inside the oracle process only
# ------------------------------------------------------------------------------
def _fetch_binance(symbol, timeout=5.0):
    last = None
    for base in ENDPOINTS:
        try:
            url = f"{base}/api/v3/ticker/price?symbol={symbol}"
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return float(json.loads(resp.read())["price"])
        except Exception as e:  # network, HTTP, JSON - all treated alike
            last = e
    raise OracleError(f"Binance fetch failed: {last}")


def _oracle_main(conn, source, symbol, bit_depth, scale):
    max_val = (1 << bit_depth) - 1
    secret = None
    epoch = 0
    verified_epoch = -1
    sim_price = 100000.0

    def check(ep):
        if secret is None:
            raise OracleError("no price fetched yet - call refresh() first")
        if ep != epoch:
            raise StaleEpochError(f"epoch {ep} is stale (current is {epoch})")

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        cmd = msg[0]
        try:
            if cmd == "refresh":
                if source == "sim":
                    sim_price *= 1.0 + random.gauss(0, 0.0003)
                    price = sim_price
                else:
                    price = _fetch_binance(symbol)
                val = int(round(price * scale))
                if not 0 <= val <= max_val:
                    raise OracleError(
                        f"price {price} does not fit in {bit_depth} bits at scale {scale}")
                secret = val
                epoch += 1
                reply = ("ok", epoch)
            elif cmd == "compare":
                _, trial, ep = msg
                check(ep)
                reply = ("ok", 1 if secret >= trial else 0)
            elif cmd == "verify":
                _, guess, ep = msg
                check(ep)
                hit = guess == secret
                if hit:
                    verified_epoch = epoch
                reply = ("ok", hit)
            elif cmd == "reveal":
                _, ep = msg
                check(ep)
                reply = ("ok", secret if verified_epoch == epoch else None)
            elif cmd == "stop":
                conn.send(("ok", None))
                return
            else:
                raise OracleError(f"unknown command {cmd!r}")
        except OracleError as e:
            reply = ("err", type(e).__name__, str(e))
        conn.send(reply)


# ------------------------------------------------------------------------------
# Client side (what the rest of the program imports)
# ------------------------------------------------------------------------------
class OracleClient:
    def __init__(self, source="binance", symbol="BTCUSDT", bit_depth=27, scale=100):
        if source not in ("binance", "sim"):
            raise ValueError("source must be 'binance' or 'sim'")
        if not symbol.isalnum():
            raise ValueError("bad symbol")
        self.source, self.symbol = source, symbol.upper()
        self.bit_depth, self.scale = bit_depth, scale

        ctx = mp.get_context("spawn")  # same behaviour on Windows/macOS/Linux
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_oracle_main,
            args=(child, source, self.symbol, bit_depth, scale),
            daemon=True,
        )
        self._proc.start()
        child.close()
        self._lock = threading.Lock()
        self._broken = False

    def _call(self, *msg, timeout=5.0):
        with self._lock:
            if self._broken:
                raise OracleError("oracle process is not responding")
            try:
                self._conn.send(msg)
                if not self._conn.poll(timeout):
                    self._broken = True  # a late reply would desync the pipe
                    raise OracleError("oracle timed out")
                reply = self._conn.recv()
            except (BrokenPipeError, EOFError, OSError) as e:
                self._broken = True
                raise OracleError(f"oracle process died: {e}")
        if reply[0] == "ok":
            return reply[1]
        raise (StaleEpochError if reply[1] == "StaleEpochError" else OracleError)(reply[2])

    def refresh(self):
        return self._call("refresh", timeout=20.0)

    def compare(self, trial, epoch):
        return self._call("compare", int(trial), epoch)

    def verify(self, guess, epoch):
        return self._call("verify", int(guess), epoch)

    def reveal(self, epoch):
        return self._call("reveal", epoch)

    def close(self):
        try:
            self._call("stop", timeout=1.0)
        except OracleError:
            pass
        self._proc.join(1.0)
        if self._proc.is_alive():
            self._proc.terminate()


# ==============================================================================
# CONFIGURATION
# ==============================================================================
SYMBOL = "BTCUSDT"
PRICE_SCALE = 100          # price units per dollar (100 = cents)
BIT_DEPTH = 27             # 2**27-1 cents = $1,342,177.27 max
DEFAULT_INTERVAL = 5.0     # seconds between prediction updates

UP_KEY = "w"               # sent with Shift when price is HIGHER than last lock
DOWN_KEY = "s"             # sent with Shift when price is LOWER than last lock
KEY_HOLD = 0.08            # seconds the key stays down (games often poll state)


def fmt_price(units):
    return f"${units / PRICE_SCALE:,.2f}"


# ==============================================================================
# BIT-POOLING SEARCH ENGINE
# Never sees the secret: it only gets 1-bit answers from the oracle client.
# ==============================================================================
class BitPoolingSearchEngine:
    def __init__(self, oracle):
        self.oracle = oracle
        self.bit_depth = oracle.bit_depth
        self.max_val = (1 << self.bit_depth) - 1

        # Hardware & pooling settings
        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5

        self.epoch = None
        self.last_lock = None   # previous converged value = reference for predictions
        self.prev_lock = None
        self._clear("INITIALIZED")

    def _clear(self, status):
        self.pooled_bits = 0
        self.current_guess = 0
        self.step_count = 0
        self.history = [0]
        self.converged = False
        self.last_status = status
        self._update_registers()

    def begin_epoch(self, epoch):
        self.epoch = epoch
        self._clear("SEARCHING")

    def predict(self):
        """Two higher/lower questions against the previous lock."""
        ref = self.last_lock
        if ref is None:
            return None
        if self.oracle.compare(ref + 1, self.epoch):
            return "HIGH"
        if not self.oracle.compare(ref, self.epoch):
            return "LOW"
        return "FLAT"

    def step(self):
        """Advances the bit-pooling search by evaluating one bit (MSB first)."""
        if self.converged or self.step_count >= self.bit_depth:
            return
        self.step_count += 1
        shift = self.bit_depth - self.step_count
        trial = self.pooled_bits | (1 << shift)

        bit = self.oracle.compare(trial, self.epoch)
        self.pooled_bits |= bit << shift
        self.current_guess = self.pooled_bits
        self.history.append(self.current_guess)

        if self.step_count >= self.bit_depth:
            if self.oracle.verify(self.current_guess, self.epoch):
                self.converged = True
                self.prev_lock, self.last_lock = self.last_lock, self.current_guess
                self.last_status = f"LOCKED & REVEALED AT {fmt_price(self.current_guess)}"
            else:
                self.last_status = "POOL MISMATCH"
        self._update_registers()

    def lock(self):
        while not self.converged and self.step_count < self.bit_depth:
            self.step()

    def pool_step(self, val):
        """Applies hardware-style bit pooling (kernel, stride, operator) to a value."""
        bits = [(val >> i) & 1 for i in range(self.bit_depth)]
        pooled_out = []
        for i in range(0, len(bits), self.stride):
            block = bits[i : i + self.kernel_size]
            if not block:
                break
            if self.pool_op == "POPCOUNT":
                res_bit = 1 if sum(block) / len(block) >= self.schmitt_trigger else 0
            elif self.pool_op == "AND":
                res_bit = 1 if all(b == 1 for b in block) else 0
            elif self.pool_op == "OR":
                res_bit = 1 if any(b == 1 for b in block) else 0
            elif self.pool_op == "XOR":
                res_bit = sum(block) % 2
            elif self.pool_op == "MAJORITY":
                res_bit = 1 if sum(block) > (len(block) / 2.0) else 0
            else:
                res_bit = block[0]
            pooled_out.append(res_bit)

        out_val = 0
        for idx, b in enumerate(pooled_out):
            out_val |= b << idx
        return out_val

    def _update_registers(self):
        self.guess_reg = self.current_guess
        # The oracle only reveals after a successful verify() in this epoch
        self.target_reg = (self.oracle.reveal(self.epoch) or 0) if self.converged else 0
        self.pooled_register = self.pool_step(self.current_guess)

    def snapshot(self, prediction=None, key=None):
        delta = None
        if self.converged and self.prev_lock is not None:
            delta = self.last_lock - self.prev_lock
        return dict(
            epoch=self.epoch, target=self.target_reg, guess=self.current_guess,
            pooled=self.pooled_register, status=self.last_status,
            steps=self.step_count, converged=self.converged,
            prediction=prediction, key=key, delta=delta,
        )


# ==============================================================================
# KEYSTROKE INJECTION
# ==============================================================================
class KeyInjector:
    def __init__(self):
        self._kb = None
        self.error = None
        if _KbController is None:
            self.error = "pynput is not available (pip install pynput)"
            return
        try:
            self._kb = _KbController()
        except Exception as e:
            self.error = str(e)

    @property
    def available(self):
        return self._kb is not None

    def send(self, prediction):
        """HIGH -> Shift+W, LOW -> Shift+S. Returns a label for the UI."""
        if not self.available:
            raise RuntimeError(self.error)
        key = UP_KEY if prediction == "HIGH" else DOWN_KEY
        with self._kb.pressed(_Key.shift):
            self._kb.press(key)
            time.sleep(KEY_HOLD)
            self._kb.release(key)
        return f"Shift+{key.upper()}"


# ==============================================================================
# LIVE FEED THREAD: refresh -> predict -> inject -> lock, every N seconds
# ==============================================================================
class LiveFeedThread(threading.Thread):
    def __init__(self, engine, injector, ctl, out, stop_evt, kick_evt):
        super().__init__(daemon=True)
        self.engine, self.injector = engine, injector
        self.ctl, self.out = ctl, out
        self.stop_evt, self.kick_evt = stop_evt, kick_evt

    def run(self):
        while not self.stop_evt.is_set():
            t0 = time.monotonic()
            self._cycle()
            # Re-read the interval continuously so UI changes apply immediately
            while not self.stop_evt.is_set():
                if self.kick_evt.is_set():
                    self.kick_evt.clear()
                    break
                if time.monotonic() - t0 >= self.ctl["interval"]:
                    break
                self.stop_evt.wait(0.1)
        self.out.put(("done", None))

    def _cycle(self):
        eng = self.engine
        try:
            epoch = eng.oracle.refresh()
            eng.begin_epoch(epoch)
            prediction = eng.predict()
            eng.lock()
        except OracleError as e:
            self.out.put(("feed_error", str(e)))
            return

        key = None
        if prediction in ("HIGH", "LOW") and self.ctl.get("inject", True):
            try:
                key = self.injector.send(prediction)
            except Exception as e:
                self.out.put(("feed_error", f"keystroke failed: {e}"))
        self.out.put(("state", eng.snapshot(prediction, key)))


# ==============================================================================
# CAMERA SOURCE HANDLING
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
# CAMERA VIEWPORT THREAD (preview only; the oracle drives the search now)
# ==============================================================================
class SimThread(threading.Thread):
    def __init__(self, cam_index, ctl, out, stop_evt):
        super().__init__(daemon=True)
        self.cam_index, self.ctl, self.out, self.stop_evt = cam_index, ctl, out, stop_evt

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out = self.ctl, self.out
        try:
            src = Source(self.cam_index)
        except RuntimeError as e:
            out.put(("error", str(e)))
            return
        try:
            while not self.stop_evt.is_set():
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    out.put(("info", "camera stopped delivering frames"))
                    break

                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                pw, ph = 320, 240
                h_img, w_img = rgb.shape[:2]
                preview = rgb[np.arange(ph) * h_img // ph][:, np.arange(pw) * w_img // pw].copy()

                if ctl.get("ir_heatmap", False):
                    preview[:, :, 0] = np.clip(preview[:, :, 0].astype(int) + 80, 0, 255)
                    preview[:, :, 2] = np.clip(preview[:, :, 2].astype(int) - 50, 0, 255)

                out.put(("frame", dict(preview=preview)))
                await asyncio.sleep(0.03)
        finally:
            src.close()


# ==============================================================================
# MAIN GUI APPLICATION
# ==============================================================================
class App:
    def __init__(self, window, args):
        self.window = window
        self.window.title(f"Bit-Pooling Oracle Lab - live {args.symbol} feed")
        self.q = queue.Queue()
        self.ctl = dict(ir_heatmap=False, inject=True, interval=max(1.0, args.interval))

        self.oracle = OracleClient(args.source, args.symbol, BIT_DEPTH, PRICE_SCALE)
        self.engine = BitPoolingSearchEngine(self.oracle)
        self.injector = KeyInjector()
        self.stop_evt = threading.Event()
        self.kick_evt = threading.Event()
        self.feed = None
        self.cam = None
        self.symbol = args.symbol

        self._build_ui(args.interval)
        self._draw_bit_rows(0, 0, 0, False)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.after(15, self._poll)

    def _build_ui(self, interval):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill="both", expand=True)

        ctrl_frame = ttk.LabelFrame(main_frame, text="Laboratory Controls")
        ctrl_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(ctrl_frame, text="Start Lab", command=self.start_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Stop Lab", command=self.stop_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Refresh Now", command=self.refresh_now).pack(side="left", padx=5, pady=5)

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="IR Thermal Heatmap", variable=self.v_ir,
                        command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get())).pack(side="left", padx=10)

        ttk.Label(ctrl_frame, text="Update every (s):").pack(side="left", padx=(15, 2))
        self.v_interval = tk.StringVar(value=f"{interval:g}")
        spin = ttk.Spinbox(ctrl_frame, from_=1, to=3600, increment=1, width=6,
                           textvariable=self.v_interval, command=self._update_interval)
        spin.pack(side="left", padx=2)
        spin.bind("<KeyRelease>", self._update_interval)
        spin.bind("<FocusOut>", self._update_interval)

        self.v_inject = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Inject keystrokes (Shift+W / Shift+S)", variable=self.v_inject,
                        command=lambda: self.ctl.update(inject=self.v_inject.get())).pack(side="left", padx=15)

        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        cam_frame = ttk.LabelFrame(display_frame, text="Optoelectronic Circuit Camera Viewport")
        cam_frame.pack(side="left", anchor="n", padx=5, pady=5)
        self.cam_canvas = tk.Canvas(cam_frame, width=320, height=240, bg="black")
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_img_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        right_panel = ttk.Frame(display_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        reg_frame = ttk.LabelFrame(right_panel, text="Bit-Pooling Hardware Registers")
        reg_frame.pack(fill="x", padx=5, pady=5)

        self.reg_w = 700
        self.reg_canvas = tk.Canvas(reg_frame, width=self.reg_w, height=140, bg="#121414", highlightthickness=0)
        self.reg_canvas.pack(padx=5, pady=5)

        self.lbl_price = ttk.Label(reg_frame, text=f"{self.symbol}: [HIDDEN]",
                                   font=("Consolas", 14, "bold"), foreground="#00e676")
        self.lbl_price.pack(anchor="w", padx=10, pady=(5, 0))
        self.lbl_pred = ttk.Label(reg_frame, text="Prediction: -", font=("Consolas", 11), foreground="#ffb300")
        self.lbl_pred.pack(anchor="w", padx=10, pady=2)
        self.lbl_status = ttk.Label(reg_frame, text="Status: IDLE", font=("Consolas", 11), foreground="#00e5ff")
        self.lbl_status.pack(anchor="w", padx=10, pady=5)

        set_frame = ttk.LabelFrame(right_panel, text="Pooling & Algorithm Settings")
        set_frame.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(set_frame, text="Pooling Operator:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.op_combo = ttk.Combobox(set_frame, values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"], state="readonly")
        self.op_combo.set("POPCOUNT")
        self.op_combo.grid(row=0, column=1, padx=5, pady=5)
        self.op_combo.bind("<<ComboboxSelected>>", self._update_pool_op)

        ttk.Label(set_frame, text="Kernel Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.kernel_spin = ttk.Spinbox(set_frame, from_=2, to=8, increment=2, width=5, command=self._update_kernel)
        self.kernel_spin.set(4)
        self.kernel_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)

    # ---- controls ----------------------------------------------------------
    def start_lab(self):
        if self.feed is not None and self.feed.is_alive():
            return
        if self.ctl["inject"] and not self.injector.available:
            messagebox.showwarning("Keystrokes unavailable", self.injector.error or "unknown error")
        self.stop_evt = threading.Event()
        self.kick_evt = threading.Event()
        self.feed = LiveFeedThread(self.engine, self.injector, self.ctl, self.q, self.stop_evt, self.kick_evt)
        self.feed.start()
        # Camera is independent: if it fails, the feed keeps running
        self.cam = SimThread(0, self.ctl, self.q, self.stop_evt)
        self.cam.start()
        self.lbl_status.config(text="Status: RUNNING")

    def stop_lab(self):
        self.stop_evt.set()
        self.lbl_status.config(text="Status: STOPPED")

    def refresh_now(self):
        if self.feed is not None and self.feed.is_alive():
            self.kick_evt.set()

    def _update_interval(self, *_):
        try:
            self.ctl["interval"] = max(1.0, float(self.v_interval.get()))
        except ValueError:
            pass

    def _update_pool_op(self, event):
        self.engine.pool_op = self.op_combo.get()

    def _update_kernel(self):
        try:
            self.engine.kernel_size = int(self.kernel_spin.get())
        except ValueError:
            pass

    # ---- drawing -----------------------------------------------------------
    def _draw_bit_rows(self, target_val, guess_val, pooled_val, exposed):
        self.reg_canvas.delete("all")
        bit_depth = self.engine.bit_depth
        pitch = min(26, (self.reg_w - 190) // bit_depth)
        box = pitch - 2
        active_target = target_val if exposed else 0

        rows = [
            ("TARGET", active_target, "#00e676", not exposed),
            ("GUESS ", guess_val, "#ffb300", False),
            ("POOLED", pooled_val, "#b388ff", False),
        ]

        y_offset = 10
        for label, val, color, hidden in rows:
            self.reg_canvas.create_text(10, y_offset + 10, text=label, fill="#88a0c0", anchor="w",
                                        font=("Consolas", 10, "bold"))
            bx = 90
            for b in range(bit_depth - 1, -1, -1):
                bit_state = (val >> b) & 1
                self.reg_canvas.create_rectangle(
                    bx, y_offset, bx + box, y_offset + 22,
                    fill=color if (bit_state and not hidden) else "#1e2430",
                    outline="#38445c")
                txt_val = "?" if hidden else str(bit_state)
                txt_color = "#000000" if (bit_state and not hidden) else "#708090"
                self.reg_canvas.create_text(bx + box // 2, y_offset + 11, text=txt_val, fill=txt_color,
                                            font=("Consolas", 8, "bold"))
                bx += pitch
            num_str = "[HIDDEN]" if hidden else f"[{val}]"
            self.reg_canvas.create_text(bx + 5, y_offset + 11, text=num_str, fill="#d2e0f0", anchor="w",
                                        font=("Consolas", 9))
            y_offset += 40

    def _show_state(self, d):
        self._draw_bit_rows(d["target"], d["guess"], d["pooled"], d["converged"])
        self.lbl_status.config(
            text=f"Status: {d['status']} | Bit Step: {d['steps']}/{self.engine.bit_depth} | Epoch {d['epoch']}")
        if d["converged"]:
            self.lbl_price.config(text=f"{self.symbol}: {fmt_price(d['target'])}")

        pred = d["prediction"] or "- (baseline lock)"
        delta = f"  ({'+' if d['delta'] >= 0 else '-'}${abs(d['delta']) / PRICE_SCALE:,.2f})" if d["delta"] is not None else ""
        key = f"  -> sent {d['key']}" if d["key"] else ""
        self.lbl_pred.config(text=f"Prediction: {pred}{delta}{key}")

    def _poll(self):
        try:
            for _ in range(100):
                ev = self.q.get_nowait()
                if ev[0] == "frame":
                    self.photo = ppm_photo(ev[1]["preview"])
                    self.cam_canvas.itemconfig(self.cam_img_id, image=self.photo)
                elif ev[0] == "state":
                    self._show_state(ev[1])
                elif ev[0] == "feed_error":
                    self.lbl_status.config(text=f"Status: {ev[1]}")
                elif ev[0] == "error":
                    messagebox.showerror("Camera Error", ev[1])
        except queue.Empty:
            pass
        self.window.after(30, self._poll)

    def _close(self):
        self.stop_evt.set()
        self.oracle.close()
        self.window.destroy()


def parse_args():
    p = argparse.ArgumentParser(description="Bit-pooling lab driven by a live Binance price.")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="seconds between updates")
    p.add_argument("--symbol", default=SYMBOL, help="Binance symbol, e.g. BTCUSDT")
    p.add_argument("--source", choices=["binance", "sim"], default="binance",
                   help="'sim' = offline random-walk price for testing")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = App(root, args)
    root.geometry("1160x640")
    root.mainloop()
