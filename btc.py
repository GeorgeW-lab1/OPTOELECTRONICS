"""
Bit-Pooling Higher-Lower Search & Circuit Camera Lab 
(Tkinter / OpenCV Framework with Visual Bit Registers & Auto-Revealing Oracle)
Adapted from the RP-style optoelectronic decision circuit framework.
"""

import asyncio
import hashlib
import hmac
import json
import os
from itertools import zip_longest
import queue
import random
import sys
import secrets
import threading
import time
import tkinter as tk
import urllib.request
from decimal import Decimal, ROUND_HALF_EVEN
from tkinter import ttk, messagebox

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:  # WebSocket market data (pip install websocket-client)
    import websocket
except ImportError:
    websocket = None

try:  # keystroke injection (pip install pynput)
    from pynput.keyboard import Controller as KbController, Key
except Exception:
    KbController = Key = None


# ==============================================================================
# BIT-POOLING HIGHER-LOWER ORACLE CLASS
# ==============================================================================
class BitPoolingOracle:
    """
    Encapsulates the secret target and uses bit-pooling across successive 
    frames/queries to accumulate bits MSB-first, mimicking the optoelectronic circuit.
    """
    def __init__(self, bit_depth=10, target=None):
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1
        self._target = int(np.clip(target, 0, self.max_val)) if target is not None else random.randint(0, self.max_val)
        
        # Pooling configuration matching hardware settings
        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5
        
        # Fixed bit-interleave wiring for the pooling stage (see set_interleave)
        self.interleave_mode = "NONE"
        self._wire = list(range(bit_depth - 1, -1, -1))
        self._lut = None

        # Pooling accumulation state
        self.pooled_bits = 0
        self.pooled_count = 0
        self.is_done = False
    def reveal(self):
        if not self.is_done:
            return 0
        return self._target
    def query_bit(self, trial_val):
        """
        Oracle evaluation: determines if the secret target is >= the pooled trial value.
        Returns the individual bit (0 or 1) for the current bit position.
        """
        if self.is_done:
            return 0
        
        bit = 1 if self._target >= trial_val else 0
        return bit

    INTERLEAVE_MODES = ("NONE", "SHUFFLE", "ODD_EVEN", "REVERSE")

    def set_interleave(self, mode):
        """
        Builds a FIXED wiring (like a hardware bit-interleaver) that the pooling stage reads through.
        The MSB-first search itself is never reordered: its comparison (target >= trial) only makes sense
        in numeric order, and bits arrive MSB-first. Only the pooled VIEW of the word is permuted, via a
        precomputed lookup table, so applying it per step is a single array read.
          SHUFFLE  : riffle the high half with the low half (b9 b4 b8 b3 b7 b2 b6 b1 b5 b0 for 10 bits),
                     so every pooling window mixes significant and noisy bits
          ODD_EVEN : de-interleave, all odd-position bits then all even-position bits
          REVERSE  : bit reversal
        """
        if mode not in self.INTERLEAVE_MODES:
            mode = "NONE"
        n = self.bit_depth
        msb = list(range(n - 1, -1, -1))  # source bit indices, MSB first
        if mode == "SHUFFLE":
            h = (n + 1) // 2
            wire = [b for pair in zip_longest(msb[:h], msb[h:]) for b in pair if b is not None]
        elif mode == "ODD_EVEN":
            wire = msb[0::2] + msb[1::2]
        elif mode == "REVERSE":
            wire = msb[::-1]
        else:
            wire = msb
        self.interleave_mode = mode
        self._wire = wire
        self._lut = [self._permute(v) for v in range(1 << n)] if n <= 16 else None

    def _permute(self, v):
        n = self.bit_depth
        w = 0
        for k, src in enumerate(self._wire):
            w |= ((v >> src) & 1) << (n - 1 - k)
        return w

    def interleave(self, v):
        v &= self.max_val
        return self._lut[v] if self._lut is not None else self._permute(v)

    def pool_step(self, val):
        """Applies hardware-style bit pooling (kernel, stride, operator) to the interleaved value."""
        val = self.interleave(val)
        bits = [(val >> i) & 1 for i in range(self.bit_depth)]
        
        pooled_out = []
        for i in range(0, len(bits), self.stride):
            block = bits[i : i + self.kernel_size]
            if not block:
                break
                
            if self.pool_op == "POPCOUNT":
                res = sum(block) / len(block)
                res_bit = 1 if res >= self.schmitt_trigger else 0
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
            out_val |= (b << idx)
        return out_val

    @property
    def target(self):
        """Raises PermissionError if accessed before search completion."""
        if not self.is_done:
            raise PermissionError("SECURITY VIOLATION: Oracle target is classified until fully pooled!")
        return self._target


# ==============================================================================
# BIT-POOLING SEARCH ENGINE
# ==============================================================================
class BitPoolingSearchEngine:
    def __init__(self, bit_depth=10):
        self.lock = threading.RLock()
        self.sealed = False   # True while the target is backed by an unopened BTC commitment
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1
        
        self.oracle = BitPoolingOracle(self.bit_depth)
        self.current_guess = 0
        self.step_count = 0
        self.history = []
        self.converged = False
        self.last_status = "INITIALIZED"

        # Hardware & Pooling Settings
        self.kernel_size = 4
        self.stride = 2
        self.pool_op = "POPCOUNT"
        self.schmitt_trigger = 0.5
        
        self.interleave_mode = "NONE"
        self.wired_register = 0

        # Registers
        self.guess_reg = 0
        self.target_reg = 0
        self.pooled_register = 0
        
        self._update_registers()

    def reset(self, new_target=None, sealed=False):
        with self.lock:
            self.oracle = BitPoolingOracle(self.bit_depth, new_target)
            # a fresh oracle must keep the user's pooling + interleave settings (BTC ticks reset it constantly)
            self.oracle.kernel_size, self.oracle.stride = self.kernel_size, self.stride
            self.oracle.pool_op, self.oracle.schmitt_trigger = self.pool_op, self.schmitt_trigger
            self.oracle.set_interleave(self.interleave_mode)
            self.current_guess = 0
            self.step_count = 0
            self.history = [self.current_guess]
            self.converged = False
            self.sealed = sealed
            self.last_status = "SEALED" if sealed else "READY"
            self._update_registers()

    def step(self):
        with self.lock:
            self._step()

    def _step(self):
        """Advances the bit-pooling search by evaluating one bit per step."""
        if self.converged or self.step_count >= self.bit_depth:
            return

        self.step_count += 1
        
        # Build trial value based on current pooled bits (MSB first)
        shift = self.bit_depth - self.step_count
        trial_val = self.oracle.pooled_bits | (1 << shift)
        
        # Query oracle for the bit
        bit = self.oracle.query_bit(trial_val)
        
        # Accumulate bit into oracle pool
        self.oracle.pooled_bits |= (bit << shift)
        self.current_guess = self.oracle.pooled_bits
        
        self.history.append(self.current_guess)
        self._update_registers()

        # Check if full bit depth has been gathered
        if self.step_count >= self.bit_depth:
            if self.current_guess == self.oracle._target:
                self.oracle.is_done = True  # Automatically unlocks oracle access
                self.converged = True
                self.last_status = f"LOCKED & REVEALED AT {self.current_guess}"
            else:
                self.last_status = "POOL MISMATCH"

    def set_interleave(self, mode):
        with self.lock:
            self.oracle.set_interleave(mode)
            self.interleave_mode = self.oracle.interleave_mode
            self._update_registers()

    def _update_registers(self):
        self.guess_reg = self.current_guess
        try:
            self.target_reg = self.oracle.target
        except PermissionError:
            self.target_reg = 0  # Masked securely while search is active
            
        self.pooled_register = self.oracle.pool_step(self.current_guess)
        self.wired_register = self.oracle.interleave(self.current_guess)


# ==============================================================================
# BTC SEALED-PRICE ORACLE (BINANCE FEED + TIMESTAMP-KEYED COMMITMENT)
# ==============================================================================
PRICE_Q = Decimal("0.0001")  # every price is held to exactly 4 decimal places


class SealedPriceVault:
    """
    Holds exactly one price as an HMAC-SHA256 commitment keyed by its timestamp.
    The plaintext is only handed back by open(), which the predictor calls when
    the NEXT price arrives. A random nonce stops anyone brute-forcing the small
    price space from the commitment alone.
    """
    def __init__(self):
        self._price = None
        self._ts = None
        self._nonce = None
        self.commitment = None
        self.ts_us = None

    @staticmethod
    def _mac(ts_us, price, nonce):
        return hmac.new(str(ts_us).encode(), f"{price:.8f}|{nonce}".encode(), hashlib.sha256).hexdigest()

    def is_sealed(self):
        return self._price is not None

    def seal(self, price, ts_us):
        self._price, self._ts = price, ts_us
        self._nonce = secrets.token_hex(16)
        self.commitment = self._mac(ts_us, price, self._nonce)
        self.ts_us = ts_us
        return self.commitment

    def open(self):
        if self._price is None:
            return None
        ok = hmac.compare_digest(self.commitment, self._mac(self._ts, self._price, self._nonce))
        rec = dict(price=self._price, ts_us=self._ts, commitment=self.commitment, verified=ok)
        self._price = self._ts = self._nonce = None
        return rec


class BinanceFeed(threading.Thread):
    """
    Precision BTCUSDT raw-trade stream from Binance.
      * microsecond exchange timestamps (?timeUnit=MICROSECOND)
      * prices kept as exact Decimal strings, never floats
      * local clock offset vs Binance server time measured on every (re)connect (min-RTT of 5 samples),
        so lag/latency are skew-corrected; accuracy is about +/- rtt/2 (the REST clock is millisecond-grained)
      * every trade is counted, even when throttled; the emitted tick carries the trade count since the last one
    Emits ('btc_tick', {price, ts_us, recv_us, evt_us, n_trades}) at most once per min_interval_s.
    """
    HOSTS = [
        "wss://stream.binance.com:9443/ws/btcusdt@trade?timeUnit=MICROSECOND",
        "wss://stream.binance.com:443/ws/btcusdt@trade?timeUnit=MICROSECOND",
        "wss://data-stream.binance.vision/ws/btcusdt@trade?timeUnit=MICROSECOND",
    ]
    TIME_URL = "https://api.binance.com/api/v3/time"

    def __init__(self, out, stop_evt, min_interval_s=0.3):
        super().__init__(daemon=True)
        self.out, self.stop_evt = out, stop_evt
        self.min_interval_s = min_interval_s
        self.host_idx = 0
        self.offset_us = 0   # add to local epoch-us to get Binance-clock epoch-us

    @classmethod
    def measure_clock_offset(cls, samples=5):
        """Returns (offset_us, rtt_us) from the lowest-RTT of several /time samples (NTP-style midpoint)."""
        best = None
        for _ in range(samples):
            t0 = time.time_ns()
            with urllib.request.urlopen(urllib.request.Request(cls.TIME_URL, headers={"User-Agent": "bit-pool-lab/1.0"}), timeout=3) as r:
                srv_ms = json.loads(r.read().decode())["serverTime"]
            t1 = time.time_ns()
            rtt = t1 - t0
            off = srv_ms * 1_000_000 - (t0 + rtt // 2)
            if best is None or rtt < best[0]:
                best = (rtt, off)
        return best[1] // 1000, best[0] // 1000

    def run(self):
        backoff = 1.0
        while not self.stop_evt.is_set():
            url = self.HOSTS[self.host_idx % len(self.HOSTS)]
            ws = None
            try:
                try:
                    self.offset_us, rtt_us = self.measure_clock_offset()
                    clock = f"clock {self.offset_us / 1000:+.2f}ms +/-{rtt_us / 2000:.2f}ms"
                except Exception:
                    clock = "clock unsynced (local time)"
                ws = websocket.create_connection(url, timeout=5)
                ws.settimeout(2)  # short recv timeout so Stop is responsive; pings are auto-answered
                self.out.put(("btc_status", f"{url.split('/')[2]} | {clock}"))
                backoff = 1.0
                last_emit, n_trades = 0.0, 0
                while not self.stop_evt.is_set():
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        raise ConnectionError("stream closed")
                    ev = json.loads(raw)
                    if ev.get("e") != "trade":
                        continue
                    n_trades += 1
                    now = time.perf_counter()
                    if now - last_emit < self.min_interval_s:
                        continue
                    last_emit = now
                    recv_us = time.time_ns() // 1000 + self.offset_us
                    self.out.put(("btc_tick", dict(price=Decimal(ev["p"]), ts_us=int(ev["T"]), recv_us=recv_us,
                                                   evt_us=recv_us - int(ev["E"]), n_trades=n_trades)))
                    n_trades = 0
            except Exception as e:
                if self.stop_evt.is_set():
                    break
                self.out.put(("btc_error", f"{type(e).__name__}: {e}"))
                self.host_idx += 1
                self.stop_evt.wait(backoff)
                backoff = min(backoff * 2, 15.0)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass


class PythFeed(threading.Thread):
    """
    Pyth Network BTC/USD aggregate (Hermes SSE stream). Pyth publishes prices as an integer with expo -8,
    i.e. real sub-cent precision, which is where the 4 decimal places come from (spot exchanges tick at $0.01).
    Needs an API key (Hermes moved behind Authorization: Bearer on 2026-08-26). Pyth only stamps updates with
    whole seconds, so ts_us is the LOCAL receive time and exchange lag/latency are not available (shown as '-').
    Emits ('btc_tick', {price, ts_us, recv_us, evt_us, n_trades, pub_s}); n_trades = updates seen since last tick.
    """
    BASE = "https://pyth.dourolabs.app/hermes"
    BTC_USD = "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"

    def __init__(self, out, stop_evt, api_key, min_interval_s=0.3):
        super().__init__(daemon=True)
        self.out, self.stop_evt, self.api_key = out, stop_evt, api_key
        self.min_interval_s = min_interval_s
        self.offset_us = 0

    def run(self):
        url = f"{self.BASE}/v2/updates/price/stream?ids[]=0x{self.BTC_USD}&parsed=true&encoding=hex"
        backoff = 1.0
        while not self.stop_evt.is_set():
            resp = None
            try:
                req = urllib.request.Request(url, headers={
                    "Authorization": f"Bearer {self.api_key}", "Accept": "text/event-stream",
                    "User-Agent": "bit-pool-lab/1.0"})
                resp = urllib.request.urlopen(req, timeout=3)  # updates arrive ~every 400 ms; a stall means reconnect
                self.out.put(("btc_status", "Pyth Hermes | exchange timestamps are 1 s grained (local clock used)"))
                backoff = 1.0
                last_emit, n, last_slot = 0.0, 0, None
                for raw in resp:
                    if self.stop_evt.is_set():
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    parsed = json.loads(line[5:]).get("parsed") or []
                    if not parsed:
                        continue
                    upd = parsed[0]
                    slot = (upd.get("metadata") or {}).get("slot")
                    if slot is not None and slot == last_slot:
                        continue
                    last_slot = slot
                    n += 1
                    now = time.perf_counter()
                    if now - last_emit < self.min_interval_s:
                        continue
                    last_emit = now
                    pr = upd["price"]
                    price = Decimal(pr["price"]).scaleb(int(pr["expo"]))
                    recv_us = time.time_ns() // 1000
                    self.out.put(("btc_tick", dict(price=price, ts_us=recv_us, recv_us=recv_us, evt_us=None,
                                                   n_trades=n, pub_s=int(pr["publish_time"]))))
                    n = 0
                if self.stop_evt.is_set():
                    break
                raise ConnectionError("stream ended")
            except Exception as e:
                if self.stop_evt.is_set():
                    break
                self.out.put(("btc_error", f"{type(e).__name__}: {e}"))
                self.stop_evt.wait(backoff)
                backoff = min(backoff * 2, 15.0)
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass


class BtcPredictor:
    """
    Per tick t:
      1. open the commitment for P(t-1)  -> now public, HMAC verified
      2. score the pending prediction (direction of P(t) vs P(t-1))
      3. seal P(t) and load the oracle target = centre + round((P(t)-P(t-1)) / step_usd)
    The bit-pooling search then resolves that target MSB-first. Target > centre means the last
    move was up, < centre down. The prediction for the next tick is the same direction (momentum).
    """
    def __init__(self, engine, step_usd=Decimal("0.10")):
        self.engine = engine
        self.vault = SealedPriceVault()
        self.step_usd = Decimal(str(step_usd))
        self.n_trades = None     # raw trades seen since the previous tick
        self.opened = None
        self.prediction = None
        self.prediction_issued = False
        self.hits = 0
        self.total = 0
        self.last_score = ""
        self.tick_count = 0
        # fetch / time-distance stats (milliseconds)
        self.gap_us = None       # trade-time distance to the previous tick
        self.avg_gap_us = None   # running mean of gap_us
        self.lag_us = None       # local receive time minus Binance trade time
        self.evt_us = None       # Binance event time -> local receive
        self.recv_us = None      # local receive time of the latest tick
        self._gaps = 0
        self._prev_ts = None

    def on_tick(self, price, ts_us, recv_us=None, evt_us=None, n_trades=None):
        self.n_trades = n_trades
        price = Decimal(price).quantize(PRICE_Q, rounding=ROUND_HALF_EVEN)
        if self._prev_ts is not None:
            self.gap_us = ts_us - self._prev_ts
            self._gaps += 1
            self.avg_gap_us = self.gap_us if self.avg_gap_us is None else \
                self.avg_gap_us + (self.gap_us - self.avg_gap_us) / self._gaps
        self._prev_ts = ts_us
        self.recv_us = recv_us
        self.evt_us = evt_us
        self.lag_us = None if (recv_us is None or recv_us == ts_us) else recv_us - ts_us  # equal => source has no exchange timestamp
        rec = self.vault.open()
        base = None
        if rec:
            self.opened = rec
            base = rec["price"]
            if self.prediction in ("UP", "DOWN"):
                actual = "UP" if price > base else "DOWN" if price < base else "FLAT"
                if actual == "FLAT":
                    self.last_score = f"{self.prediction} -> FLAT (not scored)"
                else:
                    self.total += 1
                    self.hits += int(actual == self.prediction)
                    self.last_score = f"{self.prediction} -> {actual} {'HIT' if actual == self.prediction else 'MISS'}"
        self.vault.seal(price, ts_us)
        self.tick_count += 1
        self.prediction = None
        self.prediction_issued = False

        centre = 1 << (self.engine.bit_depth - 1)
        units = 0 if base is None else int(((price - base) / self.step_usd).to_integral_value(rounding=ROUND_HALF_EVEN))
        self.engine.reset(int(np.clip(centre + units, 0, self.engine.max_val)), sealed=True)

    def maybe_predict(self):
        """Once per tick, after the search has converged. Returns 'UP' / 'DOWN' / 'FLAT' or None."""
        if self.prediction_issued or not self.engine.converged:
            return None
        centre = 1 << (self.engine.bit_depth - 1)
        g = self.engine.current_guess
        self.prediction = "UP" if g > centre else "DOWN" if g < centre else "FLAT"
        self.prediction_issued = True
        return self.prediction


class KeyInjector:
    """Injects Shift+W (UP) or Shift+S (DOWN) into whichever window currently has focus."""
    KEYS = {"UP": "w", "DOWN": "s"}

    def __init__(self):
        self.armed = False
        self.kb = None
        if KbController is not None:
            try:
                self.kb = KbController()
            except Exception:
                self.kb = None

    def send(self, direction):
        key = self.KEYS.get(direction)
        if not (key and self.armed and self.kb):
            return False
        try:
            self.kb.press(Key.shift)
            try:
                self.kb.press(key)
                time.sleep(0.03)
                self.kb.release(key)
            finally:
                self.kb.release(Key.shift)
            return True
        except Exception:
            return False


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
# BACKGROUND SIMULATION THREAD
# ==============================================================================
class SimThread(threading.Thread):
    def __init__(self, cam_index, ctl, out, engine):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.ctl = ctl
        self.out = out
        self.engine = engine

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        ctl, out, engine = self.ctl, self.out, self.engine
        try:
            src = Source(self.cam_index)
        except RuntimeError as e:
            out.put(("error", str(e)))
            out.put(("done", None))
            return
        out.put(("info", f"USB camera {self.cam_index} opened"))

        t_prev = time.perf_counter()
        idx = 0
        prev_even = False
        try:
            while not ctl["stop"]:
                gray = await asyncio.to_thread(src.read)
                if gray is None:
                    out.put(("info", "camera stopped delivering frames"))
                    break
                now = time.perf_counter()
                dt = min(max(now - t_prev, 1e-3), 0.25)
                t_prev = now

                small = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA)
                motion_detected = idx > 5 and np.std(small) > 5.0
                x = int(np.std(small) * 10) if motion_detected else 0

                # Two sequential even motion triggers advance the bit-pooling step
                even_now = x > 0 and x % 2 == 0
                even_state = even_now and prev_even
                prev_even = even_now

                if even_state and ctl.get("auto_step", True) and not engine.converged:
                    engine.step()

                rgb = src.rgb if src.rgb is not None else np.stack([gray] * 3, axis=-1)
                pw, ph = 320, 240
                h_img, w_img = rgb.shape[:2]
                preview = rgb[np.arange(ph) * h_img // ph][:, np.arange(pw) * w_img // pw].copy()
                
                if ctl.get("ir_heatmap", False):
                    preview[:, :, 0] = np.clip(preview[:, :, 0].astype(int) + 80, 0, 255)
                    preview[:, :, 2] = np.clip(preview[:, :, 2].astype(int) - 50, 0, 255)

                # Target exposed automatically on convergence or manual force reveal
                exposed_target = engine.oracle._target if ((engine.converged or ctl.get("force_reveal", False)) and not engine.sealed) else 0

                out.put(("frame", dict(
                    preview=preview, idx=idx, dt=dt, target=exposed_target,
                    guess=engine.current_guess, pooled=engine.pooled_register,
                    status=engine.last_status, steps=engine.step_count,
                    history=list(engine.history), converged=engine.converged
                )))
                idx += 1
                await asyncio.sleep(0.03)
        finally:
            src.close()
            out.put(("done", None))


# ==============================================================================
# MAIN GUI APPLICATION
# ==============================================================================
class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Bit-Pooling Oracle Search Lab — Auto-Revealing Registers")
        self.q = queue.Queue()
        self.ctl = dict(stop=False, ir_heatmap=False, auto_step=True, force_reveal=False)
        self.engine = BitPoolingSearchEngine(bit_depth=10)
        self.thread = None
        self.btc = BtcPredictor(self.engine)
        self.keys = KeyInjector()
        self.feed = None
        self.feed_stop = threading.Event()
        self.last_key = "none"
        self.clock_note = ""

        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.after(15, self._poll)
        self.window.after(20, self._btc_autostep)

    def _build_ui(self):
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Controls Top Frame
        ctrl_frame = ttk.LabelFrame(main_frame, text="Laboratory Controls")
        ctrl_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(ctrl_frame, text="Start Lab", command=self.start_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Stop Lab", command=self.stop_lab).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Step Bit Pool", command=self.step_search).pack(side="left", padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Reset Pool", command=self.reset_search).pack(side="left", padx=5, pady=5)

        self.v_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="IR Thermal Heatmap", variable=self.v_ir,
                        command=lambda: self.ctl.update(ir_heatmap=self.v_ir.get())).pack(side="left", padx=10)

        # Target Management Frame
        target_frame = ttk.LabelFrame(main_frame, text="Hidden Target Management (Auto-Revealing Oracle)")
        target_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(target_frame, text="Target (0-1023):").pack(side="left", padx=5, pady=5)
        self.v_custom_target = tk.StringVar(value="[SECURE]")
        self.e_custom_target = ttk.Entry(target_frame, width=8, textvariable=self.v_custom_target)
        self.e_custom_target.pack(side="left", padx=5, pady=5)

        ttk.Button(target_frame, text="Set Custom", command=self.set_custom_target).pack(side="left", padx=2, pady=5)
        ttk.Button(target_frame, text="Randomize Target", command=self.randomize_target).pack(side="left", padx=5, pady=5)

        self.v_reveal = tk.BooleanVar(value=False)
        self.chk_reveal = ttk.Checkbutton(target_frame, text="Force Reveal Target", variable=self.v_reveal,
                                          command=lambda: self.ctl.update(force_reveal=self.v_reveal.get()))
        self.chk_reveal.pack(side="left", padx=15, pady=5)

        # BTC Sealed Oracle Frame
        btc_frame = ttk.LabelFrame(main_frame, text="BTC/USDT Sealed Oracle (Binance) + Keystroke Injection")
        btc_frame.pack(fill="x", padx=5, pady=5)
        row = ttk.Frame(btc_frame)
        row.pack(fill="x", padx=5, pady=3)
        self.src_combo = ttk.Combobox(row, values=["Pyth (4 dp, API key)", "Binance (2 dp native)"], state="readonly", width=20)
        self.src_combo.set("Pyth (4 dp, API key)")
        self.src_combo.pack(side="left", padx=5)
        ttk.Button(row, text="Start BTC Feed", command=self.start_btc).pack(side="left", padx=5)
        ttk.Button(row, text="Stop BTC Feed", command=self.stop_btc).pack(side="left", padx=5)
        ttk.Label(row, text="$ per count:").pack(side="left", padx=(15, 2))
        self.step_spin = ttk.Spinbox(row, from_=0.0001, to=100, increment=0.05, width=8, command=self._update_step_usd)
        self.step_spin.set("0.10")
        self.step_spin.pack(side="left")
        self.step_spin.bind("<Return>", lambda e: self._update_step_usd())
        ttk.Label(row, text="min tick ms:").pack(side="left", padx=(15, 2))
        self.tick_spin = ttk.Spinbox(row, from_=50, to=5000, increment=50, width=6, command=self._update_tick)
        self.tick_spin.set("300")
        self.tick_spin.pack(side="left")
        self.tick_spin.bind("<Return>", lambda e: self._update_tick())
        self.v_arm = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Arm Shift+W / Shift+S injection", variable=self.v_arm,
                        command=self._on_arm).pack(side="left", padx=15)

        row2 = ttk.Frame(btc_frame)
        row2.pack(fill="x", padx=5, pady=3)
        ttk.Label(row2, text="Pyth API key:").pack(side="left", padx=5)
        self.v_pyth_key = tk.StringVar(value=os.environ.get("PYTH_API_KEY", ""))
        ttk.Entry(row2, textvariable=self.v_pyth_key, show="*", width=40).pack(side="left", padx=5)

        mono = ("Consolas", 9)
        self.lbl_feed = ttk.Label(btc_frame, text="Feed: OFF", font=mono)
        self.lbl_feed.pack(anchor="w", padx=10)
        self.lbl_seal = ttk.Label(btc_frame, text="SEALED  -", font=mono)
        self.lbl_seal.pack(anchor="w", padx=10)
        self.lbl_open = ttk.Label(btc_frame, text="OPENED  -", font=mono)
        self.lbl_open.pack(anchor="w", padx=10)
        self.lbl_time = ttk.Label(btc_frame, text="TIME    -", font=mono)
        self.lbl_time.pack(anchor="w", padx=10)
        self.lbl_pred = ttk.Label(btc_frame, text="PRED    -", font=mono)
        self.lbl_pred.pack(anchor="w", padx=10, pady=(0, 5))

        # Viewport and Registers Frame
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: Circuit Camera Viewport
        cam_frame = ttk.LabelFrame(display_frame, text="Optoelectronic Circuit Camera Viewport")
        cam_frame.pack(side="left", padx=5, pady=5)
        self.cam_canvas = tk.Canvas(cam_frame, width=320, height=240, bg="black")
        self.cam_canvas.pack(padx=5, pady=5)
        self.cam_img_id = self.cam_canvas.create_image(0, 0, anchor="nw")

        # Right: Register & Hardware Status Panel
        right_panel = ttk.Frame(display_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        reg_frame = ttk.LabelFrame(right_panel, text="Bit-Pooling Hardware Registers")
        reg_frame.pack(fill="x", padx=5, pady=5)

        self.reg_canvas = tk.Canvas(reg_frame, width=480, height=180, bg="#121414", highlightthickness=0)
        self.reg_canvas.pack(padx=5, pady=5)

        self.lbl_status = ttk.Label(reg_frame, text="Status: IDLE", font=("Consolas", 11), foreground="#00e5ff")
        self.lbl_status.pack(anchor="w", padx=10, pady=5)

        # Settings Configuration Panel
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

        ttk.Label(set_frame, text="Bit Interleave:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.il_combo = ttk.Combobox(set_frame, values=list(BitPoolingOracle.INTERLEAVE_MODES), state="readonly")
        self.il_combo.set("NONE")
        self.il_combo.grid(row=2, column=1, padx=5, pady=5)
        self.il_combo.bind("<<ComboboxSelected>>", self._update_interleave)

    def start_lab(self):
        self.ctl["stop"] = False
        self.thread = SimThread(0, self.ctl, self.q, self.engine)
        self.thread.start()

    def stop_lab(self):
        self.ctl["stop"] = True

    def step_search(self):
        self.engine.step()

    def reset_search(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")

    def set_custom_target(self):
        try:
            val = int(self.v_custom_target.get())
            if 0 <= val <= self.engine.max_val:
                self.engine.reset(val)
                messagebox.showinfo("Target Set", "New custom target configured for bit pooling.")
            else:
                messagebox.showerror("Invalid Range", f"Target must be between 0 and {self.engine.max_val}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for the custom target.")

    def randomize_target(self):
        self.engine.reset()
        self.v_custom_target.set("[SECURE]")

    # ---------------- BTC oracle controls ----------------
    def _read_step_usd(self):
        try:
            return Decimal(str(max(0.0001, float(self.step_spin.get()))))
        except ValueError:
            return Decimal("0.10")

    def _read_tick_s(self):
        try:
            return max(0.05, float(self.tick_spin.get()) / 1000.0)
        except ValueError:
            return 0.3

    def _update_tick(self):
        if self.feed is not None:
            self.feed.min_interval_s = self._read_tick_s()

    def _update_step_usd(self):
        self.btc.step_usd = self._read_step_usd()

    def start_btc(self):
        if self.feed is not None:
            return
        use_pyth = self.src_combo.get().startswith("Pyth")
        key = self.v_pyth_key.get().strip()
        if use_pyth and not key:
            messagebox.showerror("Pyth API key needed", "Paste a Pyth API key (or set PYTH_API_KEY), or pick the Binance source.")
            return
        if not use_pyth and websocket is None:
            messagebox.showerror("WebSocket client missing", "Run: pip install websocket-client")
            return
        self.btc = BtcPredictor(self.engine, self._read_step_usd())
        self.feed_stop = threading.Event()
        self.feed = (PythFeed(self.q, self.feed_stop, key, self._read_tick_s()) if use_pyth
                     else BinanceFeed(self.q, self.feed_stop, self._read_tick_s()))
        self.ctl["auto_step"] = False  # the BTC loop drives the bit steps, not camera motion
        self.feed.start()
        self.lbl_feed.config(text="Feed: CONNECTING...")

    def stop_btc(self):
        self.feed_stop.set()
        self.feed = None
        self.ctl["auto_step"] = True
        self.lbl_feed.config(text="Feed: OFF")

    def _on_arm(self):
        if self.v_arm.get():
            if self.keys.kb is None:
                messagebox.showerror("Injection unavailable", "pynput is missing or has no display access.\nRun: pip install pynput")
                self.v_arm.set(False)
                return
            if not messagebox.askokcancel(
                    "Arm keystroke injection",
                    "Shift+W / Shift+S will be typed into whichever window has focus "
                    "each time a prediction is made. Continue?"):
                self.v_arm.set(False)
                return
        self.keys.armed = self.v_arm.get()

    def _btc_autostep(self):
        eng = self.engine
        self._refresh_time_label()
        if self.feed is not None and eng.sealed:
            if not eng.converged:
                eng.step()
            else:
                pred = self.btc.maybe_predict()
                if pred:
                    sent = self.keys.send(pred)
                    names = {"UP": "Shift+W", "DOWN": "Shift+S"}
                    if sent:
                        self.last_key = f"{names[pred]} sent"
                    else:
                        self.last_key = "none (flat)" if pred == "FLAT" else "not sent (disarmed)"
                    self._refresh_btc_labels()
        self.window.after(20, self._btc_autostep)

    def _refresh_time_label(self):
        b = self.btc
        if self.feed is None or b.recv_us is None:
            return
        fmt = lambda x: "-" if x is None else f"{x / 1000:,.3f}"
        now_us = time.time_ns() // 1000 + self.feed.offset_us
        age_us = now_us - b.recv_us
        self.lbl_time.config(text=f"TIME    gap={fmt(b.gap_us)}ms  avg={fmt(b.avg_gap_us)}ms  lag={fmt(b.lag_us)}ms  "
                                  f"net={fmt(b.evt_us)}ms  n={b.n_trades}  sealed={fmt(age_us)}ms")

    def _refresh_btc_labels(self):
        b, v = self.btc, self.btc.vault
        if v.is_sealed():
            self.lbl_seal.config(text=f"SEALED  ts={v.ts_us}  hmac={v.commitment[:24]}...")
        if b.opened:
            o = b.opened
            self.lbl_open.config(text=f"OPENED  ts={o['ts_us']}  ${o['price']:,.4f}  hmac {'OK' if o['verified'] else 'MISMATCH'}")
        self.lbl_pred.config(text=f"PRED    next={b.prediction or '-'}  score={b.hits}/{b.total}  "
                                  f"last={b.last_score or '-'}  key={self.last_key}")

    def _redraw_registers(self):
        eng = self.engine
        with eng.lock:
            exposed = (eng.converged or self.v_reveal.get()) and not eng.sealed
            target = eng.oracle._target if exposed else 0
            guess, pooled, wired = eng.current_guess, eng.pooled_register, eng.wired_register
            status = f"Status: {eng.last_status} | Bit Step: {eng.step_count}/{eng.bit_depth}"
        self._draw_bit_rows(target, guess, pooled, exposed, wired)
        self.lbl_status.config(text=status)

    def _update_pool_op(self, event):
        op = self.op_combo.get()
        self.engine.pool_op = op
        self.engine.oracle.pool_op = op

    def _update_interleave(self, event=None):
        self.engine.set_interleave(self.il_combo.get())

    def _update_kernel(self):
        try:
            k = int(self.kernel_spin.get())
            self.engine.kernel_size = k
            self.engine.oracle.kernel_size = k
        except ValueError:
            pass

    def _draw_bit_rows(self, target_val, guess_val, pooled_val, revealed, wired_val=0):
        self.reg_canvas.delete("all")
        bit_depth = self.engine.bit_depth
        
        # Automatically unmask target when converged or explicitly forced
        is_exposed = (revealed or self.engine.converged) and not self.engine.sealed
        active_target = target_val if is_exposed else 0

        rows = [
            ("TARGET", active_target, "#00e676", not is_exposed),
            ("GUESS ", guess_val, "#ffb300", False),
            ("WIRED ", wired_val, "#4dd0e1", False),
            ("POOLED", pooled_val, "#b388ff", False)
        ]

        y_offset = 10
        for label, val, color, hidden in rows:
            self.reg_canvas.create_text(10, y_offset + 10, text=label, fill="#88a0c0", anchor="w", font=("Consolas", 10, "bold"))
            
            bx = 90
            for b in range(bit_depth - 1, -1, -1):
                bit_state = (val >> b) & 1
                self.reg_canvas.create_rectangle(
                    bx, y_offset, bx + 22, y_offset + 22, 
                    fill=color if (bit_state and not hidden) else "#1e2430", 
                    outline="#38445c"
                )
                txt_val = "?" if hidden else str(bit_state)
                txt_color = "#000000" if (bit_state and not hidden) else "#708090"
                self.reg_canvas.create_text(bx + 11, y_offset + 11, text=txt_val, fill=txt_color, font=("Consolas", 9, "bold"))
                bx += 26
            
            num_str = "[HIDDEN]" if hidden else f"[{val:4d}]"
            self.reg_canvas.create_text(bx + 5, y_offset + 11, text=num_str, fill="#d2e0f0", anchor="w", font=("Consolas", 9))
            y_offset += 40

    def _poll(self):
        try:
            for _ in range(100):
                ev = self.q.get_nowait()
                if ev[0] == "frame":
                    d = ev[1]
                    self.photo = ppm_photo(d["preview"])
                    self.cam_canvas.itemconfig(self.cam_img_id, image=self.photo)
                    self.lbl_status.config(text=f"Status: {d['status']} | Bit Step: {d['steps']}/{self.engine.bit_depth}")
                elif ev[0] == "error":
                    messagebox.showerror("Camera Error", ev[1])
                elif ev[0] == "btc_tick" and self.feed is not None:
                    self.btc.on_tick(ev[1]["price"], ev[1]["ts_us"], ev[1].get("recv_us"), ev[1].get("evt_us"), ev[1].get("n_trades"))
                    self.lbl_feed.config(text=f"Feed: LIVE  ticks={self.btc.tick_count}  | {self.clock_note}")
                    self._refresh_btc_labels()
                elif ev[0] == "btc_status" and self.feed is not None:
                    self.clock_note = ev[1]
                    self.lbl_feed.config(text=f"Feed: CONNECTED {self.clock_note}")
                elif ev[0] == "btc_error" and self.feed is not None:
                    self.lbl_feed.config(text=f"Feed: ERROR {ev[1][:90]}")
        except queue.Empty:
            pass
        self._redraw_registers()
        self.window.after(30, self._poll)

    def _close(self):
        self.ctl["stop"] = True
        self.feed_stop.set()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("860x880")
    root.mainloop()
