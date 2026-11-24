import csv
import hashlib
import math
import queue
import random
import struct
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from pynput.keyboard import Controller as KeyController
    from pynput.keyboard import Key
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

try:
    import cv2
    from PIL import Image, ImageTk
    HAS_WEBCAM_LIBS = True
except ImportError:
    HAS_WEBCAM_LIBS = False


"""
END-TO-END PHOTONIC + MODAL LOGIC BTC & SHA-256 SURFACE PREDICTOR
(SELECT -> DRIVER A, 256-BIT RANDOM BINARY QUEUE BETWEEN OR -> 74HC74,
 NO-FLIP PROBE COMPARING QUEUE VS. AND BEFORE 150 ms DELAY + WEBCAM & BINANCE BTC,
 CIRCUIT-DRIVEN BTC + FULL 256-BIT SHA-256 SURFACE PREDICTION,
 CIRCUIT UP -> Shift+W, CIRCUIT DOWN -> Shift+S, NO WARMUP)
"""


# ---------------------------------------------------------------------------
# SSMHMF — Static Symmetric-Memetic Horizon Majority Function
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SSMHMFState:
    a: int
    b: int
    c: int
    ab: int
    ac: int
    bc: int
    surface_active: int
    signature: str
    surface_identity: int
    expanded_set: tuple[int, int, int, int, int, int]
    expanded_sum: int
    parent_majority: int
    expanded_identity: int


def evaluate_sshmf(a: int, b: int, c: int) -> SSMHMFState:
    """Deterministic single-pass SSMHMF evaluation for Boolean inputs A, B, C."""
    A = 1 if int(a) else 0
    B = 1 if int(b) else 0
    C = 1 if int(c) else 0

    ab = A & B
    ac = A & C
    bc = B & C

    surface_active = int(bool(ab | ac | bc))
    signature = f"{A}{B}{C}"
    surface_identity = 4 * A + 2 * B + C

    expanded_set = (A, B, C, ab, ac, bc)
    expanded_sum = sum(expanded_set)
    parent_majority = 1 if expanded_sum >= 4 else 0
    expanded_identity = 32 * A + 16 * B + 8 * C + 4 * ab + 2 * ac + bc

    return SSMHMFState(
        a=A,
        b=B,
        c=C,
        ab=ab,
        ac=ac,
        bc=bc,
        surface_active=surface_active,
        signature=signature,
        surface_identity=surface_identity,
        expanded_set=expanded_set,
        expanded_sum=expanded_sum,
        parent_majority=parent_majority,
        expanded_identity=expanded_identity,
    )


def verify_sshmf_all_8() -> list[SSMHMFState]:
    states = [evaluate_sshmf(*[(i >> 2) & 1, (i >> 1) & 1, i & 1]) for i in range(8)]
    assert len({s.surface_identity for s in states}) == 8, "Duplicate 3-bit surface identities"
    assert len({s.expanded_identity for s in states}) == 8, "Duplicate 6-bit set identities"
    for s in states:
        assert s.surface_active == (1 if (s.a + s.b + s.c) >= 2 else 0)
        assert s.parent_majority == (1 if s.expanded_sum >= 4 else 0)
    return states


# ---------------------------------------------------------------------------
# Full 256-Bit SHA-256 Data Surface Predicted Via Circuit Nodes
# ---------------------------------------------------------------------------

def _bytes_to_bits256(digest: bytes) -> list[int]:
    """Convert 32-byte SHA-256 digest into 256 MSB-first bits (index 0 = MSB of byte 0)."""
    bits: list[int] = []
    for byte in digest[:32]:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _bits256_to_bytes(bits: list[int]) -> bytes:
    """Pack 256 bits into 32 bytes."""
    out = bytearray(32)
    for idx in range(256):
        if bits[idx] & 1:
            out[idx // 8] |= 1 << (7 - (idx % 8))
    return bytes(out)


class SHA256CircuitSurfacePredictor:
    """
    Predicts the entire 256-bit (32-byte / 64-nibble / 8x32-bit word) SHA-256
    data surface directly from the live photonic/logic circuit nodes.

    Each bit position i in [0..255] is evaluated from a deterministic circuit
    tap combining the physical/logic signals of the circuit:
      - pre_delay_and, delayed_and, D_queued, Q_SELECT, not_Q, Y_or, D_raw_or, a, b
      - continuous analog RC voltage & optical power phase projection
      - SSMHMF state (expanded_identity, parent_majority, surface_identity)
      - 256-bit queue tap bit (i % queue_len)
    """

    NUM_BITS = 256

    def __init__(self):
        self.target_digest: bytes = hashlib.sha256(b"init:0").digest()
        self.target_bits: list[int] = _bytes_to_bits256(self.target_digest)
        self.pred_bits: list[int] = [0] * self.NUM_BITS
        self.pred_digest: bytes = bytes(32)
        self.bit_matches: list[int] = [0] * self.NUM_BITS

        self.last_bit_hits: int = 0
        self.last_bit_acc: float = 0.0
        self.last_hamming_dist: int = self.NUM_BITS
        self.surface_pred_bit: int = 0
        self.surface_target_bit: int = 0
        self.surface_hit: int = 0

        self.total_steps_evaluated: int = 0
        self.total_bit_hits_acc: int = 0
        self.total_bits_evaluated: int = 0
        self.surface_hits_total: int = 0
        self.surface_trials_total: int = 0

        # Precompute static per-bit circuit tap masks & phase weights across all 256 bits
        self._tap_masks: list[int] = []
        self._phase_weights: list[float] = []
        for i in range(self.NUM_BITS):
            h = hashlib.blake2b(f"sha256_surface_bit:{i}".encode("ascii"), digest_size=8).digest()
            mask_val = int.from_bytes(h[:6], "big")
            weight_val = (int.from_bytes(h[6:8], "big") / 65535.0) * 2.0 - 1.0
            self._tap_masks.append(mask_val)
            self._phase_weights.append(weight_val)

    def reset(self):
        self.__init__()

    @staticmethod
    def _build_surface_message(
        step_index: int,
        time_s: float,
        btc_price_usd: float | None,
        btc_delta_usd: float,
        queue_bits: list[int],
        pre_delay_and: int,
        delayed_and: int,
        d_queued: int,
        q: int,
    ) -> bytes:
        price_centi = int(round((btc_price_usd or 60000.0) * 100.0))
        delta_centi = int(round(btc_delta_usd * 100.0))
        q_word = 0
        for b in queue_bits[:64]:
            q_word = ((q_word << 1) | (int(b) & 1)) & 0xFFFFFFFFFFFFFFFF
        payload = struct.pack(
            ">QdqqBBBBBBI",
            step_index & 0xFFFFFFFFFFFFFFFF,
            float(time_s),
            price_centi,
            delta_centi,
            int(pre_delay_and) & 1,
            int(delayed_and) & 1,
            int(d_queued) & 1,
            int(q) & 1,
            1,
            1,
            q_word & 0xFFFFFFFF,
        )
        return payload

    def predict_and_score(
        self,
        step_index: int,
        time_s: float,
        a: int,
        b: int,
        pre_delay_and: int,
        delayed_and: int,
        y_or: int,
        d_raw_or: int,
        d_queued: int,
        q: int,
        not_q: int,
        rc_voltage: float,
        optical_tx_mw: float,
        sshmf: SSMHMFState,
        queue_bits: list[int],
        btc_price_usd: float | None,
        btc_delta_usd: float,
    ) -> dict:
        # 1. Compute live 256-bit SHA-256 target data surface
        msg = self._build_surface_message(
            step_index=step_index,
            time_s=time_s,
            btc_price_usd=btc_price_usd,
            btc_delta_usd=btc_delta_usd,
            queue_bits=queue_bits,
            pre_delay_and=pre_delay_and,
            delayed_and=delayed_and,
            d_queued=d_queued,
            q=q,
        )
        # Double-SHA256 surface over the circuit/market state frame
        self.target_digest = hashlib.sha256(hashlib.sha256(msg).digest()).digest()
        self.target_bits = _bytes_to_bits256(self.target_digest)

        # 2. Pack live circuit logic vector into a node bus
        logic_bus = (
            ((int(pre_delay_and) & 1) << 0)
            | ((int(delayed_and) & 1) << 1)
            | ((int(d_queued) & 1) << 2)
            | ((int(q) & 1) << 3)
            | ((int(not_q) & 1) << 4)
            | ((int(y_or) & 1) << 5)
            | ((int(d_raw_or) & 1) << 6)
            | ((int(a) & 1) << 7)
            | ((int(b) & 1) << 8)
            | ((int(sshmf.parent_majority) & 1) << 9)
            | ((int(sshmf.surface_active) & 1) << 10)
            | ((int(sshmf.expanded_identity) & 0x3F) << 11)
            | ((int(q & d_queued) & 1) << 17)
            | ((int(pre_delay_and ^ delayed_and) & 1) << 18)
            | ((int(d_raw_or ^ d_queued) & 1) << 19)
        )

        # Analog / photonic continuous projection from RC voltage & optical power
        analog_phase = math.sin(rc_voltage * 3.141592653589793 + optical_tx_mw * 0.17)
        q_len = max(1, len(queue_bits))

        # 3. Predict all 256 bits of the SHA-256 data surface via the circuit
        new_pred_bits = [0] * self.NUM_BITS
        new_matches = [0] * self.NUM_BITS
        hits = 0

        for idx in range(self.NUM_BITS):
            mask_val = self._tap_masks[idx]
            # Parity of active circuit node taps selected by this bit's mask
            selected = logic_bus & mask_val
            # Fast population parity via Kernighan's bit clear
            parity = 0
            tmp = selected
            while tmp:
                parity ^= 1
                tmp &= tmp - 1

            q_tap_bit = int(queue_bits[idx % q_len]) & 1
            analog_bit = 1 if (analog_phase * self._phase_weights[idx] + (0.25 if q else -0.25)) >= 0.0 else 0

            # Circuit combination across logic parity, 256-bit queue tap, and analog/SELECT state
            pred_bit = (parity ^ q_tap_bit ^ analog_bit) & 1
            if q and (idx & 1):
                pred_bit ^= pre_delay_and & 1
            elif not q and not (idx & 1):
                pred_bit ^= delayed_and & 1

            new_pred_bits[idx] = pred_bit
            tgt_bit = self.target_bits[idx]
            matched = 1 if pred_bit == tgt_bit else 0
            new_matches[idx] = matched
            hits += matched

        self.pred_bits = new_pred_bits
        self.pred_digest = _bits256_to_bytes(self.pred_bits)
        self.bit_matches = new_matches

        self.last_bit_hits = hits
        self.last_bit_acc = hits / self.NUM_BITS
        self.last_hamming_dist = self.NUM_BITS - hits

        # Aggregate surface-level prediction (majority across the 256 predicted bits)
        self.surface_pred_bit = 1 if sum(self.pred_bits) >= (self.NUM_BITS // 2) else 0
        self.surface_target_bit = 1 if sum(self.target_bits) >= (self.NUM_BITS // 2) else 0
        self.surface_hit = int(self.surface_pred_bit == self.surface_target_bit)

        self.total_steps_evaluated += 1
        self.total_bit_hits_acc += hits
        self.total_bits_evaluated += self.NUM_BITS
        self.surface_hits_total += self.surface_hit
        self.surface_trials_total += 1

        return self.summary()

    def cumulative_bit_accuracy(self) -> float:
        if self.total_bits_evaluated <= 0:
            return 0.0
        return self.total_bit_hits_acc / self.total_bits_evaluated

    def cumulative_surface_accuracy(self) -> float:
        if self.surface_trials_total <= 0:
            return 0.0
        return self.surface_hits_total / self.surface_trials_total

    def summary(self) -> dict:
        pred_hex = self.pred_digest.hex()
        tgt_hex = self.target_digest.hex()
        nibble_hits = sum(
            1
            for n_idx in range(64)
            if self.pred_bits[n_idx * 4 : n_idx * 4 + 4] == self.target_bits[n_idx * 4 : n_idx * 4 + 4]
        )
        return {
            "pred_hex": pred_hex,
            "target_hex": tgt_hex,
            "bit_hits": self.last_bit_hits,
            "bit_acc": self.last_bit_acc,
            "cum_bit_acc": self.cumulative_bit_accuracy(),
            "hamming_dist": self.last_hamming_dist,
            "nibble_hits": nibble_hits,
            "surface_pred_bit": self.surface_pred_bit,
            "surface_target_bit": self.surface_target_bit,
            "surface_hit": self.surface_hit,
            "cum_surface_acc": self.cumulative_surface_accuracy(),
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def mw_to_dbm(power_mw: float) -> float:
    return 10.0 * math.log10(max(power_mw, 1e-12))


def db_to_linear(db_val: float) -> float:
    return 10.0 ** (db_val / 10.0)


def pct(hits: int, n: int) -> float:
    return hits / n if n else 0.0


def z_score(hits: int, n: int) -> float:
    """z-score of hit count vs a fair coin."""
    if n < 1:
        return 0.0
    return (hits - 0.5 * n) / math.sqrt(0.25 * n)


# ---------------------------------------------------------------------------
# Market data: price + order book, with endpoint fallback
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

BINANCE_HOSTS = (
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api.binance.com",
)


def _http_json(url: str, params: dict | None = None, timeout: tuple = (1.0, 1.5)):
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class _SourceChain:
    """Tries sources in order, starting from whichever worked last time."""

    COOLDOWN_S = 60.0

    def __init__(self, sources: list):
        self.sources = sources
        self.best = 0
        self.cool: dict[int, float] = {}
        self._lock = threading.Lock()

    def call(self):
        if not HAS_REQUESTS:
            raise RuntimeError("requests package not installed (pip install requests)")
        errors: list[str] = []
        n = len(self.sources)
        now = time.monotonic()
        with self._lock:
            order = [(self.best + k) % n for k in range(n)]
            active = [i for i in order if i not in self.cool]
            if not active:
                probation = [i for i in order if self.cool[i] <= now]
                active = probation[:1]
        if not active:
            raise RuntimeError("all price sources failed recently; retrying shortly")
        for i in active:
            label, fn = self.sources[i]
            try:
                value = fn()
                with self._lock:
                    self.best = i
                    self.cool.pop(i, None)
                return value, label
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                with self._lock:
                    self.cool[i] = time.monotonic() + self.COOLDOWN_S
        raise RuntimeError(" | ".join(errors[-3:]))


def _binance_price_fn(host: str):
    def fn():
        payload = _http_json(f"{host}/api/v3/ticker/price", {"symbol": "BTCUSDT"})
        if "price" not in payload:
            raise ValueError(f"unexpected payload: {payload}")
        return float(payload["price"])
    return fn


def _coinbase_price() -> float:
    return float(_http_json("https://api.exchange.coinbase.com/products/BTC-USD/ticker")["price"])


def _kraken_price() -> float:
    data = _http_json("https://api.kraken.com/0/public/Ticker", {"pair": "XBTUSD"})
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    key = next(iter(data["result"]))
    return float(data["result"][key]["c"][0])


def _binance_depth_fn(host: str):
    def fn():
        data = _http_json(f"{host}/api/v3/depth", {"symbol": "BTCUSDT", "limit": 20})
        bids = [(float(p), float(q)) for p, q, *_ in data["bids"]]
        asks = [(float(p), float(q)) for p, q, *_ in data["asks"]]
        return bids, asks
    return fn


def _coinbase_depth():
    data = _http_json("https://api.exchange.coinbase.com/products/BTC-USD/book", {"level": 2})
    bids = [(float(p), float(q)) for p, q, *_ in data["bids"][:20]]
    asks = [(float(p), float(q)) for p, q, *_ in data["asks"][:20]]
    return bids, asks


def _kraken_depth():
    data = _http_json("https://api.kraken.com/0/public/Depth", {"pair": "XBTUSD", "count": 20})
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    key = next(iter(data["result"]))
    book = data["result"][key]
    bids = [(float(p), float(q)) for p, q, *_ in book["bids"]]
    asks = [(float(p), float(q)) for p, q, *_ in book["asks"]]
    return bids, asks


PRICE_CHAIN = _SourceChain(
    [(h.split("//")[1], _binance_price_fn(h)) for h in BINANCE_HOSTS]
    + [("coinbase(BTC-USD)", _coinbase_price), ("kraken(XBTUSD)", _kraken_price)]
)
DEPTH_CHAIN = _SourceChain(
    [(h.split("//")[1], _binance_depth_fn(h)) for h in BINANCE_HOSTS]
    + [("coinbase(BTC-USD)", _coinbase_depth), ("kraken(XBTUSD)", _kraken_depth)]
)


def fetch_btc_usdt_price() -> tuple[float, str]:
    return PRICE_CHAIN.call()


def book_imbalance(bids: list, asks: list, levels: int = 10) -> float:
    """Distance-weighted bid/ask volume imbalance in [-1, 1]. Positive = more bid pressure."""
    b = sum(q / (i + 1) for i, (_, q) in enumerate(bids[:levels]))
    a = sum(q / (i + 1) for i, (_, q) in enumerate(asks[:levels]))
    total = a + b
    return (b - a) / total if total > 0 else 0.0


class BinanceBtcPoller:
    """Polls BTC spot price and order-book imbalance with simulated fallback."""

    SIM_SOURCE = "SIMULATED (random walk)"

    def __init__(self, poll_interval_s: float = 1.0, feed_mode: str = "Auto"):
        self.poll_interval_s = max(0.25, float(poll_interval_s))
        self.feed_mode = feed_mode
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._depth_thread = threading.Thread(target=self._depth_loop, daemon=True)

        self.price_usd: float | None = None
        self.prev_price_usd: float | None = None
        self.delta_usd: float = 0.0
        self.delta_cents: int = 0
        self.direction: str = "FLAT"
        self.last_fetch_mono: float = 0.0
        self.last_error: str | None = None
        self.fetch_count: int = 0
        self.source: str = ""
        self.depth_source: str = ""
        self.imbalance: float | None = None
        self.epoch = 0

        self._kind: str | None = None
        self._live_fail = 0
        self._depth_fail = 0
        self._sim_price = 60000.0
        self._sim_rng = random.Random()

    def start(self):
        if not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        if not self._depth_thread.is_alive():
            self._depth_thread = threading.Thread(target=self._depth_loop, daemon=True)
            self._depth_thread.start()

    def stop(self, timeout: float = 1.5):
        self._stop_event.set()
        for t in (self._thread, self._depth_thread):
            if t.is_alive():
                t.join(timeout=timeout)

    def set_interval(self, interval_s: float):
        with self._lock:
            self.poll_interval_s = max(0.25, float(interval_s))

    def set_feed_mode(self, mode: str):
        if mode in ("Auto", "Live only", "Simulated"):
            with self._lock:
                self.feed_mode = mode

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _publish(self, price: float, source: str, kind: str):
        now_mono = time.perf_counter()
        with self._lock:
            if kind != self._kind:
                self._kind = kind
                self.epoch += 1
                self.price_usd = None
            if self.price_usd is not None:
                self.prev_price_usd = self.price_usd
                self.delta_usd = price - self.price_usd
            else:
                self.prev_price_usd = price
                self.delta_usd = 0.0

            self.delta_cents = int(round(self.delta_usd * 100.0))
            if self.delta_usd > 0.0:
                self.direction = "UP"
            elif self.delta_usd < 0.0:
                self.direction = "DOWN"
            else:
                self.direction = "FLAT"

            self.price_usd = price
            self.source = source
            self.last_fetch_mono = now_mono
            if kind == "live":
                self.last_error = None
            self.fetch_count += 1

    def _poll_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                interval = self.poll_interval_s
                mode = self.feed_mode
            published = False

            if mode != "Simulated":
                try:
                    price, source = fetch_btc_usdt_price()
                    self._live_fail = 0
                    self._publish(price, source, "live")
                    published = True
                except Exception as exc:
                    self._live_fail += 1
                    with self._lock:
                        self.last_error = str(exc)

            if not published:
                use_sim = mode == "Simulated" or (
                    mode == "Auto"
                    and (self._live_fail >= 2 or (self._live_fail >= 1 and self._kind is None))
                )
                if use_sim:
                    with self._lock:
                        if self._kind == "live" and self.price_usd is not None:
                            self._sim_price = self.price_usd
                    self._sim_price *= 1.0 + self._sim_rng.gauss(0.0, 0.00005)
                    self._publish(self._sim_price, self.SIM_SOURCE, "sim")

            self._stop_event.wait(interval)

    def _depth_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                interval = self.poll_interval_s
                simulated = self._kind == "sim"
            if simulated:
                with self._lock:
                    self.imbalance = self._sim_rng.uniform(-1.0, 1.0)
                    self.depth_source = "SIMULATED"
                self._stop_event.wait(interval)
                continue
            try:
                (bids, asks), label = DEPTH_CHAIN.call()
                imb = book_imbalance(bids, asks)
                with self._lock:
                    self.imbalance = imb
                    self.depth_source = label
                self._depth_fail = 0
                wait = interval
            except Exception:
                self._depth_fail += 1
                with self._lock:
                    self.imbalance = None
                    self.depth_source = ""
                wait = 30.0 if self._depth_fail >= 3 else 2.0
                if self._depth_fail >= 3:
                    self._depth_fail = 0
            self._stop_event.wait(wait)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "price_usd": self.price_usd,
                "prev_price_usd": self.prev_price_usd,
                "delta_usd": self.delta_usd,
                "delta_cents": self.delta_cents,
                "direction": self.direction,
                "last_fetch_mono": self.last_fetch_mono,
                "last_error": self.last_error,
                "fetch_count": self.fetch_count,
                "source": self.source,
                "depth_source": self.depth_source,
                "imbalance": self.imbalance,
                "epoch": self.epoch,
                "simulated": self._kind == "sim",
            }


# ---------------------------------------------------------------------------
# Market feature engine (immediate start, no warmup)
# ---------------------------------------------------------------------------

class SignalEngine:
    """
    Turns raw ticks into three direction bits (1 = UP, 0 = DOWN) immediately on tick 1:
      mom   : sign of the sum of the last MOM_K price changes
      ob    : order-book imbalance sign (falls back to tick-bounce fade if no book)
      trend : price above/below an EMA(EMA_SPAN)
    """

    NAMES = ("mom", "ob", "trend")
    MOM_K = 3
    EMA_SPAN = 20
    OB_DEADBAND = 0.03
    STAT_WINDOW = 200
    MIN_STAT = 1
    FLIP_BELOW = 0.47
    UNFLIP_ABOVE = 0.53
    MIN_TICKS = 0

    def __init__(self):
        self.ticks = 0
        self.last_price: float | None = None
        self.ema: float | None = None
        self.deltas: deque[float] = deque(maxlen=self.MOM_K)
        self.last_nonzero_sign = 0
        self.imbalance: float | None = None
        self.ob_from_book = False
        self.raw = {n: 1 for n in self.NAMES}
        self.flipped = {n: False for n in self.NAMES}
        self.hits = {n: deque(maxlen=self.STAT_WINDOW) for n in self.NAMES}

    @property
    def ready(self) -> bool:
        return True

    def eff(self, name: str) -> int:
        return self.raw[name] ^ int(self.flipped[name])

    def rate(self, name: str) -> float | None:
        h = self.hits[name]
        return (sum(h) / len(h)) if h else None

    def score(self, observed_up: int):
        if self.ticks == 0:
            return
        for n in self.NAMES:
            h = self.hits[n]
            h.append(int(self.raw[n] == observed_up))
            if len(h) >= self.MIN_STAT:
                r = sum(h) / len(h)
                if not self.flipped[n] and r < self.FLIP_BELOW:
                    self.flipped[n] = True
                elif self.flipped[n] and r > self.UNFLIP_ABOVE:
                    self.flipped[n] = False

    def update(self, price: float, imbalance: float | None):
        self.ticks += 1
        delta = 0.0 if self.last_price is None else price - self.last_price
        self.last_price = price
        self.deltas.append(delta)

        if delta > 0:
            self.last_nonzero_sign = 1
        elif delta < 0:
            self.last_nonzero_sign = -1

        alpha = 2.0 / (self.EMA_SPAN + 1.0)
        prev_ema = price if self.ema is None else self.ema
        self.ema = prev_ema + alpha * (price - prev_ema)

        s = sum(self.deltas)
        if s > 0:
            self.raw["mom"] = 1
        elif s < 0:
            self.raw["mom"] = 0

        dev = price - self.ema
        if dev > 0.005:
            self.raw["trend"] = 1
        elif dev < -0.005:
            self.raw["trend"] = 0

        self.imbalance = imbalance
        if imbalance is not None:
            self.ob_from_book = True
            if imbalance > self.OB_DEADBAND:
                self.raw["ob"] = 1
            elif imbalance < -self.OB_DEADBAND:
                self.raw["ob"] = 0
        else:
            self.ob_from_book = False
            if self.last_nonzero_sign:
                self.raw["ob"] = 1 if self.last_nonzero_sign < 0 else 0


# ---------------------------------------------------------------------------
# Webcam worker
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bench model (circuit-driven predictive power + full 256-bit SHA-256 surface)
# ---------------------------------------------------------------------------

class PhotonicBenchModel:
    """
    End-to-end photonic + logic discrete-step simulation engine structured
    like the BPP circuit (256-bit random binary queue between OR and 74HC74,
    no-flip probe vs pre-delay AND, SELECT -> Driver A feedback) supporting
    Binance BTC, USB Webcam, and combined "Webcam + Binance" mode with
    predictive output and full 256-bit SHA-256 data surface prediction
    derived directly from the circuit's physical/logic nodes.
    Circuit UP maps to Shift+W; Circuit DOWN maps to Shift+S.
    """

    DT = 0.005
    RC_TAU = 0.050
    SUPPLY = 5.0
    THRESHOLD = 0.7
    AND_DELAY = 0.150
    CLOCK_PERIOD = 0.100
    INPUT_PERIOD = 0.100
    SETTLE_STEPS = 25  # 125 ms (> 5 RC time constants after a tick)
    STAT_WINDOW = 200
    MIN_STAT = 1
    FLIP_BELOW = 0.47
    UNFLIP_ABOVE = 0.53

    def __init__(
        self,
        probe_duration: float = 2.0,
        seed: int = 7,
        mode: str = "Webcam + Binance",
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
        high_conf_only: bool = False,
        sshmf_gate_mode: str = "2-of-3 M(A,B,C)",
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
        self.high_conf_only = bool(high_conf_only)
        self.sshmf_gate_mode = (
            sshmf_gate_mode
            if sshmf_gate_mode in ("Off", "2-of-3 M(A,B,C)", "4-of-6 P(S)")
            else "2-of-3 M(A,B,C)"
        )
        self.sshmf_key_allowed = False
        self.sshmf_injected_count = 0

        self.delay_steps = max(1, int(round(self.AND_DELAY / self.DT)))
        self.clock_steps = max(1, int(round(self.CLOCK_PERIOD / self.DT)))
        self.input_steps = max(1, int(round(self.INPUT_PERIOD / self.DT)))

        self.sha256_surface = SHA256CircuitSurfacePredictor()
        self.sha256_info: dict = self.sha256_surface.summary()

        self.reset()

    def set_queue_length(self, new_length: int):
        new_length = max(1, int(new_length))
        self.d_queue_length = new_length
        while len(self.d_bit_queue) > new_length:
            self.d_bit_queue.popleft()
        while len(self.d_bit_queue) < new_length:
            self.d_bit_queue.appendleft(self.queue_rng.randrange(2))
        self.d_queued = self.d_bit_queue[0]

    def reset(self):
        self.input_rng = random.Random(self.seed)
        self.photonic_rng = random.Random(self.seed + 2)
        self.queue_rng = random.Random(self.seed + 3)

        self.step_index = 0
        self.time = 0.0

        self.x = 0
        self.random_a_bit = 1
        self.cam_a_bit = 0
        self.cam_b_bit = 0
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

        self.last_epoch = 0
        self._reset_market_state()

        self.sha256_surface.reset()
        self.sha256_info = self.sha256_surface.summary()

        self.and_history: list[int] = []
        self.optical_and_history: list[float] = []
        self.records: list[dict] = []

    def _reset_market_state(self):
        self.engine = SignalEngine()
        self.last_fetch_count = 0
        self.steps_since_tick = 0
        self.pending_emit = False
        self.emit_now = False
        self.decision_now = False
        self.signal_label = "READY"

        self.btc_price_usd: float | None = None
        self.btc_delta_usd = 0.0
        self.btc_delta_cents = 0
        self.btc_observed_up = 0
        self.btc_book_imbalance: float | None = None

        self.feat_m = 1
        self.feat_o = 1
        self.feat_t = 1
        self.sshmf = evaluate_sshmf(self.feat_m, self.feat_o, self.feat_t)

        # Circuit-derived prediction state
        self.circuit_raw_up = 1
        self.circuit_inv = False
        self.circuit_hits: deque[int] = deque(maxlen=self.STAT_WINDOW)
        self.circuit_conf = 2
        self.btc_pred_up = 1
        self.btc_pred_label = "UP"
        self.pred_latched: int | None = None
        self.pred_conf_latched = 0

        self.bet_prediction = 0
        self.bet_won = 0
        self.bet_wins_total = 0
        self.bet_trials_total = 0

        self.btc_pred_hits = 0
        self.btc_pred_evaluated = 0
        self.hc_hits = 0
        self.hc_evaluated = 0
        self.persist_hits = 0
        self.persist_evaluated = 0
        self.up_hits = 0

    @property
    def is_live_btc_mode(self) -> bool:
        return self.mode in ("Binance BTC", "Webcam + Binance")

    @property
    def finished(self) -> bool:
        return not self.is_live_btc_mode and self.step_index >= self.total_steps

    def get_bet_wins(self) -> list[dict]:
        return [rec for rec in self.records if rec["bet_won"] == 1]

    def circuit_rate(self) -> float | None:
        if not self.circuit_hits:
            return None
        return sum(self.circuit_hits) / len(self.circuit_hits)

    def _update_sshmf_gate(self):
        if self.sshmf_gate_mode == "Off":
            self.sshmf_key_allowed = True
        elif self.sshmf_gate_mode == "4-of-6 P(S)":
            self.sshmf_key_allowed = bool(self.sshmf.parent_majority)
        else:
            self.sshmf_key_allowed = bool(self.sshmf.surface_active)

    # ----- market tick handling -----

    def _on_btc_tick(self, price: float, imbalance: float | None):
        prev_price = self.engine.last_price
        delta = 0.0 if prev_price is None else price - prev_price
        observed_up = 1 if delta > 0.0 else 0

        if delta != 0.0:
            if self.pred_latched is not None:
                self._score_prediction(observed_up)
            self.engine.score(observed_up)

        self.engine.update(price, imbalance)
        self.feat_m = self.engine.eff("mom")
        self.feat_o = self.engine.eff("ob")
        self.feat_t = self.engine.eff("trend")

        self.btc_price_usd = price
        self.btc_delta_usd = delta
        self.btc_delta_cents = int(round(delta * 100.0))
        self.btc_observed_up = observed_up
        self.btc_book_imbalance = imbalance

        self.steps_since_tick = 0
        self.pending_emit = True
        self.pred_latched = None

    def _score_prediction(self, observed_up: int):
        hit = int(self.pred_latched == observed_up)
        self.btc_pred_evaluated += 1
        self.btc_pred_hits += hit
        if self.pred_conf_latched >= 3:
            self.hc_evaluated += 1
            self.hc_hits += hit

        # Track raw circuit node accuracy and adapt polarity if needed
        raw_hit = int(self.circuit_raw_up == observed_up)
        self.circuit_hits.append(raw_hit)
        if len(self.circuit_hits) >= self.MIN_STAT:
            r = sum(self.circuit_hits) / len(self.circuit_hits)
            if not self.circuit_inv and r < self.FLIP_BELOW:
                self.circuit_inv = True
            elif self.circuit_inv and r > self.UNFLIP_ABOVE:
                self.circuit_inv = False

        self.up_hits += observed_up
        sign = self.engine.last_nonzero_sign
        if sign:
            self.persist_evaluated += 1
            self.persist_hits += int((1 if sign > 0 else 0) == observed_up)

        self.bet_prediction = int(self.pred_latched)
        self.bet_won = hit
        self.bet_trials_total += 1
        self.bet_wins_total += hit

    def _latch_decision(self):
        self.pending_emit = False
        self.decision_now = True
        self.pred_latched = self.btc_pred_up
        self.pred_conf_latched = self.circuit_conf
        if self.circuit_conf >= 3 or not self.high_conf_only:
            # Circuit UP -> Shift+W, Circuit DOWN -> Shift+S
            self.signal_label = "Shift+W" if self.btc_pred_up else "Shift+S"
            self.emit_now = True
        else:
            self.signal_label = "ABSTAIN"

    # ----- inputs -----

    def _update_inputs(self, cam_a: int | None = None, cam_b: int | None = None):
        self.cam_a_bit = int(cam_a) if cam_a is not None else 0
        self.cam_b_bit = int(cam_b) if cam_b is not None else 0

        if self.mode == "Webcam + Binance":
            if cam_a is not None:
                self.driver_a_raw = int(self.feat_m) & int(cam_a)
            else:
                self.driver_a_raw = int(self.feat_m)

            if cam_b is not None:
                self.x = int(self.feat_o) & int(cam_b)
            else:
                self.x = int(self.feat_o)
            return

        if self.mode == "Binance BTC":
            self.x = int(self.feat_o)
            self.driver_a_raw = int(self.feat_m)
            return

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

    # ----- main step -----

    def step(
        self,
        cam_a: int | None = None,
        cam_b: int | None = None,
        brightness_a: float = 0.0,
        brightness_b: float = 0.0,
        threshold_px: float = 130.0,
        btc_snap: dict | None = None,
    ) -> bool:
        if self.finished:
            return False

        self.emit_now = False
        self.decision_now = False

        price_usd = btc_snap.get("price_usd") if btc_snap else None
        imbalance = btc_snap.get("imbalance") if btc_snap else None
        epoch = int(btc_snap.get("epoch", 0)) if btc_snap else 0
        if epoch != self.last_epoch:
            self.last_epoch = epoch
            self._reset_market_state()
        fetch_count = int(btc_snap.get("fetch_count", 0)) if btc_snap else 0
        if price_usd is not None and fetch_count > self.last_fetch_count:
            self.last_fetch_count = fetch_count
            self._on_btc_tick(float(price_usd), imbalance)
        self.steps_since_tick += 1

        self._update_inputs(cam_a=cam_a, cam_b=cam_b)
        self.sshmf = evaluate_sshmf(self.feat_m, self.feat_o, self.feat_t)
        self._update_sshmf_gate()

        # BPP feedback: SELECT (Q) gates Driver A directly
        self.driver_a_select_gate = self.q if self.feed_select_to_driver_a else 1
        self.a = self.driver_a_raw & self.driver_a_select_gate

        # BPP clean photonic front-end: Laser -> Coupler -> PD A & PD B
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

        # Circuit-driven predictive readout from physical/logic nodes
        self.circuit_raw_up = int(
            bool(self.pre_delay_and | (self.q & self.d_queued) | (self.feat_t & self.y))
        )
        eff_circuit_up = self.circuit_raw_up ^ int(self.circuit_inv)
        self.circuit_conf = (
            int(self.a == eff_circuit_up)
            + int(self.b == eff_circuit_up)
            + int(self.d_queued == eff_circuit_up)
            + int(self.feat_t == eff_circuit_up)
        )
        self.btc_pred_up = eff_circuit_up
        self.btc_pred_label = "UP" if self.btc_pred_up else "DOWN"

        # Predict the entire 256-bit SHA-256 data surface via the circuit on every step
        self.sha256_info = self.sha256_surface.predict_and_score(
            step_index=completed,
            time_s=self.time,
            a=self.a,
            b=self.b,
            pre_delay_and=self.pre_delay_and,
            delayed_and=self.delayed_and,
            y_or=self.y,
            d_raw_or=self.d,
            d_queued=self.d_queued,
            q=self.q,
            not_q=self.not_q,
            rc_voltage=self.rc_voltage,
            optical_tx_mw=self.optical_tx_mw,
            sshmf=self.sshmf,
            queue_bits=list(self.d_bit_queue),
            btc_price_usd=self.btc_price_usd,
            btc_delta_usd=self.btc_delta_usd,
        )

        if self.pending_emit and self.steps_since_tick >= self.SETTLE_STEPS:
            self._latch_decision()

        # Compare queue output (D_queued) directly against AND BEFORE delay (pre_delay_and), no flip noise
        probe_active = completed <= self.probe_steps
        match_cell: int | str = ""
        if probe_active:
            matched = int(self.d_queued == self.pre_delay_and)
            self.last_match = matched
            self.probe_count += 1
            self.probe_matches += matched
            self.probe_mismatches += 1 - matched
            self.lag_steps_sum += self.last_lag_steps
            match_cell = matched

        if not self.is_live_btc_mode or self.decision_now:
            price_usd_rec = self.btc_price_usd
            self.records.append({
                "time_s": round(self.time, 6),
                "mode": self.mode,
                "btc_price_usd": round(price_usd_rec, 2) if price_usd_rec is not None else "",
                "btc_delta_usd": round(self.btc_delta_usd, 4),
                "btc_delta_cents": self.btc_delta_cents,
                "btc_observed_up": self.btc_observed_up,
                "book_imbalance": round(self.btc_book_imbalance, 4) if self.btc_book_imbalance is not None else "",
                "feat_mom": self.feat_m,
                "feat_ob": self.feat_o,
                "feat_trend": self.feat_t,
                "sshmf_sig": self.sshmf.signature,
                "sshmf_id_3b": self.sshmf.surface_identity,
                "sshmf_ab": self.sshmf.ab,
                "sshmf_ac": self.sshmf.ac,
                "sshmf_bc": self.sshmf.bc,
                "sshmf_active_2of3": self.sshmf.surface_active,
                "sshmf_set_sum": self.sshmf.expanded_sum,
                "sshmf_parent_4of6": self.sshmf.parent_majority,
                "sshmf_id_6b": self.sshmf.expanded_identity,
                "sshmf_gate_mode": self.sshmf_gate_mode,
                "sshmf_key_allowed": int(self.sshmf_key_allowed),
                "circuit_raw_up": self.circuit_raw_up,
                "circuit_inv": int(self.circuit_inv),
                "btc_pred_up": self.btc_pred_up,
                "btc_pred_dir": self.btc_pred_label,
                "vote_conf": self.circuit_conf,
                "key_emitted": self.signal_label,
                "bet_prediction": self.bet_prediction,
                "bet_won": self.bet_won,
                "sha256_pred_hex": self.sha256_info["pred_hex"],
                "sha256_target_hex": self.sha256_info["target_hex"],
                "sha256_bit_hits": self.sha256_info["bit_hits"],
                "sha256_bit_acc": round(self.sha256_info["bit_acc"], 6),
                "sha256_cum_bit_acc": round(self.sha256_info["cum_bit_acc"], 6),
                "sha256_hamming_dist": self.sha256_info["hamming_dist"],
                "sha256_nibble_hits": self.sha256_info["nibble_hits"],
                "sha256_surface_pred_bit": self.sha256_info["surface_pred_bit"],
                "sha256_surface_hit": self.sha256_info["surface_hit"],
                "cam_a_bit": self.cam_a_bit,
                "cam_b_bit": self.cam_b_bit,
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


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class PhotonicCircuitSimulator(tk.Tk):
    ROI_A = (0.58, 0.25, 0.92, 0.75)  # Right ROI
    ROI_B = (0.08, 0.25, 0.42, 0.75)  # Left ROI

    def __init__(self):
        super().__init__()
        self.title("Photonic & Modal Logic BTC + Full 256-Bit SHA-256 Surface Predictor")
        self.geometry("1380x960")
        self.minsize(960, 620)

        self.after_id: str | None = None
        self.cam_after_id: str | None = None
        self.resize_after_id: str | None = None
        self.running = False
        self.model: PhotonicBenchModel | None = None
        self.key_controller = KeyController() if HAS_PYNPUT else None
        self.btc_poller = BinanceBtcPoller(poll_interval_s=1.0)
        self.last_emitted_key: str = "-"
        self.playback_speed = 1.0
        self._last_tick_wall = 0.0
        self._step_accum = 0.0
        self._last_draw = 0.0

        self.webcam: WebcamWorker | None = None
        self.latest_capture_time: float | None = None
        self.preview_image = None
        self.brightness_a = 0.0
        self.brightness_b = 0.0
        self.detector_a = 0
        self.detector_b = 0

        self.duration_var = tk.StringVar(value="3.0")
        self.speed_var = tk.StringVar(value="1.0")
        self.seed_var = tk.StringVar(value="7")
        self.queue_len_var = tk.StringVar(value="256")
        self.mode_var = tk.StringVar(value="Webcam + Binance")
        self.manual_x_var = tk.BooleanVar(value=False)
        self.ignore_startup_var = tk.BooleanVar(value=False)
        self.feedback_driver_a_var = tk.BooleanVar(value=True)
        self.enable_keys_var = tk.BooleanVar(value=False)
        self.hc_only_var = tk.BooleanVar(value=False)
        self.sshmf_gate_var = tk.StringVar(value="2-of-3 M(A,B,C)")
        self.btc_poll_var = tk.StringVar(value="1.0")
        self.feed_var = tk.StringVar(value="Auto")

        self.cam_index_var = tk.StringVar(value="0")
        self.cam_thresh_var = tk.DoubleVar(value=130.0)

        self.laser_mw_var = tk.StringVar(value="10.0")
        self.er_db_var = tk.StringVar(value="20.0")
        self.split_a_var = tk.StringVar(value="0.5")
        self.loss_db_var = tk.StringVar(value="1.5")
        self.resp_var = tk.StringVar(value="0.85")
        self.tia_gain_var = tk.StringVar(value="2.5")
        self.rin_std_var = tk.StringVar(value="0.01")

        self.status_var = tk.StringVar()
        self.photonic_var = tk.StringVar()
        self.sha256_var = tk.StringVar()
        self.stats_var = tk.StringVar()
        self.cam_status_var = tk.StringVar(
            value="USB Cam: off" if HAS_WEBCAM_LIBS else "USB Cam: cv2/PIL unavailable"
        )

        self._build_ui()
        self.reset_simulation()
        if HAS_REQUESTS:
            self.btc_poller.start()
        else:
            self.after(
                300,
                lambda: messagebox.showwarning(
                    "Missing package",
                    "The 'requests' package is not installed, so no BTC price can be fetched.\n\n"
                    "Run:  pip install requests\nthen restart the program.",
                ),
            )
        if HAS_WEBCAM_LIBS:
            self.after(200, self._auto_start_webcam)
            self._poll_webcam()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _auto_start_webcam(self):
        if self.webcam is None and HAS_WEBCAM_LIBS:
            try:
                idx = int(self.cam_index_var.get())
                if idx >= 0:
                    self.webcam = WebcamWorker(idx)
                    self.webcam.start()
                    self.cam_status_var.set(f"Opening USB cam {idx}...")
            except (ValueError, tk.TclError):
                pass

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
            root,
            text="Logic, Timing & Circuit Predictor Controls (Flip noise = 0.0, Circuit UP -> Shift+W, Circuit DOWN -> Shift+S, Full 256-Bit SHA-256 Surface)",
            padding=7,
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

        ttk.Label(logic_frame, text="BTC Poll (s)").grid(row=0, column=8, padx=(10, 4), sticky="w")
        poll_entry = ttk.Entry(logic_frame, textvariable=self.btc_poll_var, width=6)
        poll_entry.grid(row=0, column=9, padx=4, sticky="w")
        poll_entry.bind("<FocusOut>", self._sync_live_vars)
        poll_entry.bind("<Return>", self._sync_live_vars)

        ttk.Label(logic_frame, text="Feed").grid(row=0, column=10, padx=(10, 4), sticky="w")
        feed_box = ttk.Combobox(
            logic_frame,
            textvariable=self.feed_var,
            values=["Auto", "Live only", "Simulated"],
            state="readonly",
            width=10,
        )
        feed_box.grid(row=0, column=11, padx=4, sticky="w")
        feed_box.bind("<<ComboboxSelected>>", self._sync_live_vars)

        ttk.Label(logic_frame, text="Flip noise").grid(row=0, column=12, padx=(10, 4), sticky="w")
        ttk.Label(logic_frame, text="0.0 (OFF)", font=("Arial", 9, "bold")).grid(
            row=0, column=13, padx=4, sticky="w"
        )

        ttk.Label(logic_frame, text="Input mode").grid(row=1, column=0, padx=4, pady=5, sticky="w")
        mode_box = ttk.Combobox(
            logic_frame,
            textvariable=self.mode_var,
            values=[
                "Webcam + Binance",
                "Binance BTC",
                "USB Webcam",
                "Random",
                "Square wave",
                "Manual",
                "Direct (x -> a, b)",
                "Photonic TIA",
            ],
            state="readonly",
            width=18,
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
        ).grid(row=1, column=6, columnspan=3, padx=6, sticky="w")

        ttk.Checkbutton(
            logic_frame,
            text="Keys only when >=3/4 agree",
            variable=self.hc_only_var,
            command=self._sync_live_vars,
        ).grid(row=1, column=9, columnspan=2, padx=6, sticky="w")

        ttk.Label(logic_frame, text="SSMHMF").grid(row=1, column=11, padx=(4, 2), sticky="e")
        sshmf_box = ttk.Combobox(
            logic_frame,
            textvariable=self.sshmf_gate_var,
            values=["2-of-3 M(A,B,C)", "4-of-6 P(S)", "Off"],
            state="readonly",
            width=14,
        )
        sshmf_box.grid(row=1, column=12, padx=2, sticky="w")
        sshmf_box.bind("<<ComboboxSelected>>", self._sync_live_vars)

        ttk.Checkbutton(
            logic_frame,
            text="Inject Shift+W/S on SSMHMF",
            variable=self.enable_keys_var,
            state="normal" if HAS_PYNPUT else "disabled",
        ).grid(row=1, column=13, padx=4, sticky="w")

        cam_frame = ttk.LabelFrame(
            logic_frame,
            text="USB Webcam ROI Input (Left = B gates x/book, Right = A gates Driver A/mom)",
            padding=5,
        )
        cam_frame.grid(row=2, column=0, columnspan=14, sticky="ew", pady=(4, 0))

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

        btns = ttk.Frame(root, padding=(10, 2))
        btns.pack(fill="x")

        self.run_btn = ttk.Button(btns, text="Run", command=self.toggle_running)
        self.run_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Step 5 ms", command=self.single_step).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset", command=self.reset_simulation).pack(side="left", padx=4)
        ttk.Button(btns, text="Run to end", command=self.run_to_end).pack(side="left", padx=4)
        ttk.Button(btns, text="Export CSV", command=self.export_csv).pack(side="left", padx=4)
        ttk.Button(btns, text="Dump Bet Wins", command=self.dump_bet_wins_gui).pack(side="left", padx=4)
        ttk.Button(btns, text="SHA-256 Surface", command=self.inspect_sha256_surface_gui).pack(side="left", padx=4)

        ttk.Label(root, textvariable=self.status_var, padding=(12, 2)).pack(fill="x")
        ttk.Label(root, textvariable=self.photonic_var, padding=(12, 2)).pack(fill="x")
        ttk.Label(
            root,
            textvariable=self.sha256_var,
            padding=(12, 2),
            font=("Consolas", 9, "bold"),
            wraplength=1340,
        ).pack(fill="x")

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

        ttk.Label(root, textvariable=self.stats_var, padding=(12, 2), wraplength=1300).pack(fill="x")

        self.waveform = tk.Canvas(
            root, background="#f7fafc", height=180, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        self.waveform.pack(fill="x", padx=10, pady=(0, 10))
        self.waveform.bind("<Configure>", self._schedule_redraw)

    # ---------------------------------------------------------
    # Full 256-Bit SHA-256 Surface Inspector
    # ---------------------------------------------------------

    def inspect_sha256_surface_gui(self):
        if self.model is None:
            return
        surf = self.model.sha256_surface
        info = self.model.sha256_info

        modal = tk.Toplevel(self)
        modal.title("Full 256-Bit SHA-256 Data Surface Predicted Via Circuit")
        modal.geometry("1020x620")

        header_frame = ttk.Frame(modal, padding=8)
        header_frame.pack(fill="x")

        ttk.Label(
            header_frame,
            text=(
                f"Step: {surf.total_steps_evaluated} | "
                f"Bit Match: {info['bit_hits']}/256 ({info['bit_acc']:.2%}) | "
                f"Cum Bit Acc: {info['cum_bit_acc']:.2%} | "
                f"Nibble Hits: {info['nibble_hits']}/64 | "
                f"Hamming Dist: {info['hamming_dist']} | "
                f"Cum Surface Acc: {info['cum_surface_acc']:.2%}"
            ),
            font=("Consolas", 9, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            header_frame,
            text=f"Circuit Pred Hex (256-bit): {info['pred_hex']}",
            font=("Consolas", 9),
        ).pack(anchor="w", pady=(2, 0))

        ttk.Label(
            header_frame,
            text=f"SHA-256 Target Hex (256-bit): {info['target_hex']}",
            font=("Consolas", 9),
        ).pack(anchor="w")

        txt_frame = ttk.Frame(modal)
        txt_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        sy = ttk.Scrollbar(txt_frame, orient="vertical")
        box = tk.Text(
            txt_frame,
            wrap="none",
            font=("Consolas", 9),
            yscrollcommand=sy.set,
        )
        sy.config(command=box.yview)
        sy.pack(side="right", fill="y")
        box.pack(side="left", fill="both", expand=True)

        box.insert("end", "32-Bit Word / Byte Rows Across Full 256-Bit SHA-256 Surface (Bits 0..255):\n")
        box.insert("end", "-" * 116 + "\n")
        for word_idx in range(8):
            b_start = word_idx * 32
            b_end = b_start + 32
            p_str = "".join(str(x) for x in surf.pred_bits[b_start:b_end])
            t_str = "".join(str(x) for x in surf.target_bits[b_start:b_end])
            m_str = "".join("." if x else "x" for x in surf.bit_matches[b_start:b_end])
            w_hits = sum(surf.bit_matches[b_start:b_end])
            box.insert(
                "end",
                f"W{word_idx} [{b_start:3d}..{b_end-1:3d}] ({w_hits:2d}/32)\n"
                f"  PRED : {p_str}\n"
                f"  TGT  : {t_str}\n"
                f"  MATCH: {m_str}\n\n",
            )
        box.configure(state="disabled")

    # ---------------------------------------------------------
    # Bet Wins Data Dump
    # ---------------------------------------------------------

    def dump_bet_wins_gui(self):
        if self.model is None or not self.model.records:
            messagebox.showinfo("No data", "Run the simulation or step first.")
            return

        wins = self.model.get_bet_wins()
        total_steps = len(self.model.records)
        win_rate = (len(wins) / total_steps) if total_steps else 0.0

        win_modal = tk.Toplevel(self)
        win_modal.title(f"Bet Wins Data Dump ({len(wins)} / {total_steps} rows)")
        win_modal.geometry("980x500")

        toolbar = ttk.Frame(win_modal, padding=6)
        toolbar.pack(fill="x")

        def save_wins_csv():
            if not wins:
                messagebox.showinfo("No wins", "No bet win records to export.", parent=win_modal)
                return
            path = filedialog.asksaveasfilename(
                parent=win_modal,
                title="Save Bet Wins CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
            )
            if not path:
                return
            try:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(wins[0].keys()))
                    writer.writeheader()
                    writer.writerows(wins)
            except OSError as exc:
                messagebox.showerror("Export failed", str(exc), parent=win_modal)
                return
            messagebox.showinfo("Saved", f"Exported {len(wins)} bet win records to CSV.", parent=win_modal)

        ttk.Label(
            toolbar,
            text=f"Wins: {len(wins)} | Rows: {total_steps} | Win Rate: {win_rate:.2%}",
            font=("Arial", 10, "bold"),
        ).pack(side="left")
        ttk.Button(toolbar, text="Save Wins CSV", command=save_wins_csv).pack(side="right")

        text_frame = ttk.Frame(win_modal)
        text_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        scroll_y = ttk.Scrollbar(text_frame, orient="vertical")
        scroll_x = ttk.Scrollbar(text_frame, orient="horizontal")
        box = tk.Text(
            text_frame,
            wrap="none",
            font=("Consolas", 9),
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
        )
        scroll_y.config(command=box.yview)
        scroll_x.config(command=box.xview)

        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        box.pack(side="left", fill="both", expand=True)

        header = (
            f"{'Time(s)':>9} | {'BTC_USD':>10} | {'dUSD':>8} | {'Pred':<4} | {'Conf':>4} | "
            f"{'Key':<8} | {'256bHit':>7} | {'CamAB':>5} | {'preAND':>6} | {'Dq':>2} | {'Q':>2}\n"
        )
        box.insert("end", header)
        box.insert("end", "-" * len(header) + "\n")

        for r in wins:
            btc_str = f"{r['btc_price_usd']:.2f}" if isinstance(r["btc_price_usd"], (int, float)) else "-"
            cam_ab_str = f"{r['cam_a_bit']}/{r['cam_b_bit']}"
            line = (
                f"{r['time_s']:9.3f} | "
                f"{btc_str:>10} | "
                f"{r['btc_delta_usd']:+8.2f} | "
                f"{r['btc_pred_dir']:<4} | "
                f"{r['vote_conf']:4d} | "
                f"{r['key_emitted']:<8} | "
                f"{r['sha256_bit_hits']:>4d}/256 | "
                f"{cam_ab_str:>5} | "
                f"{r['pre_delay_and']:6d} | "
                f"{r['D_queued']:2d} | "
                f"{r['Q_SELECT']:2d}\n"
            )
            box.insert("end", line)

        box.configure(state="disabled")

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
        btc_poll_s = float(self.btc_poll_var.get())

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
        if not math.isfinite(btc_poll_s) or not (0.25 <= btc_poll_s <= 60.0):
            raise ValueError("BTC poll interval must be in [0.25, 60.0] s.")

        return (
            duration, speed, seed, q_len, laser_mw, er_db,
            split_a, loss_db, resp, tia_gain, rin_std, btc_poll_s
        )

    def _sync_live_vars(self, _event=None, redraw: bool = True):
        if self.model is not None:
            self.model.mode = self.mode_var.get()
            self.model.manual_x = bool(self.manual_x_var.get())
            self.model.ignore_startup = bool(self.ignore_startup_var.get())
            self.model.feed_select_to_driver_a = bool(self.feedback_driver_a_var.get())
            self.model.high_conf_only = bool(self.hc_only_var.get())
            self.model.sshmf_gate_mode = self.sshmf_gate_var.get()
            self.model._update_sshmf_gate()
            try:
                q_len = int(self.queue_len_var.get())
                if 1 <= q_len <= 4096 and q_len != self.model.d_queue_length:
                    self.model.set_queue_length(q_len)
            except (ValueError, tk.TclError):
                pass
        self.btc_poller.set_feed_mode(self.feed_var.get())
        try:
            poll_s = float(self.btc_poll_var.get())
            if math.isfinite(poll_s) and 0.25 <= poll_s <= 60.0:
                self.btc_poller.set_interval(poll_s)
        except (ValueError, tk.TclError):
            pass
        if redraw:
            self.redraw()

    def _maybe_emit_key(self):
        m = self.model
        if m is None or not m.decision_now:
            return
        self.last_emitted_key = m.signal_label
        if not m.emit_now or not m.sshmf_key_allowed:
            return
        if m.signal_label not in ("Shift+W", "Shift+S"):
            return
        if not self.enable_keys_var.get() or self.key_controller is None:
            return
        key_char = "w" if m.signal_label == "Shift+W" else "s"
        try:
            with self.key_controller.pressed(Key.shift):
                self.key_controller.press(key_char)
                self.key_controller.release(key_char)
            m.sshmf_injected_count += 1
        except Exception:
            pass

    def _step_once(self, sync: bool = True) -> bool:
        assert self.model is not None
        if sync:
            self._sync_live_vars(redraw=False)
        fresh = self._webcam_fresh()
        btc_snap = self.btc_poller.snapshot() if HAS_REQUESTS else None
        ok = self.model.step(
            cam_a=self.detector_a if fresh else None,
            cam_b=self.detector_b if fresh else None,
            brightness_a=self.brightness_a,
            brightness_b=self.brightness_b,
            threshold_px=float(self.cam_thresh_var.get()),
            btc_snap=btc_snap,
        )
        if ok:
            self._maybe_emit_key()
        return ok

    def pause(self):
        self.running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.run_btn.configure(text="Run")

    def reset_simulation(self):
        self.pause()
        try:
            (
                duration, speed, seed, q_len, laser_mw, er_db,
                split_a, loss_db, resp, tia_gain, rin_std, btc_poll_s
            ) = self._parse_config()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.btc_poller.set_interval(btc_poll_s)
        self.playback_speed = speed
        self.last_emitted_key = "-"
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
            high_conf_only=self.hc_only_var.get(),
            sshmf_gate_mode=self.sshmf_gate_var.get(),
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
        self._last_tick_wall = time.perf_counter()
        self._step_accum = 0.0
        self._tick()

    def _tick(self):
        self.after_id = None
        if not self.running or self.model is None:
            return
        try:
            self.playback_speed = float(self.speed_var.get())
        except (ValueError, tk.TclError):
            pass

        now = time.perf_counter()
        elapsed = min(0.25, max(0.0, now - self._last_tick_wall))
        self._last_tick_wall = now
        self._step_accum += elapsed * max(0.01, self.playback_speed) / self.model.DT
        n = min(400, int(self._step_accum))
        self._step_accum -= n

        for i in range(n):
            if not self._step_once(sync=(i == 0)):
                break
            if self.model.finished:
                break

        if now - self._last_draw >= 0.033 or self.model.finished:
            self.redraw()
            self._last_draw = now

        if self.model.finished:
            self.pause()
            self.redraw()
            return

        self.after_id = self.after(5, self._tick)

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
        if self.model.is_live_btc_mode:
            messagebox.showinfo(
                "Live mode",
                "Live BTC mode has no end, and price ticks only arrive in real time. Use Run instead.",
            )
            return
        while not self.model.finished:
            self._step_once(sync=False)
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

        btc_str = f"${m.btc_price_usd:,.2f}" if m.btc_price_usd is not None else "polling..."
        s_info = m.sha256_info
        self._section(10, 10, 1140, 125, f"PHOTONIC BENCH — mode: {m.mode} | BTC={btc_str} | Cam={m.cam_a_bit}/{m.cam_b_bit}")
        self._section(10, 145, 1140, 115, "PARALLEL DRIVERS (SELECT -> Driver A) / TIA + RC THRESHOLD")
        self._section(10, 270, 1140, 200, "MAIN CIRCUIT — pre-delay AND -> 150 ms delay + 256-Bit Random Queue -> 74HC74 -> CIRCUIT PRED + SHA-256 SURFACE")
        self._section(10, 480, 1140, 100, "NO-FLIP PROBE — compares 256-bit queue out (D_q) vs. AND BEFORE 150 ms delay")

        eng = m.engine
        imb_txt = f"{eng.imbalance:+.2f}" if eng.imbalance is not None else "n/a"
        ob_src = "book" if eng.ob_from_book else "bounce"
        self._box(
            560, 35, 260, 45,
            f"BTC m/o/t = {m.feat_m}/{m.feat_o}/{m.feat_t} ({ob_src} {imb_txt})\n"
            f"dUSD={m.btc_delta_usd:+.2f} | obs={'UP' if m.btc_observed_up else 'DN'}",
            m.btc_observed_up,
        )
        self._box(880, 35, 220, 45, f"Laser ({m.laser_power_mw:g} mW)\nTx = {m.optical_tx_mw:.2f} mW", m.x)
        self._box(310, 35, 210, 45, f"Coupler / x={m.x}\nphi={m.optical_phase_rad:.2f} rad", m.x)
        self._wire([880, 57, 820, 57], m.x)
        self._wire([560, 57, 520, 57], m.x)

        self._box(
            40, 82, 220, 42,
            f"PD A / Cam A ({self.brightness_a:.0f}->{m.cam_a_bit})\n{m.optical_a_mw:.2f} mW | {m.photocurrent_a_ua:.1f} uA",
            m.driver_a_raw,
        )
        self._box(
            310, 82, 220, 42,
            f"PD B / Cam B ({self.brightness_b:.0f}->{m.cam_b_bit})\n{m.optical_b_mw:.2f} mW | {m.photocurrent_b_ua:.1f} uA",
            m.x,
        )
        self._wire([420, 80, 420, 82], m.x)
        self._wire([310, 57, 150, 57, 150, 82], m.x)

        fb_str = f"Q={m.q}" if m.feed_select_to_driver_a else "OFF"
        self._box(
            40, 182, 225, 56,
            f"Driver A (SEL fb: {fb_str})\nraw={m.driver_a_raw}, sel_gate={m.driver_a_select_gate} -> a={m.a}",
            m.a,
        )
        self._box(295, 182, 190, 56, f"Driver B / x\nx = {m.x}", m.x)
        self._box(515, 182, 210, 56, f"RC (tau={m.RC_TAU*1000:.0f}ms)\nVc = {m.rc_voltage:.3f} V")
        self._box(760, 182, 260, 56, f"Comparator > {m.THRESHOLD} V\nb = {m.b}", m.b)

        self._wire([150, 124, 150, 182], m.driver_a_raw)
        self._wire([420, 124, 420, 182], m.x)
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

        # Circuit-driven prediction + full 256-bit SHA-256 surface box
        inv_tag = "INV" if m.circuit_inv else "norm"
        self._box(
            655, 294, 355, 68,
            f"CIRCUIT = preAND | (Q&Dq) | (t&Y) -> raw={m.circuit_raw_up} ({inv_tag})\n"
            f"pred={m.btc_pred_label} conf={m.circuit_conf}/4 [{m.signal_label}] (last={self.last_emitted_key})\n"
            f"SHA256 256b: {s_info['bit_hits']}/256 ({s_info['bit_acc']:.1%}) | {s_info['pred_hex'][:12]}..",
            m.btc_pred_up,
        )

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

        if m.is_live_btc_mode:
            signals = [
                ("circ UP (W)", "btc_pred_up", "#1976a3"),
                ("256b surf", "sha256_surface_pred_bit", "#2c7a7b"),
                ("D (OR out)", "D_raw_or", "#b36a20"),
                ("D_queued", "D_queued", "#7753a6"),
                ("Q / SELECT", "Q_SELECT", "#bd4258"),
            ]
            visible = m.records[-150:]
            n = len(visible)
            x_of = lambda i, rec: left + plot_w * (i / max(1, n - 1))
        else:
            signals = [
                ("pre_del AND", "pre_delay_and", "#1976a3"),
                ("256b surf", "sha256_surface_pred_bit", "#2c7a7b"),
                ("D (OR out)", "D_raw_or", "#b36a20"),
                ("D_queued", "D_queued", "#7753a6"),
                ("Q / SELECT", "Q_SELECT", "#bd4258"),
            ]
            stride = max(1, len(m.records) // 1800)
            visible = m.records[::stride]
            if m.records and (not visible or visible[-1] is not m.records[-1]):
                visible.append(m.records[-1])
            total_dur = max(m.DT, m.total_duration)
            x_of = lambda i, rec: left + plot_w * (rec["time_s"] / total_dur)

        for idx, (label, key, color) in enumerate(signals):
            base_y = 24 + idx * 28
            p.create_text(8, base_y, text=label, anchor="w", fill="#34495e")
            p.create_line(left, base_y + 7, right, base_y + 7, fill="#dce4eb")

            points = [left, base_y + 7]
            prev_y = base_y + 7
            for i, rec in enumerate(visible):
                x = x_of(i, rec)
                y = base_y + 7 - 14 * rec[key]
                points.extend([x, prev_y, x, y])
                prev_y = y

            if len(points) >= 4:
                p.create_line(*points, fill=color, width=2)

        if m.is_live_btc_mode:
            p.create_text(left, 167, text=f"last {len(visible)} circuit decisions (UP->Shift+W, DOWN->Shift+S, 256-bit SHA-256 surface)", anchor="w")
        else:
            total_dur = max(m.DT, m.total_duration)
            probe_x = left + plot_w * (m.probe_duration / total_dur)
            p.create_line(probe_x, 8, probe_x, 152, fill="#888888", dash=(4, 3))
            p.create_text(left, 167, text="0 s", anchor="w")
            p.create_text(probe_x, 167, text=f"Probe stops: {m.probe_duration:g} s")
            p.create_text(right, 167, text=f"{m.total_duration:g} s", anchor="e")

    def _update_status(self):
        m = self.model
        assert m is not None
        snap = self.btc_poller.snapshot() if HAS_REQUESTS else {}

        if m.is_live_btc_mode:
            state = "Running (live)" if self.running else "Paused (live)"
            time_str = f"Time: {m.time:.1f} s"
        else:
            state = "Finished" if m.finished else ("Running" if self.running else "Paused")
            time_str = f"Time: {m.time:.3f} / {m.total_duration:g} s"

        fb_str = f"ON ({m.driver_a_select_gate})" if m.feed_select_to_driver_a else "OFF"
        btc_price = snap.get("price_usd")
        btc_err = snap.get("last_error")
        src = snap.get("source") or "?"
        dsrc = snap.get("depth_source") or "no book"
        if btc_price is not None and snap.get("simulated"):
            btc_info = f"SIMULATED ${btc_price:,.2f}"
        elif btc_price is not None:
            btc_info = f"BTC=${btc_price:,.2f} [{src} | book: {dsrc}]"
        elif not HAS_REQUESTS:
            btc_info = "BTC: 'requests' NOT INSTALLED"
        elif btc_err:
            btc_info = f"BTC err: {btc_err}"
        else:
            btc_info = "BTC: connecting..." if self.btc_poller.running else "BTC: poller not running"

        cam_fresh_str = "LIVE" if self._webcam_fresh() else "STALE/OFF"
        self.status_var.set(
            f"{state}  |  {time_str}  |  Mode: {m.mode}  |  Cam({cam_fresh_str})={m.cam_a_bit}/{m.cam_b_bit}  |  "
            f"SELECT->DriverA: {fb_str}  |  {btc_info}  |  Circuit={m.btc_pred_label} ({m.circuit_conf}/4) [{m.signal_label}]"
        )
        self.photonic_var.set(
            f"Tx: {m.optical_tx_mw:.2f} mW ({mw_to_dbm(m.optical_tx_mw):.1f} dBm)  |  "
            f"PD-A: {m.photocurrent_a_ua:.1f} uA (CamA={self.brightness_a:.0f}->{m.cam_a_bit})  |  "
            f"PD-B: {m.photocurrent_b_ua:.1f} uA (CamB={self.brightness_b:.0f}->{m.cam_b_bit})"
        )

        s_info = m.sha256_info
        self.sha256_var.set(
            f"SHA-256 256-Bit Surface (Circuit-Predicted): "
            f"Bits={s_info['bit_hits']}/256 ({s_info['bit_acc']:.1%}) | CumBitAcc={s_info['cum_bit_acc']:.1%} | "
            f"Nibbles={s_info['nibble_hits']}/64 | Hamming={s_info['hamming_dist']} | "
            f"SurfAcc={s_info['cum_surface_acc']:.1%} | PRED={s_info['pred_hex'][:24]}.. | TGT={s_info['target_hex'][:24]}.."
        )

        match_pct = (m.probe_matches / m.probe_count) if m.probe_count else 0.0
        avg_lag_ms = (
            (m.lag_steps_sum / m.probe_count) * m.DT * 1000.0
            if m.probe_count else 0.0
        )
        n_all, h_all = m.btc_pred_evaluated, m.btc_pred_hits
        n_hc, h_hc = m.hc_evaluated, m.hc_hits
        c_rate = m.circuit_rate()
        c_rate_str = f"{c_rate:.1%}" if c_rate is not None else "--"
        c_pol = "INV" if m.circuit_inv else "norm"
        self.stats_var.set(
            f"pre_delay_AND={m.pre_delay_and}, D_queued={m.d_queued}, del_AND={m.delayed_and}, Q={m.q}  |  "
            f"Queue: {len(m.d_bit_queue)}/{m.d_queue_length} (avg lag {avg_lag_ms:.1f} ms)  |  "
            f"Queue==PreDelayAND: {m.probe_matches}/{m.probe_count} ({match_pct:.1%})  |  "
            f"Circuit ALL: {h_all}/{n_all} ({pct(h_all, n_all):.1%}, z={z_score(h_all, n_all):+.2f})  |  "
            f">=3/4-agree: {h_hc}/{n_hc} ({pct(h_hc, n_hc):.1%})  |  raw={c_rate_str} ({c_pol})  |  "
            f"SSMHMF[{m.sshmf.signature} I_s={m.sshmf.surface_identity} "
            f"M={m.sshmf.surface_active} P={m.sshmf.parent_majority} I_S={m.sshmf.expanded_identity} "
            f"gate={m.sshmf_gate_mode} inj={m.sshmf_injected_count}]"
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
        self.btc_poller.stop()
        self._unbind_mousewheel()
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        if self.cam_after_id is not None:
            self.after_cancel(self.cam_after_id)
        if self.webcam is not None:
            self.webcam.stop()
        self.destroy()


def run_check():
    print(f"requests installed : {HAS_REQUESTS}")
    print(f"pynput installed   : {HAS_PYNPUT}")
    print(f"opencv/PIL         : {HAS_WEBCAM_LIBS}")
    if not HAS_REQUESTS:
        print("\n-> pip install requests")
        return
    for name, chain in (("PRICE", PRICE_CHAIN), ("ORDER BOOK", DEPTH_CHAIN)):
        print(f"\n{name} sources:")
        for label, fn in chain.sources:
            t0 = time.perf_counter()
            try:
                val = fn()
                shown = f"{val:,.2f}" if isinstance(val, float) else f"{len(val[0])} bids / {len(val[1])} asks"
                print(f"  OK    {label:<24} {shown}  ({time.perf_counter() - t0:.2f}s)")
            except Exception as exc:
                print(f"  FAIL  {label:<24} {str(exc)[:90]}  ({time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    if "--check" in sys.argv:
        run_check()
    else:
        app = PhotonicCircuitSimulator()
        app.mainloop()