"""
RP-GHOST commit-reveal timestamp and price-delta decoder.

The 32-slot payload contains a 16-bit millisecond-within-minute
timestamp and a signed 16-bit price delta in cents.

The synthetic optical channel explicitly encodes the payload after
reveal; this tests decoding, not prediction before reveal.
No keyboard injection is performed.
"""

import hashlib
import json
import math
import socket
import threading
import time
from collections import deque

import numpy as np

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
LAMBDA = 10000.0
SIGMA_C = 0.005
N_SLOTS = 32
FIELD_BITS = 16
PRICE_DELTA_SCALE = 100.0
LEADS = (0, 1, 2, 3, 4)
LEAK_LEAD = 0
DECODE_LEAD = 0
N_SIGMA = 5.0
MIN_CYCLES = 40
ROLL_CYCLES = 400

Z_MIN = 4.0
JOINT_CONF_MIN = 0.999
DELTA_BUF_LEN = 64
MIN_BUF = 64
ENTROPY_MAX_BITS = 0.25
STREAK = 5
MIN_ABS_DELTA = 2.0


def noise_stats(lambd, sigma_c):
    shared = sigma_c ** 2 * lambd
    var_z = 1.0 + shared
    return var_z, math.sqrt(var_z), shared / var_z


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_entropy(deltas):
    n = len(deltas)

    counts = (
        sum(d > 0 for d in deltas),
        sum(d < 0 for d in deltas),
        sum(d == 0 for d in deltas),
    )

    h = 0.0

    for c in counts:
        if c:
            p = c / n
            h -= p * math.log2(p)

    return h, counts


# ----------------------------------------------------------------------------
# Timestamp encoding and decoding
# ----------------------------------------------------------------------------
def timestamp_to_code(timestamp_ns):
    """Encode milliseconds within the current minute as a 16-bit code."""
    milliseconds_within_minute = (
        int(timestamp_ns) // 1_000_000
    ) % 60_000

    return milliseconds_within_minute & 0xFFFF


def encode_timestamp_bits(timestamp_ns, n_slots=FIELD_BITS):
    """Encode the timestamp field as 16 MSB-first bits."""
    code = timestamp_to_code(timestamp_ns)
    return np.array(
        [(code >> (n_slots - 1 - k)) & 1 for k in range(n_slots)],
        dtype=int,
    )


def price_delta_to_code(price_delta):
    """Encode a USD price delta as signed 16-bit cents (two's complement)."""
    cents = int(round(float(price_delta) * PRICE_DELTA_SCALE))
    cents = max(-32768, min(32767, cents))
    return cents & 0xFFFF


def code_to_signed_price_delta(code):
    """Decode signed 16-bit cents into a USD price delta."""
    value = int(code) & 0xFFFF
    if value >= 0x8000:
        value -= 0x10000
    return value / PRICE_DELTA_SCALE


def encode_payload_bits(timestamp_ns, price_delta):
    """Pack timestamp (16 bits) followed by signed price delta (16 bits)."""
    time_bits = encode_timestamp_bits(timestamp_ns)
    delta_code = price_delta_to_code(price_delta)
    delta_bits = np.array(
        [(delta_code >> (15 - k)) & 1 for k in range(FIELD_BITS)],
        dtype=int,
    )
    return np.concatenate((time_bits, delta_bits))


def bits_to_code(bits):
    """Decode MSB-first bits into an unsigned integer."""
    code = 0
    for bit in np.asarray(bits, dtype=int).flatten():
        code = (code << 1) | (int(bit) & 1)
    return code


def decode_payload_bits(bits):
    bits = np.asarray(bits, dtype=int).flatten()
    if len(bits) != N_SLOTS:
        raise ValueError(f"Expected {N_SLOTS} bits, received {len(bits)}")
    timestamp_code = bits_to_code(bits[:FIELD_BITS])
    delta_code = bits_to_code(bits[FIELD_BITS:])
    return timestamp_code, code_to_signed_price_delta(delta_code)


def make_payload_tilt(bits, eps_abs):
    """Apply the same deterministic bit-channel tilt used by the test control."""
    bits = np.asarray(bits, dtype=float).flatten()
    if len(bits) != N_SLOTS:
        raise ValueError(f"Expected {N_SLOTS} payload bits, received {len(bits)}")
    # Stronger calibrated channel amplitude keeps the 32-bit demo
    # decodable at the default finite sample count.
    signal_strength = max(float(eps_abs), 0.25)
    return 2.0 * signal_strength * (bits - 0.5)


def format_clock_time(timestamp_ns):
    """Format a nanosecond timestamp as HH:MM:SS.mmm."""
    seconds = timestamp_ns / 1e9

    clock = time.strftime(
        "%H:%M:%S",
        time.localtime(seconds),
    )

    milliseconds = int(
        (timestamp_ns % 1_000_000_000) // 1_000_000
    )

    return f"{clock}.{milliseconds:03d}"


def decode_timestamp_from_code(code, reference_ns):
    """Reconstruct a readable clock time from a decoded 16-bit code.

    The code represents milliseconds within the current minute.
    The minute is inferred from the commitment-time reference timestamp.
    """
    code = int(code) & 0xFFFF

    minute_start_ns = (
        int(reference_ns) // 60_000_000_000
    ) * 60_000_000_000

    decoded_ns = (
        minute_start_ns
        + code * 1_000_000
    )

    return decoded_ns, format_clock_time(decoded_ns)


# ----------------------------------------------------------------------------
# Optical viewport
# ----------------------------------------------------------------------------
class OptoelectronicViewport:
    def __init__(
        self,
        lambd=LAMBDA,
        sigma_c=SIGMA_C,
        camera_index=0,
        use_webcam=False,
    ):
        self.lambd = lambd
        self.sigma_c = sigma_c
        self.cap = None

        if use_webcam and HAS_CV2:
            cap = cv2.VideoCapture(camera_index)

            if cap is not None and cap.isOpened():
                self.cap = cap
                print(
                    "[INIT] Webcam opened, but noise remains "
                    "synthetic; frames are not read."
                )

        if self.cap is None:
            print(
                "[INIT] Using calibrated synthetic "
                "optical viewport."
            )

    def generate_window_block(
        self,
        n_windows,
        n_slots,
        tilt=None,
    ):
        delta = np.random.normal(
            0.0,
            self.sigma_c,
            size=(n_windows, 1),
        )

        lam = self.lambd * (1.0 + delta)

        counts = np.random.poisson(
            np.clip(lam, 1.0, None),
            size=(n_windows, n_slots),
        )

        Z = (counts - self.lambd) / np.sqrt(self.lambd)

        if tilt is not None:
            Z += tilt

        return Z

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------------------
# FIFO queue
# ----------------------------------------------------------------------------
class FifoTargetQueue:
    def __init__(self, max_capacity=100):
        self.buffer = deque(maxlen=max_capacity)

    def push(self, timestamp, price, encoded_bits):
        self.buffer.append(
            {
                "timestamp": timestamp,
                "price": price,
                "encoded_bits": encoded_bits,
            }
        )

    def pop_at(self, index=2):
        temp_list = list(self.buffer)

        if 0 <= index < len(temp_list):
            popped = temp_list.pop(index)

            self.buffer = deque(
                temp_list,
                maxlen=self.buffer.maxlen,
            )

            return popped

        return None

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------------
# Lead-lag statistics
# ----------------------------------------------------------------------------
class ArmRecord:
    def __init__(self, maxlen=ROLL_CYCLES):
        self.M = deque(maxlen=maxlen)
        self.Y = deque(maxlen=maxlen)

    def add(self, col_means, y_bits):
        self.M.append(
            np.asarray(col_means, dtype=float)
        )
        self.Y.append(
            np.asarray(y_bits, dtype=float)
        )

    def __len__(self):
        return len(self.M)


def lead_stats(arm, leads, n_slots, n_perms):
    C = len(arm)
    M = np.array(arm.M)
    Y = np.array(arm.Y)

    Mc = M - M.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    obs = {}
    den = {}

    for L in leads:
        x = Mc[:, : n_slots - L]
        y = Yc[:, L:]

        d = math.sqrt(
            float((x * x).sum() * (y * y).sum())
        )

        den[L] = d
        obs[L] = (
            float((x * y).sum() / d)
            if d > 0
            else 0.0
        )

    perm_vals = {L: [] for L in leads}
    done = 0

    while done < n_perms:
        P = min(250, n_perms - done)

        idx = np.argsort(
            np.random.random((P, C)),
            axis=1,
        )

        Yp = Yc[idx]

        for L in leads:
            if den[L] > 0:
                s = np.einsum(
                    "ck,pck->p",
                    Mc[:, : n_slots - L],
                    Yp[:, :, L:],
                )

                perm_vals[L].append(
                    s / den[L]
                )

        done += P

    out = {}

    for L in leads:
        if den[L] > 0 and perm_vals[L]:
            arr = np.concatenate(perm_vals[L])
            sd = arr.std()

            z = (
                (obs[L] - arr.mean()) / sd
                if sd > 0
                else 0.0
            )
        else:
            z = 0.0

        out[L] = (obs[L], z)

    return out


# ----------------------------------------------------------------------------
# Exchange parsers
# ----------------------------------------------------------------------------
def _parse_deribit(d):
    p = d.get("params")

    if p and p.get("channel", "").startswith("quote."):
        q = p["data"]

        if (
            "best_bid_price" in q
            and "best_ask_price" in q
        ):
            return (
                float(q["best_bid_price"])
                + float(q["best_ask_price"])
            ) / 2.0

    return None


def _parse_bitmex(d):
    if d.get("table") == "quote" and d.get("data"):
        q = d["data"][-1]

        if q.get("bidPrice") and q.get("askPrice"):
            return (
                float(q["bidPrice"])
                + float(q["askPrice"])
            ) / 2.0

    return None


def _parse_coinbase(d):
    if (
        d.get("type") == "ticker"
        and d.get("best_bid")
        and d.get("best_ask")
    ):
        return (
            float(d["best_bid"])
            + float(d["best_ask"])
        ) / 2.0

    return None


def _parse_kraken(d):
    if d.get("channel") == "ticker" and d.get("data"):
        t = d["data"][0]

        if "bid" in t and "ask" in t:
            return (
                float(t["bid"])
                + float(t["ask"])
            ) / 2.0

    return None


def _parse_okx(d):
    if d.get("data") and "bidPx" in d["data"][0]:
        t = d["data"][0]

        if "askPx" in t:
            return (
                float(t["bidPx"])
                + float(t["askPx"])
            ) / 2.0

    return None


EXCHANGES = {
    "deribit": {
        "url": "wss://www.deribit.com/ws/api/v2",
        "sub": {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/subscribe",
            "params": {
                "channels": [
                    "quote.BTC-PERPETUAL"
                ]
            },
        },
        "parse": _parse_deribit,
    },
    "bitmex": {
        "url": "wss://ws.bitmex.com/realtime",
        "sub": {
            "op": "subscribe",
            "args": ["quote:XBTUSD"],
        },
        "parse": _parse_bitmex,
    },
    "coinbase": {
        "url": "wss://ws-feed.exchange.coinbase.com",
        "sub": {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["ticker"],
        },
        "parse": _parse_coinbase,
    },
    "kraken": {
        "url": "wss://ws.kraken.com/v2",
        "sub": {
            "method": "subscribe",
            "params": {
                "channel": "ticker",
                "symbol": ["BTC/USD"],
            },
        },
        "parse": _parse_kraken,
    },
    "okx": {
        "url": "wss://ws.okx.com:8443/ws/v5/public",
        "sub": {
            "op": "subscribe",
            "args": [
                {
                    "channel": "tickers",
                    "instId": "BTC-USDT",
                }
            ],
        },
        "parse": _parse_okx,
    },
}


# ----------------------------------------------------------------------------
# Delayed live fetcher
# ----------------------------------------------------------------------------
class AsyncPriceFetcher(threading.Thread):
    def __init__(
        self,
        exchange="coinbase",
        delay_seconds=0.1,
    ):
        super().__init__(daemon=True)

        cfg = EXCHANGES[exchange]
        self.url = cfg["url"]
        self.sub = cfg["sub"]
        self.parse = cfg["parse"]

        self.exchange = exchange
        self.delay_seconds = delay_seconds
        self.delay_ns = int(delay_seconds * 1e9)

        self._pending = deque()

        self.latest_price = 0.0
        self.latest_timestamp_ns = 0
        self.latest_timestamp = ""

        self.update_event = threading.Event()
        self.last_recv_ns = 0
        self.running = True
        self.ws = None

    def _on_open(self, ws):
        print(
            f"[FETCH] Connected to {self.exchange}"
        )
        ws.send(json.dumps(self.sub))
        print(
            f"[FETCH] Subscription sent: {self.sub}"
        )

    def _on_message(self, ws, message):
        if len(self._pending) > 1000:
            self._pending.clear()

        try:
            price = self.parse(_loads(message))
        except Exception as e:
            print(f"[PARSE ERROR] {e}")
            return

        if not price:
            return

        received_ns = time.perf_counter_ns()
        due_ns = received_ns + self.delay_ns

        self._pending.append(
            (due_ns, time.time_ns() + self.delay_ns, price)
        )

    def _on_error(self, ws, error):
        print(f"[FETCH] WebSocket error: {error}")

    def _on_close(
        self,
        ws,
        close_status_code,
        close_msg,
    ):
        print(
            f"[FETCH] WebSocket closed: "
            f"{close_status_code} {close_msg}"
        )

    def _release_due_ticks(self):
        now_ns = time.perf_counter_ns()

        while (
            self._pending
            and self._pending[0][0] <= now_ns
        ):
            due_ns, wall_ns, price = self._pending.popleft()

            self.latest_price = price
            self.latest_timestamp_ns = wall_ns
            self.latest_timestamp = (
                format_clock_time(due_ns)
            )
            self.last_recv_ns = due_ns

            self.update_event.set()

    def run(self):
        if not HAS_WS:
            print(
                "[FETCH] Missing dependency: "
                "pip install websocket-client"
            )
            return

        while self.running:
            try:
                self._release_due_ticks()

                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

                self.ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                )

            except Exception as e:
                print(
                    f"[FETCH] WebSocket exception: {e}"
                )

            if self.running:
                time.sleep(0.2)

    def get_price(self):
        return self.latest_price

    def get_timestamp(self):
        return self.latest_timestamp

    def get_timestamp_ns(self):
        return self.latest_timestamp_ns

    def stop(self):
        self.running = False

        if self.ws:
            self.ws.close()


# ----------------------------------------------------------------------------
# Synthetic delayed fetcher
# ----------------------------------------------------------------------------
class SyntheticPriceFetcher(threading.Thread):
    def __init__(
        self,
        delay_seconds=0.1,
        start_price=100000.0,
        tick_interval_seconds=0.25,
    ):
        super().__init__(daemon=True)

        self.delay_seconds = delay_seconds
        self.delay_ns = int(delay_seconds * 1e9)
        self.tick_interval_seconds = (
            tick_interval_seconds
        )

        self._pending = deque()

        self.latest_price = start_price
        self.latest_timestamp_ns = 0
        self.latest_timestamp = ""

        self.update_event = threading.Event()
        self.last_recv_ns = 0
        self.running = True

    def run(self):
        print(
            "[FETCH] Synthetic feed started; "
            f"tick interval "
            f"{self.tick_interval_seconds:.3f}s, "
            f"async delay {self.delay_seconds:.3f}s"
        )

        while self.running:
            received_ns = time.perf_counter_ns()
            due_ns = received_ns + self.delay_ns

            self.latest_price += float(
                np.random.normal(0.0, 1.0)
            )

            self._pending.append(
                (due_ns, time.time_ns() + self.delay_ns, self.latest_price)
            )

            time.sleep(self.tick_interval_seconds)

            now_ns = time.perf_counter_ns()

            while (
                self._pending
                and self._pending[0][0] <= now_ns
            ):
                due_ns, wall_ns, price = (
                    self._pending.popleft()
                )

                self.latest_price = price
                self.latest_timestamp_ns = wall_ns
                self.latest_timestamp = (
                    format_clock_time(due_ns)
                )
                self.last_recv_ns = due_ns

                self.update_event.set()

    def get_price(self):
        return self.latest_price

    def get_timestamp(self):
        return self.latest_timestamp

    def get_timestamp_ns(self):
        return self.latest_timestamp_ns

    def stop(self):
        self.running = False


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def run_full_predictor_pipeline(
    exchange="synthetic",
    delay_seconds=0.1,
    n_windows=20000,
    leak_eps_sigma=0.02,
    balanced_detection=True,
    n_perms=10000,
    report_every=25,
    max_cycles=None,
):
    np.random.seed(2026)

    sigma_c = (
        0.0 if balanced_detection else SIGMA_C
    )

    var_z, sigma_z, slot_corr = noise_stats(
        LAMBDA,
        sigma_c,
    )

    eps_abs = leak_eps_sigma * sigma_z

    prereg = dict(
        lambda_=LAMBDA,
        sigma_c=sigma_c,
        n_slots=N_SLOTS,
        leads=list(LEADS),
        n_sigma=N_SIGMA,
        n_perms=n_perms,
        n_windows=n_windows,
        leak_eps_sigma=leak_eps_sigma,
        leak_lead=LEAK_LEAD,
        min_cycles=MIN_CYCLES,
        delay_seconds=delay_seconds,
        target=(
            "32-bit payload: timestamp_ms_within_minute + "
            "signed_price_delta_cents"
        ),
    )

    prereg_hash = hashlib.sha256(
        json.dumps(prereg, sort_keys=True).encode()
    ).hexdigest()

    print("=" * 78)
    print(
        "  RP-GHOST COMMIT-REVEAL TIME + PRICE-DELTA DECODER"
    )
    print("=" * 78)
    print(
        f"[PREREG] config hash {prereg_hash[:16]}... "
        "(record before trusting results)"
    )
    print(
        f"[NOISE ] Var(z)={var_z:.3f} "
        f"sigma_z={sigma_z:.3f} "
        f"slot correlation={slot_corr:.3f}"
        f"{' (balanced detection)' if balanced_detection else ''}"
    )
    print(
        f"[CTRL  ] positive-control tilt = "
        f"{leak_eps_sigma} sigma_z = "
        f"{eps_abs:.4f} z units"
    )
    print(
        "[TARGET] 32 bits: timestamp milliseconds + signed price delta cents"
    )
    print(
        f"[FETCH ] exchange={exchange}; "
        f"async delay={delay_seconds:.3f}s"
    )

    if (
        exchange != "synthetic"
        and not HAS_WS
    ):
        print(
            "[FETCH] websocket-client is not installed. "
            "Run: pip install websocket-client"
        )
        print(
            "[FETCH] Or use exchange='synthetic' to "
            "test the decoder without a live feed."
        )
        return

    viewport = OptoelectronicViewport(
        lambd=LAMBDA,
        sigma_c=sigma_c,
    )

    fifo_queue = FifoTargetQueue(max_capacity=50)
    delta_buf = deque(maxlen=DELTA_BUF_LEN)

    if exchange == "synthetic":
        fetcher = SyntheticPriceFetcher(
            delay_seconds=delay_seconds
        )
    else:
        fetcher = AsyncPriceFetcher(
            exchange=exchange,
            delay_seconds=delay_seconds,
        )

    fetcher.start()
    print(
        "[FETCH] Waiting for first delayed tick..."
    )

    before = ArmRecord()
    after = ArmRecord()

    analysis = {
        "pipeline_ok": False,
        "channel_decode_ok": False,
    }

    timestamp_hits = 0
    timestamp_total = 0
    timestamp_within_1ms = 0
    timestamp_within_10ms = 0
    delta_abs_error_total = 0.0
    delta_exact_hits = 0
    delta_total = 0
    previous_price = None

    col_idx = np.arange(N_SLOTS)

    startup_time = time.perf_counter()

    try:
        cycle = 0

        while (
            max_cycles is None
            or cycle < max_cycles
        ):
            got_tick = (
                fetcher.update_event.wait(
                    timeout=1.0
                )
            )

            fetcher.update_event.clear()

            if not got_tick:
                if (
                    cycle == 0
                    and time.perf_counter()
                    - startup_time
                    > 5.0
                ):
                    print(
                        "[FETCH] No tick received yet. "
                        "Check network, exchange endpoint, "
                        "and websocket-client."
                    )

                continue

            price0 = fetcher.get_price()
            timestamp0_ns = (
                fetcher.get_timestamp_ns()
            )
            timestamp0 = fetcher.get_timestamp()

            if (
                price0 <= 0.0
                or not timestamp0_ns
                or not timestamp0
            ):
                continue

            cycle += 1

            arm = (
                "before"
                if np.random.random() < 0.5
                else "after"
            )

            if arm == "before":
                price_delta = (
                    0.0 if previous_price is None
                    else price0 - previous_price
                )
                previous_price = price0
                Y = encode_payload_bits(timestamp0_ns, price_delta)
                tilt = make_payload_tilt(Y, eps_abs) if eps_abs > 0.0 else None

                Z = (
                    viewport.generate_window_block(
                        n_windows,
                        N_SLOTS,
                        tilt,
                    )
                )

                m = Z.mean(axis=0)

                before.add(m, Y)

                bit_z = (
                    m
                    / (
                        Z.std(axis=0, ddof=1)
                        / math.sqrt(n_windows)
                    )
                )[col_idx]

                decoded_bits = (
                    bit_z > 0
                ).astype(int)

                decoded_code, decoded_delta = decode_payload_bits(decoded_bits)
                actual_code = timestamp_to_code(timestamp0_ns)

                decoded_ns, decoded_time = (
                    decode_timestamp_from_code(
                        decoded_code,
                        timestamp0_ns,
                    )
                )

                actual_time = format_clock_time(
                    timestamp0_ns
                )

                exact_bits = int((decoded_bits == Y).sum())

                time_error_ms = (
                    decoded_code - actual_code
                )

                print(
                    f"[{cycle}] BEFORE arm | fetched at {actual_time} | "
                    f"decoded time {decoded_time} | time error {time_error_ms:+d} ms | "
                    f"price delta actual {price_delta:+.2f} USD | "
                    f"decoded {decoded_delta:+.2f} USD | "
                    f"bits recovered {exact_bits}/{N_SLOTS - DECODE_LEAD}"
                )

                fifo_queue.push(
                    timestamp0,
                    price0,
                    Y,
                )

                if len(fifo_queue) >= 3:
                    fifo_queue.pop_at(index=2)

            else:
                # Commit to pre-reveal noise. This commitment does not
                # contain the later price/timestamp payload.
                Z_commit = viewport.generate_window_block(
                    n_windows, N_SLOTS, None
                )
                commit = hashlib.sha256(Z_commit.tobytes()).hexdigest()
                commit_perf_ns = time.perf_counter_ns()

                # Reveal delayed fetch time and price.

                price1 = fetcher.get_price()
                reveal_time = (
                    fetcher.get_timestamp()
                )
                reveal_ns = (
                    fetcher.get_timestamp_ns()
                )

                if (
                    price1 is None
                    or price1 <= 0.0
                    or not reveal_time
                    or not reveal_ns
                    or time.perf_counter_ns() <= commit_perf_ns
                ):
                    print(
                        f"[{cycle}] AFTER arm: no delayed "
                        "tick available after commitment; "
                        "discarded"
                    )
                    continue

                price_delta = (
                    0.0 if previous_price is None
                    else price1 - previous_price
                )
                previous_price = price1
                Y = encode_payload_bits(reveal_ns, price_delta)

                # Post-reveal optical encoding: this is a decoder test,
                # not evidence of predicting information before reveal.
                tilt = make_payload_tilt(Y, eps_abs)
                Z = viewport.generate_window_block(n_windows, N_SLOTS, tilt)
                m = Z.mean(axis=0)
                col_z = m / (Z.std(axis=0, ddof=1) / math.sqrt(n_windows))
                bit_z = col_z[col_idx]
                decoded_bits = (bit_z > 0).astype(int)
                decoded_code, decoded_delta = decode_payload_bits(decoded_bits)

                actual_code = timestamp_to_code(reveal_ns)
                decoded_ns, decoded_time = decode_timestamp_from_code(
                    decoded_code, reveal_ns
                )

                after.add(m, Y)

                time_error_ms = (
                    decoded_code - actual_code
                )

                if time_error_ms == 0:
                    timestamp_hits += 1

                if abs(time_error_ms) <= 1:
                    timestamp_within_1ms += 1

                if abs(time_error_ms) <= 10:
                    timestamp_within_10ms += 1

                timestamp_total += 1
                delta_error = decoded_delta - price_delta
                delta_abs_error_total += abs(delta_error)
                delta_total += 1
                if abs(delta_error) < 0.0051:
                    delta_exact_hits += 1

                # Per-cycle certainty.
                price_z = np.abs(bit_z)
                min_z = float(price_z.min())

                joint_conf = float(
                    np.prod(
                        [
                            _phi(z)
                            for z in price_z
                        ]
                    )
                )

                certain = (
                    min_z >= Z_MIN
                    and joint_conf
                    >= JOINT_CONF_MIN
                )

                # Delta-buffer gates.
                delta_buf.append(time_error_ms)

                h, (
                    n_pos,
                    n_neg,
                    n_zero,
                ) = sign_entropy(delta_buf)

                low_entropy = (
                    len(delta_buf) >= MIN_BUF
                    and h <= ENTROPY_MAX_BITS
                )

                dominant = (
                    1
                    if n_pos > n_neg
                    else (
                        -1
                        if n_neg > n_pos
                        else 0
                    )
                )

                agrees = (
                    time_error_ms != 0
                    and dominant != 0
                    and (time_error_ms > 0)
                    == (dominant > 0)
                )

                recent = list(delta_buf)[
                    -STREAK:
                ]

                streak_ok = (
                    len(recent) == STREAK
                    and (
                        all(d > 0 for d in recent)
                        or all(d < 0 for d in recent)
                    )
                )

                big_enough = (
                    abs(time_error_ms)
                    >= MIN_ABS_DELTA
                )

                gates = {
                    "positive control recovered": analysis[
                        "pipeline_ok"
                    ],
                    "after-arm channel signal": analysis[
                        "channel_decode_ok"
                    ],
                    "certain": certain,
                    "low entropy": low_entropy,
                    "agrees": agrees,
                    "streak": streak_ok,
                    "big enough": big_enough,
                }

                failed = [
                    name
                    for name, passed in gates.items()
                    if not passed
                ]

                reveal_delay_ms = (
                    time.perf_counter_ns() - commit_perf_ns
                ) / 1e6

                print(
                    f"[{cycle}] AFTER arm | commit "
                    f"{commit[:10]} | fetched at "
                    f"{reveal_time} | reveal "
                    f"+{reveal_delay_ms:.1f} ms | "
                    f"decoded time {decoded_time} | "
                    f"time error {time_error_ms:+d} ms | "
                    f"price delta {price_delta:+.2f} USD -> "
                    f"decoded {decoded_delta:+.2f} USD | "
                    f"min|z| {min_z:.2f} | "
                    f"entropy {h:.2f}"
                )

                if failed:
                    print(
                        f"    [GATE] Withheld: "
                        f"{', '.join(failed)}"
                    )
                else:
                    print(
                        "    [GATE] All gates passed; "
                        "clock time decoded; "
                        "no action taken."
                    )

            if cycle % report_every == 0:
                if (
                    len(before) >= MIN_CYCLES
                    and len(after) >= MIN_CYCLES
                ):
                    sb = lead_stats(
                        before,
                        LEADS,
                        N_SLOTS,
                        n_perms,
                    )

                    sa = lead_stats(
                        after,
                        LEADS,
                        N_SLOTS,
                        n_perms,
                    )

                    pc_rho, pc_z = sb[LEAK_LEAD]

                    analysis["pipeline_ok"] = (
                        pc_z >= N_SIGMA
                    )

                    best_L = max(
                        LEADS,
                        key=lambda L: sa[L][1],
                    )

                    analysis[
                        "channel_decode_ok"
                    ] = sa[best_L][1] >= N_SIGMA

                    print(
                        "\n[ANALYSIS] " + "-" * 62
                    )

                    print(
                        f"  cycles: before {len(before)}, "
                        f"after {len(after)} | null = "
                        f"{n_perms:,} shuffles of Y"
                    )

                    print(
                        f"  positive control, lead "
                        f"+{LEAK_LEAD}: "
                        f"rho={pc_rho:+.4f} "
                        f"z={pc_z:+.1f} -> "
                        f"{'RECOVERED' if analysis['pipeline_ok'] else 'MISSED'}"
                    )

                    for L in LEADS:
                        print(
                            f"  after-arm encoded-channel lead +{L}: "
                            f"rho={sa[L][0]:+.4f} "
                            f"z={sa[L][1]:+.1f}"
                        )

                    print(
                        f"  rho_leak = "
                        f"{sb[LEAK_LEAD][0] - sa[LEAK_LEAD][0]:+.4f} | "
                        f"rho_after_channel = "
                        f"{sa[LEAK_LEAD][0]:+.4f}"
                    )

                    verdict = (
                        "post-reveal payload channel recovered"
                        if analysis["channel_decode_ok"]
                        else "post-reveal payload channel not recovered"
                    )

                    print(f"  verdict: {verdict}")

                    if timestamp_total:
                        print(
                            f"  exact time matches: "
                            f"{timestamp_hits}/"
                            f"{timestamp_total} "
                            f"({timestamp_hits / timestamp_total:.6f})"
                        )

                        print(
                            f"  within 1 ms: "
                            f"{timestamp_within_1ms}/"
                            f"{timestamp_total} "
                            f"({timestamp_within_1ms / timestamp_total:.6f})"
                        )

                        print(
                            f"  within 10 ms: "
                            f"{timestamp_within_10ms}/"
                            f"{timestamp_total} "
                            f"({timestamp_within_10ms / timestamp_total:.6f})"
                        )

                    if delta_total:
                        print(
                            f"  price-delta exact (cent) matches: "
                            f"{delta_exact_hits}/{delta_total} | "
                            f"mean absolute delta error: "
                            f"${delta_abs_error_total / delta_total:.4f}"
                        )

                    print(
                        "[ANALYSIS] "
                        + "-" * 62
                        + "\n"
                    )

                else:
                    print(
                        f"[ANALYSIS] waiting for "
                        f"{MIN_CYCLES} cycles per arm "
                        f"(before {len(before)}, "
                        f"after {len(after)})"
                    )

    except KeyboardInterrupt:
        pass

    finally:
        fetcher.stop()
        viewport.release()

        print(
            "\n[DONE] Clock-time decoder pipeline "
            "terminated safely."
        )


if __name__ == "__main__":
    # Use "synthetic" first to verify the decoder without a live feed.
    run_full_predictor_pipeline(
        exchange="synthetic",
        delay_seconds=0.1,
        n_windows=2000,
        leak_eps_sigma=0.02,
        balanced_detection=False,
        n_perms=100,
        report_every=10,
    )
