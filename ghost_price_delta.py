
import hashlib
import heapq
import json
import math
import threading
import time
import queue
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


# ============================================================================
# CONFIGURATION
# ============================================================================

LAMBDA = 10000.0
SIGMA_C = 0.005

N_SLOTS = 32
FIELD_BITS = 16
PRICE_DELTA_SCALE = 100.0

REVEAL_DELAY = 10.1
SAMPLE_INTERVAL = 1.0

N_WINDOWS = 2000
N_PERMS = 100
REPORT_EVERY = 25

LEADS = (0, 1, 2, 3, 4)

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

EXCHANGE = "binance"


# ============================================================================
# HELPERS
# ============================================================================

def format_clock_time(timestamp_ns):
    seconds = timestamp_ns / 1e9
    clock = time.strftime("%H:%M:%S", time.localtime(seconds))
    ms = int((timestamp_ns % 1_000_000_000) // 1_000_000)
    return f"{clock}.{ms:03d}"


def noise_stats(lambd, sigma_c):
    shared = sigma_c ** 2 * lambd
    var_z = 1.0 + shared
    return var_z, math.sqrt(var_z), shared / var_z


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_entropy(values):
    n = len(values)

    if not n:
        return 0.0, (0, 0, 0)

    counts = (
        sum(x > 0 for x in values),
        sum(x < 0 for x in values),
        sum(x == 0 for x in values),
    )

    h = 0.0

    for c in counts:
        if c:
            p = c / n
            h -= p * math.log2(p)

    return h, counts


# ============================================================================
# 32-BIT PAYLOAD
# ============================================================================

def timestamp_to_code(timestamp_ns):
    ms = (int(timestamp_ns) // 1_000_000) % 60000
    return ms & 0xFFFF


def price_delta_to_code(price_delta):
    cents = int(round(float(price_delta) * 100.0))
    cents = max(-32768, min(32767, cents))
    return cents & 0xFFFF


def code_to_signed_price_delta(code):
    value = int(code) & 0xFFFF

    if value >= 0x8000:
        value -= 0x10000

    return value / 100.0


def encode_payload_bits(timestamp_ns, price_delta):
    time_code = timestamp_to_code(timestamp_ns)
    delta_code = price_delta_to_code(price_delta)

    time_bits = [
        (time_code >> (15 - k)) & 1
        for k in range(16)
    ]

    delta_bits = [
        (delta_code >> (15 - k)) & 1
        for k in range(16)
    ]

    return np.array(time_bits + delta_bits, dtype=int)


def bits_to_code(bits):
    code = 0

    for bit in bits:
        code = (code << 1) | (int(bit) & 1)

    return code


def decode_payload_bits(bits):
    bits = np.asarray(bits, dtype=int).flatten()

    if len(bits) != N_SLOTS:
        raise ValueError("Invalid payload length")

    timestamp_code = bits_to_code(bits[:16])
    delta_code = bits_to_code(bits[16:])

    return (
        timestamp_code,
        code_to_signed_price_delta(delta_code)
    )


def decode_timestamp_from_code(code, reference_ns):
    minute_start = (
        int(reference_ns) // 60_000_000_000
    ) * 60_000_000_000

    decoded_ns = minute_start + (int(code) & 0xFFFF) * 1_000_000

    return decoded_ns, format_clock_time(decoded_ns)


def make_payload_tilt(bits, eps_abs):
    bits = np.asarray(bits, dtype=float)

    strength = max(float(eps_abs), 0.25)

    return 2.0 * strength * (bits - 0.5)


# ============================================================================
# OPTICAL CHANNEL
# ============================================================================

class OptoelectronicViewport:

    def __init__(self, lambd=LAMBDA, sigma_c=SIGMA_C):
        self.lambd = lambd
        self.sigma_c = sigma_c

        print("[OPTICAL] Synthetic photon-counting channel initialized")

    def generate_window_block(self, n_windows, n_slots, tilt=None):

        shared_noise = np.random.normal(
            0.0,
            self.sigma_c,
            size=(n_windows, 1)
        )

        lam = self.lambd * (1.0 + shared_noise)

        counts = np.random.poisson(
            np.clip(lam, 1.0, None),
            size=(n_windows, n_slots)
        )

        Z = (
            counts - self.lambd
        ) / np.sqrt(self.lambd)

        if tilt is not None:
            Z += tilt

        return Z


# ============================================================================
# ASYNCHRONOUS LIVE PRICE FETCHER
# ============================================================================

def parse_binance(d):

    if d.get("s") != "BTCUSDT":
        return None

    bid = d.get("b")
    ask = d.get("a")

    if bid is None or ask is None:
        return None

    bid = float(bid)
    ask = float(ask)

    if bid <= 0 or ask <= 0:
        return None

    return (bid + ask) / 2.0


class AsyncPriceFetcher(threading.Thread):

    def __init__(self, exchange="binance"):

        super().__init__(daemon=True)

        if exchange != "binance":
            raise ValueError("This version uses Binance BTCUSDT")

        self.url = (
            "wss://stream.binance.com:443/ws/"
            "btcusdt@bookTicker"
        )

        self.ticks = queue.Queue(maxsize=1)

        self.running = True
        self.ws = None

        self.received = 0
        self.last_message_ns = 0

    def _on_open(self, ws):
        print("[FETCH] Binance WebSocket connected")
        print("[FETCH] BTCUSDT bookTicker active")

    def _on_message(self, ws, message):

        try:
            data = _loads(message)
            price = parse_binance(data)

            if price is None:
                return

            received_ns = time.time_ns()
            received_perf_ns = time.perf_counter_ns()

            tick = {
                "price": price,
                "received_ns": received_ns,
                "received_perf_ns": received_perf_ns,
            }

            # Keep only the newest exchange tick.
            try:
                self.ticks.put_nowait(tick)
            except queue.Full:
                try:
                    self.ticks.get_nowait()
                except queue.Empty:
                    pass

                self.ticks.put_nowait(tick)

            self.received += 1
            self.last_message_ns = received_ns

        except Exception as e:
            print(f"[FETCH] Parse error: {e}")

    def _on_error(self, ws, error):
        print(f"[FETCH] WebSocket error: {error}")

    def _on_close(self, ws, code, message):
        print(f"[FETCH] WebSocket closed: {code} {message}")

    def run(self):

        if not HAS_WS:
            print("[FETCH] Install websocket-client")
            return

        while self.running:

            try:

                print(f"[FETCH] Connecting: {self.url}")

                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )

                self.ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10
                )

            except Exception as e:
                print(f"[FETCH] Connection exception: {e}")

            if self.running:
                print("[FETCH] Reconnecting in 3 seconds")
                time.sleep(3)

    def stop(self):
        self.running = False

        if self.ws:
            self.ws.close()


# ============================================================================
# ASYNCHRONOUS FORCED REVEAL SCHEDULER
# ============================================================================

class AsyncRevealScheduler(threading.Thread):

    def __init__(self, output_queue):

        super().__init__(daemon=True)

        self.output_queue = output_queue

        self.condition = threading.Condition()
        self.pending = []

        self.sequence = 0
        self.running = True

    def schedule(self, delay_seconds, callback):

        due_perf_ns = (
            time.perf_counter_ns()
            + int(delay_seconds * 1e9)
        )

        due_wall_ns = (
            time.time_ns()
            + int(delay_seconds * 1e9)
        )

        with self.condition:

            self.sequence += 1

            heapq.heappush(
                self.pending,
                (
                    due_perf_ns,
                    self.sequence,
                    due_wall_ns,
                    callback
                )
            )

            self.condition.notify()

        return due_wall_ns

    def run(self):

        while self.running:

            item = None

            with self.condition:

                while self.running and not self.pending:
                    self.condition.wait()

                if not self.running:
                    break

                due_ns = self.pending[0][0]
                remaining = (
                    due_ns - time.perf_counter_ns()
                ) / 1e9

                if remaining > 0:
                    self.condition.wait(timeout=remaining)
                    continue

                item = heapq.heappop(self.pending)

            if item is None:
                continue

            due_perf_ns, sequence, due_wall_ns, callback = item

            actual_perf_ns = time.perf_counter_ns()
            actual_wall_ns = time.time_ns()

            try:

                result = callback(
                    due_wall_ns,
                    actual_wall_ns
                )

                result["scheduled_reveal_ns"] = due_wall_ns
                result["actual_reveal_ns"] = actual_wall_ns
                result["actual_reveal_perf_ns"] = actual_perf_ns

                self.output_queue.put(result)

            except Exception as e:
                print(f"[REVEAL] Callback error: {e}")

    def stop(self):

        with self.condition:
            self.running = False
            self.condition.notify_all()


# ============================================================================
# LEAD-LAG RECORDS
# ============================================================================

class ArmRecord:

    def __init__(self, maxlen=ROLL_CYCLES):
        self.M = deque(maxlen=maxlen)
        self.Y = deque(maxlen=maxlen)

    def add(self, col_means, y_bits):
        self.M.append(np.asarray(col_means, dtype=float))
        self.Y.append(np.asarray(y_bits, dtype=float))

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

        x = Mc[:, :n_slots - L]
        y = Yc[:, L:]

        d = math.sqrt(
            float((x * x).sum() * (y * y).sum())
        )

        den[L] = d

        obs[L] = (
            float((x * y).sum() / d)
            if d > 0 else 0.0
        )

    perm_vals = {L: [] for L in leads}

    done = 0

    while done < n_perms:

        P = min(250, n_perms - done)

        idx = np.argsort(
            np.random.random((P, C)),
            axis=1
        )

        Yp = Yc[idx]

        for L in leads:

            if den[L] > 0:

                s = np.einsum(
                    "ck,pck->p",
                    Mc[:, :n_slots - L],
                    Yp[:, :, L:]
                )

                perm_vals[L].append(s / den[L])

        done += P

    out = {}

    for L in leads:

        if den[L] > 0 and perm_vals[L]:

            arr = np.concatenate(perm_vals[L])
            sd = arr.std()

            z = (
                (obs[L] - arr.mean()) / sd
                if sd > 0 else 0.0
            )

        else:
            z = 0.0

        out[L] = (obs[L], z)

    return out


# ============================================================================
# COMMITMENT CREATION
# ============================================================================

def make_commitment(arm, timestamp_ns, price, price_delta):

    if arm == "BEFORE":

        # The payload is committed before the reveal timer starts.
        bits = encode_payload_bits(
            timestamp_ns,
            price_delta
        )

        material = (
            bits.tobytes()
            + str(timestamp_ns).encode()
            + repr(price).encode()
            + repr(price_delta).encode()
        )

        commitment = hashlib.sha256(material).hexdigest()

        return commitment, bits

    # AFTER arm: independent optical noise commitment.
    # The payload is not encoded into this commitment.
    noise = np.random.normal(
        0.0,
        1.0,
        size=(256, N_SLOTS)
    )

    commitment = hashlib.sha256(
        noise.tobytes()
    ).hexdigest()

    return commitment, None


# ============================================================================
# REVEAL AND DECODER
# ============================================================================

def process_reveal(
    event,
    viewport,
    before,
    after,
    delta_buf,
    analysis,
    counters
):

    arm = event["arm"]

    fetch_ns = event["fetch_ns"]
    fetch_price = event["price"]
    price_delta = event["price_delta"]

    scheduled_ns = event["scheduled_reveal_ns"]
    actual_ns = event["actual_reveal_ns"]

    if arm == "BEFORE":

        # Payload was committed before the delay.
        # Its timestamp represents the scheduled reveal.
        Y = event["committed_bits"].copy()
        payload_timestamp_ns = scheduled_ns

    else:

        # AFTER arm: payload is first encoded here,
        # after the independent commitment and reveal delay.
        payload_timestamp_ns = actual_ns

        Y = encode_payload_bits(
            payload_timestamp_ns,
            price_delta
        )

    # Optical encoding occurs only after reveal.
    _, sigma_z, _ = noise_stats(
        LAMBDA,
        SIGMA_C
    )

    eps_abs = 0.02 * sigma_z

    tilt = make_payload_tilt(Y, eps_abs)

    Z = viewport.generate_window_block(
        N_WINDOWS,
        N_SLOTS,
        tilt
    )

    m = Z.mean(axis=0)

    col_z = (
        m
        / (
            Z.std(axis=0, ddof=1)
            / math.sqrt(N_WINDOWS)
        )
    )

    bit_z = col_z.copy()

    decoded_bits = (bit_z > 0).astype(int)

    decoded_code, decoded_delta = decode_payload_bits(
        decoded_bits
    )

    actual_code = timestamp_to_code(
        payload_timestamp_ns
    )

    decoded_ns, decoded_time = decode_timestamp_from_code(
        decoded_code,
        actual_ns
    )

    time_error_ms = (
        (decoded_code - actual_code + 30000) % 60000
    ) - 30000

    exact_bits = int(
        np.sum(decoded_bits == Y)
    )

    if arm == "BEFORE":
        before.add(m, Y)
    else:
        after.add(m, Y)

    counters["revealed"] += 1

    if time_error_ms == 0:
        counters["timestamp_hits"] += 1

    if abs(time_error_ms) <= 1:
        counters["within_1ms"] += 1

    if abs(time_error_ms) <= 10:
        counters["within_10ms"] += 1

    counters["timestamp_total"] += 1

    delta_error = decoded_delta - price_delta

    counters["delta_abs_error"] += abs(delta_error)
    counters["delta_total"] += 1

    if abs(delta_error) < 0.0051:
        counters["delta_exact"] += 1

    # Confidence statistics.
    min_z = float(np.abs(bit_z).min())

    joint_conf = float(
        np.prod([
            phi(z)
            for z in np.abs(bit_z)
        ])
    )

    certain = (
        min_z >= Z_MIN
        and joint_conf >= JOINT_CONF_MIN
    )

    delta_buf.append(time_error_ms)

    entropy, counts = sign_entropy(delta_buf)

    n_pos, n_neg, n_zero = counts

    low_entropy = (
        len(delta_buf) >= MIN_BUF
        and entropy <= ENTROPY_MAX_BITS
    )

    dominant = (
        1 if n_pos > n_neg
        else -1 if n_neg > n_pos
        else 0
    )

    agrees = (
        time_error_ms != 0
        and dominant != 0
        and (time_error_ms > 0) == (dominant > 0)
    )

    recent = list(delta_buf)[-STREAK:]

    streak_ok = (
        len(recent) == STREAK
        and (
            all(x > 0 for x in recent)
            or all(x < 0 for x in recent)
        )
    )

    big_enough = abs(time_error_ms) >= MIN_ABS_DELTA

    gates = {
        "positive control recovered": analysis["pipeline_ok"],
        "after-arm channel signal": analysis["channel_decode_ok"],
        "certain": certain,
        "low entropy": low_entropy,
        "agrees": agrees,
        "streak": streak_ok,
        "big enough": big_enough
    }

    failed = [
        name
        for name, passed in gates.items()
        if not passed
    ]

    elapsed_ms = (
        event["actual_reveal_perf_ns"]
        - event["commit_perf_ns"]
    ) / 1e6

    scheduled_time = format_clock_time(scheduled_ns)
    actual_time = format_clock_time(actual_ns)
    fetch_time = format_clock_time(fetch_ns)
    commit_time = format_clock_time(event["commit_ns"])

    print()
    print("=" * 78)
    print(f"[{counters['revealed']}] {arm} ARM | FORCED DATA REVEAL")
    print("=" * 78)

    print(f"  Commitment:       {event['commitment'][:24]}...")
    print(f"  Fetched at:       {fetch_time}")
    print(f"  Committed at:     {commit_time}")
    print(f"  Scheduled reveal: {scheduled_time}")
    print(f"  Actual reveal:    {actual_time}")
    print(f"  Elapsed delay:    {elapsed_ms:.2f} ms")

    print("-" * 78)

    print(f"  BTC price:        ${fetch_price:,.2f}")
    print(f"  Actual delta:     {price_delta:+.2f} USD")

    print("-" * 78)

    print(f"  Decoded time:     {decoded_time}")
    print(f"  Time error:       {time_error_ms:+d} ms")
    print(f"  Decoded delta:    {decoded_delta:+.2f} USD")
    print(f"  Bit recovery:     {exact_bits}/{N_SLOTS}")

    print("-" * 78)

    print(f"  Minimum |z|:      {min_z:.2f}")
    print(f"  Joint confidence: {joint_conf:.8f}")
    print(f"  Entropy:          {entropy:.4f}")

    if failed:
        print(f"  [GATE] WITHHELD: {', '.join(failed)}")
    else:
        print("  [GATE] All gates passed; no external action taken.")

    print("=" * 78)

    return time_error_ms


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def run_analysis(before, after, analysis, counters):

    if (
        len(before) < MIN_CYCLES
        or len(after) < MIN_CYCLES
    ):
        print(
            f"[ANALYSIS] Waiting for {MIN_CYCLES} cycles "
            f"per arm: BEFORE={len(before)}, AFTER={len(after)}"
        )
        return

    sb = lead_stats(
        before,
        LEADS,
        N_SLOTS,
        N_PERMS
    )

    sa = lead_stats(
        after,
        LEADS,
        N_SLOTS,
        N_PERMS
    )

    pc_rho, pc_z = sb[0]

    analysis["pipeline_ok"] = pc_z >= N_SIGMA

    best_L = max(
        LEADS,
        key=lambda L: sa[L][1]
    )

    analysis["channel_decode_ok"] = (
        sa[best_L][1] >= N_SIGMA
    )

    print()
    print("[ANALYSIS] " + "-" * 65)

    print(
        f"BEFORE positive control: "
        f"rho={pc_rho:+.4f} z={pc_z:+.2f}"
    )

    for L in LEADS:
        print(
            f"AFTER lead +{L}: "
            f"rho={sa[L][0]:+.4f} "
            f"z={sa[L][1]:+.2f}"
        )

    print(
        f"Positive control: "
        f"{'RECOVERED' if analysis['pipeline_ok'] else 'NOT RECOVERED'}"
    )

    print(
        f"AFTER channel: "
        f"{'RECOVERED' if analysis['channel_decode_ok'] else 'NOT RECOVERED'}"
    )

    print(
        f"Exact timestamp matches: "
        f"{counters['timestamp_hits']}/"
        f"{counters['timestamp_total']}"
    )

    print(
        f"Within 1 ms: "
        f"{counters['within_1ms']}/"
        f"{counters['timestamp_total']}"
    )

    print(
        f"Within 10 ms: "
        f"{counters['within_10ms']}/"
        f"{counters['timestamp_total']}"
    )

    if counters["delta_total"]:

        print(
            f"Exact price-delta matches: "
            f"{counters['delta_exact']}/"
            f"{counters['delta_total']}"
        )

        print(
            f"Mean absolute delta error: "
            f"${counters['delta_abs_error'] / counters['delta_total']:.4f}"
        )

    print("[ANALYSIS] " + "-" * 65)
    print()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_predictor_pipeline(
    exchange=EXCHANGE,
    delay_seconds=REVEAL_DELAY,
    max_cycles=None
):

    np.random.seed(2026)

    var_z, sigma_z, slot_corr = noise_stats(
        LAMBDA,
        SIGMA_C
    )

    prereg = {
        "exchange": exchange,
        "delay_seconds": delay_seconds,
        "lambda": LAMBDA,
        "sigma_c": SIGMA_C,
        "slots": N_SLOTS,
        "windows": N_WINDOWS,
        "leads": list(LEADS),
        "arms": ["BEFORE", "AFTER"],
        "reveal": "asynchronous forced reveal"
    }

    prereg_hash = hashlib.sha256(
        json.dumps(
            prereg,
            sort_keys=True
        ).encode()
    ).hexdigest()

    print("=" * 78)
    print(" RP-GHOST ASYNCHRONOUS COMMIT-REVEAL DECODER")
    print("=" * 78)

    print(f"[PREREG] {prereg_hash}")
    print(f"[EXCHANGE] {exchange}")
    print(f"[DELAY] {delay_seconds:.3f} seconds")
    print(f"[OPTICAL] lambda={LAMBDA}")
    print(f"[OPTICAL] Var(z)={var_z:.3f}")
    print(f"[OPTICAL] sigma(z)={sigma_z:.3f}")
    print(f"[OPTICAL] slot correlation={slot_corr:.4f}")

    print()
    print("[ARMS]")
    print("  BEFORE: Payload commitment -> delay -> forced reveal")
    print("  AFTER:  Noise commitment -> delay -> payload creation/reveal")
    print()
    print("[DISPLAY] Both arms independently display their reveal events.")
    print("[FETCH] WebSocket receiver remains asynchronous.")
    print("[DECODER] Payload is decoded only after reveal.")
    print()

    if not HAS_WS:
        print("[ERROR] Install websocket-client:")
        print("        pip install websocket-client")
        return

    viewport = OptoelectronicViewport()

    fetcher = AsyncPriceFetcher(exchange)

    reveal_queue = queue.Queue()

    scheduler = AsyncRevealScheduler(
        reveal_queue
    )

    before = ArmRecord()
    after = ArmRecord()

    delta_buf = deque(maxlen=DELTA_BUF_LEN)

    analysis = {
        "pipeline_ok": False,
        "channel_decode_ok": False
    }

    counters = {
        "scheduled": 0,
        "revealed": 0,
        "timestamp_hits": 0,
        "within_1ms": 0,
        "within_10ms": 0,
        "timestamp_total": 0,
        "delta_exact": 0,
        "delta_total": 0,
        "delta_abs_error": 0.0
    }

    previous_price = None
    last_sample_perf_ns = 0

    fetcher.start()
    scheduler.start()

    print("[START] Waiting for live BTCUSDT ticks...")

    try:

        while (
            max_cycles is None
            or counters["scheduled"] < max_cycles
            or counters["revealed"] < counters["scheduled"]
        ):

            # --------------------------------------------------------------
            # 1. RECEIVE LIVE TICK WITHOUT BLOCKING REVEAL SCHEDULER
            # --------------------------------------------------------------

            try:
                tick = fetcher.ticks.get(timeout=0.02)
            except queue.Empty:
                tick = None

            if tick is not None:

                now_perf = time.perf_counter_ns()

                if (
                    now_perf - last_sample_perf_ns
                    >= int(SAMPLE_INTERVAL * 1e9)
                    and (
                        max_cycles is None
                        or counters["scheduled"] < max_cycles
                    )
                ):

                    last_sample_perf_ns = now_perf

                    price = tick["price"]
                    fetch_ns = tick["received_ns"]

                    price_delta = (
                        0.0
                        if previous_price is None
                        else price - previous_price
                    )

                    previous_price = price

                    # Alternate randomized arms.
                    arm = (
                        "BEFORE"
                        if np.random.random() < 0.5
                        else "AFTER"
                    )

                    commit_ns = time.time_ns()
                    commit_perf_ns = time.perf_counter_ns()

                    # BEFORE payload timestamp is the scheduled reveal.
                    # AFTER payload timestamp is generated at actual reveal.
                    scheduled_payload_ns = (
                        fetch_ns + int(delay_seconds * 1e9)
                    )

                    commitment, committed_bits = make_commitment(
                        arm,
                        scheduled_payload_ns,
                        price,
                        price_delta
                    )

                    print()
                    """
                    print(
                        f"[COMMIT] {arm} | "
                        f"BTC=${price:,.2f} | "
                        f"delta={price_delta:+.2f} | "
                        f"hash={commitment[:16]}..."
                    )
                    
                    if arm == "BEFORE":
                        print(
                            "[COMMIT] Payload hash committed "
                            "before forced reveal."
                        )
                    else:
                        print(
                            "[COMMIT] Independent noise hash committed; "
                            "payload not yet encoded."
                        )
                    """
                    # Capture immutable values for the reveal callback.
                    captured = {
                        "arm": arm,
                        "price": price,
                        "price_delta": price_delta,
                        "fetch_ns": fetch_ns,
                        "commit_ns": commit_ns,
                        "commit_perf_ns": commit_perf_ns,
                        "commitment": commitment,
                        "committed_bits": (
                            committed_bits.copy()
                            if committed_bits is not None
                            else None
                        )
                    }

                    def reveal_callback(
                        scheduled_ns,
                        actual_ns,
                        data=captured
                    ):

                        return {
                            **data,
                            "arm": data["arm"],
                            "scheduled_reveal_ns": scheduled_ns,
                            "actual_reveal_ns": actual_ns
                        }

                    scheduled_ns = scheduler.schedule(
                        delay_seconds,
                        reveal_callback
                    )

                    counters["scheduled"] += 1
                    """
                    print(
                        f"[SCHEDULE] {arm} | "
                        f"forced reveal at "
                        f"{format_clock_time(scheduled_ns)}"
                    )
                    """
            # --------------------------------------------------------------
            # 2. PROCESS FORCED REVEAL EVENTS
            # --------------------------------------------------------------

            while True:

                try:
                    event = reveal_queue.get_nowait()
                except queue.Empty:
                    break

                process_reveal(
                    event,
                    viewport,
                    before,
                    after,
                    delta_buf,
                    analysis,
                    counters
                )

                # ----------------------------------------------------------
                # 3. PERIODIC STATISTICAL ANALYSIS
                # ----------------------------------------------------------

                if (
                    counters["revealed"] > 0
                    and counters["revealed"] % REPORT_EVERY == 0
                ):

                    if (
                        not hasattr(
                            run_full_predictor_pipeline,
                            "_last_report"
                        )
                        or run_full_predictor_pipeline._last_report
                        != counters["revealed"]
                    ):

                        run_full_predictor_pipeline._last_report = (
                            counters["revealed"]
                        )

                        run_analysis(
                            before,
                            after,
                            analysis,
                            counters
                        )

    except KeyboardInterrupt:

        print("\n[STOP] Keyboard interrupt received.")

    finally:

        print("[SHUTDOWN] Stopping scheduler and exchange...")

        scheduler.stop()
        fetcher.stop()

        scheduler.join(timeout=2.0)
        fetcher.join(timeout=2.0)

        print()
        print("=" * 78)
        print(" FINAL RP-GHOST REPORT")
        print("=" * 78)

        print(
            f"Scheduled: {counters['scheduled']}"
        )

        print(
            f"Revealed:  {counters['revealed']}"
        )

        print(
            f"BEFORE samples: {len(before)}"
        )

        print(
            f"AFTER samples:  {len(after)}"
        )

        print(
            f"Exchange messages received: {fetcher.received}"
        )

        if counters["timestamp_total"]:

            print(
                f"Timestamp exact: "
                f"{counters['timestamp_hits']}/"
                f"{counters['timestamp_total']}"
            )

            print(
                f"Timestamp within 1 ms: "
                f"{counters['within_1ms']}/"
                f"{counters['timestamp_total']}"
            )

        if counters["delta_total"]:

            print(
                f"Price delta exact: "
                f"{counters['delta_exact']}/"
                f"{counters['delta_total']}"
            )

            print(
                f"Mean delta error: "
                f"${counters['delta_abs_error'] / counters['delta_total']:.4f}"
            )

        print("=" * 78)
        print("[DONE] No keyboard injection or trading performed.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    run_full_predictor_pipeline(
        exchange="binance",
        delay_seconds=10.1,
        max_cycles=None
    )
