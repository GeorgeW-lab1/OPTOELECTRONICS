#!/usr/bin/env python3
"""
Webcam-driven simulation of the optoelectronic circuit, now with a DELAY LINE
between the AND gate and the flip-flop, plus a meter that measures what the
AND-accumulating "denoiser" recovers from the camera (photon) leak.

  IR detector A/B (two webcam ROIs) -> Driver A (randomized) / Driver B (deterministic)
  Driver B -> RC node C -> NPN "IF V(C) > 0.7 V THEN output HIGH"
  A and B_out -> OR / AND gates
  AND --[ delay tau_d ]--> AND_dly
  Y (time code, latched from an isolated number) forces D HIGH
  D = Y OR AND_dly                  [only with --leak]
  74HC74 D flip-flop clocked at 10 Hz -> Q = SELECT bit
  SELECT bits pooled into a word; attacker AND-accumulates words; SHA-256
  commitment confirms a correct accumulated guess.

Leak model
  SELECT_i = secret_i OR AND_dly(t_i).  Noise can only turn 0 -> 1, so
  acc &= SELECT strips it (a Z-channel).  The delay moves the sampling instant
  of the AND path relative to the slow RC/Schmitt state (B_out), which changes
  how correlated the corrupting bits are from slot to slot and word to word.

Measurements (per run / sweep point)
  AND duty          fraction of frames with AND high
  p_leak            P(AND_dly = 1 | secret bit = 0): corruption probability
  lag-1 autocorr    correlation of sampled AND_dly between successive slots
  leak bits/word    Z-channel mutual information I(secret ; SELECT) * nbits
  words to recover  words until the AND-accumulator equals the secret
  residual Hamming  bits still wrong in the accumulator after each word

  NOTE: the true secret is used ONLY to score the attacker (Hamming distance,
  p_leak). The attacker itself only sees SELECT words and the commitment oracle.

BTC loop (optional, --btc)
  price feed (Binance/Coinbase/CoinGecko, or offline synthetic) -> momentum predictor
  -> EventDelay(--delay-ms) -> key tap:  UP = Shift+W,  DOWN = Shift+S
  Dry-run (prints only) unless --inject.  The graded hit rate vs a coin flip is reported.
  This path is independent of D / SELECT, so the leak measurements are unaffected.
  Toy signal, not financial advice.

Usage
  # leak attack + delay, offline, virtual time
  python optoelectronic_sim_delay.py --leak --synthetic --headless --fast
  # delay sweep
  python optoelectronic_sim_delay.py --leak --synthetic --headless --fast --sweep-delay 0,50,100,150,300,600 --trials 10
  # BTC loop offline (dry-run keys)
  python optoelectronic_sim_delay.py --leak --synthetic --headless --fast --btc --btc-seconds 60
  # BTC live price, dry-run keys
  python optoelectronic_sim_delay.py --leak --synthetic --headless --btc --btc-seconds 120
  # BTC live price, REAL Shift+W / Shift+S to the focused window (5 s countdown)
  python optoelectronic_sim_delay.py --leak --synthetic --headless --btc --inject
  # run continually until Ctrl+C / q: attack cycles (new secret each time) + BTC loop
  python optoelectronic_sim_delay.py --leak --synthetic --headless --btc --forever
  # real webcam + window, continual
  python optoelectronic_sim_delay.py --leak --btc --forever

Keys (window): q = quit, r = restart word
Requires: numpy; opencv-python (webcam / window only); pynput (only for --inject)
"""
import argparse
import collections
import hashlib
import math
import secrets
import time
import numpy as np

VCC = 5.0
V_BE = 0.7
R_OHM = 10e3
C_FARAD = 15e-6          # tau = RC = 0.15 s
CLK_PERIOD = 0.100       # 10 Hz
PHOTONS_PER_LEVEL = 4.0  # synthetic camera: photon shot-noise scale


# --------------------------------------------------------------------------- time
class Clock:
    """Real time, or virtual time (--fast) advanced one frame per tick."""
    def __init__(self, virtual=False, frame_dt=1 / 30):
        self.virtual, self.frame_dt, self.t = virtual, frame_dt, 0.0

    def now(self):
        return self.t if self.virtual else time.time()

    def tick(self):
        if self.virtual:
            self.t += self.frame_dt


# --------------------------------------------------------------------------- sources
class Camera:
    def __init__(self, index, clock, synthetic=False, seed=1):
        self.synthetic, self.clock, self.index = synthetic, clock, index
        self.t0 = clock.now()
        self.rng = np.random.default_rng(seed)
        if not synthetic:
            import cv2
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open webcam {index}")

    def read(self):
        """Return (gray uint8 image, bgr image)."""
        if self.synthetic:
            t = self.clock.now() - self.t0
            h, w = 120, 160
            base = np.zeros((h, w), np.float32)
            base[:, : w // 2] = 110 + 90 * np.sin(2 * np.pi * 0.4 * t)       # B side
            base[:, w // 2:] = 110 + 90 * np.sin(2 * np.pi * 0.7 * t + 1.0)  # A side
            # photon shot noise: Poisson counts, not Gaussian read noise
            lam = np.clip(base, 0, None) * PHOTONS_PER_LEVEL
            img = np.clip(self.rng.poisson(lam) / PHOTONS_PER_LEVEL, 0, 255).astype(np.uint8)
            return img, np.dstack([img, img, img])
        import cv2
        for attempt in range(30):                      # long runs: ride out camera hiccups
            ok, bgr = self.cap.read()
            if ok:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr
            time.sleep(0.2)
            self.cap.release()
            self.cap = cv2.VideoCapture(self.index)
        raise RuntimeError("Webcam read failed repeatedly")

    def release(self):
        if not self.synthetic:
            self.cap.release()


# --------------------------------------------------------------------------- circuit blocks
class DriverB:
    """Deterministic Schmitt comparator on detector-B intensity; threshold set by instance x."""
    def __init__(self, x):
        self.thr = 0.30 + 0.40 * (((x * 2654435761) % 2**32) / 2**32)
        self.hyst = 0.03
        self.state = False

    def step(self, intensity):
        if self.state and intensity < self.thr - self.hyst:
            self.state = False
        elif not self.state and intensity > self.thr + self.hyst:
            self.state = True
        return VCC if self.state else 0.0


class DriverA:
    """Randomized: hash parity of the detector-A ROI (toggles with photon noise)."""
    def step(self, roi):
        digest = hashlib.sha256(np.ascontiguousarray(roi).tobytes()).digest()
        return VCC if digest[0] & 1 else 0.0


class IfThenLink:
    """RC node + NPN: IF V(C) > 0.7 V THEN output HIGH."""
    def __init__(self):
        self.vc = 0.0

    def step(self, v_drive, dt):
        self.vc += (v_drive - self.vc) * (1 - np.exp(-dt / (R_OHM * C_FARAD)))
        return VCC if self.vc > V_BE else 0.0


class DelayLine:
    """Pure transport delay (e.g. RC/buffer chain) between the AND gate and D."""
    def __init__(self, delay_s):
        self.delay = delay_s
        self.q = collections.deque()   # (timestamp, value)
        self.out = False

    def step(self, t, v):
        self.q.append((t, bool(v)))
        # newest sample that is at least `delay` old becomes the output
        while self.q and self.q[0][0] <= t - self.delay:
            self.out = self.q.popleft()[1]
        return self.out


class EventDelay:
    """Delays discrete events by a fixed time. Unlike sampling a level, a slow frame can never skip one."""
    def __init__(self, delay_s):
        self.delay, self.q = delay_s, collections.deque()

    def push(self, t, payload):
        self.q.append((t + self.delay, payload))

    def due(self, now):
        out = []
        while self.q and self.q[0][0] <= now:
            out.append(self.q.popleft()[1])
        return out


class TimeCodeLatch:
    def __init__(self, value, nbits):
        self.value, self.nbits = value, nbits

    def bit(self, slot):
        return (self.value >> (self.nbits - 1 - slot)) & 1


class DFlipFlop74HC74:
    def __init__(self):
        self.q = 0

    def clock(self, d):
        self.q = int(d)


def hi(v):
    return v > VCC / 2


# --------------------------------------------------------------------------- BTC feed / predictor / keys
class SyntheticPriceFeed:
    """Offline random walk with mild momentum (so the predictor has something to find)."""
    def __init__(self, poll_s, seed=7, start=65000.0):
        self.rng = np.random.default_rng(seed)
        self.poll_s, self.price, self.mom, self.next_t = poll_s, start, 0.0, None

    def update(self, now):
        if self.next_t is None:
            self.next_t = now
        if now < self.next_t:
            return None
        self.next_t += self.poll_s
        self.mom = 0.3 * self.mom + self.rng.normal(0, 4e-4)
        self.price *= math.exp(self.mom)
        return self.price

    def close(self):
        pass


class LivePriceFeed:
    """Polls a public spot-price endpoint in a background thread (never blocks the frame loop).
    Uses one keep-alive HTTPS connection (no TLS handshake per poll) and a fixed poll cadence."""
    SOURCES = {
        "coinbase": ("https://api.coinbase.com/v2/prices/BTC-USD/spot",
                     lambda j: float(j["data"]["amount"])),
        "binance": ("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                    lambda j: float(j["price"])),
        "coingecko": ("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                      lambda j: float(j["bitcoin"]["usd"])),
    }
    MIN_POLL = {"coinbase": 0.4, "binance": 0.1, "coingecko": 5.0}   # stay under public rate limits

    def __init__(self, source, poll_s):
        import threading
        floor = self.MIN_POLL[source]
        if poll_s < floor:
            print(f"[btc] {source}: poll {poll_s}s is too fast for its public rate limit; using {floor}s")
            poll_s = floor
        self.source, self.poll_s = source, poll_s
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._latest, self._seq, self._seen, self.err = None, 0, 0, None
        self.polls = self.errors = 0
        self.lat_ms = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import http.client
        import json
        import urllib.parse
        url, parse = self.SOURCES[self.source]
        u = urllib.parse.urlparse(url)
        path = u.path + ("?" + u.query if u.query else "")
        conn, next_t, consec = None, time.monotonic(), 0
        while not self._stop.is_set():
            try:
                if conn is None:
                    conn = http.client.HTTPSConnection(u.netloc, timeout=5)
                t0 = time.monotonic()
                conn.request("GET", path, headers={"User-Agent": "optoelectronic-sim"})
                r = conn.getresponse()
                body = r.read()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                p = parse(json.loads(body))
                with self._lock:
                    self._latest, self._seq = p, self._seq + 1
                self.polls += 1
                consec = 0
                self.lat_ms += ((time.monotonic() - t0) * 1000 - self.lat_ms) / min(self.polls, 20)
            except Exception as e:           # network down, rate-limited, geo-blocked...
                self.err, self.errors, consec = str(e), self.errors + 1, consec + 1
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            next_t += self.poll_s
            wait = next_t - time.monotonic()
            if wait < 0:                     # can't keep up: don't build a backlog
                next_t, wait = time.monotonic(), 0
            self._stop.wait(wait + min(5.0, 0.5 * consec))   # back off while failing

    def update(self, now):
        with self._lock:
            if self._seq != self._seen:
                self._seen = self._seq
                return self._latest
        return None

    def close(self):
        self._stop.set()


class Predictor:
    """Momentum t-stat on recent log-returns -> direction (+1 up, -1 down, 0 no call).
    Repeated prices are ignored (they carry no information and give zero variance).
    Scores itself: each call is graded against the next *changed* price."""
    def __init__(self, window=12, thr=0.5):
        self.window = max(2, window)
        self.px = collections.deque(maxlen=self.window + 1)
        self.thr, self.pending = thr, None
        self.n = self.hits = self.calls = self.moves = 0

    def update(self, price):
        """Returns (direction, score), or None if the price did not change."""
        if self.px and price == self.px[-1]:
            return None
        self.moves += 1
        if self.pending and self.pending[0] != 0:
            self.n += 1
            self.hits += int((price > self.pending[1]) == (self.pending[0] > 0))
        self.px.append(price)
        d, score = 0, 0.0
        need = min(6, self.px.maxlen)          # was a fixed 6: never reached when window < 5
        if len(self.px) >= need:
            r = np.diff(np.log(np.array(self.px)))
            sd = r.std(ddof=1) if len(r) > 1 else 0.0
            if sd > 0:
                score = float(r.mean() / (sd / math.sqrt(len(r))))
                d = 1 if score > self.thr else -1 if score < -self.thr else 0
        self.calls += int(d != 0)
        self.pending = (d, price)
        return d, score

    def stats(self):
        z = (self.hits - self.n / 2) / math.sqrt(self.n / 4) if self.n else 0.0
        return dict(n=self.n, hits=self.hits, rate=self.hits / self.n if self.n else float("nan"),
                    z=z, calls=self.calls, moves=self.moves)


class KeyInjector:
    """Taps Shift+W / Shift+S. Dry-run (print only) unless enabled AND pynput works."""
    def __init__(self, enabled, cooldown=1.0, quiet=False):
        self.kb, self.cooldown, self.last, self.suppressed = None, cooldown, -1e9, 0
        self.sent, self.count, self.quiet = collections.deque(maxlen=50), 0, quiet
        if enabled:
            try:
                from pynput.keyboard import Controller, Key
                self.kb, self.Key = Controller(), Key
            except Exception as e:
                print(f"[key] pynput unavailable ({e}); falling back to dry-run")

    def combo(self, letter, now):
        if now - self.last < self.cooldown - 0.01:   # tolerance for frame-time jitter
            self.suppressed += 1
            return False
        self.last = now
        label = f"Shift+{letter.upper()}"
        if self.kb:
            with self.kb.pressed(self.Key.shift):
                self.kb.press(letter)
                self.kb.release(letter)
        if not self.quiet:
            print(f"[key] {label}" + ("" if self.kb else "  (dry-run)"))
        self.sent.append(label)
        self.count += 1
        return True


# --------------------------------------------------------------------------- commitment
def commit_hash(salt, value, nbits):
    return hashlib.sha256(f"{salt}:{value:0{nbits // 4}X}".encode()).hexdigest()


class Verifier:
    """Holds only a salted SHA-256 commitment, never the number."""
    def __init__(self, value, nbits, salt=None):
        self.nbits = nbits
        self.salt = salt or secrets.token_hex(8)
        self.commitment = commit_hash(self.salt, value, nbits)

    def check(self, value):
        return secrets.compare_digest(commit_hash(self.salt, value, self.nbits), self.commitment)


# --------------------------------------------------------------------------- attacker + meter
class AndAccumulator:
    """Noise only adds 1s, so AND-ing SELECT words strips it."""
    def __init__(self, nbits):
        self.nbits = nbits
        self.acc = (1 << nbits) - 1
        self.ones = np.zeros(nbits, int)
        self.n = 0

    def update(self, word):
        self.acc &= word
        self.n += 1
        for i in range(self.nbits):
            self.ones[i] += (word >> (self.nbits - 1 - i)) & 1

    def confidence(self):
        zero_pos = [i for i in range(self.nbits)
                    if not (self.acc >> (self.nbits - 1 - i)) & 1]
        if self.n == 0 or not zero_pos:
            return 0.0
        p = float(np.mean([self.ones[i] / self.n for i in zero_pos]))
        return (1 - p ** self.n) ** (self.nbits - len(zero_pos))


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def zchannel_info(p):
    """I(X;Y) in bits for uniform input through a Z-channel (1->1 always, 0->1 w.p. p)."""
    return h2(0.5 + 0.5 * p) - 0.5 * h2(p)


class LeakMeter:
    """Scores the leak. Uses the true secret only for scoring, never for attacking."""
    def __init__(self, secret, nbits):
        self.secret, self.nbits = secret, nbits
        mk = lambda n: collections.deque(maxlen=n)      # bounded so it can run forever
        self.y, self.and_raw, self.and_dly = mk(200000), mk(200000), mk(200000)
        self.frames = self.and_frames = 0
        self.hamming = mk(100000)

    def frame(self, g_and):
        self.frames += 1
        self.and_frames += int(g_and)

    def slot(self, y, g_and, g_and_dly):
        self.y.append(int(y)); self.and_raw.append(int(g_and)); self.and_dly.append(int(g_and_dly))

    def word(self, acc):
        self.hamming.append(bin(acc ^ self.secret).count("1"))

    def summary(self):
        y, ad = np.array(self.y), np.array(self.and_dly, float)
        zeros = ad[y == 0]
        p_leak = float(zeros.mean()) if len(zeros) else float("nan")
        ac = float(np.corrcoef(ad[:-1], ad[1:])[0, 1]) if len(ad) > 2 and ad.std() > 0 else 0.0
        return dict(duty=self.and_frames / max(1, self.frames), p_leak=p_leak, autocorr=ac,
                    bits_per_word=zchannel_info(p_leak) * self.nbits if p_leak == p_leak else 0.0,
                    hamming=list(self.hamming))


# --------------------------------------------------------------------------- drawing
def draw(frame, roiB, roiA, text_lines, traces):
    import cv2
    h, w = frame.shape[:2]
    out = frame.copy()
    cv2.rectangle(out, (roiB[0], roiB[1]), (roiB[2], roiB[3]), (60, 60, 255), 2)
    cv2.putText(out, "IR det B", (roiB[0] + 4, roiB[1] + 18), 0, 0.5, (60, 60, 255), 1)
    cv2.rectangle(out, (roiA[0], roiA[1]), (roiA[2], roiA[3]), (60, 200, 60), 2)
    cv2.putText(out, "IR det A", (roiA[0] + 4, roiA[1] + 18), 0, 0.5, (60, 200, 60), 1)
    y = h - 8 - 16 * len(text_lines)
    for line in text_lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16

    names = ["A", "B_out", "OR", "AND", "AND_dly", "Y", "D", "SELECT"]
    pw, rowh = max(w, 480), 22
    panel = np.full((rowh * len(names) + 8, pw, 3), 25, np.uint8)
    for i, n in enumerate(names):
        y0 = 4 + i * rowh
        cv2.putText(panel, n, (4, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        data = list(traces[n])[-(pw - 70):]
        pts = [(60 + k, y0 + (4 if v else 17)) for k, v in enumerate(data)]
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(panel, p, q, (0, 220, 255), 1)
    if panel.shape[1] != out.shape[1]:
        pad = np.zeros((out.shape[0], panel.shape[1] - out.shape[1], 3), np.uint8)
        out = np.hstack([out, pad])
    return np.vstack([out, panel])


# --------------------------------------------------------------------------- one session
def run_session(cfg, delay_ms, seed=1, verbose=True):
    nbits = max(4, cfg.bits - cfg.bits % 4)
    mask = (1 << nbits) - 1
    secret = (int(cfg.secret, 16) if cfg.secret else secrets.randbits(nbits)) & mask
    hexw = nbits // 4
    verifier = Verifier(secret, nbits)
    max_words = cfg.runs or (200 if (cfg.sweep_delay or not cfg.leak) else 0)

    clock = Clock(virtual=cfg.fast)
    cam = Camera(cfg.cam, clock, cfg.synthetic, seed)
    drvA, drvB, link = DriverA(), DriverB(cfg.x), IfThenLink()
    delay = DelayLine(delay_ms / 1000.0)
    latch, ff = TimeCodeLatch(secret, nbits), DFlipFlop74HC74()
    traces = collections.defaultdict(lambda: collections.deque(maxlen=600))
    attacker, meter = AndAccumulator(nbits), LeakMeter(secret, nbits)

    slot, select_bits, runs, recovered = 0, [], 0, None
    HID = "*" * hexw
    last_t = clock.now()
    next_clk = last_t + CLK_PERIOD
    prev_I_B = None

    # --- BTC prediction -> delay -> key injection
    feed = None
    if cfg.btc:
        src = "synthetic" if cfg.fast else cfg.btc_source
        feed = (SyntheticPriceFeed(cfg.btc_poll) if src == "synthetic"
                else LivePriceFeed(src, cfg.btc_poll))
    pred = Predictor(cfg.btc_window, cfg.btc_thr)
    keys = KeyInjector(cfg.inject, cfg.key_cooldown, quiet=cfg.forever and not cfg.show_ticks)
    key_delay = EventDelay(delay_ms / 1000.0)
    last_dir = last_fired = 0
    last_price, last_score, ticks = None, 0.0, 0
    run_until = (float("inf") if cfg.forever else
                 ((cfg.btc_seconds if cfg.btc_seconds is not None else 60.0) if cfg.btc else 0))
    attack_done, t_start = False, clock.now()
    cycle = cfg.forever and not cfg.no_cycle       # continual mode: new secret after every recovery
    cycles, cycle_runs, cycle_words = 0, 0, collections.deque(maxlen=1000)
    next_status = t_start + cfg.status_every
    show_ticks = verbose and (not cfg.forever or cfg.show_ticks)

    try:
        while True:
            gray, bgr = cam.read()
            clock.tick()
            now = clock.now()
            dt, last_t = now - last_t, now
            h, w = gray.shape
            roiB_box = (0, 0, w // 2 - 4, h - 1)
            roiA_box = (w // 2 + 4, 0, w - 1, h - 1)
            roiB = gray[roiB_box[1]:roiB_box[3], roiB_box[0]:roiB_box[2]]
            roiA = gray[roiA_box[1]:roiA_box[3], roiA_box[0]:roiA_box[2]]

            I_B = float(roiB.mean()) / 255.0
            d_int = 0.0 if prev_I_B is None else I_B - prev_I_B
            prev_I_B = I_B

            vA, vB = drvA.step(roiA), drvB.step(I_B)
            vB_out = link.step(vB, dt)
            a, b = hi(vA), hi(vB_out)
            g_or, g_and = a or b, a and b
            g_and_dly = delay.step(now, g_and)             # <-- AND -> delay -> D
            meter.frame(g_and)

            if feed:
                px = feed.update(now)
                if px is not None:
                    ticks += 1
                    res = pred.update(px)
                    if res is not None:                    # price actually moved
                        last_price = px
                        last_dir, last_score = res
                        if last_dir and (not cfg.on_change or last_dir != last_fired):
                            key_delay.push(now, "w" if last_dir > 0 else "s")   # AND-style delay, then key
                            last_fired = last_dir
                        if show_ticks:
                            arrow = {1: "UP", -1: "DOWN", 0: "--"}[last_dir]
                            print(f"[btc] ${px:,.2f}  score={last_score:+.2f}  predict={arrow}")
                for letter in key_delay.due(now):
                    keys.combo(letter, now)

            y = bool(latch.bit(slot))
            d = y or (g_and_dly and cfg.leak)

            for k, v in zip(["A", "B_out", "OR", "AND", "AND_dly", "Y", "D", "SELECT"],
                            [a, b, g_or, g_and, g_and_dly, y, d, ff.q]):
                traces[k].append(int(v))

            if now >= next_clk:                            # 10 Hz clock edge
                next_clk += CLK_PERIOD
                meter.slot(y, g_and, g_and_dly)
                ff.clock(d)
                select_bits.append(ff.q)
                slot += 1

                if slot == nbits and attack_done:
                    slot, select_bits = 0, []
                elif slot == nbits:
                    val = int("".join(map(str, select_bits)), 2)
                    attacker.update(val)                   # attacker sees only SELECT
                    ok = verifier.check(attacker.acc)      # commitment oracle
                    meter.word(attacker.acc)
                    runs += 1
                    cycle_runs += 1
                    if verbose and (ok or not cfg.forever):
                        tag = "RECOVERED" if ok else "waiting"
                        sh = f"{val:0{hexw}X}" if (ok or cfg.reveal) else HID
                        gh = f"{attacker.acc:0{hexw}X}" if (ok or cfg.reveal) else HID
                        print(f"run {runs:3d}  SELECT={sh}  ACC={gh}  "
                              f"hamming={meter.hamming[-1]:2d}  conf~{attacker.confidence():.4f}  {tag}"
                              + (f"  (hidden={secret:0{hexw}X})" if cfg.reveal else ""))
                    slot, select_bits = 0, []
                    failed = (not ok) and bool(max_words) and cycle_runs >= max_words
                    if ok:
                        recovered = attacker.acc
                        cycles += 1
                        cycle_words.append(cycle_runs)
                        if verbose and cfg.forever:
                            print(f"[cycle {cycles}] secret recovered in {cycle_runs} words")
                    if ok or failed:
                        if cycle:                   # continual: fresh secret, attack again
                            secret = secrets.randbits(nbits)
                            verifier = Verifier(secret, nbits)
                            latch = TimeCodeLatch(secret, nbits)
                            attacker = AndAccumulator(nbits)
                            meter.secret = secret
                            cycle_runs = 0
                        elif run_until:
                            attack_done = True      # keep the circuit running for the BTC loop
                        else:
                            break

            if cfg.forever and now >= next_status:
                next_status = now + cfg.status_every
                el, st = max(1e-9, now - t_start), pred.stats()
                msg = f"[status] t={el:7.0f}s  attack cycles={cycles}"
                if cycle_words:
                    msg += f" (mean {np.mean(cycle_words):.1f} words)"
                if feed:
                    msg += (f" | btc fetch={ticks / el:.1f}/s moves={st['moves'] / el:.1f}/s "
                            f"calls={st['calls']} hit={st['rate']:.2f} keys={keys.count} supp={keys.suppressed}")
                    if getattr(feed, "err", None):
                        msg += f" err={feed.err}"
                print(msg)

            if run_until and now - t_start >= run_until:
                break

            if cfg.headless:
                if not cfg.fast:
                    time.sleep(0.01)
                continue

            import cv2
            sel = "".join(map(str, select_bits))
            lines = [f"I_B={I_B:.2f} dI={d_int:+.3f}  Vc={link.vc:.2f}V  thr={drvB.thr:.2f}  delay={delay_ms:.0f}ms",
                     f"A={int(a)} Bout={int(b)} OR={int(g_or)} AND={int(g_and)} "
                     f"ANDdly={int(g_and_dly)} Y={int(y)} D={int(d)} Q={ff.q}",
                     f"slot {slot}/{nbits}  SELECT bits={sel}",
                     f"ACC hamming={meter.hamming[-1] if meter.hamming else nbits}  runs={runs}"]
            if cfg.btc:
                lines.append(f"BTC ${last_price or 0:,.0f} score={last_score:+.2f} dir={last_dir:+d} "
                             f"keys={keys.count} hit={pred.stats()['rate']:.2f}")
            cv2.imshow("optoelectronic sim", draw(bgr, roiB_box, roiA_box, lines, traces))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                slot, select_bits = 0, []
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if feed:
            feed.close()
        cam.release()
        if not cfg.headless:
            import cv2
            cv2.destroyAllWindows()

    s = meter.summary()
    s.update(recovered=recovered is not None, words=(cycle_words[-1] if cycle_words else runs),
             secret=secret, acc=(recovered if recovered is not None else attacker.acc),
             cycles=cycles, cycle_words=list(cycle_words), total_words=runs,
             commitment=verifier.commitment, hexw=hexw,
             btc=pred.stats(), btc_ticks=ticks, keys_count=keys.count, suppressed=keys.suppressed,
             btc_err=getattr(feed, 'err', None), btc_errors=getattr(feed, 'errors', 0),
             lat_ms=getattr(feed, 'lat_ms', 0.0), elapsed=max(1e-9, clock.now() - t_start))
    return s


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="fake camera (no hardware)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="virtual time (30 fps steps, no sleeping); needs --synthetic --headless")
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--secret", default=None, help="hidden number, hex (default random)")
    ap.add_argument("--reveal", action="store_true")
    ap.add_argument("--x", type=int, default=12345)
    ap.add_argument("--leak", action="store_true", help="enable D = Y OR AND_dly")
    ap.add_argument("--delay-ms", type=float, default=100.0, help="AND -> D delay and prediction -> key delay")
    ap.add_argument("--sweep-delay", default=None, help="comma list of delays in ms, e.g. 0,50,150,300")
    ap.add_argument("--trials", type=int, default=100000000, help="trials per sweep point")
    ap.add_argument("--runs", type=int, default=0, help="stop after N words (0 = until recovered)")
    ap.add_argument("--btc", action="store_true", help="fetch BTC price, predict, fire delayed keys")
    ap.add_argument("--btc-source", default="binance",
                    choices=["coinbase", "binance", "coingecko", "synthetic"])
    ap.add_argument("--btc-poll", type=float, default=0.1, help="seconds between price polls")
    ap.add_argument("--btc-window", type=int, default=4, help="returns used by the predictor")
    ap.add_argument("--btc-thr", type=float, default=5e-8, help="|t-stat| needed to make a call")
    ap.add_argument("--btc-seconds", type=float, default=None,
                    help="total seconds to run the BTC loop (default 60 with --btc)")
    ap.add_argument("--inject", action="store_true",
                    help="REALLY send Shift+W (up) / Shift+S (down) to the focused window; default is dry-run")
    ap.add_argument("--key-cooldown", type=float, default=0.1, help="min seconds between injected combos")
    ap.add_argument("--on-change", action="store_true",
                    help="fire only when the predicted direction flips (default: every directional call)")
    ap.add_argument("--forever", action="store_true",
                    help="run continually until Ctrl+C / q: attack repeats with a fresh secret, BTC loop never stops")
    ap.add_argument("--no-cycle", action="store_true", help="with --forever: attack once, then only the BTC loop")
    ap.add_argument("--status-every", type=float, default=10.0, help="seconds between [status] lines (--forever)")
    ap.add_argument("--show-ticks", action="store_true", help="with --forever: print every price tick / key")
    cfg = ap.parse_args()

    if cfg.fast and not (cfg.synthetic and cfg.headless):
        ap.error("--fast requires --synthetic and --headless")

    if cfg.forever and cfg.fast:
        ap.error("--forever can't be combined with --fast (virtual time would spin the CPU forever)")
    if cfg.forever and cfg.sweep_delay:
        ap.error("--forever can't be combined with --sweep-delay")
    if cfg.inject and not cfg.btc:
        ap.error("--inject needs --btc")
    if cfg.inject and not cfg.fast:
        print("!! --inject: real Shift+W / Shift+S will go to whichever window has focus.")
        print("!! Starting in 5 s. Focus the target now. Ctrl+C aborts.")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            return

    if not cfg.sweep_delay:
        if not cfg.leak:
            print("Note: without --leak, D == Y, so the first SELECT word equals the secret.")
        s = run_session(cfg, cfg.delay_ms)
        print(f"\ncommitment : {s['commitment']}")
        print(f"delay      : {cfg.delay_ms:.0f} ms")
        print(f"AND duty   : {s['duty']:.3f}")
        print(f"p_leak     : {s['p_leak']:.3f}   (P[AND_dly=1 | secret bit=0])")
        print(f"autocorr   : {s['autocorr']:+.3f}   (lag-1, successive slots)")
        print(f"leak/word  : {s['bits_per_word']:.2f} of {cfg.bits} bits (Z-channel)")
        if cfg.btc:
            b, el = s["btc"], s["elapsed"]
            print(f"BTC fetches: {s['btc_ticks']} ({s['btc_ticks'] / el:.1f}/s, latency~{s['lat_ms']:.0f} ms, "
                  f"errors {s['btc_errors']})  price moves: {b['moves']} ({b['moves'] / el:.1f}/s)")
            print(f"BTC calls  : {b['calls']} directional   graded: {b['n']}   hit rate: {b['rate']:.2f} "
                  f"(z={b['z']:+.2f} vs coin-flip)")
            print(f"keys sent  : {s['keys_count']}   suppressed by cooldown: {s['suppressed']}")
            if s["btc_err"]:
                print(f"BTC feed error (last): {s['btc_err']}")
        if s["cycles"] > 1:
            print(f"attack cycles: {s['cycles']}   mean words/cycle: {np.mean(s['cycle_words']):.1f}   "
                  f"total words: {s['total_words']}")
        if s["recovered"]:
            print(f"SECRET RECOVERED after {s['words']} words: {s['acc']:0{s['hexw']}X}")
        else:
            print(f"Not recovered after {s['words']} words; best guess {s['acc']:0{s['hexw']}X}")
        return

    delays = [float(v) for v in cfg.sweep_delay.split(",")]
    cfg.headless, cfg.btc = True, False
    print(f"Sweep: {cfg.trials} trials/point, {cfg.bits}-bit secret, leak={'on' if cfg.leak else 'OFF'}\n")
    print(f"{'delay ms':>8} {'AND duty':>9} {'p_leak':>7} {'autocorr':>9} {'bits/word':>10} "
          f"{'recovered':>10} {'words (mean)':>13} {'Hamming@5':>10}")
    for dly in delays:
        res = [run_session(cfg, dly, seed=1000 + i, verbose=False) for i in range(cfg.trials)]
        rec = [r["words"] for r in res if r["recovered"]]
        h5 = [r["hamming"][min(4, len(r["hamming"]) - 1)] for r in res]
        print(f"{dly:8.0f} {np.mean([r['duty'] for r in res]):9.3f} "
              f"{np.nanmean([r['p_leak'] for r in res]):7.3f} "
              f"{np.mean([r['autocorr'] for r in res]):+9.3f} "
              f"{np.mean([r['bits_per_word'] for r in res]):10.2f} "
              f"{len(rec):>6}/{cfg.trials:<3} "
              f"{(np.mean(rec) if rec else float('nan')):13.1f} {np.mean(h5):10.2f}")


if __name__ == "__main__":
    main()
