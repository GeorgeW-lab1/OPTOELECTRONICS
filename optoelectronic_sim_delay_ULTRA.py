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

Usage
  python optoelectronic_sim_delay.py --leak --synthetic --headless --delay-ms 150
  python optoelectronic_sim_delay.py --leak --synthetic --headless --fast --sweep-delay 0,50,100,150,300,600 --trials 10
  python optoelectronic_sim_delay.py --leak --delay-ms 150            # real webcam + window

Keys (window): q = quit, r = restart word
Requires: numpy, opencv-python (opencv only for webcam / window)
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
        self.synthetic, self.clock = synthetic, clock
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
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Webcam read failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr

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
        self.y, self.and_raw, self.and_dly = [], [], []
        self.frames = self.and_frames = 0
        self.hamming = []

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
                    hamming=self.hamming)


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

                if slot == nbits:
                    val = int("".join(map(str, select_bits)), 2)
                    attacker.update(val)                   # attacker sees only SELECT
                    ok = verifier.check(attacker.acc)      # commitment oracle
                    meter.word(attacker.acc)
                    runs += 1
                    if verbose:
                        tag = "RECOVERED" if ok else "waiting"
                        sh = f"{val:0{hexw}X}" if (ok or cfg.reveal) else HID
                        gh = f"{attacker.acc:0{hexw}X}" if (ok or cfg.reveal) else HID
                        print(f"run {runs:3d}  SELECT={sh}  ACC={gh}  "
                              f"hamming={meter.hamming[-1]:2d}  conf~{attacker.confidence():.4f}  {tag}"
                              + (f"  (hidden={secret:0{hexw}X})" if cfg.reveal else ""))
                    slot, select_bits = 0, []
                    if ok:
                        recovered = attacker.acc
                        break
                    if max_words and runs >= max_words:
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
            cv2.imshow("optoelectronic sim", draw(bgr, roiB_box, roiA_box, lines, traces))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                slot, select_bits = 0, []
    finally:
        cam.release()
        if not cfg.headless:
            import cv2
            cv2.destroyAllWindows()

    s = meter.summary()
    s.update(recovered=recovered is not None, words=runs, secret=secret, acc=attacker.acc,
             commitment=verifier.commitment, hexw=hexw)
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
    ap.add_argument("--delay-ms", type=float, default=10000000.0, help="AND -> D delay (default ~RC tau)")
    ap.add_argument("--sweep-delay", default=None, help="comma list of delays in ms, e.g. 0,50,150,300")
    ap.add_argument("--trials", type=int, default=10, help="trials per sweep point")
    ap.add_argument("--runs", type=int, default=0, help="stop after N words (0 = until recovered)")
    cfg = ap.parse_args()

    if cfg.fast and not (cfg.synthetic and cfg.headless):
        ap.error("--fast requires --synthetic and --headless")

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
        if s["recovered"]:
            print(f"SECRET RECOVERED after {s['words']} words: {s['acc']:0{s['hexw']}X}")
        else:
            print(f"Not recovered after {s['words']} words; best guess {s['acc']:0{s['hexw']}X}")
        return

    delays = [float(v) for v in cfg.sweep_delay.split(",")]
    cfg.headless = True
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
