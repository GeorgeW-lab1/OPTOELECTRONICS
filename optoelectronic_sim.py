#!/usr/bin/env python3
"""
Webcam-driven optoelectronic circuit, symmetry-restored version.

  IR detector A/B (two webcam ROIs) -> Driver A (randomized) / Driver B (deterministic)
  Driver B -> RC node C -> NPN "IF V(C) > 0.7 V THEN HIGH"          (IF/THEN link)

  MATRIX BIAS: for fair independent bits, OR is 75/25 and AND is 25/75 (De Morgan duals).
  SYMMETRY RESTORATION (pure gates):
      G = OR . NOT(AND) = A xor B          -> P(G=1) = 50 %, the balanced reference
  DUAL RAIL: 74HC74 Q (1-bits) and Qbar (0-bits). Every slot asserts exactly one rail,
      so a missing bit is a present signal, and Q == Qbar flags a hardware fault.
  ADAPTATION: a feedback controller retunes Driver B's threshold so that P(B_out) -> 0.5.
      With P(B)=0.5, P(A xor B)=0.5 for ANY bias of A, so G stays balanced when the
      randomized detector drifts. This is the "matrix changes from measurement" loop.
  D = Y (time code only). Gate noise never reaches D unless a leak mode is chosen.

Usage:
  python optoelectronic_sim_balanced.py
  python optoelectronic_sim_balanced.py --secret A5C3 --reveal
  python optoelectronic_sim_balanced.py --synthetic --headless --runs 3
  python optoelectronic_sim_balanced.py --synthetic --headless --runs 3 --leak and
  python optoelectronic_sim_balanced.py --no-adapt          # open-loop for comparison

Keys (window): q = quit, r = restart word.   Requires: numpy, opencv-python
"""
import argparse
import hashlib
import secrets
import time
import collections
import numpy as np

VCC = 5.0
V_BE = 0.7
R_OHM = 10e3
C_FARAD = 15e-6          # tau = RC = 0.15 s
CLK_PERIOD = 0.100       # 10 Hz


# --------------------------------------------------------------------------- sources
class Camera:
    def __init__(self, index, synthetic=False):
        self.synthetic = synthetic
        self.t0 = time.time()
        self.rng = np.random.default_rng(1)
        if not synthetic:
            import cv2
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open webcam {index}")

    def read(self):
        """Return (gray uint8 image, bgr image)."""
        if self.synthetic:
            h, w = 240, 320
            t = time.time() - self.t0
            base = np.zeros((h, w), np.float32)
            base[:, : w // 2] = 110 + 90 * np.sin(2 * np.pi * 0.4 * t)       # B side
            base[:, w // 2:] = 110 + 90 * np.sin(2 * np.pi * 0.7 * t + 1.0)  # A side
            img = np.clip(base + self.rng.normal(0, 3, (h, w)), 0, 255).astype(np.uint8)
            return img, np.dstack([img] * 3)
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
    """Deterministic Schmitt comparator on detector-B intensity.
    Threshold starts from problem instance x and is retuned by the adaptive loop."""
    THR_MIN, THR_MAX = 0.05, 0.95

    def __init__(self, x):
        self.thr = 0.30 + 0.40 * (((x * 2654435761) % 2**32) / 2**32)
        self.hyst = 0.03
        self.state = False

    def retune(self, delta):
        self.thr = float(np.clip(self.thr + delta, self.THR_MIN, self.THR_MAX))

    def step(self, intensity):
        if self.state and intensity < self.thr - self.hyst:
            self.state = False
        elif not self.state and intensity > self.thr + self.hyst:
            self.state = True
        return VCC if self.state else 0.0


class DriverA:
    """Randomized: parity of the sensor-noise LSBs in the detector-A ROI."""
    def step(self, roi):
        return VCC if int(np.bitwise_and(roi, 1).sum()) & 1 else 0.0


class IfThenLink:
    """RC node + NPN: IF V(C) > 0.7 V THEN output HIGH."""
    def __init__(self):
        self.vc = 0.0

    def step(self, v_drive, dt):
        self.vc += (v_drive - self.vc) * (1 - np.exp(-dt / (R_OHM * C_FARAD)))
        return VCC if self.vc > V_BE else 0.0


class TimeCodeLatch:
    """Latch holding the isolated number; one bit (MSB first) per clock slot."""
    def __init__(self, value, nbits):
        self.value, self.nbits = value, nbits

    def bit(self, slot):
        return (self.value >> (self.nbits - 1 - slot)) & 1


class DFlipFlop74HC74:
    """D flip-flop with both rails. q = 1-bit rail, qbar = 0-bit rail."""
    def __init__(self):
        self.q = 0

    def clock(self, d):
        self.q = int(d)

    @property
    def qbar(self):
        return 1 - self.q


def hi(v):
    return v > VCC / 2


# --------------------------------------------------------------------------- symmetry restoration
def balanced_gate(a, b):
    """OR . NOT(AND): OR (75 %) and its De Morgan dual AND (25 %) cancel -> XOR, 50 %."""
    return (a or b) and not (a and b)


class BiasMonitor:
    """Running estimates of P(1) for every node plus the adaptive threshold controller."""
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.p = collections.defaultdict(lambda: 0.5)
        self.word = collections.defaultdict(list)

    def update(self, **nodes):
        for k, v in nodes.items():
            self.p[k] += self.alpha * (int(v) - self.p[k])
            self.word[k].append(int(v))

    def word_report(self):
        rep = {k: float(np.mean(v)) for k, v in self.word.items()}
        self.word.clear()
        return rep


def adapt(driver_b, monitor, gain=0.02):
    """Drive P(B_out) to 0.5. Too many HIGH frames -> raise threshold, too few -> lower it.
    P(A xor B) = pA + pB - 2 pA pB equals 0.5 for any pA once pB = 0.5."""
    driver_b.retune(gain * (monitor.p["B_out"] - 0.5))


class Verifier:
    """Holds only a salted SHA-256 commitment to the hidden number, never the number itself."""
    def __init__(self, value, nbits, salt=None):
        self.nbits = nbits
        self.salt = salt or secrets.token_hex(8)
        self.commitment = self._h(value)

    def _h(self, value):
        return hashlib.sha256(f"{self.salt}:{value:0{self.nbits // 4}X}".encode()).hexdigest()

    def check(self, value):
        return secrets.compare_digest(self._h(value), self.commitment)


# --------------------------------------------------------------------------- drawing
def draw(frame, roiB, roiA, text_lines, traces, names):
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

    pw, rowh = max(w, 520), 22
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


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="fake camera (no hardware)")
    ap.add_argument("--headless", action="store_true", help="no window, print results")
    ap.add_argument("--bits", type=int, default=16, help="word length (multiple of 4)")
    ap.add_argument("--secret", default=None, help="hidden number, hex (default: random, never shown)")
    ap.add_argument("--reveal", action="store_true", help="print the hidden number after each check")
    ap.add_argument("--x", type=int, default=12345, help="problem instance x")
    ap.add_argument("--leak", choices=["none", "or", "and", "xor"], default="none",
                    help="let a gate output reach D (fault injection; default none)")
    ap.add_argument("--no-adapt", action="store_true", help="disable the threshold feedback loop")
    ap.add_argument("--runs", type=int, default=0, help="stop after N words (0 = forever)")
    args = ap.parse_args()

    nbits = args.bits - args.bits % 4
    mask = (1 << nbits) - 1
    secret = (int(args.secret, 16) if args.secret else secrets.randbits(nbits)) & mask
    hexw = nbits // 4
    verifier = Verifier(secret, nbits)

    cam = Camera(args.cam, args.synthetic)
    drvA, drvB, link = DriverA(), DriverB(args.x), IfThenLink()
    latch, ff = TimeCodeLatch(secret, nbits), DFlipFlop74HC74()
    mon = BiasMonitor()
    names = ["A", "B_out", "OR", "AND", "G", "Y", "D", "Q", "Qbar"]
    traces = collections.defaultdict(lambda: collections.deque(maxlen=600))

    slot, q_bits, qb_bits, y_bits, runs = 0, [], [], [], 0
    last_t = time.time()
    next_clk = last_t + CLK_PERIOD
    prev_I_B = None
    print(f"Hidden number committed: sha256(salt:number) = {verifier.commitment}")
    print(f"salt={verifier.salt}  bits={nbits}  initial driver-B thr={drvB.thr:.2f}  "
          f"adapt={'off' if args.no_adapt else 'on'}  leak={args.leak}")

    try:
        while True:
            gray, bgr = cam.read()
            now = time.time()
            dt, last_t = now - last_t, now
            h, w = gray.shape
            roiB_box = (0, 0, w // 2 - 4, h - 1)
            roiA_box = (w // 2 + 4, 0, w - 1, h - 1)
            roiB = gray[roiB_box[1]:roiB_box[3], roiB_box[0]:roiB_box[2]]
            roiA = gray[roiA_box[1]:roiA_box[3], roiA_box[0]:roiA_box[2]]

            I_B = float(roiB.mean()) / 255.0
            d_int = 0.0 if prev_I_B is None else I_B - prev_I_B
            prev_I_B = I_B

            vA = drvA.step(roiA)
            vB = drvB.step(I_B)
            vB_out = link.step(vB, dt)                      # IF V(C) > 0.7 THEN HIGH
            a, b = hi(vA), hi(vB_out)
            g_or, g_and = a or b, a and b
            g = balanced_gate(a, b)                         # symmetry-restored reference
            y = bool(latch.bit(slot))                       # time code forces HIGH
            leak_src = {"none": False, "or": g_or, "and": g_and, "xor": g}[args.leak]
            d = y or leak_src                               # protected: D = Y

            mon.update(A=a, B_out=b, OR=g_or, AND=g_and, G=g)
            if not args.no_adapt:
                adapt(drvB, mon)

            for k, v in zip(names, [a, b, g_or, g_and, g, y, d, ff.q, ff.qbar]):
                traces[k].append(int(v))

            if now >= next_clk:                             # 10 Hz clock edge
                next_clk += CLK_PERIOD
                ff.clock(d)
                q_bits.append(ff.q)
                qb_bits.append(ff.qbar)
                y_bits.append(int(y))
                slot += 1
                if slot == nbits:
                    val = int("".join(map(str, q_bits)), 2)
                    val_bar = int("".join(str(1 - x) for x in qb_bits), 2)   # decode from 0-rail
                    rails_ok = all(q != qb for q, qb in zip(q_bits, qb_bits))
                    zero_errs = sum(1 for q, yy in zip(q_bits, y_bits) if q and not yy)  # 0 read as 1
                    ok = verifier.check(val) and verifier.check(val_bar) and rails_ok
                    rep = mon.word_report()
                    print(f"SELECT = {val:0{hexw}X}  Qbar-decoded = {val_bar:0{hexw}X}  -> "
                          f"{'VERIFIED' if ok else 'MISMATCH'}  zero-bit errors={zero_errs}"
                          + (f"   (hidden = {secret:0{hexw}X})" if args.reveal else ""))
                    print(f"   P(1): A={rep['A']:.2f} B={rep['B_out']:.2f} "
                          f"OR={rep['OR']:.2f} AND={rep['AND']:.2f} G=A^B={rep['G']:.2f}   "
                          f"thr={drvB.thr:.3f}")
                    runs += 1
                    slot, q_bits, qb_bits, y_bits = 0, [], [], []
                    if args.runs and runs >= args.runs:
                        break

            if args.headless:
                time.sleep(0.01)
                continue

            import cv2
            sel = "".join(map(str, q_bits))
            sel_hex = f"{int(sel, 2):X}" if sel else "-"
            lines = [f"I_B={I_B:.2f} dI={d_int:+.3f}  Vc={link.vc:.2f}V  thr={drvB.thr:.3f}",
                     f"A={int(a)} B={int(b)} OR={int(g_or)} AND={int(g_and)} G={int(g)} Y={int(y)} "
                     f"D={int(d)} Q={ff.q} Qbar={ff.qbar}",
                     f"running P(1): OR={mon.p['OR']:.2f} AND={mon.p['AND']:.2f} "
                     f"G={mon.p['G']:.2f} B={mon.p['B_out']:.2f}",
                     f"slot {slot}/{nbits}  SELECT={sel}  hex={sel_hex}",
                     f"hidden: {'revealed ' + format(secret, '0%dX' % hexw) if args.reveal else '********'}"]
            cv2.imshow("optoelectronic sim (balanced)", draw(bgr, roiB_box, roiA_box, lines, traces, names))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                slot, q_bits, qb_bits, y_bits = 0, [], [], []
    finally:
        cam.release()
        if not args.headless:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
