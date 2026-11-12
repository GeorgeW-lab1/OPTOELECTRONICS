#!/usr/bin/env python3
"""
Webcam-driven simulation of the optoelectronic circuit, with an attacker that
recovers the "hidden" number using only what leaks.

  IR detector A/B (two webcam ROIs) -> Driver A (randomized) / Driver B (deterministic)
  Driver B -> RC node C -> NPN "IF V(C) > 0.7 V THEN output HIGH"  (IF/THEN link)
  A and B_out -> OR / AND gates
  Y (time code, latched from an isolated number) forces D HIGH
  D = Y OR (A AND B_out)            [AND path only reaches D with --leak]
  74HC74 D flip-flop clocked at 10 Hz (T = 100 ms) -> SELECT bit
  SELECT bits are pooled into a word and compared with the commitment.

Attacks implemented
  1. AND-accumulation (statistical leak, needs --leak):
       SELECT = secret | noise   (noise can only ADD ones)
       acc &= SELECT  every word  -> acc converges to the secret.
       Each guess is tested against the SHA-256 commitment; stops when it matches.
  2. Offline brute force (--bruteforce):
       The commitment and salt are printed, and a 16-bit space is tiny.
       Uses nothing from the circuit, only commitment + salt.

Usage:
  python optoelectronic_sim_attack.py --leak                       # webcam 0, window
  python optoelectronic_sim_attack.py --leak --synthetic --headless --runs 40
  python optoelectronic_sim_attack.py --bruteforce --bits 16
  python optoelectronic_sim_attack.py --leak --secret A5C3 --reveal --synthetic --headless

Keys (window): q = quit, r = restart word.
Requires: numpy, opencv-python (opencv only for webcam / window)
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
            t = time.time() - self.t0
            h, w = 240, 320
            base = np.zeros((h, w), np.float32)
            base[:, : w // 2] = 110 + 90 * np.sin(2 * np.pi * 0.4 * t)       # B side
            base[:, w // 2:] = 110 + 90 * np.sin(2 * np.pi * 0.7 * t + 1.0)  # A side
            img = np.clip(base + self.rng.normal(0, 3, (h, w)), 0, 255).astype(np.uint8)
            bgr = np.dstack([img, img, img])
            return img, bgr
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
    """Deterministic: Schmitt comparator on detector-B intensity; threshold set by problem instance x."""
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
    """Randomized: parity of the sensor-noise LSBs in detector-A ROI."""
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
    """Latch holding the isolated number; outputs one bit (MSB first) per clock slot."""
    def __init__(self, value, nbits):
        self.value, self.nbits = value, nbits

    def bit(self, slot):
        return (self.value >> (self.nbits - 1 - slot)) & 1


class DFlipFlop74HC74:
    def __init__(self):
        self.q = 0

    def clock(self, d):
        self.q = int(d)

    @property
    def qbar(self):
        return 1 - self.q


def hi(v):
    return v > VCC / 2


# --------------------------------------------------------------------------- commitment
def commit_hash(salt, value, nbits):
    return hashlib.sha256(f"{salt}:{value:0{nbits // 4}X}".encode()).hexdigest()


class Verifier:
    """Holds only a salted SHA-256 commitment to the hidden number, never the number itself."""
    def __init__(self, value, nbits, salt=None):
        self.nbits = nbits
        self.salt = salt or secrets.token_hex(8)
        self.commitment = commit_hash(self.salt, value, nbits)

    def check(self, value):
        return secrets.compare_digest(commit_hash(self.salt, value, self.nbits), self.commitment)


# --------------------------------------------------------------------------- attacks
def brute_force(commitment, salt, nbits):
    """Attack 2: offline search of the whole space using only commitment + salt."""
    t0 = time.time()
    for v in range(1 << nbits):
        if secrets.compare_digest(commit_hash(salt, v, nbits), commitment):
            return v, time.time() - t0
    return None, time.time() - t0


class AndAccumulator:
    """Attack 1: noise only adds 1s, so AND-ing SELECT words strips it.

    Also tracks how often each bit position was 1, to show which bits are still
    uncertain (a bit that is 1 in every run so far could still be a noisy 0).
    """
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
        """Estimated chance the current guess is right, assuming the AND path
        fires with probability p per 0-bit (p estimated from bits already known to be 0)."""
        zero_pos = [i for i in range(self.nbits)
                    if not (self.acc >> (self.nbits - 1 - i)) & 1]
        if self.n == 0 or not zero_pos:
            return 0.0
        p = float(np.mean([self.ones[i] / self.n for i in zero_pos]))
        # a hidden 0 still showing as 1 in all n runs happens with prob p^n
        ones_in_acc = self.nbits - len(zero_pos)
        return (1 - p ** self.n) ** ones_in_acc


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

    names = ["A", "B_out", "OR", "AND", "Y", "D", "SELECT"]
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


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="USB webcam index")
    ap.add_argument("--synthetic", action="store_true", help="fake camera (no hardware)")
    ap.add_argument("--headless", action="store_true", help="no window, print results")
    ap.add_argument("--bits", type=int, default=16, help="word length (multiple of 4)")
    ap.add_argument("--secret", default=None,
                    help="hidden isolated number, hex (default: random, never shown)")
    ap.add_argument("--reveal", action="store_true", help="print the hidden number (for debugging)")
    ap.add_argument("--x", type=int, default=12345, help="problem instance x")
    ap.add_argument("--leak", action="store_true",
                    help="AND-gate output also feeds D (noise can corrupt 0-bits)")
    ap.add_argument("--bruteforce", action="store_true",
                    help="attack the commitment offline (salt + hash only), then exit")
    ap.add_argument("--runs", type=int, default=0, help="stop after N words (0 = until recovered)")
    args = ap.parse_args()

    nbits = max(4, args.bits - args.bits % 4)
    mask = (1 << nbits) - 1
    secret = (int(args.secret, 16) if args.secret else secrets.randbits(nbits)) & mask
    hexw = nbits // 4
    verifier = Verifier(secret, nbits)

    print(f"Hidden number committed: sha256(salt:number) = {verifier.commitment}")
    print(f"salt={verifier.salt}  bits={nbits}  driver-B threshold={DriverB(args.x).thr:.2f}")

    # ---- attack 2: brute force from commitment only
    if args.bruteforce:
        if nbits > 28:
            print(f"{nbits} bits = {1 << nbits:,} candidates; too large to brute force here.")
            return
        found, dt = brute_force(verifier.commitment, verifier.salt, nbits)
        print(f"BRUTE FORCE: recovered = {found:0{hexw}X} in {dt:.3f}s "
              f"({1 << nbits:,} candidates max)")
        if args.reveal:
            print(f"(hidden = {secret:0{hexw}X})")
        return

    if not args.leak:
        print("Note: without --leak, D == Y exactly, so the first SELECT word already "
              "equals the secret; the statistical leak is the AND path (--leak).")

    cam = Camera(args.cam, args.synthetic)
    drvA, drvB, link = DriverA(), DriverB(args.x), IfThenLink()
    latch, ff = TimeCodeLatch(secret, nbits), DFlipFlop74HC74()
    traces = collections.defaultdict(lambda: collections.deque(maxlen=600))
    attacker = AndAccumulator(nbits)

    slot, select_bits, runs = 0, [], 0
    recovered = None
    last_t = time.time()
    next_clk = last_t + CLK_PERIOD
    prev_I_B = None

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
            y = bool(latch.bit(slot))                       # time code forces HIGH
            d = y or (g_and and args.leak)

            for k, v in zip(["A", "B_out", "OR", "AND", "Y", "D", "SELECT"],
                            [a, b, g_or, g_and, y, d, ff.q]):
                traces[k].append(int(v))

            if now >= next_clk:                             # 10 Hz clock edge
                next_clk += CLK_PERIOD
                ff.clock(d)
                select_bits.append(ff.q)
                slot += 1
                if slot == nbits:
                    val = int("".join(map(str, select_bits)), 2)
                    attacker.update(val)                    # noise only adds 1s -> AND strips it
                    ok = verifier.check(attacker.acc)       # test guess against commitment
                    runs += 1
                    print(f"run {runs:3d}  SELECT={val:0{hexw}X}  ACC guess={attacker.acc:0{hexw}X}  "
                          f"conf~{attacker.confidence():.4f}  "
                          f"{'RECOVERED' if ok else 'not yet'}"
                          + (f"  (hidden={secret:0{hexw}X})" if args.reveal else ""))
                    slot, select_bits = 0, []
                    if ok:
                        recovered = attacker.acc
                        break
                    if args.runs and runs >= args.runs:
                        break

            if args.headless:
                time.sleep(0.01)
                continue

            import cv2
            sel = "".join(map(str, select_bits))
            sel_hex = f"{int(sel, 2):X}" if sel else "-"
            lines = [f"I_B={I_B:.2f} dI={d_int:+.3f}  Vc={link.vc:.2f}V  thr={drvB.thr:.2f}",
                     f"A={int(a)} Bout={int(b)} OR={int(g_or)} AND={int(g_and)} Y={int(y)} D={int(d)} Q={ff.q}",
                     f"slot {slot}/{nbits}  SELECT bits={sel}  hex={sel_hex}",
                     f"ACC guess={attacker.acc:0{hexw}X}  runs={runs}"]
            cv2.imshow("optoelectronic sim", draw(bgr, roiB_box, roiA_box, lines, traces))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                slot, select_bits = 0, []
    finally:
        cam.release()
        if not args.headless:
            import cv2
            cv2.destroyAllWindows()

    if recovered is not None:
        print(f"\nSECRET RECOVERED after {runs} words: {recovered:0{hexw}X}")
    else:
        print(f"\nNot recovered after {runs} words; best guess {attacker.acc:0{hexw}X}. "
              f"Run more words, or use --bruteforce.")


if __name__ == "__main__":
    main()
