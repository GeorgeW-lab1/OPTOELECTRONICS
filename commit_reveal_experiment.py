#!/usr/bin/env python3
"""
Webcam version of the commit-reveal protocol (Sections 5-7 of the paper).

Webcam ROI A / ROI B  = detectors A and B (viewport noise)
Software latch        = Y_k = (N >> (15-k)) & 1, D_k = Y_k OR (A_k AND B_k)
Driver A              = fires when standardised noise z_k > 1.0 sigma
Driver B              = passes with probability 0.8 (as in the paper's simulation)

Two arms, randomly interleaved window by window:
  before : N is drawn BEFORE the noise window opens (leak / shared cause can act)
  after  : N is drawn AFTER the noise window closes and its hash is committed
           (leak impossible; any surviving correlation is "anticipation" or a flaw)

Usage:
  python webcam_commit_reveal.py collect --windows 2000 --out run1.npz
  python webcam_commit_reveal.py collect --simulate --sim-leak 0.02 --windows 5000 --out sim.npz
  python webcam_commit_reveal.py analyze run1.npz
  python webcam_commit_reveal.py analyze run1.npz --inject 0.02   # positive control

Needs: numpy, opencv-python (OpenCV only for real capture).
"""
import argparse
import hashlib
import secrets
import sys
import time

import numpy as np

SLOTS = 16          # bits per window
T = 0.1             # slot length, 10 Hz clock
THRESH = 1.0        # Driver A threshold in sigma
P_B = 0.8           # Driver B pass probability
LEADS = (1, 2, 3, 4)


# ----------------------------------------------------------------- capture
class WebcamDetector:
    """Returns (A, B) brightness per slot from two ROIs."""

    def __init__(self, index, roi_a, roi_b):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            sys.exit("Cannot open webcam %d" % index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Lock exposure/gain so the camera does not "correct" the noise away.
        # Property values differ per driver; check with v4l2-ctl / your driver docs.
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual mode on V4L2
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.roi_a, self.roi_b = roi_a, roi_b
        for _ in range(30):          # warm up
            self.cap.read()

    def _roi_mean(self, g, roi):
        x, y, w, h = roi
        return float(g[y:y + h, x:x + w].mean())

    def read_slot(self, t_end):
        """Average ROI brightness over all frames that arrive before t_end."""
        a, b = [], []
        while True:
            ok, frame = self.cap.read()
            if not ok:
                sys.exit("Webcam read failed")
            g = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            a.append(self._roi_mean(g, self.roi_a))
            b.append(self._roi_mean(g, self.roi_b))
            if time.perf_counter() >= t_end:
                break
        return float(np.mean(a)), float(np.mean(b))

    def close(self):
        self.cap.release()


class SimDetector:
    """Poisson photoelectrons with a shared slow term (paper Eq. 5)."""

    def __init__(self, lam=10000.0, sigma_c=0.005, seed=None):
        self.lam, self.sc = lam, sigma_c
        self.rng = np.random.default_rng(seed)
        self.delta = 0.0
        self.pending_tilt = 0.0

    def new_window(self):
        self.delta = self.rng.normal(0, self.sc)

    def read_slot(self, t_end=None):
        mu = self.lam * (1 + self.delta)
        n = self.rng.poisson(mu, size=2).astype(float)
        a = n[0] + self.pending_tilt * np.sqrt(self.lam)
        self.pending_tilt = 0.0
        return a, n[1]

    def close(self):
        pass


# ------------------------------------------------------------------ collect
def draw_N():
    """Stand-in source of N. Replace with a hardware QRNG or public beacon
    that shares no power/clock/mains reference with the camera."""
    return secrets.randbits(SLOTS)


def bits(N):
    return np.array([(N >> (SLOTS - 1 - k)) & 1 for k in range(SLOTS)], dtype=np.int8)


def collect(args):
    rng = np.random.default_rng(secrets.randbits(64))
    if args.simulate:
        det = SimDetector(seed=secrets.randbits(32))
    else:
        det = WebcamDetector(args.camera, tuple(args.roi_a), tuple(args.roi_b))

    # Preregistration: hash of this script and the parameters
    with open(__file__, "rb") as f:
        prereg = hashlib.sha256(f.read() + repr((SLOTS, T, THRESH, P_B, LEADS, args.windows)).encode()).hexdigest()
    print("Preregistration hash (store it elsewhere NOW):", prereg)

    A = np.zeros((args.windows, SLOTS))
    B = np.zeros((args.windows, SLOTS))
    Ns = np.zeros(args.windows, dtype=np.int64)
    arm = np.zeros(args.windows, dtype=np.int8)       # 1 = after, 0 = before
    commits, nonces = [], []

    try:
        for w in range(args.windows):
            arm[w] = rng.integers(0, 2)
            if args.simulate:
                det.new_window()

            if arm[w] == 0:                            # BEFORE arm
                N = draw_N()
                y = bits(N)
            if arm[w] == 0 and args.simulate and args.sim_leak:
                pass                                   # tilt applied per slot below

            t0 = time.perf_counter() + 0.2
            for k in range(SLOTS):
                if args.simulate and arm[w] == 0 and args.sim_leak and k + 2 < SLOTS:
                    det.pending_tilt = 2 * args.sim_leak * (y[k + 2] - 0.5)
                t_end = t0 + (k + 1) * T
                A[w, k], B[w, k] = det.read_slot(t_end)
                if not args.simulate:
                    while time.perf_counter() < t_end:
                        pass
                elif False:
                    pass

            if arm[w] == 1:                            # AFTER arm
                nonce = secrets.token_bytes(16)
                digest = hashlib.sha256(nonce + A[w].tobytes() + B[w].tobytes()).hexdigest()
                commits.append(digest)                 # commitment made BEFORE N exists
                nonces.append(nonce)
                N = draw_N()
            else:
                commits.append("")
                nonces.append(b"")
            Ns[w] = N
            if (w + 1) % 50 == 0:
                print("window %d/%d" % (w + 1, args.windows))
    finally:
        det.close()

    np.savez(args.out, A=A, B=B, N=Ns, arm=arm, prereg=prereg,
             commits=np.array(commits), nonces=np.array(nonces, dtype=object))
    print("saved", args.out)


# ------------------------------------------------------------------ analyse
def standardise(x, window=400):
    """Remove slow drift (rolling mean over `window` slots) and scale to unit sigma."""
    flat = x.reshape(-1)
    k = np.ones(window) / window
    trend = np.convolve(np.pad(flat, (window // 2, window - window // 2 - 1), mode="edge"), k, mode="valid")
    r = flat - trend
    return (r / r.std()).reshape(x.shape)


def corr_at_leads(z, Y, Q, mask):
    out = {}
    for L in LEADS:
        zz = z[mask][:, :SLOTS - L].ravel()
        yy = Y[mask][:, L:].ravel().astype(float)
        qq = Q[mask][:, L:].ravel().astype(float)
        ry = np.corrcoef(zz, yy)[0, 1] if yy.std() > 0 else np.nan
        rq = np.corrcoef(zz, qq)[0, 1] if qq.std() > 0 else np.nan
        out[L] = (ry, rq, zz.size)
    return out


def analyse(args):
    d = np.load(args.file, allow_pickle=True)
    A, Bd, N, arm = d["A"], d["B"], d["N"], d["arm"]

    # Verify commitments (reveal step)
    ok = 0
    for w in np.where(arm == 1)[0]:
        h = hashlib.sha256(d["nonces"][w] + A[w].tobytes() + Bd[w].tobytes()).hexdigest()
        ok += (h == str(d["commits"][w]))
    print("commit-reveal check: %d/%d after-arm commitments verify" % (ok, (arm == 1).sum()))

    z = standardise(A - Bd) if args.balanced else standardise(A)
    rng = np.random.default_rng(12345)
    Y = np.array([bits(int(n)) for n in N])
    zB_pass = rng.random(Y.shape) < P_B
    Aflag = z > THRESH
    Q = (Y | (Aflag & zB_pass)).astype(np.int8)
    print("noise flips on zero slots: %.1f%%" % (100 * ((Q == 1) & (Y == 0)).sum() / (Y == 0).sum()))

    if args.inject:                                    # positive control (before arm only)
        z = z.copy()
        for w in np.where(arm == 0)[0]:
            z[w, :SLOTS - 2] += 2 * args.inject * (Y[w, 2:] - 0.5)
        print("positive control: injected eps=%.3f at L=+2 into before arm" % args.inject)

    results = {}
    for name, a in (("before", 0), ("after", 1)):
        mask = arm == a
        res = corr_at_leads(z, Y, Q, mask)
        results[name] = res
        n_w = mask.sum()
        print("\n%s arm: %d windows" % (name.upper(), n_w))
        print(" L   rho(z,Y)   5sigma   rho(z,Q)   pairs")
        for L in LEADS:
            ry, rq, n = res[L]
            print("%2d  %+8.4f  %7.4f  %+8.4f  %7d" % (L, ry, 5 / np.sqrt(n), rq, n))

    # Permutation null on rho(z,Y): shuffle N across windows within each arm
    print("\nPermutation test (shuffle N across windows), p-values for rho(z,Y):")
    for name, a in (("before", 0), ("after", 1)):
        idx = np.where(arm == a)[0]
        for L in LEADS:
            obs = results[name][L][0]
            null = []
            for _ in range(args.perms):
                perm = rng.permutation(idx)
                zz = z[idx][:, :SLOTS - L].ravel()
                yy = Y[perm][:, L:].ravel().astype(float)
                null.append(np.corrcoef(zz, yy)[0, 1])
            p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
            print("  %-6s L=%d  rho=%+.4f  p=%.4f" % (name, L, obs, p))

    print("\nContrast (paper Eq. 13):")
    for L in LEADS:
        rb, ra = results["before"][L][0], results["after"][L][0]
        print("  L=%d  rho_leak = before-after = %+.4f   rho_anticipation = after = %+.4f" % (L, rb - ra, ra))
    n_after = results["after"][1][2]
    print("\nSmallest after-arm coupling detectable at 5 sigma: eps_b ~ %.4f sigma" % (5 / np.sqrt(n_after)))
    print("Note: rho(z,Q) at nonzero lead is expected to be > 0 (shared-beam floor), not anticipation.")


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--windows", type=int, default=100)
    c.add_argument("--out", default="run.npz")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--roi-a", type=int, nargs=4, default=[100, 100, 64, 64], metavar=("X", "Y", "W", "H"))
    c.add_argument("--roi-b", type=int, nargs=4, default=[300, 100, 64, 64], metavar=("X", "Y", "W", "H"))
    c.add_argument("--simulate", action="store_true", help="no camera: Poisson + shared-term simulator")
    c.add_argument("--sim-leak", type=float, default=0.0, help="simulated leak (sigma) in before arm, 2 slots ahead")
    c.set_defaults(fn=collect)

    a = sub.add_parser("analyze")
    a.add_argument("file")
    a.add_argument("--perms", type=int, default=200)
    a.add_argument("--balanced", action="store_true", help="use A-B to cancel the common beam term")
    a.add_argument("--inject", type=float, default=0.0, help="positive-control tilt (sigma) at L=+2")
    a.set_defaults(fn=analyse)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
