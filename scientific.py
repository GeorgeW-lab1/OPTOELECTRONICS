#!/usr/bin/env python3
"""
rp_camera_sim.py - RP-style optoelectronic decision circuit, driven by raw camera frames.

Signal chain (follows the supplied diagrams / RP-GHOST appendices):

  camera frame (96x128 gray, "raw" as far as the driver allows)
    -> detector A = right ROI mean, detector B = left ROI mean   (the delta-intensity pair)
    -> motion-energy mask E = |frame_t - frame_t-1| > thr, count x_t, digest d_t = SHA-256(E)
    -> instance x_n (odd 32-bit number built from the digest; 50% are forced prime)
    -> Driver A (randomized): r = 2 + SHA256(E, detA, detB) mod (x-3); A = 1 iff r is a
                              Miller-Rabin witness of compositeness   (A = 0 always on primes)
    -> Driver B (deterministic): c(x) = [x >= 4] charges RC node (tau) -> NPN emitter follower
                              VB = max(0, VC - 0.7); B = 1 iff VB >= 3.5 V (74HC logic HIGH)
    -> AND(a,b) -> 150 ms delay -> and_dly ;  OR(a,b) = leak/time-code path
    -> Y = time-code latch bit (isolated number, MSB first, one bit per clock)
    -> D = and_dly OR Y   ("leak" mode, as in the drawing)   or   D = Y   ("isolated" mode)
    -> 74HC74: Q <= D on each clock edge. SELECT = Q, observe = not Q
    -> SELECT bits are pooled into hex words and compared with the isolated number
  Independent probe: AND(a,b) with flip noise p, runs for half the duration (main runs 2x longer).
  H-instrumentation (A.9): nonce picked AFTER the digest from os.urandom, SHA-256d vs 2^-k target.

Usage:
  python rp_camera_sim.py --list
  python rp_camera_sim.py --device 0 --ticks 600
  python rp_camera_sim.py --synthetic --ticks 2000 --y-mode isolated
  python rp_camera_sim.py --device 0 --dark      # cover the lens: pure sensor noise
  python rp_camera_sim.py --device 0 --live      # realtime dashboard
Needs: numpy (+ opencv-python for a real camera).
"""
import argparse, csv, hashlib, math, os, random, struct, sys, time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

H, W = 96, 128          # 12,288 pixels, as in RP-GHOST Appendix A/D
VS, VBE, VIH = 5.0, 0.7, 3.5


# ----------------------------------------------------------------------------- camera
class RawCamera:
    """OpenCV capture with RGB conversion disabled where the driver allows it."""

    def __init__(self, device=0, width=640, height=480, fourcc=None, fps=30):
        if cv2 is None:
            sys.exit("opencv-python is not installed (pip install opencv-python), or use --synthetic")
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device, backend)
        if not self.cap.isOpened():
            sys.exit(f"cannot open camera {device}")
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)       # ask for unconverted frames
        # try to freeze auto controls so the noise floor is the sensor's, not the ISP's
        for prop, val in ((cv2.CAP_PROP_AUTO_EXPOSURE, 1), (cv2.CAP_PROP_AUTOFOCUS, 0),
                          (cv2.CAP_PROP_AUTO_WB, 0)):
            self.cap.set(prop, val)
        ok, f = self.cap.read()
        if not ok:
            sys.exit("camera opened but returned no frame")
        print(f"[camera] frame shape={f.shape} dtype={f.dtype} "
              f"fourcc={int(self.cap.get(cv2.CAP_PROP_FOURCC)):#x}")

    @staticmethod
    def to_gray(f):
        if f.dtype == np.uint16:                        # 10/12/16-bit raw: keep top 8 bits
            f = (f >> 8).astype(np.uint8) if f.max() > 4095 else (f >> 2).astype(np.uint8)
        if f.ndim == 3 and f.shape[2] == 2:             # YUYV packed: Y is channel 0
            return f[:, :, 0]
        if f.ndim == 3 and f.shape[2] == 3:
            return (0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]).astype(np.uint8)
        if f.ndim == 1 or (f.ndim == 2 and f.shape[0] == 1):  # flat raw buffer, best effort
            f = f.reshape(-1)
            return np.resize(f, (H, W)).astype(np.uint8)
        return f

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        g = self.to_gray(f)
        return cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)

    def close(self):
        self.cap.release()


class SyntheticSensor:
    """Poisson (shot-noise-limited) sensor, 1 e-/DN, with an optional moving blob."""

    def __init__(self, dark=False, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.dark = dark
        yy, xx = np.mgrid[0:H, 0:W]
        self.yy, self.xx = yy, xx

    def read(self):
        self.t += 1
        mean = 100.0 + 0.0 * self.xx
        if not self.dark:
            cx = W / 2 + 40 * math.sin(self.t / 9.0)
            cy = H / 2 + 25 * math.cos(self.t / 13.0)
            mean = mean + 80 * np.exp(-((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / 60.0)
        return np.clip(self.rng.poisson(mean), 0, 255).astype(np.uint8)

    def close(self):
        pass


# ----------------------------------------------------------------------------- number theory
SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    if n < 2:
        return False
    for p in SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in SMALL_PRIMES:                       # deterministic for n < 3.3e24
        if is_witness(a, n, d, s):
            return False
    return True


def is_witness(a, n, d=None, s=None):
    """True iff a proves n composite (Miller-Rabin). Never true for a prime n."""
    if d is None:
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
    y = pow(a, d, n)
    if y in (1, n - 1):
        return False
    for _ in range(s - 1):
        y = y * y % n
        if y == n - 1:
            return False
    return True


def next_prime(n):
    n |= 1
    while not is_prime(n):
        n += 2
    return n


# ----------------------------------------------------------------------------- circuit blocks
class TimeCodeLatch:
    """Isolated number, one bit per clock slot, MSB first."""

    def __init__(self, value, nbits):
        self.value, self.nbits = value & ((1 << nbits) - 1), nbits

    def bit(self, slot):
        return (self.value >> (self.nbits - 1 - (slot % self.nbits))) & 1


class DFlipFlop74HC74:
    def __init__(self):
        self.q = 0

    def clock(self, d):
        self.q = int(d)
        return self.q

    @property
    def qbar(self):
        return 1 - self.q


class DriverB:
    """c(x) charges C through R; NPN emitter follower; HIGH once VB >= 3.5 V."""

    def __init__(self, tau_ms, noise=True):
        self.tau = tau_ms / 1000.0
        self.noise = noise

    def run(self, x, t_charge_s):
        c = 1 if x >= 4 else 0
        vc = VS * (1 - math.exp(-t_charge_s / self.tau)) * c      # C is discharged before each trial
        if self.noise:
            vc += random.gauss(0, math.sqrt(1.38e-23 * 300 / 1e-6))  # kT/C for 1 uF (negligible)
        vb = max(0.0, vc - VBE)
        return int(vb >= VIH), vc, vb


class Delay:
    def __init__(self, ticks):
        self.buf = [0] * max(0, ticks)

    def step(self, v):
        if not self.buf:
            return v
        self.buf.append(v)
        return self.buf.pop(0)



# ----------------------------------------------------------------------------- noise decoder
def decode_select(sel, nbits, latch, delay_ticks, probe_meas, probe_err):
    """
    Reconstruct the isolated number from SELECT in 'leak' mode.
    Channel:  S_i = Y_j OR L_i   (j = i mod nbits, L_i = and_dly leak). Errors are only 0->1.
    1) Z-channel ML decoder: Y_j = AND over all repeats of slot j. One observed 0 proves Y_j = 0.
       Wrong only if the leak is 1 in EVERY repeat: P ~ p_leak^m.
    2) Decoded noise: where Y_j = 0 is known, S_i = L_i exactly, so the leak stream is recovered.
    3) Probe-assisted: the independent AND probe (delayed by the 150 ms stage) predicts which
       slots are leaking, so a leak-free slot reads the bit correctly in a single pass.
    """
    sel = np.array(sel, dtype=np.uint8)
    m = len(sel) // nbits
    if m < 1:
        return
    arr = sel[:m * nbits].reshape(m, nbits)
    truth = np.array([latch.bit(j) for j in range(nbits)], dtype=np.uint8)
    yhat = arr.all(axis=0).astype(np.uint8)

    raw_err = int((sel[:m * nbits] != np.tile(truth, m)).sum())
    dec_err = int((np.tile(yhat, m) != np.tile(truth, m)).sum())
    print("\n=== NOISE DECODER (Z-channel, repeat-AND) ===")
    print(f"raw SELECT bit errors      : {raw_err}/{m * nbits}")
    print(f"after decoding             : {dec_err}/{m * nbits}  "
          f"(number = 0x{int(''.join(map(str, yhat)), 2):0{nbits // 4}X}, {m} repeats)")

    # decoded noise = the leak stream, known wherever Y_j decoded as 0
    zero_cols = yhat == 0
    if zero_cols.any():
        leak = arr[:, zero_cols].astype(float)
        pl = float(leak.mean())
        flat = leak.T.reshape(-1)
        ac = float(np.corrcoef(flat[:-1], flat[1:])[0, 1]) if flat.std() > 0 else float("nan")
        print(f"decoded leak stream        : p_leak = {pl:.3f}, lag-1 autocorr = {ac:+.3f}, "
              f"{leak.size} samples (autocorr ~0 means the leak is memoryless noise)")
    else:
        pl = 0.5
        print("decoded leak stream        : no zero bits in the number, leak not observable")

    # convergence with number of repeats
    print("repeats -> bit errors in the decoded word (expected ~ zero_bits * p_leak^m)")
    nz = int(zero_cols.sum())
    for mm in sorted({1, 2, 3, 4, 6, 8, 12, 16, 32, 64, m} & set(range(1, m + 1))):
        e = int((arr[:mm].all(axis=0) != truth).sum())
        print(f"   m={mm:4d}: {e:2d}/{nbits}   expected {nz * pl ** mm:.3g}")
    if 0 < pl < 1:
        need = math.ceil(math.log(1e-6 / max(nz, 1)) / math.log(pl))
        print(f"repeats for <1e-6 word error: {need}  (= {need * nbits * 0.1:.1f} s at 100 ms)")

    # probe-assisted single-pass decode (probe covers the first ticks only)
    n_probe = len(probe_meas)
    idx = [i for i in range(len(sel)) if 0 <= i - delay_ticks < n_probe]
    if idx and 0 < pl < 1:
        i = np.array(idx)
        pm = np.array([probe_meas[k - delay_ticks] for k in idx])
        sbits = sel[i]
        tbits = np.array([latch.bit(k) for k in idx])
        clean = pm == 0                                    # probe says: no leak in this slot
        raw_e = float((sbits != tbits).mean())
        cl_e = float((sbits[clean] != tbits[clean]).mean()) if clean.any() else float("nan")
        q1 = pl * (1 - probe_err) / (pl * (1 - probe_err) + (1 - pl) * probe_err)
        q0 = pl * probe_err / (pl * probe_err + (1 - pl) * (1 - probe_err))
        print(f"probe-assisted single pass : {clean.mean():.1%} of slots flagged leak-free; "
              f"bit error there {cl_e:.3%} (all slots: {raw_e:.1%})")
        print(f"   P(leak | probe says 0) = {q0:.4f}, P(leak | probe says 1) = {q1:.4f}")
        print("   a slot read 1 with probe 0 is a 1 with confidence "
              f"{1 - q0:.4f}; erased slots (probe 1) are filled from other repeats")


# ----------------------------------------------------------------------------- live dashboard
class LiveView:
    """Realtime matplotlib dashboard: frame, mask, RC link, timeline, decoded word, convergence."""

    def __init__(self, nbits, latch, every, tau_ms, clock_ms, save=None):
        import matplotlib
        import matplotlib.pyplot as plt
        from collections import deque
        self.plt, self.every, self.save = plt, max(1, every), save
        self.nbits, self.tau, self.T = nbits, tau_ms / 1000.0, clock_ms / 1000.0
        self.truth = np.array([latch.bit(j) for j in range(nbits)], dtype=bool)
        self.ones = np.ones(nbits, bool)
        self.seen = np.zeros(nbits, int)
        self.ssum = np.zeros(nbits)
        self.conv, self.errhist = [], []
        self.win = 80
        names = ("Y", "leak", "A", "B", "Q")
        self.tl = {k: deque(maxlen=self.win) for k in names}
        self.raw_err = deque(maxlen=64)
        self.latch = latch
        self.interactive = matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo")

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("RP circuit - live") if self.interactive else None
        gs = self.fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)
        self.ax_f, self.ax_m, self.ax_rc = (self.fig.add_subplot(gs[0, i]) for i in range(3))
        self.ax_tl, self.ax_b, self.ax_c = (self.fig.add_subplot(gs[1, i]) for i in range(3))

        self.im_f = self.ax_f.imshow(np.zeros((H, W)), cmap="gray", vmin=0, vmax=255)
        self.ax_f.axvline(W / 2, color="orange", lw=1)
        self.ax_f.text(W * 0.25, 6, "B (left)", color="orange", ha="center", fontsize=8)
        self.ax_f.text(W * 0.75, 6, "A (right)", color="orange", ha="center", fontsize=8)
        self.ax_f.set_title("camera frame (detectors A/B)", fontsize=9); self.ax_f.axis("off")
        self.im_m = self.ax_m.imshow(np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
        self.ax_m.set_title("motion mask E -> SHA-256 digest", fontsize=9); self.ax_m.axis("off")
        self.t_m = self.ax_m.text(0, H + 8, "", color="black", fontsize=8, clip_on=False)

        t = np.linspace(0, self.T, 200)
        self.ax_rc.plot(t * 1000, VS * (1 - np.exp(-t / self.tau)), color="tab:blue")
        self.ax_rc.axhline(VBE, color="gray", ls="--", lw=1)
        self.ax_rc.axhline(VIH + VBE, color="tab:red", ls="--", lw=1)
        self.ax_rc.text(1, VBE + 0.1, "0.7 V NPN on", fontsize=7)
        self.ax_rc.text(1, VIH + VBE + 0.1, "4.2 V: VB = 3.5 V logic HIGH", fontsize=7, color="tab:red")
        self.dot_rc, = self.ax_rc.plot([], [], "o", color="k")
        self.t_rc = self.ax_rc.text(0.98, 0.05, "", transform=self.ax_rc.transAxes, ha="right", fontsize=8)
        self.ax_rc.set_title(f"IF/THEN link: V(C), tau={tau_ms:g} ms", fontsize=9)
        self.ax_rc.set_xlabel("ms"); self.ax_rc.set_ylim(0, 5.3)

        self.lines = {}
        for i, k in enumerate(names):
            ln, = self.ax_tl.plot([], [], drawstyle="steps-post", lw=1.4)
            self.lines[k] = (ln, i * 1.5)
        self.ax_tl.set_yticks([i * 1.5 + 0.5 for i in range(len(names))]); self.ax_tl.set_yticklabels(names)
        self.ax_tl.set_xlim(0, self.win); self.ax_tl.set_ylim(-0.3, len(names) * 1.5)
        self.ax_tl.set_title("last 80 clocks (Q = Y OR leak in 'leak' mode)", fontsize=9)

        self.ax_b.set_xlim(-0.5, nbits - 0.5); self.ax_b.set_ylim(-0.2, 3); self.ax_b.axis("off")
        self.t_tgt = [self.ax_b.text(j, 2.2, "", ha="center", va="center", fontsize=11,
                      family="monospace") for j in range(nbits)]
        self.t_dec = [self.ax_b.text(j, 1.2, "", ha="center", va="center", fontsize=11,
                      family="monospace", bbox=dict(boxstyle="round", fc="white")) for j in range(nbits)]
        self.ax_b.text(-0.5, 2.6, "isolated number (truth)", fontsize=8)
        self.ax_b.text(-0.5, 1.75, "decoded from SELECT", fontsize=8)
        self.t_hex = self.ax_b.text(nbits / 2 - 0.5, 0.5, "", ha="center", fontsize=11, family="monospace")
        self.ax_b.text(-0.5, 0.0, "green=certain  yellow=1 unconfirmed  red=wrong", fontsize=7)
        self.ax_b.set_title("online decoder", fontsize=9)
        for j in range(nbits):
            self.t_tgt[j].set_text(str(int(self.truth[j])))

        self.ax_c.set_title("decoded-word bit errors vs repeats", fontsize=9)
        self.ax_c.set_xlabel("repeats m"); self.ax_c.set_ylabel("bit errors")
        self.l_c, = self.ax_c.plot([], [], "o-", color="tab:green", label="measured")
        self.l_e, = self.ax_c.plot([], [], "--", color="gray", label="expected")
        self.ax_c.legend(fontsize=7, loc="center right")
        self.t_c = self.ax_c.text(0.98, 0.55, "", transform=self.ax_c.transAxes, fontsize=8, ha="right")
        self.fig.suptitle("RP optoelectronic circuit - live", fontsize=11)
        if self.interactive:
            plt.ion(); plt.show(block=False)

    def push(self, n, raw, E, A, B, Y, leak, Q, vc, vb, x, in_L, xcount, det_a, det_b):
        j = n % self.nbits
        self.ones[j] &= bool(Q); self.seen[j] += 1; self.ssum[j] += Q
        self.raw_err.append(int(Q != self.truth[j]))
        for k, v in zip(("Y", "leak", "A", "B", "Q"), (Y, leak, A, B, Q)):
            self.tl[k].append(v)
        obs = self.seen > 0
        err = int((self.ones[obs] != self.truth[obs]).sum())
        self.errhist.append(err)
        if j == self.nbits - 1:
            self.conv.append((n // self.nbits + 1, err))
        if n % self.every:
            return
        self.im_f.set_data(raw); self.im_m.set_data(E)
        self.t_m.set_text(f"count={xcount}  x={x}  {'composite' if in_L else 'prime'}")
        tm = self.T * 1000
        self.dot_rc.set_data([tm], [vc]); self.t_rc.set_text(f"V(C)={vc:.2f} V  VB={vb:.2f} V  B={B}")
        for k, (ln, off) in self.lines.items():
            d = np.array(self.tl[k], float)
            ln.set_data(np.arange(len(d)), d + off)
        zero = (~self.ones) & obs
        pl = float(self.ssum[zero].sum() / max(1, self.seen[zero].sum())) if zero.any() else 0.5
        for jj in range(self.nbits):
            if not obs[jj]:
                self.t_dec[jj].set_text("?"); self.t_dec[jj].get_bbox_patch().set_facecolor("lightgray"); continue
            d = bool(self.ones[jj]); self.t_dec[jj].set_text(str(int(d)))
            if d != self.truth[jj]:
                c = "tab:red"
            elif not d or (1 - pl ** self.seen[jj]) > 0.999:
                c = "lightgreen"
            else:
                c = "khaki"
            self.t_dec[jj].get_bbox_patch().set_facecolor(c)
        val = int("".join("1" if b else "0" for b in self.ones), 2)
        tgt = int("".join("1" if b else "0" for b in self.truth), 2)
        self.t_hex.set_text(f"decoded 0x{val:0{self.nbits // 4}X}   truth 0x{tgt:0{self.nbits // 4}X}   "
                            f"{'MATCH' if val == tgt else 'converging'}")
        if self.conv:
            ms = np.array([c[0] for c in self.conv]); es = np.array([c[1] for c in self.conv])
            self.l_c.set_data(ms, es)
            nz = int(zero.sum())
            self.l_e.set_data(ms, nz * pl ** ms)
            self.ax_c.set_xlim(0.5, max(ms.max(), 3) + 0.5)
            self.ax_c.set_ylim(-0.3, max(es.max(), 2) + 1)
        rate = sum(self.raw_err) / max(1, len(self.raw_err))
        self.t_c.set_text(f"p_leak={pl:.3f}  raw SELECT error (last 64)={rate:.1%}\n"
                          f"tick {n}  repeats {n // self.nbits + 1}  decoded errors {err}")
        self.fig.canvas.draw_idle()
        if self.interactive:
            self.fig.canvas.flush_events(); self.plt.pause(0.001)

    def finish(self):
        self.fig.canvas.draw()
        if self.save:
            self.fig.savefig(self.save, dpi=110)
            print(f"live snapshot saved to {self.save}")
        if self.interactive:
            self.plt.ioff(); self.plt.show()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="probe camera indices 0-5 and exit")
    ap.add_argument("--synthetic", action="store_true", help="simulated Poisson sensor, no camera")
    ap.add_argument("--dark", action="store_true", help="no scene motion (cover lens / flat field)")
    ap.add_argument("--fourcc", default=None, help="e.g. GREY, Y800, YUYV, MJPG")
    ap.add_argument("--ticks", type=int, default=400, help="probe length; main circuit runs 2x")
    ap.add_argument("--clock-ms", type=float, default=100.0)
    ap.add_argument("--tau-ms", type=float, default=50.0, help="RC node tau (diagram: 50 ms)")
    ap.add_argument("--delay-ms", type=float, default=150.0)
    ap.add_argument("--thr", type=int, default=6, help="motion threshold in DN")
    ap.add_argument("--y-mode", choices=("leak", "isolated"), default="leak")
    ap.add_argument("--number", type=lambda s: int(s, 0), default=0xA5C3)
    ap.add_argument("--nbits", type=int, default=16)
    ap.add_argument("--probe-err", type=float, default=0.01)
    ap.add_argument("--target-bits", type=int, default=4, help="k for SHA-256d H-probe")
    ap.add_argument("--realtime", action="store_true", help="pace to the clock (synthetic only)")
    ap.add_argument("--log", default=None, help="CSV log path")
    ap.add_argument("--no-decode", action="store_true", help="skip the noise decoder")
    ap.add_argument("--live", action="store_true", help="realtime dashboard (needs matplotlib)")
    ap.add_argument("--live-every", type=int, default=2, help="redraw every N clocks")
    ap.add_argument("--save-live", default=None, help="save final dashboard PNG")
    ap.add_argument("--fast", action="store_true", help="do not pace synthetic runs to the clock")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    if a.list:
        if cv2 is None:
            sys.exit("opencv-python not installed")
        for i in range(6):
            c = cv2.VideoCapture(i)
            print(f"index {i}: {'OK' if c.isOpened() else '-'}")
            c.release()
        return

    random.seed(a.seed)
    src = SyntheticSensor(a.dark, a.seed) if a.synthetic else RawCamera(a.device, fourcc=a.fourcc)
    T = a.clock_ms / 1000.0
    latch = TimeCodeLatch(a.number, a.nbits)
    ff = DFlipFlop74HC74()
    drvB = DriverB(a.tau_ms)
    delay = Delay(math.ceil(a.delay_ms / a.clock_ms))
    acc = 1                                            # accumulator: running AND of and_dly

    n_main = 2 * a.ticks
    prev = src.read().astype(np.int16)
    s1 = np.zeros((H, W)); s2 = np.zeros((H, W)); nf = 0     # Welford-free running sums (Fano)
    log = []
    sel_bits = []
    live = LiveView(a.nbits, latch, a.live_every, a.tau_ms, a.clock_ms, a.save_live) if a.live else None
    if live and a.synthetic and not a.fast:
        a.realtime = True
    st = dict(prime_acc=0, prime_n=0, comp_acc=0, comp_n=0, b_hi=0, probe_n=0, probe_bad=0,
              hits=0, hit_and_acc=0, acc_n=0)
    probe_meas = []
    trial_hist = []                                    # (in_L, and)
    t0 = time.time()

    for n in range(n_main):
        tick0 = time.time()
        raw = src.read()
        f = raw.astype(np.int16)
        s1 += raw; s2 += raw.astype(np.float64) ** 2; nf += 1

        # detectors: right ROI = A, left ROI = B
        det_a, det_b = float(raw[:, W // 2:].mean()), float(raw[:, :W // 2].mean())
        E = (np.abs(f - prev) > a.thr).astype(np.uint8)
        prev = f
        digest = hashlib.sha256(E.tobytes()).digest()
        xcount = int(E.sum())

        # instance x_n: odd 32-bit from digest; half of the frames forced prime
        x = (int.from_bytes(digest[8:12], "big") | 1 | (1 << 31))
        if digest[12] & 1:
            x = next_prime(x)
        in_L = not is_prime(x)

        # Driver A (randomized witness): r from the same frame (shared cause) + ROI signals
        h_r = hashlib.sha256(E.tobytes() + struct.pack("<dd", round(det_a, 3), round(det_b, 3))).digest()
        r = 2 + int.from_bytes(h_r, "big") % (x - 3)
        A = int(is_witness(r, x))

        # Driver B + IF/THEN link
        B, vc, vb = drvB.run(x, T)

        and_ab = A & B
        or_ab = A | B
        and_dly = delay.step(and_ab)
        acc &= and_dly

        Y = latch.bit(n)
        D = (and_dly | Y) if a.y_mode == "leak" else Y
        Q = ff.clock(D)
        sel_bits.append(Q)
        if live:
            live.push(n, raw, E, A, B, Y, and_dly, Q, vc, vb, x, in_L, xcount, det_a, det_b)

        # H-instrumentation: nonce is drawn AFTER the digest from an independent source
        nonce = os.urandom(8)
        hh = hashlib.sha256(hashlib.sha256(digest + nonce).digest()).digest()
        hit = int.from_bytes(hh, "big") < (1 << (256 - a.target_bits))
        st["hits"] += hit
        st["acc_n"] += 1
        st["hit_and_acc"] += hit * and_ab

        # statistics
        if n < a.ticks:                                # independent probe stops at duration
            meas = and_ab ^ (random.random() < a.probe_err)
            st["probe_n"] += 1
            probe_meas.append(int(meas))
            st["probe_bad"] += int(meas != and_ab)
        st["b_hi"] += B
        if in_L:
            st["comp_n"] += 1; st["comp_acc"] += and_ab
        else:
            st["prime_n"] += 1; st["prime_acc"] += and_ab
        trial_hist.append((in_L, and_ab))
        log.append((n, digest.hex()[:16], xcount, round(det_a, 2), round(det_b, 2), x, int(in_L),
                    r, A, B, round(vc, 3), round(vb, 3), and_ab, or_ab, and_dly, Y, Q, ff.qbar,
                    nonce.hex(), int(hit)))

        if n % 100 == 0:
            print(f"t={n:5d} x={x:<11d} L={int(in_L)} A={A} B={B} VB={vb:4.2f}V Y={Y} "
                  f"Q={Q} (count={xcount}, dA-dB={det_a - det_b:+.2f})")

        if a.realtime and a.synthetic:
            time.sleep(max(0, T - (time.time() - tick0)))

    src.close()
    if live:
        live.finish()

    # ------------------------------------------------------------------ report
    print("\n=== RESULTS ===")
    print(f"ticks: main {n_main}, probe {a.ticks}; wall {time.time() - t0:.1f}s; y-mode={a.y_mode}")
    print(f"Driver B reached logic HIGH: {st['b_hi'] / n_main:.1%} of trials "
          f"(tau={a.tau_ms} ms, clock={a.clock_ms} ms; need t >= 1.83*tau)")
    if st["prime_n"]:
        print(f"Soundness   : accepts on primes = {st['prime_acc']}/{st['prime_n']} (must be 0)")
    if st["comp_n"]:
        p = st["comp_acc"] / st["comp_n"]
        print(f"Completeness: p-hat on composites = {p:.3f} (RP needs >= 0.5; MR gives >= 0.75 when B=1)")
        comp = [acc_ for in_l, acc_ in trial_hist if in_l]
        for k in (1, 2, 5):
            fails = sum(1 for i in range(0, len(comp) - k + 1, k) if not any(comp[i:i + k]))
            groups = max(1, (len(comp) - k + 1 + k - 1) // k)
            print(f"   k={k}: empirical false-reject {fails / groups:.4f}   (bound (1-p)^k = {(1 - p) ** k:.4f})")
    print(f"Probe       : {st['probe_bad']}/{st['probe_n']} flipped readings "
          f"({st['probe_bad'] / max(1, st['probe_n']):.3%}, set p={a.probe_err})")

    # hex pooling of SELECT and reconstruction of the isolated number
    words = [sel_bits[i:i + a.nbits] for i in range(0, len(sel_bits) - a.nbits + 1, a.nbits)]
    hexs = ["".join(f"{int(''.join(map(str, w[i:i+4])), 2):X}" for i in range(0, a.nbits - a.nbits % 4, 4))
            for w in words]
    target = f"{a.number & ((1 << a.nbits) - 1):0{a.nbits // 4}X}"
    ok = sum(h == target for h in hexs)
    bit_err = sum(sel_bits[i] != latch.bit(i) for i in range(len(sel_bits)))
    print(f"Time code   : isolated number 0x{target}; SELECT pooled words matching = {ok}/{len(hexs)}; "
          f"bit errors {bit_err}/{len(sel_bits)}")
    print(f"   first words from SELECT: {' '.join(hexs[:6])}")
    if a.y_mode == "leak" and ok < len(hexs):
        print("   -> in 'leak' mode the delayed AND path ORs into D, so SELECT cannot reconstruct the "
              "number. Use --y-mode isolated to see exact reconstruction.")

    if a.y_mode == "leak" and not a.no_decode:
        decode_select(sel_bits, a.nbits, latch, delay.buf.__len__(), probe_meas, a.probe_err)

    # H-instrumentation
    n = st["acc_n"]; p0 = 2.0 ** -a.target_bits
    z = (st["hits"] - n * p0) / math.sqrt(n * p0 * (1 - p0))
    print(f"H-probe     : SHA-256d below 2^-{a.target_bits}: {st['hits']}/{n} "
          f"(chance {n * p0:.1f}), z={z:+.2f}  [nonce chosen after the frame digest]")

    # per-pixel Fano estimate (D.8 estimator), DN-domain
    if nf > 2:
        mean = s1 / nf
        var = (s2 - s1 ** 2 / nf) / (nf - 1)
        m = (mean > 50) & (mean < 230)
        if m.any():
            print(f"Fano (var/mean, DN domain) over {int(m.sum())} pixels: {np.mean(var[m] / mean[m]):.3f} "
                  f"(Poisson sensor at 1 e-/DN gives ~1; real cameras need the photon-transfer gain)")

    if a.log:
        with open(a.log, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow("tick digest16 count detA detB x in_L r A B VC VB and or and_dly Y Q Qbar nonce hit".split())
            w.writerows(log)
        print(f"log written to {a.log}")


if __name__ == "__main__":
    main()
