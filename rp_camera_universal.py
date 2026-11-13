#!/usr/bin/env python3
"""
rp_camera_universal.py
======================
Camera-driven one-sided-error (RP-style) circuit, with a T-P symmetric
self-referential monitor, a pre-registered H-instrumentation log and the
two-level Z-channel recogniser, behind a universal USB-camera front end.

Chain per frame (one frame edge = one clock edge, as in RP-GHOST A.8):

    camera -> 128x96 gray -> digest d_t, motion mask E_t
    group x  = odd integer derived from the digest at group start   (instance)
    Driver A = Miller-Rabin witness test, r = 2 + SHA256(E_t) mod (x-3)
    Driver B = deterministic cheap check c(x) -> RC link -> NPN follower
    gates    : D = A AND B AND Y        (Q = A.B.Y, soundness holds for any r)
    74HC74   : Q latched on the frame edge, Qbar = observe
    k trials : accept if any allowed trial gave Q = 1 (set-only latch)
    T-P loop : antilinear norm-preserving feedback monitor (diagnostic only)
    H-log    : digest, count, witness, Q, post-frame nonce, SHA256d, outcome
    Z-chan   : optional Level-1 / Level-2 recogniser fed by camera bits

Nothing is fed back from Q/Qbar into the drivers (no path Q -> drivers), so
the nonce is always fixed after the digest and the soundness argument
P(Q=1 | x not in L) = 0 is preserved.

Usage
-----
  python rp_camera_universal.py --list                 # find cameras
  python rp_camera_universal.py                        # auto-detect camera
  python rp_camera_universal.py --source 1 --frames 3000 --k 5
  python rp_camera_universal.py --source /dev/video2
  python rp_camera_universal.py --source http://host/stream.mjpg
  python rp_camera_universal.py --synthetic            # no camera needed
  python rp_camera_universal.py --zchannel --preview --log run.csv

Requires: numpy; opencv-python for real cameras (falls back to synthetic).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import math
import os
import platform
import secrets
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:  # synthetic source still works
    cv2 = None


# =============================================================================
# 1. Universal camera front end
# =============================================================================
FRAME_W, FRAME_H = 128, 96  # 12,288 pixels, as in RP-GHOST Appendix A/D


def _quiet_opencv() -> None:
    if cv2 is None:
        return
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass


def _backends():
    """Capture backends to try, in order of preference for this OS."""
    if cv2 is None:
        return []
    system = platform.system()
    if system == "Windows":
        names = ["CAP_DSHOW", "CAP_MSMF", "CAP_ANY"]
    elif system == "Darwin":
        names = ["CAP_AVFOUNDATION", "CAP_ANY"]
    else:
        names = ["CAP_V4L2", "CAP_ANY"]
    return [(n, getattr(cv2, n)) for n in names if hasattr(cv2, n)]


def _try_open(src, backend, req_w, req_h, req_fps):
    cap = cv2.VideoCapture(src, backend)
    if not cap.isOpened():
        cap.release()
        return None
    if req_w:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    if req_h:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    if req_fps:
        cap.set(cv2.CAP_PROP_FPS, req_fps)
    for _ in range(15):  # warm-up: many UVC cams return junk first frames
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return cap
        time.sleep(0.03)
    cap.release()
    return None


def list_cameras(max_index: int = 10):
    found = []
    if cv2 is None:
        return found
    _quiet_opencv()
    for idx in range(max_index):
        for name, be in _backends():
            cap = _try_open(idx, be, 0, 0, 0)
            if cap is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append((idx, name, w, h))
                cap.release()
                break
    return found


class CameraSource:
    """Any USB camera index, device path, file or stream URL."""

    def __init__(self, source="auto", req_w=640, req_h=480, req_fps=30, max_index=10):
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        _quiet_opencv()
        self.cap, self.desc = None, ""
        candidates = []
        if str(source).lower() == "auto":
            candidates = [(i, f"index {i}") for i in range(max_index)]
            if platform.system() == "Linux":
                candidates += [(p, p) for p in sorted(glob.glob("/dev/video*"))]
        elif str(source).isdigit():
            candidates = [(int(source), f"index {source}")]
        else:
            candidates = [(source, str(source))]
        for src, label in candidates:
            for name, be in _backends():
                cap = _try_open(src, be, req_w, req_h, req_fps)
                if cap is not None:
                    self.cap, self.desc = cap, f"{label} via {name}"
                    return
        raise RuntimeError(f"no working camera found for source={source!r}")

    def read(self):
        for _ in range(30):  # tolerate transient dropouts
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            time.sleep(0.01)
        return None

    def close(self):
        if self.cap is not None:
            self.cap.release()


class SyntheticSource:
    """Poisson-noise 'sensor' with slow drift; used when no camera is present."""

    def __init__(self, fps=30.0):
        self.period = 1.0 / fps
        self.t0 = time.time()
        self.n = 0
        self.desc = "synthetic Poisson sensor"
        yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
        self.base = 90 + 50 * (xx / FRAME_W)

    def read(self):
        self.n += 1
        due = self.t0 + self.n * self.period
        delay = due - time.time()
        if delay > 0:
            time.sleep(delay)
        drift = 20 * math.sin(self.n / 40.0)
        lam = np.clip(self.base + drift, 1, 250)
        return np.clip(np.random.poisson(lam), 0, 255).astype(np.uint8)

    def close(self):
        pass


# =============================================================================
# 2. Number theory for Driver A / ground truth
# =============================================================================
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime_det(n: int) -> bool:
    """Deterministic Miller-Rabin, exact for n < 3.3e24."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL_PRIMES:
        y = pow(a, d, n)
        if y in (1, n - 1):
            continue
        for _ in range(s - 1):
            y = y * y % n
            if y == n - 1:
                break
        else:
            return False
    return True


def mr_witness(r: int, x: int) -> bool:
    """True iff r proves x composite. Never true for prime x, for any r."""
    d, s = x - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    y = pow(r, d, x)
    if y in (1, x - 1):
        return False
    for _ in range(s - 1):
        y = y * y % x
        if y == x - 1:
            return False
    return True


def derive_x(digest: bytes, xbits: int) -> int:
    v = int.from_bytes(digest[:8], "big") & ((1 << xbits) - 1)
    return v | (1 << (xbits - 1)) | 1  # odd, full width, >= 5


def cheap_check(x: int) -> int:
    """Driver B: deterministic, over-inclusive precondition.
    Must be 1 for every composite (x in L). It may also be 1 for primes; the
    AND with A keeps soundness. Trial division certifies small primes only."""
    if x < (1 << 20):
        i = 2
        while i * i <= x:
            if x % i == 0:
                return 1
            i += 1
        return 0  # provably prime
    return 1


# =============================================================================
# 3. Circuit blocks: RC/NPN link, time-code latch, 74HC74
# =============================================================================
class IfThenLink:
    """Capacitor C charged through R by B's drive; NPN emitter follower.
    V(C)=Vs(1-exp(-t/RC)), VB=max(0,VC-0.7), logic HIGH at VIH=3.5 V, which
    needs V(C) ~ 4.2 V, i.e. t = RC ln(Vs/(Vs-VIH-VBE)) ~ 1.83 RC."""

    VS, VBE, VIH = 5.0, 0.7, 3.5

    def __init__(self, rc_s: float):
        self.rc = rc_s
        self.vc = 0.0

    def t_logic(self) -> float:
        return self.rc * math.log(self.VS / (self.VS - self.VIH - self.VBE))

    def run(self, c_bit: int, frame_period_s: float) -> tuple[int, float]:
        self.vc = 0.0  # discharged before each trial
        self.vc = self.VS * (1.0 - math.exp(-frame_period_s / self.rc)) * c_bit
        vb = max(0.0, self.vc - self.VBE)
        return int(vb >= self.VIH), vb


class TimeCodeLatch:
    """Serialises `value` MSB first, one bit per clock slot. Y gates D."""

    def __init__(self, pattern: str = "1011"):
        self.nbits = len(pattern)
        self.value = int(pattern, 2)

    def bit(self, slot: int) -> int:
        s = slot % self.nbits
        return (self.value >> (self.nbits - 1 - s)) & 1


class DFlipFlop74HC74:
    def __init__(self):
        self.q = 0

    def clock(self, d: int) -> int:
        self.q = int(bool(d))
        return self.q

    @property
    def qbar(self) -> int:
        return 1 - self.q


# =============================================================================
# 4. T-P symmetric self-referential loop (corrected engine, as supplied)
# =============================================================================
class TPSymmetryEngine:
    def __init__(self, theta: float = 0.7, gain: float = 1.3):
        self.gain = gain
        self.P = np.fliplr(np.eye(2)).astype(np.complex128)  # parity
        c, s = np.cos(theta), np.sin(theta)
        self.K = np.array([[c, -s], [s, c]], dtype=np.complex128)  # coupling
        self.state = np.zeros((2, 1), dtype=np.complex128)

    def parity(self, x):
        return self.P @ x

    def time_reversal(self, x):
        return np.conj(x)  # antiunitary T

    def reaction(self, a):
        return self.K @ self.time_reversal(self.parity(a))

    def step(self, a):
        r = self.reaction(a)
        self.state = a + 1j * self.gain * r
        return self.state, r


def _same_ray(x, y, tol):
    return np.sqrt(max(0.0, 1.0 - abs(np.vdot(x, y)) ** 2)) < tol


class TPMonitor:
    """Runs the T-P loop alongside the circuit. The camera perturbs the loop
    state a little each frame; a static scene lets it settle to ||S|| = 1+g.
    Diagnostic only: it does not touch the drivers."""

    def __init__(self, gain=1.3, theta=0.7, eps=0.05):
        self.engine = TPSymmetryEngine(theta, gain)
        self.gain, self.eps = gain, eps
        self.a = None
        self.norm = 0.0
        self.settled = False

    def update(self, brightness: float, motion: float):
        v = np.array([[brightness], [1j * motion]], dtype=np.complex128)
        nv = np.linalg.norm(v)
        v = v / nv if nv > 0 else np.array([[1.0], [0.0]], dtype=np.complex128)
        if self.a is None:
            self.a = v
        seed = self.a + self.eps * v
        seed /= np.linalg.norm(seed)
        out, _ = self.engine.step(seed)
        self.norm = float(np.linalg.norm(out))
        nxt = out / self.norm
        self.settled = _same_ray(seed, nxt, 1e-3)
        self.a = nxt
        return self.norm


# =============================================================================
# 5. Two-level Z-channel recogniser (Level 1 support set, Level 2 Hamming ball)
# =============================================================================
class ZChannel:
    """Camera blocks give R_i(t). Observation (dual polarity, see note):
         X_i = 1 if K_i = 0 else R_i
    U_m = indices never seen as 0. All true zeros stay in U_m, true ones
    leave U_m with probability 1-p each cycle. Let Z = NOT K. Then Z <= B_m
    bitwise (the paper's 'K <= B_m' holds for the *complement* under this
    polarity), so candidates are Z = B_m XOR e_F, F in U_m, K = NOT Z,
    enumerated by increasing |F| and checked against C = SHA256(salt || K).

    NOTE: K is generated in software here, so this is a simulation of the
    channel with camera-derived R_i, not recovery from a physical secret.
    """

    def __init__(self, n=64, bmax=20000):
        self.n, self.bmax = n, bmax
        self.full = (1 << n) - 1
        self.K = secrets.randbits(n)
        self.salt = os.urandom(16)
        self._nb = (n + 7) // 8
        self.C = hashlib.sha256(self.salt + self.K.to_bytes(self._nb, "big")).digest()
        self.U = self.full
        self.m = 0
        self.T_total = 0
        self.solved_at = None
        self.recovered = None
        self.ema = None
        self.ones_seen = 0
        self.bits_seen = 0
        rows = 8
        self.grid = (rows, n // rows)
        self.proof = None
    def _bits(self, gray: np.ndarray) -> int:
        rows, cols = self.grid
        ye = np.linspace(0, gray.shape[0], rows + 1).astype(int)
        xe = np.linspace(0, gray.shape[1], cols + 1).astype(int)
        blocks = np.array([[gray[ye[i]:ye[i + 1], xe[j]:xe[j + 1]].mean()
                            for j in range(cols)] for i in range(rows)]).ravel()
        if self.ema is None:
            self.ema = blocks.copy()
        r = blocks > self.ema
        self.ema = 0.9 * self.ema + 0.1 * blocks
        mask = 0
        for i, bit in enumerate(r):
            if bit:
                mask |= 1 << i
        self.ones_seen += int(r.sum())
        self.bits_seen += r.size
        return mask

    def step(self, gray: np.ndarray):
        if self.solved_at is not None:
            return
        R = self._bits(gray)
        X = ((~self.K) & self.full) | (self.K & R)
        self.U &= X
        self.m += 1
        if self.m >= 3:
            self._decode()

    def _decode(self):
        idx = [i for i in range(self.n) if (self.U >> i) & 1]
        u, spent = len(idx), 0

        for k in range(0, u + 1):
            layer = math.comb(u, k)
            if spent + layer > self.bmax:
                return  # search budget exhausted

            for F in itertools.combinations(idx, k):
                z = self.U
                for i in F:
                    z ^= 1 << i

                candidate = (~z) & self.full
                spent += 1
                self.T_total += 1

                digest = hashlib.sha256(
                    self.salt + candidate.to_bytes(self._nb, "big")
                ).digest()

                if digest == self.C:
                    self.solved_at = self.m
                    self.recovered = candidate
                    self.proof = {
                        "hidden_number": candidate,
                        "binary": format(candidate, f"0{self.n}b"),
                        "salt_hex": self.salt.hex(),
                        "commitment_hex": self.C.hex(),
                        "candidate_digest_hex": digest.hex(),
                        "hash_match": digest == self.C,
                        "independent_recheck": (
                            hashlib.sha256(
                                self.salt
                                + candidate.to_bytes(self._nb, "big")
                            ).digest()
                            == self.C
                        ),
                        "frames_used": self.m,
                        "candidates_tested": self.T_total,
                        "unknown_positions_at_recovery": u,
                        "corrections_applied": k,
                    }
                    return

    @property
    def p_hat(self):
        return self.ones_seen / self.bits_seen if self.bits_seen else float("nan")


# =============================================================================
# 6. Main run loop
# =============================================================================
def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def leading_zero_bits(h: bytes) -> int:
    v = int.from_bytes(h, "big")
    return 256 - v.bit_length()


def run(args):
    # ---- source
    src = None
    if not args.synthetic:
        try:
            src = CameraSource(args.source, args.width, args.height, args.fps)
        except Exception as exc:  # noqa: BLE001
            print(f"[camera] {exc}; falling back to synthetic source", file=sys.stderr)
    if src is None:
        src = SyntheticSource(args.fps)
    print(f"[source] {src.desc}")

    link = IfThenLink(args.rc_ms / 1000.0)
    latch = TimeCodeLatch(args.ybits)
    ff = DFlipFlop74HC74()
    ff_loop = DFlipFlop74HC74()  # optional Qbar -> D demo (never feeds drivers)
    tp = TPMonitor()
    zc = ZChannel(args.nmodes) if args.zchannel else None

    writer, logf = None, None
    if args.log:
        logf = open(args.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["t", "digest", "count", "x", "in_L", "r", "A", "B", "Y", "Q",
                         "Qbar", "nonce", "hash", "lz_bits", "hit", "tp_norm", "mean"])

    # ---- statistics
    S = dict(frames=0, groups=0, accepts=0, in_L_groups=0, false_rej=0,
             sound_viol=0, A_on_L=0, A_on_L_n=0, ones=0, bits=0, hits=0,
             timing_fail=0, repeats=0)
    feat, outc = [], []
    prev, prev_digest = None, None
    # group state
    x = in_L = None
    trials = 0
    accepted = 0
    t_prev = time.time()
    dt_ema = 1.0 / args.fps
    stop = False

    try:
        while not stop and (args.frames <= 0 or S["frames"] < args.frames):
            g = src.read()
            if g is None:
                print("[source] stream ended", file=sys.stderr)
                break
            now = time.time()
            dt_ema = 0.95 * dt_ema + 0.05 * max(1e-4, now - t_prev)
            t_prev = now
            t = S["frames"]

            # ---- frame features (digest, motion mask E, count)
            digest = hashlib.sha256(g.tobytes()).digest()
            if digest == prev_digest:
                S["repeats"] += 1
            prev_digest = digest
            if prev is None:
                diff = g
            else:
                diff = np.abs(g.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
            E = (diff >> 1).tobytes()
            count = int((diff >= 4).sum())
            mean = float(g.mean())
            prev = g

            # ---- group start: new instance x
            if x is None:
                x = derive_x(digest, args.xbits)
                in_L = not is_prime_det(x)  # ground truth: composite
                trials, accepted = 0, 0

            # ---- drivers
            r = 2 + int.from_bytes(hashlib.sha256(E).digest(), "big") % (x - 3)
            A = int(mr_witness(r, x))
            c = cheap_check(x)
            B, _vb = link.run(c, dt_ema)
            if c and not B:
                S["timing_fail"] += 1
            Y = latch.bit(t)

            # ---- gates + flip-flop (Q = A.B.Y), set-only accumulator
            D = A & B & Y
            Q = ff.clock(D)
            accepted |= Q
            if args.loop_qbar:
                ff_loop.clock(ff_loop.qbar)  # x = NOT x, period 2, diagnostic

            if in_L:
                S["A_on_L"] += A
                S["A_on_L_n"] += 1

            # ---- H-instrumentation: nonce chosen AFTER the digest exists
            nonce = secrets.randbits(64)
            h = sha256d(digest + nonce.to_bytes(8, "big"))
            lz = leading_zero_bits(h)
            hit = int(lz >= args.target_bits)
            bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
            S["ones"] += int(bits.sum())
            S["bits"] += bits.size
            S["hits"] += hit
            feat.append(mean)
            outc.append(hit)

            tpn = tp.update(mean / 255.0, count / (FRAME_W * FRAME_H))
            if zc is not None:
                zc.step(g)

            if writer:
                writer.writerow([t, digest.hex()[:16], count, x, int(in_L), r, A, B, Y, Q,
                                 ff.qbar, nonce, h.hex()[:16], lz, hit, f"{tpn:.5f}",
                                 f"{mean:.3f}"])

            # ---- k allowed trials make one RP decision
            if Y:
                trials += 1
            if trials >= args.k:
                S["groups"] += 1
                S["accepts"] += accepted
                if in_L:
                    S["in_L_groups"] += 1
                    if not accepted:
                        S["false_rej"] += 1
                elif accepted:
                    S["sound_viol"] += 1  # must stay 0
                if args.verbose:
                    print(f"group {S['groups']:5d} x={x:<8d} {'composite' if in_L else 'prime    '}"
                          f" -> {'ACCEPT' if accepted else 'reject'}")
                x = None

            S["frames"] += 1

            if S["frames"] % args.report == 0:
                p = S["A_on_L"] / S["A_on_L_n"] if S["A_on_L_n"] else float("nan")
                zmsg = ""
                if zc is not None:
                    zmsg = (f" | Z m={zc.m} |U|={bin(zc.U).count('1')}"
                            f" solved={'yes@%d' % zc.solved_at if zc.solved_at else 'no'}")
                print(f"[t={S['frames']:6d}] fps~{1 / dt_ema:5.1f} p^={p:.3f} "
                      f"sound_viol={S['sound_viol']} ||S||={tpn:.4f}{zmsg}")

            if args.preview and cv2 is not None:
                try:
                    view = cv2.resize(g, (FRAME_W * 4, FRAME_H * 4), interpolation=cv2.INTER_NEAREST)
                    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
                    cv2.putText(view, f"x={x} A={A} B={B} Y={Y} Q={Q} ||S||={tpn:.3f}",
                                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    cv2.imshow("rp_camera_universal (q to quit)", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop = True
                except cv2.error:
                    args.preview = False
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if logf:
            logf.close()
        if cv2 is not None and args.preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    summarise(S, feat, outc, args, link, tp, zc, dt_ema)


def summarise(S, feat, outc, args, link, tp, zc, dt):
    print("\n=========== SUMMARY (estimators for A.9 / C.7) ===========")
    print(f"frames={S['frames']}  groups={S['groups']}  k={args.k}  Y={args.ybits}"
          f"  measured fps~{1 / dt:.1f}")
    print(f"link: 1.83RC = {link.t_logic() * 1000:.2f} ms vs frame {dt * 1000:.2f} ms;"
          f" timing failures = {S['timing_fail']}")
    print(f"static/repeated frames (r not independent): {S['repeats']}")
    print(f"SOUNDNESS: accepts on primes = {S['sound_viol']}  (must be 0)")
    if S["A_on_L_n"]:
        p = S["A_on_L"] / S["A_on_L_n"]
        print(f"(v) witness rate p^ on x in L = {p:.4f} over {S['A_on_L_n']} frames"
              f"  (Miller-Rabin theory: >= 0.75)")
        if S["in_L_groups"]:
            print(f"    false-reject groups: {S['false_rej']}/{S['in_L_groups']} = "
                  f"{S['false_rej'] / S['in_L_groups']:.4f}   bound (1-p^)^k ~ {(1 - p) ** args.k:.2e}")
    if S["bits"]:
        eps = S["ones"] / S["bits"] - 0.5
        z = eps / (0.5 / math.sqrt(S["bits"]))
        print(f"(iii) hash-bit bias eps = {eps:+.2e}  (z = {z:+.2f}; 5-sigma needs |z|>5)")
    if len(set(outc)) > 1:
        rho = float(np.corrcoef(feat, outc)[0, 1])
        print(f"(iv) corr(frame mean, hash<target) rho = {rho:+.4f};"
              f" 5-sigma detectable |rho| > {5 / math.sqrt(len(outc)):.4f}")
    else:
        print(f"(iv) corr undefined (hits={S['hits']}; lower --target-bits for more hits)")
    print(f"T-P monitor: ||S|| = {tp.norm:.4f}  (fixed point 1+g = 2.3000)  settled={tp.settled}")
    if zc is not None:
        print(f"Z-channel: m={zc.m} cycles, p^={zc.p_hat:.3f}, hash evaluations={zc.T_total}, "
              + (f"RECOVERED at m={zc.solved_at}, correct={zc.recovered == zc.K}"
                 if zc.solved_at else f"not recovered, unresolved |U|={bin(zc.U).count('1')}"))
        print("  (software K; camera bits only supply R_i(t). Not a physical secret recovery.)")
    print("A null result at these sample sizes is informative: it bounds g, eps and rho.")
    if zc is not None:
        print(f"Z-channel: m={zc.m} cycles, p^={zc.p_hat:.3f}, "
              f"hash evaluations={zc.T_total}")

        if zc.proof:
            p = zc.proof
            print("\n---------- HIDDEN NUMBER RECOVERED ----------")
            print(f"hidden number : {p['hidden_number']}")
            print(f"binary        : {p['binary']}")
            print(f"frames used   : {p['frames_used']}")
            print(f"candidates    : {p['candidates_tested']}")
            print(f"unknown bits  : {p['unknown_positions_at_recovery']}")
            print(f"corrections   : {p['corrections_applied']}")
            print(f"salt          : {p['salt_hex']}")
            print(f"commitment    : {p['commitment_hex']}")
            print(f"candidate hash: {p['candidate_digest_hex']}")
            print(f"hash match    : {p['hash_match']}")
            print(f"independent   : {p['independent_recheck']}")
            print("PROOF: candidate hashes to the original commitment.")
        else:
            print(f"not recovered; unresolved |U|="
                  f"{bin(zc.U).count('1')}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", help="auto | camera index | /dev/videoN | file | URL")
    ap.add_argument("--list", action="store_true", help="list working camera indices and exit")
    ap.add_argument("--synthetic", action="store_true", help="use the built-in Poisson sensor")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=2000, help="0 = run until Ctrl-C / q")
    ap.add_argument("--k", type=int, default=5, help="allowed trials per RP decision")
    ap.add_argument("--ybits", default="1011", help="time-code pattern; 1 = bin allowed")
    ap.add_argument("--xbits", type=int, default=24, help="bit width of instance x")
    ap.add_argument("--rc-ms", type=float, default=3.0, help="RC of the IF/THEN link, ms")
    ap.add_argument("--target-bits", type=int, default=8, help="leading zero bits for a hash 'hit'")
    ap.add_argument("--zchannel", action="store_true", help="run the two-level Z-channel recogniser")
    ap.add_argument("--nmodes", type=int, default=64)
    ap.add_argument("--loop-qbar", action="store_true", help="Qbar->D period-2 demo (diagnostic)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--log", default="", help="CSV log path for the H-instrumentation record")
    ap.add_argument("--report", type=int, default=250)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.xbits < 4:
        ap.error("--xbits must be >= 4")
    if set(args.ybits) - {"0", "1"} or "1" not in args.ybits:
        ap.error("--ybits must be a binary string containing at least one 1")

    if args.list:
        cams = list_cameras()
        if not cams:
            print("no cameras found")
        for idx, be, w, h in cams:
            print(f"index {idx}: {w}x{h} via {be}")
        return
    run(args)


if __name__ == "__main__":
    main()
