#!/usr/bin/env python3
"""
photon_suite.py - v2 decoder only, single file.

Simulated known-solution optoelectronic circuit read through a Z-channel:
    secret bit 1 -> always observed 1
    secret bit 0 -> observed 1 with probability p (photon noise), else 0

  Level 1: a bit ever seen as 0 is certainly 0. U = bits never seen 0.
  Level 2: try 0, 1, 2, ... flips (1 -> 0) over U, checking each candidate with a
           verification oracle. Cost is sum_k C(|U|, k), bounded by a per-word
           hash budget (no 2^|U| cap, no posterior gate).

Oracles: salted SHA-256 commitment (decode) or N % candidate == 0 (factor).

NOTE: a decoder, not a solver. The simulated latch is loaded with the answer by the
simulator. It cannot factor a number nobody has the factors of; `factor` shows that.

Usage:
    python photon_suite.py decode [--bits 64] [--trials 60] [--budget 5000] [--scaling] [--reveal]
    python photon_suite.py factor [--bits 32] [--runs 20] [--rho-max 44]
"""
import argparse
import hashlib
import itertools
import random
import secrets
import time
from math import comb, gcd

import numpy as np

VCC, V_BE = 5.0, 0.7
TAU = 10e3 * 15e-6           # RC = 0.15 s
CLK = 0.100                  # 10 Hz clock, one bit slot per tick
FRAME_DT = 0.005             # 200 fps simulation


def commit(salt, v, nbits):
    return hashlib.sha256(f"{salt}:{v:0{(nbits + 3) // 4}X}".encode()).hexdigest()


class PhotonCircuit:
    """Poisson photon front end. D = Y OR (A AND B_out), Y = latched secret bit."""

    def __init__(self, secret, nbits, rng, mean_photons=40.0, thr=0.45):
        self.secret, self.nbits, self.rng = secret, nbits, rng
        self.lam, self.thr, self.hyst = mean_photons, thr, 0.03
        self.state, self.vc, self.t = False, 0.0, 0.0

    def _b_out(self, intensity):
        if self.state and intensity < self.thr - self.hyst:
            self.state = False
        elif not self.state and intensity > self.thr + self.hyst:
            self.state = True
        drive = VCC if self.state else 0.0
        self.vc += (drive - self.vc) * (1 - np.exp(-FRAME_DT / TAU))
        return self.vc > V_BE

    def word(self):
        bits, frames = 0, int(round(CLK / FRAME_DT))
        for slot in range(self.nbits):
            d = 0
            for _ in range(frames):
                self.t += FRAME_DT
                nB = self.rng.poisson(self.lam * (1 + 0.8 * np.sin(2 * np.pi * 0.4 * self.t)))
                nA = self.rng.poisson(self.lam * (1 + 0.8 * np.sin(2 * np.pi * 0.7 * self.t + 1)))
                d = bool(nA & 1) and self._b_out(nB / (2 * self.lam))
            y = (self.secret >> (self.nbits - 1 - slot)) & 1
            bits = (bits << 1) | int(bool(y) or d)
        return bits


class PhotonStatistics:
    """Level 1: count zeros per bit."""

    def __init__(self, nbits):
        self.nbits, self.n = nbits, 0
        self.zeros = np.zeros(nbits, int)

    def update(self, word):
        self.n += 1
        for i in range(self.nbits):
            if not (word >> (self.nbits - 1 - i)) & 1:
                self.zeros[i] += 1

    def noise_rate(self):
        """p = P(observe 1 | secret 0), from bits proven zero (reporting only)."""
        z = self.zeros > 0
        if not z.any() or self.n == 0:
            return 0.5
        return float(np.clip(1.0 - self.zeros[z].sum() / (self.n * z.sum()), 0.01, 0.97))


def decode_v2(stats, verify, nbits, budget=5000):
    """Level 2: flip 0,1,2,... bits of U (never seen 0) until verify() passes
    or the per-word hash budget would be exceeded."""
    U = [i for i in range(nbits) if stats.zeros[i] == 0]
    masks = [1 << (nbits - 1 - i) for i in U]
    base, tries = sum(masks), 0
    for k in range(len(U) + 1):
        if tries + comb(len(U), k) > budget:
            break
        for flip in itertools.combinations(masks, k):
            cand = base
            for m in flip:
                cand ^= m
            tries += 1
            if verify(cand):
                return cand, len(U), tries
    return None, len(U), tries


def read_until_solved(latch_value, nbits, verify, rng, max_words=200, budget=5000):
    circ, stats = PhotonCircuit(latch_value, nbits, rng), PhotonStatistics(nbits)
    total = 0
    for w in range(1, max_words + 1):
        stats.update(circ.word())
        cand, nU, tries = decode_v2(stats, verify, nbits, budget)
        total += tries
        if cand is not None:
            return dict(cand=cand, words=w, tries=total, nU=nU, p=stats.noise_rate())
    return dict(cand=None, words=max_words, tries=total, nU=-1, p=stats.noise_rate())


def sha256_target(salt: str, value: int, nbits: int) -> str:
    return commit(salt, value, nbits)


def sha256_crack(target: str, salt: str, nbits: int, budget: int,
                 start: int = 0, stop: int | None = None) -> int | None:
    """Brute-force SHA-256 preimage search over integers in [start, stop)."""
    if stop is None:
        stop = 1 << nbits

    width = (nbits + 3) // 4
    checked = 0

    for value in range(start, stop):
        if checked >= budget:
            return None
        digest = hashlib.sha256(f"{salt}:{value:0{width}X}".encode()).hexdigest()
        checked += 1
        if digest == target:
            return value

    return None


def cmd_sha256(a):
    """Solve a real SHA-256 commitment by brute force.

    Example:
        python photon_suite.py sha256 --bits 24 --target <hex digest> --salt deadbeef
    """
    if a.target:
        target = a.target.lower().removeprefix("0x")
        if len(target) != 64:
            raise SystemExit("target must be a 64-hex-character SHA-256 digest")
    else:
        # Generate a challenge internally, then solve it.
        rng = random.Random(a.seed)
        secret = rng.getrandbits(a.bits)
        salt = secrets.token_hex(8)
        target = sha256_target(salt, secret, a.bits)
        print(f"challenge: SHA-256(\"{salt}:<value>\") = {target}")
        print(f"secret space: {a.bits} bits ({1 << a.bits:,} candidates)")

    started = time.time()
    found = sha256_crack(target, a.salt, a.bits, a.budget,
                         start=a.start, stop=a.stop)

    if found is None:
        print(f"NOT FOUND within budget={a.budget:,} hashes "
              f"({time.time() - started:.3f}s)")
        return

    check = sha256_target(a.salt, found, a.bits)
    print(f"SOLVED: value={found} "
          f"({time.time() - started:.3f}s, verified={check == target})")


def cmd_decode(a):
    sizes = [16, 32, 64, 128, 256] if a.scaling else [a.bits - a.bits % 4]
    for n in sizes:
        trials = a.trials if n <= 128 else max(20, a.trials // 3)
        res = [commitment_attack(n, s, a.budget) for s in range(trials)]
        W = np.array([r["words"] for r in res])
        T = np.array([r["tries"] for r in res])
        print(f"bits={n:3d} trials={trials:3d} success={np.mean([r['ok'] for r in res]):4.0%}  "
              f"words mean={W.mean():5.2f} p90={np.percentile(W, 90):4.0f} max={W.max():3d}  "
              f"hashes mean={T.mean():8.1f} max={T.max():7d}  "
              f"hw~{W.mean() * n * CLK:5.0f}s  (brute force 2^{n} = {2.0 ** n:.2e})")
        if a.reveal:
            r = res[0]
            print(f"   e.g. secret={r['secret']:0{(n + 3) // 4}X} "
                  f"guess={(r['cand'] or 0):0{(n + 3) // 4}X}")


# ------------------------------------------------------------ factoring reality check
_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def rand_prime(bits, rng):
    while True:
        c = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(c):
            return c


def brent_rho(n, rng):
    if n % 2 == 0:
        return 2
    while True:
        y, c, m = rng.randrange(1, n), rng.randrange(1, n), 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g, k = gcd(q, n), k + m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
        if g != n:
            return g


def cmd_factor(a):
    rng = random.Random(7)
    p, q = rand_prime(a.bits, rng), rand_prime(a.bits, rng)
    N = p * q
    verify = lambda c: 1 < c < N and N % c == 0
    print(f"N = {N}  ({N.bit_length()} bits)\n")

    r = read_until_solved(p, a.bits, verify, np.random.default_rng(0), max_words=60)
    print(f"A) latch preloaded with p : recovered={r['cand'] == p}  words={r['words']}"
          f"   <- only because p was put there")
    ok = sum(read_until_solved(0, a.bits, verify, np.random.default_rng(s), max_words=60)
             ["cand"] is not None for s in range(a.runs))
    print(f"B) latch empty (real case): solved {ok}/{a.runs} runs after 60 words each")

    print("\nC) real factoring cost, Brent rho, balanced semiprimes (median of 5):")
    for b in range(24, a.rho_max + 1, 4):
        ts = []
        for t in range(5):
            rr = random.Random(100 * b + t)
            n = rand_prime(b, rr) * rand_prime(b, rr)
            t0 = time.time()
            brent_rho(n, rr)
            ts.append(time.time() - t0)
        print(f"   {2 * b:3d}-bit N ({len(str(n)):2d} digits): {np.median(ts):8.3f}s")
    print("\n   Beyond ~100 bits use ECM / quadratic sieve / GNFS (CADO-NFS, msieve, YAFU).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("decode")
    s = sub.add_parser("sha256", help="brute-force a SHA-256 commitment")
    s.add_argument("--bits", type=int, default=24,
                   help="secret bit width; search space is 2**bits")
    s.add_argument("--target", default="",
                   help="optional 64-hex SHA-256 digest to crack")
    s.add_argument("--salt", default="",
                   help="salt used when creating/checking the commitment")
    s.add_argument("--budget", type=int, default=1_000_000,
                   help="maximum SHA-256 computations")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int, default=0,
                   help="exclusive upper bound; defaults to 2**bits")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sha256)
    d.add_argument("--bits", type=int, default=16)
    d.add_argument("--trials", type=int, default=20)
    d.add_argument("--budget", type=int, default=5000)
    d.add_argument("--scaling", action="store_true")
    d.add_argument("--reveal", action="store_true")
    d.set_defaults(fn=cmd_decode)
    f = sub.add_parser("factor")
    f.add_argument("--bits", type=int, default=32)
    f.add_argument("--runs", type=int, default=20)
    f.add_argument("--rho-max", type=int, default=44)
    f.set_defaults(fn=cmd_factor)
    a = ap.parse_args()
    if not a.cmd:
        a = ap.parse_args(["decode"])
    a.fn(a)


if __name__ == "__main__":
    main()
