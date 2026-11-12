#!/usr/bin/env python3
"""
photon_suite.py - everything in one file.

Simulated known-solution optoelectronic circuit read out through a Z-channel:
    secret bit 1 -> always observed 1
    secret bit 0 -> observed 1 with probability p (photon noise), else 0

  Level 1 (statistics): a bit ever seen as 0 is certainly 0. Estimate p from those.
  Level 2 (code):       enumerate candidates over the never-seen-0 set U by number
                        of flips (v2) or over 2^|U| with a cap (v1), and test each
                        against a verification oracle.

Oracles:
    - salted SHA-256 commitment   (decode mode)
    - N % candidate == 0          (factor mode)

IMPORTANT: this is a *decoder*, not a solver. The simulated latch is loaded with the
answer by the simulator. It cannot factor a number nobody has the factors of; factor
mode demonstrates exactly that (A works only if p is preloaded, B fails, C is real
factoring cost).

Usage:
    python photon_suite.py decode                       # 16-bit, v1 vs v2
    python photon_suite.py decode --bits 64 --trials 60
    python photon_suite.py decode --scaling --budget 5000
    python photon_suite.py factor                       # 32-bit primes, 64-bit N
    python photon_suite.py factor --bits 24
"""
import argparse
import hashlib
import itertools
import random
import secrets
import time
from math import comb, gcd

import numpy as np

# ============================================================== circuit constants
VCC, V_BE = 5.0, 0.7
TAU = 10e3 * 15e-6           # RC = 0.15 s
CLK = 0.100                  # 10 Hz clock (one bit slot per tick)
FRAME_DT = 0.005             # simulated frame period (200 fps)


# ============================================================== oracles
def commit(salt, v, nbits):
    return hashlib.sha256(f"{salt}:{v:0{(nbits + 3) // 4}X}".encode()).hexdigest()


# ============================================================== photon channel
class PhotonCircuit:
    """Simulated analog front end with Poisson photon arrivals.
    D = Y OR (A AND B_out), where Y is the latched secret bit."""

    def __init__(self, secret, nbits, rng, mean_photons=40.0, thr=0.45):
        self.secret, self.nbits, self.rng = secret, nbits, rng
        self.lam = mean_photons
        self.thr, self.hyst = thr, 0.03
        self.state, self.vc, self.t = False, 0.0, 0.0

    def _b_out(self, intensity):
        if self.state and intensity < self.thr - self.hyst:
            self.state = False
        elif not self.state and intensity > self.thr + self.hyst:
            self.state = True
        drive = VCC if self.state else 0.0
        self.vc += (drive - self.vc) * (1 - np.exp(-FRAME_DT / TAU))
        return self.vc > V_BE                       # NPN: Vc > 0.7 -> HIGH

    def word(self):
        """Clock out one nbits-wide SELECT word."""
        bits = 0
        frames_per_clk = int(round(CLK / FRAME_DT))
        for slot in range(self.nbits):
            d = 0
            for _ in range(frames_per_clk):
                self.t += FRAME_DT
                nB = self.rng.poisson(self.lam * (1 + 0.8 * np.sin(2 * np.pi * 0.4 * self.t)))
                nA = self.rng.poisson(self.lam * (1 + 0.8 * np.sin(2 * np.pi * 0.7 * self.t + 1)))
                a = bool(nA & 1)                    # shot-noise parity
                b = self._b_out(nB / (2 * self.lam))
                d = (a and b)                       # value at the sampling edge
            y = (self.secret >> (self.nbits - 1 - slot)) & 1
            bits = (bits << 1) | int(bool(y) or d)
        return bits


# ============================================================== level 1
class PhotonStatistics:
    """Per-bit Z-channel evidence."""

    def __init__(self, nbits):
        self.nbits, self.n = nbits, 0
        self.zeros = np.zeros(nbits, int)           # times each bit was seen as 0

    def update(self, word):
        self.n += 1
        for i in range(self.nbits):
            if not (word >> (self.nbits - 1 - i)) & 1:
                self.zeros[i] += 1

    def proven_zero(self):
        return self.zeros > 0

    def noise_rate(self):
        """Estimate p = P(observe 1 | secret 0) from bits proven zero (biased low early)."""
        z = self.proven_zero()
        if not z.any() or self.n == 0:
            return 0.5
        p = 1.0 - self.zeros[z].sum() / (self.n * z.sum())
        return float(np.clip(p, 0.01, 0.97))

    def posterior_one(self):
        p, post = self.noise_rate(), np.zeros(self.nbits)
        post[~self.proven_zero()] = 1.0 / (1.0 + p ** self.n)
        return post


# ============================================================== level 2 decoders
def decode_v1(stats, verify, nbits, max_ambiguous=12, thr=0.999):
    """Original: enumerate ambiguous bits (posterior < thr), capped at |U| <= 12."""
    post = stats.posterior_one()
    base = 0
    for i in range(nbits):
        if post[i] >= 0.5:
            base |= 1 << (nbits - 1 - i)
    amb = [i for i in range(nbits) if 0 < post[i] < thr]
    if len(amb) > max_ambiguous:
        return None, len(amb), 0
    tries = 0
    for k in range(len(amb) + 1):
        for flip in itertools.combinations(amb, k):
            cand = base
            for i in flip:
                cand &= ~(1 << (nbits - 1 - i))
            tries += 1
            if verify(cand):
                return cand, len(amb), tries
    return None, len(amb), tries


def decode_v2(stats, verify, nbits, budget=5000):
    """Improved: flip 0, 1, 2, ... of the never-seen-0 bits (1 -> 0); cost is
    sum_k C(|U|, k) bounded by a per-word hash budget instead of a 2^|U| cap."""
    U = [i for i in range(nbits) if stats.zeros[i] == 0]
    masks = [1 << (nbits - 1 - i) for i in U]
    base = sum(masks)
    tries = 0
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


# ============================================================== driver
def read_until_solved(secret_in_latch, nbits, verify, decoder, rng,
                      max_words=200, **kw):
    """Clock words, update statistics, try to decode after each word."""
    circ, stats = PhotonCircuit(secret_in_latch, nbits, rng), PhotonStatistics(nbits)
    total = 0
    for w in range(1, max_words + 1):
        stats.update(circ.word())
        cand, nU, tries = decoder(stats, verify, nbits, **kw)
        total += tries
        if cand is not None:
            return dict(cand=cand, words=w, tries=total, nU=nU, p=stats.noise_rate())
    return dict(cand=None, words=max_words, tries=total, nU=-1, p=stats.noise_rate())


def commitment_attack(nbits, seed, decoder, **kw):
    rng = np.random.default_rng(seed)
    secret = int(rng.integers(0, 1 << nbits)) if nbits < 63 else secrets.randbits(nbits)
    salt = secrets.token_hex(8)
    target = commit(salt, secret, nbits)
    verify = lambda c: commit(salt, c, nbits) == target
    r = read_until_solved(secret, nbits, verify, decoder, rng, **kw)
    r.update(ok=r["cand"] == secret, secret=secret)
    return r


def summarize(label, res):
    W = np.array([r["words"] for r in res])
    T = np.array([r["tries"] for r in res])
    print(f"  {label:<18} succ={np.mean([r['ok'] for r in res]):4.0%}  "
          f"words mean={W.mean():5.2f} max={W.max():3d}  "
          f"hashes mean={T.mean():9.1f} max={T.max():8d}")


def cmd_decode(a):
    sizes = [16, 32, 64, 128, 256] if a.scaling else [a.bits - a.bits % 4]
    for n in sizes:
        trials = a.trials if n <= 128 else max(20, a.trials // 3)
        print(f"n={n}  (brute force 2^{n} = {2.0 ** n:.2e}, trials={trials})")
        r1 = [commitment_attack(n, s, decode_v1) for s in range(trials)]
        r2 = [commitment_attack(n, s, decode_v2, budget=a.budget) for s in range(trials)]
        summarize("v1 (cap<=12)", r1)
        summarize(f"v2 (budget {a.budget})", r2)
        w1, w2 = (np.mean([r["words"] for r in x]) for x in (r1, r2))
        print(f"  -> words saved {w1 - w2:.2f} ({(w1 - w2) / w1:.0%}); hw time v2 ~ "
              f"{w2 * n * CLK:.0f}s vs v1 ~ {w1 * n * CLK:.0f}s")
        if a.reveal:
            r = r2[0]
            print(f"  e.g. secret={r['secret']:0{(n + 3) // 4}X} "
                  f"guess={(r['cand'] or 0):0{(n + 3) // 4}X}")
        print()


# ============================================================== factoring demo
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
    """Pollard-Brent rho: finds a nontrivial factor in ~sqrt(smallest prime factor)."""
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
                g = gcd(q, n)
                k += m
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
    bits = a.bits
    p, q = rand_prime(bits, rng), rand_prime(bits, rng)
    N = p * q
    verify = lambda c: 1 < c < N and N % c == 0
    print(f"N = {N}  ({N.bit_length()} bits)\n")

    r = read_until_solved(p, bits, verify, decode_v2, np.random.default_rng(0), max_words=60)
    print(f"A) latch preloaded with p : recovered={r['cand'] == p}  words={r['words']}"
          f"   <- only because p was put there")

    ok = 0
    for s in range(a.runs):
        r = read_until_solved(0, bits, verify, decode_v2, np.random.default_rng(s), max_words=60)
        ok += r["cand"] is not None
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


# ============================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("decode", help="commitment attack, v1 vs v2")
    d.add_argument("--bits", type=int, default=16)
    d.add_argument("--trials", type=int, default=20)
    d.add_argument("--budget", type=int, default=5000)
    d.add_argument("--scaling", action="store_true")
    d.add_argument("--reveal", action="store_true")
    d.set_defaults(fn=cmd_decode)

    f = sub.add_parser("factor", help="factoring reality check")
    f.add_argument("--bits", type=int, default=32, help="bits per prime")
    f.add_argument("--runs", type=int, default=20)
    f.add_argument("--rho-max", type=int, default=44, help="largest prime size for rho timing")
    f.set_defaults(fn=cmd_factor)

    a = ap.parse_args()
    if not a.cmd:
        a = ap.parse_args(["decode"])
    a.fn(a)


if __name__ == "__main__":
    main()
