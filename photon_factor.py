#!/usr/bin/env python3
"""
Two-level decoding of a known-solution optoelectronic circuit.

Circuit (causal, left to right):
    photons -> detector A (shot-noise parity)  \
    photons -> detector B -> comparator -> RC -> NPN  (B_out)  -> AND -> OR -> D -> 74HC74 -> SELECT
    isolated number -> latch -> Y ---------------------------------------^

    D = Y OR (A AND B_out)

The circuit already "knows" the solution: Y carries it. Photon noise can only
OR extra 1s onto the output, so the channel is a Z-channel:
    secret bit 1 -> always observed 1
    secret bit 0 -> observed 1 with probability p_i, else 0

LEVEL 1 (photon-statistics layer, per bit)
    - A observed 0 is certain evidence of secret 0.
    - Estimate the noise rate p from bits already proven zero.
    - For bits never seen 0, posterior P(secret=1 | n all-ones) = 1/(1 + p^n)
      (uniform prior).

LEVEL 2 (code layer, across bits)
    - Bits with posterior below a threshold form an ambiguous set U.
    - Enumerate the 2^|U| candidates (|U| is small, ~log n), most likely first,
      and test each against the salted SHA-256 commitment (verification oracle).

Cost: ~ log(n)/log(1/p) words and 2^|U| hash checks, versus 2^n for brute force.

Usage:
    python photon_two_level.py                       # 16-bit demo
    python photon_two_level.py --bits 64 --trials 200
    python photon_two_level.py --scaling
"""
import argparse
import hashlib
import itertools
import secrets
import numpy as np

VCC, V_BE = 5.0, 0.7
TAU = 10e3 * 15e-6           # RC = 0.15 s
CLK = 0.100               
FRAME_DT = 0.005             # simulated frame period (200 fps)


# ------------------------------------------------------------------ commitment
def commit(salt, v, nbits):
    return hashlib.sha256(f"{salt}:{v:0{nbits // 4}X}".encode()).hexdigest()


# ------------------------------------------------------------------ photon channel
class PhotonCircuit:
    """Simulated analog front end with Poisson photon arrivals."""
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
        return self.vc > V_BE                      # NPN: IF Vc > 0.7 THEN HIGH

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
                a = bool(nA & 1)                   # shot-noise parity (randomized driver)
                b = self._b_out(nB / (2 * self.lam))
                d = (a and b)                      # last frame before the edge is sampled
            y = (self.secret >> (self.nbits - 1 - slot)) & 1
            bits = (bits << 1) | int(bool(y) or d)  # D = Y OR (A AND B_out)
        return bits


# ------------------------------------------------------------------ level 1
class PhotonStatistics:
    """Per-bit Z-channel evidence."""
    def __init__(self, nbits):
        self.nbits, self.n = nbits, 0
        self.zeros = np.zeros(nbits, int)          # times each bit was seen as 0

    def update(self, word):
        self.n += 1
        for i in range(self.nbits):
            if not (word >> (self.nbits - 1 - i)) & 1:
                self.zeros[i] += 1

    def proven_zero(self):
        return self.zeros > 0

    def noise_rate(self):
        """Estimate p = P(observe 1 | secret 0) from bits proven zero."""
        z = self.proven_zero()
        if not z.any() or self.n == 0:
            return 0.5
        # bits proven zero: ones-rate = 1 - zeros/n  (slightly biased low; clipped)
        p = 1.0 - self.zeros[z].sum() / (self.n * z.sum())
        return float(np.clip(p, 0.01, 0.97))

    def posterior_one(self):
        """P(secret bit = 1) for each bit, uniform prior."""
        p, post = self.noise_rate(), np.zeros(self.nbits)
        z = self.proven_zero()
        post[~z] = 1.0 / (1.0 + p ** self.n)       # never seen 0
        return post                                # proven zeros -> 0.0


# ------------------------------------------------------------------ level 2
def list_decode(stats, commitment, salt, nbits, max_ambiguous=12, thr=0.999):
    """Enumerate candidates over ambiguous bits only; verify by commitment."""
    post = stats.posterior_one()
    base = 0
    for i in range(nbits):
        if post[i] >= thr or (post[i] > 0 and post[i] >= 0.5):
            base |= 1 << (nbits - 1 - i)           # tentatively 1
    amb = [i for i in range(nbits) if 0 < post[i] < thr]
    if len(amb) > max_ambiguous:
        return None, len(amb), 0
    tries = 0
    # try flipping fewest ambiguous bits first (1 is already the prior-heavy guess)
    for k in range(len(amb) + 1):
        for flip in itertools.combinations(amb, k):
            cand = base
            for i in flip:
                cand &= ~(1 << (nbits - 1 - i))
            tries += 1
            if commit(salt, cand, nbits) == commitment:
                return cand, len(amb), tries
    return None, len(amb), tries


# ------------------------------------------------------------------ driver
def attack(nbits, seed, max_words=200, reveal=False):
    rng = np.random.default_rng(seed)
    secret = int(rng.integers(0, 1 << nbits)) if nbits < 63 else secrets.randbits(nbits)
    salt = secrets.token_hex(8)
    commitment = commit(salt, secret, nbits)
    circ, stats = PhotonCircuit(secret, nbits, rng), PhotonStatistics(nbits)
    for w in range(1, max_words + 1):
        stats.update(circ.word())
        cand, n_amb, tries = list_decode(stats, commitment, salt, nbits)
        if cand is not None:
            return dict(ok=cand == secret, words=w, amb=n_amb, tries=tries,
                        p=stats.noise_rate(), secret=secret, guess=cand)
    return dict(ok=False, words=max_words, amb=-1, tries=0, p=stats.noise_rate(),
                secret=secret, guess=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--reveal", action="store_true")
    a = ap.parse_args()

    sizes = [8, 16, 32, 64, 128] if a.scaling else [a.bits - a.bits % 4]
    for n in sizes:
        res = [attack(n, s) for s in range(a.trials)]
        okr = np.mean([r["ok"] for r in res])
        print(f"bits={n:3d}  success={okr:5.0%}  "
              f"words mean={np.mean([r['words'] for r in res]):5.1f} "
              f"max={max(r['words'] for r in res):3d}  "
              f"hash checks mean={np.mean([r['tries'] for r in res]):7.1f}  "
              f"p_hat={np.mean([r['p'] for r in res]):.2f}  "
              f"(brute force: 2^{n} = {2.0 ** n:.2e})")
        if a.reveal:
            r = res[0]
            print(f"   e.g. secret={r['secret']:0{n // 4}X} guess="
                  f"{(r['guess'] or 0):0{n // 4}X}")


if __name__ == "__main__":
    main()
