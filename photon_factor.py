#!/usr/bin/env python3
"""Photon two-level decoder + integer factorisation.

Recovers a hidden integer from noisy photon observations, then factors it.
Usage:
  python photon_factor.py                # 16-bit demo
  python photon_factor.py --bits 24 --trials 5
  python photon_factor.py --factor 600851475143
"""
import argparse, hashlib, itertools, secrets, math, random, time, sys
import numpy as np

VCC, V_BE = 5.0, 0.7
TAU = 10e3 * 15e-6
CLK = 0.100
FRAME_DT = 0.005

def commit(salt, v, nbits):
    return hashlib.sha256(f"{salt}:{v:0{nbits // 4}X}".encode()).hexdigest()

class PhotonCircuit:
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
        return self.vc > V_BE
    def word(self):
        bits = 0
        frames = int(round(CLK / FRAME_DT))
        for slot in range(self.nbits):
            d = 0
            for _ in range(frames):
                self.t += FRAME_DT
                nB = self.rng.poisson(self.lam * (1 + 0.8*np.sin(2*np.pi*0.4*self.t)))
                nA = self.rng.poisson(self.lam * (1 + 0.8*np.sin(2*np.pi*0.7*self.t + 1)))
                a = bool(nA & 1)
                b = self._b_out(nB / (2*self.lam))
                d = a and b
            y = (self.secret >> (self.nbits-1-slot)) & 1
            bits = (bits << 1) | int(bool(y) or d)
        return bits

class PhotonStatistics:
    def __init__(self, nbits):
        self.nbits, self.n = nbits, 0
        self.zeros = np.zeros(nbits, int)
    def update(self, word):
        self.n += 1
        for i in range(self.nbits):
            if not (word >> (self.nbits-1-i)) & 1:
                self.zeros[i] += 1
    def proven_zero(self):
        return self.zeros > 0
    def noise_rate(self):
        z = self.proven_zero()
        if not z.any() or self.n == 0:
            return 0.5
        p = 1.0 - self.zeros[z].sum() / (self.n * z.sum())
        return float(np.clip(p, 0.01, 0.97))
    def posterior_one(self):
        p = self.noise_rate()
        post = np.zeros(self.nbits)
        z = self.proven_zero()
        post[~z] = 1.0 / (1.0 + p ** self.n)
        return post

def list_decode(stats, commitment, salt, nbits, max_ambiguous=14, thr=0.999):
    post = stats.posterior_one()
    base = 0
    for i in range(nbits):
        if post[i] >= 0.5:
            base |= 1 << (nbits-1-i)
    amb = [i for i in range(nbits) if 0 < post[i] < thr]
    if len(amb) > max_ambiguous:
        return None, len(amb), 0
    tries = 0
    for k in range(len(amb)+1):
        for flip in itertools.combinations(amb, k):
            cand = base
            for i in flip:
                cand &= ~(1 << (nbits-1-i))
            tries += 1
            if commit(salt, cand, nbits) == commitment:
                return cand, len(amb), tries
    return None, len(amb), tries

def photon_recover(nbits, seed, max_words=200):
    rng = np.random.default_rng(seed)
    secret = int(rng.integers(0, 1 << nbits))
    salt = secrets.token_hex(8)
    commitment = commit(salt, secret, nbits)
    circ = PhotonCircuit(secret, nbits, rng)
    stats = PhotonStatistics(nbits)
    for w in range(1, max_words+1):
        stats.update(circ.word())
        cand, n_amb, tries = list_decode(stats, commitment, salt, nbits)
        if cand is not None:
            return cand, w, n_amb, tries
    return None, max_words, -1, 0

def is_prime(n):
    if n < 2: return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n % p == 0: return n == p
    d, s = n-1, 0
    while d % 2 == 0: d//=2; s+=1
    for a in (2,3,5,7,11,13,17,19,23,29,31,37):
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1: break
        else: return False
    return True

def pollard_rho(n):
    if n % 2 == 0: return 2
    while True:
        c = random.randrange(1, n)
        x = y = random.randrange(2, n)
        d = 1
        while d == 1:
            x = (x*x+c) % n
            y = (y*y+c) % n; y = (y*y+c) % n
            d = math.gcd(abs(x-y), n)
        if d != n: return d

def factor(n):
    fs = []
    for p in (2,3,5):
        while n % p == 0: fs.append(p); n//=p
    f, add, i = 7, [4,2,4,2,4,6,2,6], 0
    while f <= 100000 and f*f <= n:
        while n % f == 0: fs.append(f); n//=f
        f += add[i]; i = (i+1) % 8
    stack=[n] if n>1 else []
    while stack:
        m=stack.pop()
        if m==1: continue
        if is_prime(m): fs.append(m); continue
        d=pollard_rho(m); stack += [d, m//d]
    return sorted(fs)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bits', type=int, default=16)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--factor', type=int, default=None)
    a = ap.parse_args()
    if a.factor:
        t=time.perf_counter(); fs=factor(a.factor); dt=time.perf_counter()-t
        assert math.prod(fs)==a.factor
        print(f'{a.factor} = ' + ' * '.join(map(str,fs)))
        print(f'verified: True  time: {dt:.6f}s')
        sys.exit()
    for s in range(a.trials):
        secret, words, amb, tries = photon_recover(a.bits, s)
        if secret is None:
            print(f'trial {s}: recovery failed')
            continue
        fs = factor(secret)
        assert math.prod(fs)==secret
        print(f'trial {s}: recovered {secret} in {words} words '
              f'({tries} hash checks); factors: ' + ' * '.join(map(str,fs)))
