#!/usr/bin/env python3
"""needle.py - camera -> digest -> nonce -> circuit -> hit, isolated from scientific.py.

  frame -> motion mask E -> digest = SHA-256(E)              (instance commitment)
  x     = odd 32-bit number from digest, 50% forced prime    (the problem instance)
  nonce chosen AFTER the digest; h = SHA-256d(digest||nonce)
  Driver A (randomized) : r = 2 + h mod (x-3);  A = 1 iff r is a Miller-Rabin witness of compositeness
                          (A = 0 always when x is prime)
  Driver B (RC link)    : c = [x >= 4] charges C for one clock; B = 1 iff VB = VC - 0.7 >= 3.5 V
  AND(A,B)              : the circuit's decision
  hit                   : h < 2^(256-k)                      (the CSV 'hit' column)

Needle: a nonce with  hit AND AND(A,B).  It exists only if x is composite (A is 0 on primes), so the
search for it is a one-sided (RP-style) decision: found => composite, certificate = the nonce.

Usage:  python needle.py --device 0 --frames 40 --log out.csv
        python needle.py --synthetic
Needs numpy (+ opencv-python for a real camera).
"""
import argparse, csv, hashlib, math, os, sys
import numpy as np

H, W = 96, 128
VS, VBE, VIH = 5.0, 0.7, 3.5
sha = lambda b: hashlib.sha256(b).digest()

# ---- camera ---------------------------------------------------------------------------
class Camera:
    def __init__(self, device):
        import cv2
        self.cv2, self.cap = cv2, cv2.VideoCapture(device)
        if not self.cap.isOpened():
            sys.exit(f"cannot open camera {device} (use --synthetic)")
    def read(self):
        ok, f = self.cap.read()
        if not ok: sys.exit("camera read failed")
        g = f if f.ndim == 2 else self.cv2.cvtColor(f, self.cv2.COLOR_BGR2GRAY)
        return self.cv2.resize(g, (W, H), interpolation=self.cv2.INTER_AREA)

class Synthetic:
    def __init__(self): self.rng = np.random.default_rng()
    def read(self): return np.clip(self.rng.poisson(100.0, (H, W)), 0, 255).astype(np.uint8)

# ---- number theory --------------------------------------------------------------------
SP = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_witness(a, n):                          # True iff a proves n composite; never true for primes
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    y = pow(a, d, n)
    if y in (1, n - 1): return False
    for _ in range(s - 1):
        y = y * y % n
        if y == n - 1: return False
    return True

def is_prime(n):
    for p in SP:
        if n % p == 0: return n == p
    return not any(is_witness(a, n) for a in SP)

def next_prime(n):
    n |= 1
    while not is_prime(n): n += 2
    return n

# ---- circuit --------------------------------------------------------------------------
def driver_b(x, T=0.1, tau=0.05):              # RC node + NPN emitter follower, HIGH at VB >= 3.5 V
    vc = VS * (1 - math.exp(-T / tau)) * (1 if x >= 4 else 0)
    return int(max(0.0, vc - VBE) >= VIH)

def circuit(digest, x, B, nonce, k):
    """One trial: returns (A, B, AND, hit)."""
    h = sha(sha(digest + nonce))
    A = int(is_witness(2 + int.from_bytes(h, "big") % (x - 3), x))
    hit = int(int.from_bytes(h, "big") < (1 << (256 - k)))
    return A, B, A & B, hit

def needle(digest, x, B, k, limit):
    """Search nonces until hit AND AND(A,B). Returns (nonce, tries) or (None, limit)."""
    for i in range(limit):
        nonce = i.to_bytes(4, "big")
        A, _, and_ab, hit = circuit(digest, x, B, nonce, k)
        if hit and and_ab:
            return nonce, i + 1
    return None, limit

# ---- main -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--thr", type=int, default=6, help="motion threshold in DN")
    ap.add_argument("--k", type=int, default=6, help="hash target bits: 2^-k per nonce")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    src = Synthetic() if a.synthetic else Camera(a.device)
    prev = src.read().astype(np.int16)
    limit = 64 << a.k                           # search cap (~64x expected cost)
    rows = []
    st = dict(hits=0, hit_and=0, found_comp=0, comp=0, found_prime=0, prime=0, tries=[])
    print(f"{'t':>3} {'count':>5} {'digest':<16} {'x':>10} L A B AND hit  {'needle':<8} {'tries':>6}")
    for t in range(a.frames):
        f = src.read().astype(np.int16)
        E = (np.abs(f - prev) > a.thr).astype(np.uint8); prev = f
        digest = sha(E.tobytes())               # 1. commit to the frame first
        x = int.from_bytes(digest[8:12], "big") | 1 | (1 << 31)
        if digest[12] & 1: x = next_prime(x)    # 50% of instances forced prime
        in_L = int(not is_prime(x))
        B = driver_b(x)

        nonce = os.urandom(8)                   # 2. nonce AFTER the digest -> circuit -> hit
        A, _, and_ab, hit = circuit(digest, x, B, nonce, a.k)
        st["hits"] += hit; st["hit_and"] += hit * and_ab

        nd, tries = needle(digest, x, B, a.k, limit)   # 3. needle search through the circuit
        st["comp" if in_L else "prime"] += 1
        if nd is not None:
            st["found_comp" if in_L else "found_prime"] += 1; st["tries"].append(tries)
        print(f"{t:>3} {int(E.sum()):>5} {digest.hex()[:16]} {x:>10} {in_L} {A} {B}  {and_ab}   {hit}  "
              f"{nd.hex() if nd else '-':<8} {tries if nd else '>'+str(limit):>6}")
        rows.append((t, digest.hex()[:16], int(E.sum()), x, in_L, A, B, and_ab, nonce.hex(), hit,
                     nd.hex() if nd else "", tries if nd else 0))

    n, p0 = a.frames, 2.0 ** -a.k
    print(f"\nrandom-nonce hits         : {st['hits']}/{n} (chance {n * p0:.2f}); hits with AND=1: {st['hit_and']}")
    print(f"needle on composites      : {st['found_comp']}/{st['comp']} found"
          + (f", mean tries {np.mean(st['tries']):.0f} (2^k = {1 << a.k})" if st["tries"] else ""))
    print(f"needle on primes          : {st['found_prime']}/{st['prime']} found (must be 0: A=0 on primes)")
    if a.log:
        with open(a.log, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow("tick digest16 count x in_L A B and nonce hit needle_nonce needle_tries".split())
            w.writerows(rows)
        print(f"log written to {a.log}")

if __name__ == "__main__":
    main()
