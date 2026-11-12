#!/usr/bin/env python3
"""circuit_sort.py - sort a small dataset with the RP circuit from needle.py / scientific.py.

How the circuit does the sorting
1. READ : every input bit is stored as a problem instance x (bit 1 = composite, bit 0 = prime) and
read back by the full chain Driver A (Miller-Rabin witness from nonce) + Driver B (RC)
-> AND(A,B). One-sided: a prime is never read as 1; a composite is read as 1 once any
nonce hits a witness (a few retries make a miss vanishingly rare).
2. SORT : the circuit's AND/OR pair is a 1-bit compare-exchange cell:
(low, high) = (AND(a,b), OR(a,b)) = (min, max)
An odd-even transposition network of those cells sorts any 0/1 vector (0-1 principle).
3. LIFT : numbers are sorted by threshold decomposition: for each t, bit_i = [x_i >= t] is sorted
as above, and output[i] = sum over t of sorted_bit_t[i]. Only AND/OR gates touch the data.

The dataset comes from the camera: digest = SHA-256(motion mask) seeds a hash stream that makes
n values in 0..m-1 for any n (--pattern: random, sorted, reversed, dups).

Usage:
python circuit_sort.py --device 0 --frames 3
python circuit_sort.py --synthetic --n 200 --m 16 --pattern reversed
python circuit_sort.py --synthetic --data 5,3,7,0,2,6,1,4
python circuit_sort.py --synthetic --n 200 --m 16 --pattern reversed --save dataset.json
python circuit_sort.py --synthetic --load dataset.json

Needs numpy (+ opencv-python for a real camera).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

H, W = 96, 128
VS, VBE, VIH = 5.0, 0.7, 3.5

sha = lambda b: hashlib.sha256(b).digest()


# ---- camera ---------------------------------------------------------------------------

class Camera:
    def __init__(self, device):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(device)

        if not self.cap.isOpened():
            sys.exit(f"cannot open camera {device} (use --synthetic)")

    def read(self):
        ok, f = self.cap.read()

        if not ok:
            sys.exit("camera read failed")

        g = f if f.ndim == 2 else self.cv2.cvtColor(f, self.cv2.COLOR_BGR2GRAY)
        return self.cv2.resize(g, (W, H), interpolation=self.cv2.INTER_AREA)


class Synthetic:
    def __init__(self):
        self.rng = np.random.default_rng()

    def read(self):
        return np.clip(self.rng.poisson(100.0, (H, W)), 0, 255).astype(np.uint8)


# ---- number theory --------------------------------------------------------------------

SP = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_witness(a, n):
    """True iff a proves n composite; never true for primes."""
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


def is_prime(n):
    for p in SP:
        if n % p == 0:
            return n == p

    return not any(is_witness(a, n) for a in SP)


def next_prime(n):
    n |= 1

    while not is_prime(n):
        n += 2

    return n


# ---- circuit --------------------------------------------------------------------------

def driver_b(x, T=0.1, tau=0.05):
    """RC node + NPN emitter follower, HIGH at VB >= 3.5 V."""
    vc = VS * (1 - math.exp(-T / tau)) * (1 if x >= 4 else 0)
    return int(max(0.0, vc - VBE) >= VIH)


def circuit_and(digest, x, nonce):
    """Driver A (witness r from the nonce) AND Driver B -> the circuit's decision for instance x."""
    h = int.from_bytes(sha(sha(digest + nonce)), "big")
    A = int(is_witness(2 + h % (x - 3), x))
    return A & driver_b(x)


def gate(a, b):
    """The circuit's AND/OR pair = compare-exchange cell."""
    return a & b, a | b  # (min, max)


# ---- bit storage + read through the circuit -------------------------------------------

def instance(digest, label, bit):
    base = int.from_bytes(sha(digest + label)[:4], "big") | 1 | (1 << 31)

    if bit:  # 1 -> composite
        while is_prime(base):
            base += 2

        return base

    return next_prime(base)  # 0 -> prime


def read_bit(digest, label, bit, retries, st):
    x = instance(digest, label, bit)

    for _ in range(retries):  # fresh nonce each try, drawn after the digest
        st["trials"] += 1

        if circuit_and(digest, x, os.urandom(8)):
            return 1

    return 0


# ---- sorting network ------------------------------------------------------------------

def sort_bits(bits, st):
    """Odd-even transposition network, n rounds."""
    v = list(bits)

    for rnd in range(len(v)):
        for i in range(rnd % 2, len(v) - 1, 2):
            v[i], v[i + 1] = gate(v[i], v[i + 1])
            st["gates"] += 1

    return v


def circuit_sort(digest, data, m, retries, st, verbose=True):
    n = len(data)
    out = [0] * n

    for t in range(1, m):
        true_bits = [int(d >= t) for d in data]
        read = [
            read_bit(digest, f"{t}:{i}".encode(), b, retries, st)
            for i, b in enumerate(true_bits)
        ]

        st["miss"] += sum(r != b for r, b in zip(read, true_bits))
        srt = sort_bits(read, st)
        out = [o + s for o, s in zip(out, srt)]

        if verbose:
            print(
                f" t={t:<2} in {''.join(map(str, read))} "
                f"-> sorted {''.join(map(str, srt))}"
            )

    return out


# ---- test dataset ---------------------------------------------------------------------

def make_dataset(digest, n, m, pattern="random"):
    """n values in 0..m-1 from a SHA-256 counter stream seeded by the frame digest (any length)."""
    d = [
        int.from_bytes(sha(digest + b"data" + i.to_bytes(8, "big"))[:8], "big") % m
        for i in range(n)
    ]

    if pattern == "sorted":
        d.sort()
    elif pattern == "reversed":
        d.sort(reverse=True)
    elif pattern == "dups":
        d = [v % min(m, 3) for v in d]  # only 3 distinct values

    return d


def save_dataset(path, digest, n, m, pattern, data, sorted_data, sorted_ok):
    """Save the original dataset, its parameters, and the circuit-sorted result."""
    payload = {
        "digest": digest.hex(),
        "n": n,
        "m": m,
        "pattern": pattern,
        "data": data,
        "sorted_data": sorted_data,
        "sorted_ok": bool(sorted_ok),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def load_dataset(path):
    """Load a dataset previously saved by save_dataset()."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    required = {"digest", "n", "m", "pattern", "data", "sorted_data", "sorted_ok"}
    missing = required - payload.keys()

    if missing:
        raise ValueError(
            f"dataset file is missing fields: {', '.join(sorted(missing))}"
        )

    digest = bytes.fromhex(payload["digest"])
    data = [int(v) for v in payload["data"]]
    sorted_data = [int(v) for v in payload["sorted_data"]]

    if len(data) != payload["n"]:
        raise ValueError(
            f"dataset length mismatch: n={payload['n']} "
            f"but file contains {len(data)} values"
        )

    if len(sorted_data) != payload["n"]:
        raise ValueError(
            f"sorted result length mismatch: n={payload['n']} "
            f"but file contains {len(sorted_data)} values"
        )

    if any(not (0 <= v < payload["m"]) for v in data):
        raise ValueError("one or more loaded values are outside 0..m-1")

    return digest, data, sorted_data, payload


def show(v):
    if len(v) <= 16:
        return str(v)

    return (
        f"[{', '.join(map(str, v[:8]))}, ... "
        f"{', '.join(map(str, v[-4:]))}] (len {len(v)})"
    )


# ---- main -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument(
        "--frames", type=int, default=3, help="one fresh dataset per frame"
    )
    ap.add_argument(
        "--thr", type=int, default=1, help="motion threshold in DN"
    )
    ap.add_argument(
        "--n", type=int, default=8, help="dataset length (any size >= 1)"
    )
    ap.add_argument(
        "--pattern",
        choices=("random", "sorted", "reversed", "dups"),
        default="random",
    )
    ap.add_argument(
        "--m", type=int, default=8, help="values are 0..m-1"
    )
    ap.add_argument(
        "--data",
        default=None,
        help="comma list to sort instead of a generated dataset",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=8,
        help="nonce tries per bit read (RP repeats)",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="save dataset and sorted result to this JSON file",
    )
    ap.add_argument(
        "--load",
        default=None,
        help="load a saved dataset and its sorted result from this JSON file",
    )
    a = ap.parse_args()

    if a.load and (a.data or a.save):
        sys.exit("--load cannot be combined with --data or --save")

    src = Synthetic() if a.synthetic else Camera(a.device)
    prev = src.read().astype(np.int16)

    good = 0
    st = dict(trials=0, gates=0, miss=0)

    for t in range(a.frames):
        f = src.read().astype(np.int16)
        E = (np.abs(f - prev) > a.thr).astype(np.uint8)
        prev = f
        digest = sha(E.tobytes())  # instance commitment for this frame

        if a.load:
            digest, data, saved_sorted, meta = load_dataset(a.load)
            m = meta["m"]

            print(
                f"frame {t} loaded {a.load} digest {digest.hex()[:16]} "
                f"n={meta['n']} m={m} pattern={meta['pattern']} "
                f"data {show(data)}"
            )
            print(f" saved sorted result {show(saved_sorted)}")

        elif a.data:
            data = [int(v) for v in a.data.split(",")]
            m = max(a.m, max(data) + 1)
            print(f"frame {t} digest {digest.hex()[:16]} data {show(data)}")

        else:
            data = make_dataset(digest, a.n, a.m, a.pattern)
            m = a.m
            print(f"frame {t} digest {digest.hex()[:16]} data {show(data)}")

        t0 = time.time()
        out = circuit_sort(
            digest,
            data,
            m,
            a.retries,
            st,
            verbose=len(data) <= 16,
        )

        ok = out == sorted(data)
        good += ok

        if a.save:
            save_path = a.save

            if a.frames > 1:
                root, ext = os.path.splitext(a.save)
                save_path = f"{root}_{t}{ext or '.json'}"

            save_dataset(
                save_path,
                digest,
                len(data),
                m,
                a.pattern,
                data,
                out,
                ok,
            )

            print(f" saved dataset and sorted result to {save_path}")

        print(
            f" sorted {show(out)} {'OK' if ok else 'MISMATCH'} "
            f"({time.time() - t0:.1f}s)\n"
        )

    print(
        f"{good}/{a.frames} sorted correctly | circuit trials {st['trials']} | "
        f"AND/OR cells {st['gates']} | "
        f"bit reads that missed a 1: {st['miss']} (one-sided errors only)"
    )


if __name__ == "__main__":
    main()
