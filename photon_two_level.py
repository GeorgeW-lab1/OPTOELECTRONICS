#!/usr/bin/env python3
"""
v2: Level 2 enumerates by number of flips (Hamming weight) over the never-seen-0
set U, bounded by a per-word hash budget, instead of requiring |U| <= 12 and
waiting for the 0.999 posterior gate.

Key change: after n words, a true 1 is in U with certainty, and a true 0 is in U
with probability p^n. So the number of flips needed k ~ Binomial(n/2, p^n), which
is tiny long before the posterior hits 0.999. Cost is sum_k C(|U|, k), not 2^|U|.

Usage:
    python photon_v2.py                  # compare v1 vs v2 at several sizes
    python photon_v2.py --budget 50000   # allow deeper search per word
"""
import argparse
import itertools
import secrets
import time
from math import comb

import numpy as np

import photon_two_level as v1
from photon_two_level import PhotonCircuit, PhotonStatistics, commit


def list_decode_v2(stats, commitment, salt, nbits, budget=5000):
    """Try k = 0,1,2,... flips (1 -> 0) over U, stopping before exceeding `budget`."""
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
            if commit(salt, cand, nbits) == commitment:
                return cand, len(U), tries
    return None, len(U), tries


def attack_v2(nbits, seed, max_words=200, budget=5000):
    rng = np.random.default_rng(seed)
    secret = int(rng.integers(0, 1 << nbits)) if nbits < 63 else secrets.randbits(nbits)
    salt = secrets.token_hex(8)
    commitment = commit(salt, secret, nbits)
    circ, stats = PhotonCircuit(secret, nbits, rng), PhotonStatistics(nbits)
    total = 0
    for w in range(1, max_words + 1):
        stats.update(circ.word())
        cand, nU, tries = list_decode_v2(stats, commitment, salt, nbits, budget)
        total += tries                       # cumulative across all words
        if cand is not None:
            return dict(ok=cand == secret, words=w, tries=total, nU=nU)
    return dict(ok=False, words=max_words, tries=total, nU=-1)


def attack_v1(nbits, seed):
    r = v1.attack(nbits, seed)
    return dict(ok=r["ok"], words=r["words"], tries=r["tries"], nU=r["amb"])


def summarize(label, res):
    W = np.array([r["words"] for r in res]); T = np.array([r["tries"] for r in res])
    print(f"  {label:<14} succ={np.mean([r['ok'] for r in res]):4.0%}  "
          f"words mean={W.mean():5.2f} max={W.max():3d}  "
          f"hashes mean={T.mean():9.1f} p90={np.percentile(T, 90):9.0f} max={T.max():8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--trials", type=int, default=60)
    a = ap.parse_args()

    for n in [16, 32, 64, 128, 256]:
        trials = a.trials if n <= 128 else max(20, a.trials // 3)
        print(f"n={n}  (brute force 2^{n} = {2.0 ** n:.1e}, trials={trials})")
        t = time.time(); r1 = [attack_v1(n, s) for s in range(trials)]
        summarize("v1 (cap<=12)", r1)
        r2 = [attack_v2(n, s, budget=a.budget) for s in range(trials)]
        summarize(f"v2 (budget {a.budget})", r2)
        w1 = np.mean([r["words"] for r in r1]); w2 = np.mean([r["words"] for r in r2])
        print(f"  -> words saved: {w1 - w2:.2f} ({(w1 - w2) / w1:.0%}), "
              f"hw time saved ~{(w1 - w2) * n * v1.CLK:.0f}s\n")


if __name__ == "__main__":
    main()
