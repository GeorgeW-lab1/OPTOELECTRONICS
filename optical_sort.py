#!/usr/bin/env python3
"""
optical_sort.py - the Wagenknecht optical bit-pooling pipeline, applied to datasets.

Pipeline (paper section in brackets), run for ALL records in parallel as tensor ops:
  oracle   : holds the dataset; publishes only salt s and per-record c_k = SHA256(s || x_k)   [1]
  channel  : each record is a rows x cols grid of black/white cells observed every cycle
             - mode "bit"   : reveal ~ Bernoulli(r), flip ~ Bernoulli(eps)                    [2.5, 3]
             - mode "pixel" : 255/0 cells + Gaussian pixel noise, BxB block mean pooling
                              (reshape + mean over axes (2,4)), confidence gate |p-tau|>=c0   [2.1-2.3]
  recogniser: sees only frames + commitments. Accumulates ones/zeros votes (majority = ML) [4.1],
              waits for cover, hashes the majority word, and for failures runs a Chase-type
              search over the L least-reliable bits (<= t flips, per-record budget qmax)     [5]
  output   : verified records only are sorted (parallel sample-sort) and saved raw.

Raw format: headerless little-endian binary. Commitment file: 16-byte salt + K*32-byte digests.

Supported dtypes: uint/int/float with itemsize 1,2,4,8 (order-preserving key mapping).
NOT a factoriser and not a compression/sorting breakthrough: it is a noisy-channel recovery
demo whose verifier is a hash. See paper sections 6-7 and 9.

Line-per-line text datasets (optical pipeline ON by default, --plain to skip):
  python optical_sort.py lines-gen data.txt --n 100000 [--min-len 0 --max-len 40 --space-prob 0.12]
      (lines are STRINGS by default: pronounceable words; --style random --alphabet alnum for noise)
  python optical_sort.py lines-gen ints data.txt --n 100000 [--lo -1000 --hi 1000]
      (generation streams line per line in chunks; path '-' writes to stdout)
  python optical_sort.py sort    in.txt  out.txt      (auto-detects ints vs strings)
  python optical_sort.py ints    in.txt  out.txt [--desc] [--unique] [--plain]
  python optical_sort.py strings in.txt  out.txt [--desc] [--unique] [--ignore-case] [--plain]
  python optical_sort.py to-raw in.txt out.raw | from-raw in.raw out.txt
  python optical_sort.py lines-bench ints|strings --n 50000
  Strings: every line is ONE variable-length grid record with its own SHA-256 commitment
  (line lengths are public); ints: every line is one int64 8x8 record. --commit-out saves
  salt+digests (and <file>.lens with the public lengths for strings).

CLI (fixed-width binary records):
  python optical_sort.py gen   data.raw --n 100000 --dtype uint32
  python optical_sort.py run   data.raw sorted.raw --dtype uint32 [--mode pixel --sigma 40]
  python optical_sort.py bench --n 50000 --dtype uint32
"""
import argparse, hashlib, itertools, math, os, sys, time
import numpy as np
from fastsort import save_raw, load_raw, parallel_sort, _gen

GRID = {8: (2, 4), 16: (4, 4), 32: (4, 8), 64: (8, 8)}   # m bits -> (rows, cols)
EULER = 0.5772156649015329


# ------------------------------------------------ order-preserving keys
def to_key(a):
    a = np.ascontiguousarray(a)
    bits = a.dtype.itemsize * 8
    U = np.dtype(f"<u{a.dtype.itemsize}")
    u = a.view(U)
    sign = U.type(1) << U.type(bits - 1)
    if a.dtype.kind == "u":
        return u
    if a.dtype.kind == "i":
        return u ^ sign
    if a.dtype.kind == "f":
        neg = (u >> U.type(bits - 1)).astype(bool)
        return np.where(neg, ~u, u ^ sign)
    raise TypeError("unsupported dtype")


def from_key(k, dtype):
    dt = np.dtype(dtype).newbyteorder("<")
    bits = dt.itemsize * 8
    U = k.dtype
    sign = U.type(1) << U.type(bits - 1)
    if dt.kind == "u":
        return k.view(dt)
    if dt.kind == "i":
        return (k ^ sign).view(dt)
    top = (k >> U.type(bits - 1)).astype(bool)
    return np.where(top, k ^ sign, ~k).view(dt)


# ------------------------------------------------ hashing helpers
def be_bytes(words, nb):
    """uint64 words -> concatenated big-endian fixed-length byte strings (nb bytes each)."""
    return np.ascontiguousarray(
        words.astype(">u8").view(np.uint8).reshape(-1, 8)[:, 8 - nb:]).tobytes()


def sha_rows(salt, raw, nb):
    cp = hashlib.sha256(salt).copy
    d = []
    for i in range(0, len(raw), nb):
        h = cp(); h.update(raw[i:i + nb]); d.append(h.digest())
    return np.frombuffer(b"".join(d), np.uint8).reshape(-1, 32)


def save_commit(path, salt, commits):
    with open(path, "wb") as f:
        f.write(salt); f.write(np.ascontiguousarray(commits).tobytes())


def load_commit(path):
    raw = open(path, "rb").read()
    return raw[:16], np.frombuffer(raw[16:], np.uint8).reshape(-1, 32)


# ------------------------------------------------ channels
def grid_for(m):
    """rows x cols grid for m bits (near-square; spare cells at the end stay black)."""
    if m in GRID:
        return GRID[m]
    cols = math.isqrt(m - 1) + 1
    return -(-m // cols), cols


def make_observer(words, m, mode, rng, r, eps, sigma, block, tau, c0, truth_fn=None):
    shifts = np.arange(m, dtype=np.uint64)
    rows, cols = grid_for(m)

    def truth(idx):
        return ((words[idx][:, None] >> shifts) & np.uint64(1)).astype(bool)
    if truth_fn is not None:                      # custom record source (e.g. variable-length lines)
        truth = truth_fn

    if mode == "bit":
        def observe(idx):
            b = truth(idx)
            reveal = rng.random(b.shape, dtype=np.float32) < r
            return reveal, b ^ (rng.random(b.shape, dtype=np.float32) < eps)
        return observe

    step = max(1, int(8e6 // (m * block * block)))      # records per pooling batch (memory cap)

    def observe(idx):
        rv, ob = [], []
        for s in range(0, len(idx), step):
            b = truth(idx[s:s + step])
            K = b.shape[0]
            if b.shape[1] < rows * cols:                              # spare grid cells: black
                b = np.concatenate([b, np.zeros((K, rows * cols - b.shape[1]), bool)], 1)
            lv = b.reshape(K, rows, 1, cols, 1).astype(np.float32) * 255
            frame = lv + rng.standard_normal((K, rows, block, cols, block), dtype=np.float32) * sigma
            p = frame.mean(axis=(2, 4)).reshape(K, -1)[:, :m]        # <- the spatial pooling
            valid = np.abs(p - tau) >= c0                            # confidence gate
            rv.append((rng.random(p.shape, dtype=np.float32) < r) & valid)
            ob.append(p > tau)
        return np.concatenate(rv), np.concatenate(ob)
    return observe


def _Q(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def effective_channel(r, sigma, block, tau, c0):
    """Section 2.3: worst-case (eps_eff, r_eff) over black/white cells."""
    s = sigma / block
    eps = rho = 0.0
    for d in (255 - tau, tau):
        eps = max(eps, _Q((d + c0) / s))
        rho = max(rho, _Q((d - c0) / s) - _Q((d + c0) / s))
    return eps, r * (1 - rho)


def theory(m, r, eps):
    Z = 2 * math.sqrt(eps * (1 - eps))
    base = 1 - r * (1 - Z)
    lam = -math.log(1 - r) if r < 1 else float("inf")
    H = 0.0 if eps in (0, 1) else -eps * math.log2(eps) - (1 - eps) * math.log2(1 - eps)
    cap = r * (1 - H)
    return dict(Z=Z, base=base, exponent=-math.log(base) if base > 0 else float("inf"),
                cover_mean=(math.log(m) + EULER) / lam, cover_sd=math.pi / (math.sqrt(6) * lam),
                capacity=cap, shannon_T=1 / cap if cap > 0 else float("inf"),
                T_union=lambda delta: math.log(m / delta) / -math.log(base))


# ------------------------------------------------ recogniser
def recognise(commits, salt, nb, observe, T_max, L, t, qmax, rng):
    K, m = len(commits), nb * 8
    ones = np.zeros((K, m), np.int16); zeros = np.zeros((K, m), np.int16)
    weights = np.uint64(1) << np.arange(m, dtype=np.uint64)
    out = np.zeros(K, np.uint64)
    done = np.zeros(K, bool)
    done_cycle = np.zeros(K, np.int32); cover_cycle = np.zeros(K, np.int32)
    tested = np.zeros(K, np.int64)
    L = min(L, m); t = min(t, L)
    combos = [np.array(list(itertools.combinations(range(L), k)), dtype=np.intp)
              for k in range(1, t + 1)]
    active = np.arange(K)
    hashes = 0
    cycle = 0
    while active.size and cycle < T_max:
        cycle += 1
        rev, obs = observe(active)
        ones[active] += (rev & obs); zeros[active] += (rev & ~obs)
        o, z = ones[active], zeros[active]
        seen = ((o + z) > 0).all(1)
        new = seen & (cover_cycle[active] == 0)
        cover_cycle[active[new]] = cycle
        idx = np.nonzero(seen)[0]
        if not idx.size:
            continue
        ai = active[idx]
        o, z = o[idx], z[idx]
        packed = ((o > z).astype(np.uint64) * weights).sum(1, dtype=np.uint64)
        ok = (sha_rows(salt, be_bytes(packed, nb), nb) == commits[ai]).all(1)
        hashes += len(ai); tested[ai] += 1
        out[ai[ok]] = packed[ok]; done[ai[ok]] = True; done_cycle[ai[ok]] = cycle
        fail = np.nonzero(~ok)[0]
        if t and fail.size:
            marg = np.abs(o[fail].astype(np.int32) - z[fail]).astype(np.float32)
            marg += rng.random(marg.shape, dtype=np.float32) * 0.5   # random tie-break (paper s.9.5)
            pos = np.argpartition(marg, L - 1, axis=1)[:, :L]         # L least-reliable bits
            for j, f in enumerate(fail):
                k = ai[f]
                room = qmax - tested[k]
                if room <= 0:
                    continue
                w = weights[pos[j]]
                masks = np.concatenate([w[c].sum(1, dtype=np.uint64) for c in combos])[:room]
                cand = packed[f] ^ masks
                hit = (sha_rows(salt, be_bytes(cand, nb), nb) == commits[k]).all(1)
                tested[k] += len(cand); hashes += len(cand)
                if hit.any():
                    out[k] = cand[np.argmax(hit)]; done[k] = True; done_cycle[k] = cycle
        active = active[~done[active]]
    return out, done, dict(cycles=cycle, hashes=hashes, done_cycle=done_cycle,
                           cover_cycle=cover_cycle)


# ------------------------------------------------ end to end
def run_arrays(data, mode="bit", r=0.45, eps=0.02, sigma=40.0, block=16, tau=127, c0=25,
               T_max=800, L=12, t=2, qmax=20_000, seed=0, commit_out=None, threads=None,
               verbose=True, sort=True):
    data = np.ascontiguousarray(data)
    nb = data.dtype.itemsize; m = nb * 8
    if m not in GRID:
        raise ValueError("itemsize must be 1, 2, 4 or 8 bytes")
    rng = np.random.default_rng(seed)
    keys = to_key(data)
    words = keys.astype(np.uint64)

    t0 = time.perf_counter()
    salt = rng.bytes(16)                                              # oracle
    commits = sha_rows(salt, be_bytes(words, nb), nb)
    if commit_out:
        save_commit(commit_out, salt, commits)
    t_commit = time.perf_counter() - t0

    if mode == "pixel":
        eps_e, r_e = effective_channel(r, sigma, block, tau, c0)
    else:
        eps_e, r_e = eps, r
    th = theory(m, r_e, eps_e)

    obs = make_observer(words, m, mode, rng, r, eps, sigma, block, tau, c0)
    t0 = time.perf_counter()
    rec, done, st = recognise(commits, salt, nb, obs, T_max, L, t, qmax, rng)
    t_rec = time.perf_counter() - t0

    t0 = time.perf_counter()
    if sort:
        result = from_key(parallel_sort(rec[done].astype(keys.dtype), threads), data.dtype)
    else:                                   # original order; unverified slots are zero (see `done`)
        result = from_key(rec.astype(keys.dtype), data.dtype)
    t_sort = time.perf_counter() - t0

    st.update(n=len(data), verified=int(done.sum()), t_commit=t_commit, t_recognise=t_rec,
              t_sort=t_sort, theory=th, eps_eff=eps_e, r_eff=r_e)
    if verbose:
        dc = st["done_cycle"][done]; cc = st["cover_cycle"][st["cover_cycle"] > 0]
        print(f"records {len(data):,} ({m}-bit, grid {GRID[m][0]}x{GRID[m][1]}, mode={mode})")
        print(f"channel: r_eff={r_e:.3f} eps_eff={eps_e:.4g}  Z={th['Z']:.3f}  "
              f"exponent={th['exponent']:.3f}/cycle  capacity={th['capacity']:.3f} bit/cell/frame")
        print(f"cover time  observed mean {cc.mean():.2f} (sd {cc.std():.2f}) | "
              f"theory {th['cover_mean']:.2f} (sd {th['cover_sd']:.2f})" if cc.size else "cover: none")
        if dc.size:
            print(f"verify cycle mean {dc.mean():.2f}, p99 {np.percentile(dc, 99):.0f}, max {dc.max()} | "
                  f"plain-majority bound T(1e-2)={th['T_union'](1e-2):.1f}  Shannon floor={th['shannon_T']:.2f}")
        print(f"verified {st['verified']:,}/{len(data):,}  hashes {st['hashes']:,}  "
              f"commit {t_commit:.2f}s recognise {t_rec:.2f}s sort {t_sort:.3f}s")
        if st["verified"] < len(data):
            print(f"WARNING: {len(data) - st['verified']:,} records unverified and excluded "
                  f"(raise --tmax / --chase-t / --qmax)")
    return result, done, st


# ------------------------------------------------ native variable-length records (lines)
def recognise_bytes(commits, salt, nb, observe, T_max, L, t, qmax, rng):
    """Same recogniser as `recognise`, for records of nb bytes (any nb): votes -> packbits ->
    hash check -> Chase search over the L least-reliable bits of the WHOLE record."""
    K, m = len(commits), nb * 8
    ones = np.zeros((K, m), np.int16); zeros = np.zeros((K, m), np.int16)
    out = np.zeros((K, nb), np.uint8)
    done = np.zeros(K, bool)
    done_cycle = np.zeros(K, np.int32); cover_cycle = np.zeros(K, np.int32)
    tested = np.zeros(K, np.int64)
    L = min(L, m); t = min(t, L)
    combos = [np.array(list(itertools.combinations(range(L), k)), dtype=np.intp)
              for k in range(1, t + 1)]
    active = np.arange(K); hashes = 0; cycle = 0
    while active.size and cycle < T_max:
        cycle += 1
        rev, obs = observe(active)
        ones[active] += (rev & obs); zeros[active] += (rev & ~obs)
        o, z = ones[active], zeros[active]
        seen = ((o + z) > 0).all(1)
        new = seen & (cover_cycle[active] == 0)
        cover_cycle[active[new]] = cycle
        idx = np.nonzero(seen)[0]
        if not idx.size:
            continue
        ai = active[idx]; o, z = o[idx], z[idx]
        packed = np.packbits(o > z, axis=1)                         # (n, nb) majority record
        ok = (sha_rows(salt, packed.tobytes(), nb) == commits[ai]).all(1)
        hashes += len(ai); tested[ai] += 1
        out[ai[ok]] = packed[ok]; done[ai[ok]] = True; done_cycle[ai[ok]] = cycle
        fail = np.nonzero(~ok)[0]
        if t and fail.size:
            marg = np.abs(o[fail].astype(np.int32) - z[fail]).astype(np.float32)
            marg += rng.random(marg.shape, dtype=np.float32) * 0.5
            pos = np.argpartition(marg, L - 1, axis=1)[:, :L]
            for j, f in enumerate(fail):
                k = ai[f]; room = qmax - tested[k]
                if room <= 0:
                    continue
                parts = []
                for c in combos:
                    sel = pos[j][c]                                  # (V_k, k) bit positions
                    cand = np.repeat(packed[f][None, :], len(c), 0)
                    ri = np.arange(len(c))
                    for col in range(c.shape[1]):
                        cand[ri, sel[:, col] >> 3] ^= (128 >> (sel[:, col] & 7)).astype(np.uint8)
                    parts.append(cand)
                cand = np.concatenate(parts)[:room]
                hit = (sha_rows(salt, cand.tobytes(), nb) == commits[k]).all(1)
                tested[k] += len(cand); hashes += len(cand)
                if hit.any():
                    out[k] = cand[np.argmax(hit)]; done[k] = True; done_cycle[k] = cycle
        active = active[~done[active]]
    return out, done, dict(cycles=cycle, hashes=hashes, done_cycle=done_cycle,
                           cover_cycle=cover_cycle)


def recover_lines(lines, mode="bit", r=0.45, eps=0.02, sigma=40.0, block=16, tau=127, c0=25,
                  T_max=800, L=12, t=2, qmax=20_000, seed=0, commit_out=None, verbose=True):
    """Each line is ONE record: its own grid (8*len cells), its own SHA-256 commitment, recovered
    by pooling + voting + Chase search, verified as a whole. Line lengths are public (they fix the
    grid size). Records are processed in groups of equal length. Returns (list with None for
    unverified lines, stats)."""
    t0 = time.perf_counter()
    n = len(lines)
    rng = np.random.default_rng(seed)
    salt = rng.bytes(16)
    lens = np.fromiter(map(len, lines), np.int64, n)
    commits_all = np.zeros((n, 32), np.uint8)
    out = [None] * n
    empty = np.nonzero(lens == 0)[0]
    if empty.size:
        commits_all[empty] = np.frombuffer(hashlib.sha256(salt).digest(), np.uint8)
        for i in empty.tolist():
            out[i] = b""
    eps_e, r_e = effective_channel(r, sigma, block, tau, c0) if mode == "pixel" else (eps, r)
    hashes = verified = 0
    cov_sum = cov_n = 0.0
    exp_cov = 0.0
    dcs = []
    groups = np.unique(lens[lens > 0]).tolist()
    for nb in groups:
        idx = np.nonzero(lens == nb)[0]
        mat = np.frombuffer(b"".join([lines[i] for i in idx.tolist()]), np.uint8).reshape(-1, nb)
        commits = sha_rows(salt, mat.tobytes(), nb)
        commits_all[idx] = commits
        m = 8 * nb
        obs = make_observer(None, m, mode, rng, r, eps, sigma, block, tau, c0,
                            truth_fn=lambda ids, mat=mat: np.unpackbits(mat[ids], axis=1).astype(bool))
        rec, done, st = recognise_bytes(commits, salt, nb, obs, T_max, L, t, qmax, rng)
        recb = rec.tobytes()
        for j, i in enumerate(idx.tolist()):
            if done[j]:
                out[i] = recb[j * nb:(j + 1) * nb]
        hashes += st["hashes"]; verified += int(done.sum())
        cc = st["cover_cycle"][st["cover_cycle"] > 0]
        cov_sum += cc.sum(); cov_n += cc.size
        exp_cov += cc.size * theory(m, r_e, eps_e)["cover_mean"]
        dcs.append(st["done_cycle"][done])
    if commit_out:
        save_commit(commit_out, salt, commits_all)
        lens.astype(np.uint32).tofile(commit_out + ".lens")
    verified += int(empty.size)
    dt = time.perf_counter() - t0
    st = dict(n=n, verified=verified, hashes=hashes, seconds=dt, groups=len(groups))
    if verbose and n:
        dc = np.concatenate(dcs) if dcs else np.empty(0)
        avg_m = 8 * lens.mean()
        th = theory(max(int(avg_m), 8), r_e, eps_e)
        print(f"lines {n:,} ({len(groups)} distinct lengths, mean {avg_m:.0f} cells/line, "
              f"{int(empty.size):,} empty) mode={mode}")
        print(f"channel: r_eff={r_e:.3f} eps_eff={eps_e:.4g}  Z={th['Z']:.3f}  "
              f"exponent={th['exponent']:.3f}/cycle  capacity={th['capacity']:.3f} bit/cell/frame")
        if cov_n:
            print(f"cover time  observed mean {cov_sum / cov_n:.2f} | theory (length-weighted) {exp_cov / cov_n:.2f}")
        if dc.size:
            print(f"verify cycle mean {dc.mean():.2f}, p99 {np.percentile(dc, 99):.0f}, max {dc.max()}")
        print(f"verified {verified:,}/{n:,}  hashes {hashes:,}  recover {dt:.2f}s")
    return out, st


# ================================================== line-per-line datasets
# One integer / one string per line. Optical pipeline is ON by default here (--plain to skip).
LINE_CMDS = {"sort", "ints", "strings", "to-raw", "from-raw", "lines-gen", "gen-lines", "lines-bench"}


def load_ints(path):
    """One integer per line. Returns int64 ndarray, or a Python list if values overflow int64."""
    raw = open(path, "rb").read()
    if not raw.strip():
        return np.empty(0, np.int64)
    toks = None
    try:
        arr = np.fromstring(raw.decode("ascii"), dtype=np.int64, sep="\n")
        info = np.iinfo(np.int64)
        saturated = arr.size and (arr.max() == info.max or arr.min() == info.min)
        if not saturated:                           # NumPy clamps overflowing values to the limits
            if arr.size == raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1):
                return arr                          # no blank lines, nothing truncated
            toks = raw.split()
            if arr.size == len(toks):
                return arr
    except (ValueError, UnicodeDecodeError, DeprecationWarning):
        pass
    toks = toks or raw.split()
    vals = [int(t) for t in toks]                   # raises on genuinely bad lines
    try:
        return np.array(vals, dtype=np.int64)
    except OverflowError:
        return vals


def save_ints(path, arr):
    seq = arr.tolist() if isinstance(arr, np.ndarray) else arr
    with open(path, "w") as f:
        f.write("\n".join(map(str, seq)))
        if len(seq):
            f.write("\n")


def sort_ints(arr, desc=False, unique=False, optical=True, chan=None, threads=None):
    if isinstance(arr, list):                       # big ints: no vector path, no optical
        if optical:
            print("note: values exceed int64, optical stage skipped", file=sys.stderr)
        return sorted(set(arr) if unique else arr, reverse=desc)
    if optical:
        out, done, _ = run_arrays(arr, threads=threads, **(chan or {}))
    else:
        out = parallel_sort(arr, threads)
    if unique and out.size:
        out = out[np.concatenate(([True], out[1:] != out[:-1]))]
    return out[::-1].copy() if desc else out


def load_lines(path):
    """Returns (list of bytes lines, crlf flag, has_nul flag)."""
    raw = open(path, "rb").read()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    crlf = bool(lines) and all(l.endswith(b"\r") for l in lines)
    if crlf:
        lines = [l[:-1] for l in lines]
    return lines, crlf, b"\x00" in raw


def save_lines(path, lines, crlf=False):
    nl = b"\r\n" if crlf else b"\n"
    with open(path, "wb") as f:
        if lines:
            f.write(nl.join(lines)); f.write(nl)


def _bytewise_sort(lines, has_nul, threads):
    """Bytewise sort: NumPy 'S' arrays (memcmp order, needs no NUL bytes) with the parallel
    sample-sort for big inputs, otherwise Python's list sort."""
    n = len(lines)
    if n >= 200_000 and not has_nul:
        w = max(map(len, lines))
        if 0 < w <= 128 and w * n <= 1_500_000_000:
            return parallel_sort(np.array(lines, dtype=f"S{w}"), threads).tolist()
    lines = list(lines)
    lines.sort()
    return lines


def sort_strings(lines, has_nul=False, desc=False, unique=False, ignore_case=False,
                 optical=True, chan=None, threads=None):
    if optical:
        res, _ = recover_lines(lines, **(chan or {}))
        lost = sum(x is None for x in res)
        lines = [x for x in res if x is not None]
        if lost:
            print(f"WARNING: {lost:,} strings excluded (unverified)", file=sys.stderr)
    out = sorted(lines, key=bytes.lower) if ignore_case else _bytewise_sort(lines, has_nul, threads)
    if unique:
        key = bytes.lower if ignore_case else (lambda x: x)
        res, prev = [], None
        for l in out:
            k = key(l)
            if k != prev:
                res.append(l); prev = k
        out = res
    if desc:
        out.reverse()
    return out


ALPHABETS = {"lower": b"abcdefghijklmnopqrstuvwxyz",
             "alnum": b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
             "ascii": bytes(range(33, 127))}


def stream_gen_ints(out, n, lo=-10**12, hi=10**12, seed=0, chunk=1_000_000):
    """Write n random integers in [lo, hi], one per line, streaming in chunks (flat memory).
    `out` is a binary file object."""
    rng = np.random.default_rng(seed)
    left = n
    while left:
        c = min(chunk, left)
        arr = rng.integers(lo, hi, c, dtype=np.int64, endpoint=True)
        out.write(("\n".join(map(str, arr.tolist())) + "\n").encode())
        left -= c
    return n


VOWELS, CONSONANTS = b"aeiou", b"bcdfghjklmnprstvwz"


def stream_gen_strings(out, n, min_len=3, max_len=20, alphabet="lower", seed=0, chunk=1_000_000,
                       style="words", space_prob=0.0):
    """Write n string lines, streaming in chunks (flat memory). Lengths are uniform in
    [min_len, max_len] (min_len 0 gives empty lines).
      style "words"  : pronounceable (consonant/vowel alternating), like real words
      style "random" : characters drawn uniformly from `alphabet` (lower|alnum|ascii|literal)
      space_prob     : chance that an interior character becomes a space (multi-word lines)
    Fully vectorised: one mask-gather per chunk."""
    rng = np.random.default_rng(seed)
    if style == "random":
        al = np.frombuffer(ALPHABETS.get(alphabet, alphabet if isinstance(alphabet, bytes)
                                         else alphabet.encode()), np.uint8)
        if not al.size or (al == 10).any() or (al == 13).any():
            raise ValueError("alphabet must be non-empty and contain no newline characters")
    elif style != "words":
        raise ValueError("style must be 'words' or 'random'")
    cons, vow = np.frombuffer(CONSONANTS, np.uint8), np.frombuffer(VOWELS, np.uint8)
    w = max_len
    cols = np.arange(w + 1)
    chunk = max(1, min(chunk, 50_000_000 // (w + 1)))
    left = n
    while left:
        c = min(chunk, left)
        lens = rng.integers(min_len, max_len + 1, c)
        shape = (c, w + 1)
        if style == "random":
            M = al[rng.integers(0, al.size, shape)]
        else:
            M = np.where(cols[None, :] % 2 == 0, cons[rng.integers(0, cons.size, shape)],
                         vow[rng.integers(0, vow.size, shape)])
        if space_prob > 0:                                      # interior spaces, never doubled
            sp = (rng.random(shape) < space_prob) & (cols[None, :] >= 1) & \
                 (cols[None, :] < lens[:, None] - 1)
            sp[:, 1:] &= ~sp[:, :-1]
            M[sp] = 32
        M[cols[None, :] == lens[:, None]] = 10                  # newline right after each line
        out.write(M[cols[None, :] <= lens[:, None]].astype(np.uint8).tobytes())
        left -= c
    return n


def gen_ints(n, seed=0):                                    # small in-memory helpers
    return np.random.default_rng(seed).integers(-10**12, 10**12, n, dtype=np.int64)


def gen_strings(n, seed=0):
    import io
    buf = io.BytesIO(); stream_gen_strings(buf, n, seed=seed)
    return buf.getvalue().split(b"\n")[:-1]


def sniff_kind(path, sample=1 << 20):
    """Guess what a line-per-line text file holds: 'ints', 'strings', or None if it looks binary."""
    import re
    with open(path, "rb") as f:
        raw = f.read(sample)
    if not raw:
        return "strings"
    body = raw if len(raw) < sample else raw[:raw.rfind(b"\n") + 1] or raw   # drop cut-off last line
    if body.translate(None, bytes(range(32, 127)) + b"\n\r\t") and b"\x00" in body[:4096]:
        return None
    try:
        body.decode("utf-8")
    except UnicodeDecodeError:
        if b"\x00" in body:
            return None
    toks = body.split()
    if toks and all(re.fullmatch(rb"[+-]?\d+", t) for t in toks[:200_000]) and \
            len(toks) >= body.count(b"\n") * 0.99:
        return "ints"
    return "strings"


def looks_like_text(path):
    with open(path, "rb") as f:
        b = f.read(65536)
    return len(b) >= 256 and not b.translate(None, bytes(range(32, 127)) + b"\n\r\t")


def line_main(argv=None, default_optical=True):
    ap = argparse.ArgumentParser(prog="optical_sort.py (lines)", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    num = lambda s: int(s.replace("_", ""))

    def common(p, strings):
        p.add_argument("--desc", action="store_true"); p.add_argument("--unique", action="store_true")
        if strings:
            p.add_argument("--ignore-case", action="store_true")
        p.add_argument("--optical", action="store_true", help="route through optical pipeline")
        p.add_argument("--plain", action="store_true", help="skip optical pipeline (fast path only)")
        p.add_argument("--mode", choices=["bit", "pixel"], default="bit")
        p.add_argument("--r", type=float, default=0.45); p.add_argument("--eps", type=float, default=0.02)
        p.add_argument("--sigma", type=float, default=40.0); p.add_argument("--block", type=int, default=16)
        p.add_argument("--tau", type=int, default=127); p.add_argument("--c0", type=int, default=25)
        p.add_argument("--tmax", type=int, default=800)
        p.add_argument("--chase-L", type=int, default=12); p.add_argument("--chase-t", type=int, default=2)
        p.add_argument("--qmax", type=int, default=20_000); p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threads", type=int)

    g = sp.add_parser("gen", help="generate a line-per-line dataset, streamed (path '-' = stdout)")
    g.add_argument("target", nargs="+", metavar="[ints|strings] PATH",
                   help="PATH, optionally preceded by the kind; lines are STRINGS by default")
    g.add_argument("--n", type=num, default=100_000); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--min-len", type=int, default=3); g.add_argument("--max-len", type=int, default=20)
    g.add_argument("--style", choices=["words", "random"], default="words",
                   help="strings: pronounceable words (default) or uniform random characters")
    g.add_argument("--alphabet", default="lower", help="style random: lower | alnum | ascii | literal")
    g.add_argument("--space-prob", type=float, default=0.0, help="strings: chance of a space inside a line")
    g.add_argument("--lo", type=num, default=-10**12); g.add_argument("--hi", type=num, default=10**12)
    g.add_argument("--chunk", type=num, default=1_000_000, help="lines per write")
    pi = sp.add_parser("ints"); pi.add_argument("src"); pi.add_argument("dst"); pi.add_argument("--commit-out"); common(pi, False)
    ps = sp.add_parser("strings"); ps.add_argument("src"); ps.add_argument("dst"); ps.add_argument("--commit-out"); common(ps, True)
    b = sp.add_parser("bench"); b.add_argument("kind", choices=["ints", "strings"]); b.add_argument("--n", type=num, default=100_000); common(b, True)
    so = sp.add_parser("sort", help="auto-detect ints vs strings in a text file and sort it")
    so.add_argument("src"); so.add_argument("dst"); so.add_argument("--commit-out"); common(so, True)
    tr = sp.add_parser("to-raw"); tr.add_argument("src"); tr.add_argument("dst")
    fr = sp.add_parser("from-raw"); fr.add_argument("src"); fr.add_argument("dst"); fr.add_argument("--dtype", default="int64")
    a = ap.parse_args(argv)
    if a.cmd == "sort":
        kind = sniff_kind(a.src)
        if kind is None:
            ap.error(f"{a.src} looks binary, not line-per-line text; use: run {a.src} <dst> --dtype uint32")
        a.cmd = kind
        print(f"detected {kind} (one per line)", file=sys.stderr)

    optical = getattr(a, "optical", False) or (default_optical and not getattr(a, "plain", False))
    def chan():
        c = dict(mode=a.mode, r=a.r, eps=a.eps, sigma=a.sigma, block=a.block, tau=a.tau,
                 c0=a.c0, T_max=a.tmax, L=a.chase_L, t=a.chase_t, qmax=a.qmax, seed=a.seed)
        if getattr(a, "commit_out", None):
            c["commit_out"] = a.commit_out
        return c

    if a.cmd == "gen":
        if len(a.target) > 2 or (len(a.target) == 2 and a.target[0] not in ("ints", "strings")):
            ap.error("gen: expected [ints|strings] PATH")
        kind, path = (a.target if len(a.target) == 2 else ("strings", a.target[0]))
        to_stdout = path == "-"
        f = sys.stdout.buffer if to_stdout else open(path, "wb")
        t = time.perf_counter()
        try:
            if kind == "ints":
                stream_gen_ints(f, a.n, a.lo, a.hi, a.seed, a.chunk)
            else:
                stream_gen_strings(f, a.n, a.min_len, a.max_len, a.alphabet, a.seed, a.chunk,
                                   a.style, a.space_prob)
        finally:
            if to_stdout:
                f.flush()
            else:
                f.close()
        print(f"wrote {a.n:,} {kind}, one per line, in {time.perf_counter() - t:.2f}s -> "
              f"{'stdout' if to_stdout else path}", file=sys.stderr if to_stdout else sys.stdout)
    elif a.cmd == "to-raw":
        arr = load_ints(a.src)
        save_raw(np.asarray(arr, dtype=np.int64), a.dst); print(f"{len(arr):,} ints -> {a.dst} (int64 raw)")
    elif a.cmd == "from-raw":
        arr = load_raw(a.src, a.dtype); save_ints(a.dst, arr); print(f"{len(arr):,} ints -> {a.dst}")
    elif a.cmd == "ints":
        arr = load_ints(a.src); t = time.perf_counter()
        out = sort_ints(arr, a.desc, a.unique, optical, chan(), a.threads)
        dt = time.perf_counter() - t
        save_ints(a.dst, out); print(f"sorted {len(out):,} ints in {dt:.3f}s -> {a.dst}")
    elif a.cmd == "strings":
        lines, crlf, nul = load_lines(a.src); t = time.perf_counter()
        out = sort_strings(lines, nul, a.desc, a.unique, a.ignore_case, optical, chan(), a.threads)
        dt = time.perf_counter() - t
        save_lines(a.dst, out, crlf); print(f"sorted {len(out):,} strings in {dt:.3f}s -> {a.dst}")
    else:                                            # bench: text round trip through temp files
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
            if a.kind == "ints":
                with open(src, "wb") as f:
                    stream_gen_ints(f, a.n, seed=a.seed)
                arr = load_ints(src); ref = sorted(arr.tolist())
                t = time.perf_counter(); out = sort_ints(arr, a.desc, a.unique, optical, chan(), a.threads); dt = time.perf_counter() - t
                save_ints(dst, out); got = load_ints(dst).tolist()
                ref = sorted(set(ref)) if a.unique else ref
            else:
                with open(src, "wb") as f:
                    stream_gen_strings(f, a.n, seed=a.seed)
                lines, crlf, nul = load_lines(src)
                ref = sorted(lines, key=bytes.lower if a.ignore_case else None)
                t = time.perf_counter(); out = sort_strings(lines, nul, a.desc, a.unique, a.ignore_case, optical, chan(), a.threads); dt = time.perf_counter() - t
                save_lines(dst, out, crlf); got = load_lines(dst)[0]
                if a.unique:
                    seen, r2 = None, []
                    for l in ref:
                        k = l.lower() if a.ignore_case else l
                        if k != seen:
                            r2.append(l); seen = k
                    ref = r2
            ref = ref[::-1] if a.desc else ref
            norm = (lambda L: [x.lower() for x in L]) if a.ignore_case else (lambda L: L)
            ok = norm(got) == norm(ref)
            print(f"{a.kind}: n={a.n:,} optical={optical} sort {dt:.3f}s | text round-trip vs Python: "
                  f"{'OK' if ok else 'MISMATCH'}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in LINE_CMDS:
        rename = {"lines-gen": "gen", "gen-lines": "gen", "lines-bench": "bench"}
        return line_main([rename.get(sys.argv[1], sys.argv[1])] + sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    num = lambda s: int(s.replace("_", ""))

    def chan(p):
        p.add_argument("--dtype", default="uint32")
        p.add_argument("--mode", choices=["bit", "pixel"], default="bit")
        p.add_argument("--r", type=float, default=0.45); p.add_argument("--eps", type=float, default=0.02)
        p.add_argument("--sigma", type=float, default=40.0); p.add_argument("--block", type=int, default=16)
        p.add_argument("--tau", type=int, default=127); p.add_argument("--c0", type=int, default=25)
        p.add_argument("--tmax", type=int, default=800)
        p.add_argument("--chase-L", type=int, default=12); p.add_argument("--chase-t", type=int, default=2)
        p.add_argument("--qmax", type=int, default=20_000); p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threads", type=int)

    g = sp.add_parser("gen"); g.add_argument("path"); g.add_argument("--n", type=num, default=100_000); g.add_argument("--dtype", default="uint32")
    r_ = sp.add_parser("run"); r_.add_argument("src"); r_.add_argument("dst"); r_.add_argument("--commit-out"); r_.add_argument("--mmap", action="store_true"); chan(r_)
    b = sp.add_parser("bench"); b.add_argument("--n", type=num, default=50_000); chan(b)
    a = ap.parse_args()

    kw = lambda: dict(mode=a.mode, r=a.r, eps=a.eps, sigma=a.sigma, block=a.block, tau=a.tau, c0=a.c0,
                      T_max=a.tmax, L=a.chase_L, t=a.chase_t, qmax=a.qmax, seed=a.seed, threads=a.threads)
    if a.cmd == "gen":
        save_raw(_gen(a.n, a.dtype), a.path); print(f"wrote {a.n:,} x {a.dtype} -> {a.path}")
    elif a.cmd == "run":
        size, item = os.path.getsize(a.src), np.dtype(a.dtype).itemsize
        if size % item or looks_like_text(a.src):
            kind = sniff_kind(a.src) or "strings"
            sys.exit(f"error: {a.src} ({size:,} bytes) is not raw {a.dtype} data - `run` reads fixed-width\n"
                     f"binary records. It looks like a line-per-line text file ({kind}). Use:\n"
                     f"  python optical_sort.py sort {a.src} sorted.txt        (auto-detects ints/strings)\n"
                     f"  python optical_sort.py {kind} {a.src} sorted.txt [--plain]")
        data = load_raw(a.src, a.dtype, mmap=a.mmap)
        res, done, st = run_arrays(data, commit_out=a.commit_out, **kw())
        save_raw(res, a.dst); print(f"saved {len(res):,} sorted records -> {a.dst}")
    else:
        data = _gen(a.n, a.dtype, a.seed)
        res, done, st = run_arrays(data, **kw())
        ref = np.sort(data)
        print("EXACT MATCH with np.sort" if np.array_equal(res, ref) else
              f"differs from np.sort ({len(res)} vs {len(ref)} records)")


if __name__ == "__main__":
    main()
