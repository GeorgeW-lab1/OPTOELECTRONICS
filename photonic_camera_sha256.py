import hashlib
import time
from typing import Optional, Tuple

import numpy as np


class PhotonicCameraRecovery:
    """
    Camera-based photonic recovery of the unknown input to a SHA-256 target.

        target = SHA256(known_data || frame_bytes)

    There is no integer/bit encoding of the nonce. The unknown "nonce" is the
    spatial pattern of lit pixels in an (rows x cols) sensor frame, and the hash
    input is simply the raw frame (row-major, one byte per pixel, 0 or 1).

    The source is AND-gated: pixel i can only ever light if K_i = 1, and then
    only with probability p per exposure. The camera reads ALL n pixels in one
    exposure, so each frame costs one exposure, not n sequential reads.

    Recovery is O(n):
      * Level 1: accumulate lit pixels (cumulative OR). One lit pixel proves K_i = 1.
      * Level 2: no subset enumeration. B_m is hashed only when it gains a new
        pixel, so the solver computes at most (#lit pixels) <= n hashes in total.
    Expected exposures ~ H_k / p for k lit pixels (coupon-collector).
    """

    def __init__(self, pattern: np.ndarray, known_data: bytes, p: float = 0.01,
                 noise_sigma: float = 0.08, seed: Optional[int] = None):
        self.secret = (np.asarray(pattern) > 0).astype(np.uint8)    # physical source only
        self.shape = self.secret.shape
        self.n = self.secret.size
        self.p = p
        self.noise_sigma = noise_sigma
        self.known_data = known_data
        self.rng = np.random.default_rng(seed)
        self.target = self._digest(self.secret)                      # the only public output

    def _digest(self, frame: np.ndarray) -> str:
        return hashlib.sha256(self.known_data + frame.astype(np.uint8).tobytes()).hexdigest()

    def expose(self, count: int) -> np.ndarray:
        """`count` camera frames: shape (count, n), thresholded after sensor noise."""
        R = self.rng.random((count, self.n)) < self.p                 # photon arrivals R_i(t)
        X = R & self.secret.reshape(1, -1).astype(bool)               # X = K AND R
        S = X.astype(np.float32) + self.rng.normal(0, self.noise_sigma, X.shape).astype(np.float32)
        return S > 0.5                                                # detected clicks

    def show(self, frame: np.ndarray) -> str:
        return "\n".join("        " + "".join("#" if v else "." for v in row) for row in frame)

    def run(self, max_frames: int = 5_000_000, chunk: int = 4096,
            verbose: bool = True) -> Tuple[bool, Optional[np.ndarray], int, int]:
        if verbose:
            print(f"[*] Camera recovery: {self.shape[0]}x{self.shape[1]} sensor ({self.n} pixels), p={self.p}")
            print(f"[*] Known data : {self.known_data!r}")
            print(f"[*] Target hash: {self.target}")

        lit = np.zeros(self.n, dtype=np.uint8)       # B_m
        frames_done = 0
        hashes = 0

        while frames_done < max_frames:
            c = min(chunk, max_frames - frames_done)
            D = self.expose(c)
            first_hit = np.where(D.any(axis=0), D.argmax(axis=0), -1)

            # replay this chunk in time order: only a NEW lit pixel changes B_m
            for i in sorted((i for i in range(self.n) if first_hit[i] >= 0 and not lit[i]),
                            key=lambda i: first_hit[i]):
                lit[i] = 1
                m = frames_done + int(first_hit[i]) + 1
                hashes += 1                           # one hash per change => <= n total
                ok = self._digest(lit.reshape(self.shape)) == self.target
                if verbose:
                    print(f"    frame m={m:7d} | lit={int(lit.sum()):2d}/{self.n} | pixel {i:2d} appeared | hashes={hashes}")
                if ok:
                    frame = lit.reshape(self.shape).copy()
                    if verbose:
                        print(f"\n[+] RECOVERED at frame m={m} after {hashes} hashes (n={self.n})")
                        print(self.show(frame))
                        print(f"    Digest : {self._digest(frame)}")
                        print(f"    Matches target: True")
                    return True, frame, m, hashes
            frames_done += c

        if verbose:
            print(f"\n[-] Not recovered within {max_frames} frames ({hashes} hashes).")
        return False, None, max_frames, hashes


if __name__ == "__main__":
    # Arbitrary chosen spatial pattern (4x8 = 32 pixels); not mined, not integer-encoded.
    PATTERN = np.array([[int(c) for c in row] for row in (
        "1",
        
    )])

    sim = PhotonicCameraRecovery(PATTERN, known_data=b"known_header_data|ts=1789000000",
                                 p=0.00006, seed=42)
    t0 = time.time()
    sim.run(max_frames=5_000_000)
    print(f"    time   : {time.time() - t0:.2f}s")
