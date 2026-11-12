
#!/usr/bin/env python3
"""
photon_preimage.py - Direct photon preimage decoder.

NO BRUTE FORCE.
NO CANDIDATE ENUMERATION.
NO BIT FLIPPING.
NO SALT GRINDING.
NO RETRIES.

Architecture:
    1. Generate a secret and a random salt.
    2. Create one SHA-256 commitment.
    3. Load the secret into the simulated photon circuit.
    4. Collect photon measurements.
    5. A zero observed in any measurement identifies a zero bit.
    6. Bits never observed as zero are decoded as one.
    7. Construct exactly ONE candidate.
    8. Perform exactly ONE SHA-256 verification.

The photon circuit is simulated. It is not an independent
physical SHA-256 solver or a cryptographic preimage attack.
"""

import argparse
import hashlib
import secrets

import numpy as np


VCC = 5.0
V_BE = 0.7

TAU = 10e3 * 15e-6
CLK = 0.100
FRAME_DT = 5e-7


def commit(salt, value, nbits):
    width = (nbits + 3) // 4
    encoded = f"{salt}:{value:0{width}X}".encode()
    return hashlib.sha256(encoded).hexdigest()


def lz_bits(hexdigest):
    return 256 - int(hexdigest, 16).bit_length()


def zeros_info(h):
    hexz = len(h) - len(h.lstrip("0"))
    return f"{h} [{lz_bits(h)} zero bits, {hexz} hex 0s]"


class PhotonCircuit:

    def __init__(self, secret, nbits, rng,
                 mean_photons=40.0, thr=0.45):

        self.secret = secret
        self.nbits = nbits
        self.rng = rng

        self.lam = mean_photons
        self.thr = thr
        self.hyst = 0.03

        self.state = False
        self.vc = 0.0
        self.t = 0.0

        self.alpha = 1 - np.exp(-FRAME_DT / TAU)

    def _b_out(self, intensity):

        if self.state and intensity < self.thr - self.hyst:
            self.state = False

        elif not self.state and intensity > self.thr + self.hyst:
            self.state = True

        drive = VCC if self.state else 0.0

        self.vc += (drive - self.vc) * self.alpha

        return self.vc > V_BE

    def word(self):

        bits = 0
        frames = int(round(CLK / FRAME_DT))

        for slot in range(self.nbits):

            t = self.t + FRAME_DT * np.arange(1, frames + 1)
            self.t = float(t[-1])

            nB = self.rng.poisson(
                self.lam * (1 + 0.8 * np.sin(2 * np.pi * 0.4 * t))
            )

            nA = self.rng.poisson(
                self.lam * (1 + 0.8 * np.sin(2 * np.pi * 0.7 * t + 1))
            )

            inten = nB / (2 * self.lam)
            a_odd = nA & 1

            d = False

            for i in range(frames):
                b = self._b_out(inten[i])
                d = bool(a_odd[i]) and b

            y = (self.secret >> (self.nbits - 1 - slot)) & 1

            # Z-channel:
            # Secret 1 always produces 1.
            # Secret 0 may produce a noise-induced 1.

            measured = bool(y) or d

            bits = (bits << 1) | int(measured)

        return bits


class PhotonStatistics:

    def __init__(self, nbits):

        self.nbits = nbits
        self.n = 0
        self.zeros = np.zeros(nbits, dtype=np.int64)

    def update(self, word):

        self.n += 1

        for i in range(self.nbits):

            bit = (word >> (self.nbits - 1 - i)) & 1

            if bit == 0:
                self.zeros[i] += 1

    def unknown_positions(self):

        return [
            i for i in range(self.nbits)
            if self.zeros[i] == 0
        ]

    def direct_candidate(self):

        candidate = 0

        for i in range(self.nbits):

            bit = 1 if self.zeros[i] == 0 else 0

            candidate = (candidate << 1) | bit

        return candidate


def decode_direct(stats, verify):

    """
    Construct one candidate directly from photon statistics.

    No search.
    No combinations.
    No permutations.
    No bit flipping.

    Exactly one verification.
    """

    candidate = stats.direct_candidate()

    verified = verify(candidate)

    return candidate, verified


def cmd_preimage_photon(a):

    n = max(4, a.bits - a.bits % 4)

    rng = np.random.default_rng(a.seed)

    # Deterministic secret generation for the selected seed.
    secret_bytes = rng.bytes((n + 7) // 8)

    secret = (
        int.from_bytes(secret_bytes, "big")
        & ((1 << n) - 1)
    )

    # Generate salt ONCE. No grinding.
    salt = secrets.token_hex(8)

    # Generate commitment ONCE.
    target = commit(salt, secret, n)

    print("=" * 65)
    print("PHOTON PREIMAGE DECODER")
    print("=" * 65)

    print(f"Bits       : {n}")
    print(f"Salt       : {salt}")
    print(f"Target     : {zeros_info(target)}")
    print(f"Measurements: {a.samples}")
    print()

    stats = PhotonStatistics(n)

    circ = PhotonCircuit(
        secret,
        n,
        rng,
        mean_photons=a.photons,
        thr=a.threshold
    )

    # Acquire measurements.
    # These are observations, not candidate searches.

    for sample in range(1, a.samples + 1):

        measured = circ.word()

        stats.update(measured)

        print(
            f"measurement {sample:3d}: "
            f"{measured:0{(n + 3) // 4}X} "
            f"| observed zeros: "
            f"{int(stats.zeros.sum())}"
        )

    print()
    print("-" * 65)

    unknown = stats.unknown_positions()

    print(f"Measurements collected : {stats.n}")
    print(f"Positions never zero   : {len(unknown)}")
    print(f"Positions observed zero: {n - len(unknown)}")

    # Exactly ONE candidate is constructed.
    candidate = stats.direct_candidate()

    width = (n + 3) // 4

    print()
    print(f"Direct candidate : {candidate:0{width}X}")
    print(f"Actual secret    : {secret:0{width}X}")

    # Exactly ONE SHA-256 verification.
    verification_count = 0

    def verify(value):

        nonlocal verification_count

        verification_count += 1

        digest = commit(salt, value, n)

        print()
        print("SHA-256 VERIFICATION")
        print(f"Candidate hash : {zeros_info(digest)}")
        print(f"Target hash    : {target}")

        return digest == target

    found, success = decode_direct(stats, verify)

    print()
    print("=" * 65)

    if success:

        print("RESULT: PREIMAGE VERIFIED")

    else:

        print("RESULT: VERIFICATION FAILED")
        print("The photon measurements did not recover the committed value.")
        print("No alternative candidates were attempted.")

    print("=" * 65)

    print(f"Candidate attempts : 1")
    print(f"SHA-256 operations : {verification_count}")
    print(f"Candidate correct  : {found == secret}")

    assert verification_count == 1

    print()
    print("BRUTE FORCE OPERATIONS: 0")
    print("SALT GRINDING        : 0")
    print("BIT FLIPS            : 0")
    print("CANDIDATE RETRIES    : 0")
    print("TOTAL VERIFICATIONS  : 1")


def main():

    ap = argparse.ArgumentParser(
        description="Direct photon preimage decoder - no brute force"
    )

    ap.add_argument("--bits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--photons", type=float, default=40.0)
    ap.add_argument("--threshold", type=float, default=0.45)

    a = ap.parse_args()

    if a.bits < 4:
        ap.error("--bits must be at least 4")

    if a.samples < 1:
        ap.error("--samples must be at least 1")

    if a.photons <= 0:
        ap.error("--photons must be positive")

    if not 0 <= a.threshold <= 1:
        ap.error("--threshold must be between 0 and 1")

    cmd_preimage_photon(a)


if __name__ == "__main__":
    main()
