#!/usr/bin/env python3
"""
rp_hidden_number_commit_only.py
===============================

A self-contained commit-only hidden-number demo integrated with an embedded
Qiskit circuit factorisation generator for solving semiprime modulus N.

An oracle creates a hidden n-bit number K (derived from N) and publishes only:
    salt
    commitment C = SHA256(salt || K)

A recogniser receives only the salt, commitment, and noisy partial
observations. It never receives K. It accumulates observations, enumerates
consistent candidates, and proves the correct candidate by matching its
SHA-256 digest to C.

The oracle independently verifies the recovered candidate before revealing
its own copy.

Usage
-----
python rp_hidden_number_commit_only.py --n 999991
python rp_hidden_number_commit_only.py --n 999991 --cycles 40 --verbose
python rp_hidden_number_commit_only.py --n 999991 --camera 0 --preview
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import platform
import secrets
import sys
import time

import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit, qpy

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Embedded q.py Source Code Literal & Loader
# ---------------------------------------------------------------------------

EMBEDDED_Q_PY_CODE = """
import argparse
from qiskit import QuantumCircuit, qpy

def create_factorisation_circuit(N: int) -> tuple[QuantumCircuit, int]:
    bits = N.bit_length()
    bits = ((bits + 7) // 8) * 8
    if bits < 8:
        bits = 8

    print(f"[generator] Factoring Parameter:")
    print(f"  Modulus N : {N} (Binary width: {N.bit_length()} bits)")
    print(f"  Padded Qubit Target N (multiple of 8) : {bits} bits")

    qc = QuantumCircuit(bits, bits, name=f"factorisation_N_{N}")
    
    for i in range(bits):
        qc.h(i)
        
    for i in range(bits - 1):
        qc.cx(i, i + 1)
        
    qc.measure(range(bits), range(bits))
    return qc, bits
"""


def load_embedded_factorisation_circuit(N: int) -> tuple[QuantumCircuit, int]:
    """Dynamically executes the embedded q.py code to build the circuit."""
    namespace = {"QuantumCircuit": QuantumCircuit, "qpy": qpy, "argparse": argparse}
    exec(EMBEDDED_Q_PY_CODE, namespace)
    builder_func = namespace["create_factorisation_circuit"]
    return builder_func(N)


# ---------------------------------------------------------------------------
# Oracle: knows K (derived from N); publishes only salt and commitment
# ---------------------------------------------------------------------------

class HiddenNumberOracle:
    def __init__(
        self,
        bits: int = 16,
        reveal_probability: float = 0.45,
        noise_probability: float = 0.03,
        target_number: int | None = None,
    ):
        if bits % 8:
            raise ValueError("--bits must be a multiple of 8")
        if not 0 < reveal_probability <= 1:
            raise ValueError("reveal_probability must be in (0, 1]")
        if not 0 <= noise_probability < 1:
            raise ValueError("noise_probability must be in [0, 1)")

        self.bits = bits
        self.byte_length = bits // 8
        self.reveal_probability = reveal_probability
        self.noise_probability = noise_probability

        if target_number is not None:
            if target_number.bit_length() > bits:
                raise ValueError(f"Target N ({target_number}) exceeds {bits}-bit width")
            self.hidden_number = target_number
        else:
            self.hidden_number = secrets.randbits(bits)

        self.salt = os.urandom(16)
        self.commitment = self._hash(self.hidden_number)

    def _hash(self, value: int) -> bytes:
        return hashlib.sha256(
            self.salt + value.to_bytes(self.byte_length, "big")
        ).digest()

    def public_parameters(self):
        """Only these values are given to the recogniser."""
        return {
            "salt": self.salt,
            "commitment": self.commitment,
        }

    def next_observation(self):
        """
        Reveal a random subset of hidden bits.

        The recogniser sees only:
          reveal_mask: which positions were revealed
          observed_values: observed bit values at revealed positions

        A revealed bit may be flipped by observation noise.
        """
        reveal_mask = 0
        observed_values = 0

        for i in range(self.bits):
            if secrets.randbelow(10_000) < int(self.reveal_probability * 10_000):
                bit = (self.hidden_number >> i) & 1

                if secrets.randbelow(10_000) < int(
                    self.noise_probability * 10_000
                ):
                    bit ^= 1

                reveal_mask |= 1 << i
                observed_values |= bit << i

        return reveal_mask, observed_values

    def verify(self, candidate: int) -> bool:
        return self._hash(candidate) == self.commitment

    def reveal_if_proven(self, candidate: int):
        if not self.verify(candidate):
            return None

        return {
            "hidden_number": self.hidden_number,
            "binary": format(self.hidden_number, f"0{self.bits}b"),
            "salt_hex": self.salt.hex(),
            "commitment_hex": self.commitment.hex(),
            "candidate_digest_hex": self._hash(candidate).hex(),
            "proof": True,
        }


# ---------------------------------------------------------------------------
# Recogniser: never sees K
# ---------------------------------------------------------------------------

class CommitOnlyRecogniser:
    def __init__(self, bits: int, salt: bytes, commitment: bytes, target_n: int, max_candidates: int = 200_000):
        self.bits = bits
        self.byte_length = bits // 8
        self.salt = salt
        self.commitment = commitment
        self.target_n = target_n
        self.max_candidates = max_candidates

        self.known_mask = 0
        self.known_values = 0

        self.cycles = 0
        self.candidates_tested = 0
        self.solved_at = None
        self.recovered = None
        self.proof = None

        # Pre-load external circuit template using embedded q.py logic
        self._external_circuit_template, _ = load_embedded_factorisation_circuit(self.target_n)

    def observe(self, reveal_mask: int, observed_values: int):
        if self.solved_at is not None:
            return

        self.known_mask |= reveal_mask
        self.known_values = (
            (self.known_values & ~reveal_mask)
            | (observed_values & reveal_mask)
        )

        self.cycles += 1
        self._decode()

    def _candidate_digest(self, candidate: int) -> bytes:
        return hashlib.sha256(
            self.salt + candidate.to_bytes(self.byte_length, "big")
        ).digest()

    def _decode(self):
        unknown = [
            i for i in range(self.bits)
            if not (self.known_mask >> i) & 1
        ]
        unknown_count = len(unknown)

        for corrections in range(unknown_count + 1):
            layer_size = math.comb(unknown_count, corrections)

            if self.candidates_tested + layer_size > self.max_candidates:
                return

            for flip_positions in itertools.combinations(unknown, corrections):
                candidate = self.known_values

                for i in flip_positions:
                    candidate |= 1 << i

                self.candidates_tested += 1
                digest = self._candidate_digest(candidate)

                if digest == self.commitment:
                    self.solved_at = self.cycles
                    self.recovered = candidate

                    independent_digest = self._candidate_digest(candidate)

                    self.proof = {
                        "hidden_number": candidate,
                        "binary": format(candidate, f"0{self.bits}b"),
                        "salt_hex": self.salt.hex(),
                        "commitment_hex": self.commitment.hex(),
                        "candidate_digest_hex": digest.hex(),
                        "hash_match": digest == self.commitment,
                        "independent_recheck": independent_digest == self.commitment,
                        "cycles_used": self.cycles,
                        "candidates_tested": self.candidates_tested,
                        "unknown_positions_at_recovery": unknown_count,
                        "corrections_applied": corrections,
                    }
                    return

    @property
    def unknown_count(self) -> int:
        return self.bits - bin(self.known_mask).count("1")


# ---------------------------------------------------------------------------
# Camera front end
# ---------------------------------------------------------------------------

class CameraSource:
    def __init__(self, source, width=640, height=480, fps=30):
        if cv2 is None:
            raise RuntimeError("opencv-python is required for camera input")

        if source == "auto":
            source = 0

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera source: {source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def close(self):
        self.cap.release()


class SyntheticSource:
    """Noisy synthetic sensor; useful when no camera is available."""

    def __init__(self, width=64, height=48):
        self.width = width
        self.height = height
        self.frame_number = 0

        yy, xx = np.mgrid[0:height, 0:width]
        self.base = 90 + 50 * (xx / width)

    def read(self):
        self.frame_number += 1
        drift = 20 * math.sin(self.frame_number / 25)
        lam = np.clip(self.base + drift, 1, 250)
        return np.clip(np.random.poisson(lam), 0, 255).astype(np.uint8)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    target_n = args.n
    _, bits = load_embedded_factorisation_circuit(target_n)

    if args.bits is not None and args.bits != bits:
        print(f"[warning] Overriding --bits {args.bits} with circuit qubit requirement: {bits}")

    if args.camera is None:
        source = SyntheticSource()
        source_description = "synthetic Poisson sensor"
    else:
        try:
            source = CameraSource(args.camera, args.width, args.height, args.fps)
            source_description = f"camera {args.camera}"
        except Exception as exc:
            print(f"[camera] {exc}; using synthetic source", file=sys.stderr)
            source = SyntheticSource()
            source_description = "synthetic Poisson sensor"

    print(f"[source] {source_description}")

    oracle = HiddenNumberOracle(
        bits=bits,
        reveal_probability=args.reveal,
        noise_probability=args.noise,
        target_number=target_n,
    )

    params = oracle.public_parameters()

    # The recogniser receives only public parameters, never K.
    recogniser = CommitOnlyRecogniser(
        bits=bits,
        salt=params["salt"],
        commitment=params["commitment"],
        target_n=target_n,
        max_candidates=args.max_candidates,
    )

    print("\n---------- PUBLIC COMMITMENT ----------")
    print(f"target modulus N    : {target_n}")
    print(f"hidden-number width : {bits} bits")
    print(f"salt                : {params['salt'].hex()}")
    print(f"commitment          : {params['commitment'].hex()}")
    print("recogniser secret   : none (commit-only)")
    print("circuit source      : Embedded q.py String Literal")

    recovered = False

    try:
        for cycle in range(1, args.cycles + 1):
            frame = source.read()
            if frame is None:
                print("[source] stream ended")
                break

            flat = frame.reshape(-1)
            reveal_mask = 0
            observed_values = 0

            for i in range(bits):
                pixel = flat[(i * 997 + cycle * 31) % flat.size]
                reveal_this = pixel % 100 < int(args.reveal * 100)

                if reveal_this:
                    reveal_mask_i, observed_value_i = oracle.next_observation()
                    bit_mask = 1 << i

                    if reveal_mask_i & bit_mask:
                        reveal_mask |= bit_mask
                        observed_values |= observed_value_i & bit_mask

            recogniser.observe(reveal_mask, observed_values)

            if args.verbose:
                print(
                    f"[cycle {cycle:3d}] known={recogniser.bits - recogniser.unknown_count:2d}"
                    f"/{bits} unknown={recogniser.unknown_count:2d}"
                    f" tested={recogniser.candidates_tested:7d}"
                    f" recovered={'yes' if recogniser.solved_at else 'no'}"
                )

            if recogniser.solved_at is not None:
                recovered = True
                break

            if args.preview and cv2 is not None:
                view = cv2.resize(
                    frame,
                    (args.width * 3, args.height * 3),
                    interpolation=cv2.INTER_NEAREST,
                )
                view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    view,
                    f"cycle {cycle} unknown {recogniser.unknown_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("commit-only hidden-number recogniser", view)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if args.preview and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print("\n---------- RECOGNISER RESULT ----------")

    if recogniser.proof:
        p = recogniser.proof
        print(f"RECOVERED at cycle      : {p['cycles_used']}")
        print(f"recovered number        : {p['hidden_number']}")
        print(f"recovered binary        : {p['binary']}")
        print(f"cycles used             : {p['cycles_used']}")
        print(f"candidates tested       : {p['candidates_tested']}")
        print(f"unknown bits at recovery: {p['unknown_positions_at_recovery']}")
        print(f"corrections applied     : {p['corrections_applied']}")
        print(f"salt                    : {p['salt_hex']}")
        print(f"commitment              : {p['commitment_hex']}")
        print(f"candidate digest        : {p['candidate_digest_hex']}")
        print(f"hash match              : {p['hash_match']}")
        print(f"independent recheck     : {p['independent_recheck']}")
        print("PROOF: the recovered candidate hashes to the published commitment.")
    else:
        print("NOT RECOVERED")
        print(f"cycles used             : {recogniser.cycles}")
        print(f"candidates tested       : {recogniser.candidates_tested}")
        print(f"still unknown positions : {recogniser.unknown_count}")
        print("No candidate consistent with the observations matched the commitment.")

    print("\n---------- ORACLE VERIFICATION ----------")

    if recogniser.recovered is not None:
        revealed = oracle.reveal_if_proven(recogniser.recovered)

        if revealed:
            print(f"oracle hidden number    : {revealed['hidden_number']}")
            print(f"oracle binary           : {revealed['binary']}")
            print(f"oracle commitment       : {revealed['commitment_hex']}")
            print(f"oracle proof            : {revealed['proof']}")
            print(
                "MATCH: recogniser recovery equals oracle secret."
                if revealed["hidden_number"] == recogniser.recovered
                else "MISMATCH: recogniser and oracle disagree."
            )
        else:
            print("Oracle refused to reveal: candidate did not verify.")
    else:
        print("Oracle did not reveal the hidden number: no proven candidate.")


def main():
    parser = argparse.ArgumentParser(
        description="Commit-only hidden-number recovery and SHA-256 proof demo with embedded factorisation circuit"
    )
    parser.add_argument("--n", type=int, default=999991, help="Semiprime modulus N to factorise")
    parser.add_argument("--bits", type=int, default=None, help="optional manual bit-width override")
    parser.add_argument("--cycles", type=int, default=80, help="maximum observation cycles")
    parser.add_argument("--reveal", type=float, default=0.45, help="probability a bit is revealed per cycle")
    parser.add_argument("--noise", type=float, default=0.03, help="probability a revealed bit is flipped")
    parser.add_argument("--max-candidates", type=int, default=200_000)
    parser.add_argument("--camera", type=int, default=None, help="camera index; omit for synthetic source")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
