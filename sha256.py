
#!/usr/bin/env python3
"""
PHOTON TEMPORAL-CODE POW
Author: George Wagenknecht

Two-level photon recognition architecture.

LEVEL 1: PHYSICAL SIGNAL EXTRACTION
    Photon A / Photon B
        -> AND coincidence gate
        -> RC integration
        -> Hysteresis
        -> Reference channel
        -> Digital noise cancellation
        -> Residual temporal trace

LEVEL 2: TEMPORAL CODE RECOGNITION
    Residual trace
        -> Temporal correlation
        -> Two-level code recognition
        -> Bit confidence
        -> Reconstructed nonce

FINAL:
    Reconstructed nonce + block header
        -> Double SHA-256
        -> Target verification

NO BRUTE FORCE
NO NONCE INCREMENTING
NO CANDIDATE ENUMERATION
NO BIT FLIPPING
NO SALT GRINDING

The simulator encodes a known nonce in the photon statistics.
The decoder does not receive the nonce directly.

This is a classical stochastic simulation of a proposed
optoelectronic architecture, not a quantum hardware simulation.
"""

import argparse
import hashlib
import secrets
import struct
import time

import numpy as np


# =========================================================
# PHYSICAL PARAMETERS
# =========================================================

VCC = 5.0
V_BE = 0.7

TAU = 0.15
FRAME_DT = 0.005

VERSION = 1

# Two temporal codewords.
#
# Bit 1: first two chips active.
# Bit 0: last two chips active.
#
# The decoder does not enumerate codewords.
# It measures their temporal correlation.

CODE_ONE = np.array([1, 1, 0, 0], dtype=np.float64)
CODE_ZERO = np.array([0, 0, 1, 1], dtype=np.float64)

REFERENCE_GAIN = 0.18


# =========================================================
# HASHING
# =========================================================

def double_sha256(data):

    first = hashlib.sha256(data).digest()

    return hashlib.sha256(first).digest()


def leading_zero_bits(digest):

    return 256 - int.from_bytes(digest, "big").bit_length()


def make_target(difficulty):

    return (1 << (256 - difficulty)) - 1


# =========================================================
# INPUT PARSING
# =========================================================

def parse_integer(value):

    try:
        return int(value, 0)
    except ValueError:
        return int(value, 16)


def parse_hex(value):

    return bytes.fromhex(value.replace(" ", "").replace(":", ""))


# =========================================================
# PHOTON SOURCE
# =========================================================

class PhotonSource:

    """
    Simulated photon source.

    The source encodes the nonce in temporal photon
    statistics rather than passing the nonce to the decoder.

    For each secret bit, the source generates four temporal
    coincidence windows.

    Bit 1:
        High coincidence probability in windows 0 and 1.

    Bit 0:
        High coincidence probability in windows 2 and 3.

    Common-mode photon noise is shared with a reference
    detector channel.
    """

    def __init__(
        self,
        secret_nonce,
        nbits,
        rng,
        photons=40.0,
        signal_strength=2.5
    ):

        self._secret_nonce = secret_nonce

        self.nbits = nbits
        self.rng = rng

        self.photons = photons
        self.signal_strength = signal_strength

        self.total_photons = 0

        self.total_frames = 0
        self.total_coincidences = 0

        self.source_time = 0.0

        self.alpha = 1.0 - np.exp(
            -FRAME_DT / TAU
        )

    def _emit_chip(self, active):

        """
        Emit photons and measure one temporal chip.

        A and B are detected independently.
        Their coincidence is the AND gate output.

        R is a reference detector tracking common-mode noise.
        """

        vc = 0.0
        latch = False

        reference_sum = 0
        coincidence_sum = 0

        frames = 96

        base = 0.8

        signal = (
            self.signal_strength if active else 0.0
        )

        for _ in range(frames):

            self.source_time += FRAME_DT

            modulation = (
                1.0
                + 0.20 * np.sin(
                    2 * np.pi * 0.4 * self.source_time
                )
                + self.rng.normal(0.0, 0.08)
            )

            modulation = max(0.1, modulation)

            # Shared optical environment.
            common_noise = max(
                0.1,
                modulation * self.rng.lognormal(
                    mean=0.0,
                    sigma=0.10
                )
            )

            # Photon channels.
            nA = self.rng.poisson(
                self.photons
                * (base + signal)
                * common_noise
                / 40.0
            )

            nB = self.rng.poisson(
                self.photons
                * (base + signal)
                * common_noise
                / 40.0
            )

            nR = self.rng.poisson(
                self.photons
                * base
                * common_noise
                / 40.0
            )

            self.total_photons += nA + nB + nR
            self.total_frames += 1

            # Detector threshold.
            detector_a = nA >= 2
            detector_b = nB >= 2

            # Physical AND coincidence.
            and_output = detector_a and detector_b

            if and_output:
                self.total_coincidences += 1
                coincidence_sum += 1

            reference_output = nR >= 2

            reference_sum += int(reference_output)

            # RC integrator.
            drive = VCC if and_output else 0.0

            vc += (drive - vc) * self.alpha

            # Schmitt-trigger hysteresis.
            if latch and vc < 0.75:
                latch = False

            elif not latch and vc > 0.90:
                latch = True

        reference_rate = reference_sum / frames

        coincidence_rate = coincidence_sum / frames

        # Final RC state.
        #
        # This is the signal passed to the digital
        # noise-cancellation stage.

        return (
            float(latch),
            reference_rate,
            coincidence_rate
        )

    def capture_bit_trace(self, bit_index):

        """
        Generate one four-chip temporal trace.

        This method accesses the encoded source state.
        The downstream decoder does not.
        """

        source_bit = (
            self._secret_nonce
            >> (self.nbits - 1 - bit_index)
        ) & 1

        code = CODE_ONE if source_bit else CODE_ZERO

        trace = []

        for chip in range(4):

            latch, reference, coincidence = (
                self._emit_chip(bool(code[chip]))
            )

            trace.append({
                "latch": latch,
                "reference": reference,
                "coincidence": coincidence
            })

        return trace


# =========================================================
# LEVEL 1: DIGITAL NOISE CANCELLATION
# =========================================================

class NoiseCancellation:

    """
    Extract the residual photon signal.

    Inputs:
        RC latch output
        Reference detector
        AND coincidence statistics

    Output:
        Noise-cancelled temporal residual.
    """

    def __init__(self, reference_gain=REFERENCE_GAIN):

        self.reference_gain = reference_gain

    def process(self, trace):

        residuals = []
        diagnostics = []

        for sample in trace:

            latch = sample["latch"]
            reference = sample["reference"]

            residual = (
                latch
                - self.reference_gain * reference
            )

            residuals.append(residual)

            diagnostics.append({
                "latch": latch,
                "reference": reference,
                "coincidence": sample["coincidence"],
                "residual": residual
            })

        return (
            np.asarray(residuals, dtype=np.float64),
            diagnostics
        )


# =========================================================
# LEVEL 2: TEMPORAL CODE RECOGNITION
# =========================================================

class TemporalCodeDecoder:

    """
    Recognise the photon temporal code.

    No candidate enumeration.

    The decoder measures the difference between the
    first and second halves of the four-chip trace.

    Positive correlation -> bit 1.
    Negative correlation -> bit 0.
    """

    def __init__(self, nbits):

        self.nbits = nbits

        self.confidence = []
        self.temporal_scores = []

        self.decoded_bits = []

    def decode_bit(self, residual):

        if len(residual) != 4:
            raise ValueError("Expected four temporal chips")

        first_half = np.mean(residual[:2])

        second_half = np.mean(residual[2:])

        score = first_half - second_half

        bit = int(score > 0)

        # Confidence is a normalized temporal margin.
        scale = np.mean(np.abs(residual)) + 1e-9

        confidence = float(
            np.clip(abs(score) / scale, 0.0, 1.0)
        )

        self.temporal_scores.append(float(score))
        self.confidence.append(confidence)

        self.decoded_bits.append(bit)

        return bit, confidence, score

    def decode(self, residual_traces):

        candidate = 0

        for residual in residual_traces:

            bit, confidence, score = self.decode_bit(
                residual
            )

            candidate = (candidate << 1) | bit

        return candidate


# =========================================================
# BLOCK HEADER
# =========================================================

def create_header_prefix(args):

    if args.prefix_hex:

        prefix = parse_hex(args.prefix_hex)

        if len(prefix) != 76:
            raise ValueError(
                "Block prefix must be exactly 76 bytes"
            )

        return prefix

    # Custom 76-byte header prefix.
    version = struct.pack(">I", VERSION)

    previous_hash = secrets.token_bytes(32)

    merkle_root = secrets.token_bytes(32)

    timestamp = int(time.time())

    bits = struct.pack(">I", args.difficulty)

    return (
        version
        + previous_hash
        + merkle_root
        + struct.pack(">I", timestamp)
        + bits
    )


# =========================================================
# POW VERIFICATION
# =========================================================

def verify_pow(prefix, nonce, target):

    """
    Exactly one candidate.
    Exactly one double-SHA-256 computation.
    """

    nonce_bytes = struct.pack(">I", nonce)

    header = prefix + nonce_bytes

    digest = double_sha256(header)

    hash_value = int.from_bytes(digest, "big")

    success = hash_value <= target

    return header, digest, success


# =========================================================
# COMPLETE PHOTON PIPELINE
# =========================================================

def run(args):

    rng = np.random.default_rng(args.seed)

    # The nonce is encoded into the photon source.
    # It is not given to the decoder.

    if args.solution_nonce is None:

        source_nonce = secrets.randbits(32)

    else:

        source_nonce = args.solution_nonce

    if not 0 <= source_nonce <= 0xFFFFFFFF:
        raise ValueError("Nonce must fit in 32 bits")

    # Build challenge.
    prefix = create_header_prefix(args)

    if args.target:

        target = int(
            args.target.removeprefix("0x"),
            16
        )

        if not 0 <= target <= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF:
            raise ValueError("Target must fit in 256 bits")

    else:

        target = make_target(args.difficulty)

    # Instantiate source.
    source = PhotonSource(
        source_nonce,
        32,
        rng,
        photons=args.photons,
        signal_strength=args.signal_strength
    )

    cancellation = NoiseCancellation()

    decoder = TemporalCodeDecoder(32)

    print("=" * 72)
    print("PHOTON TEMPORAL-CODE PROOF OF WORK")
    print("=" * 72)

    print("Architecture : Two-level photon recognition")
    print("Physical gate: AND")
    print("Signal path  : RC + hysteresis + DNC")
    print("Decoder      : Temporal code correlation")
    print("Hash         : Double SHA-256")
    print("Nonce width  : 32 bits")

    print()
    print("BLOCK CHALLENGE")
    print("-" * 50)

    print("Prefix       :", prefix.hex())
    print("Target       :", f"{target:064X}")

    print()
    print("LEVEL 1: PHOTON ACQUISITION")
    print("-" * 50)

    acquisition_start = time.perf_counter()

    residual_traces = []

    for i in range(32):

        trace = source.capture_bit_trace(i)

        residual, diagnostics = cancellation.process(
            trace
        )

        residual_traces.append(residual)

        print(
            f"Position {i:02d} | "
            f"AND={np.mean(residual):+.4f} | "
            f"reference={np.mean([x['reference'] for x in diagnostics]):.3f}"
        )

    acquisition_time = (
        time.perf_counter() - acquisition_start
    )

    print()
    print("LEVEL 2: TEMPORAL CODE RECOGNITION")
    print("-" * 50)

    decoding_start = time.perf_counter()

    candidate = decoder.decode(residual_traces)

    decoding_time = (
        time.perf_counter() - decoding_start
    )

    print()

    print(
        "Decoded nonce :",
        f"{candidate:08X}"
    )

    print(
        "Mean confidence:",
        f"{np.mean(decoder.confidence):.4f}"
    )

    print(
        "Temporal correlation range:",
        f"{min(decoder.temporal_scores):+.4f}",
        "to",
        f"{max(decoder.temporal_scores):+.4f}"
    )

    # -----------------------------------------------------
    # ONE HASH VERIFICATION
    # -----------------------------------------------------

    print()
    print("FINAL POW VERIFICATION")
    print("-" * 50)

    hash_start = time.perf_counter()

    header, digest, success = verify_pow(
        prefix,
        candidate,
        target
    )

    hash_time = time.perf_counter() - hash_start

    score = leading_zero_bits(digest)

    print("Candidate hash :", digest.hex())

    print("Leading zeros  :", score)

    print("Target         :", f"{target:064X}")

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print()
    print("=" * 72)

    if success:

        print("RESULT: PHOTON-DECODED POW VERIFIED")

    else:

        print("RESULT: POW VERIFICATION FAILED")

        print(
            "The decoded nonce did not satisfy the supplied target."
        )

    print("=" * 72)

    print()
    print("DIAGNOSTICS")
    print("-" * 50)

    print(
        "Source nonce (simulation only):",
        f"{source_nonce:08X}" if args.reveal_source else "HIDDEN"
    )

    if args.reveal_source:

        print(
            "Decoder matches source:",
            candidate == source_nonce
        )

    print("Photon frames     :", source.total_frames)

    print("Photon counts     :", source.total_photons)

    print("AND coincidences  :", source.total_coincidences)

    print("Decoded bits      :", len(decoder.decoded_bits))

    print("Candidate count   : 1")

    print("SHA-256 primitives: 2")

    print("PoW verifications : 1")

    print("Nonce increments  : 0")

    print("Candidate search  : DISABLED")

    print()
    print("TIMING")
    print("-" * 50)

    print(f"Acquisition : {acquisition_time:.6f} s")

    print(f"Decoding    : {decoding_time:.6f} s")

    print(f"Hashing     : {hash_time:.9f} s")

    print(
        f"Total       : "
        f"{acquisition_time + decoding_time + hash_time:.6f} s"
    )

    print("=" * 72)

    return success


# =========================================================
# COMMAND LINE
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Two-level photon temporal-code PoW"
    )

    parser.add_argument(
        "--difficulty",
        type=int,
        default=0
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional explicit 256-bit target in hexadecimal"
    )

    parser.add_argument(
        "--prefix-hex",
        type=str,
        default=None,
        help="Optional exact 76-byte block header prefix"
    )

    parser.add_argument(
        "--solution-nonce",
        type=parse_integer,
        default=None,
        help="Nonce encoded in the simulated photon source"
    )

    parser.add_argument(
        "--photons",
        type=float,
        default=40.0
    )

    parser.add_argument(
        "--signal-strength",
        type=float,
        default=2.5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--reveal-source",
        action="store_true"
    )

    args = parser.parse_args()

    if not 0 <= args.difficulty <= 256:
        parser.error("Difficulty must be between 0 and 256")

    if args.photons <= 0:
        parser.error("Photon intensity must be positive")

    if args.signal_strength <= 0:
        parser.error("Signal strength must be positive")

    try:

        run(args)

    except (ValueError, OverflowError) as exc:

        parser.error(str(exc))


if __name__ == "__main__":
    main()
