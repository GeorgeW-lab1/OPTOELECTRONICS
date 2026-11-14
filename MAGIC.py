
#!/usr/bin/env python3
"""
LOCKED OPTICAL PRIVATE-NUMBER RECOVERY
======================================

MULTIPROCESS ARCHITECTURE

PROCESS 1: ORACLE
    Owns secret number.
    Creates salted SHA256 commitment.
    Generates optical images.
    Holds factors privately.

PROCESS 2: OPTICAL CHANNEL
    Transfers image frames only.
    Never transmits raw bits or integer values.

PROCESS 3: RECOGNISER
    Receives public commitment and optical images.
    Performs spatial pooling.
    Accumulates probabilistic votes.
    Performs Chase-style reconstruction.
    Never receives original number or factors.

PROCESS 4: ORACLE VERIFIER
    Implemented inside the isolated oracle process.
    Verifies reconstructed candidate.
    Releases secret and factors only after proof.

Requirements:
    pip install numpy sympy

Run:
    python locked_optical_recovery.py

Custom number:
    python locked_optical_recovery.py --number 9999999999999999
"""

import argparse
import hashlib
import itertools
import multiprocessing as mp
import queue
import secrets
import time

import numpy as np
from sympy import factorint


# ============================================================
# CONFIGURATION
# ============================================================

BYTE_LENGTH = 8

ROWS = 8
COLS = 8

CELL_SIZE = 16

REVEAL_RATE = 0.45
BIT_ERROR_RATE = 0.02
PIXEL_NOISE = 12.0

THRESHOLD = 127.5
CONFIDENCE_THRESHOLD = 25.0

MAX_EXPOSURES = 200

UNCERTAIN_BITS = 16
MAX_FLIPS = 4
MAX_HASH_TESTS = 200_000

ORACLE_TIMEOUT = 120


# ============================================================
# ENCODING UTILITIES
# ============================================================

def integer_to_bits(number):

    data = number.to_bytes(
        BYTE_LENGTH,
        byteorder="big"
    )

    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8)
    )


def bits_to_bytes(bits):

    return np.packbits(
        np.asarray(bits, dtype=np.uint8)
    ).tobytes()


def bits_to_integer(bits):

    return int.from_bytes(
        bits_to_bytes(bits),
        byteorder="big"
    )


# ============================================================
# PROCESS 1: LOCKED ORACLE
# ============================================================

def oracle_process(
    secret_number,
    public_queue,
    frame_queue,
    acknowledgement_queue,
    candidate_queue,
    result_queue
):

    """
    Entire secret exists only inside this process.

    No secret bits, integer or factors are placed
    into the recogniser's queues.
    """

    # --------------------------------------------------------
    # PRIVATE ORACLE STATE
    # --------------------------------------------------------

    secret_payload = secret_number.to_bytes(
        BYTE_LENGTH,
        byteorder="big"
    )

    secret_bits = integer_to_bits(secret_number)

    secret_factors = None

    salt = secrets.token_bytes(16)

    commitment = hashlib.sha256(
        salt + secret_payload
    ).digest()

    rng = np.random.default_rng()

    # --------------------------------------------------------
    # PUBLIC PARAMETERS
    # --------------------------------------------------------

    public_queue.put({
        "salt": salt,
        "commitment": commitment,
        "bit_count": BYTE_LENGTH * 8,
        "rows": ROWS,
        "cols": COLS,
        "cell_size": CELL_SIZE
    })

    print("[ORACLE] Secret locked.")
    print("[ORACLE] Commitment published.")

    # --------------------------------------------------------
    # OPTICAL TRANSMISSION
    # --------------------------------------------------------

    for exposure in range(1, MAX_EXPOSURES + 1):

        # Randomly reveal cells.
        revealed = (
            rng.random(len(secret_bits))
            < REVEAL_RATE
        )

        # Simulated optical classification errors.
        errors = (
            rng.random(len(secret_bits))
            < BIT_ERROR_RATE
        )

        observed = np.bitwise_xor(
            secret_bits,
            errors.astype(np.uint8)
        )

        # Erased cells remain grey.
        cells = np.full(
            len(secret_bits),
            127.5,
            dtype=np.float32
        )

        cells[revealed] = (
            observed[revealed] * 255.0
        )

        grid = cells.reshape(ROWS, COLS)

        # Render image.
        frame = np.repeat(
            np.repeat(
                grid,
                CELL_SIZE,
                axis=0
            ),
            CELL_SIZE,
            axis=1
        )

        # Simulate sensor noise.
        noise = rng.normal(
            0,
            PIXEL_NOISE,
            frame.shape
        )

        frame = np.clip(
            frame + noise,
            0,
            255
        ).astype(np.float32)

        # Only the optical image crosses the boundary.
        frame_queue.put(frame)

        # Wait for the recogniser to finish this frame.
        try:
            acknowledgement_queue.get(
                timeout=ORACLE_TIMEOUT
            )
        except queue.Empty:
            print("[ORACLE] Recogniser timeout.")
            break

        # ----------------------------------------------------
        # CHECK FOR RECOVERED CANDIDATE
        # ----------------------------------------------------

        try:
            candidate = candidate_queue.get(
                timeout=2.0
            )
        except queue.Empty:
            continue

        if not isinstance(candidate, bytes):
            continue

        if len(candidate) != BYTE_LENGTH:
            continue

        # ----------------------------------------------------
        # VERIFICATION
        # ----------------------------------------------------

        candidate_commitment = hashlib.sha256(
            salt + candidate
        ).digest()

        verified = secrets.compare_digest(
            candidate_commitment,
            commitment
        )

        if not verified:
            print("[ORACLE] Invalid candidate.")
            continue

        # ----------------------------------------------------
        # MAGIC RELEASE
        # ----------------------------------------------------

        print("\n[ORACLE] COMMITMENT VERIFIED")
        print("[ORACLE] Authorising secret release.")

        # Factors are calculated only after verification.
        secret_factors = factorint(secret_number)

        result_queue.put({
            "success": True,
            "number": secret_number,
            "factors": dict(secret_factors),
            "exposure": exposure,
            "proof": True
        })

        print("[ORACLE] Secret released.")

        return

    # --------------------------------------------------------
    # FAILURE: REMAIN LOCKED
    # --------------------------------------------------------

    frame_queue.put(None)

    result_queue.put({
        "success": False,
        "number": None,
        "factors": None,
        "proof": False
    })

    print("[ORACLE] Recovery unsuccessful.")
    print("[ORACLE] Secret remains locked.")


# ============================================================
# PROCESS 2: OPTICAL CHANNEL / SPATIAL POOLING
# ============================================================

def spatial_pool(frame, rows, cols, cell_size):

    blocks = frame.reshape(
        rows,
        cell_size,
        cols,
        cell_size
    )

    pooled = blocks.mean(
        axis=(1, 3)
    )

    return pooled.flatten()


# ============================================================
# PROCESS 3: INDEPENDENT RECOGNISER
# ============================================================

def recogniser_process(
    public,
    frame_queue,
    acknowledgement_queue,
    candidate_queue
):

    """
    This process receives no oracle reference,
    no original number and no factorisation.
    """

    salt = public["salt"]
    commitment = public["commitment"]

    bit_count = public["bit_count"]

    rows = public["rows"]
    cols = public["cols"]

    cell_size = public["cell_size"]

    # Independent observation memory.
    ones = np.zeros(
        bit_count,
        dtype=np.int64
    )

    zeros = np.zeros(
        bit_count,
        dtype=np.int64
    )

    total_hash_tests = 0

    print("[RECOGNISER] Started.")
    print("[RECOGNISER] Original number unavailable.")
    print("[RECOGNISER] Awaiting optical observations.")

    # --------------------------------------------------------
    # HASH VERIFIER
    # --------------------------------------------------------

    def verify_candidate(candidate):

        nonlocal total_hash_tests

        digest = hashlib.sha256(
            salt + candidate
        ).digest()

        total_hash_tests += 1

        return secrets.compare_digest(
            digest,
            commitment
        )

    # --------------------------------------------------------
    # CANDIDATE DECODER
    # --------------------------------------------------------

    def decode():

        nonlocal total_hash_tests

        covered = (ones + zeros) > 0

        if not np.all(covered):
            return None

        estimate = (
            ones > zeros
        ).astype(np.uint8)

        candidate = bits_to_bytes(estimate)

        # First test ordinary majority decoding.
        if verify_candidate(candidate):
            return candidate

        # Confidence-ranked positions.
        confidence = np.abs(ones - zeros)

        uncertain = np.argsort(
            confidence,
            kind="stable"
        )[:min(UNCERTAIN_BITS, bit_count)]

        # Chase-style bounded bit-flip search.
        for flip_count in range(
            1,
            min(MAX_FLIPS, len(uncertain)) + 1
        ):

            for subset in itertools.combinations(
                uncertain,
                flip_count
            ):

                if total_hash_tests >= MAX_HASH_TESTS:
                    return None

                trial = estimate.copy()

                trial[list(subset)] ^= 1

                candidate = bits_to_bytes(trial)

                if verify_candidate(candidate):
                    return candidate

        return None

    # --------------------------------------------------------
    # OBSERVATION LOOP
    # --------------------------------------------------------

    exposure = 0

    while True:

        frame = frame_queue.get()

        if frame is None:
            print("[RECOGNISER] Transmission ended.")
            break

        exposure += 1

        # Actual spatial pooling.
        pooled = spatial_pool(
            frame,
            rows,
            cols,
            cell_size
        )

        # Grey cells are erasures.
        valid = (
            np.abs(pooled - THRESHOLD)
            >= CONFIDENCE_THRESHOLD
        )

        observed = (
            pooled > THRESHOLD
        ).astype(np.uint8)

        indices = np.where(valid)[0]

        # Accumulate independent observations.
        ones[
            indices[observed[indices] == 1]
        ] += 1

        zeros[
            indices[observed[indices] == 0]
        ] += 1

        coverage = int(
            np.sum((ones + zeros) > 0)
        )

        # Acknowledge completed image.
        acknowledgement_queue.put(exposure)

        if exposure % 5 == 0:

            print(
                f"[RECOGNISER] Exposure {exposure:3d} | "
                f"Coverage {coverage}/{bit_count} | "
                f"Hashes {total_hash_tests}"
            )

        # Do not reconstruct before full coverage.
        if coverage != bit_count:
            continue

        # Reconstruct independently.
        candidate = decode()

        if candidate is not None:

            print("\n[RECOGNISER] CANDIDATE FOUND.")
            print("[RECOGNISER] SHA-256 commitment matched.")

            # Only recovered candidate is sent.
            candidate_queue.put(candidate)

            print("[RECOGNISER] Candidate submitted to oracle.")

            return

        if total_hash_tests >= MAX_HASH_TESTS:
            print("[RECOGNISER] Hash budget exhausted.")
            return


# ============================================================
# PROCESS 4: TRUSTED LAUNCHER / COORDINATOR
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Locked optical hidden-number recovery"
    )

    parser.add_argument(
        "--number",
        type=int,
        default=9_999_999_999_999_999
    )

    args = parser.parse_args()

    secret_number = args.number

    if not 0 <= secret_number < 2 ** (BYTE_LENGTH * 8):
        parser.error("Number must fit in 64 bits.")

    # Windows-compatible multiprocessing.
    ctx = mp.get_context("spawn")

    public_queue = ctx.Queue()

    frame_queue = ctx.Queue(maxsize=1)

    acknowledgement_queue = ctx.Queue()

    candidate_queue = ctx.Queue()

    result_queue = ctx.Queue()

    print("=" * 65)
    print("LOCKED OPTICAL PRIVATE-NUMBER RECOVERY")
    print("=" * 65)

    print("\nStarting isolated oracle process...")

    oracle = ctx.Process(
        target=oracle_process,
        args=(
            secret_number,
            public_queue,
            frame_queue,
            acknowledgement_queue,
            candidate_queue,
            result_queue
        ),
        name="LOCKED-ORACLE"
    )

    oracle.start()

    try:

        # Receive public information only.
        public = public_queue.get(timeout=30)

        print("\n[LAUNCHER] Public commitment received.")

        print(
            "[LAUNCHER] SHA256:",
            public["commitment"].hex()
        )

        print("\nStarting independent recogniser...")

        recogniser = ctx.Process(
            target=recogniser_process,
            args=(
                public,
                frame_queue,
                acknowledgement_queue,
                candidate_queue
            ),
            name="OPTICAL-RECOGNISER"
        )

        recogniser.start()

        # Wait for the oracle's release decision.
        result = result_queue.get(timeout=300)

        recogniser.join(timeout=10)

        print("\n" + "=" * 65)
        print("FINAL RECOVERY REPORT")
        print("=" * 65)

        if result["success"]:

            print("\n< MAGIC MOMENT >")

            print("Commitment verified: TRUE")
            print("Oracle release: TRUE")

            print(
                "\nRecovered number:",
                result["number"]
            )

            print(
                "Prime factors:",
                result["factors"]
            )

            print(
                "\nRecovery exposure:",
                result["exposure"]
            )

            print("\nSTATUS: SUCCESS")

        else:

            print("\nSTATUS: FAILED")

            print("Number remains locked.")
            print("Factors remain locked.")

        print("\nProcess separation:")
        print("Oracle process:       ", oracle.pid)
        print("Recognizer process:   ", recogniser.pid)

    except (queue.Empty, KeyboardInterrupt) as exc:

        print("\n[LAUNCHER] Interrupted or timed out.")

        if isinstance(exc, KeyboardInterrupt):
            print("[LAUNCHER] Stopping experiment.")

    finally:

        if oracle.is_alive():
            oracle.terminate()

        oracle.join(timeout=5)

        for q in (
            public_queue,
            frame_queue,
            acknowledgement_queue,
            candidate_queue,
            result_queue
        ):
            q.close()

        print("\n[LAUNCHER] Processes closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()
