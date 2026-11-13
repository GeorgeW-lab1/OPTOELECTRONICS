
#!/usr/bin/env python3
"""
rp_hidden_number_commit_only.py

PARALLEL OPTICAL BIT POOLING
COMMIT-ONLY HIDDEN NUMBER RECOVERY
CUSTOM QISKIT BACKENDV2
TQDM PROGRESS MONITORING

Architecture:
    Oracle -> displayed parallel bit grid
    Display -> USB camera -> one full-frame exposure
    Camera frame -> parallel spatial pooling
    Pooling -> parallel bit classification
    Classification -> parallel vote accumulation
    Reconstructed number -> SHA256 commitment verification
    Successful proof -> oracle reveals its stored factors

Camera mode requires the camera to be pointed at the displayed bit grid.
Synthetic mode uses the same pooling pipeline without a physical camera.

Install:
    pip install numpy opencv-python tqdm qiskit sympy

Example:
    python rp_hidden_number_commit_only.py --n 9999999999999999

Camera:
    python rp_hidden_number_commit_only.py --camera 0 --preview

Note:
    The Qiskit backend is a diagnostic circuit demonstration.
    It is not a quantum factorisation algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import secrets
import sys
import math
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    print("Missing tqdm. Install it with: pip install tqdm")
    raise

try:
    import sympy
except ImportError:
    sympy = None

try:
    from qiskit import QuantumCircuit, qpy
    from qiskit.providers import BackendV2, Options, JobV1
    from qiskit.providers import JobStatus
    from qiskit.result import Result
    from qiskit.result.models import ExperimentResult, ExperimentResultData
    from qiskit.transpiler import Target
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# ============================================================
# EMBEDDED QISKIT CIRCUIT
# ============================================================

EMBEDDED_Q_PY_CODE = """
from qiskit import QuantumCircuit

def create_factorisation_circuit(N: int = 0):
    # Small diagnostic GHZ circuit.
    # This tests the custom execution backend only.
    qc = QuantumCircuit(8, 8, name="parallel_bit_pooling_diagnostic")

    qc.h(0)

    for i in range(7):
        qc.cx(i, i + 1)

    qc.measure(range(8), range(8))
    return qc, 8
"""


def load_embedded_factorisation_circuit(N=0):
    if not QISKIT_AVAILABLE:
        return None, 8

    namespace = {
        "QuantumCircuit": QuantumCircuit,
        "qpy": qpy,
    }

    exec(EMBEDDED_Q_PY_CODE, namespace)

    return namespace["create_factorisation_circuit"](N)


# ============================================================
# CUSTOM QISKIT BACKEND
# ============================================================

if QISKIT_AVAILABLE:

    class CustomBackendJob(JobV1):

        def __init__(self, backend, result_obj):
            super().__init__(
                backend,
                job_id=secrets.token_hex(8)
            )
            self._result_obj = result_obj

        def submit(self):
            pass

        def result(self):
            return self._result_obj

        def cancel(self):
            pass

        def status(self):
            return JobStatus.DONE


    class CustomExecutionBackend(BackendV2):

        def __init__(self, name="parallel_pooling_backend"):
            super().__init__(
                name=name,
                backend_version="1.0.0"
            )
            self._target = Target(num_qubits=8)

        @classmethod
        def _default_options(cls):
            return Options(shots=64)

        @property
        def target(self):
            return self._target

        @property
        def max_circuits(self):
            return 1024

        def run(self, run_input, **options):

            if isinstance(run_input, QuantumCircuit):
                circuits = [run_input]
            else:
                circuits = list(run_input)

            shots = int(options.get("shots", 64))

            results = [
                self._execute_circuit(circuit, shots)
                for circuit in circuits
            ]

            result_obj = Result(
                backend_name=self.name,
                backend_version=self.backend_version,
                qiskit_version="1.0+",
                success=True,
                results=results
            )

            return CustomBackendJob(self, result_obj)

        def _execute_circuit(self, circuit, shots):

            # The diagnostic circuit is deliberately limited to 8 qubits.
            if circuit.num_qubits > 8:
                raise ValueError(
                    "Diagnostic backend supports at most 8 qubits"
                )

            dim = 1 << circuit.num_qubits

            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0

            for instruction in circuit.data:

                name = instruction.operation.name

                qargs = [
                    circuit.find_bit(q).index
                    for q in instruction.qubits
                ]

                if name == "h":
                    state = self._apply_h(state, qargs[0])

                elif name in ("cx", "cnot"):
                    state = self._apply_cx(
                        state,
                        qargs[0],
                        qargs[1]
                    )

            probabilities = np.abs(state) ** 2
            probabilities /= probabilities.sum()

            samples = np.random.choice(
                dim,
                size=shots,
                p=probabilities
            )

            counts = {}

            for sample in samples:
                bitstring = format(
                    sample,
                    f"0{circuit.num_qubits}b"
                )[::-1]

                counts[bitstring] = (
                    counts.get(bitstring, 0) + 1
                )

            header = {
                "name": circuit.name,
                "memory_slots": circuit.num_clbits
            }

            return ExperimentResult(
                header=header,
                success=True,
                shots=shots,
                data=ExperimentResultData(counts=counts)
            )

        @staticmethod
        def _apply_h(state, q):

            result = np.zeros_like(state)
            factor = 1.0 / np.sqrt(2.0)

            for i, amplitude in enumerate(state):

                if amplitude == 0:
                    continue

                bit = (i >> q) & 1
                i0 = i & ~(1 << q)
                i1 = i | (1 << q)

                if bit == 0:
                    result[i0] += factor * amplitude
                    result[i1] += factor * amplitude
                else:
                    result[i0] += factor * amplitude
                    result[i1] -= factor * amplitude

            return result

        @staticmethod
        def _apply_cx(state, control, target):

            result = np.zeros_like(state)

            for i, amplitude in enumerate(state):

                if amplitude == 0:
                    continue

                if (i >> control) & 1:
                    result[i ^ (1 << target)] = amplitude
                else:
                    result[i] = amplitude

            return result


# ============================================================
# ORACLE
# ============================================================

class HiddenNumberOracle:

    def __init__(
        self,
        target_number,
        bits,
        noise_probability=0.02
    ):

        if bits % 8:
            raise ValueError("Bit width must be a multiple of 8")

        if target_number < 0:
            raise ValueError("N must be positive")

        if target_number.bit_length() > bits:
            raise ValueError("N exceeds configured bit width")

        self.hidden_number = target_number
        self.bits = bits
        self.byte_length = bits // 8
        self.noise_probability = noise_probability

        self.salt = os.urandom(16)

        self.commitment = self._hash(
            self.hidden_number
        )

        self.prime_factors = self._compute_factors(
            self.hidden_number
        )

    def _hash(self, value):

        payload = value.to_bytes(
            self.byte_length,
            "big"
        )

        return hashlib.sha256(
            self.salt + payload
        ).digest()

    def _compute_factors(self, n):

        if sympy is not None:
            return [
                int(x)
                for x in sympy.factorint(n).keys()
                for _ in range(int(sympy.factorint(n)[x]))
            ]

        # Basic fallback for small inputs.
        factors = []
        d = 2
        remaining = n

        while d * d <= remaining:

            while remaining % d == 0:
                factors.append(d)
                remaining //= d

            d = 3 if d == 2 else d + 2

        if remaining > 1:
            factors.append(remaining)

        return factors

    def public_parameters(self):

        return {
            "salt": self.salt,
            "commitment": self.commitment,
        }

    def bit_vector(self):

        return np.array(
            [
                (self.hidden_number >> i) & 1
                for i in range(self.bits)
            ],
            dtype=np.uint8
        )

    def verify(self, candidate):

        return self._hash(candidate) == self.commitment

    def reveal_if_proven(self, candidate):

        if not self.verify(candidate):
            return None

        return {
            "hidden_number": self.hidden_number,
            "binary": format(
                self.hidden_number,
                f"0{self.bits}b"
            ),
            "prime_factors": self.prime_factors,
            "commitment": self.commitment.hex(),
            "proof": True,
        }


# ============================================================
# PARALLEL BIT GRID
# ============================================================

def create_bit_grid(bit_vector, rows, cols, cell_size=24):

    total = rows * cols

    padded = np.zeros(total, dtype=np.uint8)
    padded[:len(bit_vector)] = bit_vector

    grid = padded.reshape(rows, cols)

    # Black = 0, white = 1.
    pixels = grid * 255

    image = np.repeat(
        np.repeat(
            pixels,
            cell_size,
            axis=0
        ),
        cell_size,
        axis=1
    )

    # Add dark borders between cells.
    image[::cell_size, :] = 80
    image[:, ::cell_size] = 80

    return cv2.cvtColor(
        image.astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    ) if cv2 is not None else image


# ============================================================
# CAMERA SOURCE
# ============================================================

class CameraSource:

    def __init__(
        self,
        source,
        width=640,
        height=480,
        fps=30
    ):

        if cv2 is None:
            raise RuntimeError(
                "Install opencv-python for camera input"
            )

        if source == "auto":
            source = 0

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera: {source}"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            fps
        )

    def read(self):

        ok, frame = self.cap.read()

        if not ok:
            return None

        return frame

    def close(self):
        self.cap.release()


class SyntheticSource:

    def read(self, grid_image):
        return grid_image.copy()

    def close(self):
        pass


# ============================================================
# PARALLEL SPATIAL POOLING
# ============================================================

def parallel_bit_pool(
    frame,
    rows,
    cols,
    threshold,
    confidence_threshold
):

    """
    One camera frame is divided into rows x cols regions.

    All spatial cells are pooled in one vectorised operation.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV is required")

    if frame is None:
        raise ValueError("Empty camera frame")

    if frame.ndim == 3:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = frame

    h, w = gray.shape

    # Equal-sized spatial regions.
    # Resize ensures exact divisibility.
    pooled_input = cv2.resize(
        gray,
        (cols * 16, rows * 16),
        interpolation=cv2.INTER_AREA
    )

    cell_h = 16
    cell_w = 16

    # Vectorised parallel pooling:
    # (rows, cell_h, cols, cell_w)
    cells = pooled_input.reshape(
        rows,
        cell_h,
        cols,
        cell_w
    )

    pooled = cells.mean(
        axis=(1, 3),
        dtype=np.float32
    )

    # Parallel binary classification.
    observed_bits = (
        pooled > threshold
    ).astype(np.uint8).ravel()

    # Parallel confidence estimate.
    confidence = np.abs(
        pooled.ravel() - threshold
    )

    # Parallel validity mask.
    valid = (
        confidence >= confidence_threshold
    )

    return observed_bits, confidence, valid, pooled


# ============================================================
# COMMIT-ONLY RECOGNISER
# ============================================================

class CommitOnlyRecogniser:

    def __init__(
        self,
        bits,
        salt,
        commitment,
        max_candidates=200_000
    ):

        self.bits = bits
        self.byte_length = bits // 8

        self.salt = salt
        self.commitment = commitment

        self.max_candidates = max_candidates

        # Parallel observation accumulators.
        self.ones = np.zeros(
            bits,
            dtype=np.uint32
        )

        self.zeros = np.zeros(
            bits,
            dtype=np.uint32
        )

        self.cycles = 0
        self.candidates_tested = 0

        self.recovered = None
        self.proof = None

        self.backend = None
        self.circuit = None

        if QISKIT_AVAILABLE:
            self.circuit, _ = (
                load_embedded_factorisation_circuit()
            )
            self.backend = CustomExecutionBackend()

    def observe(
        self,
        observed_bits,
        valid
    ):

        observed_bits = np.asarray(
            observed_bits,
            dtype=np.uint8
        )

        valid = np.asarray(
            valid,
            dtype=bool
        )

        if len(observed_bits) != self.bits:
            raise ValueError("Incorrect observation width")

        if len(valid) != self.bits:
            raise ValueError("Incorrect validity-mask width")

        # Parallel accumulation.
        self.ones += (
            observed_bits * valid
        ).astype(np.uint32)

        self.zeros += (
            (1 - observed_bits) * valid
        ).astype(np.uint32)

        self.cycles += 1

        # Run diagnostic Qiskit circuit.
        if self.backend is not None:

            try:
                job = self.backend.run(
                    self.circuit,
                    shots=64
                )

                _ = job.result().get_counts()

            except Exception as exc:
                print(
                    f"\n[Qiskit diagnostic] {exc}",
                    file=sys.stderr
                )

        self._decode()

    def _candidate_digest(self, candidate):

        return hashlib.sha256(
            self.salt
            + candidate.to_bytes(
                self.byte_length,
                "big"
            )
        ).digest()

    def _decode(self):

        totals = self.ones + self.zeros

        known = totals > 0

        if not np.all(known):
            return

        # Majority vote across every pooled bit.
        recovered_bits = (
            self.ones > self.zeros
        ).astype(np.uint8)

        # Confidence is the difference between the two vote totals.
        confidence = np.abs(
            self.ones.astype(np.int64)
            - self.zeros.astype(np.int64)
        )

        candidate = sum(
            int(bit) << i
            for i, bit in enumerate(recovered_bits)
        )

        self.candidates_tested += 1

        digest = self._candidate_digest(candidate)

        if digest == self.commitment:
            self._accept(candidate, 0)
            return

        # If majority voting contains errors, test a bounded number
        # of flips among the least-confident positions.
        uncertain = np.argsort(confidence)[:min(16, self.bits)]

        for corrections in range(1, min(4, len(uncertain)) + 1):

            for positions in itertools.combinations(
                uncertain,
                corrections
            ):

                if self.candidates_tested >= self.max_candidates:
                    return

                corrected = candidate

                for position in positions:
                    corrected ^= 1 << int(position)

                self.candidates_tested += 1

                if self._candidate_digest(corrected) == self.commitment:
                    self._accept(corrected, corrections)
                    return

    def _accept(self, candidate, corrections):

        digest = self._candidate_digest(candidate)

        self.recovered = candidate

        self.proof = {
            "hidden_number": candidate,
            "binary": format(
                candidate,
                f"0{self.bits}b"
            ),
            "commitment_hex": self.commitment.hex(),
            "candidate_digest_hex": digest.hex(),
            "hash_match": digest == self.commitment,
            "independent_recheck": (
                self._candidate_digest(candidate)
                == self.commitment
            ),
            "cycles_used": self.cycles,
            "candidates_tested": self.candidates_tested,
            "corrections_applied": corrections,
        }

    @property
    def unknown_count(self):

        return int(
            np.count_nonzero(
                (self.ones + self.zeros) == 0
            )
        )

    @property
    def known_count(self):
        return self.bits - self.unknown_count


# ============================================================
# MAIN
# ============================================================

def run(args):

    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required. Install: pip install opencv-python"
        )

    target_n = args.n

    if target_n <= 0:
        raise ValueError("N must be positive")

    # Pad number to complete bytes.
    bits = max(
        8,
        ((target_n.bit_length() + 7) // 8) * 8
    )

    rows = args.rows
    cols = math.ceil(bits / rows)

    if args.cols is not None:
        cols = args.cols
        rows = math.ceil(bits / cols)

    total_cells = rows * cols

    oracle = HiddenNumberOracle(
        target_number=target_n,
        bits=total_cells,
        noise_probability=args.noise
    )

    params = oracle.public_parameters()

    recogniser = CommitOnlyRecogniser(
        bits=total_cells,
        salt=params["salt"],
        commitment=params["commitment"],
        max_candidates=args.max_candidates
    )

    if args.camera is None:
        source = SyntheticSource()
        source_description = "synthetic optical frame"
    else:
        try:
            source = CameraSource(
                args.camera,
                args.width,
                args.height,
                args.fps
            )
            source_description = f"USB camera {args.camera}"
        except Exception as exc:
            print(
                f"[camera] {exc}",
                file=sys.stderr
            )
            print("[camera] Falling back to synthetic mode")
            source = SyntheticSource()
            source_description = "synthetic optical frame"

    print("\n========== PARALLEL BIT POOLING ==========")
    print(f"Target N             : {target_n}")
    print(f"Number width         : {bits} bits")
    print(f"Grid                 : {rows} x {cols}")
    print(f"Parallel bit cells   : {total_cells}")
    print(f"Observation source   : {source_description}")
    print(f"Reveal probability   : {args.reveal}")
    print(f"Noise probability    : {args.noise}")
    print(f"Salt                 : {params['salt'].hex()}")
    print(f"Commitment           : {params['commitment'].hex()}")
    print(f"Qiskit available     : {QISKIT_AVAILABLE}")

    if args.camera is not None:
        print("\nPoint the USB camera at the displayed ORACLE BIT GRID.")
        print("Keep the entire grid visible and in focus.")

    # The oracle's grid is displayed separately from the camera input.
    oracle_bits = oracle.bit_vector()

    grid_image = create_bit_grid(
        oracle_bits,
        rows,
        cols,
        cell_size=args.cell_size
    )

    display_scale = args.display_scale

    display_image = cv2.resize(
        grid_image,
        (
            grid_image.shape[1] * display_scale,
            grid_image.shape[0] * display_scale
        ),
        interpolation=cv2.INTER_NEAREST
    )

   

    recovered = False

    progress = tqdm(
        range(args.cycles),
        desc="Parallel optical observations",
        unit="frame"
    )

    try:

        for cycle in progress:

            # One complete camera exposure per cycle.
            if isinstance(source, SyntheticSource):
                frame = source.read(grid_image)
            else:
                frame = source.read()

            if frame is None:
                print("\n[source] Camera frame unavailable")
                break

            # ------------------------------------------------
            # PARALLEL SPATIAL POOLING
            # ------------------------------------------------

            observed_bits, confidence, valid, pooled = (
                parallel_bit_pool(
                    frame,
                    rows,
                    cols,
                    args.threshold,
                    args.confidence_threshold
                )
            )

            # ------------------------------------------------
            # PARALLEL REVEAL MASK
            # ------------------------------------------------

            # Each cell is independently eligible for inclusion.
            reveal_mask = (
                np.random.random(total_cells)
                < args.reveal
            )

            # Simulated sensor noise is applied to the optical
            # observations, not to individual camera exposures.
            noise_mask = (
                np.random.random(total_cells)
                < args.noise
            )

            observed_bits = (
                observed_bits ^ noise_mask.astype(np.uint8)
            )

            # Confidence and reveal validity combined.
            valid = valid & reveal_mask

            # ------------------------------------------------
            # PARALLEL ACCUMULATION
            # ------------------------------------------------

            recogniser.observe(
                observed_bits,
                valid
            )

            progress.set_postfix(
                known=recogniser.known_count,
                unknown=recogniser.unknown_count,
                tested=recogniser.candidates_tested
            )

            if args.preview:

                preview = cv2.resize(
                    frame,
                    (640, 480)
                )

                cv2.imshow(
                    "CAMERA EXPOSURE",
                    preview
                )

                pooled_view = cv2.resize(
                    pooled,
                    (cols * 20, rows * 20),
                    interpolation=cv2.INTER_NEAREST
                )

                pooled_view = np.clip(
                    pooled_view,
                    0,
                    255
                ).astype(np.uint8)

                cv2.imshow(
                    "PARALLEL POOLED CELLS",
                    pooled_view
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            if recogniser.recovered is not None:
                recovered = True
                break

    except KeyboardInterrupt:
        print("\n[interrupt] Stopping observation loop")

    finally:
        source.close()

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ========================================================
    # RECOGNISER RESULT
    # ========================================================

    print("\n========== RECOGNISER RESULT ==========")

    if recogniser.proof:

        proof = recogniser.proof

        print("RECOVERY SUCCESSFUL")
        print(f"Recovered number       : {proof['hidden_number']}")
        print(f"Recovered binary       : {proof['binary']}")
        print(f"Cycles                 : {proof['cycles_used']}")
        print(f"Candidates tested      : {proof['candidates_tested']}")
        print(f"Corrections applied    : {proof['corrections_applied']}")
        print(f"Commitment             : {proof['commitment_hex']}")
        print(f"Candidate digest       : {proof['candidate_digest_hex']}")
        print(f"Hash match             : {proof['hash_match']}")
        print(f"Independent recheck    : {proof['independent_recheck']}")

        print(
            "\nPROOF: Reconstructed number matches the published SHA256 commitment."
        )

    else:

        print("NOT RECOVERED")
        print(f"Cycles                 : {recogniser.cycles}")
        print(f"Known positions        : {recogniser.known_count}")
        print(f"Unknown positions      : {recogniser.unknown_count}")
        print(f"Candidates tested      : {recogniser.candidates_tested}")

    # ========================================================
    # ORACLE REVEAL
    # ========================================================

    print("\n========== ORACLE REVEAL ==========")

    if recogniser.recovered is not None:

        revealed = oracle.reveal_if_proven(
            recogniser.recovered
        )

        if revealed:

            print(f"Oracle number          : {revealed['hidden_number']}")
            print(f"Oracle binary          : {revealed['binary']}")
            print(f"Oracle commitment      : {revealed['commitment']}")
            print(f"Prime factors          : {revealed['prime_factors']}")

            print(
                "MATCH: recogniser reconstruction equals oracle secret."
            )

        else:
            print("Oracle refused the candidate.")

    else:
        print("No verified reconstruction; oracle retains its factors.")

    print("\n========== BENCH COMPLETE ==========")


# ============================================================
# ARGUMENTS
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Parallel optical bit pooling with commit-only "
            "hidden-number recovery"
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=9999999999999999,
        help="Hidden modulus N"
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=800
    )

    parser.add_argument(
        "--reveal",
        type=float,
        default=0.45
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200000
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="USB camera index; omit for synthetic mode"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=8
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=None
    )

    parser.add_argument(
        "--cell-size",
        type=int,
        default=24
    )

    parser.add_argument(
        "--display-scale",
        type=int,
        default=2
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=127.0
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=25.0
    )

    parser.add_argument(
        "--preview",
        action="store_true"
    )

    args = parser.parse_args()

    if not 0 < args.reveal <= 1:
        parser.error("--reveal must be in (0, 1]")

    if not 0 <= args.noise < 1:
        parser.error("--noise must be in [0, 1)")

    if args.rows < 1:
        parser.error("--rows must be at least 1")

    if args.cols is not None and args.cols < 1:
        parser.error("--cols must be at least 1")

    if args.cell_size < 2:
        parser.error("--cell-size must be at least 2")

    run(args)


if __name__ == "__main__":
    main()
