
#!/usr/bin/env python3
"""
circuit_sort.py

CAMERA-DRIVEN NON-DETERMINISTIC OPTOELECTRONIC SORTING

Optical inputs:
    Right ROI -> Detector A -> stochastic photon detection
    Left ROI  -> Detector B -> deterministic RC/NPN

Main circuit:
    AND -> 150ms delay -> accumulator -> SELECT

Leak circuit:
    OR -> 74HC74 D input -> 10Hz clock -> OBSERVE

Independent probe:
    Separate photon-counting channel.
    Does not modify the main circuit.

Sorting:
    AND/OR compare-exchange network.
    Threshold decomposition for integer sorting.

Photon statistics are simulated using Poisson shot noise.
This is not a physical single-photon camera measurement.
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

VS = 5.0
VBE = 0.7
VIH = 3.5

RC_TAU = 0.050
RC_CHARGE_TIME = 0.100

MAIN_DELAY = 0.150
CLOCK_PERIOD = 0.100

PHOTON_EFFICIENCY = 0.80
PHOTON_FLUX = 20.0
DARK_COUNTS = 0.02

DETECTOR_THRESHOLD = 1

sha = lambda b: hashlib.sha256(b).digest()


# =========================================================
# CAMERA
# =========================================================

class Camera:

    def __init__(self, device):

        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(device)

        if not self.cap.isOpened():
            sys.exit(f"Cannot open camera {device}")

    def read(self):

        ok, frame = self.cap.read()

        if not ok:
            sys.exit("Camera read failed")

        if frame.ndim == 2:
            gray = frame
        else:
            gray = self.cv2.cvtColor(
                frame,
                self.cv2.COLOR_BGR2GRAY
            )

        return self.cv2.resize(
            gray,
            (W, H),
            interpolation=self.cv2.INTER_AREA
        )

    def close(self):
        self.cap.release()


class Synthetic:

    def __init__(self):
        self.rng = np.random.default_rng()

    def read(self):

        return np.clip(
            self.rng.poisson(100.0, (H, W)),
            0,
            255
        ).astype(np.uint8)


# =========================================================
# OPTICAL DETECTORS
# =========================================================

def photon_detection(roi, exposure, rng):

    """
    Poisson photon-counting model.

    The photon arrival count is the source of
    stochasticity.

    roi:
        Normalized optical intensity [0,1].

    exposure:
        Exposure time in seconds.

    rng:
        Random number generator for photon statistics.
    """

    intensity = np.clip(
        np.asarray(roi, dtype=float),
        0.0,
        1.0
    )

    mean_intensity = float(intensity.mean())

    photon_rate = (
        mean_intensity
        * PHOTON_FLUX
        * PHOTON_EFFICIENCY
    )

    dark_rate = DARK_COUNTS

    expected = (
        photon_rate + dark_rate
    ) * exposure

    detected = rng.poisson(expected)

    return int(detected), float(expected)


def detector_rois(frame):

    """
    Detector A: right half of camera image.
    Detector B: left half of camera image.
    """

    normalized = (
        np.asarray(frame, dtype=np.float64) / 255.0
    )

    right_roi = normalized[:, W // 2:]
    left_roi = normalized[:, :W // 2]

    return right_roi, left_roi


def driver_a(right_roi, exposure, rng):

    """
    Randomized Driver A.

    The input bit controls optical transmission:
        0 -> blocked
        1 -> transmitted

    Photon detection determines the output.
    No random software logic bit is generated.
    """

    count, expected = photon_detection(
        right_roi,
        exposure,
        rng
    )

    output = int(count >= DETECTOR_THRESHOLD)

    return output, count, expected


def driver_b(left_roi):

    """
    Deterministic Driver B.

    Optical intensity -> RC node -> NPN emitter follower.

    RC time constant: 50 ms.
    Charge time: 100 ms.
    Base-emitter drop: 0.7 V.
    Output HIGH threshold: 3.5 V.
    """

    intensity = np.clip(
        np.asarray(left_roi, dtype=float),
        0.0,
        1.0
    )

    optical_signal = float(intensity.mean())

    input_voltage = VS * optical_signal

    vc = input_voltage * (
        1.0 - math.exp(
            -RC_CHARGE_TIME / RC_TAU
        )
    )

    ve = max(0.0, vc - VBE)

    output = int(ve >= VIH)

    return output, vc, ve


# =========================================================
# MAIN CIRCUIT
# =========================================================

def and_gate(a, b):
    return int(bool(a) and bool(b))


def or_gate(a, b):
    return int(bool(a) or bool(b))


class FlipFlop74HC74:

    """
    Positive-edge-triggered D flip-flop.

    Clock period: 100 ms (10 Hz).
    """

    def __init__(self):

        self.q = 0
        self.last_clock = 0

    def clock(self, d, clock_signal):

        rising_edge = (
            clock_signal == 1
            and self.last_clock == 0
        )

        if rising_edge:
            self.q = int(bool(d))

        self.last_clock = clock_signal

        return self.q


class Accumulator:

    def __init__(self):

        self.total = 0
        self.reads = 0

    def update(self, value):

        self.total += int(value)
        self.reads += 1

        return self.total


class MainCircuit:

    def __init__(self, realtime=False):

        self.accumulator = Accumulator()
        self.flipflop = FlipFlop74HC74()

        self.realtime = realtime
        self.last_clock_time = time.monotonic()

        self.gates = 0
        self.and_high = 0
        self.or_high = 0

    def evaluate(self, a, b):

        # Parallel AND and OR paths.

        main_and = and_gate(a, b)
        leak_or = or_gate(a, b)

        self.gates += 2

        self.and_high += main_and
        self.or_high += leak_or

        # Main 150ms delay.

        if self.realtime:
            time.sleep(MAIN_DELAY)

        # Accumulator receives the AND result.

        accumulated = self.accumulator.update(
            main_and
        )

        # Independent leak path drives D input.

        now = time.monotonic()

        clock_signal = int(
            now - self.last_clock_time >= CLOCK_PERIOD
        )

        if clock_signal:
            self.last_clock_time = now

        observed = self.flipflop.clock(
            leak_or,
            clock_signal
        )

        return {
            "and": main_and,
            "or": leak_or,
            "accumulator": accumulated,
            "observe": observed
        }


# =========================================================
# INDEPENDENT PROBE
# =========================================================

class IndependentProbe:

    """
    Separate photon detector.

    Probe output is logged independently.
    It cannot influence the main circuit.
    """

    def __init__(self, duration=10.0):

        self.duration = duration
        self.start = time.monotonic()

        self.log = []
        self.active = True

    def measure(self, roi, exposure, rng):

        if not self.active:
            return None

        elapsed = time.monotonic() - self.start

        if elapsed >= self.duration:
            self.active = False
            return None

        count, expected = photon_detection(
            roi,
            exposure,
            rng
        )

        result = {
            "photons": count,
            "expected": expected,
            "detected": int(count > 0),
            "elapsed": elapsed
        }

        self.log.append(result)

        return result


# =========================================================
# OPTICAL READ
# =========================================================

def optical_read(
    right_roi,
    left_roi,
    optical_bit,
    exposure,
    rng,
    circuit,
    probe=None,
    probe_enabled=False
):

    """
    One optical evaluation.

    optical_bit modulates transmission into Detector A.

    Photon statistics determine Driver A.
    Detector B remains deterministic.
    """

    # Optical modulation representing the stored bit.

    if optical_bit:
        transmitted_roi = right_roi
    else:
        transmitted_roi = np.zeros_like(right_roi)

    # Detector A: photon detection.

    a, photons, expected = driver_a(
        transmitted_roi,
        exposure,
        rng
    )

    # Detector B: deterministic RC/NPN.

    b, vc, ve = driver_b(left_roi)

    # Main circuit.

    result = circuit.evaluate(a, b)

    # Independent probe.

    probe_result = None

    if probe_enabled and probe is not None:

        probe_result = probe.measure(
            right_roi,
            exposure,
            rng
        )

    return {
        "a": a,
        "b": b,
        "photons": photons,
        "expected": expected,
        "vc": vc,
        "ve": ve,
        "and": result["and"],
        "or": result["or"],
        "accumulator": result["accumulator"],
        "observe": result["observe"],
        "probe": probe_result
    }


# =========================================================
# STOCHASTIC BIT STORAGE
# =========================================================

def read_bit(
    right_roi,
    left_roi,
    bit,
    retries,
    exposure,
    rng,
    circuit,
    probe=None,
    probe_enabled=False,
    decision="majority"
):

    """
    Repeated optical reads.

    Every read obtains a fresh Poisson photon count.

    The accumulator records AND successes.

    The final bit is selected from accumulated
    photon-detection-driven circuit outputs.

    No additional random bit is generated.
    """

    successes = 0
    photons_total = 0

    for _ in range(retries):

        result = optical_read(
            right_roi,
            left_roi,
            bit,
            exposure,
            rng,
            circuit,
            probe,
            probe_enabled
        )

        successes += result["and"]
        photons_total += result["photons"]

    if decision == "majority":
        output = int(successes >= math.ceil(retries / 2))

    elif decision == "any":
        output = int(successes > 0)

    else:
        raise ValueError("Unknown decision rule")

    return output, successes, photons_total


# =========================================================
# SORTING NETWORK
# =========================================================

def gate(a, b):

    """
    AND/OR compare-exchange.

    LOW  = AND(a,b)
    HIGH = OR(a,b)
    """

    return int(a & b), int(a | b)


def sort_bits(bits, stats):

    """
    Odd-even transposition sorting network.

    Only AND/OR gates touch the data.
    """

    values = list(bits)

    for rnd in range(len(values)):

        for i in range(rnd % 2, len(values) - 1, 2):

            values[i], values[i + 1] = gate(
                values[i],
                values[i + 1]
            )

            stats["sort_gates"] += 1

    return values


def circuit_sort(
    right_roi,
    left_roi,
    data,
    m,
    retries,
    exposure,
    rng,
    stats,
    decision="majority",
    verbose=True
):

    """
    Sort integer values using threshold decomposition.

    For each threshold t:

        bit_i = [data_i >= t]

    The photon-driven circuit reads each bit.
    The AND/OR network sorts the resulting binary vector.

    Integer values are reconstructed by summing
    the sorted threshold vectors.
    """

    n = len(data)
    output = [0] * n

    circuit = MainCircuit()
    probe = IndependentProbe()

    for threshold in range(1, m):

        true_bits = [
            int(value >= threshold)
            for value in data
        ]

        read_bits = []

        for i, bit in enumerate(true_bits):

            read, successes, photons = read_bit(
                right_roi,
                left_roi,
                bit,
                retries,
                exposure,
                rng,
                circuit,
                probe,
                probe_enabled=False,
                decision=decision
            )

            read_bits.append(read)

            stats["trials"] += retries
            stats["photons"] += photons
            stats["miss"] += int(read != bit)
            stats["accumulator_reads"] += retries

        sorted_bits = sort_bits(
            read_bits,
            stats
        )

        output = [
            old + new
            for old, new in zip(output, sorted_bits)
        ]

        if verbose:

            print(
                f" t={threshold:<3} "
                f"in {''.join(map(str, read_bits))} "
                f"-> sorted "
                f"{''.join(map(str, sorted_bits))}"
            )

    stats["main_and"] += circuit.and_high
    stats["leak_or"] += circuit.or_high
    stats["main_gates"] += circuit.gates

    return output


# =========================================================
# DATASET GENERATION
# =========================================================

def make_dataset(digest, n, m, pattern="random"):

    data = [
        int.from_bytes(
            sha(
                digest
                + b"data"
                + i.to_bytes(8, "big")
            )[:8],
            "big"
        ) % m
        for i in range(n)
    ]

    if pattern == "sorted":
        data.sort()

    elif pattern == "reversed":
        data.sort(reverse=True)

    elif pattern == "dups":
        data = [
            value % min(m, 3)
            for value in data
        ]

    return data


def show(values):

    if len(values) <= 16:
        return str(values)

    return (
        f"[{', '.join(map(str, values[:8]))}, ... "
        f"{', '.join(map(str, values[-4:]))}] "
        f"(len {len(values)})"
    )


# =========================================================
# SAVE / LOAD
# =========================================================

def save_dataset(
    path,
    digest,
    data,
    sorted_data,
    metadata
):

    payload = {
        "digest": digest.hex(),
        "data": data,
        "sorted_data": sorted_data,
        "sorted_ok": sorted_data == sorted(data),
        "metadata": metadata
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def load_dataset(path):

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    digest = bytes.fromhex(payload["digest"])
    data = [int(x) for x in payload["data"]]

    return digest, data, payload


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Photon-driven optoelectronic sorting circuit"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0
    )

    parser.add_argument(
        "--synthetic",
        action="store_true"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=1
    )

    parser.add_argument(
        "--thr",
        type=int,
        default=1
    )

    parser.add_argument(
        "--n",
        type=int,
        default=8
    )

    parser.add_argument(
        "--m",
        type=int,
        default=8
    )

    parser.add_argument(
        "--pattern",
        choices=("random", "sorted", "reversed", "dups"),
        default="random"
    )

    parser.add_argument(
        "--data",
        default=None
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3200
    )

    parser.add_argument(
        "--exposure",
        type=float,
        default=0.050
    )

    parser.add_argument(
        "--decision",
        choices=("majority", "any"),
        default="majority"
    )

    parser.add_argument(
        "--photon-flux",
        type=float,
        default=PHOTON_FLUX
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed; omit for non-repeatable photon simulation"
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Apply actual 150ms software delay per main evaluation"
    )

    parser.add_argument(
        "--save",
        default=None
    )

    parser.add_argument(
        "--load",
        default=None
    )

    args = parser.parse_args()

    if args.n < 1 or args.m < 2:
        sys.exit("--n must be >= 1 and --m must be >= 2")

    if args.retries < 1:
        sys.exit("--retries must be >= 1")

    if args.exposure <= 0:
        sys.exit("--exposure must be positive")

    if args.frames < 1:
        sys.exit("--frames must be >= 1")

    if args.load and (args.data or args.save):
        sys.exit("--load cannot be combined with --data or --save")


    rng = np.random.default_rng(args.seed)

    source = (
        Synthetic()
        if args.synthetic
        else Camera(args.device)
    )

    previous = source.read().astype(np.int16)

    stats = {
        "trials": 0,
        "photons": 0,
        "miss": 0,
        "sort_gates": 0,
        "main_gates": 0,
        "main_and": 0,
        "leak_or": 0,
        "accumulator_reads": 0
    }

    correct = 0

    print("\n=== Photon-Driven Optoelectronic Sort ===")

    print(f"Input source:       {'Synthetic' if args.synthetic else 'Camera'}")
    print("Detector A:         Right ROI / photon counting")
    print("Detector B:         Left ROI / deterministic RC")
    print(f"Photon flux:        {PHOTON_FLUX}")
    print(f"Photon efficiency:  {PHOTON_EFFICIENCY}")
    print(f"Exposure:           {args.exposure * 1000:.1f} ms")
    print(f"RC time constant:   {RC_TAU * 1000:.1f} ms")
    print(f"Main delay:         {MAIN_DELAY * 1000:.1f} ms")
    print(f"Clock:              {1 / CLOCK_PERIOD:.1f} Hz")
    print(f"Decision:           {args.decision}")
    print(f"Retries:            {args.retries}")
    print(f"Random seed:        {args.seed}")

    try:

        for frame_index in range(args.frames):

            frame = source.read().astype(np.int16)

            motion = (
                np.abs(frame - previous) > args.thr
            ).astype(np.uint8)

            previous = frame

            digest = sha(motion.tobytes())

            # Split optical detector regions.

            right_roi, left_roi = detector_rois(
                motion * 255
            )

            b, vc, ve = driver_b(left_roi)

            print(
                f"\nFrame {frame_index}: "
                f"digest={digest.hex()[:16]}"
            )

            print(
                f"Detector B: HIGH={b} "
                f"Vc={vc:.3f}V "
                f"Ve={ve:.3f}V"
            )

            if args.load:

                digest, data, payload = load_dataset(
                    args.load
                )

                m = max(data) + 1 if data else 2

                print(f"Loaded: {args.load}")

            elif args.data:

                data = [
                    int(x)
                    for x in args.data.split(",")
                ]

                if not data or any(x < 0 for x in data):
                    sys.exit("Data must contain non-negative integers")

                m = max(data) + 1

            else:

                data = make_dataset(
                    digest,
                    args.n,
                    args.m,
                    args.pattern
                )

                m = args.m

            print(f"Dataset: {show(data)}")

            start = time.monotonic()

            result = circuit_sort(
                right_roi,
                left_roi,
                data,
                m,
                args.retries,
                args.exposure,
                rng,
                stats,
                decision=args.decision,
                verbose=len(data) <= 16
            )

            elapsed = time.monotonic() - start

            expected = sorted(data)
            ok = result == expected

            correct += int(ok)

            print(f"\nInput:    {show(data)}")
            print(f"Output:   {show(result)}")
            print(f"Expected: {show(expected)}")

            print(
                f"Result: {'OK' if ok else 'MISMATCH'}"
            )

            print(f"Elapsed: {elapsed:.3f}s")

            if args.save:

                path = args.save

                if args.frames > 1:

                    root, ext = os.path.splitext(path)

                    path = (
                        f"{root}_{frame_index}"
                        f"{ext or '.json'}"
                    )

                save_dataset(
                    path,
                    digest,
                    data,
                    result,
                    {
                        "retries": args.retries,
                        "exposure": args.exposure,
                        "decision": args.decision
                    }
                )

                print(f"Saved: {path}")

    finally:

        if isinstance(source, Camera):
            source.close()

    print("\n=== Circuit Statistics ===")

    print(f"Photon detections:       {stats['photons']}")
    print(f"Circuit trials:          {stats['trials']}")
    print(f"Main AND/OR gates:       {stats['main_gates']}")
    print(f"Sorting AND/OR gates:    {stats['sort_gates']}")
    print(f"Accumulator reads:       {stats['accumulator_reads']}")
    print(f"Main AND HIGH:           {stats['main_and']}")
    print(f"Leak OR HIGH:            {stats['leak_or']}")
    print(f"Bit-read errors:         {stats['miss']}")

    print(
        f"\nCorrect datasets: "
        f"{correct}/{args.frames}"
    )


if __name__ == "__main__":
    main()
