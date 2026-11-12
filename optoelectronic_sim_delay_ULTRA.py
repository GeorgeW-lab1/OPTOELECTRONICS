
#!/usr/bin/env python3
"""
Webcam Optoelectronic AND Circuit — Independent Secondary Probe

Topology
========

                         +--> MAIN AND --> configurable delay --> AND_dly --+
Photon ROIs -> Drivers --|                                                    OR --> DFF
                         +--> SECONDARY AND --> probe measurement            |
                                                    (no delay)                |
                                                                            Y -+

Main delay modes:
    post : AND(a, b) -> delay
    a    : delay(a), AND with b
    b    : AND with delay(b)

The secondary probe always receives the live, undelayed a and b signals.
It is not part of the main signal path and is not used as a time-aligned
taint label for delayed DFF samples.

Install:
    pip install numpy opencv-python matplotlib

Run:
    python webcam_and_probe.py

Examples:
    python webcam_and_probe.py --camera 0
    python webcam_and_probe.py --synthetic
    python webcam_and_probe.py --delay-ms 150 --delay-mode post --probe
    python webcam_and_probe.py --delay-ms 250 --delay-mode a --probe --leak
"""

import argparse
import hashlib
import secrets
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class Config:
    camera: int = 0
    synthetic: bool = False
    duration: float = 30.0
    sample_hz: float = 10.0
    delay_ms: float = 150.0
    delay_mode: str = "post"
    leak: bool = True
    probe: bool = False
    probe_err: float = 0.0
    roi_threshold: float = 0.5
    driver_a_mode: str = "intensity"
    ema_alpha: float = 0.25
    rc_tau_ms: float = 50.0
    npn_threshold: float = 0.7
    schmitt_low: float = 0.42
    schmitt_high: float = 0.58
    words: int = 100
    seed: int = 42
    plot: bool = True
    show_camera: bool = True
    trace_file: str = ""


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def bit(value):
    return 1 if bool(value) else 0


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def bits_to_string(values):
    return "".join(str(bit(v)) for v in values)


# ---------------------------------------------------------------------
# Webcam and synthetic photon source
# ---------------------------------------------------------------------

class PhotonSource:
    """
    Reads two image regions from a webcam.

    The left half is detector B.
    The right half is detector A.

    In synthetic mode, two noisy photon-intensity fields are generated.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cap = None
        self.frame = None
        self.synthetic_rng = np.random.default_rng(cfg.seed)
        self.synthetic_t = 0.0

        if not cfg.synthetic:
            if cv2 is None:
                raise RuntimeError(
                    "OpenCV is required for webcam mode. "
                    "Install with: pip install opencv-python"
                )

            self.cap = cv2.VideoCapture(cfg.camera)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera index {cfg.camera}. "
                    "Use --synthetic to run without a webcam."
                )

    def read(self, dt):
        if self.cfg.synthetic:
            return self._synthetic_read(dt)

        ok, frame = self.cap.read()

        if not ok or frame is None:
            raise RuntimeError("Webcam frame could not be read.")

        self.frame = frame
        height, width = frame.shape[:2]
        midpoint = width // 2

        roi_b = frame[:, :midpoint]
        roi_a = frame[:, midpoint:]

        intensity_a = float(np.mean(cv2.cvtColor(
            roi_a, cv2.COLOR_BGR2GRAY
        ))) / 255.0

        intensity_b = float(np.mean(cv2.cvtColor(
            roi_b, cv2.COLOR_BGR2GRAY
        ))) / 255.0

        return intensity_a, intensity_b, frame

    def _synthetic_read(self, dt):
        self.synthetic_t += dt
        t = self.synthetic_t

        wave_a = 0.5 + 0.35 * np.sin(2.0 * np.pi * 0.37 * t)
        wave_b = 0.5 + 0.35 * np.sin(2.0 * np.pi * 0.37 * t + 1.1)

        noise_a = self.synthetic_rng.normal(0.0, 0.09)
        noise_b = self.synthetic_rng.normal(0.0, 0.09)

        intensity_a = clamp(wave_a + noise_a, 0.0, 1.0)
        intensity_b = clamp(wave_b + noise_b, 0.0, 1.0)

        return intensity_a, intensity_b, None

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------
# Driver A
# ---------------------------------------------------------------------

class DriverA:
    """
    Converts detector A intensity into a digital signal.

    intensity:
        Uses an EMA-smoothed intensity threshold.

    parity:
        Hashes a quantised intensity value and uses its parity.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.ema = 0.0

    def step(self, intensity):
        alpha = self.cfg.ema_alpha
        self.ema = alpha * intensity + (1.0 - alpha) * self.ema

        if self.cfg.driver_a_mode == "parity":
            quantised = int(clamp(self.ema, 0.0, 1.0) * 65535)
            digest = hashlib.sha256(
                str(quantised).encode("utf-8")
            ).digest()
            return bool(digest[0] & 1)

        return self.ema >= self.cfg.roi_threshold


# ---------------------------------------------------------------------
# Driver B — Schmitt trigger
# ---------------------------------------------------------------------

class DriverB:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = False

    def step(self, intensity):
        if intensity >= self.cfg.schmitt_high:
            self.state = True
        elif intensity <= self.cfg.schmitt_low:
            self.state = False

        return self.state


# ---------------------------------------------------------------------
# RC node and NPN-style threshold link
# ---------------------------------------------------------------------

class RCNode:
    def __init__(self, tau_ms):
        self.tau = max(0.000001, tau_ms / 1000.0)
        self.voltage = 0.0

    def step(self, input_value, dt):
        alpha = 1.0 - np.exp(-dt / self.tau)
        target = 1.0 if input_value else 0.0
        self.voltage += alpha * (target - self.voltage)
        return self.voltage


class IfThenLink:
    """
    NPN-style threshold model:

        IF V(C) > threshold:
            output HIGH
        ELSE:
            output LOW
    """

    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def step(self, voltage):
        return voltage > self.threshold


# ---------------------------------------------------------------------
# Delay line
# ---------------------------------------------------------------------

class DelayLine:
    """
    Discrete-time delay line.

    The delay is represented by a queue of sampled values. The requested
    delay is rounded to the nearest simulation sample.
    """

    def __init__(self, delay_ms, sample_hz, initial=False):
        self.sample_hz = max(0.1, sample_hz)
        self.dt = 1.0 / self.sample_hz
        self.delay_ms = max(0.0, delay_ms)
        self.delay_samples = int(round(
            (self.delay_ms / 1000.0) / self.dt
        ))
        self.queue = deque(
            [bool(initial)] * (self.delay_samples + 1)
        )

    def step(self, value):
        self.queue.append(bool(value))
        return self.queue.popleft()

    @property
    def effective_delay_ms(self):
        return self.delay_samples * 1000.0 / self.sample_hz


# ---------------------------------------------------------------------
# Logic gates
# ---------------------------------------------------------------------

def and_gate(a, b):
    return bool(a and b)


def or_gate(a, b):
    return bool(a or b)


# ---------------------------------------------------------------------
# D flip-flop
# ---------------------------------------------------------------------

class DFlipFlop:
    def __init__(self, initial=False):
        self.q = bool(initial)

    def clock(self, d):
        self.q = bool(d)
        return self.q


# ---------------------------------------------------------------------
# Salted SHA-256 commitment
# ---------------------------------------------------------------------

class Commitment:
    """
    Commits to a secret bit string using SHA-256.

    This is a commitment/checking mechanism, not a source of information
    about the secret.
    """

    def __init__(self, secret_bits=None):
        if secret_bits is None:
            secret_bits = secrets.randbits(64)

        self.secret_bits = int(secret_bits)
        self.secret_text = format(self.secret_bits, "064b")
        self.salt = secrets.token_bytes(16)

        payload = self.salt + self.secret_text.encode("ascii")
        self.digest = sha256_hex(payload)

    def verify(self, candidate):
        candidate = str(candidate)

        if len(candidate) != len(self.secret_text):
            return False

        payload = self.salt + candidate.encode("ascii")
        return sha256_hex(payload) == self.digest

    def public_info(self):
        return {
            "salt": self.salt.hex(),
            "commitment": self.digest,
            "length": len(self.secret_text),
        }


# ---------------------------------------------------------------------
# AND accumulator
# ---------------------------------------------------------------------

class AndAccumulator:
    """
    Maintains a bitwise AND accumulator over observed main-path AND values.

    This reports the accumulator state. It does not independently recover
    the committed secret.
    """

    def __init__(self):
        self.value = None
        self.count = 0
        self.history = []

    def update(self, observed):
        observed = bit(observed)

        if self.value is None:
            self.value = observed
        else:
            self.value &= observed

        self.count += 1
        self.history.append(self.value)
        return self.value

    def summary(self):
        return {
            "count": self.count,
            "value": self.value,
        }


# ---------------------------------------------------------------------
# Leakage and probe meter
# ---------------------------------------------------------------------

class LeakMeter:
    """
    Measures main-path leakage and secondary-probe measurement error.

    Main-path p_leak:
        P(AND_dly = 1 | Y = 0)

    Probe error:
        Fraction of secondary probe measurements that differ from the
        secondary branch's own undelayed AND output.
    """

    def __init__(self):
        self.y_values = []
        self.and_values = []
        self.and_dly_values = []

        self.probe_estimates = []
        self.probe_truth = []

        self.a_values = []
        self.b_values = []

    def slot(
        self,
        y,
        and_value,
        and_dly,
        a=None,
        b=None,
        probe_estimate=None,
        probe_truth=None,
    ):
        self.y_values.append(bit(y))
        self.and_values.append(bit(and_value))
        self.and_dly_values.append(bit(and_dly))

        if a is not None:
            self.a_values.append(bit(a))

        if b is not None:
            self.b_values.append(bit(b))

        if probe_estimate is not None and probe_truth is not None:
            self.probe_estimates.append(bit(probe_estimate))
            self.probe_truth.append(bit(probe_truth))

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    def summary(self):
        y = np.asarray(self.y_values, dtype=np.int8)
        and_dly = np.asarray(self.and_dly_values, dtype=np.int8)

        y_zero = y == 0

        if np.any(y_zero):
            p_leak = float(np.mean(and_dly[y_zero]))
        else:
            p_leak = 0.0

        probe_error = None

        if self.probe_estimates:
            estimates = np.asarray(
                self.probe_estimates, dtype=np.int8
            )
            truth = np.asarray(self.probe_truth, dtype=np.int8)
            probe_error = float(np.mean(estimates != truth))

        correlation = None

        if len(self.a_values) > 1 and len(self.b_values) > 1:
            a = np.asarray(self.a_values, dtype=float)
            b = np.asarray(self.b_values, dtype=float)

            if np.std(a) > 0 and np.std(b) > 0:
                correlation = float(np.corrcoef(a, b)[0, 1])

        return {
            "samples": len(self.y_values),
            "p_leak": p_leak,
            "probe_error": probe_error,
            "correlation_ab": correlation,
            "main_and_duty": self._mean(self.and_values),
            "delayed_and_duty": self._mean(self.and_dly_values),
            "probe_samples": len(self.probe_estimates),
        }


# ---------------------------------------------------------------------
# Trace storage
# ---------------------------------------------------------------------

class Trace:
    def __init__(self):
        self.data = {
            "time": [],
            "intensity_a": [],
            "intensity_b": [],
            "a": [],
            "b_driver": [],
            "rc_voltage": [],
            "b_out": [],
            "main_a": [],
            "main_b": [],
            "main_and": [],
            "and_dly": [],
            "y": [],
            "d": [],
            "q": [],
            "probe_and": [],
            "probe_measurement": [],
        }

    def add(self, **values):
        for key in self.data:
            if key in values:
                self.data[key].append(values[key])

    def save_csv(self, filename):
        import csv

        keys = list(self.data.keys())

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)

            rows = zip(*(self.data[key] for key in keys))
            writer.writerows(rows)


# ---------------------------------------------------------------------
# Independent secondary probe
# ---------------------------------------------------------------------

class SecondaryAndProbe:
    """
    Independent parallel AND branch.

    It receives the raw live driver outputs. It does not receive:
        - delayed A
        - delayed B
        - main AND output
        - delayed main AND output

    Measurement error is applied only after the secondary AND gate.
    """

    def __init__(self, error_probability=0.0, seed=1):
        self.error_probability = clamp(
            error_probability, 0.0, 1.0
        )
        self.rng = np.random.default_rng(seed)

    def step(self, a, b):
        truth = and_gate(a, b)
        measured = truth

        if self.rng.random() < self.error_probability:
            measured = not measured

        return truth, measured


# ---------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------

def run_session(cfg):
    rng = np.random.default_rng(cfg.seed)

    source = PhotonSource(cfg)
    driver_a = DriverA(cfg)
    driver_b = DriverB(cfg)
    rc_node = RCNode(cfg.rc_tau_ms)
    npn = IfThenLink(cfg.npn_threshold)

    main_delay = DelayLine(
        cfg.delay_ms,
        cfg.sample_hz,
        initial=False,
    )

    dff = DFlipFlop()
    accumulator = AndAccumulator()
    meter = LeakMeter()
    trace = Trace()

    probe = SecondaryAndProbe(
        error_probability=cfg.probe_err,
        seed=cfg.seed + 100,
    )

    commitment = Commitment()

    dt = 1.0 / max(0.1, cfg.sample_hz)
    total_samples = max(1, int(cfg.duration * cfg.sample_hz))

    print("\n=== Webcam Optoelectronic AND Circuit ===")
    print(f"Input mode:       {'synthetic' if cfg.synthetic else 'webcam'}")
    print(f"Main delay mode:  {cfg.delay_mode}")
    print(f"Requested delay:  {cfg.delay_ms:.2f} ms")
    print(f"Effective delay:  {main_delay.effective_delay_ms:.2f} ms")
    print(f"Secondary probe:  {'enabled' if cfg.probe else 'disabled'}")
    print(f"Leak path:        {'enabled' if cfg.leak else 'disabled'}")
    print(f"Sample rate:      {cfg.sample_hz:.2f} Hz")
    print(f"Duration:         {cfg.duration:.2f} s")
    print("----------------------------------------")
    print("Commitment:", commitment.digest)
    print("----------------------------------------")

    start = time.monotonic()
    next_sample = start

    try:
        for index in range(total_samples):
            now = time.monotonic()

            if not cfg.synthetic and cfg.show_camera:
                intensity_a, intensity_b, frame = source.read(dt)
            else:
                intensity_a, intensity_b, frame = source.read(dt)

            # ---------------------------------------------------------
            # Detector and driver stages
            # ---------------------------------------------------------

            a = driver_a.step(intensity_a)

            b_driver = driver_b.step(intensity_b)

            # Driver B -> RC node -> NPN-style threshold link.
            rc_voltage = rc_node.step(b_driver, dt)
            b_out = npn.step(rc_voltage)

            # ---------------------------------------------------------
            # Fan-out point
            #
            # The driver outputs are now available to two independent
            # branches:
            #
            # 1. Main delayed AND branch.
            # 2. Secondary undelayed probe branch.
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # Main branch
            # ---------------------------------------------------------

            if cfg.delay_mode == "a":
                main_a = main_delay.step(a)
                main_b = b_out
                main_and = and_gate(main_a, main_b)
                and_dly = main_and

            elif cfg.delay_mode == "b":
                main_a = a
                main_b = main_delay.step(b_out)
                main_and = and_gate(main_a, main_b)
                and_dly = main_and

            else:
                main_a = a
                main_b = b_out
                main_and = and_gate(main_a, main_b)
                and_dly = main_delay.step(main_and)

            # OR branch from the current main-branch inputs.
            y = or_gate(main_a, main_b)

            # D = Y OR delayed main AND when leakage is enabled.
            if cfg.leak:
                d = or_gate(y, and_dly)
            else:
                d = y

            # ---------------------------------------------------------
            # Independent secondary probe branch
            #
            # IMPORTANT:
            # This always uses raw, live a and b_out.
            # It is not connected to main_delay.
            # ---------------------------------------------------------

            probe_truth = None
            probe_measurement = None

            if cfg.probe:
                probe_truth, probe_measurement = probe.step(a, b_out)

            # ---------------------------------------------------------
            # Clock and accumulator
            # ---------------------------------------------------------

            q = dff.clock(d)
            accumulator_value = accumulator.update(and_dly)

            # Metering is performed at each sample.
            meter.slot(
                y=y,
                and_value=main_and,
                and_dly=and_dly,
                a=main_a,
                b=main_b,
                probe_estimate=probe_measurement,
                probe_truth=probe_truth,
            )

            # ---------------------------------------------------------
            # Trace collection
            # ---------------------------------------------------------

            trace.add(
                time=index * dt,
                intensity_a=intensity_a,
                intensity_b=intensity_b,
                a=bit(a),
                b_driver=bit(b_driver),
                rc_voltage=rc_voltage,
                b_out=bit(b_out),
                main_a=bit(main_a),
                main_b=bit(main_b),
                main_and=bit(main_and),
                and_dly=bit(and_dly),
                y=bit(y),
                d=bit(d),
                q=bit(q),
                probe_and=(
                    bit(probe_truth)
                    if probe_truth is not None else np.nan
                ),
                probe_measurement=(
                    bit(probe_measurement)
                    if probe_measurement is not None else np.nan
                ),
            )

            # ---------------------------------------------------------
            # Optional camera display
            # ---------------------------------------------------------

            if (
                frame is not None
                and cfg.show_camera
                and cv2 is not None
            ):
                height, width = frame.shape[:2]
                midpoint = width // 2

                cv2.line(
                    frame,
                    (midpoint, 0),
                    (midpoint, height),
                    (0, 255, 255),
                    2,
                )

                status = (
                    f"A={bit(a)} B={bit(b_out)} "
                    f"AND={bit(main_and)} "
                    f"DELAY={bit(and_dly)} "
                    f"D={bit(d)} Q={bit(q)}"
                )

                cv2.putText(
                    frame,
                    status,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                if cfg.probe:
                    probe_status = (
                        f"SECONDARY AND={bit(probe_truth)} "
                        f"MEASURED={bit(probe_measurement)}"
                    )

                    cv2.putText(
                        frame,
                        probe_status,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2,
                    )

                cv2.imshow("Optoelectronic AND Circuit", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

            # ---------------------------------------------------------
            # Console output
            # ---------------------------------------------------------

            if index % max(1, int(cfg.sample_hz)) == 0:
                line = (
                    f"\rSample {index + 1:5d}/{total_samples} "
                    f"A={bit(a)} B={bit(b_out)} "
                    f"AND={bit(main_and)} "
                    f"AND_dly={bit(and_dly)} "
                    f"Y={bit(y)} D={bit(d)} Q={bit(q)}"
                )

                if cfg.probe:
                    line += (
                        f" | Probe={bit(probe_truth)}"
                        f" Measured={bit(probe_measurement)}"
                    )

                print(line, end="", flush=True)

            # Maintain a roughly fixed simulation sample interval.
            next_sample += dt
            sleep_time = next_sample - time.monotonic()

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        source.close()

        if cv2 is not None:
            cv2.destroyAllWindows()

    print("\n\n=== Session Results ===")

    meter_result = meter.summary()
    accumulator_result = accumulator.summary()

    print(f"Samples:              {meter_result['samples']}")
    print(f"Main AND duty:        {meter_result['main_and_duty']:.4f}")
    print(f"Delayed AND duty:     {meter_result['delayed_and_duty']:.4f}")
    print(f"P(AND_dly=1 | Y=0):   {meter_result['p_leak']:.4f}")

    if meter_result["correlation_ab"] is not None:
        print(
            "A/B correlation:      "
            f"{meter_result['correlation_ab']:.4f}"
        )

    if cfg.probe:
        print(
            "Probe samples:        "
            f"{meter_result['probe_samples']}"
        )

        if meter_result["probe_error"] is not None:
            print(
                "Probe measurement error: "
                f"{meter_result['probe_error']:.4f}"
            )

        print(
            "Probe topology:       independent, parallel, undelayed"
        )

    print(
        "AND accumulator:      "
        f"{accumulator_result['value']}"
    )
    print(
        "Accumulator samples:  "
        f"{accumulator_result['count']}"
    )

    print("\n=== Commitment Verification ===")
    print("Commitment:", commitment.digest)
    print("Salt:", commitment.salt.hex())
    print("Secret length:", len(commitment.secret_text), "bits")

    # The true secret is deliberately not supplied to the accumulator.
    # The commitment is verified only against an explicitly supplied
    # candidate. No secret recovery is claimed here.
    print(
        "Verification: no candidate supplied; "
        "the commitment has not been opened."
    )

    if cfg.trace_file:
        trace.save_csv(cfg.trace_file)
        print(f"\nTrace saved to: {cfg.trace_file}")

    if cfg.plot and plt is not None:
        plot_trace(trace, cfg)

    return {
        "meter": meter_result,
        "accumulator": accumulator_result,
        "commitment": commitment.public_info(),
        "trace": trace,
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_trace(trace, cfg):
    if plt is None:
        print("Matplotlib is unavailable; skipping plots.")
        return

    data = trace.data
    t = data["time"]

    if not t:
        return

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
    )

    axes[0].plot(t, data["intensity_a"], label="Detector A intensity")
    axes[0].plot(t, data["intensity_b"], label="Detector B intensity")
    axes[0].set_ylabel("Intensity")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(t, data["a"], where="post", label="Driver A")
    axes[1].step(t, data["b_out"], where="post", label="Driver B out")
    axes[1].plot(t, data["rc_voltage"], label="RC voltage")
    axes[1].set_ylabel("Driver / RC")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    axes[2].step(t, data["main_and"], where="post", label="Main AND")
    axes[2].step(t, data["and_dly"], where="post", label="Main AND delayed")

    if cfg.probe:
        axes[2].step(
            t,
            data["probe_and"],
            where="post",
            label="Secondary AND truth",
        )
        axes[2].step(
            t,
            data["probe_measurement"],
            where="post",
            label="Secondary probe measurement",
            alpha=0.7,
        )

    axes[2].set_ylabel("AND branches")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)

    axes[3].step(t, data["y"], where="post", label="Y")
    axes[3].step(t, data["d"], where="post", label="D")
    axes[3].step(t, data["q"], where="post", label="DFF Q")
    axes[3].set_ylabel("Output")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(
        "Optoelectronic Circuit — Main Delayed AND and Independent Probe"
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Webcam-driven optoelectronic AND circuit with "
            "an independent undelayed secondary AND probe."
        )
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")

    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)

    parser.add_argument("--delay-ms", type=float, default=150.0)
    parser.add_argument(
        "--delay-mode",
        choices=("post", "a", "b"),
        default="post",
    )

    parser.add_argument("--leak", action="store_true", default=True)
    parser.add_argument("--no-leak", action="store_false", dest="leak")

    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-err", type=float, default=0.0)

    parser.add_argument("--roi-threshold", type=float, default=0.5)
    parser.add_argument(
        "--driver-a-mode",
        choices=("intensity", "parity"),
        default="intensity",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.25)

    parser.add_argument("--rc-tau-ms", type=float, default=50.0)
    parser.add_argument("--npn-threshold", type=float, default=0.7)

    parser.add_argument("--schmitt-low", type=float, default=0.42)
    parser.add_argument("--schmitt-high", type=float, default=0.58)

    parser.add_argument("--words", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-plot", action="store_false", dest="plot")
    parser.add_argument(
        "--no-camera-window",
        action="store_false",
        dest="show_camera",
    )

    parser.add_argument("--trace-file", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.duration <= 0:
        raise SystemExit("--duration must be greater than zero.")

    if args.sample_hz <= 0:
        raise SystemExit("--sample-hz must be greater than zero.")

    if args.delay_ms < 0:
        raise SystemExit("--delay-ms cannot be negative.")

    if not 0.0 <= args.probe_err <= 1.0:
        raise SystemExit("--probe-err must be between 0 and 1.")

    if not 0.0 <= args.ema_alpha <= 1.0:
        raise SystemExit("--ema-alpha must be between 0 and 1.")

    if args.schmitt_low >= args.schmitt_high:
        raise SystemExit(
            "--schmitt-low must be lower than --schmitt-high."
        )

    cfg = Config(
        camera=args.camera,
        synthetic=args.synthetic,
        duration=args.duration,
        sample_hz=args.sample_hz,
        delay_ms=args.delay_ms,
        delay_mode=args.delay_mode,
        leak=args.leak,
        probe=args.probe,
        probe_err=args.probe_err,
        roi_threshold=args.roi_threshold,
        driver_a_mode=args.driver_a_mode,
        ema_alpha=args.ema_alpha,
        rc_tau_ms=args.rc_tau_ms,
        npn_threshold=args.npn_threshold,
        schmitt_low=args.schmitt_low,
        schmitt_high=args.schmitt_high,
        words=args.words,
        seed=args.seed,
        plot=args.plot,
        show_camera=args.show_camera,
        trace_file=args.trace_file,
    )

    run_session(cfg)


if __name__ == "__main__":
    main()
