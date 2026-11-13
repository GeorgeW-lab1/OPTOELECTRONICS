
#!/usr/bin/env python3
"""
circuit_sort.py

CONTINUOUS CAMERA-DRIVEN OPTOELECTRONIC BTC DIRECTION PREDICTOR

Optical inputs:
    Right ROI -> Detector A -> stochastic photon detection
    Left ROI  -> Detector B -> deterministic RC/NPN model

Main circuit:
    AND -> 150ms delay -> accumulator -> prediction signal

Leak circuit:
    OR -> 74HC74 D input -> 10Hz clock -> OBSERVE

Independent probe:
    Separate photon-counting measurement.

Prediction:
    The optical circuit generates a signed prediction signal.
    Positive signal = BTC predicted HIGHER -> Shift+W
    Negative signal = BTC predicted LOWER  -> Shift+S

Timing:
    Capture -> circuit evaluation -> 150ms circuit delay
    -> additional 1 second -> BTC/USD fetch -> prediction -> keypress

BTC price does not determine the prediction direction.
It is used as the reference price for the predicted value.

Install:
    pip install numpy opencv-python pyautogui

Run:
    python circuit_sort.py
    python circuit_sort.py --synthetic
    python circuit_sort.py --synthetic --no-inject
"""

import argparse
import hashlib
import json
import math
import time
import urllib.request
from datetime import datetime, timezone

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

H = 96
W = 128

MAIN_DELAY = 0.150
POST_DELAY = 1.000
CLOCK_PERIOD = 0.100

VS = 5.0
VBE = 0.7
VIH = 3.5

RC_TAU = 0.050

PHOTON_EFFICIENCY = 0.80
PHOTON_FLUX = 20.0
DARK_COUNTS = 0.02

MOTION_THRESHOLD = 1

MAX_PREDICTED_CHANGE_PERCENT = 0.50
PREDICTION_HORIZON_SECONDS = 60.0

BTC_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"

LOG_FILE = "circuit_sort_log.jsonl"


# ============================================================
# CAMERA INPUT
# ============================================================

class CameraSource:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    def read(self):
        ok, frame = self.cap.read()

        if not ok:
            raise RuntimeError("Camera frame capture failed.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (W, H))

    def close(self):
        self.cap.release()


# ============================================================
# SYNTHETIC INPUT
# ============================================================

class SyntheticSource:
    def __init__(self):
        self.phase = 0.0

    def read(self):
        self.phase += 0.13

        x = np.arange(W)[None, :]
        y = np.arange(H)[:, None]

        wave = (
            35.0
            + 25.0 * np.sin(x * 0.09 + self.phase)
            + 20.0 * np.cos(y * 0.11 - self.phase * 0.7)
        )

        noise = np.random.normal(0, 8, (H, W))
        frame = np.clip(wave + noise, 0, 255)

        return frame.astype(np.uint8)

    def close(self):
        pass


# ============================================================
# DETECTOR REGIONS
# ============================================================

def detector_rois(frame):
    mid_x = frame.shape[1] // 2

    left = frame[:, :mid_x]
    right = frame[:, mid_x:]

    return right, left


# ============================================================
# DETECTOR A: STOCHASTIC PHOTON DETECTOR
# ============================================================

def photon_detection(roi):
    intensity = float(np.mean(roi)) / 255.0

    expected = max(
        0.0,
        intensity * PHOTON_FLUX * PHOTON_EFFICIENCY
        + DARK_COUNTS
    )

    detected = int(np.random.poisson(expected))

    return {
        "intensity": intensity,
        "expected_photons": expected,
        "detected_photons": detected,
        "active": detected > 0
    }


# ============================================================
# DETECTOR B: DETERMINISTIC RC / NPN
# ============================================================

class DetectorB:
    def __init__(self):
        self.capacitor_voltage = 0.0
        self.last_time = time.monotonic()

    def read(self, roi):
        now = time.monotonic()
        dt = max(0.001, now - self.last_time)
        self.last_time = now

        intensity = float(np.mean(roi)) / 255.0
        input_voltage = intensity * VS

        if input_voltage > VBE:
            target_voltage = input_voltage
        else:
            target_voltage = 0.0

        alpha = 1.0 - math.exp(-dt / RC_TAU)

        self.capacitor_voltage += alpha * (
            target_voltage - self.capacitor_voltage
        )

        npn_on = self.capacitor_voltage >= VIH

        return {
            "intensity": intensity,
            "input_voltage": input_voltage,
            "capacitor_voltage": self.capacitor_voltage,
            "npn_on": bool(npn_on)
        }


# ============================================================
# 74HC74 FLIP-FLOP MODEL
# ============================================================

class FlipFlop74HC74:
    def __init__(self, clock_period=CLOCK_PERIOD):
        self.clock_period = clock_period
        self.q = 0
        self.d = 0
        self.last_clock = time.monotonic()
        self.clock_edges = 0

    def set_d(self, value):
        self.d = int(bool(value))

    def clock(self):
        now = time.monotonic()
        elapsed = now - self.last_clock

        edges = int(elapsed / self.clock_period)

        if edges > 0:
            self.q = self.d
            self.clock_edges += edges
            self.last_clock += edges * self.clock_period

        return self.q


# ============================================================
# INDEPENDENT PHOTON PROBE
# ============================================================

class IndependentProbe:
    def measure(self, frame):
        intensity = float(np.mean(frame)) / 255.0

        expected = max(
            0.0,
            intensity * PHOTON_FLUX * PHOTON_EFFICIENCY
        )

        count = int(np.random.poisson(expected))

        return {
            "intensity": intensity,
            "expected_photons": expected,
            "detected_photons": count
        }


# ============================================================
# MAIN OPTICAL CIRCUIT
# ============================================================

class MainCircuit:
    def __init__(self):
        self.detector_b = DetectorB()
        self.flipflop = FlipFlop74HC74()

        self.accumulator = 0.0
        self.samples = 0

    def evaluate(self, detector_a, detector_b):
        a_signal = detector_a["active"]
        b_signal = detector_b["npn_on"]

        and_output = int(a_signal and b_signal)
        or_output = int(a_signal or b_signal)

        # Main circuit delay.
        time.sleep(MAIN_DELAY)

        # AND path feeds the accumulator.
        self.accumulator += float(and_output)
        self.samples += 1

        # Independent OR path feeds the flip-flop.
        self.flipflop.set_d(or_output)
        observed_q = self.flipflop.clock()

        accumulator_mean = (
            self.accumulator / self.samples
            if self.samples else 0.0
        )

        return {
            "and": and_output,
            "or": or_output,
            "accumulator": self.accumulator,
            "samples": self.samples,
            "accumulator_mean": accumulator_mean,
            "flipflop_d": self.flipflop.d,
            "flipflop_q": observed_q,
            "clock_edges": self.flipflop.clock_edges
        }


# ============================================================
# BTC/USD REFERENCE PRICE
# ============================================================

def fetch_btc_price():
    request = urllib.request.Request(
        BTC_URL,
        headers={"User-Agent": "CircuitSort/1.0"}
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return float(payload["data"]["amount"])


# ============================================================
# CIRCUIT-DRIVEN PREDICTOR
# ============================================================

class CircuitPredictor:
    """
    Experimental circuit-driven prediction mapping.

    Positive circuit signal:
        Predicted price higher.

    Negative circuit signal:
        Predicted price lower.

    The BTC reference price does not influence the direction.
    """

    def __init__(self):
        self.smoothed_signal = 0.0

    def predict(self, circuit_result, detector_a,
                detector_b, probe, spot_price):

        photon_level = min(
            1.0,
            detector_a["detected_photons"] / 10.0
        )

        voltage_level = min(
            1.0,
            detector_b["capacitor_voltage"] / VS
        )

        probe_level = min(
            1.0,
            probe["detected_photons"] / 10.0
        )

        accumulator_level = circuit_result["accumulator_mean"]

        and_signal = float(circuit_result["and"])
        or_signal = float(circuit_result["or"])
        flipflop_signal = float(circuit_result["flipflop_q"])

        # Circuit measurements form the prediction signal.
        raw_signal = (
            0.30 * (photon_level - 0.5)
            + 0.25 * (voltage_level - 0.5)
            + 0.15 * (probe_level - 0.5)
            + 0.15 * (accumulator_level - 0.5)
            + 0.10 * (and_signal - 0.5)
            + 0.05 * (or_signal - 0.5)
            + 0.05 * (flipflop_signal - 0.5)
        )

        # Smooth noisy circuit measurements.
        self.smoothed_signal = (
            0.75 * self.smoothed_signal
            + 0.25 * raw_signal
        )

        # Bound the prediction signal.
        bounded_signal = math.tanh(
            self.smoothed_signal * 2.0
        )

        # Positive = higher; negative = lower.
        predicted_change_percent = (
            bounded_signal * MAX_PREDICTED_CHANGE_PERCENT
        )

        predicted_price = spot_price * (
            1.0 + predicted_change_percent / 100.0
        )

        direction = (
            "HIGHER"
            if bounded_signal >= 0
            else "LOWER"
        )

        return {
            "raw_signal": raw_signal,
            "smoothed_signal": self.smoothed_signal,
            "bounded_signal": bounded_signal,
            "direction": direction,
            "predicted_change_percent": predicted_change_percent,
            "reference_price": spot_price,
            "predicted_price": predicted_price,
            "horizon_seconds": PREDICTION_HORIZON_SECONDS
        }


# ============================================================
# KEYBOARD INJECTION
# ============================================================

def inject_prediction(direction, enabled=True):
    """
    HIGHER -> Shift+W
    LOWER  -> Shift+S
    """

    if not enabled:
        print("Keyboard injection disabled.")
        return False

    try:
        import pyautogui

        if direction == "HIGHER":
            pyautogui.hotkey("shift", "w")
            print("KEY INJECTED: Shift+W")

        elif direction == "LOWER":
            pyautogui.hotkey("shift", "s")
            print("KEY INJECTED: Shift+S")

        else:
            print("Unknown prediction direction.")
            return False

        return True

    except ImportError:
        print("pyautogui is not installed.")
        print("Run: pip install pyautogui")
        return False

    except Exception as exc:
        print("Keyboard injection failed:", repr(exc))
        return False


# ============================================================
# LOGGING
# ============================================================

def write_log(record):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# ============================================================
# ONE COMPLETE PREDICTION CYCLE
# ============================================================

def run_cycle(frame_index, source, previous_frame,
              circuit, probe_device, predictor, inject=True):

    # 1. Capture camera frame.
    frame = source.read()

    # 2. Calculate motion.
    difference = cv2.absdiff(frame, previous_frame)

    _, motion = cv2.threshold(
        difference,
        MOTION_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    digest = hashlib.sha256(
        motion.tobytes()
    ).hexdigest()

    # 3. Split optical detector inputs.
    right_roi, left_roi = detector_rois(motion)

    # 4. Detector measurements.
    detector_a = photon_detection(right_roi)
    detector_b = circuit.detector_b.read(left_roi)

    # 5. Independent probe.
    probe = probe_device.measure(frame)

    # 6. Main optical circuit and 150ms delay.
    circuit_result = circuit.evaluate(
        detector_a,
        detector_b
    )

    # 7. Additional one-second wait.
    time.sleep(POST_DELAY)

    # 8. Fetch BTC/USD reference price.
    spot_price = fetch_btc_price()

    # 9. Generate direction from optical circuit measurements.
    prediction = predictor.predict(
        circuit_result,
        detector_a,
        detector_b,
        probe,
        spot_price
    )

    # 10. Select keyboard command from prediction.
    direction = prediction["direction"]

    key_result = inject_prediction(
        direction,
        enabled=inject
    )

    # 11. Report.
    print("\n" + "=" * 60)
    print("CYCLE:", frame_index)
    print("UTC:", datetime.now(timezone.utc).isoformat())
    print("Motion SHA256:", digest)

    print("\nDETECTOR A")
    print("Photons:", detector_a["detected_photons"])
    print("Intensity:", round(detector_a["intensity"], 5))

    print("\nDETECTOR B")
    print("RC voltage:", round(
        detector_b["capacitor_voltage"], 5
    ))
    print("NPN:", detector_b["npn_on"])

    print("\nMAIN CIRCUIT")
    print("AND:", circuit_result["and"])
    print("OR:", circuit_result["or"])
    print("Accumulator:", circuit_result["accumulator"])
    print("Accumulator mean:", round(
        circuit_result["accumulator_mean"], 5
    ))
    print("74HC74 Q:", circuit_result["flipflop_q"])

    print("\nINDEPENDENT PROBE")
    print("Photons:", probe["detected_photons"])

    print("\nBTC/USD")
    print("Reference:", round(spot_price, 2))
    print("Prediction:", direction)
    print(
        "Predicted change:",
        round(prediction["predicted_change_percent"], 6),
        "%"
    )
    print(
        "Predicted price:",
        round(prediction["predicted_price"], 2)
    )

    print("\nKEYBOARD")
    print("Direction:", direction)
    print("Command:", "Shift+W" if direction == "HIGHER" else "Shift+S")
    print("Injected:", key_result)

    # 12. Append cycle record to JSONL log.
    record = {
        "cycle": frame_index,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "motion_sha256": digest,
        "detector_a": detector_a,
        "detector_b": detector_b,
        "probe": probe,
        "circuit": circuit_result,
        "btc_usd_reference": spot_price,
        "prediction": prediction,
        "keyboard_direction": direction,
        "keyboard_injected": key_result
    }

    write_log(record)

    return frame


# ============================================================
# CONTINUOUS MAIN LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optoelectronic BTC direction predictor"
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic frames instead of webcam"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index"
    )

    parser.add_argument(
        "--no-inject",
        action="store_true",
        help="Disable keyboard injection"
    )

    args = parser.parse_args()

    if args.synthetic:
        source = SyntheticSource()
    else:
        source = CameraSource(args.camera)

    circuit = MainCircuit()
    probe_device = IndependentProbe()
    predictor = CircuitPredictor()

    previous_frame = source.read()
    frame_index = 0

    print("=" * 60)
    print("OPTOELECTRONIC BTC DIRECTION PREDICTOR")
    print("=" * 60)
    print("Input:", "Synthetic" if args.synthetic else "Camera")
    print("Main delay:", MAIN_DELAY, "seconds")
    print("Additional wait:", POST_DELAY, "seconds")
    print("Clock period:", CLOCK_PERIOD, "seconds")
    print("Higher:", "Shift+W")
    print("Lower:", "Shift+S")
    print("Keyboard injection:", not args.no_inject)
    print("Prediction horizon:", PREDICTION_HORIZON_SECONDS, "seconds")
    print("Log:", LOG_FILE)
    print("\nPress Ctrl+C to stop.\n")

    try:
        while True:
            try:
                current_frame = run_cycle(
                    frame_index,
                    source,
                    previous_frame,
                    circuit,
                    probe_device,
                    predictor,
                    inject=not args.no_inject
                )

                previous_frame = current_frame
                frame_index += 1

            except Exception as exc:
                print("\nCycle error:", repr(exc))
                print("Retrying in 2 seconds...")
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nStopping prediction loop.")

    finally:
        source.close()
        print("Camera closed.")
        print("Completed cycles:", frame_index)


if __name__ == "__main__":
    main()
