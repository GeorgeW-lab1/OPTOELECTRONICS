#!/usr/bin/env python3
"""
Webcam-driven optoelectronic simulation.

The isolated number remains fixed.

After every PriceOracle sample completes, the program:
1. Fetches a second live BTC-USD price.
2. Compares it with the BTC price the oracle just used.
3. Prints both BTC floats, the difference, and the result.
4. Does NOT save, cache, or update the isolated number.

Usage:
python optoelectronic_sim-5.py --price
python optoelectronic_sim-5.py --price --price-poll 2
python optoelectronic_sim-5.py --synthetic --headless --price --runs 5
python optoelectronic_sim-5.py --cam 1 --price --price-poll 2
python optoelectronic_sim-5.py --price --no-keys   (disable key injection)
python optoelectronic_sim-5.py --price --fetch-delay 5
    (inject key after oracle, wait 5 s more, then fetch fresh price)

Key injection (requires: pip install pynput):
Next oracle price above the last oracle price -> taps Shift+W
Next oracle price below the last oracle price -> taps Shift+S

Adaptive photonic decision (see the "matrix bias" diagram):
A classical OR gate over detectors (A,B) is biased: 25% zeros /
75% ones. AdaptivePhotonicDecision treats the 4-row truth table
as a tunable matrix [0, w, w, 1]. Measured output statistics are
fed back to adjust w (the "phase shift") until P(1) -> 0.5.
It is a diagnostic only; it does not choose keys.
Use --classical to freeze w=1 (plain OR) and compare.

Keys are decided ONLY by consecutive oracle prices (oracle v
oracle). The fresh price is printed for comparison but never
injects a key. If the oracle price is identical to the last
oracle price, the balanced photonic bit picks the key
(1 -> Shift+W, 0 -> Shift+S). Use --no-tiebreak for no key.
"""

import argparse
import collections
import json
import queue
import threading
import time
import urllib.request

import numpy as np


VCC = 5.0
V_BE = 0.7
R_OHM = 10e3
C_FARAD = 15e-6
CLK_PERIOD = 0.100


class Camera:
    def __init__(self, index, synthetic=False):
        self.synthetic = synthetic
        self.t0 = time.time()
        self.rng = np.random.default_rng(1)

        if not synthetic:
            import cv2
            self.cap = cv2.VideoCapture(index)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot open webcam {index}"
                )

    def read(self):
        import cv2

        if self.synthetic:
            t = time.time() - self.t0
            h, w = 240, 320

            base = np.zeros((h, w), np.float32)
            base[:, : w // 2] = 110 + 90 * np.sin(
                2 * np.pi * 0.4 * t
            )
            base[:, w // 2:] = 110 + 90 * np.sin(
                2 * np.pi * 0.7 * t + 1.0
            )

            image = np.clip(
                base
                + self.rng.normal(0, 3, (h, w)),
                0,
                255,
            ).astype(np.uint8)

            return (
                image,
                cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR,
                ),
            )

        ok, bgr = self.cap.read()

        if not ok:
            raise RuntimeError(
                "Webcam read failed"
            )

        return (
            cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2GRAY,
            ),
            bgr,
        )

    def release(self):
        if not self.synthetic:
            self.cap.release()


class DriverB:
    def __init__(self, x):
        self.threshold = 0.30 + 0.40 * (
            ((x * 2654435761) % 2**32)
            / 2**32
        )
        self.hysteresis = 0.03
        self.state = False

    def step(self, intensity):
        if (
            self.state
            and intensity
            < self.threshold - self.hysteresis
        ):
            self.state = False

        elif (
            not self.state
            and intensity
            > self.threshold + self.hysteresis
        ):
            self.state = True

        return VCC if self.state else 0.0


class DriverA:
    def step(self, roi):
        parity = (
            int(np.bitwise_and(roi, 1).sum())
            & 1
        )
        return VCC if parity else 0.0


class IfThenLink:
    def __init__(self):
        self.vc = 0.0

    def step(self, voltage, dt):
        self.vc += (voltage - self.vc) * (
            1
            - np.exp(
                -dt / (R_OHM * C_FARAD)
            )
        )

        return VCC if self.vc > V_BE else 0.0


class TimeCodeLatch:
    def __init__(self, value, nbits):
        self.value = value
        self.nbits = nbits

    def bit(self, slot):
        return (
            self.value
            >> (self.nbits - 1 - slot)
        ) & 1


class DFlipFlop74HC74:
    def __init__(self):
        self.q = 0

    def clock(self, d):
        self.q = int(d)

    @property
    def qbar(self):
        return 1 - self.q


def hi(voltage):
    return voltage > VCC / 2


def photon_uniform(roi_a, roi_b, frame_no):
    """
    Uniform [0,1) sample from detector shot noise: the LSBs of
    32 pixels spread over both ROIs, whitened with a murmur-style
    finalizer mixed with the frame counter (so a frozen camera
    still yields a stream).
    """
    bits = np.concatenate(
        [
            roi_a.ravel()[::997][:16],
            roi_b.ravel()[::991][:16],
        ]
    ) & 1

    word = 0
    for bit in bits:
        word = (word << 1) | int(bit)

    mask = 0xFFFFFFFF
    word ^= (frame_no * 2246822519) & mask
    word ^= word >> 16
    word = (word * 2246822519) & mask
    word ^= word >> 13
    word = (word * 3266489917) & mask
    word ^= word >> 16

    return word / 2**32


class AdaptivePhotonicDecision:
    """
    Feedback-tuned decision matrix.

    Rows (A,B) -> P(output = 1):
        (0,0) -> 0      (0,1) -> w
        (1,0) -> w      (1,1) -> 1
    w = 1 is the classical OR matrix (P(1) = 0.75 for fair inputs).
    w = 0 is AND (P(1) = 0.25). The controller integrates the error
    between the measured P(1) and the target and nudges w, like
    phase shifters adapting from detector feedback, so the output
    settles at the balanced 50/50 target.
    """

    ROWS = ((0, 0), (0, 1), (1, 0), (1, 1))

    def __init__(
        self,
        target=0.5,
        gain=0.01,
        alpha=0.05,
        adaptive=True,
    ):
        self.target = target
        self.gain = gain
        self.alpha = alpha
        self.adaptive = adaptive

        self.w = 1.0            # start as the classical OR matrix
        self.p_or = 0.75        # measured P(OR = 1)
        self.p_dec = 0.75       # measured P(decision = 1)
        self.last_bit = 0
        self.n = 0
        self.ones = 0
        self.or_ones = 0
        self.row_counts = {row: 0 for row in self.ROWS}

    def step(self, a, b, u):
        a = int(a)
        b = int(b)
        self.row_counts[(a, b)] += 1

        or_bit = a | b
        self.p_or += self.alpha * (or_bit - self.p_or)
        self.or_ones += or_bit

        if a and b:
            p_one = 1.0
        elif a or b:
            p_one = self.w
        else:
            p_one = 0.0

        bit = 1 if u < p_one else 0
        self.p_dec += self.alpha * (bit - self.p_dec)

        if self.adaptive:
            self.w = min(
                1.0,
                max(
                    0.0,
                    self.w
                    + self.gain * (self.target - self.p_dec),
                ),
            )

        self.last_bit = bit
        self.ones += bit
        self.n += 1
        return bit

    def tiebreak(self):
        return self.last_bit

    def status(self):
        return (
            f"P(OR)={self.p_or:.2f} "
            f"P(DEC)={self.p_dec:.2f} "
            f"w={self.w:.2f}"
        )

    def summary(self):
        if self.n == 0:
            return "photonic: no samples"

        total = self.n
        rows = " ".join(
            f"{a}{b}:{100 * c / total:.0f}%"
            for (a, b), c in self.row_counts.items()
        )
        return (
            f"photonic summary: n={total} "
            f"classical OR P(1)={self.or_ones / total:.3f} | "
            f"adaptive P(1)={self.ones / total:.3f} "
            f"(final w={self.w:.3f}) | rows {rows}"
        )

def live_btc_price():
    url = (
        "https://api.binance.com/api/v3/ticker/price"
        "?symbol=BTCUSDT"
    )

    with urllib.request.urlopen(
        url,
        timeout=5,
    ) as response:
        data = json.load(response)

    return float(data["price"])


def print_key_line(key, note):
    print(
        f"KEY INJECTED:     SHIFT+{key.upper()} "
        f"({'UP' if key == 'w' else 'DOWN'}, {note})"
    )


def compare_btc_prices(
    oracle_price,
    key_injector=None,
    fetch_delay=0.0,
):
    """
    Called immediately after the oracle finishes.

    Order: inject the key right away, then wait fetch_delay
    seconds, then fetch a second live BTC price and compare it
    with the price the oracle just used.

    Nothing is saved or updated.

    Key injection: only the delta between the previous oracle
    price and this oracle price decides Shift+W / Shift+S.
    If they are identical, the photonic bit decides.
    """
    oracle_delta = None

    if key_injector is not None:
        oracle_delta = key_injector.on_oracle_price(
            oracle_price
        )

        if key_injector.sent_key:
            print_key_line(
                key_injector.sent_key,
                key_injector.sent_note,
            )

    if fetch_delay > 0:
        print(
            f"Waiting {fetch_delay:g}s before fetching "
            f"fresh BTC price..."
        )
        time.sleep(fetch_delay)

    fresh_btc_price = live_btc_price()

    difference = (
        fresh_btc_price - oracle_price
    )

    if oracle_price != 0:
        percent_difference = (
            difference / oracle_price * 100
        )
    else:
        percent_difference = 0.0

    if difference == 0:
        result = "IDENTICAL"
    elif difference > 0:
        result = "FRESH PRICE HIGHER"
    else:
        result = "FRESH PRICE LOWER"

    print(
        "BTC PRICE COMPARISON AFTER ORACLE DONE:"
    )
    print(
        f"Oracle BTC price: "
        f"{oracle_price:.8f}"
    )
    print(
        f"Fresh BTC price:  "
        f"{fresh_btc_price:.8f}"
    )
    print(
        f"Difference:       "
        f"{difference:.8f}"
    )
    print(
        f"Percent change:   "
        f"{percent_difference:.8f}%"
    )
    print(f"Result:           {result}")



# Hardware scancodes (PS/2 set 1 for Windows, evdev codes for Linux).
SCANCODES_WIN = {"w": 0x11, "s": 0x1F, "shift": 0x2A}
EVDEV_CODES = {"w": 17, "s": 31, "shift": 42}


class ScancodeBackend:
    """
    Sends raw scancodes so games that ignore virtual-key
    events still see the keypress.

    Windows: SendInput with KEYEVENTF_SCANCODE (no extra install).
    Linux:   uinput via evdev (pip install evdev; needs write
             access to /dev/uinput, e.g. run with sudo).
    Other:   falls back to pynput virtual keys.
    """

    def __init__(self):
        import sys

        self.mode = None

        if sys.platform.startswith("win"):
            self._init_windows()
            self.mode = "windows"

        elif sys.platform.startswith("linux"):
            try:
                self._init_linux()
                self.mode = "linux"
            except Exception as error:
                print(
                    f"[keys] uinput unavailable ({error}); "
                    f"falling back to pynput (no scancodes)"
                )

        if self.mode is None:
            from pynput.keyboard import Controller

            self._pynput = Controller()
            self.mode = "pynput"

        print(f"[keys] backend: {self.mode}")

    def _init_windows(self):
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        ulong_p = ctypes.POINTER(ctypes.c_ulong)

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_p),
            ]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_p),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class _INPUTUNION(ctypes.Union):
            _fields_ = [
                ("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("u", _INPUTUNION),
            ]

        self._KEYBDINPUT = KEYBDINPUT
        self._INPUTUNION = _INPUTUNION
        self._INPUT = INPUT

    def _win_send(self, scan, up):
        ctypes = self._ctypes
        KEYEVENTF_SCANCODE = 0x0008
        KEYEVENTF_KEYUP = 0x0002

        flags = KEYEVENTF_SCANCODE | (
            KEYEVENTF_KEYUP if up else 0
        )
        extra = ctypes.c_ulong(0)
        ki = self._KEYBDINPUT(
            0,  # wVk must be 0 when using scancodes
            scan,
            flags,
            0,
            ctypes.pointer(extra),
        )
        event = self._INPUT(
            type=1,  # INPUT_KEYBOARD
            u=self._INPUTUNION(ki=ki),
        )
        ctypes.windll.user32.SendInput(
            1,
            ctypes.byref(event),
            ctypes.sizeof(event),
        )

    def _init_linux(self):
        from evdev import UInput, ecodes

        self._ecodes = ecodes
        self._uinput = UInput(
            {ecodes.EV_KEY: list(EVDEV_CODES.values())},
            name="price-predictor-kbd",
        )

    @staticmethod
    def _pynput_key(key):
        if key == "shift":
            from pynput.keyboard import Key

            return Key.shift

        return key

    def press(self, key):
        if self.mode == "windows":
            self._win_send(SCANCODES_WIN[key], up=False)
        elif self.mode == "linux":
            self._uinput.write(
                self._ecodes.EV_KEY, EVDEV_CODES[key], 1
            )
            self._uinput.syn()
        else:
            self._pynput.press(self._pynput_key(key))

    def release(self, key):
        if self.mode == "windows":
            self._win_send(SCANCODES_WIN[key], up=True)
        elif self.mode == "linux":
            self._uinput.write(
                self._ecodes.EV_KEY, EVDEV_CODES[key], 0
            )
            self._uinput.syn()
        else:
            self._pynput.release(self._pynput_key(key))


class KeyInjector:
    """
    Injects Shift+W (price rose) or Shift+S (price fell) as
    hardware scancodes into whichever window has focus.
    """

    def __init__(
        self,
        enabled=True,
        hold=0.05,
        threshold=0.0,
        tiebreak=None,
    ):
        self.tiebreak = tiebreak
        self.sent_note = None
        self.enabled = enabled
        self.hold = hold
        self.threshold = threshold
        self.controller = None
        self.last_key = None
        self.prev_oracle_price = None
        self.last_delta = 0.0
        self.sent_key = None

        if enabled:
            self.controller = ScancodeBackend()

    def on_oracle_price(self, price):
        """
        Called once per oracle sample. Compares it with the
        previous oracle sample and returns the delta
        (None on the first sample).

        A non-zero delta sends its key immediately. A zero
        delta (oracle identical to last oracle) uses the
        photonic bit, if one is wired in.
        """
        previous = self.prev_oracle_price
        self.prev_oracle_price = price
        self.sent_key = None
        self.sent_note = None

        if previous is None:
            return None

        delta = price - previous
        self.last_delta = delta

        if delta != 0:
            self.sent_key = self.signal(previous, price)
            self.sent_note = f"oracle delta {delta:+.8f}"

        elif self.tiebreak is not None and self.enabled:
            key = "w" if self.tiebreak() else "s"
            self.press_combo(key)
            self.sent_key = key
            self.sent_note = (
                "oracle identical to last oracle, "
                "photonic 50/50 bit"
            )

        return delta

    def signal(self, last_price, fresh_price):
        """
        Delta between two oracle prices.
        Higher -> W, lower -> S, identical -> None.
        Returns the key sent ('w', 's') or None.
        """
        if not self.enabled:
            return None

        difference = fresh_price - last_price

        if last_price != 0:
            change = difference / last_price * 100
        else:
            change = 0.0

        if difference > 0 and change > self.threshold:
            key = "w"
        elif difference < 0 and change < -self.threshold:
            key = "s"
        else:
            return None

        self.last_key = key

        self.press_combo(key)

        return key

    def press_combo(self, key):
        # Shift + W / Shift + S
        self.controller.press("shift")
        time.sleep(0.01)
        self.controller.press(key)
        time.sleep(self.hold)
        self.controller.release(key)
        time.sleep(0.01)
        self.controller.release("shift")


class PriceOracle:
    """
    Samples BTC-USD.

    After each sample and prediction are complete,
    it calls on_oracle_done(oracle_price).
    """

    URL = (
        "https://api.binance.com/api/v3/ticker/price"
        "?symbol=BTCUSDT"
    )

    def __init__(
        self,
        simulated=False,
        poll=5.0,
        window=12,
        on_oracle_done=None,
    ):
        self.simulated = simulated
        self.poll = poll
        self.window = window
        self.on_oracle_done = on_oracle_done

        self.history = collections.deque(
            maxlen=max(4 * window, 60)
        )
        self.events = queue.Queue()

        self.price = None
        self.prediction = None
        self.pending = None

        self.hits = 0
        self.total = 0
        self.absolute_error = 0.0

        self._sim_price = 60000.0
        self._drift = 0.0
        self._rng = np.random.default_rng(7)
        self._stop = threading.Event()

        threading.Thread(
            target=self._run,
            daemon=True,
        ).start()

    def _fetch(self):
        if self.simulated:
            if self._rng.random() < 0.08:
                self._drift = (
                    self._rng.choice(
                        [-1, 0, 1]
                    )
                    * 0.0012
                )

            self._sim_price *= (
                1
                + self._drift
                + self._rng.normal(
                    0,
                    0.0003,
                )
            )

            return self._sim_price

        with urllib.request.urlopen(
            self.URL,
            timeout=4,
        ) as response:
            data = json.load(response)

        return float(data["price"])


    def _predict_next(self):
        history = list(self.history)[
            -self.window:
        ]

        if len(history) < self.window:
            return None

        slope, intercept = np.polyfit(
            np.arange(len(history)),
            history,
            1,
        )

        return float(
            slope * len(history)
            + intercept
        )

    def _on_sample(self, price):
        self.history.append(price)
        self.price = price

        if self.pending is not None:
            prediction, base_price = (
                self.pending
            )

            direction_hit = (
                prediction - base_price
            ) * (
                price - base_price
            ) > 0

            self.total += 1
            self.hits += int(direction_hit)
            self.absolute_error += abs(
                price - prediction
            )

            self.events.put(
                f"ORACLE DONE: predicted "
                f"${prediction:,.2f}, "
                f"actual ${price:,.2f}, "
                f"direction "
                f"{'HIT' if direction_hit else 'MISS'}, "
                f"error "
                f"${abs(price - prediction):,.2f}, "
                f"[{self.rate()}]"
            )

        self.prediction = (
            self._predict_next()
        )

        if self.prediction is not None:
            self.pending = (
                self.prediction,
                price,
            )

            self.events.put(
                f"price ${price:,.2f} -> "
                f"predicted next "
                f"${self.prediction:,.2f}"
            )

        else:
            self.pending = None

            self.events.put(
                f"price ${price:,.2f} "
                f"(warming up "
                f"{len(self.history)}/"
                f"{self.window})"
            )

        # Oracle is now done.
        if self.on_oracle_done is not None:
            self.on_oracle_done(self.price)

    def _run(self):
        while not self._stop.is_set():
            try:
                price = self._fetch()
                self._on_sample(price)

            except Exception as error:
                self.events.put(
                    f"[oracle] error: "
                    f"{str(error)[:60]}"
                )

            self._stop.wait(self.poll)

    def rate(self):
        if self.total == 0:
            return "n/a"

        accuracy = (
            100 * self.hits / self.total
        )
        mean_error = (
            self.absolute_error / self.total
        )

        return (
            f"{self.hits}/{self.total} "
            f"({accuracy:.0f}%), "
            f"mean err "
            f"${mean_error:,.2f}"
        )

    def stop(self):
        self._stop.set()


def draw(
    frame,
    roi_b,
    roi_a,
    text_lines,
    traces,
):
    import cv2

    height, width = frame.shape[:2]
    output = frame.copy()

    cv2.rectangle(
        output,
        (roi_b[0], roi_b[1]),
        (roi_b[2], roi_b[3]),
        (60, 60, 255),
        2,
    )
    cv2.putText(
        output,
        "IR det B",
        (roi_b[0] + 4, roi_b[1] + 18),
        0,
        0.5,
        (60, 60, 255),
        1,
    )

    cv2.rectangle(
        output,
        (roi_a[0], roi_a[1]),
        (roi_a[2], roi_a[3]),
        (60, 200, 60),
        2,
    )
    cv2.putText(
        output,
        "IR det A",
        (roi_a[0] + 4, roi_a[1] + 18),
        0,
        0.5,
        (60, 200, 60),
        1,
    )

    y = height - 8 - 16 * len(text_lines)

    for line in text_lines:
        cv2.putText(
            output,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 16

    names = [
        "A",
        "B_out",
        "OR",
        "AND",
        "Y",
        "D",
        "SELECT",
        "DEC",
    ]

    panel_width = max(width, 480)
    row_height = 22

    panel = np.full(
        (
            row_height * len(names) + 8,
            panel_width,
            3,
        ),
        25,
        np.uint8,
    )

    for index, name in enumerate(names):
        y0 = 4 + index * row_height

        cv2.putText(
            panel,
            name,
            (4, y0 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        data = list(traces[name])[
            -(panel_width - 70):
        ]

        points = [
            (
                60 + position,
                y0
                + (4 if value else 17),
            )
            for position, value in enumerate(
                data
            )
        ]

        for p1, p2 in zip(
            points[:-1],
            points[1:],
        ):
            cv2.line(
                panel,
                p1,
                p2,
                (0, 220, 255),
                1,
            )

    if panel.shape[1] != output.shape[1]:
        padding = np.zeros(
            (
                output.shape[0],
                panel.shape[1]
                - output.shape[1],
                3,
            ),
            np.uint8,
        )
        output = np.hstack(
            [output, padding]
        )

    return np.vstack([output, panel])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--secret",
        default="A5C3",
    )
    parser.add_argument(
        "--x",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--leak",
        action="store_true", default=True,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--price",
        action="store_true",
    )
    parser.add_argument(
        "--price-sim",
        action="store_true",
    )
    parser.add_argument(
        "--price-poll",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--price-window",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--classical",
        action="store_true",
        help="freeze the decision matrix at plain OR (biased 75%%)",
    )
    parser.add_argument(
        "--fetch-delay",
        type=float,
        default=0.2,
        help="extra seconds to wait after the key is injected "
        "before fetching the fresh BTC price",
    )
    parser.add_argument(
        "--no-tiebreak",
        action="store_true",
        help="send no key when oracle price equals last oracle price",
    )
    parser.add_argument(
        "--no-keys",
        action="store_true",
        help="disable W/S key injection",
    )
    parser.add_argument(
        "--key-hold",
        type=float,
        default=0.05,
        help="seconds each key is held down",
    )
    parser.add_argument(
        "--key-threshold",
        type=float,
        default=0.01,
        help="min predicted move in percent before a key fires",
    )

    args = parser.parse_args()

    if not args.price:
        raise SystemExit(
            "Run with --price to compare "
            "BTC prices after the oracle finishes."
        )

    nbits = args.bits - args.bits % 4
    hex_width = nbits // 4

    isolated_number = (
        int(args.secret, 16)
        & ((1 << nbits) - 1)
    )

    camera = Camera(
        args.cam,
        args.synthetic,
    )
    driver_a = DriverA()
    driver_b = DriverB(args.x)
    if_then_link = IfThenLink()

    latch = TimeCodeLatch(
        isolated_number,
        nbits,
    )
    flip_flop = DFlipFlop74HC74()

    traces = collections.defaultdict(
        lambda: collections.deque(
            maxlen=600
        )
    )

    slot = 0
    select_bits = []
    completed_words = 0

    adaptive = AdaptivePhotonicDecision(
        adaptive=not args.classical,
    )
    frame_no = 0
    decision = 0

    key_injector = KeyInjector(
        enabled=not args.no_keys,
        hold=args.key_hold,
        threshold=args.key_threshold,
        tiebreak=(
            None
            if args.no_tiebreak
            else adaptive.tiebreak
        ),
    )

    oracle = PriceOracle(
        simulated=(
            args.price_sim
            or args.synthetic
        ),
        poll=args.price_poll,
        window=args.price_window,
        on_oracle_done=lambda price: compare_btc_prices(
            price,
            key_injector,
            args.fetch_delay,
        ),
    )

    last_time = time.time()
    next_clock = last_time + CLK_PERIOD
    previous_intensity_b = None

    print(
        f"Isolated number = "
        f"{isolated_number:0{hex_width}X} "
        f"(fixed; not saved or updated)"
    )

    try:
        while True:
            gray, bgr = camera.read()

            now = time.time()
            dt = now - last_time
            last_time = now

            height, width = gray.shape

            roi_b_box = (
                0,
                0,
                width // 2 - 4,
                height - 1,
            )
            roi_a_box = (
                width // 2 + 4,
                0,
                width - 1,
                height - 1,
            )

            roi_b = gray[
                roi_b_box[1]:roi_b_box[3],
                roi_b_box[0]:roi_b_box[2],
            ]
            roi_a = gray[
                roi_a_box[1]:roi_a_box[3],
                roi_a_box[0]:roi_a_box[2],
            ]

            intensity_b = (
                float(roi_b.mean()) / 255.0
            )

            if previous_intensity_b is None:
                delta_intensity = 0.0
            else:
                delta_intensity = (
                    intensity_b
                    - previous_intensity_b
                )

            previous_intensity_b = (
                intensity_b
            )

            voltage_a = driver_a.step(
                roi_a
            )
            voltage_b = driver_b.step(
                intensity_b
            )
            voltage_b_out = (
                if_then_link.step(
                    voltage_b,
                    dt,
                )
            )

            a = hi(voltage_a)
            b = hi(voltage_b_out)
            or_gate = a or b
            and_gate = a and b

            frame_no += 1
            decision = adaptive.step(
                a,
                b,
                photon_uniform(roi_a, roi_b, frame_no),
            )

            y = bool(latch.bit(slot))
            d = y or (
                and_gate and args.leak
            )

            for name, value in zip(
                [
                    "A",
                    "B_out",
                    "OR",
                    "AND",
                    "Y",
                    "D",
                    "SELECT",
                    "DEC",
                ],
                [
                    a,
                    b,
                    or_gate,
                    and_gate,
                    y,
                    d,
                    flip_flop.q,
                    decision,
                ],
            ):
                traces[name].append(
                    int(value)
                )

            if now >= next_clock:
                next_clock += CLK_PERIOD
                flip_flop.clock(d)
                select_bits.append(
                    flip_flop.q
                )
                slot += 1

                if slot == nbits:
                    reconstructed = int(
                        "".join(
                            map(
                                str,
                                select_bits,
                            )
                        ),
                        2,
                    )

                    if (
                        reconstructed
                        == isolated_number
                    ):
                        result = (
                            "RECONSTRUCTED"
                        )
                    else:
                        result = "MISMATCH"

                    print(
                        f"SELECT = "
                        f"{reconstructed:0{hex_width}X} "
                        f"vs isolated "
                        f"{isolated_number:0{hex_width}X} "
                        f"-> {result}"
                    )

                    print(f"PHOTONIC: {adaptive.status()}")

                    completed_words += 1
                    slot = 0
                    select_bits = []

                    if (
                        args.runs
                        and completed_words
                        >= args.runs
                    ):
                        break

            while not oracle.events.empty():
                print(
                    oracle.events.get()
                )

            if (
                args.headless
                and args.runs
                and oracle.total
                >= args.runs
            ):
                break

            if args.headless:
                time.sleep(0.01)
                continue

            import cv2

            selected = "".join(
                map(str, select_bits)
            )

            if selected:
                selected_hex = (
                    f"{int(selected, 2):X}"
                )
            else:
                selected_hex = "-"

            lines = [
                f"I_B={intensity_b:.2f} "
                f"dI={delta_intensity:+.3f} "
                f"Vc={if_then_link.vc:.2f}V "
                f"thr="
                f"{driver_b.threshold:.2f}",
                f"A={int(a)} "
                f"Bout={int(b)} "
                f"OR={int(or_gate)} "
                f"AND={int(and_gate)} "
                f"Y={int(y)} D={int(d)} "
                f"Q={flip_flop.q}",
                f"photonic {adaptive.status()} "
                f"DEC={decision}",
                f"slot {slot}/{nbits} "
                f"SELECT bits={selected} "
                f"hex={selected_hex}",
                f"isolated number = "
                f"{isolated_number:0{hex_width}X} "
                f"(fixed)",
                (
                    f"BTC oracle price: "
                    f"${oracle.price:,.2f} "
                    f"hot verify: "
                    f"{oracle.rate()}"
                    if oracle.price is not None
                    else
                    "BTC: waiting for oracle"
                ),
            ]

            cv2.imshow(
                "optoelectronic sim",
                draw(
                    bgr,
                    roi_b_box,
                    roi_a_box,
                    lines,
                    traces,
                ),
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                slot = 0
                select_bits = []

    finally:
        print(adaptive.summary())
        oracle.stop()
        camera.release()

        if not args.headless:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
