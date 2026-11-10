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

Key injection (requires: pip install pynput):
Fresh price above the last downloaded price -> taps W
Fresh price below the last downloaded price -> taps S
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


def compare_btc_prices(oracle_price, key_injector=None):
    """
    Called immediately after the oracle finishes.

    Fetches a second live BTC price and compares it
    with the price the oracle just used.

    Nothing is saved or updated.
    """
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

    # Key injection from the raw delta vs the last downloaded price.
    if key_injector is not None:
        key = key_injector.signal(
            oracle_price,
            fresh_btc_price,
        )

        if key:
            print(
                f"KEY INJECTED:     {key.upper()} "
                f"({'UP' if key == 'w' else 'DOWN'}"
                f"{', repeated: identical' if difference == 0 else ''})"
            )


class KeyInjector:
    """
    Injects W (price rose) or S (price fell) into
    whichever window currently has focus.
    """

    def __init__(self, enabled=True, hold=0.05, threshold=0.0):
        self.enabled = enabled
        self.hold = hold
        self.threshold = threshold
        self.controller = None
        self.last_key = None

        if enabled:
            from pynput.keyboard import Controller

            self.controller = Controller()

    def signal(self, last_price, fresh_price):
        """
        Delta from the last price downloaded.
        Fresh above last -> W, fresh below last -> S.
        Identical -> repeats the key of the last real move.
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
        elif difference == 0 and self.last_key:
            # Identical price: keep the direction of the last
            # real move instead of going silent.
            key = self.last_key
        else:
            return None

        if difference != 0:
            self.last_key = key

        self.controller.press(key)
        time.sleep(self.hold)
        self.controller.release(key)

        return key


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
        action="store_true",
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
        default=5.0,
    )
    parser.add_argument(
        "--price-window",
        type=int,
        default=3,
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
        default=0.0,
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

    key_injector = KeyInjector(
        enabled=not args.no_keys,
        hold=args.key_hold,
        threshold=args.key_threshold,
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
                ],
                [
                    a,
                    b,
                    or_gate,
                    and_gate,
                    y,
                    d,
                    flip_flop.q,
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
        oracle.stop()
        camera.release()

        if not args.headless:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
