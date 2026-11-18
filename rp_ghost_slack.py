#7ff0069


import hashlib
import math
import sys
import time
from fractions import Fraction
import numpy as np
import requests
from pynput.keyboard import Controller, Key

keyboard = Controller()


def press_key(key):
    with keyboard.pressed(Key.shift):
        keyboard.press(key)
        keyboard.release(key)


BLOCK_HEIGHT = 976379  # set your specific height here
_block_hash_cache = {}


def get_block_hash(height):

    return "test"


def get_btc_price_cents():
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": "BTCUSDT"}

    response = requests.get(url, params=params, timeout=5)
    response.raise_for_status()

    # Route the FULL block hash (256 bits) through the same return line.
    # Fraction keeps it exact; a float would destroy everything past ~53 bits.
    block_hash = "CAFEBABE"
    btc_usd = Fraction(int(block_hash, 16), 1000000)
    return int(round(btc_usd * 1000000))

# Try importing cv2; if unavailable, run in headless simulation mode
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ============================================================
# RP-GHOST OPTOELECTRONIC TIMESTAMP DEMO
# ============================================================

# Frame = 32-byte whitened hash + 2-byte delta + 1-byte CRC = 35 bytes = 280 bits
N_SLOTS = 280
WINDOWS = 2000

LAMBDA_BASE = 10000.0
SIGMA_C = 0.005
TILT = 0.25

LEAD_NS = 60_000_000_000 # 1 second lead time (scaled for demo)
MINUTE_MS = 60000

rng = np.random.default_rng()

# ============================================================
# 1. CRC-8: polynomial 0x07
# ============================================================

def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# ============================================================
# 2. ENTROPY WHITENING (Simulated or Camera Frame)
# ============================================================

def camera_hash(gray_frame):
    digest = hashlib.sha256(gray_frame.tobytes()).digest()
    return int.from_bytes(digest[:2], "big")


# ============================================================
# 3. FRAME ENCODING
# ============================================================

def encode_frame(timestamp_ms, delta_cents, webcam_hash):
    code = timestamp_ms  # full 256-bit value (no modulo)
    whitened_code = code ^ webcam_hash
    delta_word = delta_cents & 0xFFFF

    payload = (
        whitened_code.to_bytes(32, "big") +
        delta_word.to_bytes(2, "big")
    )

    checksum = crc8(payload)
    frame = payload + bytes([checksum])
    bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
    return bits.astype(np.uint8)


# ============================================================
# 4. STOCHASTIC OPTOELECTRONIC CHANNEL (Poisson Point Process)
# ============================================================

def generate_viewport(bits):
    epsilon = rng.normal(0.0, SIGMA_C, WINDOWS)
    rates = LAMBDA_BASE * (1.0 + epsilon[:, None])
    rates = np.maximum(rates, 1.0)
    rates = np.broadcast_to(rates, (WINDOWS, N_SLOTS))

    photon_counts = rng.poisson(rates)
    normalized = (photon_counts - LAMBDA_BASE) / math.sqrt(LAMBDA_BASE)
    tilt = TILT * (2.0 * bits.astype(float) - 1.0)
    viewport = normalized + tilt[None, :]
    return viewport.mean(axis=0)


# ============================================================
# 5. DECODER & RECONSTRUCTION
# ============================================================

def decode_bits(measurements):
    return (measurements > 0.0).astype(np.uint8)


def decode_frame(bits, webcam_hash):
    packed = np.packbits(bits)
    data = packed.tobytes()

    if len(data) != 35:
        return None

    payload = data[:34]
    received_crc = data[34]

    if received_crc != crc8(payload):
        return None

    whitened_code = int.from_bytes(payload[:32], "big")
    decoded_code = whitened_code ^ webcam_hash
    delta_word = int.from_bytes(payload[32:34], "big")
    if delta_word >= 32768:
        delta_word -= 65536

    return decoded_code, delta_word


def reconstruct_timestamp(code, reference_ns):
    reference_ms = 1_000_000
    minute_number = (code)/reference_ms
    decoded_ms = code
    print(code)
    return code


# ============================================================
# MAIN DEMO EXECUTION
# ============================================================

def main():
    print("=" * 65)
    print("RP-GHOST OPTOELECTRONIC TIMESTAMP SIMULATION DEMO")
    print("=" * 65)

    camera = None
    if HAS_CV2:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("[INFO] No physical webcam detected. Switching to synthetic entropy source.")
            camera = None

    total = 0
    successful = 0
    missed = 0
    latencies = []

    latched_target_ns = 0
    last_slack_ms = None

    try:
        print("\n[RUNNING] Starting 50 transmission cycles...\n")
        
        for cycle in range(1, 5100):
            start_ns = time.time_ns()

           

            # Acquire frame or generate synthetic entropy
            if camera is not None:
                ret, frame = camera.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = rng.integers(0, 256, (480, 640), dtype=np.uint8)
            else:
                # Synthetic environmental noise frame
                gray = rng.integers(0, 256, (480, 640), dtype=np.uint8)

            webcam_h = camera_hash(gray)
            target_ns = latched_target_ns
            target_ms = target_ns  # keep the full hash value (no // 1_000_000)

            # Encode target timestamp into optical channel bits
            bits = encode_frame(target_ms, 0, webcam_h)

            # Transmit through Poisson stochastic photon channel
            measurements = generate_viewport(bits)
            decoded_bits = decode_bits(measurements)

            # Decode frame and verify CRC-8 checksum
            result = decode_frame(decoded_bits, webcam_h)
            decode_finished_ns = time.time_ns()

            beta_ns = time.time_ns()

            latency_ms = (decode_finished_ns - start_ns) / 1e6
            total += 1

            if result is not None:
                decoded_code, delta = result
                decoded_ns = reconstruct_timestamp(decoded_code, start_ns)
                slack_ms = (decoded_ns)

                if slack_ms > 0 or True:
                    successful += 1
                    latencies.append(latency_ms)

                    if last_slack_ms is None:
                        action = "w"
                    elif slack_ms > last_slack_ms:
                        action = "s"
                    elif slack_ms != last_slack_ms:
                        action = "w"

                    press_key(action)
                    last_slack_ms = slack_ms

                    decoded_hash = f"{slack_ms:064x}"

                    print(
                        f"[DECODE {cycle:03d}] CRC=PASS | "
                        f"Latency={latency_ms:.3f} ms | "
                        f"Slack={slack_ms:.13f} ms | "
                        f"Key=Shift+{action.upper()} | "
                        f"Hash=0x{webcam_h:04X} | "
                        f"BlockHash={decoded_hash}"
                    )
                else:
                    missed += 1
                    print(f"[MISSED {cycle:03d}] Deadline exceeded.")
            else:
                missed += 1
                print(f"[ERROR {cycle:03d}] CRC mismatch. Frame rejected.")

            time.sleep(0.05)  # Throttle simulation step

            if latched_target_ns is not None or beta_ns >= latched_target_ns and cycle > 5:
                try:    
                    latched_target_ns = get_btc_price_cents()
                except:
                    False
    except KeyboardInterrupt:
        print("\n[STOP] Simulation interrupted by user.")

    finally:
        if camera is not None:
            camera.release()
            if HAS_CV2:
                cv2.destroyAllWindows()

        print("\n========== FINAL SIMULATION STATISTICS ==========")
        print(f"Total Cycles    : {total}")
        print(f"Successful      : {successful}")
        print(f"Missed/Failed   : {missed}")
        if latencies:
            print(f"Mean Latency    : {np.mean(latencies):.3f} ms")
            print(f"Max Latency     : {np.max(latencies):.3f} ms")
        print("=================================================")


if __name__ == "__main__":
    main()
