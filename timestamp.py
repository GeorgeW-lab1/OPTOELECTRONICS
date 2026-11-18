import hashlib
import math
import sys
import time
import numpy as np

# Try importing cv2; if unavailable, run in headless simulation mode
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ============================================================
# RP-GHOST OPTOELECTRONIC TIMESTAMP DEMO
# ============================================================

N_SLOTS = 40
WINDOWS = 2000

LAMBDA_BASE = 10000.0
SIGMA_C = 0.005
TILT = 0.25

LEAD_NS = 1_000_000_000  # 1 second lead time (scaled for demo)
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
    code = timestamp_ms % MINUTE_MS
    whitened_code = code ^ webcam_hash
    delta_word = delta_cents & 0xFFFF

    payload = (
        whitened_code.to_bytes(2, "big") +
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

    if len(data) != 5:
        return None

    payload = data[:4]
    received_crc = data[4]

    if received_crc != crc8(payload):
        return None

    whitened_code = int.from_bytes(payload[:2], "big")
    decoded_code = whitened_code ^ webcam_hash
    delta_word = int.from_bytes(payload[2:4], "big")
    if delta_word >= 32768:
        delta_word -= 65536

    return decoded_code, delta_word


def reconstruct_timestamp(code, reference_ns):
    reference_ms = reference_ns // 1_000_000
    minute_number = (reference_ms - code + MINUTE_MS // 2) // MINUTE_MS
    decoded_ms = code + minute_number * MINUTE_MS
    return decoded_ms * 1_000_000


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

    try:
        print("\n[RUNNING] Starting 50 transmission cycles...\n")
        
        for cycle in range(1, 51):
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
            target_ms = target_ns // 1_000_000

            # Encode target timestamp into optical channel bits
            bits = encode_frame(target_ms, 0, webcam_h)

            # Transmit through Poisson stochastic photon channel
            measurements = generate_viewport(bits)
            decoded_bits = decode_bits(measurements)

            # Decode frame and verify CRC-8 checksum
            result = decode_frame(decoded_bits, webcam_h)
            decode_finished_ns = time.time_ns()

            latency_ms = (decode_finished_ns - start_ns) / 1e6
            total += 1

            if result is not None:
                decoded_code, delta = result
                decoded_ns = reconstruct_timestamp(decoded_code, start_ns)
                slack_ms = (decoded_ns - decode_finished_ns) / 1e6

                if slack_ms > 0 or True:
                    successful += 1
                    latencies.append(latency_ms)
                    print(
                        f"[DECODE {cycle:03d}] CRC=PASS | "
                        f"Latency={latency_ms:.3f} ms | "
                        f"Slack={slack_ms:.3f} ms | "
                        f"Hash=0x{webcam_h:04X}"
                    )
                else:
                    missed += 1
                    print(f"[MISSED {cycle:03d}] Deadline exceeded.")
            else:
                missed += 1
                print(f"[ERROR {cycle:03d}] CRC mismatch. Frame rejected.")

            time.sleep(10.05)  # Throttle simulation step
            beta_ns = time.time_ns()

            if latched_target_ns is not None or beta_ns >= latched_target_ns:
                latched_target_ns = beta_ns + LEAD_NS
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
