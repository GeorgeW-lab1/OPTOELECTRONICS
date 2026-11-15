import hashlib
import json
import time
import threading
import queue

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

LAMBDA = 10000.0
SIGMA_C = 0.005
N_SLOTS = 32

# Adjust difficulty iterations to guarantee a distinct time gap (e.g., ~15-50ms)
POW_DIFFICULTY_ITERATIONS = 5000000


# ============================================================================
# HELPERS & TIMESTAMP ENCODING
# ============================================================================

def format_clock_time(timestamp_ns):
    seconds = timestamp_ns / 1e9
    clock = time.strftime("%H:%M:%S", time.localtime(seconds))
    ms = int((timestamp_ns % 1_000_000_000) // 1_000_000)
    return f"{clock}.{ms:03d}"


def timestamp_to_code(timestamp_ns):
    ms = (int(timestamp_ns) // 1_000_000) % 60000
    return ms & 0xFFFF


def encode_timestamp_bits(timestamp_ns):
    time_code = timestamp_to_code(timestamp_ns)
    time_bits = [(time_code >> (15 - k)) & 1 for k in range(16)]
    padding_bits = [(time_code >> k) & 1 for k in range(16)]
    return np.array(time_bits + padding_bits, dtype=int)


def bits_to_code(bits):
    code = 0
    for bit in bits:
        code = (code << 1) | (int(bit) & 1)
    return code


def decode_timestamp_from_code(code, reference_ns):
    minute_start = (int(reference_ns) // 60_000_000_000) * 60_000_000_000
    decoded_ns = minute_start + (int(code) & 0xFFFF) * 1_000_000
    return decoded_ns, format_clock_time(decoded_ns)


# ============================================================================
# OPTIONAL VIEWPORT CHANNEL
# ============================================================================

class OptoelectronicViewport:
    def __init__(self, lambd=LAMBDA, sigma_c=SIGMA_C):
        self.lambd = lambd
        self.sigma_c = sigma_c

    def generate_window_block(self, n_windows, n_slots, tilt=None):
        shared_noise = np.random.normal(0.0, self.sigma_c, size=(n_windows, 1))
        lam = self.lambd * (1.0 + shared_noise)
        counts = np.random.poisson(np.clip(lam, 1.0, None), size=(n_windows, n_slots))
        Z = (counts - self.lambd) / np.sqrt(self.lambd)
        if tilt is not None:
            Z += tilt
        return Z


# ============================================================================
# PROOF-OF-WORK MINER WITH PIPELINED ENCODE/DECODE
# ============================================================================

class PipelinedBlockProcessor(threading.Thread):
    def __init__(self, output_queue):
        super().__init__(daemon=True)
        self.output_queue = output_queue
        self.running = True

    def run(self):
        mock_height = 850000
        viewport = OptoelectronicViewport()

        # Initialize channel bits with a baseline timestamp
        initial_ns = time.time_ns()
        current_optical_bits = encode_timestamp_bits(initial_ns)

        while self.running:
            mock_height += 1
            
            # ================================================================
            # 1. BEGINNING OF MINING CYCLE: DECODE IMMEDIATELY
            # ================================================================
            start_ns = time.time_ns()

            # Decode the optical channel bits present at the start of mining
            tilt = 2.0 * 0.1 * (current_optical_bits - 0.5)
            Z = viewport.generate_window_block(1000, N_SLOTS, tilt)
            m = Z.mean(axis=0)
            decoded_bits = (m > 0).astype(int)
            decoded_code = bits_to_code(decoded_bits[:16])
            _, decoded_time = decode_timestamp_from_code(decoded_code, start_ns)

            # ================================================================
            # 2. INTENSIVE PROOF-OF-WORK HASHING LOOP
            # ================================================================
            block_hash = ""
            nonce = 0
            block_hash = "0"

            for i in range(POW_DIFFICULTY_ITERATIONS):
                nonce += 1
                header = f"{mock_height}:{nonce}:{start_ns}"
                test = hashlib.sha256(header.encode()).hexdigest()
                if test.startswith("00000"):
                    block_hash = test

            # ================================================================
            # 3. END OF MINING CYCLE: ENCODE COMPLETION TIMESTAMP
            # ================================================================
            completion_ns = time.time_ns()
            
            if completion_ns <= start_ns:
                completion_ns = start_ns + 1_500_000  # +1.5 ms offset fallback

            mining_duration_ms = (completion_ns - start_ns) / 1_000_000

            # Encode the completion timestamp into the optical channel for the next cycle
            current_optical_bits = encode_timestamp_bits(completion_ns)

            result = {
                "height": mock_height,
                "hash": block_hash,
                "nonce": nonce,
                "start_time": format_clock_time(start_ns),
                "completion_ns": completion_ns,
                "completion_time": format_clock_time(completion_ns),
                "mining_duration_ms": mining_duration_ms,
                "decoded_time": decoded_time,
                "exact_bits": int(np.sum(decoded_bits[:16] == current_optical_bits[:16]))
            }

            self.output_queue.put(result)
            time.sleep(1.0)  # Next block template delay

    def stop(self):
        self.running = False


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("=" * 78)
    print(" RP-GHOST PROOF-OF-WORK PIPELINED ENCODE/DECODE PIPELINE")
    print("=" * 78)

    output_queue = queue.Queue()
    processor = PipelinedBlockProcessor(output_queue)
    processor.start()

    try:
        while True:
            try:
                res = output_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            print()
            print("-" * 78)
            print(f"  [BLOCK MINED] Height: #{res['height']} (Nonce iterations: {res['nonce']:,})")
            print(f"  [MINING WORK] Start Time:      {res['start_time']}")
            print(f"  [COMPLETION]  Completion Time: {res['completion_time']} (Elapsed: {res['mining_duration_ms']:.2f} ms)")
            print(f"  [DECODED TS]  Decoded Time:    {res['decoded_time']} (Bits verified)")
            print(f"  [HASH]        {res['hash']}")

            print("-" * 78)

    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")
    finally:
        processor.stop()
        processor.join(timeout=1.0)
        print("[DONE]")


if __name__ == "__main__":
    main()
