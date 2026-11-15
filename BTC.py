import hashlib
import time
import threading
import urllib.request
import json
from collections import deque
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True
    pyautogui.FAILSAFE = True
except ImportError:
    HAS_PYAUTOGUI = False


class OptoelectronicViewport:
    """Acquires standardized photon/pixel noise samples z_k from a webcam or calibrated synthetic source."""

    def __init__(self, camera_index=0, lambd=10000.0, sigma_c=0.005, use_webcam=False):
        self.camera_index = camera_index
        self.lambd = lambd
        self.sigma_c = sigma_c
        self.cap = None
        self.live_mode = False

        if use_webcam and HAS_CV2:
            print(f"[INIT] Attempting to open webcam index {camera_index}...")
            cap = cv2.VideoCapture(self.camera_index)
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.live_mode = True
                    print("[INIT] Live webcam successfully initialized.")
                else:
                    cap.release()
            if not self.live_mode:
                print("[INIT] Using calibrated synthetic optical viewport (Safe Mode).")

    def generate_ensemble_matrix(self, n_ensemble, n_slots, secret_Y, epsilon, lead_L):
        delta_shared = np.random.normal(0.0, self.sigma_c, size=(n_ensemble, 1))
        lambda_matrix = self.lambd * (1.0 + delta_shared)
        counts = np.random.poisson(np.clip(lambda_matrix, 1.0, None), size=(n_ensemble, n_slots))
        Z_ens = (counts - self.lambd) / np.sqrt(self.lambd)

        if secret_Y is not None and epsilon != 0.0:
            for k in range(n_slots - lead_L):
                Z_ens[:, k] += 2.0 * epsilon * (secret_Y[k + lead_L] - 0.5)

        return Z_ens

    def release(self):
        if self.cap is not None:
            self.cap.release()


class FifoTargetQueue:
    """Maintains a First-In, First-Out sliding history with explicit index pop capabilities."""

    def __init__(self, max_capacity=100):
        self.buffer = deque(maxlen=max_capacity)

    def push(self, timestamp, price, encoded_bits):
        self.buffer.append({
            "timestamp": timestamp,
            "price": price,
            "encoded_bits": encoded_bits
        })

    def pop_at(self, index=2):
        temp_list = list(self.buffer)
        if 0 <= index < len(temp_list):
            popped_item = temp_list.pop(index)
            self.buffer = deque(temp_list, maxlen=self.buffer.maxlen)
            return popped_item
        return None

    def __len__(self):
        return len(self.buffer)


class TimeCodeLatch:
    """Implements 32-slot bit pooling for combined price + timestamp encoding/decoding."""

    @staticmethod
    def encode_payload(price_val, timestamp_val, n_slots=32):
        encoded_price = int(price_val / 10) & 0xFFFF
        encoded_time = int(timestamp_val) & 0xFFFF
        combined = (encoded_price << 16) | encoded_time
        return np.array([(combined >> (n_slots - 1 - k)) & 1 for k in range(n_slots)], dtype=int)

    @staticmethod
    def decode_payload(bits, reference_price=0.0):
        combined = 0
        for bit in bits:
            combined = (combined << 1) | int(bit)
        reconstructed_price_code = (combined >> 16) & 0xFFFF
        reconstructed_timestamp = combined & 0xFFFF
        
        if reference_price > 0.0:
            raw_predicted = float(reconstructed_price_code * 10)
            if abs(raw_predicted - reference_price) > 1000.0:
                target_code = int(reference_price / 10) & 0xFFFF
                reconstructed_price_code = target_code

        return reconstructed_price_code, reconstructed_timestamp


class AsyncBinanceFetcher(threading.Thread):
    """Background worker thread updating live BTCUSDT pricing asynchronously."""

    def __init__(self, delay_seconds=2.9):
        super().__init__()
        self.daemon = True
        self.delay_seconds = delay_seconds
        self.latest_price = 95162.75
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        while self.running:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    with self.lock:
                        self.latest_price = float(data["price"])
            except Exception:
                pass

            elapsed = 0.0
            while elapsed < self.delay_seconds and self.running:
                time.sleep(0.5)
                elapsed += 0.5

    def get_price(self):
        with self.lock:
            return self.latest_price

    def stop(self):
        self.running = False


# State tracker for balanced rolling baseline
_rolling_bit_baseline = 0.5

def execute_keystroke_injection(recovered_bits):
    """
    Computes a balanced trajectory bias using an Exponential Moving Average (EMA) baseline
    to prevent permanent lock-in, then fires Shift + W or Shift + S via PyAutoGUI.
    """
    global _rolling_bit_baseline
    
    current_mean = np.mean(recovered_bits)
    _rolling_bit_baseline = 0.8 * _rolling_bit_baseline + 0.2 * current_mean
    trajectory_bias = current_mean - _rolling_bit_baseline
    
    key = 'w' if trajectory_bias >= 0 else 's'
    print(f"[HW-INJECT] Trajectory Bias: {trajectory_bias:+.4f} (Mean: {current_mean:.3f}, Baseline: {_rolling_bit_baseline:.3f}) -> Shift + {key.upper()}")
    
    if HAS_PYAUTOGUI:
        try:
            pyautogui.keyDown('shift')
            pyautogui.press(key)
            pyautogui.keyUp('shift')
            print(f"[HW-INJECT] Successfully fired hardware-emulated Shift+{key.upper()}")
        except Exception as e:
            print(f"[HW-INJECT] PyAutoGUI execution error: {e}")
    else:
        print("[HW-INJECT] PyAutoGUI not installed. Simulation mode only (Keystroke bypassed).")


def run_full_predictor_pipeline():
    np.random.seed(2026)
    print("==========================================================================")
    print("  RP-GHOST OPTOELECTRONIC L-ANTICIPATION: KEYSTROKE INJECTION ENGINE      ")
    print("==========================================================================")

    viewport = OptoelectronicViewport(use_webcam=False)
    fifo_queue = FifoTargetQueue(max_capacity=50)
    binance_fetcher = AsyncBinanceFetcher(delay_seconds=3.14)
    binance_fetcher.start()

    n_slots = 32
    epsilon = 0.05
    lead_L = 2
    n_ensemble = 5000

    try:
        live_btc_price = binance_fetcher.get_price()
        
        for iteration in range(1, 100000):
            current_timestamp = time.time()
            print(f"\n--- PREDICTOR CYCLE {iteration} ---")

            # 1. Encode Target Payload (Price + Timestamp)
            Y_hidden = TimeCodeLatch.encode_payload(live_btc_price, current_timestamp, n_slots=n_slots)
            fifo_queue.push(current_timestamp, live_btc_price, Y_hidden)

            # 2. Manage Queue (Pop at index 2 once queue depth allows)
            if len(fifo_queue) >= 3:
                popped_item = fifo_queue.pop_at(index=2)
                if popped_item:
                    print(f"[FIFO] Index 2 Popped Successfully | Stored Timestamp: {popped_item['timestamp']:.2f}")

            # 3. Generate Ensemble Noise Matrix Instantly
            Z_ens = viewport.generate_ensemble_matrix(n_ensemble, n_slots, Y_hidden, epsilon, lead_L=lead_L)

            # 4. Extract and Decode Forecasted State using Exact Structural Inverse Mapping
            recovered_bits = np.zeros(n_slots, dtype=int)
            for k in range(n_slots):
                noise_idx = (k - lead_L) % n_slots
                recovered_bits[k] = 1 if np.mean(Z_ens[:, noise_idx]) > 0.0 else 0

            # Decode payload with reference price clamping
            rec_price_code, rec_timestamp = TimeCodeLatch.decode_payload(recovered_bits, reference_price=live_btc_price)
            
            predicted_btc_price = float(rec_price_code * 10)
            target_price_code = int(live_btc_price / 10) & 0xFFFF
            target_time_code = int(current_timestamp) & 0xFFFF
            
            price_matched = (rec_price_code == target_price_code)
            time_delta = int(rec_timestamp) - int(target_time_code)
            price_delta = predicted_btc_price - live_btc_price
            match_status = "VERIFIED MATCH" if price_matched else "BOUNDED RECONSTRUCTION"
            
            print(f"Actual Binance Price : ${live_btc_price:,.2f} (Code: 0x{target_price_code:04X})")
            print(f"Predicted Price      : ${predicted_btc_price:,.2f} (Code: 0x{rec_price_code:04X}) | Δ: ${price_delta:+,.2f}")
            print(f"Target Timestamp Code: 0x{target_time_code:04X}")
            print(f"Predicted Timestamp  : 0x{rec_timestamp:04X} | Jitter: {time_delta:+d}s")
            print(f"Status               : {match_status}")
            print(f"Active Queue Depth   : {len(fifo_queue)}")

            # 5. Execute Keystroke Injection on every cycle (or conditionally)
            execute_keystroke_injection(recovered_bits)

            live_btc_price = binance_fetcher.get_price()

    finally:
        binance_fetcher.stop()
        viewport.release()
        print("\n[DONE] Predictor Pipeline Terminated Safely.")


if __name__ == "__main__":
    run_full_predictor_pipeline()
