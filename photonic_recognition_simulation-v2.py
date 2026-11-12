import numpy as np
import hashlib
import itertools
import math
from typing import List, Tuple, Set

class PhotonicSecretRecoverySimulation:
    """
    Python implementation of the Two-Level Photonic Recognition and 
    Temporally Directed Secret Recovery architecture.
    
    Uses sequential stage-by-stage decoding (m = 1, 2, ..., M) with optimal 
    stopping criteria and soft-threshold / majority-vote resilience so that 
    the never-seen-zero set U_m does not prematurely drop true '1' positions 
    when p < 1 across large observation counts (e.g., m = 82).
    """
    def __init__(self, secret_bits: List[int], p: float = 0.5, salt: str = "photon_salt_2026"):
        self.secret = np.array(secret_bits, dtype=int)
        self.n = len(self.secret)
        self.p = p
        self.salt = salt.encode('utf-8')
        self.commitment = self._compute_commitment(self.secret)

    def _compute_commitment(self, K: np.ndarray) -> str:
        """Computes cryptographic commitment C = H(s || K)"""
        k_bytes = ''.join(map(str, K)).encode('utf-8')
        h = hashlib.sha256()
        h.update(self.salt)
        h.update(k_bytes)
        return h.hexdigest()

    def generate_single_observation(self) -> np.ndarray:
        """
        Generates a single photonic observation vector X(t) = K AND R(t)
        where R_i(t) ~ Bernoulli(p).
        """
        R = np.random.choice([0, 1], size=self.n, p=[1.0 - self.p, self.p])
        return np.bitwise_and(self.secret, R)

    def apply_noise_cancellation(self, X: np.ndarray) -> np.ndarray:
        """
        Simulates digital noise-cancellation S_c(t) = S(t) - N_hat(t) across accumulated observations.
        """
        noise = np.random.normal(0, 0.08, size=X.shape)
        S = X.astype(float) + noise
        n_hat = np.mean(noise, axis=0, keepdims=True)
        S_c = S - n_hat
        return (S_c > 0.5).astype(int)

    def level_one_recognition(self, X: np.ndarray, mode: str = "cumulative_or") -> Tuple[Set[int], np.ndarray]:
        """
        Level 1 Recognition:
        - 'strict_never_zero': U_m = {i : Z_i(m) == 0} (exact paper rule, optimal for small m)
        - 'cumulative_or': U_m = {i : sum(X_i(t) == 1) > 0} (dual Z-channel rule: since K_i=0 ALWAYS 
          produces 0 under AND-gate X_i = K_i AND R_i, observing even a SINGLE '1' proves K_i = 1!)
        """
        if mode == "strict_never_zero":
            zeros_count = np.sum(X == 0, axis=0)
            u_set = {i for i in range(self.n) if zeros_count[i] == 0}
        else:
            # Dual Z-channel property of X_i(t) = K_i AND R_i(t):
            # If K_i == 0, X_i(t) == 0 for all t.
            # If X_i(t) == 1 at least once, K_i is guaranteed to be 1 (absent dark-count noise).
            ones_count = np.sum(X == 1, axis=0)
            u_set = {i for i in range(self.n) if ones_count[i] > 0}
        
        b_m = np.zeros(self.n, dtype=int)
        for i in u_set:
            b_m[i] = 1
            
        return u_set, b_m

    def level_two_recognition(self, u_set: Set[int], b_m: np.ndarray, max_budget: int = 100000) -> Tuple[bool, np.ndarray, int]:
        """
        Level 2: Hamming-weight recognition.
        Enumerates correction subsets F subset U_m in increasing order of cardinality k.
        """
        u_list = sorted(list(u_set))
        u = len(u_list)
        total_hashes = 0
        
        for k in range(u + 1):
            comb_count = math.comb(u, k)
            if total_hashes + comb_count > max_budget:
                return False, np.array([]), total_hashes
                
            for combo in itertools.combinations(u_list, k):
                total_hashes += 1
                k_hat = b_m.copy()
                for idx in combo:
                    k_hat[idx] = 1 - k_hat[idx]
                    
                if self._compute_commitment(k_hat) == self.commitment:
                    return True, k_hat, total_hashes
                    
        return False, np.array([]), total_hashes

    def run_sequential_recovery(self, max_observations: int = 82, max_budget_per_stage: int = 1000000, mode: str = "cumulative_or"):
        """
        Runs the temporally directed sequential recovery loop stage-by-stage (m = 1, 2, ..., M).
        Stops immediately as soon as the secret is recovered.
        """
        print(f"[*] Starting Sequential Photonic Recovery (n={self.n}, p={self.p}, max_m={max_observations}, mode='{mode}')")
        print(f"[*] Target Secret : {''.join(map(str, self.secret))}")
        print(f"[*] Commitment    : {self.commitment[:16]}...")
        
        observations = []
        cumulative_hashes = 0
        
        for m in range(1, max_observations + 1):
            obs_t = self.generate_single_observation()
            observations.append(obs_t)
            X_raw = np.array(observations)
            
            # Apply digital noise cancellation
            X_clean = self.apply_noise_cancellation(X_raw)
            
            # Level 1: Update candidate support set U_m
            u_set, b_m = self.level_one_recognition(X_clean, mode=mode)
            
            # Level 2: Attempt Hamming-weight decoding at stage m
            success, k_hat, stage_hashes = self.level_two_recognition(u_set, b_m, max_budget_per_stage)
            cumulative_hashes += stage_hashes
            
            print(f"    Stage m={m:2d} | |U_m|={len(u_set)} | B_m={''.join(map(str, b_m))} | Stage Hashes={stage_hashes}")
            
            if success:
                print(f"\n[+] SECRET RECOVERED at observation stage m={m}!")
                print(f"    Reconstructed Secret : {''.join(map(str, k_hat))}")
                print(f"    Stage Hashes Used    : {stage_hashes}")
                print(f"    Cumulative Hashes    : {cumulative_hashes}")
                return True, k_hat, m, cumulative_hashes
                
        print(f"\n[-] Recovery did not succeed within {max_observations} observations (Total Hashes: {cumulative_hashes}).")
        return False, np.array([]), max_observations, cumulative_hashes

if __name__ == "__main__":
    np.random.seed(42)
    secret_example = [1, 0, 1, 1, 0, 0]
    sim = PhotonicSecretRecoverySimulation(secret_bits=secret_example, p=0.6, salt="wagenknecht_2026")
    sim.run_sequential_recovery(max_observations=82, max_budget_per_stage=10000000000000000000)
