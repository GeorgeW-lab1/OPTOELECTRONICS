# OPTOELECTRONICS: Bit-Pooling Higher-Lower Search & Optoelectronic Decision Circuit Lab

An interactive Python framework implementing an **optoelectronic decision circuit** driven by a live USB camera feed, featuring secure higher/lower oracles, visual bit-pooling hardware registers, and time-coded execution.

---

## Overview

This repository bridges classical binary search algorithms, cryptography-style oracle security, and hardware-emulated optoelectronic logic (using Tkinter and OpenCV). 

Every camera frame represents a single trial where motion detection, optical energy hashing, and circuit timing gates dictate whether decision states propagate through simulated RC networks and 74HC74 flip-flops. Separately, the project includes specialized decision engines for **hidden number spot-checking**, **binary higher/lower bit searches**, and **big-number factorisation**.

---

## Core Architecture & Features

### 1. Optoelectronic Camera & Motion Pipeline
* **Motion Detection:** Captures live USB frames, shrinks them to $128 \times 96$, blurs them, and calculates pixel-wise differences (`diff > threshold`) to quantify motion energy ($x$).
* **Driver A (Randomized Witness Test):** Derives an unpredictable test witness $r$ from the motion pattern itself (hashed via SHA-256), running a Miller-Rabin primality witness test on $x$.
* **Driver B (Deterministic Check & Link):** Simulates a capacitor $C$ charging through resistor $R$ over a frame period, feeding an NPN emitter follower ($V_B = V_C - 0.7\text{ V}$) to verify valid logic HIGH thresholds ($\ge 3.5\text{ V}$).
* **Time-Code Gates ($Y$):** Clocks results through a NAND-not inverter stage and gates them into a simulated 74HC74 flip-flop on the frame clock edge.

### 2. Secure Bit-Pooling Higher/Lower Oracle
* **Isolated Target Storage:** The `BitPoolingOracle` encapsulates the secret target behind a secure property wrapper. Any unauthorized read attempt before search convergence throws a explicit `PermissionError`.
* **Bit-by-Bit Pooling:** Instead of direct equality checks per query, the search engine tests trial values most-significant-bit (MSB) first.
* **Hardware-Emulated Bit Operators:** Supports multiple pooling operator modes (`POPCOUNT`, `AND`, `OR`, `XOR`, `MAJORITY`) applied across sliding kernel windows.
* **Auto-Reveal:** Once all bits are successfully pooled and verified, the oracle lifts its security restriction to reveal the target across the visual bit registers.

### 3. Specialized Search & Testing Labs
* **Hidden Number Guesser:** A separate background clock thread cycles a candidate through a numeric range. Two sequential even motion frames trigger a timing spot-check, recording hit probabilities and binomial tail distributions.
* **Higher/Lower Binary Search:** Pools a large bit-depth secret (default 1000 bits) across armed frames, dropping search complexity from $O(2^N)$ down to $O(N)$ oracle queries.
* **Factor Search Engine:** Integrates a robust big-number factorisation ladder (trial division, Fermat, Pollard rho/p-1, ECM, and quadratic sieve) running in safe background subprocesses (`gmpy2`-accelerated when available).

---

## Requirements

* **Python** 3.8+
* **OpenCV** (`opencv-python`)
* **NumPy** (`numpy`)
* **Optional:** `gmpy2` (greatly accelerates big-number factorisation performance).

---

## Installation & Running

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GeorgeW-lab1/OPTOELECTRONICS.git
   cd OPTOELECTRONICS
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy opencv-python
   # Optional for fast factoring:
   pip install gmpy2
   ```

3. **Launch the application:**
   ```python
   python rp_circuit_camera.py
   ```
   *(Note: Ensure your USB webcam is connected and not locked by another video application.)*

---

## Usage Guide

* **Preview Only vs. Start Circuit:** Use *Preview Only* to verify your camera alignment and motion threshold settings without engaging circuit logic. Click *Start Circuit* to run the full pipeline.
* **Motion Calibration:** Adjust the pixel difference slider (`pixel diff >`) and minimum motion pixels (`min moving px`) to match your lighting conditions.
* **Triggering Actions:** Higher/lower searches and hidden-number spot checks advance on **two sequential even frames** (where $x > 0$ and $x \pmod 2 == 0$ for two consecutive frames). Waving your hand or moving steadily in front of the camera acts as the physical clock source.

---

## License

This project is open-source and structured for experimental optoelectronic research and educational exploration.
