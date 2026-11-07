"""
QUERY-DRIVEN BIT ORACLE — ISOLATED WORD/SENTENCE GENERATOR

Architecture:
    Query textbox
        -> SHA-256 query bitstream
        -> Editable bitstream
        -> Hidden candidate index
        -> Higher/lower Bit Oracle
        -> Dataset word index
        -> Isolated word output

Each generated word receives its own oracle search.

Requirements:
    Python 3.10+
    No external packages required.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from collections import Counter, defaultdict
import hashlib
import re


# ============================================================
# TOKENIZATION AND QUERY BIT GENERATION
# ============================================================

def tokenize(text):
    return re.findall(
        r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[.,!?;:]",
        text.lower()
    )


def query_to_bits(query):
    """
    Convert the actual user query into a deterministic bitstream.
    The dataset is NOT used to create these bits.
    """
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    return "".join(format(byte, "08b") for byte in digest)


def text_to_index_bits(value, depth=10):
    """
    Convert binary text or ordinary words into an oracle-compatible
    index bitstream of exactly `depth` bits.

    - Binary input (0/1) is used directly.
    - Words/text are SHA-256 hashed.
    - The result is truncated or zero-padded to the oracle depth.
    """
    if not value.strip():
        return ""

    cleaned = "".join(
        ch for ch in value
        if ch in "01"
    )

    if cleaned:
        bits = cleaned

    else:
        bits = query_to_bits(value)

    if len(bits) < depth:
        bits = bits.ljust(depth, "0")

    return bits[:depth]


# ============================================================
# BIT POOLING
# ============================================================

class BitPool:

    def __init__(self):
        self.kernel = 4
        self.stride = 2
        self.operation = "POPCOUNT"
        self.threshold = 0.5

    def apply(self, value, depth=10):

        bits = [
            (value >> i) & 1
            for i in range(depth)
        ]

        output = []

        for i in range(0, len(bits), self.stride):

            block = bits[i:i + self.kernel]

            if not block:
                continue

            if self.operation == "POPCOUNT":
                result = int(
                    sum(block) / len(block) >= self.threshold
                )

            elif self.operation == "AND":
                result = int(all(block))

            elif self.operation == "OR":
                result = int(any(block))

            elif self.operation == "XOR":
                result = sum(block) % 2

            elif self.operation == "MAJORITY":
                result = int(
                    sum(block) > len(block) / 2
                )

            else:
                result = block[0]

            output.append(result)

        result = 0

        for i, bit in enumerate(output):
            result |= bit << i

        return result


# ============================================================
# ISOLATED HIGHER / LOWER BIT ORACLE
# ============================================================

class BitOracle:

    def __init__(self, depth=10):

        self.depth = depth
        self.maximum = (1 << depth) - 1

        self.target = 0
        self.low = 0
        self.high = self.maximum

        self.guess = 0
        self.steps = 0
        self.done = False

        self.history = []
        self.pool = BitPool()

    def reset(self, target):

        if not 0 <= target <= self.maximum:
            raise ValueError("Oracle target outside bit depth.")

        self.target = target
        self.low = 0
        self.high = self.maximum

        self.guess = 0
        self.steps = 0
        self.done = False

        self.history = []

    def query(self, trial):

        # Hidden comparison. Only one higher/lower bit is returned.
        return int(self.target >= trial)

    def step(self):

        if self.done:
            return self.guess

        if self.low > self.high:
            raise RuntimeError("Oracle search interval invalid.")

        middle = (self.low + self.high) // 2

        response = self.query(middle)

        self.steps += 1

        if response:
            self.low = middle
            self.guess = middle
            self.high = self.high

            if middle == self.target:
                self.done = True
                self.guess = middle
                self.history.append(
                    (middle, response, self.guess)
                )
                return self.guess

            self.low = middle + 1

        else:
            self.high = middle - 1

        self.history.append(
            (middle, response, self.guess)
        )

        if self.low > self.high:
            self.done = True
            self.guess = self.target

        return self.guess

    def run(self):

        while not self.done:
            self.step()

        return self.guess

    @property
    def pooled(self):
        return self.pool.apply(
            self.guess,
            self.depth
        )


# ============================================================
# DATASET WORD INDEX
# ============================================================

class DatasetIndex:

    def __init__(self, depth=10):

        self.depth = depth
        self.maximum_candidates = 1 << depth

        self.tokens = []
        self.unigrams = Counter()
        self.transitions = defaultdict(Counter)

        self.candidates = []
        self.target_index = None

    def load(self, text):

        self.tokens = tokenize(text)

        if not self.tokens:
            raise ValueError("Dataset is empty.")

        self.unigrams = Counter(self.tokens)

        self.transitions = defaultdict(Counter)

        # Build contextual word-index relationships.
        for i in range(len(self.tokens)):

            for width in range(1, min(8, i + 1) + 1):

                context = tuple(
                    self.tokens[i - width:i]
                )

                if len(context) == width:
                    self.transitions[context][
                        self.tokens[i]
                    ] += 1

    def candidates_for(self, query):

        query_tokens = tokenize(query)

        # Empty query: use dataset-wide unigram indexes.
        if not query_tokens:

            counts = self.unigrams

        else:

            counts = None

            # Find the longest available context.
            for width in range(
                min(8, len(query_tokens)),
                0,
                -1
            ):

                context = tuple(
                    query_tokens[-width:]
                )

                if context in self.transitions:
                    counts = self.transitions[context]
                    break

            if counts is None:
                raise ValueError(
                    "No dataset continuation found for this query. "
                    "Try a shorter query or expand the dataset."
                )

        ranked = counts.most_common(
            self.maximum_candidates
        )

        self.candidates = [
            {
                "index": i,
                "word": word,
                "count": count
            }
            for i, (word, count) in enumerate(ranked)
        ]

        return self.candidates

    def index_from_bits(self, bits):

        bits = "".join(
            c for c in bits if c in "01"
        )

        if not bits:
            raise ValueError("Oracle bitstream is empty.")

        if not self.candidates:
            raise ValueError("No dataset candidates loaded.")

        value = int(bits, 2)

        self.target_index = (
            value % len(self.candidates)
        )

        return self.target_index


# ============================================================
# MAIN APPLICATION
# ============================================================

class OracleApplication:

    def __init__(self, root):

        self.root = root
        self.root.title(
            "Query-Driven Bit Oracle — Isolated Word Generator"
        )

        self.root.geometry("1150x900")
        self.root.minsize(850, 650)

        self.dataset = DatasetIndex(depth=10)
        self.oracle = BitOracle(depth=10)

        self.oracle_runs = 0
        self.generated_words = []

        self.last_candidates = []
        self.last_query = ""

        self.auto_bits = True

        self.build_interface()

        self.load_default_dataset()

    # --------------------------------------------------------
    # INTERFACE
    # --------------------------------------------------------

    def build_interface(self):

        main = ttk.Frame(
            self.root,
            padding=10
        )

        main.pack(
            fill="both",
            expand=True
        )

        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        # DATASET

        dataset_frame = ttk.LabelFrame(
            main,
            text="1. Dataset — word vocabulary and indexes"
        )

        dataset_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )

        self.dataset_text = tk.Text(
            dataset_frame,
            height=7,
            wrap="word",
            font=("Consolas", 10)
        )

        self.dataset_text.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        ttk.Button(
            dataset_frame,
            text="Rebuild Dataset Index",
            command=self.rebuild_dataset
        ).pack(
            anchor="w",
            padx=5,
            pady=5
        )

        # QUERY

        query_frame = ttk.LabelFrame(
            main,
            text="2. User Query — editable oracle input"
        )

        query_frame.pack(
            fill="x",
            pady=5
        )

        self.query_text = tk.Text(
            query_frame,
            height=4,
            wrap="word",
            font=("Consolas", 11)
        )

        self.query_text.pack(
            fill="x",
            padx=5,
            pady=5
        )

        self.query_text.bind(
            "<KeyRelease>",
            self.query_changed
        )

        query_buttons = ttk.Frame(query_frame)
        query_buttons.pack(
            fill="x",
            padx=5,
            pady=5
        )

        ttk.Button(
            query_buttons,
            text="Derive Oracle Bits",
            command=self.derive_bits
        ).pack(side="left", padx=3)

        ttk.Button(
            query_buttons,
            text="Clear Query",
            command=self.clear_query
        ).pack(side="left", padx=3)

        # BITSTREAM

        bits_frame = ttk.LabelFrame(
            main,
            text="3. Oracle Index Bits — binary or words (10-bit)"
        )

        bits_frame.pack(
            fill="x",
            pady=5
        )

        self.bits_text = tk.Text(
            bits_frame,
            height=4,
            wrap="word",
            font=("Consolas", 10)
        )

        self.bits_text.pack(
            fill="x",
            padx=5,
            pady=5
        )

        bit_buttons = ttk.Frame(bits_frame)
        bit_buttons.pack(
            fill="x",
            padx=5,
            pady=5
        )

        ttk.Button(
            bit_buttons,
            text="Reload Query Bits",
            command=self.derive_bits
        ).pack(side="left", padx=3)

        ttk.Button(
            bit_buttons,
            text="Search Oracle",
            command=self.search_word
        ).pack(side="left", padx=3)

        ttk.Button(
            bit_buttons,
            text="Clear Bits",
            command=lambda: self.set_text(
                self.bits_text, ""
            )
        ).pack(side="left", padx=3)

        # ORACLE SETTINGS

        settings = ttk.LabelFrame(
            main,
            text="4. Bit Oracle Configuration"
        )

        settings.pack(
            fill="x",
            pady=5
        )

        ttk.Label(
            settings,
            text="Pooling operator:"
        ).pack(side="left", padx=5)

        self.operation = ttk.Combobox(
            settings,
            state="readonly",
            values=[
                "POPCOUNT",
                "AND",
                "OR",
                "XOR",
                "MAJORITY"
            ],
            width=12
        )

        self.operation.set("POPCOUNT")

        self.operation.pack(
            side="left",
            padx=5
        )

        self.operation.bind(
            "<<ComboboxSelected>>",
            self.change_operation
        )

        ttk.Label(
            settings,
            text="Sentence words:"
        ).pack(side="left", padx=5)

        self.word_count = tk.IntVar(value=12)

        ttk.Spinbox(
            settings,
            from_=1,
            to=100,
            textvariable=self.word_count,
            width=6
        ).pack(side="left", padx=5)

        ttk.Button(
            settings,
            text="Generate Sentence",
            command=self.generate_sentence
        ).pack(side="left", padx=5)

        ttk.Button(
            settings,
            text="Reset",
            command=self.reset
        ).pack(side="left", padx=5)

        # REGISTERS

        register_frame = ttk.LabelFrame(
            main,
            text="5. Oracle Registers"
        )

        register_frame.pack(
            fill="x",
            pady=5
        )

        self.register_text = tk.StringVar(
            value="Waiting for oracle."
        )

        ttk.Label(
            register_frame,
            textvariable=self.register_text,
            font=("Consolas", 10)
        ).pack(
            anchor="w",
            padx=8,
            pady=8
        )

        self.progress = ttk.Progressbar(
            register_frame,
            maximum=10,
            length=350,
            mode="determinate"
        )

        self.progress.pack(
            fill="x",
            padx=8,
            pady=5
        )

        # PREDICTION

        output_frame = ttk.LabelFrame(
            main,
            text="6. Isolated Oracle Output"
        )

        output_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )

        self.output_text = tk.Text(
            output_frame,
            height=5,
            wrap="word",
            font=("Consolas", 13),
            background="#e8f5e9"
        )

        self.output_text.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        # CANDIDATES

        candidate_frame = ttk.LabelFrame(
            main,
            text="7. Dataset Candidate Index"
        )

        candidate_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )

        self.candidate_text = tk.Text(
            candidate_frame,
            height=7,
            wrap="none",
            font=("Consolas", 9)
        )

        self.candidate_text.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        # STATUS

        self.status = tk.StringVar(
            value="Ready."
        )

        ttk.Label(
            main,
            textvariable=self.status,
            foreground="#007f5f"
        ).pack(
            fill="x",
            pady=5
        )

    # --------------------------------------------------------
    # TEXT HELPERS
    # --------------------------------------------------------

    def set_text(self, widget, value):

        widget.delete("1.0", "end")
        widget.insert("1.0", str(value))

    def get_text(self, widget):

        return widget.get(
            "1.0",
            "end-1c"
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    def load_default_dataset(self):

        default = (
            "The bit oracle searches a hidden number. "
            "The bit oracle uses higher and lower queries. "
            "The oracle selects words using dataset indexes. "
            "The pooled bits determine the candidate index. "
            "The selected word is recovered by an isolated search. "
            "A sentence is generated one word at a time. "
            "The dataset provides the vocabulary and context. "
            "The query provides the oracle bits. "
            "The system searches each word independently. "
            "Language emerges from contextual word relationships. "
            "The user can edit the query and the binary sequence. "
            "The oracle uses binary search to isolate the index. "
        )

        self.dataset_text.insert(
            "1.0",
            default
        )

        self.rebuild_dataset()

    def rebuild_dataset(self):

        try:

            self.dataset.load(
                self.get_text(self.dataset_text)
            )

            self.status.set(
                f"Dataset indexed: {len(self.dataset.tokens)} tokens | "
                f"{len(self.dataset.unigrams)} unique words."
            )

        except ValueError as error:

            messagebox.showwarning(
                "Dataset",
                str(error)
            )

    # --------------------------------------------------------
    # QUERY -> BITS
    # --------------------------------------------------------

    def query_changed(self, event=None):

        # Query editing does not erase the existing prediction.
        self.auto_bits = False

    def derive_bits(self):

        query = self.get_text(
            self.query_text
        )

        bits = text_to_index_bits(
            query,
            self.oracle.depth
        )

        self.set_text(
            self.bits_text,
            bits
        )

        self.auto_bits = True

        self.status.set(
            f"Derived {len(bits)} oracle index bits from the query."
        )

    def clear_query(self):

        self.set_text(
            self.query_text,
            ""
        )

        self.derive_bits()

    # --------------------------------------------------------
    # ORACLE BIT SEARCH
    # --------------------------------------------------------

    def change_operation(self, event=None):

        self.oracle.pool.operation = (
            self.operation.get()
        )

    def recover_index(self, query, bits):

        candidates = self.dataset.candidates_for(
            query
        )

        target_index = self.dataset.index_from_bits(
            bits
        )

        # Configure the hidden target.
        self.oracle.reset(
            target_index
        )

        self.oracle_runs += 1

        # The oracle independently performs its search.
        recovered_index = self.oracle.run()

        if recovered_index != target_index:
            raise RuntimeError(
                "Oracle verification failed."
            )

        if not 0 <= recovered_index < len(candidates):
            raise RuntimeError(
                "Recovered dataset index is invalid."
            )

        selected = candidates[recovered_index]

        self.last_candidates = candidates

        return selected, candidates

    def display_candidates(self, candidates, selected=None):

        lines = []

        for item in candidates:

            marker = (
                "  <-- RECOVERED"
                if selected is not None
                and item["index"] == selected["index"]
                else ""
            )

            lines.append(
                f'{item["index"]:4d} | '
                f'{item["word"]:<20} | '
                f'frequency={item["count"]}{marker}'
            )

        self.set_text(
            self.candidate_text,
            "\n".join(lines)
        )

    def update_registers(self):

        self.register_text.set(
            f"Guess: {self.oracle.guess:04d} | "
            f"Binary: {self.oracle.guess:010b} | "
            f"Pooled: {self.oracle.pooled:04d} | "
            f"Steps: {self.oracle.steps} | "
            f"Verified: {self.oracle.done}"
        )

        self.progress["value"] = min(
            self.oracle.steps,
            10
        )

    # --------------------------------------------------------
    # SINGLE WORD SEARCH
    # --------------------------------------------------------

    def search_word(self):

        query = self.get_text(
            self.query_text
        )

        bits_input = self.get_text(
            self.bits_text
        )

        bits = text_to_index_bits(
            bits_input,
            self.oracle.depth
        )

        if not bits:
            bits = text_to_index_bits(
                query,
                self.oracle.depth
            )

        self.set_text(
            self.bits_text,
            bits
        )

        try:

            selected, candidates = self.recover_index(
                query,
                bits
            )

            self.display_candidates(
                candidates,
                selected
            )

            self.set_text(
                self.output_text,
                selected["word"]
            )

            self.update_registers()

            self.status.set(
                f"ORACLE VERIFIED | "
                f"Word: {selected['word']} | "
                f"Dataset index: {selected['index']} | "
                f"Searches: {self.oracle_runs}"
            )

        except (ValueError, RuntimeError) as error:

            messagebox.showwarning(
                "Oracle Search",
                str(error)
            )

    # --------------------------------------------------------
    # MULTIWORD SENTENCE GENERATION
    # --------------------------------------------------------

    def generate_sentence(self):

        query = self.get_text(
            self.query_text
        )

        editable_bits_input = self.get_text(
            self.bits_text
        )

        editable_bits = text_to_index_bits(
            editable_bits_input,
            self.oracle.depth
        )

        if not editable_bits:
            editable_bits = text_to_index_bits(
                query,
                self.oracle.depth
            )

            self.set_text(
                self.bits_text,
                editable_bits
            )

        try:

            number = max(
                1,
                min(100, int(self.word_count.get()))
            )

        except (ValueError, tk.TclError):

            number = 12

        generated = tokenize(query)

        self.generated_words = []

        self.set_text(
            self.output_text,
            ""
        )

        self.root.update_idletasks()

        for position in range(number):

            current_query = " ".join(
                generated[-8:]
            )

            # The original editable bits control the first word.
            # Each subsequent word gets a fresh query-derived bitstream.
            if position == 0:

                current_bits = editable_bits

            else:

                current_bits = text_to_index_bits(
                    current_query,
                    self.oracle.depth
                )

            try:

                selected, candidates = self.recover_index(
                    current_query,
                    current_bits
                )

            except ValueError:

                # Back off context until the dataset has a continuation.
                selected = None
                candidates = None

                current_tokens = tokenize(
                    current_query
                )

                for width in range(
                    len(current_tokens) - 1,
                    0,
                    -1
                ):

                    shorter_query = " ".join(
                        current_tokens[-width:]
                    )

                    shorter_bits = text_to_index_bits(
                        shorter_query,
                        self.oracle.depth
                    )

                    try:

                        selected, candidates = self.recover_index(
                            shorter_query,
                            shorter_bits
                        )

                        break

                    except ValueError:
                        continue

                if selected is None:
                    self.status.set(
                        "Generation stopped: no dataset continuation."
                    )
                    break

            word = selected["word"]

            generated.append(word)
            self.generated_words.append(word)

            # Keep the complete sentence visible.
            sentence = " ".join(generated)

            self.set_text(
                self.output_text,
                sentence
            )

            self.display_candidates(
                candidates,
                selected
            )

            self.update_registers()

            self.status.set(
                f"Word {position + 1}/{number} | "
                f"Recovered: {word} | "
                f"Index: {selected['index']} | "
                f"Oracle searches: {self.oracle_runs}"
            )

            self.root.update_idletasks()

        final_sentence = " ".join(
            generated
        ).strip()

        if final_sentence:

            final_sentence = (
                final_sentence[0].upper()
                + final_sentence[1:]
            )

            if final_sentence[-1] not in ".!?":
                final_sentence += "."

        self.set_text(
            self.output_text,
            final_sentence
        )

        self.status.set(
            f"Generation complete | "
            f"{len(self.generated_words)} new words | "
            f"{self.oracle_runs} isolated oracle searches."
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.oracle.reset(0)

        self.oracle_runs = 0
        self.generated_words = []

        self.set_text(
            self.output_text,
            ""
        )

        self.set_text(
            self.candidate_text,
            ""
        )

        self.update_registers()

        self.status.set(
            "Oracle reset. Dataset preserved."
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = OracleApplication(root)

    root.mainloop()
