"""
QUERY-DRIVEN BIT ORACLE — NLTK WORD PREDICTION / COMPLETION

Architecture (every word gets its own isolated oracle search):

    Typing box  ---->  NLTK n-gram model (Kneser-Ney)  ---->  ranked candidate index
                                                                   |
    Intended-words box (optional, the words "meant to be typed")   |
         = hidden target index, held ONLY inside the BitOracle <---+
                                                                   |
    BitOracle reveals the target index MSB-first, one higher/lower bit per step
                                                                   |
    BitPredictor forecasts the remaining bits from the NLTK probabilities
    using only the bits revealed so far (it never sees the target)
                                                                   |
    When the oracle has finished all bits, the word is committed
    to the output box, then the next word starts with a fresh oracle.

Requirements:
    Python 3.10+
    pip install nltk
    (tokenisation, language model and detokenisation need no NLTK downloads;
     the optional "Load NLTK Brown corpus" button downloads 'brown' on demand)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from itertools import accumulate
import hashlib

try:
    from nltk.lm import KneserNeyInterpolated
    from nltk.lm.preprocessing import padded_everygram_pipeline
    from nltk.tokenize import RegexpTokenizer
    from nltk.tokenize.treebank import TreebankWordDetokenizer
except ImportError as error:  # pragma: no cover
    raise SystemExit("NLTK is required:  pip install nltk") from error


DEPTH = 10                      # oracle bit depth -> up to 1024 candidates
NGRAM_ORDER = 3                 # NLTK language model order
MAX_POOL = 1500                 # most frequent words scored per prediction
PAD = "<s>"
TERMINAL = {".", "!", "?"}
SPECIAL = {"<s>", "</s>", "<UNK>"}

_TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[.,!?;:]")
_DETOK = TreebankWordDetokenizer()


# ============================================================
# TOKENIZATION AND QUERY BIT GENERATION
# ============================================================

def tokenize(text):
    return _TOKENIZER.tokenize(text.lower())


def detokenize(tokens):
    return _DETOK.detokenize(tokens)


def query_to_bits(query):
    """Deterministic SHA-256 bitstream of the query (dataset is NOT used)."""
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    return "".join(format(byte, "08b") for byte in digest)


def text_to_index_bits(value, depth=DEPTH):
    """Binary text is used directly, anything else is hashed. Result: `depth` bits."""
    if not value.strip():
        return ""

    cleaned = "".join(ch for ch in value if ch in "01")
    bits = cleaned if cleaned else query_to_bits(value)

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

    def apply(self, value, depth=DEPTH):
        bits = [(value >> i) & 1 for i in range(depth)]
        output = []

        for i in range(0, len(bits), self.stride):
            block = bits[i:i + self.kernel]

            if not block:
                continue

            if self.operation == "POPCOUNT":
                result = int(sum(block) / len(block) >= self.threshold)
            elif self.operation == "AND":
                result = int(all(block))
            elif self.operation == "OR":
                result = int(any(block))
            elif self.operation == "XOR":
                result = sum(block) % 2
            elif self.operation == "MAJORITY":
                result = int(sum(block) > len(block) / 2)
            else:
                result = block[0]

            output.append(result)

        value_out = 0
        for i, bit in enumerate(output):
            value_out |= bit << i

        return value_out


# ============================================================
# ISOLATED HIGHER / LOWER BIT ORACLE  (MSB-first bit pooling)
# ============================================================

class BitOracle:
    """
    Holds a secret index. The only thing it ever emits is one
    higher/lower bit per step, most-significant bit first.
    The target is classified until every bit has been revealed.
    """

    def __init__(self, depth=DEPTH):
        self.depth = depth
        self.maximum = (1 << depth) - 1
        self.pool = BitPool()
        self._target = 0
        self.reset(0)

    def reset(self, target):
        if not 0 <= target <= self.maximum:
            raise ValueError("Oracle target outside bit depth.")

        self._target = target
        self.guess = 0          # pooled (revealed) bits so far
        self.steps = 0
        self.done = False
        self.history = []

    def _query(self, trial):
        # Hidden comparison: a single higher/lower bit leaves the oracle.
        return int(self._target >= trial)

    def step(self):
        if self.done:
            return None

        shift = self.depth - self.steps - 1
        trial = self.guess | (1 << shift)
        bit = self._query(trial)

        self.guess |= bit << shift
        self.steps += 1
        self.history.append((trial, bit, self.guess))

        if self.steps >= self.depth:
            self.done = True

        return bit

    @property
    def revealed(self):
        """Bits revealed so far, MSB first, as a string."""
        return format(self.guess, f"0{self.depth}b")[:self.steps]

    @property
    def target(self):
        if not self.done:
            raise PermissionError(
                "Oracle target is classified until every bit is pooled."
            )
        return self._target

    @property
    def pooled(self):
        return self.pool.apply(self.guess, self.depth)


# ============================================================
# NLTK WORD MODEL
# ============================================================

class NltkWordModel:
    """Kneser-Ney n-gram model (nltk.lm) that ranks candidate words."""

    def __init__(self, order=NGRAM_ORDER):
        self.order = order
        self.lm = None
        self.tokens = []
        self.words = []
        self.freq = {}
        self._cache = {}

    @property
    def ready(self):
        return self.lm is not None

    def fit(self, text):
        tokens = tokenize(text)

        if not tokens:
            raise ValueError("Dataset is empty.")

        sentences, current = [], []
        for token in tokens:
            current.append(token)
            if token in TERMINAL:
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)

        train, vocab = padded_everygram_pipeline(self.order, sentences)
        lm = KneserNeyInterpolated(self.order)
        lm.fit(train, vocab)

        self.lm = lm
        self.tokens = tokens
        self.freq = {
            w: c for w, c in lm.vocab.counts.items() if w not in SPECIAL
        }
        self.words = sorted(self.freq, key=lambda w: (-self.freq[w], w))
        self._cache.clear()

    def _context(self, tokens):
        """Last (order-1) tokens of the current sentence, left-padded."""
        sentence = []
        for token in reversed(tokens):
            if token in TERMINAL:
                break
            sentence.append(token)
        sentence.reverse()

        size = self.order - 1
        context = sentence[-size:] if size else []
        return tuple([PAD] * (size - len(context)) + context)

    def _score(self, word, context):
        # Back off the context manually if nltk cannot score an unseen one.
        for start in range(len(context) + 1):
            try:
                return self.lm.score(word, context[start:] or None)
            except ZeroDivisionError:
                continue
        return self.freq.get(word, 0) / max(1, sum(self.freq.values()))

    def ranked(self, tokens, prefix=None, limit=1 << DEPTH, force=()):
        """
        Candidate words ranked by NLTK probability, best first.
        Index 0 = most probable word. `prefix` restricts to word completions.
        """
        context = self._context(tokens)
        key = (context, prefix, tuple(force), limit)

        if key in self._cache:
            return self._cache[key]

        pool = self.words
        if prefix:
            pool = [w for w in pool if w.startswith(prefix)]
        pool = pool[:MAX_POOL]

        for word in force:
            if word in self.freq and word not in pool:
                pool.append(word)

        scored = [(w, max(self._score(w, context), 1e-12)) for w in pool]
        scored.sort(key=lambda item: (-item[1], item[0]))

        result = [
            {"index": i, "word": w, "prob": p, "count": self.freq.get(w, 0)}
            for i, (w, p) in enumerate(scored[:limit])
        ]

        self._cache[key] = result
        return result


# ============================================================
# BIT PREDICTOR  (never sees the oracle's target)
# ============================================================

class BitPredictor:
    """
    Forecasts the unrevealed bits of the candidate index. For every
    remaining bit it compares the NLTK probability mass of the upper
    half of the current interval against the lower half.
    """

    def __init__(self, depth=DEPTH):
        self.depth = depth

    @staticmethod
    def _mass(cum, n, a, b):
        a, b = min(a, n), min(b + 1, n)
        return cum[b] - cum[a] if b > a else 0.0

    def forecast(self, cum, candidates, revealed):
        n = len(candidates)
        remaining = self.depth - len(revealed)
        low = (int(revealed, 2) << remaining) if revealed else 0
        high = low + (1 << remaining) - 1

        interval_mass = self._mass(cum, n, low, high)
        span = 1 << remaining
        bits, p_one = "", None

        for k in range(remaining):
            half = span >> 1
            lower = self._mass(cum, n, low, low + half - 1)
            upper = self._mass(cum, n, low + half, low + span - 1)
            total = lower + upper
            p1 = (upper / total) if total > 0 else 0.0

            if k == 0:
                p_one = p1

            bit = 1 if p1 > 0.5 else 0
            bits += str(bit)

            if bit:
                low += half
            span = half

        index = low
        word = candidates[index]["word"] if index < n else None
        word_mass = candidates[index]["prob"] if index < n else 0.0

        return {
            "next_bit": int(bits[0]) if bits else None,
            "p_one": p_one,
            "predicted_bits": revealed + bits,
            "index": index,
            "word": word,
            "word_share": (word_mass / interval_mass) if interval_mass else 0.0,
            "interval": (int(revealed, 2) << remaining if revealed else 0,
                         (int(revealed, 2) << remaining if revealed else 0)
                         + (1 << remaining) - 1),
        }


# ============================================================
# PREDICTION SESSION  (pure logic, no GUI)
# ============================================================

class PredictionSession:
    """
    One run: complete a partly typed word (optional), then keep predicting
    words. Each word is found by a fresh, isolated BitOracle search while
    the BitPredictor forecasts its bits.
    """

    def __init__(self, model, oracle, predictor, typing_text,
                 intended_text="", words=5, complete_partial=True,
                 seed_bits=""):

        self.model = model
        self.oracle = oracle
        self.predictor = predictor
        self.seed_bits = seed_bits

        tokens = tokenize(typing_text)
        intended = tokenize(intended_text)
        partial = None

        ends_mid_word = bool(typing_text) and (
            typing_text[-1].isalnum() or typing_text[-1] in "'’"
        )

        if complete_partial and tokens and ends_mid_word:
            partial = tokens.pop()

        if intended and partial and not intended[0].startswith(partial):
            tokens.append(partial)      # intended word does not extend it
            partial = None

        self.tokens = tokens
        self.partial = partial
        self.intended = intended[:100]
        self.total = len(self.intended) if self.intended else max(1, min(100, words))

        self.generated = []
        self.pos = 0
        self.finished = False
        self.message = ""

        self.word_ready = False
        self.candidates = []
        self.cum = [0.0]
        self.pred_log = []
        self.hit_log = []
        self.forecast = None
        self.word_hits = 0
        self.total_hits = 0
        self.total_bits = 0

    def _stop(self, message):
        self.finished = True
        self.word_ready = False
        self.message = message

    @property
    def error(self):
        return self.finished and self.message.startswith("STOPPED")

    def begin_word(self):
        prefix = self.partial if self.pos == 0 else None
        context = self.tokens + self.generated
        force = [self.intended[self.pos]] if self.intended else []

        candidates = self.model.ranked(
            context, prefix=prefix, limit=self.oracle.maximum + 1, force=force
        )

        if not candidates:
            self._stop("STOPPED: no dataset continuation for this context.")
            return

        if self.intended:
            word = self.intended[self.pos]
            matches = [c["index"] for c in candidates if c["word"] == word]

            if not matches:
                self._stop(
                    f"STOPPED: '{word}' is not in the model's candidate index. "
                    f"Add it to the dataset and rebuild."
                )
                return

            target = matches[0]

        else:
            key = " ".join(context[-8:]) + (prefix or "")
            bits = ""

            if self.pos == 0 and self.seed_bits:
                bits = text_to_index_bits(self.seed_bits, self.oracle.depth)

            if not bits:
                bits = query_to_bits(key)[:self.oracle.depth]

            target = int(bits, 2) % len(candidates)

        self.oracle.reset(target)
        self.candidates = candidates
        self.cum = [0.0] + list(accumulate(c["prob"] for c in candidates))
        self.pred_log, self.hit_log = [], []
        self.word_hits = 0
        self.forecast = self.predictor.forecast(self.cum, candidates, "")
        self.word_ready = True

    def tick(self):
        """Forecast the next bit, let the oracle reveal it, score the forecast."""
        before = self.predictor.forecast(
            self.cum, self.candidates, self.oracle.revealed
        )
        bit = self.oracle.step()
        hit = before["next_bit"] == bit

        self.pred_log.append(before["next_bit"])
        self.hit_log.append(hit)
        self.word_hits += int(hit)
        self.total_hits += int(hit)
        self.total_bits += 1

        self.forecast = self.predictor.forecast(
            self.cum, self.candidates, self.oracle.revealed
        )

        if self.oracle.done:
            index = self.oracle.target
            word = self.candidates[index]["word"]
            self.generated.append(word)
            self.pos += 1
            self.word_ready = False

            if self.pos >= self.total:
                self.finished = True
                self.message = (
                    f"Oracle finished {len(self.generated)} word(s) | "
                    f"bit forecast accuracy {self.total_hits}/{self.total_bits}"
                )

    def advance(self):
        if self.finished:
            return
        if not self.word_ready:
            self.begin_word()
        else:
            self.tick()

    @property
    def output_text(self):
        return detokenize(self.generated)

    @property
    def merged_text(self):
        return detokenize(self.tokens + self.generated)


# ============================================================
# MAIN APPLICATION
# ============================================================

class OracleApplication:

    def __init__(self, root):
        self.root = root
        self.root.title("Bit Oracle — NLTK Word Prediction (isolated per-word oracle)")
        self.root.geometry("1280x860")
        self.root.minsize(1000, 700)

        self.model = NltkWordModel()
        self.oracle = BitOracle(DEPTH)
        self.predictor = BitPredictor(DEPTH)

        self.session = None
        self.running = False
        self.loop_id = None
        self.debounce_id = None
        self._model_text = None

        self.build_interface()
        self.load_default_dataset()

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # --------------------------------------------------------
    # INTERFACE
    # --------------------------------------------------------

    def build_interface(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1, uniform="col")
        main.columnconfigure(1, weight=1, uniform="col")
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # ---- LEFT: dataset / typing / intended ----

        data_frame = ttk.LabelFrame(
            left, text="1. Dataset — trains the NLTK Kneser-Ney model"
        )
        data_frame.pack(fill="both", expand=True, pady=4)

        self.dataset_text = tk.Text(
            data_frame, height=8, wrap="word", font=("Consolas", 10)
        )
        self.dataset_text.pack(fill="both", expand=True, padx=5, pady=5)

        row = ttk.Frame(data_frame)
        row.pack(fill="x", padx=5, pady=4)
        ttk.Button(row, text="Rebuild NLTK Model",
                   command=self.rebuild_dataset).pack(side="left", padx=3)
        ttk.Button(row, text="Load NLTK Brown corpus",
                   command=self.load_brown).pack(side="left", padx=3)

        typing_frame = ttk.LabelFrame(
            left, text="2. Typing box — words typed so far "
                       "(a half-typed last word gets completed)"
        )
        typing_frame.pack(fill="both", expand=True, pady=4)

        self.typing_text = tk.Text(
            typing_frame, height=5, wrap="word", font=("Consolas", 12)
        )
        self.typing_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.typing_text.bind("<KeyRelease>", self.typing_changed)

        opts = ttk.Frame(typing_frame)
        opts.pack(fill="x", padx=5, pady=4)

        self.v_complete = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Complete partial last word",
                        variable=self.v_complete).pack(side="left", padx=3)

        self.v_auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Auto-predict while typing",
                        variable=self.v_auto).pack(side="left", padx=10)

        intended_frame = ttk.LabelFrame(
            left, text="3. Intended words — hidden oracle target "
                       "(the words meant to be typed; optional)"
        )
        intended_frame.pack(fill="both", expand=True, pady=4)

        self.intended_text = tk.Text(
            intended_frame, height=3, wrap="word", font=("Consolas", 12)
        )
        self.intended_text.pack(fill="both", expand=True, padx=5, pady=5)

        seed_row = ttk.Frame(intended_frame)
        seed_row.pack(fill="x", padx=5, pady=4)
        ttk.Label(
            seed_row, text="Seed bits (used only if intended box is empty):"
        ).pack(side="left")
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(seed_row, textvariable=self.seed_var, width=14).pack(
            side="left", padx=5
        )

        # ---- RIGHT: controls / registers / output / candidates ----

        ctrl = ttk.LabelFrame(right, text="4. Oracle controls")
        ctrl.pack(fill="x", pady=4)

        top = ttk.Frame(ctrl)
        top.pack(fill="x", padx=5, pady=4)

        ttk.Label(top, text="Pooling:").pack(side="left", padx=3)
        self.operation = ttk.Combobox(
            top, state="readonly", width=10,
            values=["POPCOUNT", "AND", "OR", "XOR", "MAJORITY"]
        )
        self.operation.set("POPCOUNT")
        self.operation.pack(side="left", padx=3)
        self.operation.bind("<<ComboboxSelected>>", self.change_operation)

        ttk.Label(top, text="Words:").pack(side="left", padx=(10, 3))
        self.word_count = tk.IntVar(value=5)
        ttk.Spinbox(top, from_=1, to=100, textvariable=self.word_count,
                    width=5).pack(side="left")

        ttk.Label(top, text="Bit delay ms:").pack(side="left", padx=(10, 3))
        self.speed = tk.IntVar(value=120)
        ttk.Spinbox(top, from_=0, to=2000, increment=20,
                    textvariable=self.speed, width=6).pack(side="left")

        buttons = ttk.Frame(ctrl)
        buttons.pack(fill="x", padx=5, pady=4)
        ttk.Button(buttons, text="Run Oracle", command=self.run_oracle).pack(
            side="left", padx=3)
        ttk.Button(buttons, text="Step Bit", command=self.step_bit).pack(
            side="left", padx=3)
        ttk.Button(buttons, text="Pause", command=self.pause).pack(
            side="left", padx=3)
        ttk.Button(buttons, text="Reset", command=self.reset).pack(
            side="left", padx=3)
        ttk.Button(buttons, text="Append output → typing box",
                   command=self.append_output).pack(side="left", padx=10)

        reg_frame = ttk.LabelFrame(right, text="5. Oracle bit registers")
        reg_frame.pack(fill="x", pady=4)

        self.reg_canvas = tk.Canvas(
            reg_frame, width=560, height=130, bg="#121414", highlightthickness=0
        )
        self.reg_canvas.pack(padx=5, pady=5)

        self.progress = ttk.Progressbar(
            reg_frame, maximum=DEPTH, mode="determinate"
        )
        self.progress.pack(fill="x", padx=8, pady=(0, 5))

        self.forecast_var = tk.StringVar(value="Waiting for oracle.")
        ttk.Label(
            reg_frame, textvariable=self.forecast_var,
            font=("Consolas", 10), justify="left"
        ).pack(anchor="w", padx=8, pady=(0, 8))

        out_frame = ttk.LabelFrame(
            right, text="6. Output box — words finished by the oracle"
        )
        out_frame.pack(fill="both", expand=True, pady=4)

        self.output_box = tk.Text(
            out_frame, height=4, wrap="word",
            font=("Consolas", 13), background="#e8f5e9"
        )
        self.output_box.pack(fill="both", expand=True, padx=5, pady=5)

        cand_frame = ttk.LabelFrame(
            right, text="7. Candidate index — shrinks as bits are revealed"
        )
        cand_frame.pack(fill="both", expand=True, pady=4)

        self.candidate_text = tk.Text(
            cand_frame, height=9, wrap="none", font=("Consolas", 9)
        )
        self.candidate_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(
            self.root, textvariable=self.status, foreground="#007f5f"
        ).pack(fill="x", padx=10, pady=4)

    # --------------------------------------------------------
    # TEXT HELPERS
    # --------------------------------------------------------

    def set_text(self, widget, value):
        widget.delete("1.0", "end")
        widget.insert("1.0", str(value))

    def get_text(self, widget):
        return widget.get("1.0", "end-1c")

    # --------------------------------------------------------
    # DATASET / NLTK MODEL
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
        self.dataset_text.insert("1.0", default)
        self.rebuild_dataset()

    def rebuild_dataset(self):
        text = self.get_text(self.dataset_text)

        try:
            self.model.fit(text)
        except ValueError as error:
            messagebox.showwarning("Dataset", str(error))
            return False

        self._model_text = text
        self.status.set(
            f"NLTK Kneser-Ney model (order {NGRAM_ORDER}) built: "
            f"{len(self.model.tokens)} tokens | {len(self.model.words)} unique words."
        )
        return True

    def ensure_model(self):
        if self.model.ready and self._model_text == self.get_text(self.dataset_text):
            return True
        return self.rebuild_dataset()

    def load_brown(self):
        try:
            from nltk.corpus import brown
            words = brown.words()[:15000]
        except LookupError:
            if not messagebox.askyesno(
                "NLTK corpus", "The 'brown' corpus is not installed.\nDownload it now?"
            ):
                return
            try:
                import nltk
                self.status.set("Downloading NLTK 'brown' corpus...")
                self.root.update_idletasks()
                if not nltk.download("brown", quiet=True):
                    raise RuntimeError("download failed")
                from nltk.corpus import brown
                words = brown.words()[:15000]
            except Exception as error:
                messagebox.showwarning("NLTK corpus", f"Could not load 'brown': {error}")
                return

        self.set_text(self.dataset_text, " ".join(words))
        self.rebuild_dataset()

    # --------------------------------------------------------
    # RUN CONTROL
    # --------------------------------------------------------

    def change_operation(self, event=None):
        self.oracle.pool.operation = self.operation.get()

    def typing_changed(self, event=None):
        if not self.v_auto.get():
            return
        if self.debounce_id is not None:
            self.root.after_cancel(self.debounce_id)
        self.debounce_id = self.root.after(700, self.run_oracle)

    def new_session(self):
        if not self.ensure_model():
            return False

        try:
            words = int(self.word_count.get())
        except (ValueError, tk.TclError):
            words = 5

        self.session = PredictionSession(
            self.model, self.oracle, self.predictor,
            typing_text=self.get_text(self.typing_text),
            intended_text=self.get_text(self.intended_text),
            words=words,
            complete_partial=self.v_complete.get(),
            seed_bits=self.seed_var.get().strip(),
        )
        self.set_text(self.output_box, "")
        return True

    def run_oracle(self):
        self.debounce_id = None
        self.stop_loop()

        if not self.new_session():
            return

        self.running = True
        self.loop()

    def step_bit(self):
        self.stop_loop()

        if self.session is None or self.session.finished:
            if not self.new_session():
                return

        self.advance_once()

    def pause(self):
        self.stop_loop()
        self.status.set("Paused. Press Step Bit to continue bit by bit.")

    def stop_loop(self):
        self.running = False
        if self.loop_id is not None:
            self.root.after_cancel(self.loop_id)
            self.loop_id = None

    def loop(self):
        self.loop_id = None

        if not self.running:
            return

        self.advance_once()

        if self.running and not self.session.finished:
            try:
                delay = max(0, int(self.speed.get()))
            except (ValueError, tk.TclError):
                delay = 120
            self.loop_id = self.root.after(delay, self.loop)
        else:
            self.running = False

    def advance_once(self):
        try:
            self.session.advance()
        except (ValueError, RuntimeError) as error:
            self.stop_loop()
            messagebox.showwarning("Oracle", str(error))
            return

        self.render()

        if self.session.finished:
            self.stop_loop()
            if self.session.message.startswith("STOPPED"):
                messagebox.showwarning("Oracle", self.session.message)

    def append_output(self):
        if self.session is None or not self.session.generated:
            return
        self.set_text(self.typing_text, self.session.merged_text + " ")

    def reset(self):
        self.stop_loop()
        self.session = None
        self.oracle.reset(0)
        self.set_text(self.output_box, "")
        self.set_text(self.candidate_text, "")
        self.reg_canvas.delete("all")
        self.progress["value"] = 0
        self.forecast_var.set("Waiting for oracle.")
        self.status.set("Oracle reset. Dataset and model preserved.")

    def close(self):
        self.stop_loop()
        if self.debounce_id is not None:
            self.root.after_cancel(self.debounce_id)
        self.root.destroy()

    # --------------------------------------------------------
    # RENDERING
    # --------------------------------------------------------

    def draw_registers(self):
        s, o = self.session, self.oracle
        canvas = self.reg_canvas
        canvas.delete("all")

        steps = o.steps if s.word_ready or o.done else 0
        revealed = o.revealed if steps else ""
        forecast_tail = ""
        if s.forecast:
            forecast_tail = s.forecast["predicted_bits"][len(revealed):]

        rows = [
            ("ORACLE", "#00e676"),
            ("FORECAST", "#ffb300"),
            ("HIT?", "#b388ff"),
        ]

        y = 8
        for label, color in rows:
            canvas.create_text(8, y + 11, text=label, fill="#88a0c0",
                               anchor="w", font=("Consolas", 10, "bold"))
            x = 92

            for k in range(DEPTH):
                text, fill, text_color = "", "#1e2430", "#708090"

                if label == "ORACLE":
                    if k < steps:
                        bit = revealed[k]
                        text = bit
                        fill = color if bit == "1" else "#1e2430"
                        text_color = "#000000" if bit == "1" else "#d2e0f0"
                    else:
                        text = "?"

                elif label == "FORECAST":
                    if k < len(s.pred_log):
                        bit = str(s.pred_log[k])
                        text = bit
                        fill = color if bit == "1" else "#1e2430"
                        text_color = "#000000" if bit == "1" else "#d2e0f0"
                    elif k - steps < len(forecast_tail) and k >= steps:
                        bit = forecast_tail[k - steps]
                        text = bit
                        fill = "#4a3b10" if bit == "1" else "#1e2430"
                        text_color = "#ffcf5c"

                else:
                    if k < len(s.hit_log):
                        ok = s.hit_log[k]
                        text = "✓" if ok else "✗"
                        fill = "#1b5e20" if ok else "#7f1d1d"
                        text_color = "#ffffff"

                canvas.create_rectangle(x, y, x + 22, y + 22,
                                        fill=fill, outline="#38445c")
                canvas.create_text(x + 11, y + 11, text=text, fill=text_color,
                                   font=("Consolas", 9, "bold"))
                x += 26

            y += 38

    def render(self):
        s, o = self.session, self.oracle

        self.draw_registers()
        self.progress["value"] = o.steps if (s.word_ready or o.done) else 0
        self.set_text(self.output_box, s.output_text)

        fc = s.forecast
        if fc is not None and (s.word_ready or o.done):
            low, high = fc["interval"]
            if fc["p_one"] is None:
                bit_line = "all bits revealed"
            else:
                bit_line = f"{fc['next_bit']}  (P(bit=1) = {fc['p_one']:.2f})"

            shown_word = fc["word"] if fc["word"] is not None else "-"
            accuracy = (
                f"{s.word_hits}/{len(s.hit_log)}" if s.hit_log else "0/0"
            )

            self.forecast_var.set(
                f"Word {min(s.pos + (0 if o.done else 1), s.total)}/{s.total} | "
                f"oracle bit {o.steps}/{DEPTH} | pooled op value {o.pooled}\n"
                f"Interval: {low}..{high}\n"
                f"Next-bit forecast: {bit_line}\n"
                f"Forecast word: {shown_word} (index {fc['index']}, "
                f"{fc['word_share'] * 100:.0f}% of interval)\n"
                f"Bit forecast accuracy: this word {accuracy} | "
                f"total {s.total_hits}/{s.total_bits}"
            )
            self.display_candidates()

        if s.finished:
            self.status.set(s.message)
        else:
            partial = f" | completing '{s.partial}'" if (s.partial and s.pos == 0) else ""
            self.status.set(
                f"Predicting word {s.pos + 1}/{s.total}{partial} | "
                f"Generated: {s.output_text}"
            )

    def display_candidates(self):
        s, o = self.session, self.oracle
        fc = s.forecast
        n = len(s.candidates)

        low, high = fc["interval"]
        high = min(high, n - 1)
        count = max(0, high - low + 1)

        lines = [f"Interval {low}..{high} — {count} candidate(s) still possible"]
        final = o.target if o.done else None

        for index in range(low, min(high, low + 59) + 1):
            item = s.candidates[index]
            marker = ""
            if final is not None and index == final:
                marker = "  <-- ORACLE"
            elif final is None and index == fc["index"]:
                marker = "  <-- FORECAST"

            lines.append(
                f'{index:4d} | {item["word"]:<18} | '
                f'p={item["prob"]:.4f} | freq={item["count"]}{marker}'
            )

        if count > 60:
            lines.append(f"... +{count - 60} more")

        self.set_text(self.candidate_text, "\n".join(lines))


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = OracleApplication(root)
    root.mainloop()
