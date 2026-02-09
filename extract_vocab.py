"""
Extract Claude's vocabulary via count_tokens API.

Strategy (4 phases):
1.  Verify tiktoken encodings (cl100k_base, o200k_base, gpt2) — cheap, high yield
2.  Boundary-find on diverse real text — discover tokens Claude actually produces
3.  Unicode block scanning — systematic coverage of CJK, Hangul, Cyrillic, etc.
4.  Targeted extension — extend known tokens by common continuations (if needed)
"""

import json
import time
import random
from pathlib import Path

import tiktoken

from tokenizer import count_tokens, greedy_tokenize, find_boundaries_linear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_single_token(s: str) -> bool:
    """Check if string is a single token."""
    try:
        return count_tokens(s) == 1
    except Exception:
        return False


def load_vocab(vocab_file: str = "vocab_tiktoken.json") -> tuple[set[str], set[str]]:
    """Load verified vocab and checked sets from a checkpoint file."""
    path = Path(vocab_file)
    verified: set[str] = set()
    checked: set[str] = set()
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            verified = set(data.get("verified", []))
            checked = set(data.get("checked", []))
    return verified, checked


def save_vocab(
    verified: set[str],
    checked: set[str],
    vocab_file: str = "vocab_tiktoken.json",
) -> None:
    """Save verified vocab and checked sets."""
    with open(vocab_file, "w") as f:
        json.dump({"verified": sorted(verified), "checked": sorted(checked)}, f)


# ---------------------------------------------------------------------------
# Phase 1 / 1b: Verify tiktoken encodings
# ---------------------------------------------------------------------------

def extract_from_tiktoken(
    encoding_name: str = "cl100k_base",
    output_file: str = "vocab_tiktoken.json",
    rate_limit: float = 0.03,
    checkpoint_interval: int = 1000,
) -> set[str]:
    """
    Verify which tokens from a tiktoken encoding exist in Claude's vocab.

    Works with any tiktoken encoding: cl100k_base, o200k_base, gpt2, etc.
    """
    phase_label = f"tiktoken ({encoding_name})"
    print(f"\n--- Extracting from {phase_label} ---")

    enc = tiktoken.get_encoding(encoding_name)

    # Collect candidate strings
    tiktoken_tokens: list[str] = []
    for i in range(enc.n_vocab):
        try:
            token_bytes = enc.decode_single_token_bytes(i)
            token_str = token_bytes.decode("utf-8", errors="replace")
            if token_str.strip():
                tiktoken_tokens.append(token_str)
        except Exception:
            continue

    print(f"  Loaded {len(tiktoken_tokens)} tokens from {encoding_name}")

    # Load existing progress (shared across encodings)
    verified, checked = load_vocab(output_file)
    print(f"  Existing progress: {len(verified)} verified, {len(checked)} checked")

    # Filter to only unchecked tokens
    to_check = [t for t in tiktoken_tokens if t not in checked]
    print(f"  New candidates to check: {len(to_check)}")

    if not to_check:
        print("  Nothing new to check — skipping.")
        return verified

    start_time = time.time()
    new_found = 0

    for i, token in enumerate(to_check):
        try:
            if is_single_token(token):
                verified.add(token)
                new_found += 1
            checked.add(token)

            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(to_check) - i - 1) / rate / 60
                print(
                    f"    [{encoding_name}] {i+1}/{len(to_check)} "
                    f"({rate:.1f}/sec, ETA {eta:.1f}min) — "
                    f"{len(verified)} verified (+{new_found} new)"
                )
                save_vocab(verified, checked, output_file)

            time.sleep(rate_limit)

        except Exception as e:
            print(f"    Error on token {i}: {e}")
            continue

    save_vocab(verified, checked, output_file)
    print(
        f"  {phase_label} done: {len(verified)} total verified "
        f"(+{new_found} new from this encoding)"
    )
    return verified


# ---------------------------------------------------------------------------
# Phase 1c: HuggingFace tokenizer cross-reference
# ---------------------------------------------------------------------------

HF_MODELS = [
    # --- Highest priority: Claude's own older tokenizer ---
    "Xenova/claude-tokenizer",              # 65K vocab, Claude 1/2 official
    # --- Code-specific tokenizers ---
    "bigcode/starcoder2-3b",                # 49K vocab, code-focused BPE
    "Salesforce/codegen-350M-mono",         # 50K vocab, code-focused
    # --- Large multilingual tokenizers ---
    "bigscience/bloom",                     # 250K vocab, multilingual
    "Qwen/Qwen2.5-0.5B",                   # 152K vocab, CJK-heavy
    "deepseek-ai/deepseek-llm-7b-base",    # 100K vocab
    "tiiuae/falcon-7b",                     # 65K vocab
    "01-ai/Yi-6B",                          # 64K vocab
    "mistralai/Mistral-7B-v0.1",           # 32K vocab
]


def extract_from_huggingface(
    model_names: list[str] | None = None,
    output_file: str = "vocab_tiktoken.json",
    rate_limit: float = 0.03,
    checkpoint_interval: int = 1000,
) -> set[str]:
    """
    Verify which tokens from HuggingFace tokenizers exist in Claude's vocab.

    Downloads each tokenizer, extracts candidate strings, deduplicates,
    filters out already-checked tokens, and verifies against Claude's API.
    Candidates appearing in more tokenizers are checked first.
    """
    print("\n--- Phase 1c: HuggingFace tokenizer cross-reference ---")

    if model_names is None:
        model_names = HF_MODELS

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  ERROR: transformers not installed.")
        print("  Run: pip install transformers sentencepiece protobuf")
        verified, _ = load_vocab(output_file)
        return verified

    # -- Collect candidates from all tokenizers --
    from collections import Counter

    candidate_counts: Counter[str] = Counter()
    claude_source: set[str] = set()  # tokens from Claude's own tokenizer

    for model_name in model_names:
        try:
            print(f"  Loading {model_name}...", end=" ", flush=True)
            tok = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            vocab = tok.get_vocab()
            special = set(tok.all_special_tokens)

            model_tokens: set[str] = set()
            for token_str, token_id in vocab.items():
                if token_str in special:
                    continue

                # Decode token ID to get actual UTF-8 string
                # This correctly handles GPT-2 byte-level BPE and SentencePiece
                try:
                    decoded = tok.decode(
                        [token_id], clean_up_tokenization_spaces=False
                    )
                    # Skip replacement chars (incomplete UTF-8 byte tokens)
                    if decoded and "\ufffd" not in decoded:
                        model_tokens.add(decoded)
                except Exception:
                    pass

            # Remove empty strings
            model_tokens.discard("")

            # Track Claude's own tokenizer candidates for priority
            is_claude = "claude" in model_name.lower()
            if is_claude:
                claude_source.update(model_tokens)

            for t in model_tokens:
                candidate_counts[t] += 1

            print(f"{len(vocab)} vocab -> {len(model_tokens)} candidates")
        except Exception as e:
            print(f"skipped ({str(e)[:80]})")
            continue

    if not candidate_counts:
        print("  No candidates collected — skipping.")
        verified, _ = load_vocab(output_file)
        return verified

    print(f"\n  Total unique candidates across all tokenizers: {len(candidate_counts)}")
    if claude_source:
        print(f"  Of which {len(claude_source)} from Claude 1/2 tokenizer")

    # -- Filter and prioritize --
    verified, checked = load_vocab(output_file)
    print(f"  Existing progress: {len(verified)} verified, {len(checked)} checked")

    # Sort: Claude 1/2 tokens first, then by cross-tokenizer frequency, then shorter
    to_check = [
        t
        for t in sorted(
            candidate_counts,
            key=lambda t: (
                0 if t in claude_source else 1,
                -candidate_counts[t],
                len(t),
            ),
        )
        if t not in checked and t not in verified
    ]
    print(f"  New candidates to check: {len(to_check)}")
    if claude_source:
        claude_unchecked = sum(1 for t in to_check if t in claude_source)
        print(f"  Of which {claude_unchecked} from Claude 1/2 (checked first)")

    if not to_check:
        print("  Nothing new to check — skipping.")
        return verified

    # -- Verify against Claude API --
    start_time = time.time()
    new_found = 0
    window_found = 0  # tokens found in current 2000-check window

    for i, token in enumerate(to_check):
        try:
            if is_single_token(token):
                verified.add(token)
                new_found += 1
                window_found += 1
            checked.add(token)

            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(to_check) - i - 1) / rate / 60
                hit_rate = new_found / (i + 1) * 100
                print(
                    f"    [{i+1}/{len(to_check)}] "
                    f"({rate:.1f}/sec, ETA {eta:.0f}min) — "
                    f"{len(verified)} verified (+{new_found} new, "
                    f"{hit_rate:.1f}% hit rate)"
                )
                save_vocab(verified, checked, output_file)

            # Early stopping: if hit rate drops below 0.5% over 2000 checks
            if (i + 1) % 2000 == 0:
                window_rate = window_found / 2000 * 100
                if window_rate < 0.5 and i > 5000:
                    print(
                        f"    Hit rate dropped to {window_rate:.2f}% "
                        f"(last 2000 checks) — stopping early."
                    )
                    save_vocab(verified, checked, output_file)
                    break
                window_found = 0

            time.sleep(rate_limit)
        except KeyboardInterrupt:
            print("\n  Interrupted — saving progress...")
            save_vocab(verified, checked, output_file)
            return verified
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                print(f"    Rate limited, waiting 5s...")
                time.sleep(5)
            continue

    save_vocab(verified, checked, output_file)
    print(
        f"\n  Phase 1c done: {len(verified)} total verified "
        f"(+{new_found} new from HuggingFace tokenizers)"
    )
    return verified


# ---------------------------------------------------------------------------
# Phase 1d: Common English words (Claude prefers whole-word tokens)
# ---------------------------------------------------------------------------

def extract_common_words(
    output_file: str = "vocab_tiktoken.json",
    rate_limit: float = 0.03,
    checkpoint_interval: int = 1000,
) -> set[str]:
    """
    Check common English words as potential single tokens.

    Claude's tokenizer is known to encode most of the top 10K English words
    as single tokens (with space prefix). Generate candidates:
      - " word" (space-prefixed, how words appear in context)
      - "word" (raw)
      - " Word" (capitalized)
      - "Word" (capitalized raw)
    """
    print("\n--- Phase 1d: Common English word candidates ---")

    # Try to get a word frequency list
    word_list: list[str] = []

    # Strategy: extract words from our training data
    for data_file in ["training_data.jsonl", "diverse_data.jsonl"]:
        path = Path(data_file)
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    # Extract alphabetic words
                    for word in text.split():
                        cleaned = word.strip(".,;:!?\"'()[]{}/-")
                        if cleaned.isalpha() and 2 <= len(cleaned) <= 20:
                            word_list.append(cleaned.lower())
                except (json.JSONDecodeError, KeyError):
                    continue

    # Add common programming identifiers and English words
    common_extras = [
        # Top English words (many likely already found, but ensure coverage)
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "should", "been", "each", "much",
        "between", "being", "under", "never", "every", "same", "another",
        "however", "through", "while", "where", "still", "before", "here",
        "does", "many", "both", "own", "part", "right", "without", "end",
        # Common programming words
        "function", "return", "class", "import", "export", "default",
        "const", "string", "number", "boolean", "null", "undefined",
        "true", "false", "error", "data", "value", "result", "input",
        "output", "file", "path", "name", "type", "list", "dict", "set",
        "array", "object", "method", "property", "index", "length",
        "count", "size", "key", "item", "node", "element", "text",
        "message", "request", "response", "status", "code", "test",
        "config", "option", "param", "query", "model", "schema",
        "table", "column", "field", "record", "database", "server",
        "client", "user", "token", "session", "async", "await",
        "promise", "callback", "event", "handler", "listener",
        "component", "render", "state", "props", "context", "hook",
        "module", "package", "version", "build", "deploy", "docker",
        "container", "service", "endpoint", "route", "middleware",
        "authentication", "authorization", "permission", "role",
        "algorithm", "implementation", "interface", "abstract",
        "instance", "constructor", "prototype", "inheritance",
        # Common suffixes and morphological variants
        "ing", "tion", "ment", "ness", "able", "ible", "ful", "less",
        "ous", "ive", "ical", "ally", "ized", "izing", "ation",
    ]
    word_list.extend(common_extras)

    # Deduplicate and generate variants
    unique_words = sorted(set(word_list))
    candidates: list[str] = []
    for word in unique_words:
        candidates.append(" " + word)          # " hello" (space-prefixed)
        candidates.append(word)                 # "hello"
        candidates.append(" " + word.capitalize())  # " Hello"
        candidates.append(word.capitalize())    # "Hello"
        if word != word.upper():
            candidates.append(" " + word.upper())   # " HELLO"

    # Deduplicate
    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    candidates = deduped

    print(f"  Generated {len(candidates)} word candidates from {len(unique_words)} unique words")

    # Filter against already-checked
    verified, checked = load_vocab(output_file)
    to_check = [c for c in candidates if c not in checked and c not in verified]
    print(f"  New candidates to check: {len(to_check)}")

    if not to_check:
        print("  Nothing new to check — skipping.")
        return verified

    start_time = time.time()
    new_found = 0

    for i, token in enumerate(to_check):
        try:
            if is_single_token(token):
                verified.add(token)
                new_found += 1
            checked.add(token)

            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(to_check) - i - 1) / rate / 60
                hit_rate = new_found / (i + 1) * 100
                print(
                    f"    [{i+1}/{len(to_check)}] "
                    f"({rate:.1f}/sec, ETA {eta:.0f}min) — "
                    f"{len(verified)} verified (+{new_found} new, "
                    f"{hit_rate:.1f}% hit rate)"
                )
                save_vocab(verified, checked, output_file)

            time.sleep(rate_limit)
        except KeyboardInterrupt:
            print("\n  Interrupted — saving progress...")
            save_vocab(verified, checked, output_file)
            return verified
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                print(f"    Rate limited, waiting 5s...")
                time.sleep(5)
            continue

    save_vocab(verified, checked, output_file)
    print(
        f"\n  Phase 1d done: {len(verified)} total verified "
        f"(+{new_found} new from common words)"
    )
    return verified


# ---------------------------------------------------------------------------
# Phase 2: Boundary-find on real text
# ---------------------------------------------------------------------------

def smart_boundary_find(text: str, known_vocab: set[str]) -> list[str]:
    """
    Boundary-find only the spans of *text* not covered by known vocab.

    1. Greedy-tokenize locally with known_vocab.
    2. Check overall API count — if it matches, we're done.
    3. Otherwise, find the spans where local tokens are wrong and
       boundary-find just those spans.

    Returns newly-discovered tokens (strings that are single tokens
    but were not in known_vocab).
    """
    if not text or not text.strip():
        return []

    local_tokens = greedy_tokenize(text, known_vocab)
    local_count = len(local_tokens)

    try:
        api_count = count_tokens(text)
    except Exception:
        return []

    if local_count == api_count:
        return []  # Local vocab already covers this text perfectly

    # Mismatch — boundary-find the full text to discover new tokens
    try:
        boundaries = find_boundaries_linear(text)
    except Exception:
        return []

    new_tokens: list[str] = []
    for i in range(len(boundaries) - 1):
        token = text[boundaries[i] : boundaries[i + 1]]
        if token and token not in known_vocab and token.strip():
            # Verify it's really a single token
            try:
                if count_tokens(token) == 1:
                    new_tokens.append(token)
            except Exception:
                pass

    return new_tokens


def extract_from_text_corpus(
    texts: list[str],
    output_file: str = "vocab_tiktoken.json",
    rate_limit: float = 0.03,
    checkpoint_interval: int = 100,
) -> set[str]:
    """
    Phase 2: Discover tokens by boundary-finding real text.

    Uses skip-match optimization: only boundary-find texts where the local
    greedy tokenizer disagrees with the API count.
    """
    print("\n--- Phase 2: Boundary-find on text corpus ---")

    verified, checked = load_vocab(output_file)
    print(f"  Starting vocab size: {len(verified)}")

    start_time = time.time()
    texts_processed = 0
    texts_skipped = 0
    new_found = 0

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue

        # Truncate very long texts to keep API costs reasonable
        text = text[:500]

        new_tokens = smart_boundary_find(text, verified)

        if new_tokens:
            for tok in new_tokens:
                verified.add(tok)
                new_found += 1
            texts_processed += 1
        else:
            texts_skipped += 1

        if (i + 1) % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (len(texts) - i - 1) / rate / 60 if rate else 0
            print(
                f"    Text {i+1}/{len(texts)} "
                f"({rate:.1f}/sec, ETA {eta:.1f}min) — "
                f"vocab {len(verified)} (+{new_found} new), "
                f"skipped {texts_skipped}"
            )
            save_vocab(verified, checked, output_file)

        time.sleep(rate_limit)

    save_vocab(verified, checked, output_file)
    print(
        f"  Phase 2 done: {len(verified)} total verified "
        f"(+{new_found} new from text corpus)"
    )
    return verified


# ---------------------------------------------------------------------------
# Phase 3: Unicode block scanning
# ---------------------------------------------------------------------------

UNICODE_BLOCKS: dict[str, list[tuple[int, int]]] = {
    "CJK Unified": [(0x4E00, 0x9FFF)],
    "CJK Extension A": [(0x3400, 0x4DBF)],
    "Hangul Syllables": [(0xAC00, 0xD7AF)],
    "Cyrillic": [(0x0400, 0x04FF)],
    "Arabic": [(0x0600, 0x06FF)],
    "Devanagari": [(0x0900, 0x097F)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Hiragana": [(0x3040, 0x309F)],
    "Katakana": [(0x30A0, 0x30FF)],
    "Greek": [(0x0370, 0x03FF)],
    "Hebrew": [(0x0590, 0x05FF)],
    "Latin Extended": [(0x0080, 0x024F)],
    "Emoji Misc Symbols": [(0x2600, 0x26FF)],
    "Emoji Dingbats": [(0x2700, 0x27BF)],
    "Emoji Emoticons": [(0x1F600, 0x1F64F)],
    "Emoji Misc Symbols & Pictographs": [(0x1F300, 0x1F5FF)],
    "Emoji Transport & Map": [(0x1F680, 0x1F6FF)],
    "Math Symbols": [(0x2200, 0x22FF)],
    "Fullwidth Latin": [(0xFF01, 0xFF5E)],
    "CJK Symbols & Punctuation": [(0x3000, 0x303F)],
}


def scan_unicode_blocks(
    output_file: str = "vocab_tiktoken.json",
    rate_limit: float = 0.03,
    checkpoint_interval: int = 500,
    sample_limit_per_block: int = 2000,
) -> set[str]:
    """
    Phase 3: Systematically check Unicode characters and common bigrams.
    """
    print("\n--- Phase 3: Unicode block scanning ---")

    verified, checked = load_vocab(output_file)
    print(f"  Starting vocab size: {len(verified)}")

    start_time = time.time()
    total_checked = 0
    new_found = 0

    for block_name, ranges in UNICODE_BLOCKS.items():
        block_new = 0

        for range_start, range_end in ranges:
            # Collect candidate chars from the range
            chars: list[str] = []
            for cp in range(range_start, range_end + 1):
                try:
                    ch = chr(cp)
                    chars.append(ch)
                except (ValueError, OverflowError):
                    continue

            # Sample if block is very large
            if len(chars) > sample_limit_per_block:
                chars = random.sample(chars, sample_limit_per_block)

            # Check single characters
            for ch in chars:
                if ch in checked:
                    continue

                try:
                    if is_single_token(ch):
                        verified.add(ch)
                        new_found += 1
                        block_new += 1
                    checked.add(ch)
                    total_checked += 1

                    if total_checked % checkpoint_interval == 0:
                        elapsed = time.time() - start_time
                        rate = total_checked / elapsed if elapsed else 0
                        print(
                            f"    [{block_name}] checked {total_checked}, "
                            f"vocab {len(verified)} (+{new_found} new), "
                            f"{rate:.1f}/sec"
                        )
                        save_vocab(verified, checked, output_file)

                    time.sleep(rate_limit)
                except Exception:
                    continue

            # Check common bigrams within this block (sample)
            if len(chars) > 1:
                bigram_candidates = []
                sample_chars = chars[:200] if len(chars) > 200 else chars
                for c1 in sample_chars[:50]:
                    for c2 in sample_chars[:50]:
                        bigram = c1 + c2
                        if bigram not in checked:
                            bigram_candidates.append(bigram)

                # Limit bigram checks
                if len(bigram_candidates) > 500:
                    bigram_candidates = random.sample(bigram_candidates, 500)

                for bigram in bigram_candidates:
                    try:
                        if is_single_token(bigram):
                            verified.add(bigram)
                            new_found += 1
                            block_new += 1
                        checked.add(bigram)
                        total_checked += 1
                        time.sleep(rate_limit)
                    except Exception:
                        continue

        print(f"  Block '{block_name}': +{block_new} tokens")

    save_vocab(verified, checked, output_file)
    print(
        f"  Phase 3 done: {len(verified)} total verified "
        f"(+{new_found} new from unicode scanning)"
    )
    return verified


# ---------------------------------------------------------------------------
# Phase 4: Targeted extension (only if coverage < 95%)
# ---------------------------------------------------------------------------

# Most common continuation bytes in English/code text
COMMON_CONTINUATIONS = list(
    b" etaoinsrhldcumfpgwybvkxjqzETAOINSRHLDCUMFPGWYBVKXJQZ"
    b"0123456789.,;:!?'\"-_()/\\@#$%^&*+=<>{}[]|~`\n\t\r"
)


def extend_known_tokens(
    output_file: str = "vocab_tiktoken.json",
    rate_limit: float = 0.03,
    checkpoint_interval: int = 1000,
    max_token_len: int = 50,
) -> set[str]:
    """
    Phase 4: Extend known tokens by common continuation bytes.

    Only extend by the top ~100 most likely continuations (not all 256).
    """
    print("\n--- Phase 4: Targeted extension ---")

    verified, checked = load_vocab(output_file)
    print(f"  Starting vocab size: {len(verified)}")

    start_time = time.time()
    new_found = 0
    total_checked = 0

    # Sort tokens shortest-first so extensions build incrementally
    tokens_to_extend = sorted(verified, key=len)

    for token in tokens_to_extend:
        if len(token) >= max_token_len:
            continue

        for byte_val in COMMON_CONTINUATIONS:
            try:
                ch = chr(byte_val) if isinstance(byte_val, int) else byte_val
                candidate = token + ch
            except (ValueError, TypeError):
                continue

            if candidate in checked:
                continue

            try:
                if is_single_token(candidate):
                    verified.add(candidate)
                    new_found += 1
                checked.add(candidate)
                total_checked += 1

                if total_checked % checkpoint_interval == 0:
                    elapsed = time.time() - start_time
                    rate = total_checked / elapsed if elapsed else 0
                    print(
                        f"    Checked {total_checked}, "
                        f"vocab {len(verified)} (+{new_found} new), "
                        f"{rate:.1f}/sec"
                    )
                    save_vocab(verified, checked, output_file)

                time.sleep(rate_limit)
            except Exception:
                continue

    save_vocab(verified, checked, output_file)
    print(
        f"  Phase 4 done: {len(verified)} total verified "
        f"(+{new_found} new from extensions)"
    )
    return verified


# ---------------------------------------------------------------------------
# Coverage measurement
# ---------------------------------------------------------------------------

def measure_coverage(
    vocab: set[str],
    test_texts: list[str],
    max_texts: int = 1000,
) -> float:
    """
    Measure what fraction of test texts our vocab covers perfectly.

    "Covers" means greedy_tokenize count == API count.
    Returns fraction in [0, 1].
    """
    if not test_texts:
        return 0.0

    texts = test_texts[:max_texts]
    matches = 0
    errors = 0

    for text in texts:
        if not text or not text.strip():
            continue
        try:
            local_count = len(greedy_tokenize(text, vocab))
            api_count = count_tokens(text)
            if local_count == api_count:
                matches += 1
        except Exception:
            errors += 1

    total = len(texts) - errors
    coverage = matches / total if total > 0 else 0.0
    print(
        f"  Coverage: {matches}/{total} texts match "
        f"({coverage:.1%}), {errors} errors"
    )
    return coverage


# ---------------------------------------------------------------------------
# Build local tokenizer from extracted vocab
# ---------------------------------------------------------------------------

def build_tokenizer(vocab_file: str = "vocab_tiktoken.json"):
    """Build a local tokenizer from extracted vocabulary."""
    verified, _ = load_vocab(vocab_file)
    sorted_vocab = sorted(verified, key=len, reverse=True)

    def tokenize(text: str) -> list[str]:
        tokens = []
        pos = 0
        while pos < len(text):
            matched = False
            for token in sorted_vocab:
                if text[pos:].startswith(token):
                    tokens.append(token)
                    pos += len(token)
                    matched = True
                    break
            if not matched:
                tokens.append(text[pos])
                pos += 1
        return tokens

    def count(text: str) -> int:
        return len(tokenize(text))

    return tokenize, count


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def load_text_corpus() -> list[str]:
    """Load diverse text corpus for Phase 2. Tries multiple sources."""
    texts: list[str] = []

    # 1. Existing training data
    training_file = Path("training_data.jsonl")
    if training_file.exists():
        with open(training_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    t = data.get("text", "")
                    if t and t.strip() and len(t) > 10:
                        texts.append(t)
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {len(texts)} texts from training_data.jsonl")

    # 2. Diverse data if available
    diverse_file = Path("diverse_data.jsonl")
    if diverse_file.exists():
        count_before = len(texts)
        with open(diverse_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    t = data.get("text", "")
                    if t and t.strip() and len(t) > 10:
                        texts.append(t)
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {len(texts) - count_before} texts from diverse_data.jsonl")

    if not texts:
        print("  No text corpus found. Run collect_data.py or collect_diverse_data.py first.")

    return texts


def main():
    print("=" * 60)
    print("Claude Vocabulary Extraction — Multi-Phase")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Phase 1: cl100k_base (likely partially done already)
    # ---------------------------------------------------------------
    print("\n[Phase 1] tiktoken cl100k_base verification")
    vocab = extract_from_tiktoken("cl100k_base")

    # ---------------------------------------------------------------
    # Phase 1b: Cross-reference o200k_base + gpt2
    # ---------------------------------------------------------------
    print("\n[Phase 1b] tiktoken o200k_base + gpt2 cross-reference")
    vocab = extract_from_tiktoken("o200k_base")
    vocab = extract_from_tiktoken("gpt2")

    # Checkpoint coverage
    print(f"\nAfter Phase 1/1b: {len(vocab)} verified tokens")

    # ---------------------------------------------------------------
    # Phase 1c: Cross-reference HuggingFace tokenizers
    # ---------------------------------------------------------------
    print("\n[Phase 1c] HuggingFace tokenizer cross-reference")
    vocab = extract_from_huggingface()
    print(f"\nAfter Phase 1c: {len(vocab)} verified tokens")

    # ---------------------------------------------------------------
    # Phase 1d: Common English words
    # ---------------------------------------------------------------
    print("\n[Phase 1d] Common English word candidates")
    vocab = extract_common_words()
    print(f"\nAfter Phase 1d: {len(vocab)} verified tokens")

    # ---------------------------------------------------------------
    # Phase 2: Boundary-find on real text (SKIPPED — too expensive)
    # ---------------------------------------------------------------
    print("\n[Phase 2] Skipped (boundary-find too expensive for yield)")

    # ---------------------------------------------------------------
    # Phase 3: Unicode block scanning (SKIPPED — covered by tokenizer cross-refs)
    # ---------------------------------------------------------------
    print("\n[Phase 3] Skipped (unicode blocks already covered by tokenizer cross-refs)")

    # ---------------------------------------------------------------
    # Coverage check — decide whether Phase 4 is needed
    # ---------------------------------------------------------------
    print("\n--- Coverage check ---")
    test_texts = load_text_corpus()
    if test_texts:
        coverage = measure_coverage(vocab, random.sample(test_texts, min(200, len(test_texts))))
        if coverage >= 0.95:
            print(f"Coverage {coverage:.1%} >= 95% — skipping Phase 4.")
        else:
            print(f"Coverage {coverage:.1%} < 95% — running Phase 4.")
            vocab = extend_known_tokens()
            # Re-measure
            coverage = measure_coverage(
                vocab,
                random.sample(test_texts, min(200, len(test_texts))),
            )
            print(f"Final coverage: {coverage:.1%}")
    else:
        print("  No test texts available for coverage check.")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Extraction complete! Total vocabulary: {len(vocab)} tokens")
    print("=" * 60)

    # Quick sanity test
    print("\nSanity test (local tokenizer vs API):")
    tokenize_fn, count_fn = build_tokenizer()
    test_cases = ["hello world", "The quick brown fox", "def foo():", "import numpy as np"]
    for text in test_cases:
        local = count_fn(text)
        api = count_tokens(text)
        match = "OK" if local == api else "MISMATCH"
        print(f"  '{text}': local={local}, api={api} [{match}]")


if __name__ == "__main__":
    main()
