"""
Unicode cross-reference: extract Unicode token candidates from HuggingFace
tokenizers and verify them against Claude's API.

This runs AFTER reverify_vocab.py has cleaned up false positives.
Uses the fixed § sandwich count_tokens method.
"""

import json
import time
import sys
from pathlib import Path
from collections import Counter

from tokenizer import count_tokens


# Unicode-heavy tokenizers — prioritize multilingual models
UNICODE_MODELS = [
    # --- CJK-heavy ---
    "THUDM/glm-4-9b-chat",                 # 151K vocab, CJK-heavy
    "baichuan-inc/Baichuan2-7B-Base",       # 125K vocab, CJK-heavy
    "Qwen/Qwen2.5-0.5B",                   # 152K vocab, CJK-heavy
    "01-ai/Yi-6B",                          # 64K vocab, CJK-heavy
    # --- Massive multilingual ---
    "bigscience/bloom",                     # 250K vocab
    "facebook/mbart-large-50",              # 250K vocab, 50 languages
    "google/mt5-base",                      # 250K vocab, 101 languages
    # --- Other multilingual ---
    "deepseek-ai/deepseek-llm-7b-base",    # 100K vocab
    "Xenova/claude-tokenizer",              # 65K vocab, Claude 1/2
    # --- Code tokenizers (may have some unicode) ---
    "bigcode/starcoder2-3b",                # 49K vocab
    "mistralai/Mistral-7B-v0.1",           # 32K vocab
]


def load_vocab(vocab_file: str = "vocab_tiktoken.json") -> tuple[set[str], set[str]]:
    path = Path(vocab_file)
    data = json.load(open(path))
    return set(data.get("verified", [])), set(data.get("checked", []))


def save_vocab(verified: set[str], checked: set[str], vocab_file: str = "vocab_tiktoken.json"):
    with open(vocab_file, "w") as f:
        json.dump({"verified": sorted(verified), "checked": sorted(checked)}, f)


def is_single_token(s: str) -> bool:
    try:
        return count_tokens(s) == 1
    except Exception:
        return False


def extract_unicode_candidates(model_names: list[str] | None = None) -> list[str]:
    """Extract all Unicode token candidates from HuggingFace tokenizers."""
    if model_names is None:
        model_names = UNICODE_MODELS

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed.")
        return []

    candidate_counts: Counter[str] = Counter()

    for model_name in model_names:
        try:
            print(f"Loading {model_name}...", end=" ", flush=True)
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            vocab = tok.get_vocab()
            special = set(tok.all_special_tokens)

            model_candidates = set()
            for token_str, token_id in vocab.items():
                if token_str in special:
                    continue
                try:
                    decoded = tok.decode([token_id], clean_up_tokenization_spaces=False)
                    if not decoded or "\ufffd" in decoded:
                        continue
                    # Only keep candidates with at least one non-ASCII char
                    if any(ord(c) >= 128 for c in decoded):
                        model_candidates.add(decoded)
                except Exception:
                    pass

            model_candidates.discard("")

            for t in model_candidates:
                candidate_counts[t] += 1

            print(f"{len(vocab)} vocab -> {len(model_candidates)} unicode candidates")
        except Exception as e:
            print(f"skipped ({str(e)[:80]})")
            continue

    print(f"\nTotal unique unicode candidates: {len(candidate_counts)}")

    # Sort by frequency (tokens appearing in more tokenizers are more likely to be in Claude's)
    sorted_candidates = sorted(
        candidate_counts.keys(),
        key=lambda t: (-candidate_counts[t], len(t)),
    )
    return sorted_candidates


def verify_candidates(
    candidates: list[str],
    vocab_file: str = "vocab_tiktoken.json",
    checkpoint_interval: int = 1000,
):
    """Verify Unicode candidates against Claude's API."""
    verified, checked = load_vocab(vocab_file)
    print(f"Starting: {len(verified)} verified, {len(checked)} checked")

    to_check = [t for t in candidates if t not in checked and t not in verified]
    print(f"New candidates to check: {len(to_check)}")

    if not to_check:
        print("Nothing new to check.")
        return

    start = time.time()
    new_found = 0
    window_found = 0

    for i, token in enumerate(to_check):
        try:
            if is_single_token(token):
                verified.add(token)
                new_found += 1
                window_found += 1
            checked.add(token)

            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(to_check) - i - 1) / rate / 60
                hit_rate = new_found / (i + 1) * 100
                print(
                    f"  [{i+1}/{len(to_check)}] ({rate:.1f}/sec, ETA {eta:.0f}min) "
                    f"verified={len(verified)} (+{new_found} new, {hit_rate:.1f}% hit)"
                )
                save_vocab(verified, checked, vocab_file)

            # Early stopping check every 5000
            if (i + 1) % 5000 == 0:
                window_rate = window_found / 5000 * 100
                if window_rate < 0.3 and i > 10000:
                    print(f"  Hit rate dropped to {window_rate:.2f}% — stopping early.")
                    break
                window_found = 0

            time.sleep(0.03)

        except KeyboardInterrupt:
            print("\nInterrupted — saving...")
            save_vocab(verified, checked, vocab_file)
            return
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                print(f"  Rate limited, waiting 5s...")
                time.sleep(5)
            continue

    save_vocab(verified, checked, vocab_file)
    print(f"\nDone: {len(verified)} verified (+{new_found} new unicode tokens)")


def main():
    print("=" * 60)
    print("Unicode Cross-Reference — Fixed § Sandwich Method")
    print("=" * 60)

    # Step 1: Collect candidates from all tokenizers
    print("\n--- Step 1: Collecting unicode candidates ---")
    candidates = extract_unicode_candidates()

    if not candidates:
        print("No candidates found. Exiting.")
        return

    # Step 2: Verify against Claude API
    print("\n--- Step 2: Verifying candidates ---")
    verify_candidates(candidates)


if __name__ == "__main__":
    main()
