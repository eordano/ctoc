#!/usr/bin/env python3
"""
Extract Claude's vocabulary by systematically probing count_tokens.

Strategy:
1. Start with all single bytes (256 candidates)
2. For each confirmed token, try extending with each byte
3. If count(extended) == 1, it's a valid token
4. BFS through token space until no new tokens are found
"""

import json
import time
from collections import deque
from tokenizer import count_tokens


def extract_vocabulary(
    max_token_len: int = 50,
    rate_limit_delay: float = 0.02,
    checkpoint_file: str = "vocab_checkpoint.json",
    checkpoint_interval: int = 1000,
) -> set[bytes]:
    """
    Extract vocabulary using BFS through token space.

    Returns set of byte sequences that are valid tokens.
    """
    vocab: set[bytes] = set()
    visited: set[bytes] = set()
    queue: deque[bytes] = deque()

    # Start with all single bytes
    for b in range(256):
        queue.append(bytes([b]))

    api_calls = 0
    start_time = time.time()

    while queue:
        candidate = queue.popleft()

        if candidate in visited:
            continue
        visited.add(candidate)

        # Check if this is a single token
        try:
            # Convert bytes to string for API (handling encoding)
            try:
                text = candidate.decode('utf-8')
            except UnicodeDecodeError:
                # Skip invalid UTF-8 sequences for now
                continue

            c = count_tokens(text)
            api_calls += 1

            if c == 1:
                vocab.add(candidate)
                print(f"[{len(vocab):5d}] Found token: {candidate!r}")

                # Try extending with each byte
                if len(candidate) < max_token_len:
                    for b in range(256):
                        extended = candidate + bytes([b])
                        if extended not in visited:
                            queue.append(extended)

            # Rate limiting
            time.sleep(rate_limit_delay)

            # Checkpoint
            if api_calls % checkpoint_interval == 0:
                save_checkpoint(vocab, visited, checkpoint_file)
                elapsed = time.time() - start_time
                rate = api_calls / elapsed if elapsed > 0 else 0
                print(f"  ... {api_calls} API calls, {len(vocab)} tokens, {rate:.1f} calls/sec")

        except Exception as e:
            print(f"  Error on {candidate!r}: {e}")
            continue

    return vocab


def save_checkpoint(vocab: set[bytes], visited: set[bytes], filename: str):
    """Save progress to file."""
    data = {
        "vocab": [b.hex() for b in vocab],
        "visited": [b.hex() for b in visited],
    }
    with open(filename, "w") as f:
        json.dump(data, f)


def load_checkpoint(filename: str) -> tuple[set[bytes], set[bytes]]:
    """Load progress from file."""
    with open(filename, "r") as f:
        data = json.load(f)
    vocab = {bytes.fromhex(h) for h in data["vocab"]}
    visited = {bytes.fromhex(h) for h in data["visited"]}
    return vocab, visited


def main():
    print("=" * 60)
    print("Claude Vocabulary Extraction")
    print("=" * 60)
    print()
    print("This will make many API calls to discover the vocabulary.")
    print("Progress will be checkpointed to vocab_checkpoint.json")
    print()

    # Start with a small exploration
    vocab = extract_vocabulary(
        max_token_len=20,  # Start conservative
        rate_limit_delay=0.05,
    )

    print(f"\nExtracted {len(vocab)} tokens")

    # Save final vocabulary
    vocab_list = sorted([v.decode('utf-8', errors='replace') for v in vocab])
    with open("vocab.json", "w") as f:
        json.dump(vocab_list, f, indent=2, ensure_ascii=False)

    print(f"Saved to vocab.json")


if __name__ == "__main__":
    main()
