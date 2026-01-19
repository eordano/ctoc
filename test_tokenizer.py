#!/usr/bin/env python3
"""
Test Claude's tokenizer behavior via count_tokens API.

Key experiment: Is BPE prefix count monotonic?
Spoiler: NO! "hel" = 2 tokens, "hell" = 1 token.
"""

import time
from tokenizer import count_tokens, get_baseline, tokenize, is_boundary


def test_baseline():
    """Check the baseline overhead."""
    baseline = get_baseline()
    print(f"Baseline (message overhead): {baseline} tokens")
    return baseline


def test_monotonicity(test_string: str):
    """Test if count(prefix) is monotonically non-decreasing."""
    print(f"\n=== Monotonicity Test: '{test_string}' ===")

    prev_count = 0
    violations = []

    for i in range(1, len(test_string) + 1):
        prefix = test_string[:i]
        c = count_tokens(prefix)

        if c < prev_count:
            violations.append((i, prefix, prev_count, c))
            marker = f" ** VIOLATION: {prev_count} -> {c}"
        elif c > prev_count:
            marker = f" +{c - prev_count}"
        else:
            marker = ""

        print(f"  [{i:2d}] '{prefix}' -> {c} tokens{marker}")
        prev_count = c
        time.sleep(0.02)

    if violations:
        print(f"\n  Found {len(violations)} monotonicity violations!")
        print("  This means binary search won't work for token boundaries.")
    else:
        print("\n  No violations - monotonicity holds for this string.")

    return violations


def test_boundary_property(s: str):
    """Test: count(prefix) + count(suffix) == count(full) at boundaries."""
    print(f"\n=== Boundary Property Test: '{s}' ===")

    total = count_tokens(s)
    print(f"Total tokens: {total}")

    boundaries = []
    for i in range(len(s) + 1):
        prefix = s[:i]
        suffix = s[i:]

        # Skip whitespace-only for clean output
        if prefix and not prefix.strip():
            continue
        if suffix and not suffix.strip():
            continue

        prefix_count = count_tokens(prefix) if prefix else 0
        suffix_count = count_tokens(suffix) if suffix else 0
        sum_count = prefix_count + suffix_count
        is_bound = sum_count == total

        if is_bound:
            boundaries.append(i)
            marker = " <-- BOUNDARY"
        else:
            marker = ""

        print(f"  [{i:2d}] '{prefix}' | '{suffix}' -> {prefix_count} + {suffix_count} = {sum_count}{marker}")
        time.sleep(0.02)

    # Extract tokens from boundaries
    tokens = [s[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]
    print(f"\nTokens: {tokens}")
    print(f"Count: {len(tokens)} (expected: {total})")

    return boundaries, tokens


def test_single_token_detection():
    """Test detecting single tokens."""
    print("\n=== Single Token Detection ===")

    # Test various strings to see which are single tokens
    test_cases = [
        "a", "ab", "abc",
        "hello", "hell", "hel", "he", "h",
        "the", "th", "t",
        "world", "worl", "wor", "wo", "w",
        " the", " hello",  # With leading space
        "def", "import", "return",
        "1234", "123", "12", "1",
    ]

    for s in test_cases:
        c = count_tokens(s)
        is_single = "" if c == 1 else f" ({c} tokens)"
        print(f"  '{s}' -> {c} token{'s' if c != 1 else ''}{is_single}")
        time.sleep(0.02)


def test_tokenization(test_cases: list[str]):
    """Test our tokenization against the API count."""
    print("\n=== Tokenization Test ===")

    for s in test_cases:
        tokens = tokenize(s)
        actual = count_tokens(s)
        match = "" if len(tokens) == actual else f" (expected {actual})"

        print(f"  '{s[:40]}{'...' if len(s) > 40 else ''}'")
        print(f"    -> {tokens}")
        print(f"    -> {len(tokens)} tokens{match}")
        time.sleep(0.1)


def main():
    print("=" * 60)
    print("Claude Tokenizer Reverse Engineering")
    print("=" * 60)

    test_baseline()

    # Key finding: monotonicity doesn't hold
    test_monotonicity("hello")
    test_monotonicity("hello world")
    test_monotonicity("The quick brown fox")

    # Single token detection
    test_single_token_detection()

    # Boundary property
    test_boundary_property("hello world")

    # Tokenization
    test_cases = [
        "hello",
        "hello world",
        "The quick brown fox",
        "def foo():",
        "import numpy as np",
    ]
    test_tokenization(test_cases)


if __name__ == "__main__":
    main()
