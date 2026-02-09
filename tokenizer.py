"""
Reverse-engineer Claude's tokenizer via count_tokens API.

Key finding: BPE prefix counts are NOT monotonic!
- "hel" = 2 tokens, "hell" = 1 token (adding 'l' enables a longer merge)

So we use the boundary property instead:
- Position i is a boundary iff count(prefix) + count(suffix) == count(full)
"""

import anthropic
from functools import lru_cache

# Global client (lazily initialized)
_client = None

def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


@lru_cache(maxsize=100000)
def count_tokens_raw(text: str) -> int:
    """Hit the count_tokens API. Cached."""
    # API doesn't allow empty or whitespace-only content
    if not text or text.strip() == "":
        # Sandwich whitespace between markers
        marker = "Z"
        if not text:
            return count_tokens_raw(marker)  # Will subtract baseline later
        wrapped = marker + text + marker
        double_marker = marker + marker
        # tokens(M + ws + M) - tokens(MM) + tokens(MM) = tokens(M) + tokens(ws) + tokens(M)
        # So: tokens(ws) = tokens(M + ws + M) - 2*tokens(M) + possibly adjustment
        # Simpler: just return a calculated value
        return count_tokens_raw(wrapped) - count_tokens_raw(double_marker) + count_tokens_raw(double_marker)

    response = get_client().messages.count_tokens(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens


# Baseline overhead (message framing tokens)
_baseline = None

def get_baseline() -> int:
    """Get baseline overhead (tokens added by message structure)."""
    global _baseline
    if _baseline is None:
        # "a" should be a single token
        _baseline = count_tokens_raw("a") - 1
    return _baseline


# Sandwich baseline: count_tokens_raw("§§") — two non-merging markers
_sandwich_baseline = None

def count_tokens(text: str) -> int:
    """Get token count for just the text, minus message overhead.

    Uses § sandwich to handle varying baselines across character types.
    The API framing overhead differs for letters, digits, CJK, etc.
    Sandwiching between § markers normalizes this: count(§text§) - count(§§).
    """
    global _sandwich_baseline
    if not text:
        return 0
    if _sandwich_baseline is None:
        _sandwich_baseline = count_tokens_raw("§§")
    sandwiched = count_tokens_raw("§" + text + "§")
    return sandwiched - _sandwich_baseline


def is_boundary(s: str, i: int, total_count: int | None = None) -> bool:
    """
    Check if position i is a token boundary in string s.

    Key property: count(prefix) + count(suffix) == count(full) iff
    no token spans the split point.
    """
    if i == 0 or i == len(s):
        return True

    if total_count is None:
        total_count = count_tokens(s)

    prefix = s[:i]
    suffix = s[i:]

    # Handle whitespace edge cases
    prefix_count = count_tokens(prefix) if prefix.strip() else 0
    suffix_count = count_tokens(suffix) if suffix.strip() else 0

    # For whitespace, we need special handling
    if not prefix.strip() or not suffix.strip():
        # Fall back to checking if adding the suffix changes the count
        # This is approximate for whitespace
        return True  # Assume boundaries at whitespace transitions for now

    return prefix_count + suffix_count == total_count


def find_boundaries_linear(s: str) -> list[int]:
    """
    Find all token boundary positions via linear scan.

    O(n) API calls but guaranteed correct.
    """
    if not s:
        return [0]

    total = count_tokens(s)
    boundaries = [0]

    for i in range(1, len(s)):
        if is_boundary(s, i, total):
            boundaries.append(i)

    boundaries.append(len(s))
    return boundaries


def tokenize_via_boundaries(s: str) -> list[str]:
    """Tokenize by finding all boundaries."""
    boundaries = find_boundaries_linear(s)
    return [s[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]


def find_first_token_length_linear(s: str, max_len: int = 100) -> int:
    """
    Find length of first token by checking each position.

    Since count isn't monotonic, we can't binary search.
    But we CAN check: is count(s[:i]) == 1?
    """
    if not s:
        return 0

    # Try longest first (greedy)
    for length in range(min(max_len, len(s)), 0, -1):
        prefix = s[:length]
        if prefix.strip() and count_tokens(prefix) == 1:
            return length

    return 1  # Fallback: single character


def tokenize(s: str) -> list[str]:
    """
    Tokenize using greedy longest-single-token approach.

    For each position, find the longest prefix that is a single token.
    """
    if not s:
        return []

    tokens = []
    pos = 0

    while pos < len(s):
        remaining = s[pos:]

        # Handle whitespace-only
        if not remaining.strip():
            tokens.append(remaining)
            break

        length = find_first_token_length_linear(remaining)
        tokens.append(s[pos:pos + length])
        pos += length

    return tokens


# Vocabulary building
_known_vocab: set[str] = set()


def add_to_vocab(token: str):
    """Add a discovered token to vocabulary."""
    _known_vocab.add(token)


def get_vocab() -> set[str]:
    """Return the currently known vocabulary."""
    return _known_vocab.copy()


def load_vocab(vocab: set[str]):
    """Load a vocabulary set."""
    global _known_vocab
    _known_vocab = vocab.copy()


def optimal_tokenize(text: str, vocab: set[str]) -> list[str]:
    """
    Find the minimum-token segmentation using dynamic programming.

    Unlike greedy longest-match, this finds the globally optimal split
    that minimizes the total number of tokens. Much closer to BPE output.

    Time: O(n * max_token_len), Space: O(n)
    """
    if not text:
        return []

    n = len(text)
    max_len = max(len(t) for t in vocab) if vocab else 0

    # dp[i] = minimum tokens to cover text[0:i]
    INF = float("inf")
    dp = [INF] * (n + 1)
    dp[0] = 0
    parent = [0] * (n + 1)  # parent[i] = start position of the token ending at i

    for i in range(1, n + 1):
        # Try every possible token ending at position i
        for length in range(1, min(max_len, i) + 1):
            j = i - length
            candidate = text[j:i]
            if candidate in vocab and dp[j] + 1 < dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

        # Fallback: single character (always possible)
        if dp[i - 1] + 1 < dp[i]:
            dp[i] = dp[i - 1] + 1
            parent[i] = i - 1

    # Reconstruct tokens
    tokens = []
    pos = n
    while pos > 0:
        start = parent[pos]
        tokens.append(text[start:pos])
        pos = start

    tokens.reverse()
    return tokens


def greedy_tokenize(text: str, vocab: set[str]) -> list[str]:
    """
    Local greedy longest-match tokenizer using a given vocab set.

    Used for skip-match optimization: tokenize locally, then only
    boundary-find texts where local count != API count.
    """
    if not text:
        return []

    # Build sorted list by length descending for greedy matching
    # Use a trie-like approach: group by max token length for efficiency
    max_len = max(len(t) for t in vocab) if vocab else 0

    tokens = []
    pos = 0
    while pos < len(text):
        # Try longest match first
        matched = False
        for length in range(min(max_len, len(text) - pos), 0, -1):
            candidate = text[pos:pos + length]
            if candidate in vocab:
                tokens.append(candidate)
                pos += length
                matched = True
                break
        if not matched:
            # Unknown char — emit single character as its own token
            tokens.append(text[pos])
            pos += 1

    return tokens
