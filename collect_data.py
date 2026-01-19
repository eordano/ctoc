"""
Collect training data by sampling substrings from real text corpora.

Strategy:
1. Sample random substrings from Wikipedia (real language patterns)
2. Stratify by length (tiny/short/medium/long)
3. Include edge cases (numbers, unicode, punctuation)
4. Hit count_tokens API for ground truth
"""

import json
import random
import time
from pathlib import Path
from datasets import load_dataset
from tokenizer import count_tokens


def sample_substrings(text: str, length: int, n: int) -> list[str]:
    """Sample n random substrings of given length from text."""
    samples = []
    if len(text) < length:
        return samples

    for _ in range(n):
        start = random.randint(0, len(text) - length)
        samples.append(text[start:start + length])

    return samples


def generate_edge_cases() -> list[str]:
    """Generate edge cases that need explicit coverage."""
    cases = []

    # Numbers (various formats)
    for i in range(1000):
        cases.append(str(i))
    for _ in range(200):
        cases.append(str(random.randint(1000, 999999)))
    for _ in range(100):
        cases.append(f"{random.uniform(-1000, 1000):.2f}")
    for _ in range(50):
        cases.append(f"{random.uniform(0, 1):.6f}")

    # Punctuation patterns
    for p in ".,!?;:\"'()[]{}":
        cases.extend([p, p*2, p*3, p*5])
    cases.extend(["...", "???", "!!!", "---", "***", "///"])

    # Whitespace patterns
    cases.extend([" "*i for i in range(1, 20)])
    cases.extend(["\t", "\t\t", "\n", "\n\n", " \n ", "\t\n\t"])

    # Common unicode
    unicode_samples = [
        "café", "naïve", "résumé", "coöperate", "Zürich",
        "北京", "東京", "日本語", "中文",
        "Привет", "Москва", "مرحبا", "שלום",
        "🎉", "🚀", "👍", "❤️", "🔥",
        "α", "β", "γ", "δ", "π", "∑", "∞", "√", "≠", "≤", "≥",
        "©", "®", "™", "€", "£", "¥", "°",
    ]
    cases.extend(unicode_samples)

    # Mixed patterns
    cases.extend([
        "hello123", "test_case", "camelCase", "PascalCase",
        "SCREAMING_SNAKE", "kebab-case", "dot.notation",
        "user@example.com", "https://example.com",
        "v1.0.0", "2024-01-15", "12:30:45",
        "$100", "50%", "#hashtag", "@mention",
    ])

    # Repeated characters
    for c in "abcdefghijklmnopqrstuvwxyz0123456789":
        for length in [2, 5, 10, 20]:
            cases.append(c * length)

    return cases


def load_wikipedia_text(num_docs: int = 1000) -> str:
    """Load Wikipedia text from HuggingFace."""
    print(f"Loading ~{num_docs} documents...")

    # Try multiple datasets in order of preference
    datasets_to_try = [
        ("wikitext", "wikitext-103-v1", "train"),
        ("bookcorpus", None, "train"),
        ("ag_news", None, "train"),
    ]

    texts = []

    for ds_name, ds_config, split in datasets_to_try:
        try:
            print(f"  Trying {ds_name}...")
            if ds_config:
                ds = load_dataset(ds_name, ds_config, split=split, streaming=True)
            else:
                ds = load_dataset(ds_name, split=split, streaming=True)

            for i, doc in enumerate(ds):
                if i >= num_docs:
                    break

                # Handle different dataset formats
                text = doc.get("text", "")
                if not text:
                    text = doc.get("content", "")
                if not text:
                    # ag_news format
                    text = doc.get("description", "") + " " + doc.get("title", "")

                if text and len(text) > 50:
                    texts.append(text)

                if (i + 1) % 500 == 0:
                    print(f"    Loaded {i + 1} documents...")

            if texts:
                print(f"  Successfully loaded from {ds_name}")
                break

        except Exception as e:
            print(f"  Failed to load {ds_name}: {e}")
            continue

    if not texts:
        raise RuntimeError("Could not load any dataset")

    full_text = "\n\n".join(texts)
    print(f"Total text length: {len(full_text):,} characters")
    return full_text


def generate_samples(
    corpus: str,
    num_samples: int = 50000,
    length_buckets: dict = None,
) -> list[str]:
    """Generate stratified samples from corpus."""

    if length_buckets is None:
        # Default: equal split across length buckets
        length_buckets = {
            (1, 5): 0.10,      # Tiny (single tokens)
            (5, 20): 0.20,     # Short (words)
            (20, 100): 0.30,   # Medium (sentences)
            (100, 500): 0.25,  # Long (paragraphs)
            (500, 2000): 0.15, # Very long
        }

    samples = []

    # Edge cases first (always include)
    edge_cases = generate_edge_cases()
    samples.extend(edge_cases)
    print(f"Generated {len(edge_cases)} edge cases")

    # Corpus samples (use most of the budget for real text)
    corpus_budget = max(0, num_samples - len(edge_cases))

    for (min_len, max_len), fraction in length_buckets.items():
        n_samples = int(corpus_budget * fraction)
        bucket_samples = []

        attempts = 0
        max_attempts = n_samples * 5

        while len(bucket_samples) < n_samples and attempts < max_attempts:
            length = random.randint(min_len, max_len)
            if len(corpus) > length:
                start = random.randint(0, len(corpus) - length)
                sample = corpus[start:start + length]
                # Skip whitespace-only samples
                if sample.strip():
                    bucket_samples.append(sample)
            attempts += 1

        samples.extend(bucket_samples)
        print(f"Generated {len(bucket_samples)} samples for length {min_len}-{max_len}")

    # Shuffle
    random.shuffle(samples)

    return samples


def collect_data(
    output_file: str = "training_data.jsonl",
    num_samples: int = 50000,
    num_wiki_docs: int = 2000,
    rate_limit_delay: float = 0.02,
    checkpoint_interval: int = 1000,
):
    """Main collection loop."""

    # Load corpus
    corpus = load_wikipedia_text(num_wiki_docs)

    # Generate samples
    samples = generate_samples(corpus, num_samples)
    print(f"\nTotal samples to collect: {len(samples)}")

    # Load existing data to avoid duplicates
    output_path = Path(output_file)
    existing_texts = set()

    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                existing_texts.add(data["text"])
        print(f"Found {len(existing_texts)} existing samples")

    # Filter out duplicates
    samples = [s for s in samples if s not in existing_texts]
    print(f"New samples to collect: {len(samples)}")

    # Collect
    collected = 0
    errors = 0
    start_time = time.time()

    with open(output_path, "a") as f:
        for i, text in enumerate(samples):
            try:
                # Skip whitespace-only
                if not text.strip():
                    continue

                count = count_tokens(text)
                data = {"text": text, "count": count}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                collected += 1

                if collected % checkpoint_interval == 0:
                    elapsed = time.time() - start_time
                    rate = collected / elapsed if elapsed > 0 else 0
                    eta = (len(samples) - collected) / rate if rate > 0 else 0
                    print(f"  Collected {collected}/{len(samples)} ({rate:.1f}/sec, ETA: {eta/60:.1f}min)")
                    f.flush()

                time.sleep(rate_limit_delay)

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error on sample {i}: {e}")
                continue

    elapsed = time.time() - start_time
    print(f"\nDone! Collected {collected} samples in {elapsed/60:.1f} minutes")
    print(f"Errors: {errors}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    collect_data(
        num_samples=50000,
        num_wiki_docs=2000,
    )
