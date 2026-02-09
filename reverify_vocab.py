"""
Re-verify all tokens in vocab using the fixed § sandwich count_tokens method.

The old count_tokens had a baseline bug that caused false positives for
Unicode tokens and some ASCII tokens containing special characters.
This script re-checks every token and removes false positives.
"""

import json
import time
import sys
from pathlib import Path

from tokenizer import count_tokens


def reverify_all(vocab_file: str = "vocab_tiktoken.json", batch_size: int = 500):
    """Re-verify all tokens in the vocab file."""
    path = Path(vocab_file)
    data = json.load(open(path))
    verified = list(data.get("verified", []))
    checked = set(data.get("checked", []))

    print(f"Loaded {len(verified)} verified tokens, {len(checked)} checked")

    good = set()
    bad = []
    errors = []
    start = time.time()

    for i, token in enumerate(verified):
        try:
            result = count_tokens(token)
            if result == 1:
                good.add(token)
            else:
                bad.append((token, result))
        except Exception as e:
            errors.append((token, str(e)))

        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(verified) - i - 1) / rate / 60
            print(
                f"  [{i+1}/{len(verified)}] ({rate:.1f}/sec, ETA {eta:.1f}min) "
                f"good={len(good)}, bad={len(bad)}, errors={len(errors)}"
            )

        time.sleep(0.03)

    print(f"\n--- Results ---")
    print(f"  Good (confirmed single tokens): {len(good)}")
    print(f"  Bad (NOT single tokens): {len(bad)}")
    print(f"  Errors: {len(errors)}")

    if bad:
        print(f"\n  Sample bad tokens:")
        for token, count in bad[:30]:
            print(f"    {repr(token)} -> {count} tokens")

    if errors:
        print(f"\n  Sample errors:")
        for token, err in errors[:10]:
            print(f"    {repr(token)}: {err}")

    # Save cleaned vocab
    # Remove bad tokens from verified, keep them in checked
    new_verified = good
    # Add bad tokens to checked so we don't recheck them
    for token, _ in bad:
        checked.add(token)
    for token, _ in errors:
        checked.add(token)

    data_out = {
        "verified": sorted(new_verified),
        "checked": sorted(checked),
    }
    with open(vocab_file, "w") as f:
        json.dump(data_out, f)

    print(f"\nSaved: {len(new_verified)} verified (removed {len(verified) - len(new_verified)})")
    print(f"Checked: {len(checked)}")


if __name__ == "__main__":
    reverify_all()
