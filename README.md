# ctoc

Like [cloc](https://github.com/AlDanial/cloc), but counts Claude tokens instead of lines.

```
$ ctoc src/
---------------------
Ext     files  tokens
---------------------
.rs        17  52,000
.py         5  12,340
.ts         3   4,200
---------------------
SUM        25  68,540
---------------------
```

Uses a reverse-engineered vocabulary of 36,495 verified Claude tokens with a greedy longest-match tokenizer. The vocabulary is embedded at compile time — the binary is fully self-contained.

## Install

Requires [Bazel](https://bazel.build/) (or [Bazelisk](https://github.com/bazelbuild/bazelisk)):

```bash
bazel build //:ctoc
cp bazel-bin/ctoc /usr/local/bin/
```

### Cross-compile

```bash
bazel build //:ctoc --config=linux_amd64
bazel build //:ctoc --config=linux_arm64
bazel build //:ctoc --config=macos_amd64
bazel build //:ctoc --config=macos_arm64
```

Hermetic cross-compilation powered by [zig cc](https://github.com/uber/hermetic_cc_toolchain) -- no system toolchain required.

## Usage

```
ctoc [OPTIONS] PATH [PATH...]

Options:
  --by-file            Show per-file token counts
  --exclude-dir DIR    Exclude directory (repeatable)
  --include-ext EXT    Only include extension, e.g. .py (repeatable)
  --help               Show help
```

### Examples

```bash
ctoc .                                    # tokenize current project
ctoc --by-file src/                       # per-file breakdown
ctoc --include-ext .py --include-ext .js  # only Python and JS
ctoc --exclude-dir vendor .               # skip vendor/
```

## Accuracy

Tested against the Anthropic `count_tokens` API (ground truth) across 30 randomly selected source files (.rs, .py, .ts, .hs, .md, .yml, .json, .sh, .h):

| Method | Avg error vs API | Direction |
|--------|-----------------|-----------|
| **ctoc** | **+2.9%** | slightly over-counts |
| tiktoken (cl100k) | -22.7% | under-counts |
| bytes / 4 | -18.2% | under-counts |

Per-extension breakdown:

| Extension | Files | ctoc error | tiktoken error | bytes/4 error |
|-----------|------:|------------|----------------|---------------|
| .rs | 17 | +2.9% | -25.0% | -19.5% |
| .md | 4 | +4.5% | -11.8% | -10.5% |
| .py | 2 | +4.2% | -22.8% | -7.8% |
| .h | 1 | -0.9% | -21.6% | -43.4% |
| .yml | 1 | +1.1% | -21.5% | -5.0% |
| .ts | 2 | -4.3% | -22.8% | -22.3% |
| .hs | 1 | -6.3% | -27.9% | -28.8% |
| .json | 1 | +4.7% | -20.8% | -43.2% |
| .sh | 1 | -2.4% | -22.6% | -33.9% |

ctoc stays within ~5% of the real Claude token count across all file types. tiktoken and bytes/4 consistently undercount by 20-25% because they approximate a different tokenizer.

## Speed

Measured on a 1,341-file / 8.3 MB codebase (Apple M2):

| Method | Tokens/sec | Notes |
|--------|-----------|-------|
| tiktoken (cl100k) | 2,750,000 | Rust BPE, in-memory only |
| **ctoc** | **980,000** | C++ trie, includes file I/O + trie construction |
| bytes / 4 | instant | just division |

ctoc is ~3x slower than tiktoken. The difference comes from:

1. **Trie vs BPE**: tiktoken uses byte-pair encoding implemented in Rust with cache-friendly data structures optimized for sequential merges. ctoc uses a trie with pointer-chasing on every byte — poor cache locality.
2. **Startup cost**: ctoc rebuilds the trie from 36,495 tokens on every invocation (~170ms). tiktoken amortizes this via a pre-compiled vocabulary.
3. **File I/O**: ctoc's timing includes walking directories and reading files from disk. tiktoken was benchmarked on pre-loaded text.

In practice, ctoc processes ~1M tokens/sec end-to-end, which is fast enough to scan a large monorepo in a few seconds.

## How it works

1. At build time, `gen_vocab.py` converts `vocab_tiktoken.json` into a C++ array (`vocab_data.cc`)
2. At runtime, the 36,495 tokens are inserted into a trie
3. Walks source files with `std::filesystem::recursive_directory_iterator`
4. For each file, runs greedy longest-match tokenization against the trie (O(n) per file)
5. Prints a cloc-style summary table grouped by file extension

The vocabulary was extracted by probing Anthropic's `count_tokens` API endpoint ~276K times. See [REPORT.md](REPORT.md) for the full methodology.
