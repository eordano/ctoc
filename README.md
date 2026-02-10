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

- Self-contained C++17 binary — no runtime dependencies.
- Greedy longest-match tokenizer built from 36,495 reverse-engineered Claude tokens.
- ~**3% error** vs the Anthropic `count_tokens` API across 30 tested files (tiktoken and bytes/4 undercount by 20%+).
- Processes ~1M tokens/sec including file I/O.

## Install

```bash
bazel build //:ctoc
cp bazel-bin/ctoc /usr/local/bin/
```

Cross-compile with `--config={linux_amd64,linux_arm64,macos_amd64,macos_arm64}` via [hermetic zig cc](https://github.com/uber/hermetic_cc_toolchain).

## Usage

```bash
ctoc .                                    # tokenize current project
ctoc --by-file src/                       # per-file breakdown
ctoc --include-ext .py --include-ext .js  # only Python and JS
ctoc --exclude-dir vendor .               # skip vendor/
```

## How it works

1. At build time, `gen_vocab.py` converts `vocab.json` into a C++ array
2. At runtime, tokens are inserted into a trie and files are tokenized via greedy longest-match
3. Vocabulary was extracted by probing Anthropic's `count_tokens` API ~276K times — see [REPORT.md](REPORT.md)
