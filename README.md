# ctoc

Like [cloc](https://github.com/AlDanial/cloc), but counts Claude tokens instead of lines.

```
$ ctoc src/
-----------------------------------------
Language            files          tokens
-----------------------------------------
Python                 10           5,432
JavaScript              5           2,100
-----------------------------------------
SUM                    15           7,532
-----------------------------------------
```

Uses a reverse-engineered vocabulary of 36,495 verified Claude tokens with a greedy longest-match tokenizer. Achieves 95-96% accuracy vs the real tokenizer (see [REPORT.md](REPORT.md)).

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
  --vocab PATH         Path to vocab_tiktoken.json
  --help               Show help
```

### Examples

```bash
ctoc .                                    # tokenize current project
ctoc --by-file src/                       # per-file breakdown
ctoc --include-ext .py --include-ext .js  # only Python and JS
ctoc --exclude-dir vendor .               # skip vendor/
```

## How it works

1. Loads 36,495 tokens from `vocab_tiktoken.json` into a trie
2. Walks source files with `std::filesystem::recursive_directory_iterator`
3. For each file, runs greedy longest-match tokenization against the trie (O(n))
4. Prints a cloc-style summary table

The vocabulary was extracted by probing Anthropic's `count_tokens` API endpoint ~276K times. See [REPORT.md](REPORT.md) for the full methodology.
