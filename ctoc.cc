// ctoc — Count Tokens of Code
// Like cloc, but for Claude tokens.
//
// Uses a greedy longest-match tokenizer built from a reverse-engineered
// vocabulary of 36,495 verified Claude tokens (95-96% accuracy).

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// ─── Language detection ──────────────────────────────────────────────

static const std::unordered_map<std::string, std::string> EXT_TO_LANG = {
    {".py", "Python"},
    {".pyi", "Python"},
    {".js", "JavaScript"},
    {".mjs", "JavaScript"},
    {".cjs", "JavaScript"},
    {".jsx", "JavaScript"},
    {".ts", "TypeScript"},
    {".tsx", "TypeScript"},
    {".java", "Java"},
    {".kt", "Kotlin"},
    {".kts", "Kotlin"},
    {".scala", "Scala"},
    {".c", "C"},
    {".h", "C/C++ Header"},
    {".cc", "C++"},
    {".cpp", "C++"},
    {".cxx", "C++"},
    {".hpp", "C++ Header"},
    {".hxx", "C++ Header"},
    {".cs", "C#"},
    {".go", "Go"},
    {".rs", "Rust"},
    {".rb", "Ruby"},
    {".php", "PHP"},
    {".swift", "Swift"},
    {".m", "Objective-C"},
    {".mm", "Objective-C++"},
    {".r", "R"},
    {".R", "R"},
    {".lua", "Lua"},
    {".pl", "Perl"},
    {".pm", "Perl"},
    {".sh", "Shell"},
    {".bash", "Shell"},
    {".zsh", "Shell"},
    {".fish", "Shell"},
    {".ps1", "PowerShell"},
    {".dart", "Dart"},
    {".ex", "Elixir"},
    {".exs", "Elixir"},
    {".erl", "Erlang"},
    {".hrl", "Erlang"},
    {".hs", "Haskell"},
    {".ml", "OCaml"},
    {".mli", "OCaml"},
    {".fs", "F#"},
    {".fsx", "F#"},
    {".clj", "Clojure"},
    {".cljs", "ClojureScript"},
    {".vim", "Vim Script"},
    {".el", "Emacs Lisp"},
    {".sql", "SQL"},
    {".html", "HTML"},
    {".htm", "HTML"},
    {".css", "CSS"},
    {".scss", "SCSS"},
    {".sass", "Sass"},
    {".less", "Less"},
    {".xml", "XML"},
    {".xsl", "XML"},
    {".json", "JSON"},
    {".yaml", "YAML"},
    {".yml", "YAML"},
    {".toml", "TOML"},
    {".ini", "INI"},
    {".cfg", "INI"},
    {".md", "Markdown"},
    {".markdown", "Markdown"},
    {".rst", "reStructuredText"},
    {".tex", "TeX"},
    {".latex", "TeX"},
    {".proto", "Protocol Buffers"},
    {".graphql", "GraphQL"},
    {".gql", "GraphQL"},
    {".tf", "Terraform"},
    {".hcl", "HCL"},
    {".dockerfile", "Dockerfile"},
    {".cmake", "CMake"},
    {".make", "Makefile"},
    {".mk", "Makefile"},
    {".gradle", "Gradle"},
    {".sbt", "sbt"},
    {".zig", "Zig"},
    {".nim", "Nim"},
    {".v", "V"},
    {".jl", "Julia"},
    {".ipynb", "Jupyter Notebook"},
    {".vue", "Vue"},
    {".svelte", "Svelte"},
    {".astro", "Astro"},
    {".sol", "Solidity"},
    {".wasm", "WebAssembly"},
    {".wat", "WebAssembly"},
};

static const std::unordered_set<std::string> DEFAULT_EXCLUDED_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "bower_components",
    "__pycache__", ".venv", "venv", "env",
    "build", "dist", "out", "target",
    ".next", ".nuxt",
    "vendor",
    "bazel-bin", "bazel-out", "bazel-testlogs",
    ".cache", ".pytest_cache", ".mypy_cache",
    ".idea", ".vscode", ".vs",
};

static constexpr size_t MAX_FILE_SIZE = 1 * 1024 * 1024; // 1 MB
static constexpr size_t BINARY_CHECK_SIZE = 8192;

// ─── Trie ────────────────────────────────────────────────────────────

struct TrieNode {
    std::unordered_map<unsigned char, TrieNode*> children;
    bool is_terminal = false;

    ~TrieNode() {
        for (auto& [_, child] : children)
            delete child;
    }
};

class Trie {
public:
    Trie() : root_(new TrieNode()) {}
    ~Trie() { delete root_; }

    Trie(const Trie&) = delete;
    Trie& operator=(const Trie&) = delete;

    void insert(const std::string& token) {
        TrieNode* node = root_;
        for (unsigned char c : token) {
            auto it = node->children.find(c);
            if (it == node->children.end()) {
                node->children[c] = new TrieNode();
                node = node->children[c];
            } else {
                node = it->second;
            }
        }
        node->is_terminal = true;
    }

    // Returns the length of the longest match starting at data[pos], or 0.
    size_t longest_match(const std::string& data, size_t pos) const {
        TrieNode* node = root_;
        size_t best = 0;
        for (size_t i = pos; i < data.size(); ++i) {
            auto it = node->children.find(static_cast<unsigned char>(data[i]));
            if (it == node->children.end())
                break;
            node = it->second;
            if (node->is_terminal)
                best = i - pos + 1;
        }
        return best;
    }

private:
    TrieNode* root_;
};

// ─── JSON parsing (extract "verified" string array) ──────────────────

// Unescape a JSON string value (handles \n, \t, \r, \\, \", \/, \uXXXX)
static std::string json_unescape(const std::string& s, size_t start, size_t end) {
    std::string result;
    result.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
        if (s[i] == '\\' && i + 1 < end) {
            ++i;
            switch (s[i]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                case 'u': {
                    // Parse 4 hex digits
                    if (i + 4 < end) {
                        std::string hex = s.substr(i + 1, 4);
                        uint32_t cp = std::stoul(hex, nullptr, 16);
                        i += 4;
                        // Check for surrogate pair
                        if (cp >= 0xD800 && cp <= 0xDBFF && i + 2 < end &&
                            s[i + 1] == '\\' && s[i + 2] == 'u') {
                            std::string hex2 = s.substr(i + 3, 4);
                            uint32_t cp2 = std::stoul(hex2, nullptr, 16);
                            if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                                i += 6;
                            }
                        }
                        // Encode as UTF-8
                        if (cp < 0x80) {
                            result += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            result += static_cast<char>(0xC0 | (cp >> 6));
                            result += static_cast<char>(0x80 | (cp & 0x3F));
                        } else if (cp < 0x10000) {
                            result += static_cast<char>(0xE0 | (cp >> 12));
                            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            result += static_cast<char>(0xF0 | (cp >> 18));
                            result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default: result += s[i]; break;
            }
        } else {
            result += s[i];
        }
    }
    return result;
}

// Parse the "verified" array from vocab_tiktoken.json.
// Format: {"verified": ["token1", "token2", ...], "checked": [...]}
static std::vector<std::string> parse_vocab(const std::string& json) {
    std::vector<std::string> tokens;

    // Find "verified"
    size_t key_pos = json.find("\"verified\"");
    if (key_pos == std::string::npos) {
        std::cerr << "ctoc: vocab JSON missing \"verified\" key\n";
        return tokens;
    }

    // Find the opening bracket of the array
    size_t arr_start = json.find('[', key_pos);
    if (arr_start == std::string::npos) {
        std::cerr << "ctoc: malformed vocab JSON\n";
        return tokens;
    }

    // Parse strings within the array
    size_t i = arr_start + 1;
    while (i < json.size()) {
        // Skip whitespace and commas
        while (i < json.size() && (json[i] == ' ' || json[i] == '\n' ||
               json[i] == '\r' || json[i] == '\t' || json[i] == ','))
            ++i;

        if (i >= json.size() || json[i] == ']')
            break;

        if (json[i] != '"') {
            ++i;
            continue;
        }

        // Find end of string (handle escapes)
        size_t str_start = i + 1;
        size_t j = str_start;
        while (j < json.size()) {
            if (json[j] == '\\') {
                j += 2;
            } else if (json[j] == '"') {
                break;
            } else {
                ++j;
            }
        }

        tokens.push_back(json_unescape(json, str_start, j));
        i = j + 1;
    }

    return tokens;
}

// ─── Tokenizer ───────────────────────────────────────────────────────

static size_t count_tokens(const std::string& text, const Trie& trie) {
    size_t count = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t match_len = trie.longest_match(text, pos);
        if (match_len == 0) {
            // Unknown byte — count as 1 token (single-byte fallback)
            ++pos;
        } else {
            pos += match_len;
        }
        ++count;
    }
    return count;
}

// ─── File discovery ──────────────────────────────────────────────────

static bool is_binary(const std::string& data) {
    size_t check_len = std::min(data.size(), BINARY_CHECK_SIZE);
    for (size_t i = 0; i < check_len; ++i) {
        if (data[i] == '\0')
            return true;
    }
    return false;
}

static std::string read_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct FileEntry {
    fs::path path;
    std::string language;
    size_t tokens;
};

static std::string detect_language(const fs::path& path) {
    // Special filenames
    std::string filename = path.filename().string();
    if (filename == "Makefile" || filename == "makefile" || filename == "GNUmakefile")
        return "Makefile";
    if (filename == "Dockerfile")
        return "Dockerfile";
    if (filename == "CMakeLists.txt")
        return "CMake";
    if (filename == "BUILD" || filename == "BUILD.bazel")
        return "Bazel";
    if (filename == "WORKSPACE" || filename == "WORKSPACE.bazel")
        return "Bazel";
    if (filename == "MODULE.bazel")
        return "Bazel";

    std::string ext = path.extension().string();
    // Convert to lowercase for matching
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    auto it = EXT_TO_LANG.find(ext);
    if (it != EXT_TO_LANG.end())
        return it->second;
    return {};
}

static bool should_exclude_dir(const std::string& dirname,
                               const std::unordered_set<std::string>& excluded) {
    return excluded.count(dirname) > 0;
}

// Check if path starts with a bazel- symlink directory
static bool is_bazel_dir(const fs::path& path) {
    for (const auto& component : path) {
        std::string name = component.string();
        if (name.size() > 6 && name.substr(0, 6) == "bazel-")
            return true;
    }
    return false;
}

static std::vector<FileEntry> discover_files(
    const std::vector<std::string>& paths,
    const std::unordered_set<std::string>& excluded_dirs,
    const std::unordered_set<std::string>& include_exts,
    const Trie& trie)
{
    std::vector<FileEntry> files;

    for (const auto& input_path : paths) {
        fs::path p(input_path);
        std::error_code ec;

        if (!fs::exists(p, ec)) {
            std::cerr << "ctoc: " << input_path << ": No such file or directory\n";
            continue;
        }

        if (fs::is_regular_file(p, ec)) {
            // Single file — always process even if extension unknown
            std::string lang = detect_language(p);
            if (!include_exts.empty()) {
                std::string ext = p.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (include_exts.find(ext) == include_exts.end())
                    continue;
            }
            std::string content = read_file(p);
            if (content.empty() || is_binary(content))
                continue;
            if (lang.empty())
                lang = "Other";
            files.push_back({p, lang, count_tokens(content, trie)});
            continue;
        }

        if (!fs::is_directory(p, ec))
            continue;

        for (auto it = fs::recursive_directory_iterator(
                 p, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); ++it) {

            if (ec) {
                it.increment(ec);
                continue;
            }

            if (it->is_directory()) {
                std::string dirname = it->path().filename().string();
                if (should_exclude_dir(dirname, excluded_dirs) ||
                    is_bazel_dir(it->path().lexically_relative(p))) {
                    it.disable_recursion_pending();
                }
                continue;
            }

            if (!it->is_regular_file())
                continue;

            // Check file size
            auto fsize = it->file_size(ec);
            if (ec || fsize > MAX_FILE_SIZE)
                continue;

            std::string lang = detect_language(it->path());
            if (lang.empty())
                continue;

            if (!include_exts.empty()) {
                std::string ext = it->path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (include_exts.find(ext) == include_exts.end())
                    continue;
            }

            std::string content = read_file(it->path());
            if (content.empty() || is_binary(content))
                continue;

            files.push_back({it->path(), lang, count_tokens(content, trie)});
        }
    }

    return files;
}

// ─── Output formatting ──────────────────────────────────────────────

// Format a number with comma separators: 1234567 -> "1,234,567"
static std::string format_number(size_t n) {
    std::string s = std::to_string(n);
    int insert_pos = static_cast<int>(s.size()) - 3;
    while (insert_pos > 0) {
        s.insert(insert_pos, ",");
        insert_pos -= 3;
    }
    return s;
}

static void print_summary(const std::vector<FileEntry>& files) {
    // Aggregate by language
    struct LangStats {
        size_t file_count = 0;
        size_t token_count = 0;
    };
    std::unordered_map<std::string, LangStats> by_lang;
    size_t total_files = 0;
    size_t total_tokens = 0;

    for (const auto& f : files) {
        by_lang[f.language].file_count++;
        by_lang[f.language].token_count += f.tokens;
        total_files++;
        total_tokens += f.tokens;
    }

    // Sort by token count descending
    std::vector<std::pair<std::string, LangStats>> sorted(by_lang.begin(), by_lang.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.token_count > b.second.token_count; });

    // Calculate column widths
    size_t lang_w = 8; // "Language"
    for (const auto& [lang, _] : sorted)
        lang_w = std::max(lang_w, lang.size());

    std::string files_str = format_number(total_files);
    std::string tokens_str = format_number(total_tokens);
    size_t files_w = std::max(size_t(5), files_str.size());
    size_t tokens_w = std::max(size_t(6), tokens_str.size());

    for (const auto& [_, stats] : sorted) {
        files_w = std::max(files_w, format_number(stats.file_count).size());
        tokens_w = std::max(tokens_w, format_number(stats.token_count).size());
    }

    size_t total_w = lang_w + 2 + files_w + 2 + tokens_w;
    std::string line(total_w, '\xe2'); // placeholder
    // Use simple dashes for portability
    line = std::string(total_w, '-');

    std::cout << line << "\n";
    std::cout << std::left << std::setw(lang_w) << "Language"
              << "  " << std::right << std::setw(files_w) << "files"
              << "  " << std::setw(tokens_w) << "tokens" << "\n";
    std::cout << line << "\n";

    for (const auto& [lang, stats] : sorted) {
        std::cout << std::left << std::setw(lang_w) << lang
                  << "  " << std::right << std::setw(files_w) << format_number(stats.file_count)
                  << "  " << std::setw(tokens_w) << format_number(stats.token_count) << "\n";
    }

    std::cout << line << "\n";
    std::cout << std::left << std::setw(lang_w) << "SUM"
              << "  " << std::right << std::setw(files_w) << format_number(total_files)
              << "  " << std::setw(tokens_w) << format_number(total_tokens) << "\n";
    std::cout << line << "\n";
}

static void print_by_file(const std::vector<FileEntry>& files) {
    // Sort by tokens descending
    std::vector<const FileEntry*> sorted;
    sorted.reserve(files.size());
    for (const auto& f : files)
        sorted.push_back(&f);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto* a, const auto* b) { return a->tokens > b->tokens; });

    // Calculate column widths
    size_t path_w = 4; // "File"
    size_t lang_w = 8; // "Language"
    size_t tokens_w = 6; // "tokens"

    size_t total_tokens = 0;
    for (const auto* f : sorted) {
        path_w = std::max(path_w, f->path.string().size());
        lang_w = std::max(lang_w, f->language.size());
        tokens_w = std::max(tokens_w, format_number(f->tokens).size());
        total_tokens += f->tokens;
    }

    tokens_w = std::max(tokens_w, format_number(total_tokens).size());

    size_t total_w = path_w + 2 + lang_w + 2 + tokens_w;
    std::string line(total_w, '-');

    std::cout << line << "\n";
    std::cout << std::left << std::setw(path_w) << "File"
              << "  " << std::setw(lang_w) << "Language"
              << "  " << std::right << std::setw(tokens_w) << "tokens" << "\n";
    std::cout << line << "\n";

    for (const auto* f : sorted) {
        std::cout << std::left << std::setw(path_w) << f->path.string()
                  << "  " << std::setw(lang_w) << f->language
                  << "  " << std::right << std::setw(tokens_w) << format_number(f->tokens) << "\n";
    }

    std::cout << line << "\n";

    std::string sum_label = "SUM (" + std::to_string(files.size()) + " files)";
    // Pad to fill file + language columns
    size_t sum_pad = path_w + 2 + lang_w;
    std::cout << std::left << std::setw(sum_pad) << sum_label
              << "  " << std::right << std::setw(tokens_w) << format_number(total_tokens) << "\n";
    std::cout << line << "\n";
}

// ─── Vocab file resolution ──────────────────────────────────────────

static fs::path resolve_vocab_path(const std::string& explicit_path) {
    if (!explicit_path.empty())
        return fs::path(explicit_path);

    // Try relative to the binary
    std::error_code ec;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (ec) {
        // macOS: try _NSGetExecutablePath or argv[0] fallback
        // We'll handle this via the Bazel runfiles data attribute
        // which places vocab_tiktoken.json next to the binary
    }

    // Try common locations
    std::vector<fs::path> candidates;
    if (!exe.empty())
        candidates.push_back(exe.parent_path() / "vocab_tiktoken.json");

    // Bazel runfiles: binary.runfiles/_main/vocab_tiktoken.json
    if (!exe.empty()) {
        fs::path runfiles = fs::path(exe.string() + ".runfiles") / "_main" / "vocab_tiktoken.json";
        candidates.push_back(runfiles);
    }

    // Current directory
    candidates.push_back(fs::current_path() / "vocab_tiktoken.json");

    for (const auto& c : candidates) {
        if (fs::exists(c, ec))
            return c;
    }

    return "vocab_tiktoken.json";
}

// ─── Help ────────────────────────────────────────────────────────────

static void print_help() {
    std::cout <<
R"(ctoc - Count Tokens of Code

Like cloc, but counts Claude tokens instead of lines.
Uses a reverse-engineered vocabulary of 36,495 tokens (95-96% accuracy).

USAGE:
    ctoc [OPTIONS] PATH [PATH...]

OPTIONS:
    --by-file            Show per-file token counts
    --exclude-dir DIR    Exclude directory name (repeatable)
    --include-ext EXT    Only include file extension, e.g. .py (repeatable)
    --vocab PATH         Path to vocab_tiktoken.json
    --help               Show this help message

EXAMPLES:
    ctoc src/
    ctoc --by-file main.py utils.py
    ctoc --exclude-dir vendor --exclude-dir test .
    ctoc --include-ext .py --include-ext .js src/
)";
}

// ─── Main ────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    bool by_file = false;
    std::string vocab_path;
    std::vector<std::string> input_paths;
    std::unordered_set<std::string> extra_excluded_dirs;
    std::unordered_set<std::string> include_exts;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        } else if (arg == "--by-file") {
            by_file = true;
        } else if (arg == "--exclude-dir" && i + 1 < argc) {
            extra_excluded_dirs.insert(argv[++i]);
        } else if (arg == "--include-ext" && i + 1 < argc) {
            std::string ext = argv[++i];
            if (ext[0] != '.')
                ext = "." + ext;
            include_exts.insert(ext);
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg[0] == '-') {
            std::cerr << "ctoc: unknown option: " << arg << "\n";
            std::cerr << "Try 'ctoc --help' for more information.\n";
            return 1;
        } else {
            input_paths.push_back(arg);
        }
    }

    if (input_paths.empty()) {
        std::cerr << "ctoc: no input paths specified\n";
        std::cerr << "Try 'ctoc --help' for more information.\n";
        return 1;
    }

    // Merge excluded dirs
    auto excluded_dirs = DEFAULT_EXCLUDED_DIRS;
    excluded_dirs.insert(extra_excluded_dirs.begin(), extra_excluded_dirs.end());

    // Load vocab
    fs::path vp = resolve_vocab_path(vocab_path);
    std::string vocab_json = read_file(vp);
    if (vocab_json.empty()) {
        std::cerr << "ctoc: cannot read vocab file: " << vp << "\n";
        return 1;
    }

    auto vocab = parse_vocab(vocab_json);
    if (vocab.empty()) {
        std::cerr << "ctoc: no tokens found in vocab file\n";
        return 1;
    }

    // Build trie
    Trie trie;
    for (const auto& token : vocab)
        trie.insert(token);

    // Discover and tokenize files
    auto files = discover_files(input_paths, excluded_dirs, include_exts, trie);

    if (files.empty()) {
        std::cerr << "ctoc: no files found\n";
        return 1;
    }

    // Output
    if (by_file)
        print_by_file(files);
    else
        print_summary(files);

    return 0;
}
