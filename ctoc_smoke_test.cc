#include <cassert>
#include <cstdint>
#include <cstring>
#include "vocab_data.h"

static constexpr uint32_t TERM_BIT = 0x80000000u;
static constexpr uint32_t IDX_MASK = 0x7FFFFFFFu;

static size_t match_len(const char* data, size_t len, size_t pos) {
    unsigned char byte = static_cast<unsigned char>(data[pos]);
    uint32_t t = DA_BASE[0] + byte;
    if (t >= DA_TRIE_SIZE) return 1;
    uint32_t c = DA_CHECK[t];
    if (c == 0xFFFFFFFFu || (c & IDX_MASK) != 0) return 1;
    size_t best = (c & TERM_BIT) ? 1 : 0;
    uint32_t state = t;
    for (size_t i = pos + 1; i < len; ++i) {
        byte = static_cast<unsigned char>(data[i]);
        t = DA_BASE[state] + byte;
        if (t >= DA_TRIE_SIZE) break;
        c = DA_CHECK[t];
        if (c == 0xFFFFFFFFu || (c & IDX_MASK) != state) break;
        state = t;
        if (c & TERM_BIT) best = i - pos + 1;
    }
    return best ? best : 1;
}

static size_t count(const char* s) {
    size_t len = strlen(s), n = 0, pos = 0;
    while (pos < len) { pos += match_len(s, len, pos); ++n; }
    return n;
}

int main() {
    assert(DA_TRIE_SIZE > 0);
    assert(count("hello") == 1);
    assert(count("\x01") == 1);
    assert(count("") == 0);
    assert(count("Hello, world!") >= 1 && count("Hello, world!") <= 10);
}
