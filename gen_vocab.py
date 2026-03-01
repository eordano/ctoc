#!/usr/bin/env python3
"""Generate vocab_data.cc and vocab_data.h from vocab.json.

Builds a double-array tree at code-gen time and emits it as static C++ arrays.
At runtime, the tree is ready to use with zero initialization — just array
lookups, no heap allocation, no hash maps.
"""

import json
import sys
from collections import deque


TERM_BIT = 0x80000000
IDX_MASK = 0x7FFFFFFF


def build_da_trie(tokens):
    """Build a double-array tree from a list of token strings.

    Returns (base, check, array_size) where base and check are lists of uint32.
    """
    # Phase 1: Build simple trie
    node_children = [[]]  # list of (byte, child_index) per node
    node_terminal = [False]

    for token in tokens:
        cur = 0
        for byte in token.encode('utf-8'):
            existing = None
            for k, idx in node_children[cur]:
                if k == byte:
                    existing = idx
                    break
            if existing is not None:
                cur = existing
            else:
                idx = len(node_children)
                node_children.append([])
                node_terminal.append(False)
                node_children[cur].append((byte, idx))
                cur = idx
        node_terminal[cur] = True

    # Sort children by byte value
    for children in node_children:
        children.sort(key=lambda x: x[0])

    num_nodes = len(node_children)

    # Phase 2: Convert to double-array via BFS
    initial_size = num_nodes + 512
    base = [0] * initial_size
    check = [0xFFFFFFFF] * initial_size
    occupied = [False] * initial_size

    da_pos = [0] * num_nodes
    da_pos[0] = 0  # root at position 0
    occupied[0] = True

    def find_base(keys):
        n = len(occupied)
        first_key = keys[0]
        b = 0
        while True:
            fpos = b + first_key
            if fpos < n and occupied[fpos]:
                b += 1
                continue
            ok = True
            for k in keys[1:]:
                pos = b + k
                if pos < n and occupied[pos]:
                    ok = False
                    break
            if ok:
                return b
            b += 1

    queue = deque([0])

    while queue:
        trie_node = queue.popleft()
        s = da_pos[trie_node]
        ch = node_children[trie_node]

        if not ch:
            continue

        keys = [k for k, _ in ch]
        b = find_base(keys)

        max_pos = b + 256
        if max_pos >= len(base):
            new_size = max_pos + 512
            base.extend([0] * (new_size - len(base)))
            check.extend([0xFFFFFFFF] * (new_size - len(check)))
            occupied.extend([False] * (new_size - len(occupied)))

        base[s] = b

        for key, child_trie_idx in ch:
            t = b + key
            term = TERM_BIT if node_terminal[child_trie_idx] else 0
            check[t] = s | term
            occupied[t] = True
            da_pos[child_trie_idx] = t
            queue.append(child_trie_idx)

    # Trim to actual size
    actual_size = 0
    for i in range(len(occupied) - 1, -1, -1):
        if occupied[i]:
            actual_size = i + 1
            break

    return base[:actual_size], check[:actual_size], actual_size


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <vocab.json> <out.cc> <out.h>", file=sys.stderr)
        sys.exit(1)

    vocab_path, cc_path, h_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokens = data['verified']

    base, check, array_size = build_da_trie(tokens)

    # Write header
    with open(h_path, 'w', encoding='utf-8') as f:
        f.write('#ifndef VOCAB_DATA_H_\n')
        f.write('#define VOCAB_DATA_H_\n\n')
        f.write('#include <cstdint>\n')
        f.write('#include <cstddef>\n\n')
        f.write(f'static constexpr size_t DA_TRIE_SIZE = {array_size};\n')
        f.write(f'extern const uint32_t DA_BASE[{array_size}];\n')
        f.write(f'extern const uint32_t DA_CHECK[{array_size}];\n\n')
        f.write('#endif  // VOCAB_DATA_H_\n')

    # Write source
    with open(cc_path, 'w', encoding='utf-8') as f:
        f.write('#include "vocab_data.h"\n\n')

        f.write(f'const uint32_t DA_BASE[{array_size}] = {{\n')
        for i in range(0, array_size, 16):
            chunk = base[i:i+16]
            f.write('    ' + ','.join(str(v) for v in chunk) + ',\n')
        f.write('};\n\n')

        f.write(f'const uint32_t DA_CHECK[{array_size}] = {{\n')
        for i in range(0, array_size, 16):
            chunk = check[i:i+16]
            f.write('    ' + ','.join(str(v) for v in chunk) + ',\n')
        f.write('};\n')

    print(f"Generated {len(tokens)} tokens, {array_size} slots", file=sys.stderr)


if __name__ == '__main__':
    main()
