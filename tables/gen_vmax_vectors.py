#!/usr/bin/env python3
"""
gen_vmax_vectors.py — Reference vectors for VMAX (find max + index).
"""

import struct
from pathlib import Path

Q8 = 256

def q8(real):
    v = int(round(real * Q8))
    return max(-32768, min(32767, v))

def vmax_ref(vec):
    """Return (max_value, first_index_of_max)."""
    max_val = vec[0]
    max_idx = 0
    for i in range(1, len(vec)):
        if vec[i] > max_val:
            max_val = vec[i]
            max_idx = i
    return max_val, max_idx

def gen_vectors():
    cases = []
    # 1. Length 1
    cases.append([q8(3.5)])
    # 2. Strictly increasing
    cases.append([q8(1.0), q8(2.0), q8(3.0), q8(4.0)])
    # 3. Strictly decreasing
    cases.append([q8(4.0), q8(3.0), q8(2.0), q8(1.0)])
    # 4. Max at start
    cases.append([q8(10.0), q8(1.0), q8(2.0), q8(3.0)])
    # 5. Max in middle
    cases.append([q8(1.0), q8(2.0), q8(10.0), q8(3.0)])
    # 6. Max at end
    cases.append([q8(1.0), q8(2.0), q8(3.0), q8(10.0)])
    # 7. All equal (tie at index 0)
    cases.append([q8(5.0), q8(5.0), q8(5.0), q8(5.0)])
    # 8. Mixed signs, max is negative
    cases.append([q8(-3.0), q8(-1.0), q8(-5.0), q8(-2.0)])

    result = []
    for vec in cases:
        mv, mi = vmax_ref(vec)
        result.append((vec, mv, mi))
    return result

def emit_binary(path, vectors):
    buf = bytearray()
    buf += b"VM"
    buf += bytes([0x01, 0x00])
    buf += struct.pack(">H", len(vectors))
    buf += struct.pack(">H", 0)
    # Record: count[1], pad[1], vec[count*2], exp_max[2], exp_idx[2]
    for (vec, mv, mi) in vectors:
        buf += bytes([len(vec), 0])
        for v in vec: buf += struct.pack(">h", v)
        buf += struct.pack(">h", mv)
        buf += struct.pack(">H", mi)
    Path(path).write_bytes(bytes(buf))
    return len(buf)

def self_test():
    assert vmax_ref([q8(3.5)]) == (q8(3.5), 0)
    assert vmax_ref([q8(1.0), q8(2.0), q8(3.0)])[1] == 2
    assert vmax_ref([q8(5.0), q8(5.0)])[1] == 0  # tie: first
    assert vmax_ref([q8(-3.0), q8(-1.0)])[1] == 1
    print("self_test: PASS")

def main():
    self_test()
    vectors = gen_vectors()
    out_dir = Path("test")
    out_dir.mkdir(exist_ok=True)
    nbytes = emit_binary(out_dir / "vectors_vmax.bin", vectors)
    print(f"wrote test/vectors_vmax.bin ({nbytes} bytes)")
    print(f"  VMAX vectors: {len(vectors)}")
    for i, (vec, mv, mi) in enumerate(vectors):
        print(f"  [{i}] len={len(vec)} max={mv} idx={mi}")

if __name__ == "__main__":
    main()
