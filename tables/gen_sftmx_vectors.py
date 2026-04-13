#!/usr/bin/env python3
"""
gen_sftmx_vectors.py — Reference vectors for SFTMX (Q8 softmax).

Exactly matches the 6309 SFTMX algorithm:
  1. max = max(vec)
  2. exp_i = EXPTBL[clamp((max - vec[i]) >> 3, 0, 255)]
  3. sum = sum(exp_i)
  4. out_i = (exp_i << 8) / sum  (FXDIV truncation toward zero)
"""

import math
import struct
from pathlib import Path

Q8 = 256

def q8(real):
    return max(-32768, min(32767, int(round(real * Q8))))

# Build EXPTBL (must match gen_exptbl.py / ACTFN.MAC exactly)
EXPTBL = [max(0, int(math.exp(-i / 32.0) * 256.0 + 0.5)) for i in range(256)]

def fxdiv_ref(a, b):
    """Q8 division: (a << 8) / b, truncating toward zero."""
    if b == 0:
        return 0
    numerator = a * 256
    if (numerator >= 0 and b >= 0) or (numerator < 0 and b < 0):
        result = numerator // b
    else:
        result = -(abs(numerator) // abs(b))
    return max(-32768, min(32767, result))

def sftmx_ref(vec):
    """Exactly matches 6309 SFTMX step by step."""
    n = len(vec)
    mx = max(vec)

    # Step 2: exp lookup
    exps = []
    for v in vec:
        diff = mx - v  # >= 0
        if diff > 2040:
            idx = 255
        else:
            idx = diff >> 3  # integer divide by 8
        if idx > 255:
            idx = 255
        exps.append(EXPTBL[idx])

    # Sum (16-bit)
    s = sum(exps)

    # Step 3: normalize
    if s == 0:
        return [0] * n
    out = [fxdiv_ref(e, s) for e in exps]
    return out

def gen_vectors():
    cases = []

    # 1. Single element [256] -> [256] (1.0 -> 1.0)
    cases.append([q8(1.0)])

    # 2. Two equal [128, 128] -> [128, 128] (0.5, 0.5)
    cases.append([q8(0.5), q8(0.5)])

    # 3. Two unequal [256, 0] -> sigmoid-ish
    cases.append([q8(1.0), q8(0.0)])

    # 4. Three with clear winner
    cases.append([q8(3.0), q8(1.0), q8(1.0)])

    # 5. Eight uniform -> all ~1/8
    cases.append([q8(0.5)] * 8)

    # 6. Eight, one dominant
    cases.append([q8(5.0)] + [q8(0.0)] * 7)

    # 7. All-negative
    cases.append([q8(-1.0), q8(-2.0), q8(-3.0), q8(-1.0)])

    # 8. Large range (triggers index clamp)
    cases.append([q8(10.0), q8(-10.0), q8(0.0)])

    # 9. Zero vector -> uniform
    cases.append([0, 0, 0, 0])

    # 10. Alternating large positive/negative
    cases.append([q8(4.0), q8(-4.0), q8(4.0), q8(-4.0)])

    result = []
    for vec in cases:
        out = sftmx_ref(vec)
        result.append((vec, out))
    return result

def emit_binary(path, vectors):
    buf = bytearray()
    buf += b"SM"
    buf += bytes([0x01, 0x00])
    buf += struct.pack(">H", len(vectors))
    buf += struct.pack(">H", 0)
    # Record: count[1], pad[1], vec[count*2], expected[count*2]
    for (vec, exp) in vectors:
        buf += bytes([len(vec), 0])
        for v in vec:
            buf += struct.pack(">h", v)
        for v in exp:
            buf += struct.pack(">h", v)
    Path(path).write_bytes(bytes(buf))
    return len(buf)

def self_test():
    # Single element: softmax([x]) = [1.0] = [256]
    r = sftmx_ref([q8(1.0)])
    assert r == [256], f"single: {r}"

    # Two equal: should be [128, 128]
    r = sftmx_ref([q8(0.5), q8(0.5)])
    assert r == [128, 128], f"equal: {r}"

    # Four zeros: all same -> each = 256/4 = 64
    r = sftmx_ref([0, 0, 0, 0])
    assert all(abs(v - 64) <= 1 for v in r), f"zeros: {r}"

    # Sum check: outputs should sum close to 256
    for vec in [[q8(1.0), q8(0.0)], [q8(3.0), q8(1.0), q8(1.0)]]:
        r = sftmx_ref(vec)
        s = sum(r)
        assert 254 <= s <= 258, f"sum check failed: {r} sum={s}"

    print("self_test: PASS")

def main():
    self_test()
    vectors = gen_vectors()
    out_dir = Path("test")
    out_dir.mkdir(exist_ok=True)
    nbytes = emit_binary(out_dir / "vectors_sftmx.bin", vectors)
    print(f"wrote test/vectors_sftmx.bin ({nbytes} bytes)")
    print(f"  SFTMX vectors: {len(vectors)}")
    for i, (vec, exp) in enumerate(vectors):
        s = sum(exp)
        print(f"  [{i}] len={len(vec)} sum={s} out={exp}")

if __name__ == "__main__":
    main()
