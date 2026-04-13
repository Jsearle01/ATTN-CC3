#!/usr/bin/env python3
"""
gen_fxdiv_vectors.py — Reference vectors for FXDIV (Q8 division).

FXDIV computes (a << 8) / b, truncating toward zero.
This matches the PDP-11 ASHC+DIV and 6309 DIVQ behavior.
"""

import struct
from pathlib import Path

Q8 = 256

def q8(real):
    v = int(round(real * Q8))
    return max(-32768, min(32767, v))

def fxdiv_ref(a, b):
    """Q8 division: (a << 8) / b, truncating toward zero.
    Divide-by-zero: returns 32767 if a>0, -32768 if a<0, 0 if a==0."""
    if b == 0:
        if a > 0: return 32767
        if a < 0: return -32768
        return 0
    # Python integer division truncates toward negative infinity.
    # We need truncation toward zero (matching hardware DIV).
    numerator = a * 256  # a << 8
    if (numerator >= 0 and b >= 0) or (numerator < 0 and b < 0):
        # Same sign: result positive, floor division == truncation
        result = numerator // b
    else:
        # Opposite signs: result negative
        # Truncate toward zero: -(abs(num) // abs(den))
        result = -(abs(numerator) // abs(b))
    # Clamp to 16-bit signed
    result = max(-32768, min(32767, result))
    return result

def gen_vectors():
    cases = []
    # 1. 1.0 / 1.0 = 1.0
    cases.append((q8(1.0), q8(1.0)))
    # 2. 1.0 / 2.0 = 0.5
    cases.append((q8(1.0), q8(2.0)))
    # 3. 0.5 / 0.25 = 2.0
    cases.append((q8(0.5), q8(0.25)))
    # 4. Negative numerator: -1.0 / 2.0 = -0.5
    cases.append((q8(-1.0), q8(2.0)))
    # 5. Negative divisor: 1.0 / -2.0 = -0.5
    cases.append((q8(1.0), q8(-2.0)))
    # 6. Both negative: -1.0 / -2.0 = 0.5
    cases.append((q8(-1.0), q8(-2.0)))
    # 7. Saturates positive: 100.0 / 0.5 = 200.0 > 127.996
    cases.append((q8(100.0), q8(0.5)))
    # 8. Saturates negative: -100.0 / 0.5 = -200.0 < -128.0
    cases.append((q8(-100.0), q8(0.5)))
    # 9. Divide by zero (a > 0)
    cases.append((q8(1.0), 0))
    # 10. Rounding: 3 / 2 in Q8 = 768 / 512. (768*256)/512 = 384 = 1.5 exact
    # Pick a case with fractional result: 1 / 3 = 0.333...
    # a=256, b=768. (256*256)/768 = 65536/768 = 85.333... -> truncate to 85
    cases.append((q8(1.0), q8(3.0)))

    return [(a, b, fxdiv_ref(a, b)) for a, b in cases]

def emit_binary(path, vectors):
    buf = bytearray()
    buf += b"FD"           # magic
    buf += bytes([0x01, 0x00])
    buf += struct.pack(">H", len(vectors))
    buf += struct.pack(">H", 0)  # reserved
    # Each record: a[2], b[2], expected[2] = 6 bytes
    for (a, b, exp) in vectors:
        buf += struct.pack(">h", a)
        buf += struct.pack(">h", b)
        buf += struct.pack(">h", exp)
    Path(path).write_bytes(bytes(buf))
    return len(buf)

def self_test():
    assert fxdiv_ref(q8(1.0), q8(1.0)) == q8(1.0)
    assert fxdiv_ref(q8(1.0), q8(2.0)) == q8(0.5)
    assert fxdiv_ref(q8(0.5), q8(0.25)) == q8(2.0)
    assert fxdiv_ref(q8(-1.0), q8(2.0)) == q8(-0.5)
    assert fxdiv_ref(q8(1.0), q8(-2.0)) == q8(-0.5)
    assert fxdiv_ref(q8(-1.0), q8(-2.0)) == q8(0.5)
    assert fxdiv_ref(q8(1.0), 0) == 32767
    assert fxdiv_ref(q8(-1.0), 0) == -32768
    assert fxdiv_ref(0, 0) == 0
    # 1/3: (256*256)/768 = 85.333 -> 85
    assert fxdiv_ref(q8(1.0), q8(3.0)) == 85
    print("self_test: PASS")

def main():
    self_test()
    vectors = gen_vectors()
    out_dir = Path("test")
    out_dir.mkdir(exist_ok=True)
    nbytes = emit_binary(out_dir / "vectors_fxdiv.bin", vectors)
    print(f"wrote test/vectors_fxdiv.bin ({nbytes} bytes)")
    print(f"  FXDIV vectors: {len(vectors)}")
    for i, (a, b, exp) in enumerate(vectors):
        print(f"  [{i}] a={a:6d} b={b:6d} exp={exp:6d}")

if __name__ == "__main__":
    main()
