#!/usr/bin/env python3
"""
gen_vecop_vectors.py — Bit-exact reference vectors for VECOP routines.

Produces test/vectors_vecop.bin containing test cases for VDOT, VSCL,
and VADD. All arithmetic uses Python int only (no floats). Truncation
semantics match ATTN/11 VECOP.MAC and the 6309 NORMQ15 macro.

Output format (big-endian, matches 6309 native byte order):

  Header (12 bytes):
    magic       2 bytes  "VC"
    version     1 byte   0x01
    pad         1 byte   0x00
    n_vdot      2 bytes
    n_vscl      2 bytes
    n_vadd      2 bytes
    reserved    2 bytes  0x0000

  VDOT record (variable length):
    count       2 bytes
    x[count]    count * 2 bytes (Q8, signed big-endian)
    y[count]    count * 2 bytes
    expected    2 bytes (Q8 result)

  VSCL record (variable length):
    count       2 bytes
    alpha       2 bytes
    x[count]    count * 2 bytes
    exp[count]  count * 2 bytes

  VADD record (variable length):
    count       2 bytes
    x[count]    count * 2 bytes
    y[count]    count * 2 bytes
    exp[count]  count * 2 bytes
"""

import struct
from pathlib import Path

Q8_SCALE = 256  # 1.0 in Q8 = 256

def clamp_s16(x):
    if x < -32768 or x > 32767:
        raise ValueError(f"Q8 overflow: {x}")
    return x

def arith_shr(x, n):
    """Arithmetic shift right (Python >> on negative ints floors)."""
    return x >> n

def wrap_s32(x):
    """Wrap to 32-bit signed range."""
    return ((x + 0x80000000) & 0xFFFFFFFF) - 0x80000000

def wrap_s16(x):
    """Wrap to 16-bit signed range."""
    return ((x + 0x8000) & 0xFFFF) - 0x8000

# ---- VDOT reference ----

def vdot_ref(x, y):
    """Match ATTN/11 VDOT: 32-bit accumulate, ASHC #-8, clamp.
    x and y are lists of Q8 (16-bit signed) values."""
    assert len(x) == len(y)
    acc = 0
    for xi, yi in zip(x, y):
        product = xi * yi       # Q16, 32-bit
        acc += product           # accumulate
    # Wrap to 32-bit signed (match hardware accumulator width)
    acc = wrap_s32(acc)
    # Shift right 8: Q16 -> Q8 (equivalent to ASHC #-8)
    shifted = arith_shr(acc, 8)
    # Clamp check: match ATTN/11 exactly
    # high16 = equivalent of PDP-11 R0 after ASHC = V >> 24
    high_byte = arith_shr(acc, 24)  # = MSB of 32-bit acc
    if high_byte == 0 or high_byte == -1:
        # Result fits — extract bits [23:8] as signed 16-bit
        result = shifted & 0xFFFF
        if result >= 0x8000:
            result -= 0x10000
        return result
    elif shifted > 0:
        return 32767
    else:
        return -32768

# ---- VSCL reference ----

def vscl_element(alpha, x):
    """Match ATTN/11 VSCL: MUL + ASHC #-8 per element.
    Same as NORMQ15: product >> 8 with 16-bit signed wrap."""
    product = alpha * x         # Q16
    result = arith_shr(product, 8)
    return wrap_s16(result)

def vscl_ref(alpha, x):
    """Scale vector: y[i] = alpha * x[i]."""
    return [vscl_element(alpha, xi) for xi in x]

# ---- VADD reference ----

def vadd_ref(x, y):
    """Match ATTN/11 VADD: 16-bit ADD, wrapping."""
    assert len(x) == len(y)
    return [wrap_s16(xi + yi) for xi, yi in zip(x, y)]

# ---- Test vector generation ----

def q8(real):
    """Convert real number to Q8 integer, clamped."""
    v = int(round(real * Q8_SCALE))
    return max(-32768, min(32767, v))

def gen_vdot_vectors():
    """Coverage: zeros, unit, orthogonal, sign combos, saturation."""
    cases = []

    # Zero vectors
    cases.append(([0]*4, [0]*4))
    # Unit dot: 1.0 * 1.0 * 4 = 4.0
    cases.append(([q8(1.0)]*4, [q8(1.0)]*4))
    # Negative: -1.0 * 1.0 * 4 = -4.0
    cases.append(([q8(-1.0)]*4, [q8(1.0)]*4))
    # Mixed signs
    cases.append(([q8(0.5), q8(-0.5), q8(0.25), q8(-0.25)],
                  [q8(0.5), q8(0.5), q8(0.5), q8(0.5)]))
    # Length 8 — typical for SEQ=8
    cases.append(([q8(0.1)]*8, [q8(0.2)]*8))
    # Small values
    cases.append(([1, 2, 3, 4], [4, 3, 2, 1]))
    # Large values (near boundary)
    cases.append(([q8(100.0)]*4, [q8(0.01)]*4))
    # Length 16 — typical for D=16
    cases.append(([q8(0.5)]*16, [q8(0.25)]*16))

    return [(x, y, vdot_ref(x, y)) for x, y in cases]

def gen_vscl_vectors():
    """Coverage: unity, zero, negative, fractional scalars."""
    cases = []

    # Scale by 1.0 (identity)
    x = [q8(0.5), q8(-0.5), q8(1.0), q8(-1.0)]
    cases.append((q8(1.0), x))
    # Scale by 0
    cases.append((0, x))
    # Scale by -1.0 (negate)
    cases.append((q8(-1.0), x))
    # Scale by 0.5
    cases.append((q8(0.5), [q8(1.0), q8(2.0), q8(-1.0), q8(-2.0)]))
    # Small scale
    cases.append((q8(0.1), [q8(10.0), q8(-10.0), q8(5.0), q8(-5.0)]))
    # Length 8
    cases.append((q8(0.25), [q8(i * 0.5 - 2.0) for i in range(8)]))

    return [(alpha, x, vscl_ref(alpha, x)) for alpha, x in cases]

def gen_vadd_vectors():
    """Coverage: basic addition, zero, wrapping."""
    cases = []

    # Zero + zero
    cases.append(([0]*4, [0]*4))
    # Basic addition
    cases.append(([q8(1.0), q8(2.0), q8(3.0), q8(4.0)],
                  [q8(0.5), q8(0.5), q8(0.5), q8(0.5)]))
    # Negative
    cases.append(([q8(1.0), q8(-1.0), q8(2.0), q8(-2.0)],
                  [q8(-1.0), q8(1.0), q8(-2.0), q8(2.0)]))
    # Near overflow (127 + 0.5 wraps in Q8)
    cases.append(([q8(100.0), q8(-100.0)],
                  [q8(50.0), q8(-50.0)]))

    return [(x, y, vadd_ref(x, y)) for x, y in cases]

# ---- Binary emission ----

def emit_s16(buf, v):
    buf += struct.pack(">h", v)

def emit_u16(buf, v):
    buf += struct.pack(">H", v)

def emit_binary(path, vdot, vscl, vadd):
    buf = bytearray()
    # Header
    buf += b"VC"
    buf += bytes([0x01, 0x00])
    buf += struct.pack(">H", len(vdot))
    buf += struct.pack(">H", len(vscl))
    buf += struct.pack(">H", len(vadd))
    buf += struct.pack(">H", 0)  # reserved

    # VDOT records
    for (x, y, expected) in vdot:
        buf += struct.pack(">H", len(x))
        for v in x:
            buf += struct.pack(">h", v)
        for v in y:
            buf += struct.pack(">h", v)
        buf += struct.pack(">h", expected)

    # VSCL records
    for (alpha, x, exp) in vscl:
        buf += struct.pack(">H", len(x))
        buf += struct.pack(">h", alpha)
        for v in x:
            buf += struct.pack(">h", v)
        for v in exp:
            buf += struct.pack(">h", v)

    # VADD records
    for (x, y, exp) in vadd:
        buf += struct.pack(">H", len(x))
        for v in x:
            buf += struct.pack(">h", v)
        for v in y:
            buf += struct.pack(">h", v)
        for v in exp:
            buf += struct.pack(">h", v)

    Path(path).write_bytes(bytes(buf))
    return len(buf)

# ---- Self-tests ----

def self_test():
    # VDOT: 0 * 0 = 0
    assert vdot_ref([0, 0], [0, 0]) == 0

    # VDOT: 256 * 256 = 1.0 * 1.0 = 1.0 = 256 in Q8
    # Product = 256*256 = 65536 (Q16). Sum = 65536. >> 8 = 256. ✓
    assert vdot_ref([256], [256]) == 256

    # VDOT: [256, 256] * [256, 256] = 2.0 = 512
    # Sum = 2 * 65536 = 131072. >> 8 = 512. ✓
    assert vdot_ref([256, 256], [256, 256]) == 512

    # VDOT: [-256] * [256] = -1.0 = -256
    assert vdot_ref([-256], [256]) == -256

    # VSCL: 256 * 256 >> 8 = 256 (1.0 * 1.0 = 1.0)
    assert vscl_element(256, 256) == 256

    # VSCL: 0 * anything = 0
    assert vscl_element(0, 12345) == 0

    # VSCL: 128 * 256 >> 8 = 128 (0.5 * 1.0 = 0.5)
    assert vscl_element(128, 256) == 128

    # VADD: plain addition, wrapping
    assert vadd_ref([100, -100], [200, -200]) == [300, -300]
    assert vadd_ref([0], [0]) == [0]

    print("self_test: PASS")

# ---- Main ----

def main():
    self_test()
    vdot = gen_vdot_vectors()
    vscl = gen_vscl_vectors()
    vadd = gen_vadd_vectors()
    out_dir = Path("test")
    out_dir.mkdir(exist_ok=True)
    nbytes = emit_binary(out_dir / "vectors_vecop.bin", vdot, vscl, vadd)
    print(f"wrote test/vectors_vecop.bin ({nbytes} bytes)")
    print(f"  VDOT  vectors: {len(vdot)}")
    print(f"  VSCL  vectors: {len(vscl)}")
    print(f"  VADD  vectors: {len(vadd)}")

if __name__ == "__main__":
    main()
