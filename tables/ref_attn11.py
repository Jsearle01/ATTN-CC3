#!/usr/bin/env python3
"""
ref_attn11.py — Bit-exact reference generator for ATTN/11 port FXMATH.

Produces test/vectors.bin containing (input, input, expected) triples
for FX_MUL_Q8Q15 and FX_MUL_Q15Q15. The 6309 test harness loads this
file via .incbin and compares each case byte-for-byte.

All arithmetic uses Python int only. No floats in the arithmetic path.
Truncation semantics match ATTN/11's ASHC #-n: drop low bits, no rounding.

Output format (big-endian throughout, matches 6309 native byte order):

  Header (8 bytes):
    magic       2 bytes  "FX"
    version     1 byte   0x01
    pad         1 byte
    n_mul8q15   2 bytes  count of Q8×Q15 records
    n_mul15q15  2 bytes  count of Q15×Q15 records

  MUL8Q15 record (5 bytes each):
    a_q8        1 byte   signed Q8 input
    b_q15       2 bytes  signed Q15 input, big-endian
    expected    2 bytes  signed Q15 result, big-endian

  MUL15Q15 record (6 bytes each):
    a_q15       2 bytes
    b_q15       2 bytes
    expected    2 bytes
"""

import struct
from pathlib import Path

# ---- Q-format constants ------------------------------------------------

Q8_MIN,  Q8_MAX  = -128, 127
Q15_MIN, Q15_MAX = -32768, 32767

def clamp_s8(x):
    if x < Q8_MIN or x > Q8_MAX:
        raise ValueError(f"Q8 overflow: {x}")
    return x

def clamp_s16(x):
    if x < Q15_MIN or x > Q15_MAX:
        raise ValueError(f"Q15 overflow: {x}")
    return x

def arith_shr(x, n):
    """Arithmetic shift right — Python >> on negative ints already floors
    toward -infinity, matching PDP-11 ASHC #-n and 6309 arithmetic shift."""
    return x >> n

# ---- Q8 × Q15 → Q15 (truncation) --------------------------------------

def fx_mul_q8q15(a_q8, b_q15):
    """Matches 6309 FX_MUL_Q8Q15 and PDP-11 ASHC #-8 semantics.
    Raw product is Q23 in 32-bit; take bits [23:8] with 16-bit wraparound."""
    clamp_s8(a_q8)
    clamp_s16(b_q15)
    product = a_q8 * b_q15                     # Q23 scale
    result = arith_shr(product, 8)             # drop low byte
    # 16-bit signed wraparound to match LDD-from-scratch behavior on 6309.
    result = ((result + 0x8000) & 0xFFFF) - 0x8000
    return result

# ---- Q15 × Q15 → Q15 (truncation) -------------------------------------

def fx_mul_q15q15(a_q15, b_q15):
    """Matches 6309 FX_MUL_Q15Q15: MULD; STQ; LDD high word; LSLD.
    Raw product is Q30; high 16 bits are Q14, shift left 1 → Q15."""
    clamp_s16(a_q15)
    clamp_s16(b_q15)
    product = a_q15 * b_q15                    # Q30 scale
    high16 = arith_shr(product, 16)            # Q14 value
    shifted = (high16 << 1) & 0xFFFF           # LSLD, 16-bit wrap
    if shifted & 0x8000:
        shifted -= 0x10000                     # back to signed
    return shifted

# ---- Test vector generation -------------------------------------------

def gen_mul8q15_vectors():
    """Coverage: zero cases, sign combinations, boundaries, typical ranges."""
    vectors = []
    vectors += [(0, 0), (0, 12345), (0, -12345), (42, 0), (-42, 0)]
    vectors += [(1, 256), (-1, 256), (1, -256), (-1, -256)]
    vectors += [(64, 16384), (-64, 16384), (64, -16384), (-64, -16384)]
    vectors += [(127, 32767), (-128, 32767), (127, -32768), (-128, -32768)]
    vectors += [(127, 1), (-128, 1), (1, 32767), (1, -32768)]
    vectors += [(12, 4096), (-12, 4096), (25, -8192), (-25, 8192)]
    vectors += [(3, 300), (-3, 300), (15, -1500), (-15, 1500)]
    vectors += [(1, 1), (-1, -1), (1, -1), (-1, 1)]
    vectors += [(2, 128), (-2, 128), (2, -128), (-2, -128)]
    return [(a, b, fx_mul_q8q15(a, b)) for (a, b) in vectors]

def gen_mul15q15_vectors():
    """Similar coverage for Q15×Q15 backward-pass multiply."""
    vectors = []
    vectors += [(0, 0), (0, 12345), (12345, 0)]
    vectors += [(16384, 16384), (-16384, 16384), (16384, -16384), (-16384, -16384)]
    vectors += [(32767, 32767), (-32768, 32767), (32767, -32768), (-32768, -32768)]
    vectors += [(1, 1), (-1, -1), (1, -1), (-1, 1)]
    vectors += [(8192, 4096), (-8192, 4096), (8192, -4096), (-8192, -4096)]
    vectors += [(1000, 2000), (-1000, 2000), (5000, -3000), (-5000, -3000)]
    vectors += [(10, 10), (-10, 10), (100, 100), (-100, -100)]
    return [(a, b, fx_mul_q15q15(a, b)) for (a, b) in vectors]

# ---- Binary emission --------------------------------------------------

def emit_binary(path, m8, m15):
    buf = bytearray()
    buf += b"FX"
    buf += bytes([0x01, 0x00])
    buf += struct.pack(">H", len(m8))
    buf += struct.pack(">H", len(m15))
    for (a, b, r) in m8:
        buf += struct.pack(">b", a)
        buf += struct.pack(">h", b)
        buf += struct.pack(">h", r)
    for (a, b, r) in m15:
        buf += struct.pack(">h", a)
        buf += struct.pack(">h", b)
        buf += struct.pack(">h", r)
    Path(path).write_bytes(bytes(buf))
    return len(buf)

def emit_log(path, m8, m15):
    lines = []
    lines.append("# FX_MUL_Q8Q15 vectors")
    lines.append("# a_q8   b_q15   expected_q15   (a_real * b_real = r_real)")
    for (a, b, r) in m8:
        a_real = a / 256.0
        b_real = b / 32768.0
        r_real = r / 32768.0
        lines.append(
            f"  {a:5d}  {b:7d}  {r:7d}   "
            f"({a_real:+.5f} * {b_real:+.5f} = {r_real:+.5f})"
        )
    lines.append("")
    lines.append("# FX_MUL_Q15Q15 vectors")
    lines.append("# a_q15   b_q15   expected_q15")
    for (a, b, r) in m15:
        a_real = a / 32768.0
        b_real = b / 32768.0
        r_real = r / 32768.0
        lines.append(
            f"  {a:7d}  {b:7d}  {r:7d}   "
            f"({a_real:+.5f} * {b_real:+.5f} = {r_real:+.5f})"
        )
    Path(path).write_text("\n".join(lines) + "\n")

# ---- Sanity self-tests before emission --------------------------------

def self_test():
    """Hand-verified cases where bit math can be traced mentally."""
    # 0.25 Q8 × 0.25 Q15 = 0.0625: 64 * 8192 = 524288, >>8 = 2048 = 0.0625 Q15
    assert fx_mul_q8q15(64, 8192) == 2048
    assert fx_mul_q8q15(-64, 8192) == -2048
    assert fx_mul_q8q15(0, 12345) == 0
    assert fx_mul_q8q15(42, 0) == 0
    # 0.5 Q15 × 0.5 Q15 = 0.25 Q15: 16384² = 268435456, >>16 = 4096, <<1 = 8192
    assert fx_mul_q15q15(16384, 16384) == 8192
    assert fx_mul_q15q15(-16384, -16384) == 8192
    assert fx_mul_q15q15(-16384, 16384) == -8192
    print("self_test: PASS")

# ---- Main -------------------------------------------------------------

def main():
    self_test()
    m8 = gen_mul8q15_vectors()
    m15 = gen_mul15q15_vectors()
    out_dir = Path("test")
    out_dir.mkdir(exist_ok=True)
    nbytes = emit_binary(out_dir / "vectors.bin", m8, m15)
    emit_log(out_dir / "vectors.txt", m8, m15)
    print(f"wrote test/vectors.bin ({nbytes} bytes)")
    print(f"  FX_MUL_Q8Q15  vectors: {len(m8)}")
    print(f"  FX_MUL_Q15Q15 vectors: {len(m15)}")

if __name__ == "__main__":
    main()
