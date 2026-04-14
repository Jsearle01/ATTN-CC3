#!/usr/bin/env python3
"""
gen_vsadd_vectors.py — Reference vectors for VSADD.

VSADD reference uses TWO-STAGE clamp (matches ATTN/11, differs from MATOP).
  Stage 1: clamp (scalar * src[k]) >> 8 to Q8 range.
  Stage 2: clamp dst[k] + stage1 to Q8 range.
See DEVIATIONS.md for rationale on why VSADD differs from MVADD/VTMUL/OUTER.

For tests where the two-stage and single-clamp paths diverge, the
single-clamp value is emitted as a comment in vsadd_vectors.asm so
a future regression to single-clamp would fail loudly against the
two-stage expected value with the single-clamp value visible nearby.

Output: vsadd_vectors.asm (include file, FDB blocks, no leading spaces).
Companion log: vsadd_vectors.log (per-element computation trace for
tests that benefit from spot-checking — divergence test and mixed test).
"""

from pathlib import Path

Q8_MIN = -32768
Q8_MAX =  32767

def clamp_q8(x):
    """Saturating clamp to Q8 16-bit signed range."""
    if x > Q8_MAX: return Q8_MAX
    if x < Q8_MIN: return Q8_MIN
    return x

def arith_shr_8(x):
    """Arithmetic right shift by 8, floor semantics (matches 6309 MULD
    + STQ + LDD_mid byte-extraction for signed 32-bit products).

    Python's >> on int is arithmetic with floor-toward-negative-infinity,
    which is what we want:
      -1 >> 8 = -1  (floor(-0.004) = -1)
      -128 >> 8 = -1  (floor(-0.5) = -1)
      -256 >> 8 = -1  (floor(-1.0) = -1)
      -257 >> 8 = -2  (floor(-1.004) = -2)
    """
    return x >> 8

def vsadd_two_stage(dst_init, src, scalar, length):
    """VSADD reference: two-stage clamp."""
    result = []
    for k in range(length):
        product = scalar * src[k]           # signed 32-bit in Q8*Q8 range
        shifted = arith_shr_8(product)      # arithmetic >>8
        stage1 = clamp_q8(shifted)          # stage 1 clamp
        stage2 = clamp_q8(dst_init[k] + stage1)
        result.append(stage2)
    return result

def vsadd_single_clamp(dst_init, src, scalar, length):
    """Hypothetical single-clamp variant for divergence comparison.
    NOT our implementation — shown as comment only in divergence tests."""
    result = []
    for k in range(length):
        product = scalar * src[k]
        shifted = arith_shr_8(product)
        # No stage-1 clamp; add to dst then clamp the sum
        result.append(clamp_q8(dst_init[k] + shifted))
    return result

# ---- Test cases ----

def gen_tests():
    """Returns list of (name, dst_init, src, scalar, length, expected, log_detail)."""
    tests = []

    # Test 0: VSADD_ZERO_SCL — scalar = 0, dst unchanged
    tests.append((
        "VSADD_ZERO_SCL",
        [100, 200, 300, 400],
        [1000, 2000, 3000, 4000],
        0,
        4,
        False  # don't log per-element (obvious)
    ))

    # Test 1: VSADD_UNIT_SCL — scalar = 256 (1.0 in Q8), dst += src
    tests.append((
        "VSADD_UNIT_SCL",
        [100, 200, -300, 400],
        [50, 60, 70, -80],
        256,
        4,
        False
    ))

    # Test 2: VSADD_POS_NOSAT — small values, no saturation
    tests.append((
        "VSADD_POS_NOSAT",
        [100, 200, 300, 400],
        [50, 60, 70, 80],
        128,  # 0.5 in Q8
        4,
        False
    ))

    # Test 3: VSADD_NEG_NOSAT — negative scalar, mixed-sign src, sign-preserving shift
    tests.append((
        "VSADD_NEG_NOSAT",
        [1000, -1000, 500, -500],
        [100, -100, 200, -200],
        -128,  # -0.5 in Q8
        4,
        False
    ))

    # Test 4: VSADD_PRODUCT_SAT — DIVERGENCE TEST
    # product overflows Q8 after >>8, but sum (with clamped stage1) fits.
    # scalar=20000, src=20000 -> product=400,000,000, >>8=1,562,500 (overflows Q8)
    # stage1 clamped to 32767, dst=-20000, sum=12767 (fits)
    # single-clamp would give: dst + 1,562,500 -> clamped to 32767
    tests.append((
        "VSADD_PRODUCT_SAT",
        [-20000],
        [20000],
        20000,
        1,
        True  # log per-element
    ))

    # Test 5: VSADD_SUM_SAT — stage1 in range, but sum saturates
    # scalar=200, src=10000 -> product=2,000,000, >>8=7812 (fits in Q8)
    # dst=30000, sum=30000+7812=37812, clamped to 32767
    tests.append((
        "VSADD_SUM_SAT",
        [30000],
        [10000],
        200,
        1,
        False
    ))

    # Test 6: VSADD_BOTH_SAT — stage1 saturates, sum also would saturate
    # scalar=20000, src=20000 -> stage1 = 32767
    # dst=20000, sum=20000+32767=52767, clamped to 32767
    tests.append((
        "VSADD_BOTH_SAT",
        [20000],
        [20000],
        20000,
        1,
        False
    ))

    # Test 7: VSADD_LEN0 — no-op
    tests.append((
        "VSADD_LEN0",
        [1234, 5678, -4321, 9999],
        [100, 200, 300, 400],
        128,
        0,
        False
    ))

    # Test 8: VSADD_LEN1 — single-element update
    tests.append((
        "VSADD_LEN1",
        [100, 9999],   # dst[1] is sentinel; harness checks bounds
        [200, 0],
        128,  # 0.5
        1,
        False
    ))

    # Test 9: VSADD_LEN16_MIXED — 16 elements, mixed signs and saturation
    dst9 = [100, -200, 300, -400, 500, -600, 700, -800,
            20000, -20000, 15000, -15000, 100, 200, 300, 400]
    src9 = [1000, 2000, -1500, 2500, -3000, 3500, -4000, 4500,
            20000, 20000, -20000, -20000, 0, 100, -100, 50]
    scl9 = 200  # ~0.78 in Q8
    tests.append((
        "VSADD_LEN16_MIXED",
        dst9, src9, scl9, 16,
        True  # log per-element
    ))

    # Compute expected for each test
    result = []
    for (name, dst, src, scl, length, log) in tests:
        exp = vsadd_two_stage(dst, src, scl, length)
        sc  = vsadd_single_clamp(dst, src, scl, length)
        result.append((name, dst, src, scl, length, exp, sc, log))
    return result

# ---- Emit vsadd_vectors.asm ----

def emit_asm(path, tests):
    lines = []
    lines.append("* ============================================================")
    lines.append("* vsadd_vectors.asm - test vectors for VSADD harness")
    lines.append("* Generated by gen_vsadd_vectors.py - do not hand-edit")
    lines.append("* Two-stage clamp semantics (matches VSADD, not MATOP)")
    lines.append("* ============================================================")
    lines.append("")

    for i, (name, dst, src, scl, length, exp, sc, _log) in enumerate(tests):
        diverge = exp != sc
        lines.append(f"* Test {i}: {name}")
        if diverge:
            lines.append(f"*   DIVERGENCE from single-clamp semantics:")
            # Scalar formatting for single-element tests, list for multi-element
            if len(exp) == 1:
                lines.append(f"*   two-stage (ours):  {exp[0]}")
                lines.append(f"*   single-clamp:      {sc[0]}")
            else:
                lines.append(f"*   two-stage (ours):  {exp}")
                lines.append(f"*   single-clamp:      {sc}")
        # FDB blocks — no leading spaces on values
        if length > 0:
            dst_s = ",".join(str(v) for v in dst[:max(length, len(dst))])
            src_s = ",".join(str(v) for v in src[:max(length, len(src))])
            exp_s = ",".join(str(v) for v in exp) if exp else "0"
        else:
            dst_s = ",".join(str(v) for v in dst)
            src_s = ",".join(str(v) for v in src)
            exp_s = ",".join(str(v) for v in dst)  # no-op: expect dst unchanged

        lines.append(f"DST_{i}_INIT     FDB {dst_s}")
        lines.append(f"SRC_{i}          FDB {src_s}")
        lines.append(f"SCL_{i}          FDB {scl}")
        lines.append(f"LEN_{i}          FDB {length}")
        if length == 0:
            lines.append(f"EXPECT_{i}       FDB {exp_s}  ; same as DST_{i}_INIT (no-op)")
        else:
            lines.append(f"EXPECT_{i}       FDB {exp_s}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")
    return len(lines)

# ---- Emit log with per-element trace ----

def emit_log(path, tests):
    lines = []
    lines.append("VSADD reference generator log")
    lines.append("Two-stage clamp semantics")
    lines.append("=" * 60)
    lines.append("")
    for i, (name, dst, src, scl, length, exp, sc, log) in enumerate(tests):
        lines.append(f"Test {i}: {name}")
        lines.append(f"  dst_init = {dst[:length] if length > 0 else dst}")
        lines.append(f"  src      = {src[:length] if length > 0 else src}")
        lines.append(f"  scalar   = {scl}")
        lines.append(f"  length   = {length}")
        lines.append(f"  expected = {exp}")
        if exp != sc:
            lines.append(f"  single-clamp (would be): {sc}")
            lines.append(f"  *** DIVERGENCE ***")
        if log and length > 0:
            lines.append("  Per-element trace:")
            for k in range(length):
                product = scl * src[k]
                shifted = arith_shr_8(product)
                stage1 = clamp_q8(shifted)
                stage2 = clamp_q8(dst[k] + stage1)
                lines.append(f"    [{k}] dst={dst[k]}, src={src[k]}, "
                             f"prod={product}, >>8={shifted}, "
                             f"stage1={stage1}, final={stage2}")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")

# ---- Self tests ----

def self_test():
    # arith_shr_8 sanity
    assert arith_shr_8(-1) == -1
    assert arith_shr_8(-128) == -1
    assert arith_shr_8(-256) == -1
    assert arith_shr_8(-257) == -2
    assert arith_shr_8(0) == 0
    assert arith_shr_8(255) == 0
    assert arith_shr_8(256) == 1

    # Divergence test values
    # scalar=20000, src=20000, dst=-20000, length=1
    # product = 400,000,000, >>8 = 1,562,500
    # two-stage: clamp(1,562,500) = 32767; dst + 32767 = 12767
    # single:    dst + 1,562,500 = 1,542,500; clamp = 32767
    assert vsadd_two_stage([-20000], [20000], 20000, 1) == [12767]
    assert vsadd_single_clamp([-20000], [20000], 20000, 1) == [32767]

    # Zero scalar: dst unchanged
    assert vsadd_two_stage([1, 2, 3], [99, 99, 99], 0, 3) == [1, 2, 3]

    # Unit scalar (256 = 1.0 in Q8): dst += src exactly
    # product = 256*50 = 12800, >>8 = 50, stage1=50, final = 100+50 = 150
    assert vsadd_two_stage([100], [50], 256, 1) == [150]

    print("self_test: PASS")

# ---- Main ----

def main():
    self_test()
    tests = gen_tests()
    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "vsadd_vectors.asm", tests)
    emit_log(out_dir / "vsadd_vectors.log", tests)
    print(f"wrote tables/vsadd_vectors.asm ({len(tests)} tests)")
    print(f"wrote tables/vsadd_vectors.log (per-element trace)")
    print()
    print("Summary:")
    divergences = 0
    for i, (name, _, _, _, _, exp, sc, _) in enumerate(tests):
        if exp != sc:
            print(f"  Test {i} ({name}): DIVERGENCE two-stage={exp} vs single-clamp={sc}")
            divergences += 1
        else:
            print(f"  Test {i} ({name}): no divergence (expected={exp})")
    print()
    print(f"Total divergence tests: {divergences}")
    if divergences == 0:
        raise SystemExit("ERROR: No divergence tests found. "
                         "VSADD regression to single-clamp would not be caught!")

if __name__ == "__main__":
    main()
