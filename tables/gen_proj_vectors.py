#!/usr/bin/env python3
"""
PROJ reference generator.

Produces tables/proj_vectors.asm with 5 test cases.

PROJ semantics (per-product shift-accumulate, matches ATTN/11):
    for each (position i, vocab v):
        acc = 0
        for d in 0..DIM-1:
            product = Y[i][d] * Wout[d][v]
            stage1 = clamp_q8( arith_shift_right_8(product) )
            acc = clamp_q8( acc + stage1 )
        logits[i][v] = acc

Two-stage clamp per step: clamp shifted product, then clamp running sum.

Matches ATTN/11 MATOP.MAC VTMUL (lines 162-181) and prototype.shf vtmul
(lines 50-52). The PDP-11 header explicitly documents "Per-product Q8
rounding (acceptable for d_model=16)" as the deliberate precision
tradeoff. MVMUL uses a different pattern (full 32-bit row accumulation,
single end-of-row clamp). VTMUL does not.

Wout layout: [d_model][vocab] row-major. Element Wout[d][v] is at
offset (d * VOC + v) * 2 bytes from WOUT base.

Generator history note: initial version used full-sum-then-shift
semantics, which produced EXPECT values 1 LSB too high (273 vs 272
on test 0 v=0). Corrected to match actual VTMUL behavior. 6309
VTMUL and PROJ were always correct; only this generator needed fix.
"""

from pathlib import Path

def arith_shift_right_8(x):
    """Floor-toward-minus-infinity for signed values, matches 6309 ASR semantics."""
    if x >= 0:
        return x // 256
    else:
        return -((-x + 255) // 256)

def clamp_q8(x):
    """Clamp to signed 16-bit Q8 range."""
    if x > 32767: return 32767
    if x < -32768: return -32768
    return x

def proj_ref(Y, Wout, seq, dim, voc):
    """
    Per-product shift-accumulate matching ATTN/11 VTMUL and prototype.shf.
    Two-stage clamp per step: clamp shifted product, then clamp running sum.

    For each (i, v):
      acc = 0
      for d in 0..DIM-1:
        product = Y[i][d] * Wout[d][v]
        stage1 = clamp_q8( arith_shift_right_8(product) )
        acc = clamp_q8( acc + stage1 )
      logits[i][v] = acc

    Y: list of seq lists, each of length dim.
    Wout: list of dim lists, each of length voc.
    Returns: flat list of seq*voc logits (row-major: position-major).
    """
    logits = []
    for i in range(seq):
        for v in range(voc):
            acc = 0
            for d in range(dim):
                product = Y[i][d] * Wout[d][v]
                stage1 = clamp_q8(arith_shift_right_8(product))
                acc = clamp_q8(acc + stage1)
            logits.append(acc)
    return logits

def proj_ref_with_trace(Y, Wout, seq, dim, voc):
    """Like proj_ref but returns list of (i, v, [per-step tuples], final)."""
    results = []
    for i in range(seq):
        for v in range(voc):
            acc = 0
            steps = []
            for d in range(dim):
                product = Y[i][d] * Wout[d][v]
                shifted = arith_shift_right_8(product)
                stage1 = clamp_q8(shifted)
                acc_new = clamp_q8(acc + stage1)
                steps.append((d, Y[i][d], Wout[d][v], product, shifted, stage1, acc_new))
                acc = acc_new
            results.append((i, v, steps, acc))
    return results

# ---- Self-test ----

def self_test():
    # arith_shift_right_8 probes
    assert arith_shift_right_8(-1) == -1, f"ASR8(-1) should be -1, got {arith_shift_right_8(-1)}"
    assert arith_shift_right_8(-128) == -1, f"ASR8(-128) should be -1, got {arith_shift_right_8(-128)}"
    assert arith_shift_right_8(-256) == -1, f"ASR8(-256) should be -1, got {arith_shift_right_8(-256)}"
    assert arith_shift_right_8(-257) == -2, f"ASR8(-257) should be -2, got {arith_shift_right_8(-257)}"
    assert arith_shift_right_8(0) == 0
    assert arith_shift_right_8(255) == 0
    assert arith_shift_right_8(256) == 1

    # clamp_q8
    assert clamp_q8(0) == 0
    assert clamp_q8(40000) == 32767
    assert clamp_q8(-40000) == -32768
    assert clamp_q8(32767) == 32767
    assert clamp_q8(-32768) == -32768

    # Trivial proj_ref
    # Y=[[256]], Wout=[[256]], result = (256*256) >> 8 = 256
    r = proj_ref([[256]], [[256]], 1, 1, 1)
    assert r == [256], f"trivial: {r}"

    print("Self-test passed: arith_shift_right_8, clamp_q8, proj_ref")

# ---- Safety net ----

def safety_check(test_name, Y, Wout, seq, dim, voc, saturation_required):
    # Architecture limits
    assert seq <= 8, f"{test_name}: SEQ {seq} exceeds arch max 8"
    assert dim <= 16, f"{test_name}: DIM {dim} exceeds arch max 16"
    assert voc <= 10, f"{test_name}: VOC {voc} exceeds arch max 10"

    # Dimensions consistent
    assert len(Y) == seq, f"{test_name}: len(Y)={len(Y)} != SEQ={seq}"
    for i, row in enumerate(Y):
        assert len(row) == dim, f"{test_name}: len(Y[{i}])={len(row)} != DIM={dim}"
    assert len(Wout) == dim, f"{test_name}: len(Wout)={len(Wout)} != DIM={dim}"
    for d, row in enumerate(Wout):
        assert len(row) == voc, f"{test_name}: len(Wout[{d}])={len(row)} != VOC={voc}"

    # Input range: Q8 signed 16-bit
    for i, row in enumerate(Y):
        for d, v in enumerate(row):
            assert -32768 <= v <= 32767, f"{test_name}: Y[{i}][{d}]={v} out of Q8 range"
    for d, row in enumerate(Wout):
        for v, w in enumerate(row):
            assert -32768 <= w <= 32767, f"{test_name}: Wout[{d}][{v}]={w} out of Q8 range"

    # Saturation requirement (per-product semantics, matches VTMUL)
    # Clamp fires if: stage1 (shifted product) falls outside Q15, OR
    # running acc + stage1 exceeds Q15 at any step.
    if saturation_required:
        any_saturated = False
        where = None
        for i in range(seq):
            for v in range(voc):
                acc = 0
                for d in range(dim):
                    product = Y[i][d] * Wout[d][v]
                    shifted = arith_shift_right_8(product)
                    stage1_overflow = shifted > 32767 or shifted < -32768
                    stage1 = clamp_q8(shifted)
                    sum_raw = acc + stage1
                    acc_overflow = sum_raw > 32767 or sum_raw < -32768
                    acc = clamp_q8(sum_raw)
                    if stage1_overflow or acc_overflow:
                        any_saturated = True
                        where = (i, v, d, product, shifted, stage1,
                                 "stage1" if stage1_overflow else "acc")
                        break
                if any_saturated:
                    break
            if any_saturated:
                break
        if not any_saturated:
            raise ValueError(
                f"{test_name}: saturation_required but no element saturates. "
                f"Adjust inputs to ensure at least one (i,v) overflows Q15 "
                f"at stage1 or in the running acc.")
        return where
    return None

# ---- Test data ----

def q8(real):
    v = int(round(real * 256))
    return max(-32768, min(32767, v))

def test_0():
    """PROJ_SINGLE: SEQ=1 DIM=4 VOC=3, hand-verifiable."""
    Y = [[100, 200, 300, 400]]
    # Wout [DIM=4][VOC=3] row-major
    Wout = [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
        [100, 110, 120],
    ]
    return ("PROJ_SINGLE", Y, Wout, 1, 4, 3, False)

def test_1():
    """PROJ_SEQ4: SEQ=4 DIM=8 VOC=5, deterministic pattern."""
    # Y[i][d] = 10*i + d + 1
    Y = [[10*i + d + 1 for d in range(8)] for i in range(4)]
    # Wout[d][v] = d*3 + v + 2
    Wout = [[d*3 + v + 2 for v in range(5)] for d in range(8)]
    return ("PROJ_SEQ4", Y, Wout, 4, 8, 5, False)

def test_2():
    """PROJ_FULL: SEQ=8 DIM=16 VOC=10 (architecture max).
    Y[i][d] = (i*7 + d*3) % 200 - 100    (range [-100, 99])
    Wout[d][v] = (d*5 + v*2) % 100 - 50   (range [-50, 49])
    """
    Y = [[(i*7 + d*3) % 200 - 100 for d in range(16)] for i in range(8)]
    Wout = [[(d*5 + v*2) % 100 - 50 for v in range(10)] for d in range(16)]
    return ("PROJ_FULL", Y, Wout, 8, 16, 10, False)

def test_3():
    """PROJ_SAT_POS: at least one logit saturates positive.
    Y and Wout both near +30000 across all dimensions drives accumulator
    well past Q15*256.
    For DIM=4, single row, single vocab column:
      acc = 4 * 30000 * 30000 = 3.6e9
      shifted = 3.6e9 / 256 = 1.4e7 (overflows Q15)
    """
    Y = [[30000, 30000, 30000, 30000]]
    # Wout [DIM=4][VOC=3]: all columns saturating, inputs make col 0 saturate
    Wout = [
        [30000, 0, 0],
        [30000, 0, 0],
        [30000, 0, 0],
        [30000, 0, 0],
    ]
    return ("PROJ_SAT_POS", Y, Wout, 1, 4, 3, True)

def test_4():
    """PROJ_SAT_NEG: mirror of test_3 with sign-flipped Y."""
    Y = [[-30000, -30000, -30000, -30000]]
    Wout = [
        [30000, 0, 0],
        [30000, 0, 0],
        [30000, 0, 0],
        [30000, 0, 0],
    ]
    return ("PROJ_SAT_NEG", Y, Wout, 1, 4, 3, True)

def all_tests():
    return [test_0(), test_1(), test_2(), test_3(), test_4()]

# ---- Output emission ----

def emit_asm(path, tests):
    lines = []
    lines.append("* ============================================================")
    lines.append("* proj_vectors.asm — test vectors for PROJ harness")
    lines.append("* Generated by gen_proj_vectors.py — do not hand-edit.")
    lines.append("* Per-product shift-accumulate semantics matching ATTN/11 VTMUL.")
    lines.append("* Two-stage clamp per step: clamp shifted product, clamp running sum.")
    lines.append("* WOUT layout: [d_model][vocab] row-major.")
    lines.append("* Element Wout[d][v] at offset (d*VOC + v)*2 bytes from WOUT base.")
    lines.append("* ============================================================")
    lines.append("")

    for i, (name, Y, Wout, seq, dim, voc, sat_req) in enumerate(tests):
        sat_info = safety_check(name, Y, Wout, seq, dim, voc, sat_req)
        expected = proj_ref(Y, Wout, seq, dim, voc)
        cnt = seq * voc

        lines.append(f"* Test {i}: {name}")
        if name == "PROJ_FULL":
            lines.append(f"* Y pattern: Y[i][d] = (i*7 + d*3) %% 200 - 100  (range [-100, 99])")
            lines.append(f"* Wout pattern: Wout[d][v] = (d*5 + v*2) %% 100 - 50  (range [-50, 49])")
        if sat_info is not None:
            si, sv, sd, sprod, sshifted, sstage1, swhere = sat_info
            lines.append(f"* First saturating step: i={si} v={sv} d={sd} "
                         f"product={sprod} shifted={sshifted} stage1={sstage1} ({swhere} clamp)")

        lines.append(f"SEQ_{i}          FDB  {seq}")
        lines.append(f"DIM_{i}          FDB  {dim}")
        lines.append(f"VOC_{i}          FDB  {voc}")
        lines.append(f"CNT_{i}          FDB  {cnt}                    ; SEQ * VOC (output verify)")
        lines.append(f"Y_CNT_{i}        FDB  {seq * dim}                    ; SEQ * DIM (Y load)")
        lines.append(f"WOUT_CNT_{i}     FDB  {dim * voc}                    ; DIM * VOC (WOUT load)")

        # Y: SEQ*DIM words (row-major)
        y_flat = [v for row in Y for v in row]
        y_str = ",".join(str(v) for v in y_flat)
        lines.append(f"Y_{i}            FDB  {y_str}")

        # Wout: DIM*VOC words (row-major, d outer, v inner)
        w_flat = [v for row in Wout for v in row]
        w_str = ",".join(str(v) for v in w_flat)
        lines.append(f"WOUT_{i}         FDB  {w_str}")

        # Expected: SEQ*VOC words (row-major)
        exp_str = ",".join(str(v) for v in expected)
        lines.append(f"EXPECT_{i}       FDB  {exp_str}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

def emit_log(path, tests):
    lines = []
    lines.append("PROJ reference generator log")
    lines.append("Per-product shift-accumulate semantics matching ATTN/11 VTMUL")
    lines.append("Two-stage clamp per step: clamp shifted product, clamp running sum")
    lines.append("=" * 60)
    lines.append("")

    for i, (name, Y, Wout, seq, dim, voc, sat_req) in enumerate(tests):
        expected = proj_ref(Y, Wout, seq, dim, voc)
        trace = proj_ref_with_trace(Y, Wout, seq, dim, voc)
        lines.append(f"Test {i}: {name}")
        lines.append(f"  SEQ={seq} DIM={dim} VOC={voc}  ({seq*voc} outputs)")

        if name == "PROJ_FULL":
            lines.append(f"  Per-element finals:")
            for (ti, tv, tsteps, tfinal) in trace:
                lines.append(f"    i={ti} v={tv}: final={tfinal}")
        elif sat_req:
            # Show per-step detail for every (i,v) in saturation tests
            lines.append(f"  Per-step trace (all elements):")
            for (ti, tv, tsteps, tfinal) in trace:
                lines.append(f"    i={ti} v={tv}: final={tfinal}")
                for (d, yv, wv, prod, shf, st1, ac) in tsteps:
                    lines.append(f"      d={d} Y={yv} W={wv} prod={prod} "
                                 f"shf={shf} stage1={st1} acc={ac}")
        else:
            lines.append(f"  expected = {expected}")
            # Also emit per-step detail for the first element
            lines.append(f"  Per-step detail for i=0, v=0:")
            first = trace[0]
            for (d, yv, wv, prod, shf, st1, ac) in first[2]:
                lines.append(f"    d={d} Y={yv} W={wv} prod={prod} "
                             f"shf={shf} stage1={st1} acc={ac}")

        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

# ---- Main ----

def main():
    self_test()
    tests = all_tests()
    # Safety-check each test (raises on saturation failure)
    for (name, Y, Wout, seq, dim, voc, sat_req) in tests:
        safety_check(name, Y, Wout, seq, dim, voc, sat_req)

    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "proj_vectors.asm", tests)
    emit_log(out_dir / "proj_vectors.log", tests)
    print(f"wrote tables/proj_vectors.asm ({len(tests)} tests)")
    print(f"wrote tables/proj_vectors.log")
    print()
    print("Summary:")
    for i, (name, Y, Wout, seq, dim, voc, sat_req) in enumerate(tests):
        expected = proj_ref(Y, Wout, seq, dim, voc)
        print(f"  Test {i} ({name}): seq={seq} dim={dim} voc={voc} -> {len(expected)} outputs"
              + (" [SATURATES]" if sat_req else ""))

if __name__ == "__main__":
    main()
