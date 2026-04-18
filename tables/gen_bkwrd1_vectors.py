#!/usr/bin/env python3
"""
BKWRD Step 1 reference generator — dLogits, dWout, dY.

For each position i:
  1. sftmx(logits[i]) -> probs (copy, not in-place)
  2. probs[target[i]] -= 256 (one-hot subtraction)
  3. dL = probs << 7 (Q8 -> Q15)
  4. dWout += outer(Y[i], dL) (weight gradient, accumulated)
  5. dY[i] = mvmul(Wout, dL) (input gradient, per-position)

OUTER: 32-bit accumulate-and-extract per element.
  combined = mat[i][j]*256 + vx[i]*vy[j]; mat = extract(combined)
MVMUL: 32-bit row accumulation, single end-of-row extract.
  Same as VDOT-style extraction.
"""

import math
from pathlib import Path

# ---- Primitives ----

def to_s32(val):
    val &= 0xFFFFFFFF
    if val >= 0x80000000:
        val -= 0x100000000
    return val

def vdot_extract(acc_s32):
    """MSB-byte check + middle-word extraction. Matches VDOT/OUTER/MVMUL."""
    acc_u = acc_s32 & 0xFFFFFFFF
    msb = (acc_u >> 24) & 0xFF
    if msb == 0x00 or msb == 0xFF:
        result = (acc_u >> 8) & 0xFFFF
        if result >= 0x8000:
            result -= 0x10000
        return result
    elif msb & 0x80:
        return -32768
    else:
        return 32767

def wrap16(x):
    x &= 0xFFFF
    if x >= 0x8000:
        x -= 0x10000
    return x

EXPTBL = [max(0, int(math.exp(-i / 32.0) * 256.0 + 0.5)) for i in range(256)]

def fxdiv_ref(a, b):
    if b == 0:
        return 0
    numerator = a * 256
    if (numerator >= 0 and b >= 0) or (numerator < 0 and b < 0):
        result = numerator // b
    else:
        result = -(abs(numerator) // abs(b))
    return max(-32768, min(32767, result))

def sftmx_ref(vec):
    n = len(vec)
    mx = max(vec)
    exps = []
    for v in vec:
        diff = mx - v
        idx = 255 if diff > 2040 else (diff >> 3)
        if idx > 255:
            idx = 255
        exps.append(EXPTBL[idx])
    s = sum(exps)
    if s == 0:
        return [0] * n
    return [fxdiv_ref(e, s) for e in exps]

# ---- OUTER: 32-bit accumulate-and-extract ----

def outer_ref(mat, vx, vy, rows, cols):
    """mat[i][j] = extract(mat[i][j]*256 + vx[i]*vy[j])"""
    for i in range(rows):
        for j in range(cols):
            combined = to_s32(mat[i][j] * 256 + vx[i] * vy[j])
            mat[i][j] = vdot_extract(combined)
    return mat

# ---- MVMUL: 32-bit row accumulation, single extract ----

def mvmul_ref(mat, vin, rows, cols):
    """vout[i] = extract(sum_j(mat[i][j] * vin[j]))"""
    vout = []
    for i in range(rows):
        acc = 0
        for j in range(cols):
            acc = to_s32(acc + mat[i][j] * vin[j])
        vout.append(vdot_extract(acc))
    return vout

# ---- BKWRD Step 1 reference ----

def bkwrd1_ref(logits, targets, Y, Wout, seq, dim, voc):
    dWout = [[0]*voc for _ in range(dim)]
    dY = [[0]*dim for _ in range(seq)]

    for i in range(seq):
        probs = sftmx_ref(list(logits[i]))
        probs[targets[i]] -= 256
        dL = [wrap16(v << 7) for v in probs]
        dWout = outer_ref(dWout, Y[i], dL, dim, voc)
        dY[i] = mvmul_ref(Wout, dL, dim, voc)

    return dWout, dY

# ---- Self-test ----

def self_test():
    assert vdot_extract(0) == 0
    assert vdot_extract(256) == 1
    assert vdot_extract(-256) == -1
    assert wrap16((-256) << 7) == -32768
    assert wrap16(255 << 7) == 32640

    r = sftmx_ref([128, 128])
    assert r == [128, 128], f"sftmx: {r}"

    # outer trivial: mat=[[0]], vx=[256], vy=[256] -> mat=[[256]]
    m = outer_ref([[0]], [256], [256], 1, 1)
    assert m == [[256]], f"outer: {m}"

    # mvmul trivial: mat=[[256]], vin=[256] -> vout=[256]
    r = mvmul_ref([[256]], [256], 1, 1)
    assert r == [256], f"mvmul: {r}"

    # outer accumulation: two products
    m = outer_ref([[0]], [100], [200], 1, 1)
    # product = 100*200 = 20000. combined = 0 + 20000. extract: 20000>>8 = 78.
    assert m == [[78]], f"outer acc: {m}"
    m = outer_ref(m, [100], [200], 1, 1)
    # combined = 78*256 + 20000 = 19968 + 20000 = 39968. extract: 39968>>8 = 156.
    assert m == [[156]], f"outer acc2: {m}"

    print("Self-test passed")

# ---- Test data ----

def test_0():
    """BK1_TINY: SEQ=2 DIM=2 VOC=3. Hand-verifiable."""
    seq, dim, voc = 2, 2, 3
    logits = [[100, 200, 50], [300, 100, 200]]
    targets = [1, 0]
    Y = [[64, 128], [-64, 32]]
    Wout = [[10, 20, 30], [40, 50, 60]]
    return ("BK1_TINY", logits, targets, Y, Wout, seq, dim, voc)

def test_1():
    """BK1_UNIFORM: SEQ=2 DIM=4 VOC=10. Uniform logits."""
    seq, dim, voc = 2, 4, 10
    logits = [[0]*voc for _ in range(seq)]
    targets = [3, 7]
    Y = [[100, -50, 200, -100], [50, 150, -75, 25]]
    Wout = [[(d*3 + v*2) % 60 - 30 for v in range(voc)] for d in range(dim)]
    return ("BK1_UNIFORM", logits, targets, Y, Wout, seq, dim, voc)

def test_2():
    """BK1_MEDIUM: SEQ=4 DIM=8 VOC=10. Non-trivial dimensions."""
    seq, dim, voc = 4, 8, 10
    logits = [[(i*7 + v*3) % 400 - 200 for v in range(voc)] for i in range(seq)]
    targets = [3, 7, 1, 5]
    Y = [[(i*5 + d*3) % 200 - 100 for d in range(dim)] for i in range(seq)]
    Wout = [[(d*4 + v*6) % 80 - 40 for v in range(voc)] for d in range(dim)]
    return ("BK1_MEDIUM", logits, targets, Y, Wout, seq, dim, voc)

def test_3():
    """BK1_SAT: SEQ=2 DIM=4 VOC=10. Values to exercise saturation."""
    seq, dim, voc = 2, 4, 10
    logits = [[0]*voc for _ in range(seq)]
    targets = [0, 1]
    Y = [[30000, -30000, 25000, -25000], [32000, 32000, -32000, -32000]]
    Wout = [[(d*11 + v*7) % 120 - 60 for v in range(voc)] for d in range(dim)]
    return ("BK1_SAT", logits, targets, Y, Wout, seq, dim, voc)

def all_tests():
    return [test_0(), test_1(), test_2(), test_3()]

# ---- Emission ----

def emit_asm(path, test_data):
    lines = []
    lines.append("* ============================================================")
    lines.append("* bkwrd1_vectors.asm — BKWRD Step 1 test vectors")
    lines.append("* Generated by gen_bkwrd1_vectors.py — do not hand-edit.")
    lines.append("* OUTER: 32-bit accumulate-and-extract. MVMUL: row accumulation.")
    lines.append("* ============================================================")
    lines.append("")

    for idx, (name, logits, targets, Y, Wout, seq, dim, voc, dWout, dY) in enumerate(test_data):
        lines.append(f"* Test {idx}: {name}  SEQ={seq} DIM={dim} VOC={voc}")
        lines.append(f"SEQ_{idx}          FDB  {seq}")
        lines.append(f"DIM_{idx}          FDB  {dim}")
        lines.append(f"VOC_{idx}          FDB  {voc}")

        lg_flat = [v for row in logits for v in row]
        lines.append(f"LOGITS_{idx}       FDB  {','.join(str(v) for v in lg_flat)}")
        lines.append(f"TARGETS_{idx}      FDB  {','.join(str(v) for v in targets)}")

        y_flat = [v for row in Y for v in row]
        lines.append(f"Y_{idx}            FDB  {','.join(str(v) for v in y_flat)}")

        w_flat = [v for row in Wout for v in row]
        lines.append(f"WOUT_{idx}         FDB  {','.join(str(v) for v in w_flat)}")

        lines.append(f"DWOUT_CNT_{idx}    FDB  {dim*voc}")
        lines.append(f"DY_CNT_{idx}       FDB  {seq*dim}")

        dw_flat = [v for row in dWout for v in row]
        lines.append(f"EXP_DWOUT_{idx}    FDB  {','.join(str(v) for v in dw_flat)}")

        dy_flat = [v for row in dY for v in row]
        lines.append(f"EXP_DY_{idx}       FDB  {','.join(str(v) for v in dy_flat)}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

def emit_log(path, test_data_full):
    lines = []
    lines.append("BKWRD Step 1 reference generator log")
    lines.append("=" * 60)
    lines.append("")

    for idx, info in enumerate(test_data_full):
        name = info["name"]
        lines.append(f"Test {idx}: {name}  SEQ={info['seq']} DIM={info['dim']} VOC={info['voc']}")
        lines.append(f"  targets = {info['targets']}")

        if info.get("per_pos"):
            for pp in info["per_pos"]:
                lines.append(f"  pos {pp['i']}:")
                lines.append(f"    probs = {pp['probs']}")
                lines.append(f"    dL_q8 = {pp['dL_q8']}")
                lines.append(f"    dL_q15 = {pp['dL_q15']}")

        dw = info["dWout"]
        lines.append(f"  dWout:")
        for i, row in enumerate(dw):
            lines.append(f"    [{i}] = {row}")

        dy = info["dY"]
        lines.append(f"  dY:")
        for i, row in enumerate(dy):
            lines.append(f"    [{i}] = {row}")

        if info.get("saturated"):
            for s in info["saturated"]:
                lines.append(f"  SAT: {s}")

        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

# ---- Main ----

def main():
    self_test()

    tests = all_tests()
    test_data = []
    test_data_full = []

    for (name, logits, targets, Y, Wout, seq, dim, voc) in tests:
        dWout, dY = bkwrd1_ref(logits, targets, Y, Wout, seq, dim, voc)
        test_data.append((name, logits, targets, Y, Wout, seq, dim, voc, dWout, dY))

        info = {"name": name, "seq": seq, "dim": dim, "voc": voc,
                "targets": targets, "dWout": dWout, "dY": dY}

        if seq <= 2 and dim <= 4:
            per_pos = []
            dw_trace = [[0]*voc for _ in range(dim)]
            for i in range(seq):
                probs = sftmx_ref(list(logits[i]))
                dL_q8 = list(probs)
                dL_q8[targets[i]] -= 256
                dL_q15 = [wrap16(v << 7) for v in dL_q8]
                per_pos.append({"i": i, "probs": probs, "dL_q8": dL_q8, "dL_q15": dL_q15})
            info["per_pos"] = per_pos

        saturated = []
        for i, row in enumerate(dWout):
            for j, v in enumerate(row):
                if v == 32767 or v == -32768:
                    saturated.append(f"dWout[{i}][{j}] = {v}")
        for i, row in enumerate(dY):
            for j, v in enumerate(row):
                if v == 32767 or v == -32768:
                    saturated.append(f"dY[{i}][{j}] = {v}")
        if saturated:
            info["saturated"] = saturated

        test_data_full.append(info)

    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "bkwrd1_vectors.asm", test_data)
    emit_log(out_dir / "bkwrd1_vectors.log", test_data_full)

    print(f"\nwrote tables/bkwrd1_vectors.asm ({len(test_data)} tests)")
    print(f"wrote tables/bkwrd1_vectors.log")
    print("\nSummary:")
    for idx, (name, logits, targets, Y, Wout, seq, dim, voc, dWout, dY) in enumerate(test_data):
        dw_flat = [v for row in dWout for v in row]
        dy_flat = [v for row in dY for v in row]
        print(f"  Test {idx} ({name}): {len(dw_flat)} dWout + {len(dy_flat)} dY outputs")

if __name__ == "__main__":
    main()
