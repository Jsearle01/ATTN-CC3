#!/usr/bin/env python3
"""
ATTN reference generator.

Produces tables/attn_vectors.asm with 5 test cases for the full ATTN
self-attention forward pass (Steps 1-7 of src/layer.asm ATTN routine).

Reference semantics (matches ATTN/11 and src/layer.asm):
  Step 1-3:  Q/K/V = vtmul(W*, X[i])    per-product shift-accumulate
  Step 4:    Sc[i][j] = VDOT(Q[i],K[j]) >> shf   (VDOT does >>8 + clamp internally)
  Step 5:    A[i] = sftmx(Sc[i])         LUT-based Q8 softmax
  Step 6:    O[i] = vtmul(V, A[i])       rows=SEQ, cols=DIM
  Step 7:    Y[i] += X[i]                wrapping add, no clamp

VTMUL: per-product shift-accumulate, two-stage clamp per step.
VDOT: 32-bit signed accumulation (wraps at 32 bits), byte-extract bits [23:8],
  MSB-byte overflow clamp to +/-32767.
SFTMX: max-subtract, EXPTBL[clamp((max-x_i)>>3, 0, 255)], FXDIV normalize.
"""

import math
from pathlib import Path

# ---- Primitives ----

def arith_shift_right_8(x):
    if x >= 0:
        return x // 256
    else:
        return -((-x + 255) // 256)

def clamp_q8(x):
    if x > 32767: return 32767
    if x < -32768: return -32768
    return x

def wrap16(x):
    x &= 0xFFFF
    if x >= 0x8000:
        x -= 0x10000
    return x

def to_s32(val):
    val &= 0xFFFFFFFF
    if val >= 0x80000000:
        val -= 0x100000000
    return val

EXPTBL = [max(0, int(math.exp(-i / 32.0) * 256.0 + 0.5)) for i in range(256)]

# ---- VTMUL: per-product shift-accumulate ----

def vtmul_q8(mat, vin, rows, cols):
    out = []
    for j in range(cols):
        acc = 0
        for i in range(rows):
            product = mat[i][j] * vin[i]
            stage1 = clamp_q8(arith_shift_right_8(product))
            acc = clamp_q8(acc + stage1)
        out.append(acc)
    return out

# ---- VDOT: 32-bit accumulation, byte-extract bits [23:8], clamp ----

def vdot_q8(a, b, length):
    """
    Matches src/vecop.asm VDOT exactly.
    Byte-extraction (LDD VDOT_ACC+1 on big-endian 6309) uses unsigned >>8
    of the 32-bit representation, NOT Python's arithmetic >>.
    """
    acc = 0
    for i in range(length):
        acc = to_s32(acc + a[i] * b[i])
    acc_u = acc & 0xFFFFFFFF
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

# ---- FXDIV: Q8 division ----

def fxdiv_ref(a, b):
    if b == 0:
        return 0
    numerator = a * 256
    if (numerator >= 0 and b >= 0) or (numerator < 0 and b < 0):
        result = numerator // b
    else:
        result = -(abs(numerator) // abs(b))
    return max(-32768, min(32767, result))

# ---- SFTMX: LUT-based Q8 softmax ----

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

# ---- ATTN forward pass ----

def attn_ref(X, Wq, Wk, Wv, seq, dim, shf):
    Q = [vtmul_q8(Wq, X[i], dim, dim) for i in range(seq)]
    K = [vtmul_q8(Wk, X[i], dim, dim) for i in range(seq)]
    V = [vtmul_q8(Wv, X[i], dim, dim) for i in range(seq)]

    Sc = [[0]*seq for _ in range(seq)]
    for i in range(seq):
        for j in range(seq):
            dot = vdot_q8(Q[i], K[j], dim)
            Sc[i][j] = dot >> shf

    A = [sftmx_ref(Sc[i]) for i in range(seq)]
    O = [vtmul_q8(V, A[i], seq, dim) for i in range(seq)]

    Y = [[0]*dim for _ in range(seq)]
    for i in range(seq):
        for d in range(dim):
            Y[i][d] = wrap16(O[i][d] + X[i][d])

    return Y, Q, K, V, Sc, A, O

# ---- Self-test ----

def self_test():
    assert arith_shift_right_8(-1) == -1
    assert arith_shift_right_8(-256) == -1
    assert arith_shift_right_8(-257) == -2
    assert arith_shift_right_8(0) == 0
    assert arith_shift_right_8(256) == 1
    assert clamp_q8(40000) == 32767
    assert clamp_q8(-40000) == -32768
    assert wrap16(32768) == -32768
    assert wrap16(0) == 0

    r = vtmul_q8([[256, 0], [0, 256]], [100, 200], 2, 2)
    assert r == [100, 200], f"vtmul identity: {r}"

    # vdot trivial: 2*(256^2) = 131072 = 0x00020000. Middle bytes = 0x0200 = 512.
    r = vdot_q8([256, 256], [256, 256], 2)
    assert r == 512, f"vdot trivial: {r}"

    # vdot 32-bit wrap: 4*32767^2 = 4294705156 = 0xFFFC0004 (signed -262140).
    # MSB=0xFF (fits). Byte-extract middle: 0xFC00 = -1024.
    r = vdot_q8([32767]*4, [32767]*4, 4)
    assert r == -1024, f"vdot wrap: {r}"

    # sftmx
    assert sftmx_ref([256]) == [256]
    assert sftmx_ref([128, 128]) == [128, 128]
    r = sftmx_ref([0, 0, 0, 0])
    assert all(abs(v - 64) <= 1 for v in r), f"sftmx zeros: {r}"

    assert fxdiv_ref(256, 256) == 256
    assert fxdiv_ref(128, 256) == 128

    print("Self-test passed")

# ---- Test data ----

def test_0():
    """ATTN_TINY: SEQ=2 DIM=4 SHF=1. Identity weights, hand-verifiable."""
    seq, dim, shf = 2, 4, 1
    X = [[64, 64, 64, 64], [128, -128, 64, -64]]
    eye = [[256 if i == j else 0 for j in range(dim)] for i in range(dim)]
    return ("ATTN_TINY", X, eye, eye, eye, seq, dim, shf, False, False)

def test_1():
    """ATTN_NOSHIFT: SEQ=2 DIM=4 SHF=0. Exercises BEQ ATTN_S4_NOSHF."""
    seq, dim, shf = 2, 4, 0
    X = [[100, -50, 200, -100], [-200, 150, -100, 50]]
    eye = [[256 if i == j else 0 for j in range(dim)] for i in range(dim)]
    return ("ATTN_NOSHIFT", X, eye, eye, eye, seq, dim, shf, False, False)

def test_2():
    """ATTN_RESIDUAL_WRAP: X near boundaries so O+X wraps."""
    seq, dim, shf = 2, 4, 1
    X = [[30000, -30000, 32000, -32000], [25000, -25000, 28000, -28000]]
    eye = [[256 if i == j else 0 for j in range(dim)] for i in range(dim)]
    return ("ATTN_RESIDUAL_WRAP", X, eye, eye, eye, seq, dim, shf, True, False)

def test_3():
    """ATTN_SAT_SCORE: Large Q*K causes VDOT overflow -> clamp. SHF=0.
    Both positions have same-sign X so all scores clamp to the same value.
    This avoids triggering SFTMX's signed-comparison overflow when
    score range exceeds 32767 (a pre-existing ATTN/11 limitation).
    """
    seq, dim, shf = 2, 4, 0
    X = [[20000, 20000, 20000, 20000], [20000, 20000, 20000, 20000]]
    eye = [[256 if i == j else 0 for j in range(dim)] for i in range(dim)]
    return ("ATTN_SAT_SCORE", X, eye, eye, eye, seq, dim, shf, False, True)

def test_4():
    """ATTN_MEDIUM: SEQ=4 DIM=8 SHF=2. Non-trivial dimensions, production SHF.
    Non-identity weights for richer coverage. Defers SEQ=8 DIM=16 to standalone harness.
    """
    seq, dim, shf = 4, 8, 2
    X = [[(i*7 + d*3) % 200 - 100 for d in range(dim)] for i in range(seq)]
    Wq = [[(i*5 + j*2 + 1) % 60 - 30 for j in range(dim)] for i in range(dim)]
    Wk = [[(i*3 + j*7 + 2) % 60 - 30 for j in range(dim)] for i in range(dim)]
    Wv = [[(i*4 + j*5 + 3) % 60 - 30 for j in range(dim)] for i in range(dim)]
    return ("ATTN_MEDIUM", X, Wq, Wk, Wv, seq, dim, shf, False, False)

def all_tests():
    return [test_0(), test_1(), test_2(), test_3(), test_4()]

# ---- Emission ----

def emit_asm(path, test_data):
    lines = []
    lines.append("* ============================================================")
    lines.append("* attn_vectors.asm — test vectors for ATTN harness")
    lines.append("* Generated by gen_attn_vectors.py — do not hand-edit.")
    lines.append("* VTMUL per-product, VDOT single-clamp, LUT softmax, wrap residual.")
    lines.append("* ============================================================")
    lines.append("")

    for idx, (name, X, Wq, Wk, Wv, seq, dim, shf, Y_expect) in enumerate(test_data):
        lines.append(f"* Test {idx}: {name}  SEQ={seq} DIM={dim} SHF={shf}")
        lines.append(f"SEQ_{idx}          FDB  {seq}")
        lines.append(f"DIM_{idx}          FDB  {dim}")
        lines.append(f"SHF_{idx}          FDB  {shf}")
        lines.append(f"X_CNT_{idx}        FDB  {seq*dim}")
        lines.append(f"W_CNT_{idx}        FDB  {dim*dim}")
        lines.append(f"Y_CNT_{idx}        FDB  {seq*dim}")

        x_flat = [v for row in X for v in row]
        lines.append(f"X_{idx}            FDB  {','.join(str(v) for v in x_flat)}")

        wq_flat = [v for row in Wq for v in row]
        lines.append(f"WQ_{idx}           FDB  {','.join(str(v) for v in wq_flat)}")
        wk_flat = [v for row in Wk for v in row]
        lines.append(f"WK_{idx}           FDB  {','.join(str(v) for v in wk_flat)}")
        wv_flat = [v for row in Wv for v in row]
        lines.append(f"WV_{idx}           FDB  {','.join(str(v) for v in wv_flat)}")

        y_flat = [v for row in Y_expect for v in row]
        lines.append(f"EXPECT_{idx}       FDB  {','.join(str(v) for v in y_flat)}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

def emit_log(path, all_info):
    lines = []
    lines.append("ATTN reference generator log")
    lines.append("=" * 60)
    lines.append("")

    for idx, info in enumerate(all_info):
        name = info["name"]
        lines.append(f"Test {idx}: {name}  SEQ={info['seq']} DIM={info['dim']} SHF={info['shf']}")

        if info.get("intermediates"):
            for label, data in info["intermediates"]:
                if isinstance(data[0], list):
                    for i, row in enumerate(data):
                        lines.append(f"  {label}[{i}] = {row}")
                else:
                    lines.append(f"  {label} = {data}")

        y_flat = [v for row in info["Y"] for v in row]
        lines.append(f"  Y (flat) = {y_flat}")

        for (i, d, o_val, x_val, raw, wrapped) in info.get("wrap_elements", []):
            lines.append(f"  WRAP: Y[{i}][{d}]: O={o_val} + X={x_val} = {raw} -> wrap={wrapped}")

        for (i, j, dot) in info.get("sat_scores", []):
            lines.append(f"  SAT: VDOT[{i}][{j}] = {dot} (clamped)")

        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

# ---- Main ----

def main():
    self_test()

    tests = all_tests()
    test_data = []
    all_info = []

    for (name, X, Wq, Wk, Wv, seq, dim, shf, require_wrap, require_sat) in tests:
        Y, Q, K, V, Sc, A, O = attn_ref(X, Wq, Wk, Wv, seq, dim, shf)

        info = {"name": name, "seq": seq, "dim": dim, "shf": shf, "Y": Y}

        if seq <= 2:
            info["intermediates"] = [
                ("Q", Q), ("K", K), ("V", V), ("Sc", Sc), ("A", A), ("O", O)]

        wrap_elements = []
        for i in range(seq):
            for d in range(dim):
                raw = O[i][d] + X[i][d]
                wrapped = wrap16(raw)
                if raw != wrapped:
                    wrap_elements.append((i, d, O[i][d], X[i][d], raw, wrapped))
        info["wrap_elements"] = wrap_elements

        if require_wrap:
            assert len(wrap_elements) > 0, \
                f"{name}: require_wrap but no element wraps. Adjust inputs."
            print(f"  {name}: {len(wrap_elements)} wrapping element(s)")

        sat_scores = []
        for i in range(seq):
            for j in range(seq):
                dot = vdot_q8(Q[i], K[j], dim)
                if dot == 32767 or dot == -32768:
                    sat_scores.append((i, j, dot))
        info["sat_scores"] = sat_scores

        if require_sat:
            assert len(sat_scores) > 0, \
                f"{name}: require_sat but no VDOT result clamps. Adjust inputs."
            print(f"  {name}: {len(sat_scores)} saturating score(s)")

        test_data.append((name, X, Wq, Wk, Wv, seq, dim, shf, Y))
        all_info.append(info)

    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "attn_vectors.asm", test_data)
    emit_log(out_dir / "attn_vectors.log", all_info)

    print(f"\nwrote tables/attn_vectors.asm ({len(test_data)} tests)")
    print(f"wrote tables/attn_vectors.log")
    print("\nSummary:")
    for idx, (name, X, Wq, Wk, Wv, seq, dim, shf, Y) in enumerate(test_data):
        y_flat = [v for row in Y for v in row]
        print(f"  Test {idx} ({name}): SEQ={seq} DIM={dim} SHF={shf} -> {len(y_flat)} outputs")
        print(f"    Y[0] = {Y[0]}")

if __name__ == "__main__":
    main()
