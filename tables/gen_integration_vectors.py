#!/usr/bin/env python3
"""
Integration reference generator: EMBED -> ATTN -> PROJ end-to-end.

Chains validated reference functions from the three composite generators:
  - embed_ref: wrapping add (matches ATTN/11 assembly, not prototype.shf clamp)
  - attn_ref: 7-step with VTMUL per-product, VDOT unsigned byte-extract, LUT softmax
  - proj_ref: VTMUL per-product shift-accumulate

Tiny model: SEQ=2, D=4, V=10, SHF=1.
Output: tables/integration_vectors.asm + .log
"""

import math
from pathlib import Path

# ---- Primitives (copied from gen_attn_vectors.py / gen_proj_vectors.py) ----

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

# ---- EMBED ----

def embed_ref(tokens, tkemb, psemb, seq, dim):
    """Returns 2D list [seq][dim]. Wrapping add, no clamp."""
    X = []
    for i in range(seq):
        row = []
        for j in range(dim):
            row.append(wrap16(tkemb[tokens[i]][j] + psemb[i][j]))
        X.append(row)
    return X

# ---- VTMUL ----

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

# ---- VDOT ----

def vdot_q8(a, b, length):
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

# ---- FXDIV / SFTMX ----

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

# ---- ATTN ----

def attn_ref(X, Wq, Wk, Wv, seq, dim, shf):
    Q = [vtmul_q8(Wq, X[i], dim, dim) for i in range(seq)]
    K = [vtmul_q8(Wk, X[i], dim, dim) for i in range(seq)]
    V = [vtmul_q8(Wv, X[i], dim, dim) for i in range(seq)]
    Sc = [[0]*seq for _ in range(seq)]
    for i in range(seq):
        for j in range(seq):
            Sc[i][j] = vdot_q8(Q[i], K[j], dim) >> shf
    A = [sftmx_ref(Sc[i]) for i in range(seq)]
    O = [vtmul_q8(V, A[i], seq, dim) for i in range(seq)]
    Y = [[wrap16(O[i][d] + X[i][d]) for d in range(dim)] for i in range(seq)]
    return Y, Q, K, V, Sc, A, O

# ---- PROJ ----

def proj_ref(Y, Wout, seq, dim, voc):
    logits = []
    for i in range(seq):
        row = []
        for v in range(voc):
            acc = 0
            for d in range(dim):
                product = Y[i][d] * Wout[d][v]
                stage1 = clamp_q8(arith_shift_right_8(product))
                acc = clamp_q8(acc + stage1)
            row.append(acc)
        logits.append(row)
    return logits

# ---- Self-test ----

def self_test():
    assert arith_shift_right_8(-257) == -2
    assert arith_shift_right_8(256) == 1
    assert clamp_q8(40000) == 32767
    assert wrap16(32768) == -32768

    r = vtmul_q8([[256, 0], [0, 256]], [100, 200], 2, 2)
    assert r == [100, 200], f"vtmul: {r}"

    r = vdot_q8([256, 256], [256, 256], 2)
    assert r == 512, f"vdot: {r}"

    assert sftmx_ref([128, 128]) == [128, 128]
    assert fxdiv_ref(256, 256) == 256

    X = embed_ref([0], [[100, 200]], [[10, 20]], 1, 2)
    assert X == [[110, 220]], f"embed: {X}"

    print("Self-test passed")

# ---- Test data ----

def gen_test():
    seq, dim, voc, shf = 2, 4, 10, 1

    tokens = [3, 7]
    tkemb = [[(v*7 + d*3) % 200 - 100 for d in range(dim)] for v in range(voc)]
    psemb = [[(p*11 + d*5) % 100 - 50 for d in range(dim)] for p in range(seq)]
    Wq = [[(i*5 + j*3) % 60 - 30 for j in range(dim)] for i in range(dim)]
    Wk = [[(i*7 + j*2) % 60 - 30 for j in range(dim)] for i in range(dim)]
    Wv = [[(i*3 + j*7) % 60 - 30 for j in range(dim)] for i in range(dim)]
    Wout = [[(d*4 + v*6) % 80 - 40 for v in range(voc)] for d in range(dim)]

    X = embed_ref(tokens, tkemb, psemb, seq, dim)
    Y, Q, K, V, Sc, A, O = attn_ref(X, Wq, Wk, Wv, seq, dim, shf)
    logits = proj_ref(Y, Wout, seq, dim, voc)

    return {
        "seq": seq, "dim": dim, "voc": voc, "shf": shf,
        "tokens": tokens,
        "tkemb": tkemb, "psemb": psemb,
        "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wout": Wout,
        "X": X, "Y": Y, "logits": logits,
        "Q": Q, "K": K, "V": V, "Sc": Sc, "A": A, "O": O,
    }

# ---- Emission ----

def emit_asm(path, t):
    lines = []
    lines.append("* ============================================================")
    lines.append("* integration_vectors.asm — end-to-end test vectors")
    lines.append("* Generated by gen_integration_vectors.py — do not hand-edit.")
    lines.append("* EMBED (wrap) -> ATTN (VTMUL per-product, VDOT byte-extract)")
    lines.append("*   -> PROJ (VTMUL per-product)")
    lines.append("* ============================================================")
    lines.append("")
    lines.append(f"SEQ_0          FDB  {t['seq']}")
    lines.append(f"DIM_0          FDB  {t['dim']}")
    lines.append(f"VOC_0          FDB  {t['voc']}")
    lines.append(f"SHF_0          FDB  {t['shf']}")

    lines.append(f"TOKENS_0       FDB  {','.join(str(v) for v in t['tokens'])}")

    tk_flat = [v for row in t['tkemb'] for v in row]
    lines.append(f"TKEMB_0        FDB  {','.join(str(v) for v in tk_flat)}")

    ps_flat = [v for row in t['psemb'] for v in row]
    lines.append(f"PSEMB_0        FDB  {','.join(str(v) for v in ps_flat)}")

    wq_flat = [v for row in t['Wq'] for v in row]
    lines.append(f"WQ_0           FDB  {','.join(str(v) for v in wq_flat)}")
    wk_flat = [v for row in t['Wk'] for v in row]
    lines.append(f"WK_0           FDB  {','.join(str(v) for v in wk_flat)}")
    wv_flat = [v for row in t['Wv'] for v in row]
    lines.append(f"WV_0           FDB  {','.join(str(v) for v in wv_flat)}")

    wo_flat = [v for row in t['Wout'] for v in row]
    lines.append(f"WOUT_0         FDB  {','.join(str(v) for v in wo_flat)}")

    lg_flat = [v for row in t['logits'] for v in row]
    lines.append(f"EXPECT_0       FDB  {','.join(str(v) for v in lg_flat)}")

    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")

def emit_log(path, t):
    lines = []
    lines.append("Integration reference generator log")
    lines.append("EMBED -> ATTN -> PROJ end-to-end")
    lines.append(f"SEQ={t['seq']} DIM={t['dim']} VOC={t['voc']} SHF={t['shf']}")
    lines.append(f"Tokens = {t['tokens']}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("EMBED output X:")
    for i, row in enumerate(t['X']):
        lines.append(f"  X[{i}] = {row}")
    lines.append("")
    lines.append("ATTN intermediates:")
    for i, row in enumerate(t['Q']):
        lines.append(f"  Q[{i}] = {row}")
    for i, row in enumerate(t['K']):
        lines.append(f"  K[{i}] = {row}")
    for i, row in enumerate(t['V']):
        lines.append(f"  V[{i}] = {row}")
    for i, row in enumerate(t['Sc']):
        lines.append(f"  Sc[{i}] = {row}")
    for i, row in enumerate(t['A']):
        lines.append(f"  A[{i}] = {row}")
    for i, row in enumerate(t['O']):
        lines.append(f"  O[{i}] = {row}")
    lines.append("")
    lines.append("ATTN output Y:")
    for i, row in enumerate(t['Y']):
        lines.append(f"  Y[{i}] = {row}")
    lines.append("")
    lines.append("PROJ output logits:")
    for i, row in enumerate(t['logits']):
        lines.append(f"  logits[{i}] = {row}")
    lines.append("")

    # Hand spot-check: logits[0][0]
    lines.append("Hand spot-check: logits[0][0]")
    Y0 = t['Y'][0]
    Wout = t['Wout']
    acc = 0
    for d in range(t['dim']):
        product = Y0[d] * Wout[d][0]
        shifted = arith_shift_right_8(product)
        stage1 = clamp_q8(shifted)
        acc_new = clamp_q8(acc + stage1)
        lines.append(f"  d={d}: Y[0][{d}]={Y0[d]} * Wout[{d}][0]={Wout[d][0]} "
                     f"= {product} >>8= {shifted} clamp= {stage1} acc= {acc_new}")
        acc = acc_new
    lines.append(f"  logits[0][0] = {acc}")
    assert acc == t['logits'][0][0], \
        f"Spot-check mismatch: computed {acc} vs stored {t['logits'][0][0]}"
    lines.append(f"  (matches generator output: {t['logits'][0][0]}) ✓")
    lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

# ---- Main ----

def main():
    self_test()
    t = gen_test()

    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "integration_vectors.asm", t)
    emit_log(out_dir / "integration_vectors.log", t)

    lg_flat = [v for row in t['logits'] for v in row]
    print(f"wrote tables/integration_vectors.asm (1 test, {len(lg_flat)} logit outputs)")
    print(f"wrote tables/integration_vectors.log")
    print()
    print(f"EMBED X[0] = {t['X'][0]}")
    print(f"EMBED X[1] = {t['X'][1]}")
    print(f"ATTN  Y[0] = {t['Y'][0]}")
    print(f"ATTN  Y[1] = {t['Y'][1]}")
    print(f"PROJ  logits[0] = {t['logits'][0]}")
    print(f"PROJ  logits[1] = {t['logits'][1]}")

if __name__ == "__main__":
    main()
