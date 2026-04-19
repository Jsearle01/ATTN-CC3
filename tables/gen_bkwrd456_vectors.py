#!/usr/bin/env python3
"""
BKWRD Steps 4-6 reference generator.

Step 4 (BKWRD_S4):
    Part A: for i=0..SEQ-1:  dQ[i] = VTMUL(K, dSc[i],  SEQ, DIM)
    Part B: for j=0..SEQ-1:  dK[j] = VTMUL(Q, col_j(dSc), SEQ, DIM)

Step 5 (BKWRD_S5):
    dX = copy(dY)                          ; residual init
    for i in 0..SEQ-1:
        dX[i] += MVADD(Wq, dQ[i])          ; vout += mat * vin
        dWq   += OUTER(X[i], dQ[i])        ; mat  += vx(outer)vy
        dX[i] += MVADD(Wk, dK[i])
        dWk   += OUTER(X[i], dK[i])
        dX[i] += MVADD(Wv, dV[i])
        dWv   += OUTER(X[i], dV[i])

Step 6 (BKWRD_S6):
    for i in 0..SEQ-1:
        d_tok[tokens[i]][d] += dX[i][d]    ; scatter-add, clamp per element
        d_pos[i][d]         += dX[i][d]    ; direct-add,  clamp per element

VTMUL / MVADD / OUTER: per-element single-clamp via MSB-byte check +
bits-[23:8] extraction, 32-bit accumulate with sign-extended vout/mat
injected as v*256 (matches matop.asm semantics from gen_matop_vectors.py).

Step 6 clamp: signed 16-bit saturation, addend-sign determines direction.
"""

from pathlib import Path


# ---- Primitives ----

def to_s32(v):
    v &= 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    return v


def sign_extend_q8_to_q16(v):
    """Q8 value sign-extended to Q16 is simply v*256."""
    return v * 256


def clamp_extract(acc):
    """32-bit acc -> Q15 via MSB-byte check + bits [23:8] extract.
    Matches matop.asm VTMUL/MVADD/OUTER and VDOT clamp."""
    acc = to_s32(acc)
    hi = (acc >> 24) & 0xFF
    if hi == 0x00 or hi == 0xFF:
        r = (acc >> 8) & 0xFFFF
        if r >= 0x8000:
            r -= 0x10000
        return r
    return 32767 if (hi & 0x80) == 0 else -32768


def clamp_q15(x):
    """16-bit signed saturation (Step 6 clamp)."""
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return x


# ---- Per-element single-clamp primitives (match matop.asm) ----

def vtmul_ref(mat, vin, rows, cols):
    """vout[j] = clamp_chain over i of (vout[j]*256 + vin[i]*mat[i][j]).
    Output cleared on entry (VTMUL contract). Per-element single-clamp."""
    vout = [0] * cols
    for i in range(rows):
        scalar = vin[i]
        for j in range(cols):
            acc = to_s32(scalar * mat[i][j] + sign_extend_q8_to_q16(vout[j]))
            vout[j] = clamp_extract(acc)
    return vout


def mvadd_ref(mat, vin, vout_init, rows, cols):
    """vout[i] = clamp( sum_j(mat[i][j]*vin[j]) + vout[i]*256 ).
    32-bit row accumulation with sign-extended vout injection + single clamp."""
    out = list(vout_init)
    for i in range(rows):
        acc = 0
        for j in range(cols):
            acc = to_s32(acc + mat[i][j] * vin[j])
        acc = to_s32(acc + sign_extend_q8_to_q16(out[i]))
        out[i] = clamp_extract(acc)
    return out


def outer_ref(mat, vx, vy, rows, cols):
    """mat[i][j] = clamp( mat[i][j]*256 + vx[i]*vy[j] ). Per-element."""
    new_mat = [list(row) for row in mat]
    for i in range(rows):
        for j in range(cols):
            acc = to_s32(vx[i] * vy[j] + sign_extend_q8_to_q16(new_mat[i][j]))
            new_mat[i][j] = clamp_extract(acc)
    return new_mat


# ---- Step references ----

def bkwrd_s4_ref(Q, K, dSc, seq, dim):
    dQ = [vtmul_ref(K, dSc[i], seq, dim) for i in range(seq)]
    dK = []
    for j in range(seq):
        col_j = [dSc[i][j] for i in range(seq)]
        dK.append(vtmul_ref(Q, col_j, seq, dim))
    return dQ, dK


def bkwrd_s5_ref(X, dQ, dK, dV, dY, Wq, Wk, Wv, seq, dim):
    dX = [list(row) for row in dY]
    dWq = [[0] * dim for _ in range(dim)]
    dWk = [[0] * dim for _ in range(dim)]
    dWv = [[0] * dim for _ in range(dim)]
    for i in range(seq):
        dX[i] = mvadd_ref(Wq, dQ[i], dX[i], dim, dim)
        dWq = outer_ref(dWq, X[i], dQ[i], dim, dim)
        dX[i] = mvadd_ref(Wk, dK[i], dX[i], dim, dim)
        dWk = outer_ref(dWk, X[i], dK[i], dim, dim)
        dX[i] = mvadd_ref(Wv, dV[i], dX[i], dim, dim)
        dWv = outer_ref(dWv, X[i], dV[i], dim, dim)
    return dX, dWq, dWk, dWv


def bkwrd_s6_ref(tokens, dX, seq, dim, vocab):
    d_tok = [[0] * dim for _ in range(vocab)]
    d_pos = [[0] * dim for _ in range(seq)]
    clamps = []
    for i in range(seq):
        tok = tokens[i]
        for d in range(dim):
            raw_t = d_tok[tok][d] + dX[i][d]
            if raw_t > 32767 or raw_t < -32768:
                clamps.append(f"d_tok[{tok}][{d}]: raw={raw_t} -> {clamp_q15(raw_t)}")
            d_tok[tok][d] = clamp_q15(raw_t)
            raw_p = d_pos[i][d] + dX[i][d]
            if raw_p > 32767 or raw_p < -32768:
                clamps.append(f"d_pos[{i}][{d}]: raw={raw_p} -> {clamp_q15(raw_p)}")
            d_pos[i][d] = clamp_q15(raw_p)
    return d_tok, d_pos, clamps


# ---- Self-test ----

def self_test():
    assert clamp_extract(0) == 0
    assert clamp_extract(256) == 1
    assert clamp_extract(-256) == -1
    assert clamp_extract(0x7FFFFFFF) == 32767
    assert clamp_extract(-0x80000000) == -32768
    # vtmul trivial: mat=[[256]] (1.0 Q8), vin=[256] -> vout=[256]
    assert vtmul_ref([[256]], [256], 1, 1) == [256]
    # mvadd trivial: vout=[0] + mat=[[256]]*vin=[256] -> [256]
    assert mvadd_ref([[256]], [256], [0], 1, 1) == [256]
    # mvadd accumulate: vout=[100] + 0*0 -> vout unchanged (sign-ext then extract)
    assert mvadd_ref([[0]], [0], [100], 1, 1) == [100]
    # outer accumulate: mat=[[0]], vx=[100], vy=[200] -> 20000>>8=78
    assert outer_ref([[0]], [100], [200], 1, 1) == [[78]]
    # clamp_q15: saturate
    assert clamp_q15(32768) == 32767
    assert clamp_q15(-32769) == -32768
    # Column extraction sanity
    m = [[1, 2], [3, 4]]
    col0 = [m[i][0] for i in range(2)]
    assert col0 == [1, 3]
    col1 = [m[i][1] for i in range(2)]
    assert col1 == [2, 4]
    print("self-test passed")


# ---- Test cases ----

def test_0_s4_only():
    """S4-only: SEQ=2 DIM=2. Verifies dQ + dK from known Q, K, dSc."""
    seq, dim, voc = 2, 2, 10
    Q = [[100, 50], [30, -20]]
    K = [[80, 40], [-10, 60]]
    # V and A not used by Step 4 — emit zeros in workspace
    V = [[0] * dim for _ in range(seq)]
    A = [[0] * seq for _ in range(seq)]
    dSc = [[1000, -500], [2000, 1500]]  # Q15 input to Step 4
    return dict(name="BK456_T0_S4ONLY", mode="s4_only",
                seq=seq, dim=dim, voc=voc,
                Q=Q, K=K, V=V, A=A, dSc=dSc,
                dY=None, dV=None, X=None,
                Wq=None, Wk=None, Wv=None, tokens=None)


def test_1_s456_tiny():
    """Chained S4+S5+S6: SEQ=2 DIM=2 VOC=10. 8 output buffers verified."""
    seq, dim, voc = 2, 2, 10
    Q = [[100, 50], [30, -20]]
    K = [[80, 40], [-10, 60]]
    V = [[0] * dim for _ in range(seq)]
    A = [[0] * seq for _ in range(seq)]
    dSc = [[1000, -500], [2000, 1500]]
    dY = [[500, -300], [700, 400]]
    dV = [[200, -100], [350, -250]]
    X = [[10, 20], [30, -40]]
    Wq = [[50, 25], [-30, 60]]
    Wk = [[40, 70], [20, -50]]
    Wv = [[-10, 80], [60, 30]]
    tokens = [2, 5]
    return dict(name="BK456_T1_TINY", mode="chained",
                seq=seq, dim=dim, voc=voc,
                Q=Q, K=K, V=V, A=A, dSc=dSc,
                dY=dY, dV=dV, X=X,
                Wq=Wq, Wk=Wk, Wv=Wv, tokens=tokens)


def test_2_s456_medium():
    """Chained S4+S5+S6: SEQ=4 DIM=8 VOC=10."""
    seq, dim, voc = 4, 8, 10
    Q = [[((i * 7 + k * 3) % 200) - 100 for k in range(dim)] for i in range(seq)]
    K = [[((i * 5 + k * 11) % 200) - 100 for k in range(dim)] for i in range(seq)]
    V = [[0] * dim for _ in range(seq)]
    A = [[0] * seq for _ in range(seq)]
    dSc = [[((i * 17 + j * 11) % 600) - 300 for j in range(seq)] for i in range(seq)]
    dY = [[((i * 19 + k * 23) % 2000) - 1000 for k in range(dim)] for i in range(seq)]
    dV = [[((i * 29 + k * 31) % 1000) - 500 for k in range(dim)] for i in range(seq)]
    X = [[((i * 41 + k * 37) % 200) - 100 for k in range(dim)] for i in range(seq)]
    Wq = [[((i * 2 + j * 5) % 100) - 50 for j in range(dim)] for i in range(dim)]
    Wk = [[((i * 3 + j * 7) % 120) - 60 for j in range(dim)] for i in range(dim)]
    Wv = [[((i * 5 + j * 11) % 80) - 40 for j in range(dim)] for i in range(dim)]
    tokens = [3, 7, 1, 5]
    return dict(name="BK456_T2_MEDIUM", mode="chained",
                seq=seq, dim=dim, voc=voc,
                Q=Q, K=K, V=V, A=A, dSc=dSc,
                dY=dY, dV=dV, X=X,
                Wq=Wq, Wk=Wk, Wv=Wv, tokens=tokens)


def test_3_clamp():
    """Chained, designed to trigger Step 6 d_tok clamp.

    tokens=[0,0] forces d_tok[0] to accumulate dX[0] + dX[1]. Large
    positive dY pushed through Step 5 (VCPY path dominates) keeps dX
    large, so d_tok[0] element sums exceed Q15 range and clamp.
    """
    seq, dim, voc = 2, 4, 10
    Q = [[100, 50, -30, 20], [80, -40, 60, -10]]
    K = [[40, 70, -20, 50], [-30, 60, 80, 10]]
    V = [[0] * dim for _ in range(seq)]
    A = [[0] * seq for _ in range(seq)]
    dSc = [[20000, -15000], [-18000, 22000]]
    dY = [[25000] * dim, [25000] * dim]
    dV = [[15000] * dim, [15000] * dim]
    X = [[50] * dim, [50] * dim]
    Wq = [[10] * dim for _ in range(dim)]
    Wk = [[10] * dim for _ in range(dim)]
    Wv = [[10] * dim for _ in range(dim)]
    tokens = [0, 0]
    return dict(name="BK456_T3_CLAMP", mode="chained",
                seq=seq, dim=dim, voc=voc,
                Q=Q, K=K, V=V, A=A, dSc=dSc,
                dY=dY, dV=dV, X=X,
                Wq=Wq, Wk=Wk, Wv=Wv, tokens=tokens)


def all_tests():
    return [test_0_s4_only(), test_1_s456_tiny(),
            test_2_s456_medium(), test_3_clamp()]


# ---- Emission helpers ----

def flat_row_major(matrix):
    """Flatten a 2D list row-major."""
    return [v for row in matrix for v in row]


def emit_asm(path, results):
    lines = []
    lines.append("* ============================================================")
    lines.append("* bkwrd456_vectors.asm - BKWRD Steps 4-6 test vectors")
    lines.append("* Generated by gen_bkwrd456_vectors.py - do not hand-edit.")
    lines.append("*")
    lines.append("* WORK_N layout (contiguous, matches BKWRD_INIT workspace):")
    lines.append("*   [Q region SEQ*DIM*2][K region][V region][A region SEQ*SEQ*2]")
    lines.append("* Set BK_WRK = #WORK_N; BKWRD_INIT populates BK_QP/KP/VP/AP.")
    lines.append("* ============================================================")
    lines.append("")

    for idx, r in enumerate(results):
        name = r["name"]
        seq, dim, voc = r["seq"], r["dim"], r["voc"]
        mode = r["mode"]
        lines.append(f"* Test {idx}: {name}  SEQ={seq} DIM={dim} VOC={voc}  mode={mode}")
        lines.append(f"SEQ_{idx}         FDB  {seq}")
        lines.append(f"DIM_{idx}         FDB  {dim}")
        lines.append(f"VOC_{idx}         FDB  {voc}")

        q_flat = flat_row_major(r["Q"])
        k_flat = flat_row_major(r["K"])
        v_flat = flat_row_major(r["V"])
        a_flat = flat_row_major(r["A"])
        lines.append(f"WORK_{idx}        FDB  {','.join(str(v) for v in q_flat)}   ; Q region")
        lines.append(f"WK_{idx}K         FDB  {','.join(str(v) for v in k_flat)}   ; K region")
        lines.append(f"WK_{idx}V         FDB  {','.join(str(v) for v in v_flat)}   ; V region")
        lines.append(f"WK_{idx}A         FDB  {','.join(str(v) for v in a_flat)}   ; A region")

        dsc_flat = flat_row_major(r["dSc"])
        lines.append(f"DSC_{idx}         FDB  {','.join(str(v) for v in dsc_flat)}   ; dSc input -> BW_DATT")

        if mode == "chained":
            dy_flat = flat_row_major(r["dY"])
            dv_flat = flat_row_major(r["dV"])
            x_flat = flat_row_major(r["X"])
            wq_flat = flat_row_major(r["Wq"])
            wk_flat = flat_row_major(r["Wk"])
            wv_flat = flat_row_major(r["Wv"])
            lines.append(f"DY_{idx}          FDB  {','.join(str(v) for v in dy_flat)}")
            lines.append(f"DV_{idx}          FDB  {','.join(str(v) for v in dv_flat)}   ; dV input -> BW_DV")
            lines.append(f"X_{idx}           FDB  {','.join(str(v) for v in x_flat)}")
            lines.append(f"WQ_{idx}          FDB  {','.join(str(v) for v in wq_flat)}")
            lines.append(f"WK_{idx}          FDB  {','.join(str(v) for v in wk_flat)}")
            lines.append(f"WV_{idx}          FDB  {','.join(str(v) for v in wv_flat)}")
            lines.append(f"TOKS_{idx}        FDB  {','.join(str(v) for v in r['tokens'])}")

        # Counts
        lines.append(f"DQ_CNT_{idx}      FDB  {seq * dim}")
        lines.append(f"DK_CNT_{idx}      FDB  {seq * dim}")
        if mode == "chained":
            lines.append(f"DX_CNT_{idx}      FDB  {seq * dim}")
            lines.append(f"DWQ_CNT_{idx}     FDB  {dim * dim}")
            lines.append(f"DWK_CNT_{idx}     FDB  {dim * dim}")
            lines.append(f"DWV_CNT_{idx}     FDB  {dim * dim}")
            lines.append(f"DTOK_CNT_{idx}    FDB  {voc * dim}")
            lines.append(f"DPOS_CNT_{idx}    FDB  {seq * dim}")

        # Expected outputs
        lines.append(f"EXP_DQ_{idx}      FDB  {','.join(str(v) for v in flat_row_major(r['dQ']))}")
        lines.append(f"EXP_DK_{idx}      FDB  {','.join(str(v) for v in flat_row_major(r['dK']))}")
        if mode == "chained":
            lines.append(f"EXP_DX_{idx}      FDB  {','.join(str(v) for v in flat_row_major(r['dX']))}")
            lines.append(f"EXP_DWQ_{idx}     FDB  {','.join(str(v) for v in flat_row_major(r['dWq']))}")
            lines.append(f"EXP_DWK_{idx}     FDB  {','.join(str(v) for v in flat_row_major(r['dWk']))}")
            lines.append(f"EXP_DWV_{idx}     FDB  {','.join(str(v) for v in flat_row_major(r['dWv']))}")
            lines.append(f"EXP_DTOK_{idx}    FDB  {','.join(str(v) for v in flat_row_major(r['d_tok']))}")
            lines.append(f"EXP_DPOS_{idx}    FDB  {','.join(str(v) for v in flat_row_major(r['d_pos']))}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


def emit_log(path, results):
    lines = []
    lines.append("BKWRD Steps 4-6 reference generator log")
    lines.append("=" * 60)
    lines.append("")
    for idx, r in enumerate(results):
        lines.append(f"Test {idx}: {r['name']}  SEQ={r['seq']} DIM={r['dim']} VOC={r['voc']}  mode={r['mode']}")
        lines.append("  Inputs:")
        lines.append(f"    Q:    {r['Q']}")
        lines.append(f"    K:    {r['K']}")
        lines.append(f"    dSc:  {r['dSc']}")
        if r["mode"] == "chained":
            lines.append(f"    dY:   {r['dY']}")
            lines.append(f"    dV:   {r['dV']}")
            lines.append(f"    X:    {r['X']}")
            lines.append(f"    tokens: {r['tokens']}")
        lines.append("  Step 4 outputs:")
        lines.append(f"    dQ: {r['dQ']}")
        lines.append(f"    dK: {r['dK']}")
        if r["mode"] == "chained":
            lines.append("  Step 5 outputs:")
            lines.append(f"    dX:  {r['dX']}")
            lines.append(f"    dWq: {r['dWq']}")
            lines.append(f"    dWk: {r['dWk']}")
            lines.append(f"    dWv: {r['dWv']}")
            lines.append("  Step 6 outputs:")
            for v, row in enumerate(r["d_tok"]):
                if any(x != 0 for x in row):
                    lines.append(f"    d_tok[{v}]: {row}")
            lines.append(f"    d_pos: {r['d_pos']}")
            if r["clamps"]:
                lines.append("  Step 6 clamp events:")
                for c in r["clamps"]:
                    lines.append(f"    {c}")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


# ---- Main ----

def main():
    self_test()

    results = []
    for r in all_tests():
        seq, dim, voc = r["seq"], r["dim"], r["voc"]

        # Step 4
        dQ, dK = bkwrd_s4_ref(r["Q"], r["K"], r["dSc"], seq, dim)
        r["dQ"] = dQ
        r["dK"] = dK

        if r["mode"] == "chained":
            # Step 5
            dX, dWq, dWk, dWv = bkwrd_s5_ref(
                r["X"], dQ, dK, r["dV"], r["dY"],
                r["Wq"], r["Wk"], r["Wv"], seq, dim)
            r["dX"] = dX
            r["dWq"] = dWq
            r["dWk"] = dWk
            r["dWv"] = dWv

            # Step 6
            d_tok, d_pos, clamps = bkwrd_s6_ref(r["tokens"], dX, seq, dim, voc)
            r["d_tok"] = d_tok
            r["d_pos"] = d_pos
            r["clamps"] = clamps

        results.append(r)

    # Safety net: Test 3 must trigger at least one clamp
    t3 = next(r for r in results if r["name"] == "BK456_T3_CLAMP")
    if not t3.get("clamps"):
        raise RuntimeError("Test 3 (clamp trigger) did not fire any clamp — refuse emit")

    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "bkwrd456_vectors.asm", results)
    emit_log(out_dir / "bkwrd456_vectors.log", results)

    print(f"\nwrote tables/bkwrd456_vectors.asm ({len(results)} tests)")
    print("wrote tables/bkwrd456_vectors.log")
    print("\nSummary:")
    for r in results:
        seq, dim, voc = r["seq"], r["dim"], r["voc"]
        parts = [f"dQ({seq*dim}w)", f"dK({seq*dim}w)"]
        if r["mode"] == "chained":
            parts += [f"dX({seq*dim}w)",
                      f"dWq({dim*dim}w)", f"dWk({dim*dim}w)", f"dWv({dim*dim}w)",
                      f"d_tok({voc*dim}w)", f"d_pos({seq*dim}w)"]
        total = sum(int(p.split("(")[1].split("w")[0]) for p in parts)
        print(f"  {r['name']}  SEQ={seq} DIM={dim} VOC={voc}  "
              f"verify: {', '.join(parts)}  total={total}w")
        if r.get("clamps"):
            print(f"    Step 6 clamps: {len(r['clamps'])}")


if __name__ == "__main__":
    main()
