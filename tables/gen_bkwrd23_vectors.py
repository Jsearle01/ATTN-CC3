#!/usr/bin/env python3
"""
BKWRD Steps 2-3 reference generator.

Step 2 (BKWRD_S2):
    dV[j] = 0  (zeroed at Step 2 entry)
    for i,j in SEQ^2:
        dA[i][j] = VDOT(V[j], dY[i], DIM)    # Q8 * Q15 >> 8 -> Q15
        dV[j]  += VSADD(A[i][j], dY[i], DIM) # scalar * vector, two-stage clamp

Step 3 (BKWRD_S3, in-place over dA):
    for i:
        dot_ad = VDOT(A[i], dA[i], SEQ)
        for j:
            diff   = clamp16(dA[i][j] - dot_ad)
            prod32 = diff * A[i][j]
            q15    = wrap16(prod32 >> 8)         # NO clamp on extract (ATTN/11)
            dSc[i][j] = q15 >> BK_SHF            # arithmetic (sign-preserving)

VDOT extraction = MSB-byte test + bits [23:8] with clamp to +/-32767.
VSADD stage 1  = same as VDOT extract (scalar * src >> 8, clamp Q16).
VSADD stage 2  = dst += stage1, clamp on signed overflow using sign(stage1).

The workspace block (BK_WRK) is laid out Q/K/V/A contiguous. For Steps 2-3
only V and A are read; Q and K are emitted as zero-filled placeholders so
the binary image matches the layout BKWRD_INIT produces from BK_WRK.
"""

from pathlib import Path


# ---- Primitives ----

def to_s32(v):
    v &= 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    return v


def wrap16(v):
    v &= 0xFFFF
    if v >= 0x8000:
        v -= 0x10000
    return v


def vdot_extract(acc_s32):
    """MSB-byte check + bits [23:8] extraction with clamp. Matches VDOT."""
    acc_u = acc_s32 & 0xFFFFFFFF
    msb = (acc_u >> 24) & 0xFF
    if msb == 0x00 or msb == 0xFF:
        result = (acc_u >> 8) & 0xFFFF
        if result >= 0x8000:
            result -= 0x10000
        return result
    if msb & 0x80:
        return -32768
    return 32767


def vdot_ref(x_vec, y_vec, n):
    acc = 0
    for k in range(n):
        acc = to_s32(acc + x_vec[k] * y_vec[k])
    return vdot_extract(acc)


def vsadd_stage1(scalar, src_k):
    """Q8-clamped scaled product. Same bit-pattern as VDOT extract."""
    return vdot_extract(to_s32(scalar * src_k))


def vsadd_stage2(dst_k, stage1):
    """dst += stage1, clamp on signed overflow using sign of stage1."""
    result = dst_k + stage1
    if -32768 <= result <= 32767:
        return result
    return 32767 if stage1 >= 0 else -32768


def clamp16_sub(a, b):
    """D = a - b with Q15 signed-overflow clamp (matches SUBD + BVC/BPL pattern)."""
    result = a - b
    if -32768 <= result <= 32767:
        return result
    return 32767 if result > 32767 else -32768


def step3_extract(prod32):
    """>>8 with wrap (NO clamp). Matches STQ MULSCR / LDD MULSCR+1 (bits 23:8)."""
    return wrap16(prod32 >> 8)


def asr(val, shf):
    """Arithmetic right shift (sign-preserving). Matches 6309 ASRD loop."""
    return val >> shf  # Python >> is arithmetic for negatives


# ---- Step 2 + Step 3 reference ----

def step2_ref(A, V, dY, seq, dim):
    dA = [[0] * seq for _ in range(seq)]
    dV = [[0] * dim for _ in range(seq)]
    for i in range(seq):
        for j in range(seq):
            dA[i][j] = vdot_ref(V[j], dY[i], dim)
            scalar = A[i][j]
            for k in range(dim):
                s1 = vsadd_stage1(scalar, dY[i][k])
                dV[j][k] = vsadd_stage2(dV[j][k], s1)
    return dA, dV


def step3_ref(A, dA_in, seq, shf):
    """In-place: returns dSc (same shape as dA)."""
    dSc = [row[:] for row in dA_in]
    for i in range(seq):
        dot_ad = vdot_ref(A[i], dSc[i], seq)
        for j in range(seq):
            diff = clamp16_sub(dSc[i][j], dot_ad)
            q15 = step3_extract(to_s32(diff * A[i][j]))
            dSc[i][j] = asr(q15, shf)
    return dSc


# ---- Self-test ----

def self_test():
    assert vdot_extract(0) == 0
    assert vdot_extract(256) == 1
    assert vdot_extract(-256) == -1
    assert vdot_extract(0x7FFFFFFF) == 32767
    assert vdot_extract(-0x80000000) == -32768
    # VDOT with mixed signs
    assert vdot_ref([100, -50], [1000, -500], 2) == (100 * 1000 + (-50) * (-500)) >> 8
    # VSADD stage 1 and 2
    assert vsadd_stage1(64, 1000) == (64 * 1000) >> 8
    assert vsadd_stage2(10000, 30000) == 32767  # +clamp
    assert vsadd_stage2(-10000, -30000) == -32768  # -clamp
    assert vsadd_stage2(100, -50) == 50
    # Clamp subtract
    assert clamp16_sub(30000, -10000) == 32767
    assert clamp16_sub(-30000, 10000) == -32768
    assert clamp16_sub(100, 50) == 50
    # Step3 extract (no clamp)
    assert step3_extract(to_s32(135 * 256 + 200)) == 135  # middle bytes
    # ASR matches Python >>
    assert asr(-135, 1) == -68
    assert asr(135, 1) == 67
    print("self-test passed")


# ---- Test data ----

def test_0_step2_only():
    """TINY Step-2-only: SEQ=2 DIM=2. Hand-verifiable."""
    seq, dim, shf = 2, 2, 0
    A = [[64, 192], [128, 128]]      # Q8
    V = [[100, -50], [-20, 80]]      # Q8
    dY = [[1000, -500], [2000, -1500]]  # Q15
    return ("BK23_T0_S2ONLY", seq, dim, shf, A, V, dY, "step2_only")


def test_1_chained_tiny():
    """TINY chained: same inputs as Test 0 but SHF=1 to exercise the shift."""
    seq, dim, shf = 2, 2, 1
    A = [[64, 192], [128, 128]]
    V = [[100, -50], [-20, 80]]
    dY = [[1000, -500], [2000, -1500]]
    return ("BK23_T1_CHAIN_TINY", seq, dim, shf, A, V, dY, "chained")


def test_2_chained_medium():
    """MEDIUM chained: SEQ=4 DIM=8 SHF=3."""
    seq, dim, shf = 4, 8, 3
    # Synthetic Q8 non-negative A (attention-like: not softmax, just test data)
    A = [[((i * 3 + j * 7) % 200) + 20 for j in range(seq)] for i in range(seq)]
    # Synthetic Q8 V (signed)
    V = [[((i * 5 + k * 11) % 200) - 100 for k in range(dim)] for i in range(seq)]
    # Synthetic Q15 dY
    dY = [[((i * 13 + k * 17) % 4000) - 2000 for k in range(dim)] for i in range(seq)]
    return ("BK23_T2_CHAIN_MED", seq, dim, shf, A, V, dY, "chained")


def test_3_chained_clamp():
    """CLAMP trigger: designed to exercise Step 3's BVC/BPL clamp both ways.

    dA after Step 2 saturates to [32767, -32768] in row 0 and [-32768, 32767]
    in row 1. dot_ad signs are opposite per row so both +clamp and -clamp fire.
    """
    seq, dim, shf = 2, 4, 0
    # A rows different so dot_ad[0] != dot_ad[1]
    A = [[50, 200], [60, 180]]
    # V asymmetric so dot products saturate predictably
    V = [[32000, 32000, 0, 0], [0, 0, -32000, -32000]]
    dY = [[32000, 32000, 32000, 32000],
          [-32000, -32000, -32000, -32000]]
    return ("BK23_T3_CLAMP", seq, dim, shf, A, V, dY, "chained")


def all_tests():
    return [test_0_step2_only(),
            test_1_chained_tiny(),
            test_2_chained_medium(),
            test_3_chained_clamp()]


# ---- Emission ----

def emit_asm(path, results):
    lines = []
    lines.append("* ============================================================")
    lines.append("* bkwrd23_vectors.asm — BKWRD Steps 2-3 test vectors")
    lines.append("* Generated by gen_bkwrd23_vectors.py — do not hand-edit.")
    lines.append("* Workspace layout per test: [Q zeros][K zeros][V data][A data]")
    lines.append("* so BK_WRK = WORK_N puts V at BK_WRK+2*SEQ*DIM*2, A at +3*.")
    lines.append("* ============================================================")
    lines.append("")

    for idx, r in enumerate(results):
        name = r["name"]
        seq, dim, shf = r["seq"], r["dim"], r["shf"]
        mode = r["mode"]
        lines.append(f"* Test {idx}: {name}  SEQ={seq} DIM={dim} SHF={shf}  mode={mode}")
        lines.append(f"SEQ_{idx}         FDB  {seq}")
        lines.append(f"DIM_{idx}         FDB  {dim}")
        lines.append(f"SHF_{idx}         FDB  {shf}")

        # Workspace block: Q zeros + K zeros + V data + A data
        qk_words = seq * dim
        q_zeros = ','.join(['0'] * qk_words)
        k_zeros = ','.join(['0'] * qk_words)
        v_flat = [v for row in r["V"] for v in row]
        a_flat = [v for row in r["A"] for v in row]

        lines.append(f"WORK_{idx}        FDB  {q_zeros}   ; Q region (unused)")
        lines.append(f"WK_{idx}K         FDB  {k_zeros}   ; K region (unused)")
        lines.append(f"WK_{idx}V         FDB  {','.join(str(v) for v in v_flat)}   ; V region")
        lines.append(f"WK_{idx}A         FDB  {','.join(str(v) for v in a_flat)}   ; A region")

        # dY data (separate buffer, points at BK_DY)
        dy_flat = [v for row in r["dY"] for v in row]
        lines.append(f"DY_{idx}          FDB  {','.join(str(v) for v in dy_flat)}")

        # Expected outputs
        lines.append(f"DA_CNT_{idx}      FDB  {seq * seq}")
        lines.append(f"DV_CNT_{idx}      FDB  {seq * dim}")

        if mode == "step2_only":
            da_flat = [v for row in r["dA"] for v in row]
            lines.append(f"EXP_DA_{idx}      FDB  {','.join(str(v) for v in da_flat)}")
        # All tests emit EXP_DV (post-Step-2 state; Step 3 doesn't touch dV)
        dv_flat = [v for row in r["dV"] for v in row]
        lines.append(f"EXP_DV_{idx}      FDB  {','.join(str(v) for v in dv_flat)}")

        if mode == "chained":
            dsc_flat = [v for row in r["dSc"] for v in row]
            lines.append(f"EXP_DSC_{idx}     FDB  {','.join(str(v) for v in dsc_flat)}")

        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


def emit_log(path, results):
    lines = []
    lines.append("BKWRD Steps 2-3 reference generator log")
    lines.append("=" * 60)
    lines.append("")

    for idx, r in enumerate(results):
        lines.append(f"Test {idx}: {r['name']}  SEQ={r['seq']} DIM={r['dim']} SHF={r['shf']}  mode={r['mode']}")
        lines.append("  A:")
        for i, row in enumerate(r["A"]):
            lines.append(f"    [{i}] = {row}")
        lines.append("  V:")
        for i, row in enumerate(r["V"]):
            lines.append(f"    [{i}] = {row}")
        lines.append("  dY:")
        for i, row in enumerate(r["dY"]):
            lines.append(f"    [{i}] = {row}")
        lines.append("  dA (post-Step-2):")
        for i, row in enumerate(r["dA"]):
            lines.append(f"    [{i}] = {row}")
        lines.append("  dV (post-Step-2):")
        for i, row in enumerate(r["dV"]):
            lines.append(f"    [{i}] = {row}")
        if r["mode"] == "chained":
            lines.append("  dot_ad per row:")
            for i, v in enumerate(r["dot_ad"]):
                lines.append(f"    row {i}: {v}")
            lines.append("  dSc (post-Step-3, in-place over dA):")
            for i, row in enumerate(r["dSc"]):
                lines.append(f"    [{i}] = {row}")
            if r["clamps"]:
                lines.append("  Step 3 clamp events:")
                for c in r["clamps"]:
                    lines.append(f"    {c}")
        if r["saturations"]:
            lines.append("  VDOT saturations in Step 2:")
            for s in r["saturations"]:
                lines.append(f"    {s}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


# ---- Main ----

def main():
    self_test()

    results = []
    for (name, seq, dim, shf, A, V, dY, mode) in all_tests():
        # Step 2
        dA, dV = step2_ref(A, V, dY, seq, dim)

        # Detect saturations from Step 2
        saturations = []
        for i in range(seq):
            for j in range(seq):
                if dA[i][j] in (32767, -32768):
                    saturations.append(f"dA[{i}][{j}] = {dA[i][j]}")

        r = {"name": name, "seq": seq, "dim": dim, "shf": shf, "mode": mode,
             "A": A, "V": V, "dY": dY, "dA": dA, "dV": dV,
             "saturations": saturations, "clamps": []}

        if mode == "chained":
            # Track clamps during Step 3 for the log
            dot_ad = []
            clamps = []
            dSc = [row[:] for row in dA]
            for i in range(seq):
                d = vdot_ref(A[i], dSc[i], seq)
                dot_ad.append(d)
                for j in range(seq):
                    raw = dSc[i][j] - d
                    if raw > 32767:
                        clamps.append(f"row {i}, j={j}: diff={raw} -> +32767")
                    elif raw < -32768:
                        clamps.append(f"row {i}, j={j}: diff={raw} -> -32768")
                    diff = clamp16_sub(dSc[i][j], d)
                    q15 = step3_extract(to_s32(diff * A[i][j]))
                    dSc[i][j] = asr(q15, shf)
            r["dot_ad"] = dot_ad
            r["dSc"] = dSc
            r["clamps"] = clamps

        results.append(r)

    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "bkwrd23_vectors.asm", results)
    emit_log(out_dir / "bkwrd23_vectors.log", results)

    print(f"\nwrote tables/bkwrd23_vectors.asm ({len(results)} tests)")
    print("wrote tables/bkwrd23_vectors.log")
    print("\nSummary:")
    for idx, r in enumerate(results):
        suffix = f" (+dSc {r['seq']*r['seq']}w)" if r["mode"] == "chained" else " (+dA verify only)"
        print(f"  Test {idx} ({r['name']}): SEQ={r['seq']} DIM={r['dim']} SHF={r['shf']}  "
              f"dV={r['seq']*r['dim']}w{suffix}")
        if r["clamps"]:
            print(f"    clamps in Step 3: {len(r['clamps'])}")


if __name__ == "__main__":
    main()
