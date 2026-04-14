#!/usr/bin/env python3
"""
gen_embed_vectors.py — Reference vectors for EMBED.

EMBED reference uses wrapping arithmetic (matches ATTN/11 assembly).
The prototype.shf applies clamp16; the PDP-11 assembly does NOT.
This is a pre-existing ATTN/11 inconsistency. Test 5 (WRAP) exercises
a case where the two paths diverge and emits both values as a comment
to catch any future regression to clamping semantics.

Output: tables/embed_vectors.asm (include file, FDB blocks)
Companion log: tables/embed_vectors.log (per-element trace for
Test 2 (max seq) and Test 5 (wrap divergence)).
"""

from pathlib import Path

def wrap16(x):
    """Wrap to signed 16-bit: any int -> [-32768, 32767]."""
    return ((x + 32768) % 65536) - 32768

def clamp16(x):
    """Reference for divergence comment — NOT used in expected output."""
    if x > 32767: return 32767
    if x < -32768: return -32768
    return x

def embed_ref(tokens, tkemb, psemb, seq, dim):
    """
    out[i][j] = wrap16(tkemb[tokens[i]][j] + psemb[i][j])
    Flat list: out[i*dim + j].
    """
    out = []
    for i in range(seq):
        tok = tokens[i]
        for j in range(dim):
            tk_val = tkemb[tok][j]
            ps_val = psemb[i][j]
            raw_sum = tk_val + ps_val
            out.append(wrap16(raw_sum))
    return out

# ---- Test cases ----

def gen_tests():
    """Returns list of (name, tokens, tkemb, psemb, seq, dim, expected)."""
    tests = []

    # Test 0: EMBED_SINGLE_TOKEN — SEQ=1, D=4
    # V=5 rows, pick row 3. Known distinct values.
    tokens_0 = [3]
    tkemb_0 = [
        [100, 200, 300, 400],     # row 0
        [500, 600, 700, 800],     # row 1
        [-100, -200, -300, -400], # row 2
        [1000, 2000, 3000, 4000], # row 3 (selected)
        [5, 10, 15, 20],          # row 4
    ]
    psemb_0 = [
        [10, 20, 30, 40],         # position 0
    ]
    tests.append(("EMBED_SINGLE_TOKEN", tokens_0, tkemb_0, psemb_0, 1, 4))

    # Test 1: EMBED_SEQ2 — SEQ=2, D=4
    # Two different tokens, two positions.
    tokens_1 = [1, 4]
    tkemb_1 = [
        [100, 100, 100, 100],     # row 0
        [200, 200, 200, 200],     # row 1 (selected for position 0)
        [300, 300, 300, 300],     # row 2
        [400, 400, 400, 400],     # row 3
        [500, 500, 500, 500],     # row 4 (selected for position 1)
    ]
    psemb_1 = [
        [1, 2, 3, 4],             # position 0
        [10, 20, 30, 40],         # position 1
    ]
    tests.append(("EMBED_SEQ2", tokens_1, tkemb_1, psemb_1, 2, 4))

    # Test 2: EMBED_MAX_SEQ — SEQ=8, D=16 (architecture max)
    # tokens = [0, 1, 2, ..., 7]
    # tkemb: V=10 rows with pattern tkemb[t][j] = t*100 + j
    # psemb: SEQ=8 rows with pattern psemb[i][j] = i*10 + j*2
    tokens_2 = list(range(8))
    tkemb_2 = [[t * 100 + j for j in range(16)] for t in range(10)]
    psemb_2 = [[i * 10 + j * 2 for j in range(16)] for i in range(8)]
    tests.append(("EMBED_MAX_SEQ", tokens_2, tkemb_2, psemb_2, 8, 16))

    # Test 3: EMBED_TOKEN_ZERO — SEQ=2, D=4
    # Both positions use token 0. Confirms row 0 is a real lookup.
    tokens_3 = [0, 0]
    tkemb_3 = [
        [1111, 2222, 3333, 4444], # row 0
        [9999, 9999, 9999, 9999], # row 1 (must NOT be read)
    ]
    psemb_3 = [
        [1, 2, 3, 4],             # position 0
        [10, 20, 30, 40],         # position 1
    ]
    tests.append(("EMBED_TOKEN_ZERO", tokens_3, tkemb_3, psemb_3, 2, 4))

    # Test 4: EMBED_TOKEN_MAX — SEQ=1, D=4
    # tokens = [9] (V-1 in architecture, last valid row)
    tokens_4 = [9]
    tkemb_4 = [[0]*4 for _ in range(10)]  # rows 0-8 are zero
    tkemb_4[9] = [-1000, -2000, -3000, -4000]  # row 9
    psemb_4 = [[5, 10, 15, 20]]
    tests.append(("EMBED_TOKEN_MAX", tokens_4, tkemb_4, psemb_4, 1, 4))

    # Test 5: EMBED_WRAP — DIVERGENCE TEST
    # SEQ=1, D=1. tkemb[0][0]=30000, psemb[0][0]=10000.
    # Raw sum = 40000 (overflows Q15).
    # Wrapped: 40000 - 65536 = -25536 = $9C40
    # Clamp16 would give: 32767 = $7FFF
    tokens_5 = [0]
    tkemb_5 = [[30000]]
    psemb_5 = [[10000]]
    tests.append(("EMBED_WRAP", tokens_5, tkemb_5, psemb_5, 1, 1))

    # Compute expected
    result = []
    for (name, tokens, tkemb, psemb, seq, dim) in tests:
        expected = embed_ref(tokens, tkemb, psemb, seq, dim)
        result.append((name, tokens, tkemb, psemb, seq, dim, expected))
    return result

# ---- Safety net ----

def verify(tests):
    """Refuse to emit if invariants break."""
    # 1. Test 5 must actually wrap (wrap16 and clamp16 differ)
    test5 = next(t for t in tests if t[0] == "EMBED_WRAP")
    _, tokens, tkemb, psemb, seq, dim, _ = test5
    raw = tkemb[tokens[0]][0] + psemb[0][0]
    wrapped = wrap16(raw)
    clamped = clamp16(raw)
    if wrapped == clamped:
        raise SystemExit(f"ERROR: Test 5 no longer diverges (wrap={wrapped}, clamp={clamped}). "
                         f"Inputs changed — divergence test lost.")

    # 2. All tokens in range [0, len(tkemb)-1]
    for (name, tokens, tkemb, psemb, seq, dim, _) in tests:
        V = len(tkemb)
        for i, t in enumerate(tokens):
            if t < 0 or t >= V:
                raise SystemExit(f"ERROR: {name} token[{i}]={t} out of range [0,{V-1}]")

    # 3. Buffer sizes consistent
    for (name, tokens, tkemb, psemb, seq, dim, expected) in tests:
        if len(tokens) != seq:
            raise SystemExit(f"ERROR: {name} len(tokens)={len(tokens)} != seq={seq}")
        if len(psemb) != seq:
            raise SystemExit(f"ERROR: {name} len(psemb)={len(psemb)} != seq={seq}")
        for row in tkemb:
            if len(row) != dim:
                raise SystemExit(f"ERROR: {name} tkemb row len={len(row)} != dim={dim}")
        for row in psemb:
            if len(row) != dim:
                raise SystemExit(f"ERROR: {name} psemb row len={len(row)} != dim={dim}")
        if len(expected) != seq * dim:
            raise SystemExit(f"ERROR: {name} expected len={len(expected)} != seq*dim={seq*dim}")

# ---- Emit tables/embed_vectors.asm ----

def emit_asm(path, tests):
    lines = []
    lines.append("* ============================================================")
    lines.append("* embed_vectors.asm — test vectors for EMBED harness")
    lines.append("* Generated by gen_embed_vectors.py — do not hand-edit")
    lines.append("* Wrapping arithmetic semantics (matches ATTN/11 assembly)")
    lines.append("* ============================================================")
    lines.append("")

    for i, (name, tokens, tkemb, psemb, seq, dim, expected) in enumerate(tests):
        lines.append(f"* Test {i}: {name}")
        if name == "EMBED_WRAP":
            raw = tkemb[tokens[0]][0] + psemb[0][0]
            w = wrap16(raw)
            c = clamp16(raw)
            lines.append(f"*   DIVERGENCE from clamp16 semantics:")
            lines.append(f"*   wrapping (ours/assembly):  {w}  (${w & 0xFFFF:04X})")
            lines.append(f"*   clamp16 (prototype):        {c}   (${c & 0xFFFF:04X})")
            lines.append(f"*")
            lines.append(f"*   tkemb[0][0] = {tkemb[0][0]}")
            lines.append(f"*   psemb[0][0] = {psemb[0][0]}")
            lines.append(f"*   raw sum     = {raw} (overflows Q15)")
            lines.append(f"*   wrap: {raw} - 65536 = {w}")

        # SEQ, DIM, and precomputed counts
        V = len(tkemb)
        out_cnt = seq * dim
        tkemb_cnt = V * dim
        lines.append(f"SEQ_{i}          FDB {seq}")
        lines.append(f"DIM_{i}          FDB {dim}")
        lines.append(f"CNT_{i}          FDB {out_cnt}         ; seq * dim (output word count)")
        lines.append(f"TKEMB_CNT_{i}    FDB {tkemb_cnt}         ; V * dim (tkemb word count)")

        # TOK
        toks = ",".join(str(t) for t in tokens)
        lines.append(f"TOK_{i}          FDB {toks}")

        # TKEMB (V rows of dim words each)
        tkemb_flat = []
        for row in tkemb:
            tkemb_flat.extend(row)
        tk_s = ",".join(str(v) for v in tkemb_flat)
        lines.append(f"* TKEMB_{i} rows: V={V}, dim={dim}, row-major")
        lines.append(f"TKEMB_{i}        FDB {tk_s}")

        # PSEMB (seq rows of dim words)
        psemb_flat = []
        for row in psemb:
            psemb_flat.extend(row)
        ps_s = ",".join(str(v) for v in psemb_flat)
        lines.append(f"* PSEMB_{i} rows: seq={seq}, dim={dim}")
        lines.append(f"PSEMB_{i}        FDB {ps_s}")

        # EXPECT (seq*dim words)
        exp_s = ",".join(str(v) for v in expected)
        if name == "EMBED_WRAP":
            lines.append(f"EXPECT_{i}       FDB {exp_s}       ; wrapping; clamp16 would be 32767")
        else:
            lines.append(f"EXPECT_{i}       FDB {exp_s}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

# ---- Emit embed_vectors.log ----

def emit_log(path, tests):
    lines = []
    lines.append("EMBED reference generator log")
    lines.append("Wrapping arithmetic semantics")
    lines.append("=" * 60)
    lines.append("")
    for i, (name, tokens, tkemb, psemb, seq, dim, expected) in enumerate(tests):
        lines.append(f"Test {i}: {name}")
        lines.append(f"  seq={seq} dim={dim} tokens={tokens}")
        lines.append(f"  V={len(tkemb)} (tkemb rows)")

        if name == "EMBED_WRAP":
            raw = tkemb[tokens[0]][0] + psemb[0][0]
            w = wrap16(raw)
            c = clamp16(raw)
            lines.append(f"  *** DIVERGENCE ***")
            lines.append(f"  tkemb[0][0]={tkemb[0][0]}, psemb[0][0]={psemb[0][0]}")
            lines.append(f"  raw_sum = {raw}")
            lines.append(f"  wrap16  = {w}  (expected)")
            lines.append(f"  clamp16 = {c}  (what prototype would give)")
            lines.append(f"  expected[0] = {expected[0]}")

        elif name == "EMBED_MAX_SEQ":
            lines.append(f"  Per-element trace (128 elements):")
            for i_row in range(seq):
                tok = tokens[i_row]
                for j in range(dim):
                    tk_val = tkemb[tok][j]
                    ps_val = psemb[i_row][j]
                    raw = tk_val + ps_val
                    w = wrap16(raw)
                    out_idx = i_row * dim + j
                    lines.append(f"    out[{i_row},{j}] = tkemb[{tok}][{j}]={tk_val} + "
                                 f"psemb[{i_row}][{j}]={ps_val} = {raw} -> {w}  "
                                 f"(expect[{out_idx}]={expected[out_idx]})")

        else:
            lines.append(f"  expected = {expected}")

        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")

# ---- Self tests ----

def self_test():
    # wrap16 sanity
    assert wrap16(0) == 0
    assert wrap16(1) == 1
    assert wrap16(32767) == 32767
    assert wrap16(32768) == -32768
    assert wrap16(40000) == -25536
    assert wrap16(-32768) == -32768
    assert wrap16(-32769) == 32767
    assert wrap16(-40000) == 25536

    # clamp16 sanity
    assert clamp16(0) == 0
    assert clamp16(40000) == 32767
    assert clamp16(-40000) == -32768
    assert clamp16(32767) == 32767

    # Basic EMBED
    # tokens=[0], tkemb=[[10, 20]], psemb=[[1, 2]], seq=1, dim=2
    r = embed_ref([0], [[10, 20]], [[1, 2]], 1, 2)
    assert r == [11, 22], f"basic: {r}"

    # Token indexing
    r = embed_ref([1], [[10, 20], [30, 40]], [[1, 2]], 1, 2)
    assert r == [31, 42], f"token_idx: {r}"

    # Wrap case
    r = embed_ref([0], [[30000]], [[10000]], 1, 1)
    assert r == [-25536], f"wrap: {r}"

    print("self_test: PASS")

# ---- Main ----

def main():
    self_test()
    tests = gen_tests()
    verify(tests)
    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)
    emit_asm(out_dir / "embed_vectors.asm", tests)
    emit_log(out_dir / "embed_vectors.log", tests)
    print(f"wrote tables/embed_vectors.asm ({len(tests)} tests)")
    print(f"wrote tables/embed_vectors.log (per-element trace for MAX_SEQ and WRAP)")
    print()
    print("Summary:")
    for i, (name, tokens, tkemb, psemb, seq, dim, expected) in enumerate(tests):
        if name == "EMBED_WRAP":
            raw = tkemb[tokens[0]][0] + psemb[0][0]
            w = wrap16(raw)
            c = clamp16(raw)
            print(f"  Test {i} ({name}): DIVERGENCE wrap={w} vs clamp={c}")
        else:
            print(f"  Test {i} ({name}): seq={seq} dim={dim}, {len(expected)} outputs")

if __name__ == "__main__":
    main()
