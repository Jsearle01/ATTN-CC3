# PROJ Implementation Plan

Exploration report for PROJ, the forward-pass output projection.
Based on reading LAYER.MAC (lines 72-132), FORWRD.MAC (call site),
TRAIN.MAC (weight and buffer layouts), and prototype.shf.

## Section A — Algorithm (Q1)

**PROJ is a per-position VTMUL loop.** Not MVMUL.

For each position `i` in the sequence:
```
logits[i] = Wout^T · Y[i]
```

The implementation (LAYER.MAC lines 104-121):

1. Outer loop over SEQ positions.
2. Per iteration: call VTMUL with (mat=Wout, vin=Y[i], vout=logits[i], rows=d_model, cols=vocab).
3. Advance YIN pointer by d_model*2 bytes (one input row).
4. Advance LOG pointer by vocab*2 bytes (one output row).
5. Decrement SEQ counter; loop until zero.

The VTMUL call uses ATTN/11's "self-modifying code" pattern: PROJ patches
VTMUL's inline `.WORD` parameters at labels PJ.P1-P5 before each JSR R5
(ATTN/11 line 104-116). This is a PDP-11 idiom. On 6309 we fill our
MP_* DP parameter block instead — cleaner.

**No bias.** No learned bias vector in the prototype (line 83: `Wout`
is the only output-projection learnable). No bias handling in the
assembly.

**No activation.** No tanh/ReLU applied in PROJ. Softmax is NOT called
inside PROJ — see Q3.

**Per-position, not full-sequence.** PROJ calls VTMUL once per
position. The prototype confirms this pattern: line 124 `((vmap (fn
[yi] (vtmul Wout yi))) Y)` — vmap over positions, VTMUL per position.

## Section B — Output dimension (Q2)

- **Input:** `Y[seq_len][d_model]` = `[8][16]` = 128 Q8 words = 256 bytes
- **Output:** `logits[seq_len][vocab_size]` = `[8][10]` = 80 Q8 words = 160 bytes
- **Weight matrix:** `Wout[d_model][vocab_size]` = `[16][10]` = 160 Q8 words = 320 bytes

Weight layout confirmation (prototype.shf line 83):
`Wout (q16 ... (dims d v))` — shape is `[d, v]` = `[d_model, vocab]`.

So `Wout` is stored row-major with **d_model rows and vocab_size
columns**. PROJ calls VTMUL to compute `Wout^T · Y[i]`:
- VTMUL treats `mat` as the raw `[d_model][vocab]` matrix.
- Implicitly transposes during the multiply: `vout[j] = sum_i(mat[i][j] * vin[i])`.
- So `vout` has `cols = vocab` elements, summing over `rows = d_model` inputs.

VTMUL's signature (from our matop.asm) accepts `mat[rows×cols]`,
`vin[rows]`, writes `vout[cols]`. For PROJ: rows=d_model=16,
cols=vocab=10. The transpose is baked into VTMUL's semantics.

## Section C — Softmax integration (Q3)

**Softmax is NOT applied inside PROJ.** PROJ produces raw Q8 logits.

Evidence:
1. ATTN/11 LAYER.MAC PROJ ends with the VTMUL loop and RTS. No SFTMX call.
2. The prototype.shf line 124 produces `logits` (raw scores).
3. In training (TRAIN.MAC), softmax over logits is called only in
   CLOSS (loss computation) and in BKWRD (gradient step 1).
4. The ATTN/11 training loop uses `argmax(logits)` for accuracy
   measurement, which doesn't require softmax.

**For our test harness:** expected values are raw Q8 logits (what
VTMUL produces), not probabilities. No SFTMX dependency in PROJ's
test path.

## Section D — Memory layout (Q4)

**Input `Y[seq_len][d_model]`:**
- ATTN/11 stores at `YY` (TRAIN.MAC line 523: `YY: .BLKW 8.*16.`).
- This is the attention output with residual — ATTN writes here.
- In our memory map, this maps to the forward cache region. Looking at
  equates.inc, we have FC_X, FC_Q, FC_K, FC_V, FC_ATT, FC_CTX,
  FC_LOGITS, FC_PROBS. **FC_CTX** is the closest analog (SEQ*D words
  at offset 4*DSEQ + SEQ*SEQ in FWD_CACHE). But the naming is
  ATTN-internal; we'll want a dedicated FC_Y or similar, added when
  ATTN is implemented. **For PROJ testing, the input address is
  caller-provided; we just need a test buffer.**

**Output `logits[seq_len][vocab_size]`:**
- ATTN/11: `LOGITS: .BLKW 8.*10.` (80 words = 160 bytes).
- Our equates already has `FC_LOGITS` (equates.inc line 70: `FC_LOGITS EQU FC_CTX+DSEQ`).
- **But FC_LOGITS is a Q15 region in our equates comment; PROJ's output
  is Q8. This is a labeling question we'll face when main.asm is
  written.** For Stage 1 test harness, PROJ's output goes to a test
  buffer, not FC_LOGITS.

**Weights `Wout`:**
- ATTN/11: `WOTQ8: .BLKW 16.*10.` (Q8 copy of Wout, 160 words = 320
  bytes). WOTH/WOTL (Q16 pair, 320 bytes each) are the master
  accumulators.
- In our memory map: `W_OUT` at WEIGHT_BASE + offsets (Stage 1
  contract). 320 bytes for the Q16 representation.

**Harness allocation:** matches t_embed pattern. Use runtime buffers
in $1760+ range for Y_BUF, WOUT_BUF, and LOGITS_BUF. Copy from
vectors include, call PROJ, verify logits against expected.

## Section E — Calling convention (Q5)

**ATTN/11 call (FORWRD.MAC line 20-21):**
```
JSR  R5, PROJ
.WORD YY, WOTQ8, LOGITS, SEQ.LN, D.MODL, VOCAB
```

Six inline parameters, fetched via `MOV (R5)+, ...` in the routine
prologue (LAYER.MAC lines 91-96).

**Our 6309 equivalent:** DP parameter block matching EMBED's style.
6 caller-filled fields + some internal state (advancing pointers,
pre-computed row sizes).

Proposed parameter block:
```
PR_YIN          EQU  BW_SCRATCH+100   ; 2B input pointer (Y buffer)
PR_WOT          EQU  BW_SCRATCH+102   ; 2B Wout weight matrix pointer
PR_LOG          EQU  BW_SCRATCH+104   ; 2B output logits pointer
PR_SEQ          EQU  BW_SCRATCH+106   ; 2B sequence length
PR_DIM          EQU  BW_SCRATCH+108   ; 2B d_model
PR_VOC          EQU  BW_SCRATCH+110   ; 2B vocab size
```

Internal state:
```
PR_YSZ          EQU  BW_SCRATCH+112   ; 2B input row size bytes (= DIM*2)
PR_LSZ          EQU  BW_SCRATCH+114   ; 2B output row size bytes (= VOC*2)
```

8 fields × 2 bytes = 16 bytes. BW_SCRATCH grows 100 → 116 bytes.
BW_END $4484 → $4494. STACK_TOP gap 892 → 876 bytes.

**PROJ does not preserve caller's Y/logits pointers** (not needed —
caller re-fills DP per call). EMBED preserved EM_TOK via EM_TPTR
because the caller might reuse EM_TOK; for PROJ, the pointers in
PR_YIN and PR_LOG advance during the outer loop and are not expected
to survive the call. Document this in the header.

## Section F — Primitive gap analysis

Every operation in PROJ maps cleanly to existing primitives.

| PROJ operation | Primitive | Our status |
|----------------|-----------|------------|
| Per-position VTMUL call | VTMUL | ✓ MATOP |
| Pointer advance by row size | `LEAX D,X` inline | ✓ 6309 native |
| Decrement outer counter | `DEC ,S / BNE` | ✓ standard pattern |
| Row size pre-computation (DIM*2, VOC*2) | LSLD | ✓ standard |

**No missing primitives.** PROJ is a pure wrapper around VTMUL, no
novel operations.

**MULSCR shared scratch:** VTMUL uses MULSCR internally for the 32-bit
accumulation and single-clamp pattern. PROJ does NOT touch MULSCR
directly — it only calls VTMUL and lets MATOP manage the scratch. No
conflict.

## Section G — Memory budget impact

**Code size estimate:** ~60-80 bytes.

Structure:
- Prologue: 12-18 bytes (load params, compute YSZ/LSZ)
- Outer loop setup: 4-6 bytes (counter on stack)
- Per-iteration: fill MP_* block (~20 bytes), JSR VTMUL (3 bytes), advance pointers (6 bytes), decrement+branch (4 bytes) ≈ 33 bytes
- Loop tail + RTS: 4-6 bytes

Expected total: ~65-75 bytes. Plan says ~50-80. On target.

**Parameter block:** 16 bytes (6 caller + 2 internal).

**BW_SCRATCH evolution:**

| Step | Size | End | Gap | Reason |
|------|------|-----|-----|--------|
| After EMBED | 100 | $4484 | 892 | EM_* block |
| After PROJ | 116 | $4494 | 876 | PR_* block + YSZ/LSZ |

892 → 876 bytes of stack headroom. Still ample.

**Output buffer for test harness:** Max case is SEQ=8, VOCAB=10 = 80
words = 160 bytes. Fits in harness scratch region without issue.

**Code region runway:** After EMBED, code end is ~$10AF. PROJ adds
~75 bytes → ~$10FA. That leaves ~$0906 = 2310 bytes to WEIGHT_BASE
$1A00. ATTN plan estimates 250 bytes; plenty of room.

## Section H — Risk assessment

**PROJ risk: LOW.**

Justification:
- Pure wrapper around an existing validated primitive (VTMUL, 5/5
  tested in MATOP).
- No new algorithmic content — just a loop with parameter setup.
- No clamping decisions — VTMUL's single-clamp is what we want
  (matches MATOP convention, produces raw Q8 logits as the prototype
  specifies).
- No softmax integration, no bias, no activation — all would be
  Medium risk territory but absent here.
- No primitive gaps.
- Pointer-advance math is straightforward (LEAX by pre-computed row
  size bytes).

Risks that would escalate:
- If `tests against prototype.shf fail`, it would likely be a VTMUL
  call signature misalignment (passing wrong mat/vin/vout/rows/cols)
  rather than a PROJ-internal bug. Test harness targets VTMUL
  behavior indirectly.
- If `sequence-loop pointer advance is off by one row`, the first
  position would be correct and later ones shifted. Easy to detect in
  SEQ≥2 tests.

## Section I — Implementation order proposal

No new primitives needed. Standard 5-step sequence matching EMBED:

1. **Equates expansion** (PR_* parameter block + 2 internal fields).
   BW_SCRATCH 100 → 116 bytes. Review gate.

2. **PROJ source** appended to src/vecop.asm after EMBED. Review gate.

3. **Reference generator** `tables/gen_proj_vectors.py`. Computes
   VTMUL output per position against raw Q8 arithmetic, bit-exact
   with what our MATOP VTMUL produces. Follows gen_embed_vectors.py
   pattern. Review gate.

4. **Test harness** `test/t_proj.asm`. ~6 test vectors (see Section J).
   Review gate.

5. **Assemble, run under MAME, commit.**

### Section J — Proposed test coverage

6 test vectors, following EMBED's coverage pattern:

1. **PROJ_SINGLE_POS** — SEQ=1, D=4, V=3. One position, tiny matrix.
   Hand-computable.

2. **PROJ_SEQ2** — SEQ=2, D=4, V=3. Two positions, verify each
   computed independently with correct pointer advance.

3. **PROJ_FULL** — SEQ=8, D=16, V=10 (architecture max). 80 output
   elements. This is the realistic case.

4. **PROJ_IDENTITY_WEIGHTS** — SEQ=2, D=4, V=4, with Wout = identity.
   Output should equal input (element-wise, for each position).
   Validates that the transpose-multiply semantics are correct.

5. **PROJ_ZERO_INPUT** — SEQ=2, D=4, V=3, Y all zeros. Output should
   be all zeros regardless of Wout.

6. **PROJ_SATURATION** — SEQ=1, D=4, V=2, values chosen to cause
   VTMUL's single-clamp to saturate. Confirms clamp is active and
   direction is correct. (Same saturation logic as MVMUL tests —
   single-clamp, accumulating dot product overflows Q8.)

**No divergence test needed.** PROJ inherits VTMUL's single-clamp
semantics; there's no alternate path to diverge from. If we ever
regressed VTMUL, the MATOP test suite would catch it first.

## Section K — Open questions

**Q: VTMUL rows/cols mapping.** Our MATOP VTMUL:
```
vout[j] = sum_i(mat[i][j] * vin[i])  for j in 0..cols-1
```
Takes `rows` = number of rows in mat and input vector length, `cols` =
number of columns and output vector length. For PROJ: rows = d_model
(16), cols = vocab (10). Consistent with ATTN/11's PROJ call (line
111 patches PJ.P4 = DIM = 16, PJ.P5 = VOC = 10). **Confirmed.**

**Q: What happens if SEQ=0?** ATTN/11's PROJ has `DEC PJ.SEQ / BNE
PJ.LP` at the bottom — if SEQ=0 on entry, it decrements to $FFFF,
BNE stays taken, loops ~65535 times. Classic underflow bug. Our
implementation should guard with `BEQ exit` at entry (like VSADD,
EMBED, VCPY/VCLR). **Document this as a deviation from ATTN/11**:
our PROJ handles SEQ=0 gracefully; ATTN/11's would hang. Match
the pattern of all our other looping routines.

**Q: Does PROJ need to preserve caller's Y/Wout/LOG pointers?** Looking
at FORWRD.MAC, after PROJ returns the caller doesn't access them
again. ATTN/11's PROJ doesn't preserve them either (PJ.YIN and
PJ.LOG are both advanced in place during the loop). We can match
— advance PR_YIN and PR_LOG during the loop. After RTS, caller's
original pointers are overwritten in DP. If a caller needs them
preserved, it saves them before calling. **Document behavior in
header.**
