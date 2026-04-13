# LAYER Implementation Plan

Exploration report for the LAYER layer of the ATTN-CC3 port.
Based on reading LAYER.MAC, BKWRD.MAC, UPDAT.MAC, FORWRD.MAC,
TRAIN.MAC, prototype.shf, and prototype-float.shf.

## Section A — Routine Inventory

ATTN/11 distributes "LAYER" across three files. What the task
prompt calls "LAYER" is actually the combination of:

### nn11/LAYER.MAC (library level)

**1. EMBED** — Token + position embedding lookup
- Input: tokens[SEQ], tok_embed[V×D], pos_embed[SEQ×D]
- Output: X[SEQ×D] = tok_embed[token[i]] + pos_embed[i] per position
- Primitives called: none (manual loop with MOV+ADD)
- PDP-11: ~30 instructions
- Private storage: 7 words (EM.TOK through EM.RSZ)
- Risk: Low. Straightforward copy+add loop.

**2. PROJ** — Output projection: logits = Y . Wout
- Input: Y[SEQ×D], Wout[D×V]
- Output: logits[SEQ×V]
- Primitives called: VTMUL (per row)
- PDP-11: ~25 instructions + VTMUL calls
- Private storage: 8 words (PJ.YIN through PJ.LSZ)
- Risk: Low. Loop calling VTMUL with pointer advancement.
- Note: PDP-11 uses self-modifying code to patch VTMUL inline
  params. On 6309, we fill our DP parameter block instead.

**3. ATTN** — Self-attention forward pass (the big one)
- Input: X[SEQ×D], Wq[D×D], Wk[D×D], Wv[D×D], workspace
- Output: Y[SEQ×D] = attention(X) + X (residual)
- Algorithm (7 steps):
  1. Q = batch_VTMUL(Wq, X)
  2. K = batch_VTMUL(Wk, X)
  3. V = batch_VTMUL(Wv, X)
  4. S[i][j] = VDOT(Q[i], K[j]) >> sqrt_shift
  5. A[i] = SFTMX(S[i]) per row
  6. O[i] = VTMUL(V, A[i]) per row (weighted sum)
  7. Y[i] = O[i] + X[i] (residual, element-wise)
- Primitives: VTMUL, VDOT, SFTMX, VADD (element-wise add)
- Private storage: 25 words (AT.XIN through AT.BC)
- PDP-11: ~100 instructions + primitive calls
- Risk: **High**. Complex nested loops, workspace partitioning,
  many pointer arithmetic chains, self-modifying code pattern.
- Internal subroutine: AT.BPR (batch project, calls VTMUL in loop)

### model/BKWRD.MAC (model level, backward pass)

**4. BKWRD** — Full backward pass
- Input: Forward cache (X, Q, K, V, A, Y, logits), targets
- Output: All gradient accumulators (dWq, dWk, dWv, dWout, dTokEmb, dPosEmb)
- Algorithm (6 steps):
  1. dLogits: softmax(logits) - one_hot(target), shift to Q15
  2. dWout += OUTER(Y[i], dL); dY[i] = MVMUL(Wout, dL)
  3. Backward O=A.V: dA[i][j] = VDOT(V[j], dY[i]); dV += VSADD
  4. Backward softmax: dSc = A*(dA-dot(A,dA)) >> 8 >> sqrt_shift
  5. dQ = VTMUL(K, dSc); dK = VTMUL(Q, dSc_col) (column extract)
  6. dX accumulation + embedding gradients
- Primitives: SFTMX, MVMUL, OUTER, VTMUL, MVADD, VDOT, VCPY, VCLR
- **New primitive needed**: VSADD (vector scale-add, defined in BKWRD.MAC)
- **New primitive needed**: VCPY (vector copy, from VECOP.MAC)
- **New primitive needed**: VCLR (vector clear, from VECOP.MAC)
- Risk: **High**. Most complex routine in the system. Column
  extraction, softmax backward, saturating element-add.

### model/UPDAT.MAC (model level, SGD)

**5. UPDAT** — SGD weight update
- Updates Q16 weight accumulators: w -= grad >> (lr_shift - 1)
- Zeros gradients after reading
- Per-weight: 32-bit subtraction with sign-extension
- Risk: Medium. 32-bit arithmetic on the Q16 accumulators.

**6. INITW** — Weight initialization (random Q16)
- LCG PRNG, random Q8 → Q16 via shift
- Risk: Low.

**7. RAND** — Pseudo-random number generator (15-bit LCG)
- Risk: Low.

## Section B — Forward vs. Backward Separation

### Forward pass (FORWRD.MAC calls):
1. EMBED → X[SEQ×D] (stored in XX)
2. ATTN → Y[SEQ×D] (stored in YY, workspace in WORK)
3. PROJ → logits[SEQ×V] (stored in LOGITS)

Forward activations that must persist for backward:
- XX (embeddings): SEQ×D = 128 words = 256 bytes
- YY (attention output): SEQ×D = 128 words = 256 bytes
- LOGITS: SEQ×V = 80 words = 160 bytes
- WORK: contains Q, K, V, S (attention scores = softmax weights A)
  - Q: SEQ×D = 256 bytes
  - K: SEQ×D = 256 bytes
  - V: SEQ×D = 256 bytes
  - S/A: SEQ×SEQ = 64 words = 128 bytes (overwritten in-place by SFTMX)
  Total WORK: 3×256 + 128 = 896 bytes

### Backward pass (BKWRD.MAC):
Uses its own workspace:
- DL: V = 10 words = 20 bytes (per-position logit gradient)
- DY: SEQ×D = 256 bytes
- DA: SEQ×SEQ = 128 bytes (reused as dSc)
- DQQ: SEQ×D = 256 bytes
- DKK: SEQ×D = 256 bytes
- DVV: SEQ×D = 256 bytes
- DXX: SEQ×D = 256 bytes
- DTMP: D = 16 words = 32 bytes
Total backward workspace: ~1460 bytes

### Weight storage:
- Q16 accumulators (high + low): 1216 × 4 = 4864 bytes
- Q8 copies: 1216 × 2 = 2432 bytes
- Gradient Q15: 1216 × 2 = 2432 bytes

### Shared routines:
SFTMX is called in both forward (attention) and backward (logit
gradient computation). No state conflict — it's stateless.

## Section C — Memory Impact

### Code size estimates

| Component | Estimated bytes |
|-----------|----------------|
| EMBED | ~80 |
| PROJ | ~60 |
| ATTN | ~250 |
| AT.BPR (internal) | ~50 |
| BKWRD | ~500 |
| VSADD | ~80 |
| VCPY/VCLR | ~20 |
| UPDAT | ~100 |
| INITW/RAND | ~60 |
| Private storage (DP) | ~80 |
| **Total LAYER** | **~1280** |

### Current state
- Code region: $0600-$0B97 (1432 bytes, all routines + EXPTBL)
- BW_SCRATCH: $4420-$446F (80 bytes, SF_ block fills to +79)
- STACK_TOP: $4800 (912 bytes headroom from BW_END)

### Projected after LAYER
- Code end: ~$0B97 + 1280 = ~$1097
- This **crosses** STR_BASE ($1800)... actually it doesn't since
  $0B97 + $0500 = $1097. STR_BASE at $1800 has 1768 bytes margin.
- But the production binary (main.asm) also needs weight data starting
  at WEIGHT_BASE ($1A00). Code at $1097 leaves ~$0900 (2304 bytes)
  before weights. Adequate.

### New DP parameter block needs
LAYER needs extensive private storage for ATTN (25 words = 50 bytes)
plus EMBED (7 words), PROJ (8 words), BKWRD state (~10 words).

These **cannot** fit in BW_SCRATCH (only 6 bytes of reserved headroom
left at +74..+79). Options:
1. **Reuse MP_* block** for LAYER's per-call parameter setup, since
   LAYER calls MATOP routines which use MP_*. LAYER fills MP_* before
   each MATOP call — no conflict.
2. **Put ATTN/EMBED/PROJ private storage at harness scratch** ($1700
   region) for test, and in the production binary use dedicated
   addresses in the data region.
3. **Allocate a new LAYER scratch region** outside BW_SCRATCH. Since
   ATTN/EMBED/PROJ are called at the top level (not inside inner
   loops of other routines), their private storage can overlap with
   backward workspace addresses that aren't in use during forward pass.

Recommendation: Option 3 — overlay ATTN private storage (AT_*) onto
backward workspace addresses during forward pass, since forward and
backward never run concurrently. Specifically, use BWD_WORK ($3E00+)
region directly for AT_* storage, since the backward workspace
values are meaningless during forward pass and vice versa.

### Forward cache (activation storage)
The Stage 1 memory map already reserves:
- FWD_CACHE ($3800): 1.5KB for forward activations (X, Q, K, V, ATT,
  CTX, LOGITS, PROBS). This matches what ATTN/11 stores.
- BWD_WORK ($3E00): 2.0KB for backward workspace.

These regions were designed for exactly this purpose. No new allocation
needed — just use the existing FC_* and BW_* addresses from equates.inc.

## Section D — Primitive Gap Analysis

### Missing primitives (must be added)

**1. VSADD** — Vector scale-add: dst[k] += (scalar × src[k]) >> 8
- Source: BKWRD.MAC lines 463-499
- Two-stage clamp: ASHC >> 8 + clamp, then ADD with overflow clamp
- Used by: BKWRD step 2 (dV accumulation)
- Our decision: implement with single-clamp (matching our MATOP
  deviation) or match ATTN/11's two-stage? The MATOP precedent
  says single-clamp. But VSADD adds per-element, not per-row-dot.
- **Recommend: match ATTN/11 two-stage for VSADD** since the
  intermediate value IS the natural Q8 result of one multiply,
  and the addition is a separate logical step. This differs from
  MVADD where we accumulate across a whole row.
- Estimated size: ~60 bytes

**2. VCPY** — Vector copy: dst = src
- Source: VECOP.MAC lines 149-152
- Trivial: MOV loop. On 6309, could use TFM for speed.
- Estimated size: ~15 bytes (TFM version)

**3. VCLR** — Vector clear: vec = 0
- Source: VECOP.MAC lines 163-166
- Trivial: CLR loop. On 6309, ZEROFILL macro already exists.
- Estimated size: ~10 bytes

**4. ASH (multi-bit shift)** — Used in BKWRD for sqrt_shift and
   lr_shift. PDP-11 ASH shifts by a variable amount in one
   instruction. On 6309, we need a shift loop or unrolled shifts
   for known shift amounts.
- For SQRTSH=2 (fixed for D=16): two ASRD instructions.
- For LR shifts (1, 4, 6): unrolled or loop.
- Not a standalone primitive — inline in each use site.

### Primitives we have and LAYER uses:
- VTMUL ✓ (ATTN steps 1-3, 6; PROJ; BKWRD steps 4-5)
- MVMUL ✓ (BKWRD step 1: dY = MVMUL(Wout, dL))
- MVADD ✓ (BKWRD step 5: dX accumulation)
- OUTER ✓ (BKWRD step 1: dWout; step 5: dWq/dWk/dWv)
- VDOT ✓ (ATTN step 4: scores; BKWRD step 2: dA)
- SFTMX ✓ (ATTN step 5; BKWRD step 1: logit softmax)
- FXDIV ✓ (used by SFTMX internally)
- VADD ✓ (ATTN step 7: residual add Y+=X)

## Section E — 6309 Translation Notes

### EMBED
- Token ID × row_size: PDP-11 uses MUL (signed 16×16→32). On 6309,
  token IDs are 0-9 and row_size is 32 (D=16, ×2 bytes). Product
  fits in 16 bits. Use MULD or simple repeated-addition (ID×32 =
  ID<<5). Fastest: LDB tokenID / CLRA / 5× LSLD.
- Inner loop: MOV+ADD per element → LDD+ADDD+STD on 6309.

### PROJ
- PDP-11 self-modifying code patches VTMUL inline params. On 6309,
  fill MP_* parameter block and JSR VTMUL. Cleaner.

### ATTN
- **AT.BPR**: batch project subroutine. Calls VTMUL in a loop,
  advancing input/output pointers. On 6309: fill MP_* block per
  iteration, JSR VTMUL.
- **Score computation (step 4)**: Calls VDOT with R0/R1/R2 params.
  On 6309: X/Y/B params per our VDOT convention.
- **ASH AT.SHF, R0**: variable-count arithmetic shift. For D=16,
  SQRTSH=2, so this is always >>2. On 6309: two ASRD instructions.
  If we want to support variable D, need a shift loop. For Stage 1
  (D=16 fixed), hardcode two ASRD.
- **Residual add (step 7)**: SOB loop with ADD. On 6309: stack
  counter + LDD/ADDD/STD loop (same pattern as VADD).

### BKWRD
- **Softmax backward (step 3)**: The most complex translation.
  PDP-11 does: MUL + ASHC #-8 + ASH #-SQRTSH per element, with
  overflow clamping on subtraction. On 6309: MULD + NORMQ15 +
  two ASRD (for SQRTSH=2).
- **Column extraction (step 4, dK)**: Copies column j from a
  row-major matrix to a temp vector. Stride = SEQ×2 bytes.
  On 6309: indexed addressing with stride increment.
- **Element-wise saturating add (step 6)**: Embedding gradient
  accumulation with BVC/clamp pattern. Matches our single-clamp
  add pattern but at the scalar level.

### UPDAT
- **32-bit subtraction**: PDP-11 does ADD of negated value with
  ADC for carry + sign extension. On 6309: SUBD + SBCB/SBCA or
  ADD-negated same pattern. The sign-extension on the PDP-11
  (TST + BPL + DEC) maps to TSTA + BPL + LDA #$FF + STA.
- **ASH for lr_shift**: Variable shift. For LR.ATN=1: one ASRD.
  For LR.EMB=4: four ASRD. For LR.OUT=6: six ASRD. Unroll or loop.

## Section F — Fixed-Point Hazards

### SFTMX max count
SFTMX is called with:
- SEQ=8 (attention scores): 16-bit sum OK (max 8×256 = 2048)
- VOCAB=10 (logit softmax in BKWRD): 16-bit sum OK (max 10×256 = 2560)

No calls exceed our documented max count of 127. **No hazard.**

### MULSCR dual-use
ATTN calls VTMUL (which uses MULSCR via NORMQ15 inside VSCL loop),
VDOT (uses MULSCR for 32-bit accumulation), and SFTMX (calls FXDIV
which uses MULSCR for dividend construction). These are sequential,
not nested — each completes before the next starts. **No conflict.**

BKWRD's softmax backward step does: MUL + ASHC per element. If we
use MULD + NORMQ15, that uses MULSCR. But this is also sequential.
**No conflict** as long as we don't call MATOP routines from inside
a MULSCR-dependent loop.

### Double-clamp in BKWRD
BKWRD step 3 (softmax backward) has: SUB + overflow clamp, then
MUL + ASHC clamp, then implicit store. This is a three-stage
pipeline of clamped operations. We should match ATTN/11's two-stage
behavior here rather than inventing a single-clamp variant —
the operations are logically distinct (subtraction, multiplication,
shift) and the intermediate values have semantic meaning.

BKWRD step 6 (embedding gradient accumulation) uses ADD with
BVC/clamp — saturating addition. This is the standard two-stage
pattern and should be matched directly.

### EXPTBL lookup tables
EXPTBL is already inline. No additional tables needed for LAYER.
LOGTBL (for loss computation in TRAIN) is deferred.

## Section G — Implementation Order

1. **VCPY + VCLR** — Trivial. Add to vecop.asm. Quick test.
2. **VSADD** — New primitive. Add to vecop.asm or a new file.
   Test with dedicated vectors.
3. **EMBED** — Forward pass entry point. Depends on no new
   primitives. Test with known token→embedding mapping.
4. **PROJ** — Forward pass exit. Calls VTMUL. Test with known
   Y and Wout.
5. **ATTN** — The big one. Calls VTMUL, VDOT, SFTMX, VADD.
   Test with tiny matrices (2×2 or 3×2) where hand-computation
   is feasible. This is the hardest to test because it's a
   composite of 7 steps.
6. **BKWRD** — Backward pass. Depends on everything. Test by
   running forward+backward on a tiny model and checking gradient
   shapes/signs rather than exact values (exact matching requires
   the full training loop).
7. **UPDAT + INITW + RAND** — SGD update. Test by initializing
   weights, running one forward+backward+update cycle, and verifying
   weight change direction.

### Validation strategy
- Steps 1-4: per-routine test harness with reference vectors
  (matching FXMATH/VECOP/MATOP pattern).
- Step 5 (ATTN): integration test. Compute attention on a small
  example (SEQ=2, D=4) where every intermediate tensor can be
  hand-traced. The Python prototype (prototype.shf) is the reference.
- Steps 6-7: system test. Full training loop on the real model
  parameters. This is essentially TRAIN and goes beyond LAYER scope.

Alternative order for faster validation:
- Implement EMBED+PROJ+ATTN as "LAYER forward" and test end-to-end
  before implementing backward. This gives us a working forward pass
  that we can validate against the prototype, reducing risk before
  tackling the complex backward pass.

## Section H — Open Questions

**Q1: Should LAYER include BKWRD and UPDAT, or just forward?**
ATTN/11 separates them into nn11/LAYER.MAC (forward primitives) and
model/{BKWRD,UPDAT,FORWRD}.MAC (model-specific orchestration). Our
build order says "LAYER" covers forward. Backward and training loop
are "TRAIN" (step 7 in the original plan). **Recommend: implement
LAYER as forward-only (EMBED+PROJ+ATTN), defer BKWRD+UPDAT to TRAIN.**

**Q2: ATTN's workspace allocation.**
ATTN partitions WORK into Q|K|V|S contiguously. Our memory map has
FWD_CACHE ($3800, 1.5KB) for this purpose. Is 1.5KB enough?
- Q: SEQ×D×2 = 8×16×2 = 256 bytes
- K: 256 bytes
- V: 256 bytes
- S: SEQ×SEQ×2 = 8×8×2 = 128 bytes
- Total: 896 bytes. FWD_CACHE is 1.5KB = 1536 bytes. **Fits with
  640 bytes to spare** (used for FC_X, FC_CTX, FC_LOGITS, FC_PROBS
  per equates.inc).

**Q3: SFTMX in-place vs. separate output.**
Our SFTMX writes to SF_OUT (separate output pointer). ATTN/11's
SFTMX overwrites input in-place. For ATTN step 5, scores S are
overwritten with attention weights A — this is intentional and
the original S values are not needed after softmax. Our SFTMX
can handle this by setting SF_OUT = SF_VEC.

**Q4: Does ATTN use VADD for the residual?**
No — ATTN/11 uses a raw ADD loop (SOB). We can use our VADD
primitive for clarity, or write a direct loop. VADD matches.

**Q5: BKWRD's VSADD — is it the same as VSCL + VADD?**
No. VSADD is: dst[k] += (scalar × src[k]) >> 8. It's a fused
scale-and-accumulate that can't be decomposed into VSCL+VADD
without an intermediate buffer. It must be a dedicated primitive.

## Section I — Risk Assessment

| Routine | Risk | Justification |
|---------|------|---------------|
| VCPY | Low | TFM or trivial loop |
| VCLR | Low | ZEROFILL macro or trivial loop |
| VSADD | Medium | New primitive, saturating add, two-stage clamp |
| EMBED | Low | Copy+add loop, no primitives |
| PROJ | Low | VTMUL loop with pointer advance |
| ATTN | **High** | 7-step composite, 25 words private state, nested loops, workspace partitioning, multiple primitive calls |
| BKWRD | **High** | Most complex code in the system, 6 steps, column extraction, softmax backward, saturating accumulation |
| UPDAT | Medium | 32-bit arithmetic, variable shift, sign extension |
| INITW/RAND | Low | LCG PRNG, shift conversion |

### Critical path
ATTN is the gating risk. If ATTN works correctly on a small
example, the forward pass is validated and backward pass becomes
a matter of applying known patterns (MATOP primitives) in the
right order. If ATTN has bugs, they're hard to diagnose because
the 7-step pipeline means errors compound.

Recommendation: invest heavily in ATTN testing with a small model
(SEQ=2, D=4) where every intermediate value can be computed by
hand or via the Python prototype.
