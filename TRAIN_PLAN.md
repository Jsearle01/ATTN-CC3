# TRAIN_PLAN.md — Training phase implementation plan

Based on reading all ATTN/11 source (TRAIN.MAC, FORWRD.MAC, BKWRD.MAC,
UPDAT.MAC, all nn11/*.MAC) and prototype.shf (lines 132-292).

---

## Source file survey

### ATTN/11 (PDP-11 assembly)

| File | Lines | Role |
|------|-------|------|
| TRAIN.MAC | 541 | Training loop orchestration, loss, reporting, test, data, strings |
| model/FORWRD.MAC | 73 | Forward pass (EMBED→ATTN→PROJ), Q16→Q8 weight conversion |
| model/BKWRD.MAC | 501 | Backward pass (6 steps), Q15 gradients, includes VSADD inline |
| model/UPDAT.MAC | 177 | SGD update (Q16 accumulators), weight init (RAND+LCG), ZEROG |
| IO.MAC | 153 | Console I/O: PUTC, PUTS, PUTDEC, PUTOCT, PUTQ8, PUTVEC |
| nn11/LAYER.MAC | 309 | EMBED, ATTN, PROJ (forward-only composites) |
| nn11/MATOP.MAC | 260 | MVMUL, MVADD, VTMUL, OUTER |
| nn11/VECOP.MAC | 168 | VDOT, VSCL, VADD, VCPY, VCLR, VMAX |
| nn11/ACTFN.MAC | 165 | SFTMX (softmax) |
| nn11/FXMATH.MAC | 88 | MUL8Q15, MUL15Q15, FXDIV |

All training code exists in ATTN/11. This is NOT a prototype-only
implementation. **Risk is significantly lower than if we were
implementing from scratch.**

### prototype.shf

Lines 132-201: `backward` function with Q15 gradient arithmetic.
Lines 202-210: `sgd-update` with Q16 accumulators.
Lines 212-217: `cross-entropy-float` (monitoring loss, not used in backward).
Lines 219-291: Training loop (`train` function).

---

## Q1 — Loss function

**Cross-entropy loss** on logits, computed in CLOSS (TRAIN.MAC:170-215).

Algorithm:
1. For each position i: copy logits[i] to temp, apply SFTMX → probabilities
2. Look up `target[i]` probability → p (Q8, range [0, 256])
3. If p ≥ 256 (1.0): loss contribution = 0
4. Otherwise: loss contribution = LOGTBL[p] (Q12 encoded `-ln(p/256)*4096`)
5. Accumulate all positions in a 32-bit sum (Q12)
6. Average: 32-bit sum >>3 (divide by SEQ=8)
7. Return Q12 loss in R0

LOGTBL is a 257-entry Q12 table of `-ln(x/256)*4096` for x=0..256.
LOGTBL[0] = 22713 (large penalty for zero probability).

**Loss is for monitoring only** — it's NOT used in the backward pass.
The backward pass computes `dL/d(logits) = softmax(logits) - one_hot(target)`
directly, without computing the scalar loss value.

---

## Q2 — Gradient format (Q8/Q15/Q16 mixed precision)

Confirmed from TRAIN.MAC header and BKWRD.MAC header:

| Stage | Format | Precision |
|-------|--------|-----------|
| Forward activations | Q8 (Q7.8) | 16-bit signed, 1/256 resolution |
| Gradients | Q15 (Q0.15) | 16-bit signed, 1/32768 resolution |
| Weight accumulators | Q16 (Q0.16) | 32-bit signed (hi:lo word pair), 1/65536 resolution |

**Mixed-precision multiplication:**
- Forward: Q8 × Q8 = Q16, >>8 → Q8 (via VTMUL per-product or VDOT accumulate)
- Backward: Q8 × Q15 = Q23, >>8 → Q15 (same shift, different interpretation)
- Both use MUL + ASHC #-8 on PDP-11, MULD + >>8 extraction on 6309

**Key insight:** The backward pass reuses the SAME primitives (VTMUL,
MVMUL, MVADD, OUTER, VDOT, VSADD) as the forward pass. The Q8×Q15→Q15
arithmetic works identically to Q8×Q8→Q8 in the hardware — the shift
amount (>>8) is the same. Only the interpretation of the result differs
(Q15 gradient vs Q8 activation).

**dLogits conversion:** After computing `softmax(logits) - one_hot(target)`
(which is in Q8 range [-256, 255]), the result is shifted <<7 to Q15.
This <<7 is the only new operation not in the forward path.

---

## Q3 — Backward pass structure (from BKWRD.MAC)

Six steps, reversing the forward pass:

### Step 1: dLogits, dWout, dY (BKWRD.MAC:24-98)
For each position i:
```
dL[i] = softmax(logits[i]) - one_hot(target[i])   ; Q8
dL[i] <<= 7                                        ; Q8 → Q15
dWout += OUTER(Y[i], dL[i])    ; Q8 × Q15 → Q23, >>8 → Q15
dY[i]  = MVMUL(Wout, dL[i])   ; Q8 × Q15 → Q23, >>8 → Q15
```

### Step 2: Backward O=A@V → dA, dV (BKWRD.MAC:100-167)
dO = dY (residual: no copy needed, dY IS dO).
For each (i, j):
```
dA[i][j] = VDOT(V[j], dY[i])          ; Q8 × Q15 → Q15
dV[j]   += VSADD(A[i][j], dY[i])      ; A[i][j] (Q8 scalar) × dY[i] (Q15 vec)
```

### Step 3: Backward softmax → dSc (BKWRD.MAC:169-216)
For each row i of attention weights:
```
dot_ad = VDOT(A[i], dA[i])                     ; Q8 × Q15 → Q15
dSc[i][j] = A[i][j] * (dA[i][j] - dot_ad) >> 8  ; Q8 × Q15 → Q23, >>8 → Q15
dSc[i][j] >>= sqrt_shift                         ; undo the forward scaling
```
The subtraction `dA[i][j] - dot_ad` includes overflow clamping (BVC/BPL/BMI).

### Step 4: Backward Q·K^T → dQ, dK (BKWRD.MAC:218-276)
```
dQ[i] = VTMUL(K, dSc[i], SEQ, DIM)     ; K^T · dSc_row
dK[j] = VTMUL(Q, dSc_col_j, SEQ, DIM)  ; Q^T · dSc_column
```
dK requires extracting columns of dSc into a temp vector (DTMP) before
VTMUL, because dSc is stored row-major but we need column vectors.

### Step 5: Backward projections → dX (BKWRD.MAC:278-393)
```
dX = copy(dY)                           ; residual pathway
For each position i:
  dX[i] += MVADD(Wq, dQ[i])            ; Wq · dQ (not transpose!)
  dWq   += OUTER(X[i], dQ[i])          ; weight gradient
  dX[i] += MVADD(Wk, dK[i])
  dWk   += OUTER(X[i], dK[i])
  dX[i] += MVADD(Wv, dV[i])
  dWv   += OUTER(X[i], dV[i])
```

### Step 6: Backward embedding (BKWRD.MAC:395-448)
```
For each position i:
  d_tok[token[i]] += dX[i]     ; scatter-add to token embedding gradient
  d_pos[i]        += dX[i]     ; position embedding gradient
```
Both additions include overflow clamping (BVC/TST/BMI/BPL pattern).

---

## Q4 — Primitive inventory: what we have vs what we need

### Existing primitives (already implemented + tested)

| Primitive | Location | Used in backward? |
|-----------|----------|-------------------|
| VTMUL | matop.asm | ✅ Step 4 (dQ, dK) |
| MVMUL | matop.asm | ✅ Step 1 (dY) — but see note |
| MVADD | matop.asm | ✅ Step 5 (dX accumulation) |
| OUTER | matop.asm | ✅ Steps 1, 5 (weight gradients) |
| VDOT | vecop.asm | ✅ Steps 2, 3 (dot products) |
| VSADD | vecop.asm | ✅ Step 2 (dV accumulation) |
| VCPY | vecop.asm | ✅ Step 5 (dY→dX copy) |
| VCLR | vecop.asm | ✅ Gradient zeroing |
| SFTMX | actfn.asm | ✅ Step 1 (softmax of logits for dL) |
| VMAX | vecop.asm | ✅ Accuracy counting (argmax) |
| FXDIV | fxmath.asm | ✅ Inside SFTMX |

**Note on MVMUL:** BKWRD.MAC Step 1 uses MVMUL (mat × vin → vout)
for `dY[i] = Wout · dL[i]`. Our MVMUL computes `vout[i] = sum_j(mat[i][j] * vin[j])`
with 32-bit row accumulation and single end-of-row clamp. This is the
correct operation for the backward pass through PROJ. Already implemented.

### New operations needed

| Operation | Where | Complexity | Notes |
|-----------|-------|------------|-------|
| **dL <<= 7** | Step 1 | Trivial | Per-element ASH #7 on Q8 vector → Q15. ~5 lines of assembly. |
| **Clamped add (scatter)** | Step 6 | Simple | `d_tok[tok[i]][d] += dX[i][d]` with BVC/clamp. ~15 lines inline. |
| **Clamped subtract** | Step 3 | Simple | `dA[i][j] - dot_ad` with BVC/clamp. ~8 lines inline. |
| **Q16 weight update** | UPDAT | Moderate | 32-bit subtract: `w_hi:w_lo -= sign_extend(grad >> (shift-1))`. ~20 lines. |
| **Q16→Q8 conversion** | CVT16 | Simple | For each weight: `ASHC #-8` (PDP-11) = extract middle word of hi:lo pair. ~10 lines. |
| **Weight initialization** | INITW | Simple | LCG PRNG + Q8→Q16 conversion. ~15 lines. |
| **Cross-entropy loss** | CLOSS | Moderate | SFTMX on logits, LOGTBL lookup, 32-bit accumulation. ~40 lines. Used for monitoring only. |
| **Sample generation** | GENSM | Simple | PRNG + mod-10 for random digits, reverse for targets. ~20 lines. |
| **Column extraction** | Step 4 | Trivial | Extract column j from row-major matrix into temp. ~6 lines inline in BKWRD. |

**No new MATOP/VECOP primitives needed.** All backward operations map
to existing primitives (VTMUL, MVMUL, MVADD, OUTER, VDOT, VSADD).
The "new" operations are inline arithmetic (shifts, clamped adds) and
training infrastructure (loss, PRNG, weight format conversion).

---

## Q5 — Weight update mechanism

**Pure SGD with shift-encoded learning rates.** No momentum, no Adam.

From UPDAT.MAC:
```
w_q16 -= grad_q15 >> (lr_shift - 1)
```

Learning rate shifts (from TRAIN.MAC):
```
LR.EMB = 4   → tok_emb, pos_emb: grad >> 3  (lr ≈ 0.08 * 2/256)
LR.ATN = 1   → Wq, Wk, Wv:      grad >> 0  (lr ≈ 0.5 * 2/256)
LR.OUT = 6   → Wout:             grad >> 5  (lr ≈ 0.01 * 2/256)
```

Implementation in UPDAT.MAC:UP.DO:
1. Read gradient (Q15), zero the gradient slot immediately
2. ASH by -(lr_shift-1) → delta (Q15)
3. Negate delta (avoids PDP-11 SUB carry issues)
4. ADD -delta to w_lo (16-bit), ADC carry to w_hi
5. If -delta was negative: DEC w_hi (sign extension)

Weight storage: **split hi/lo arrays** (TKEH/TKEL, WQH/WQL, etc.).
Each weight is a 32-bit Q16 value stored as two separate 16-bit words
in parallel arrays. This avoids the need for 32-bit memory access
but complicates pointer management.

On 6309, we already have the equates for this layout (WEIGHT_BASE
at $1A00, split hi/lo in equates.inc). We can use LDQ/STQ for
32-bit access if contiguous, or maintain the split layout for
PDP-11 compatibility.

---

## Q6 — Memory budget for training

### Forward cache (must survive for backward pass)

| Buffer | Size (SEQ=8, DIM=16, VOC=10) |
|--------|-----|
| XX (EMBED output X) | 256 B |
| WORK (Q+K+V+S/A cached by ATTN) | 896 B |
| YY (ATTN output Y) | 256 B |
| LOGITS | 160 B |
| **Forward cache total** | **1568 B** |

### Gradient buffers (Q15, same shape as what they're gradients of)

| Buffer | Size |
|--------|------|
| DL (dLogits temp, one position) | 20 B |
| DY [SEQ][DIM] | 256 B |
| DA [SEQ][SEQ] (reused as dSc) | 128 B |
| DQQ [SEQ][DIM] | 256 B |
| DKK [SEQ][DIM] | 256 B |
| DVV [SEQ][DIM] | 256 B |
| DXX [SEQ][DIM] | 256 B |
| DTMP [DIM] (temp for column extraction) | 32 B |
| **Gradient buffer total** | **1460 B** |

### Weight gradient accumulators (Q15)

| Buffer | Size |
|--------|------|
| DTKE [VOC][DIM] | 320 B |
| DPSE [SEQ][DIM] | 256 B |
| DWQ [DIM][DIM] | 512 B |
| DWK [DIM][DIM] | 512 B |
| DWV [DIM][DIM] | 512 B |
| DWOUT [DIM][VOC] | 320 B |
| **Weight gradient total** | **2432 B** |

### Q16 weight accumulators (32-bit per weight = 2× weight size)

Already allocated in equates.inc: WEIGHT_BASE at $1A00, ~4.8 KB.
This is the hi/lo split storage for all weights.

### Q8 weight copies (for forward/backward computation)

Same shape as weight gradients: ~2432 B. These are the Q8 versions
of the Q16 accumulators, regenerated each step by CVT16.

### Total training memory

| Category | Bytes |
|----------|-------|
| Forward cache | 1568 |
| Gradient buffers | 1460 |
| Weight gradient accumulators | 2432 |
| Q16 weight accumulators | 4864 (already at WEIGHT_BASE) |
| Q8 weight copies | 2432 |
| Training state (tokens, targets, misc) | ~100 |
| LOGTBL | 514 |
| Stack | 256+ |
| **Total** | **~13,600 B** |

### CoCo3 RAM budget

Available RAM: $0600 (code start) to $4800 (STACK_TOP) = ~16.5 KB.
Code occupies ~$0600-$1200 (~3 KB for all primitives + composites).
Remaining: ~13.5 KB for data.

**Tight but feasible.** The equates.inc memory map already allocates
WEIGHT_BASE ($1A00, 4.8 KB), GRAD_BASE ($2C00, 2.4 KB),
FWD_CACHE ($3800, 1.5 KB), BWD_WORK ($3E00, 2 KB). Total: ~11 KB.
With Q8 weight copies at ~2.4 KB, we need ~13.4 KB. The gap between
BWD_WORK end ($4420) and STACK_TOP ($4800) is 960 B for stack.

Some overlap is possible: Q8 weight copies could be generated in-place
by overwriting a shared buffer, but BKWRD reads both Q8 weights AND
cached forward values, so they can't share storage during backward.

**Verdict: fits, but with <1 KB stack headroom.** Monitor stack depth
during testing. The deepest call chain is BKWRD → VTMUL → inner loops
with ~20 bytes of stack per call level. Should be fine.

---

## Q7 — Training loop structure

From TRAIN.MAC:

- **Batch size: 1** (one sequence per step)
- **Steps: 350** (configurable via NSTEP; ATTN/11 uses 350, prototype uses 500)
- **Reporting: every 50 steps** (RPRT=50)
- **No mini-batching** — single forward+backward+update per step
- **Task: digit reversal** — input 8 random digits [0-9], target is reversed sequence
- **Sample generation:** LCG PRNG, mod-10 for digits, reverse for targets
- **Convergence:** fixed iteration count, no early stopping
- **Weight init:** random Q8 in [-128, 127], converted to Q16 via <<8
- **Per-step sequence:** GENSM → CVT16 → FORWRD → BKWRD → UPDAT → COUNT

The training loop also maintains running hit/total counters for accuracy
reporting, reset every RPRT steps.

Final test: 10 samples (NTEST=10), forward-only, print input→prediction
with OK/FAIL tags plus aggregate accuracy.

---

## Primitive gap analysis

### No new MATOP/VECOP primitives needed

Every backward operation maps to an existing primitive:

| Backward operation | Primitive | Already tested? |
|-------------------|-----------|-----------------|
| dY = Wout · dL | MVMUL | ✅ 21/21 MATOP |
| dWout += Y^T · dL | OUTER | ✅ 21/21 MATOP |
| dX += Wq · dQ | MVADD | ✅ 21/21 MATOP |
| dWq += X^T · dQ | OUTER | ✅ |
| dQ = K^T · dSc_row | VTMUL | ✅ 5/5 PROJ, 5/5 ATTN |
| dK = Q^T · dSc_col | VTMUL | ✅ |
| dA[i][j] = V[j]·dY[i] | VDOT | ✅ 44/44 VECOP |
| dV += A[i][j]·dY[i] | VSADD | ✅ 10/10 VSADD |
| dot(A,dA) | VDOT | ✅ |

### New code needed (all inline or simple routines)

1. **BKWRD composite** (~300 lines, matching BKWRD.MAC's 6-step structure)
   - Inline: dL shift <<7, clamped subtract for softmax backward,
     column extraction for dK
   - Calls: SFTMX, VCLR, VCPY, VDOT, VSADD, VTMUL, MVMUL, MVADD, OUTER

2. **UPDAT routine** (~80 lines)
   - CVT16: Q16 hi:lo → Q8 (LDQ/ASHC-equivalent or byte extraction)
   - SGD update: 32-bit subtract with sign extension
   - ZEROG: just VCLR calls
   - INITW: LCG PRNG + Q8→Q16 init

3. **CLOSS routine** (~50 lines)
   - SFTMX on logits, LOGTBL lookup, 32-bit accumulation, >>3 average
   - LOGTBL: 257-entry Q12 table (514 bytes, matches TRAIN.MAC data)

4. **Training infrastructure** (~80 lines)
   - GENSM (sample generation)
   - COUNT (accuracy tracking)
   - REPORT (screen output)
   - Main training loop

---

## Risk assessment: **Medium**

### Low-risk elements
- All backward operations use tested primitives (no new MATOP/VECOP)
- ATTN/11 source provides exact reference for every step
- Mixed Q8/Q15 arithmetic uses same shift (>>8) as forward path
- Simple SGD (no momentum/Adam state to track)
- Forward pass already validated end-to-end

### Medium-risk elements
- **BKWRD is 501 lines** of PDP-11 assembly — largest single routine to port.
  Careful register-to-stack translation needed (PDP-11 uses R0-R5 freely;
  6309 has A/B/D/W/X/Y/U with different constraints).
- **Softmax backward** (Step 3) has inline clamped arithmetic that's
  easy to get wrong (overflow detection, sign-dependent clamping).
- **Q16 weight accumulators** use 32-bit split storage; pointer
  management for hi/lo arrays is error-prone.
- **Memory tightness** (~960 B stack headroom). Deep call chains in
  BKWRD → VTMUL → inner loops could overflow if not careful.
- **Testing the backward pass** is harder than forward: gradients
  require a reference generator that chains forward+backward, and
  small numerical differences accumulate through the chain.

### Not a risk
- MAME callback budget: training loop runs to completion in the
  emulated machine (no Lua callback needed during training itself).
  Test harnesses still use the PC-stall template for verification,
  but the main training binary would run autonomously until HALT.

---

## Implementation order (phased)

### Phase 1: CLOSS — Loss computation
- Port CLOSS and LOGTBL from TRAIN.MAC
- Generate `gen_logtbl.py` (or use existing `gen_logtbl.py` if present)
- Test: feed known logits + targets, compare Q12 loss against reference
- Depends on: SFTMX (already validated)

### Phase 2: BKWRD Step 1 — dLogits, dWout, dY
- Implement the <<7 shift and SFTMX-based dL computation
- Test OUTER for weight gradients, MVMUL for input gradients
- Single-step test: given known forward cache + targets, verify dWout and dY
- Depends on: SFTMX, MVMUL, OUTER (all validated)

### Phase 3: BKWRD Steps 2-3 — dA, dV, dSc (softmax backward)
- Most complex step: VDOT for dA, VSADD for dV accumulation
- Softmax backward with clamped subtract + per-element multiply
- Test: given known A, dY → verify dA, dV, dSc
- Depends on: VDOT, VSADD (validated)

### Phase 4: BKWRD Steps 4-6 — dQ, dK, dX, embedding grads
- VTMUL for dQ/dK, MVADD+OUTER for dX/dW accumulation
- Column extraction for dK (simple inline loop)
- Embedding scatter-add with clamping
- Test: full backward pass (all steps chained)
- Depends on: VTMUL, MVADD, OUTER, VCPY (all validated)

### Phase 5: UPDAT — Weight update + initialization
- CVT16: Q16→Q8 conversion
- SGD update: 32-bit arithmetic
- INITW: LCG PRNG
- Test: known gradient → apply update → verify new Q16 values → verify Q8 conversion

### Phase 6: Training loop integration
- GENSM, COUNT, REPORT, main loop
- Screen output formatting
- End-to-end: run N training steps, verify loss decreases and accuracy increases
- Final validation: compare against ATTN/11's expected training trajectory

---

## Key architectural decisions for next session

1. **Weight storage layout:** ATTN/11 uses split hi/lo arrays. 6309
   has LDQ/STQ for contiguous 32-bit access. Do we maintain split
   layout (simpler port) or use contiguous (better 6309 fit)?

2. **BKWRD parameter passing:** ATTN/11 uses JSR R5 with inline
   parameter blocks. Our forward pass uses DP-block parameters
   (MP_*, AT_*, etc. in BW_SCRATCH). Extend this pattern to
   backward-specific parameters (BK_* in BW_SCRATCH)?

3. **LOGTBL placement:** 514 bytes. Current tables.asm holds EXPTBL
   (512 B). LOGTBL would double the table budget. Verify it fits
   in the code/tables region ($0200-$17FF).

4. **Gradient buffer overlap with production memory map:** The gradient
   buffers (GRAD_BASE at $2C00, 2.4 KB in equates.inc) may need
   expansion for the backward workspace buffers (DY, DA, DQ, DK,
   DV, DX, DTMP = 1460 B). Current BWD_WORK at $3E00 has 2 KB but
   includes MP_*/SF_*/EM_*/PR_*/AT_* parameter blocks.

5. **Test strategy for backward pass:** Unit-test each BKWRD step
   independently (reference generator produces per-step intermediates),
   or test only the full chain (forward+backward)? Per-step is more
   diagnostic but requires more generator work.
