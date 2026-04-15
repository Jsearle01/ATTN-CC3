# ATTN Implementation Plan

Exploration report for ATTN, the self-attention composite.
Based on reading LAYER.MAC (lines 134-280), FORWRD.MAC (call site
lines 14-17), prototype.shf (lines 106-121), and cross-referencing
the 6309 primitives we have (VDOT, VTMUL, SFTMX) against the seven
composition steps.

ATTN is the highest-risk routine in the LAYER work: 9 caller
parameters, 25 words of private state, nested SEQ×SEQ inner loop,
variable-count scaling shift, in-place softmax on interior buffer
rows, and a per-position weighted-aggregation VTMUL call that
re-uses the same primitive with different dimensions than Steps
1–3. **No new primitives are required** — the existing VDOT,
VTMUL, SFTMX, and an inline ASR-by-N loop cover the 7 steps — but
the composition bookkeeping and the workspace layout make this
the most delicate layer routine to stand up.

## Section A — Parameter block (Q1)

ATTN takes **9 caller-filled fields** ([LAYER.MAC:143-149](.claude/ATTN-11-main/nn11/LAYER.MAC#L143-L149)):

| Field | Width | Purpose |
|-------|-------|---------|
| xin   | ptr | input X, `[seq_len][d_model]` Q8 (from EMBED) |
| wq    | ptr | Wq weight matrix, `[d_model][d_model]` Q8 |
| wk    | ptr | Wk weight matrix, `[d_model][d_model]` Q8 |
| wv    | ptr | Wv weight matrix, `[d_model][d_model]` Q8 |
| yout  | ptr | output Y, `[seq_len][d_model]` Q8 (written in Step 6; Step 7 adds X in-place) |
| work  | ptr | scratch workspace ≥ `3*seq*d + seq*seq` words |
| seq_len | word | 8 for Stage 1 |
| d_model | word | 16 for Stage 1 |
| sqrtsh | word | ASR count = `log2(sqrt(d_model))` = **2** for d=16 |

FORWRD.MAC call site ([FORWRD.MAC:14-17](.claude/ATTN-11-main/model/FORWRD.MAC#L14-L17)):
```
JSR     R5, ATTN
.WORD   XX, WQQ8, WKQ8, WVQ8, YY, WORK
.WORD   SEQ.LN, D.MODL, SQRTSH
```

So ATTN's inline param list is 50 % bigger than EMBED's (6) or
PROJ's (6). On 6309 our calling convention replaces ATTN/11's
inline-words-after-JSR pattern with a DP parameter block set
before entry — no JSR R5 stream reads.

## Section B — Workspace layout (Q2)

ATTN writes four intermediate tensors into the caller-provided
`work` buffer. The region is partitioned at entry
([LAYER.MAC:171-181](.claude/ATTN-11-main/nn11/LAYER.MAC#L171-L181)):

```
AT.QQ = work                        ; Q[seq][dim], SEQ*DIM words
AT.KK = AT.QQ + SEQ*DIM*2           ; K[seq][dim]
AT.VV = AT.KK + SEQ*DIM*2           ; V[seq][dim]
AT.SS = AT.VV + SEQ*DIM*2           ; S[seq][seq], SEQ*SEQ words
```

For Stage 1 (SEQ=8, DIM=16):

| Buffer | Size (words) | Size (bytes) |
|--------|--------------|--------------|
| Q      | 128          | 256          |
| K      | 128          | 256          |
| V      | 128          | 256          |
| S      | 64           | 128          |
| **Total** | **448**   | **896**      |

Notes:
- No separate `A` (softmax output) or `O` buffer: `S` is softmax'd
  in-place in Step 5; Step 6 writes directly to the caller's `yout`.
- The prototype names `A` and `O` conceptually, but the assembly
  reuses `S` for `A` and `Y` for `O`. Our port mirrors the assembly.

**Memory-map impact.** 896 bytes does not fit in BW_SCRATCH (and
shouldn't — BW_SCRATCH is for DP parameter blocks, not tensors).
It also conflicts with the current FWD_CACHE allocation if the
cache continues to size fields as if Q8 were 1 byte per element
(see Q2 flag below). For Stage 1, the **test harness allocates
its own 896-byte WRK buffer in the test-buffer region** (around
`$1B00` — well above OUT_BUF). Production integration against
FORWRD must resolve the FWD_CACHE sizing question.

**FWD_CACHE byte/word discrepancy — flag.** `equates.inc` defines
`Q8SZ EQU 1` and sizes FC_X, FC_Q, FC_K, FC_V as `DSEQ` (= 128)
bytes each. But all forward-path code loads/stores 16-bit words
(`LDD`, `ADDD`). At 2 bytes per element, SEQ*D words needs 256
bytes, not 128 — so the current FWD_CACHE layout is half the size
it should be. `FC_ATT` at SEQ*SEQ = 64 bytes has the same issue if
scores are 16-bit (they are — the VDOT result is Q8 in a 16-bit
word). This mismatch predates ATTN and is not fixed here; the
ATTN test harness sidesteps it by allocating its own workspace.

## Section C — Private state, 25 words (Q3)

Enumerated from [LAYER.MAC:282-307](.claude/ATTN-11-main/nn11/LAYER.MAC#L282-L307):

**Caller-filled (9 words):** AT.XIN, AT.WQ, AT.WK, AT.WV, AT.YOT,
AT.WRK, AT.SEQ, AT.DIM, AT.SHF.

**Derived at entry (2 words):**
- AT.RSZ = DIM*2 (row size in bytes, reused everywhere)
- AT.SSZ = SEQ*2 (seq-element stride in bytes, used Steps 4/6)

**Workspace base pointers (4 words):** AT.QQ, AT.KK, AT.VV, AT.SS —
computed at entry from `work`.

**Loop-state cursors (5 words):** AT.QI (advancing Q row ptr in
Steps 4/6; wait — actually Step 4 only), AT.KJ (K row ptr, reset
per outer), AT.SI (S write/read ptr, reused Steps 4/5/6), AT.YI
(Y output ptr, Step 6 only), plus one free slot. Actually the
assembly lists: QI, KJ, SI, YI, OC, IC = 6 cursor/counter words.

**Loop counters (2 words):** AT.OC (outer), AT.IC (inner).

**BPR helper state (4 words):** AT.BW, AT.BI, AT.BO, AT.BC —
used only by the AT.BPR batch-projection subroutine that Steps 1–3
call. These live separate from QI/KJ/SI because BPR runs inside
Steps 1–3 but outer loop state isn't active yet (Steps 4/6 reuse
QI/KJ/SI later).

Total: 9 + 2 + 4 + 6 + 4 = **25 words = 50 bytes**.

Plus the **self-modifying-code slots** (ATTN/11-only; obsolete on
6309): AT.V1-V5 (5 words) inside AT.BPR's VTMUL call, AT.T1-T5
(5 words) in Step 6's VTMUL call. On 6309 these collapse into
`STD MP_*` writes before each `JSR VTMUL` — no inline slots.

**6309 collapsing opportunity.** AT.BW/BI/BO/BC can time-share
storage with AT.QI/KJ/SI/IC because BPR runs only during Steps
1–3 (before the Step 4 outer loop begins). Saves 8 bytes. Will
mark aliased in equates.inc.

## Section D — Scaling shift AT.SHF (Q4)

**Shift count.** For d_model=16, AT.SHF = 2 (right-shift by 2
approximates division by sqrt(16)=4). Caller-provided, not
hard-coded in ATTN.

**Where applied.** Inside Step 4's inner loop, per score,
immediately after VDOT
([LAYER.MAC:207-214](.claude/ATTN-11-main/nn11/LAYER.MAC#L207-L214)):

```
AT.S2:  MOV     AT.QI, R0
        MOV     AT.KJ, R1
        MOV     AT.DIM, R2
        JSR     PC, VDOT
        ASH     AT.SHF, R0     ; /= sqrt(d)
        MOV     AT.SI, R1
        MOV     R0, (R1)+
```

On PDP-11 ASH takes a variable-count shift register operand; ATTN
negates AT.SHF once at line 183 (`NEG AT.SHF`) so ASH with negative
count does a right-shift.

**On 6309 there is no variable-count shift instruction.** The shift
must be emitted as an inline loop (ASRD N times) or a small
helper. For Stage 1 AT.SHF is always 2, so unrolling two ASRDs
would suffice — but the parameter interface is variable, so we
implement it as a 2-instruction loop (`ASRD / DEC / BNE`). Keeping
AT.SHF positive (no NEG) simplifies the counter: decrement
directly to zero.

Cost: ~2 bytes per ASRD + 3 bytes for loop frame = ~7 bytes code.
Runtime: DIM shifts happen per Step-4 inner iteration = SEQ*SEQ
× AT.SHF shift instructions total = 8*8*2 = 128 shift ops per
ATTN call. Negligible.

## Section E — Softmax integration (Q5)

Step 5 ([LAYER.MAC:222-230](.claude/ATTN-11-main/nn11/LAYER.MAC#L222-L230))
is a single SEQ-iteration loop calling SFTMX once per row:

```
AT.S3:  MOV     AT.SI, R0        ; R0 = ptr to S[i][:]
        MOV     AT.SEQ, R1        ; R1 = vector length (SEQ)
        JSR     PC, SFTMX
        ADD     AT.SSZ, AT.SI     ; advance to next row
        DEC     AT.OC
        BNE     AT.S3
```

**ATTN/11 SFTMX operates in-place** (input ptr = output ptr in its
calling convention).

**Our 6309 SFTMX** ([src/actfn.asm:31](attn6309/src/actfn.asm#L31))
uses a DP parameter block with separate `SF_VEC` (input) and
`SF_OUT` (output). For in-place semantics we set
`SF_VEC = SF_OUT = S[i]` before each call. `SF_LEN = SEQ = 8`.

Cost: SEQ=8 SFTMX calls per ATTN. SFTMX is 10/10 validated.

## Section F — Residual connection (Q6)

Step 7 ([LAYER.MAC:252-258](.claude/ATTN-11-main/nn11/LAYER.MAC#L252-L258)):

```
MOV     AT.YOT, R0
MOV     AT.XIN, R1
MOV     AT.SEQ, R2
MUL     AT.DIM, R2      ; R2:R3 = total elems
AT.S5:  ADD     (R1)+, (R0)+
SOB     R3, AT.S5
```

This is a plain element-wise ADD with SOB loop over SEQ*DIM
elements — 128 words for Stage 1. **No clamp, no BVS check.**
Values wrap on overflow.

**Divergence from prototype.shf** (`Y (clamp16 (+ O X))` at line
121). Same pattern as EMBED's divergence: the PDP-11 assembly
does not clamp the residual add even though the prototype says
clamp. We inherit the ATTN/11 assembly behavior; this is not a
port-introduced deviation. Will add a divergence test vector
that exercises wrap-on-overflow in the residual (one element with
`O[i][j] + X[i][j]` overflowing Q15).

We do **not** call VSADD for this: VSADD is two-stage-clamped
scale-add, wrong semantics. A tight inline ADDD loop matches the
ATTN/11 SOB loop byte-for-byte semantically.

## Section G — Call order and VTMUL reuse (Q7)

ATTN calls VTMUL **four times per ATTN invocation × SEQ positions
each** = 4×SEQ = 32 VTMUL calls total (minus optimizations). Call
parameters differ between Steps 1–3 and Step 6:

| Call | MP_MAT | MP_VIN | MP_OUT | MP_ROW | MP_COL |
|------|--------|--------|--------|--------|--------|
| Step 1 per pos | Wq | X[i] | Q[i] | DIM | DIM |
| Step 2 per pos | Wk | X[i] | K[i] | DIM | DIM |
| Step 3 per pos | Wv | X[i] | V[i] | DIM | DIM |
| Step 6 per pos | V | S[i] | Y[i] | SEQ | DIM |

Steps 1–3 share `MP_ROW = MP_COL = DIM` and can set these once
before the triple. Step 6 has `MP_ROW = SEQ, MP_COL = DIM` — must
re-set `MP_ROW` before entering Step 6.

ATTN/11 factors the Steps-1–3 triple into helper `AT.BPR`
([LAYER.MAC:263-280](.claude/ATTN-11-main/nn11/LAYER.MAC#L263-L280))
— a per-position VTMUL loop parameterized by input/weight/output
pointers. We will do the same (an internal `ATTN_BPR` label).

Note Steps 1–3 are conceptually independent projections of the
*same* X through three different weight matrices. They could be
fused into a single loop (one position at a time, compute Q/K/V
for that position, advance). ATTN/11 does not fuse — it runs
three full SEQ loops. We mirror the ATTN/11 structure; fusion is
a future optimization that changes cache behavior.

## Section H — Primitive gap analysis

**No new primitives required.** All seven steps map to existing,
validated 6309 primitives:

| Step | Operation | Primitive | Validation status |
|------|-----------|-----------|-------------------|
| 1 | Q = X @ Wq (per pos) | VTMUL | MATOP 21/21; semantics verified vs ATTN/11 VTMUL during PROJ Step 4 |
| 2 | K = X @ Wk (per pos) | VTMUL | same |
| 3 | V = X @ Wv (per pos) | VTMUL | same |
| 4 | S[i][j] = Q[i]·K[j]/sqrt(d) | VDOT + inline ASR-loop | VDOT validated; ASR-loop is trivial inline code |
| 5 | A = softmax(S) per row | SFTMX | 10/10 validated |
| 6 | Y = S @ V (per pos, VTMUL-style) | VTMUL | same |
| 7 | Y += X | inline ADDD loop | matches EMBED's inner-loop pattern |

**Important: Step 4 is VDOT-based, not a new matmul.** The
nested `SEQ × SEQ` score-matrix computation is written as a
doubly-nested loop with a VDOT call at the leaf — not as a call
to MVMUL or a new "transpose-matmul" primitive. This was the item
flagged High in the prior risk estimate; reading the source
reveals it's actually Medium — the loop bookkeeping is non-
trivial but the arithmetic kernel is a known-good primitive.

## Section I — Memory budget

**Code size estimate.** PDP-11 ATTN is ~127 instructions (lines
154–280 of LAYER.MAC minus the data). At ~3 bytes per 6309
instruction on average (our EMBED is 86 bytes for ~30
instructions, PROJ is 91 bytes for ~30 instructions), ATTN will
be in the 280–380 byte range — roughly 4× PROJ. Budgeted as **~350
bytes**.

Code-end after ATTN commit: `$11E7 (t_proj end) + ATTN body + t_attn
harness`. Still well below WEIGHT_BASE at $1A00.

**Parameter / state block.** With AT.BW/BI/BO/BC aliased onto
AT.QI/KJ/SI/IC as noted in Section C, the unique storage count is:

- 9 caller fields: 18 bytes
- 2 derived sizes (RSZ, SSZ): 4 bytes
- 4 workspace bases (QQ, KK, VV, SS): 8 bytes
- 4 cursor/counter slots (QI/KJ/SI/YI + OC/IC): 12 bytes
- (BPR helpers alias onto cursor slots)

**Total: ~42 bytes of unique AT_* storage.** Doubling to 50 for
wiggle room suggests BW_SCRATCH expansion 116 → ~170 bytes. New
BW_END = `$4420 + 170` = `$44CA`. Within BWD_WORK budget
($3E00–$4600 = 2048 bytes; usage would be ~1770 bytes, 278 bytes
headroom).

**Workspace (test-harness-allocated).** 896 bytes for Q/K/V/S at
a test buffer address (proposed `$1B00` — above OUT_BUF at
`$19C0 + 160 = $1A60`, leaving a small gap).

**Total new storage**: 50 bytes (AT_* params) + 896 bytes (WRK
buffer) = 946 bytes. BW_SCRATCH expansion is the only equates
change; WRK lives in the harness.

## Section J — Risk assessment per step

Re-evaluated after reading the source (prior estimates revised
down in two places):

| Step | Risk | Justification |
|------|------|---------------|
| 1 | **Medium** | Three near-identical VTMUL calls via helper. Bookkeeping errors likely: wrong weight ptr, wrong workspace offset. Mitigated by AT.BPR helper. |
| 2 | **Medium** | Same as Step 1. |
| 3 | **Medium** | Same as Step 1. |
| 4 | **Medium** (was High) | Not a new matmul — doubly-nested VDOT loop. Existing primitive. Risk is loop-cursor discipline (QI advances once per outer iter, KJ resets each outer) and the per-score ASR shift. |
| 5 | **Low** | SEQ calls to validated SFTMX with SF_VEC=SF_OUT=row ptr. Simple wrapper. |
| 6 | **Medium** | VTMUL with MP_ROW=SEQ (not DIM — easy to miss). Per-position loop similar to Step 1 but with different dimensions. |
| 7 | **Low** | 128-iteration ADDD loop, no clamp, mirrors EMBED's inner pattern. |

Highest-risk failure modes are composition/bookkeeping:
- Forgetting to reset MP_ROW when moving from Steps 1–3 to Step 6.
- Off-by-one in Step 4's nested cursor advancement.
- SF_VEC not pointing at the correct S row in Step 5.

None of these are algorithmic surprises — they're the kind of bugs
a per-step test harness catches in one run each.

## Section K — Implementation order

**Recommendation: Option A (monolithic) with per-element test
reporting.**

Reasoning: the per-step test-hook approach (Option B or C) adds
conditional assembly or internal entry points that complicate both
the ATTN source and the harness. The 7 steps are each shallow
(Steps 1/2/3 are copies, Step 5 is one-liner, Step 7 is a loop).
A failing composite produces a *specific* output element mismatch
at a *specific* (i, j) or (i, v) index. With a well-instrumented
harness (per-element printout of first divergence, not just
PASS/FAIL), the failing element's index combined with the ATTN/11
source tells you which step is responsible:

- Mismatch in `Y[i][*]` at all positions, all dims: Step 1–3 wrong
  (Q, K, or V corrupted before Step 4).
- Mismatch in Y spreading out from specific position: Step 5 or 6.
- Mismatch proportional to X[i][j] additive contribution: Step 7.
- Score magnitude off by factor of 2/4/8: Step 4 shift wrong.

Instrumentation plan for the harness: print the first
`(position, dim)` where OUT_BUF diverges from EXPECT, not just
"FAIL". If the diagnostic runs out of budget, we add per-step hooks
as a Phase 2 iteration — but start monolithic.

**Stretch consideration.** If ATTN fails and the diagnostic cost
stays high across 2–3 iterations, escalate to Option C (hybrid):
make `ATTN_Q`, `ATTN_K`, `ATTN_V`, `ATTN_SC`, `ATTN_SFT`, `ATTN_O`,
`ATTN_RES` internal labels public so the harness can call any
prefix and verify the intermediate workspace. This is a retroactive
mitigation, not the default.

## Section L — Phase 2 outline

Phase 2 is iterative. Expect one or more fix-and-retry cycles.
Gated reviews at every step:

1. **Equates update** (`include/equates.inc`)
   - AT_* parameter block, +~50 bytes BW_SCRATCH expansion.
   - Review before proceeding.

2. **ATTN source** (`src/layer.asm` append)
   - Monolithic ATTN entry, internal AT_BPR helper.
   - Inline variable-count ASR loop for AT.SHF.
   - Steps 1–3 via AT_BPR, Step 4 nested VDOT, Step 5 SFTMX loop,
     Step 6 per-position VTMUL, Step 7 inline ADDD loop.
   - Zero-length guard on AT_SEQ at entry (matches EMBED/PROJ
     defensive-hardening pattern).
   - Review before proceeding.

3. **Reference generator** (`tables/gen_attn_vectors.py`)
   - Must replicate per-product VTMUL semantics (lesson from PROJ
     Step 4 — do not repeat the full-sum-then-shift mistake).
   - Must replicate VDOT's 32-bit accumulation → ASHC #-8 →
     clamp Q15 semantics for scores.
   - Must replicate our SFTMX's exact LUT-based softmax.
   - ATTN-specific safety net: at least one test must exercise
     the residual wrap (Step 7 no-clamp divergence) and at least
     one must exercise the scaling shift on non-trivial values.
   - Review before regenerating vectors.

4. **Test harness** (`test/t_attn.asm`)
   - 5 tests spanning single-position, multi-position,
     architecture-max (SEQ=8, DIM=16), residual wrap divergence,
     and saturation/scaling edge cases.
   - Same RECORD_PASS/FAIL/bitmap framework as PROJ/EMBED.
   - Per-element first-divergence reporting (new vs prior
     harnesses — for diagnosability).
   - Caller allocates 896-byte WRK buffer at ~$1B00.
   - Review before assembling.

5. **Run and iterate**
   - MAME run, collect RESULT_BITS, LAST_FAIL_ID, first-divergence
     (position, dim) if any test fails.
   - One diagnostic per failure (no speculative fixes).
   - Most likely iteration: element-off-by-one, MP_ROW not reset
     before Step 6, or SF_VEC/SF_OUT aliasing mistake.

6. **Regression check**
   - On 5/5 pass: re-run t_embed, t_proj, t_vcpycl, t_vsadd to
     confirm no regressions from BW_SCRATCH expansion / layer.asm
     append.

7. **Commit and push** (Jsearle01/ATTN-CC3 main)
   - Update DEVIATIONS.md with the residual no-clamp entry (if not
     already covered by EMBED's same-pattern entry).
   - Update LAYER_PLAN.md to mark ATTN complete.
   - Stop before Step 6 (integration into FORWRD.asm).

## Section M — Flags for Phase 2

Items that may surprise during implementation or review:

1. **FWD_CACHE byte sizing.** `equates.inc`'s Q8SZ=1 and
   `DSEQ=128` allocations assume 1-byte-per-element storage, but
   all code uses 16-bit words. This mismatch predates ATTN; the
   Stage 1 ATTN test harness sidesteps it by allocating its own
   WRK buffer. Flag as a pre-existing memory-map issue to resolve
   before FORWRD integration (not in this commit).

2. **Variable-count shift on 6309.** No single-instruction ASH
   equivalent. Inline ASR-loop adds 3 extra cycles per bit of
   AT.SHF. For AT.SHF=2 (Stage 1 only) the cost is 6–8 cycles
   per score, × 64 scores = ~500 extra cycles per ATTN call.
   Negligible in the overall ATTN cost (dominated by 32 VTMUL +
   64 VDOT + 8 SFTMX calls).

3. **Step 7 residual wrap divergence.** The assembly does not
   clamp `O + X`. This is the same class of divergence as EMBED
   already documented. Test harness must include at least one
   vector that triggers wrap so any future "clamp the residual"
   fix is caught immediately.

4. **MP_* overlap across VTMUL calls.** MP_* fields are shared
   scratch for MATOP primitives. ATTN sets them repeatedly (32
   times per call). No aliasing concern — ATTN is the only caller
   of MP_* during its own execution, and each `STD MP_*` / `JSR
   VTMUL` pair is atomic within a single iteration.

5. **BPR helper state aliasing.** AT.BW/BI/BO/BC aliased onto
   AT.QI/KJ/SI/IC in the 6309 port saves 8 bytes. Documented in
   equates.inc comment block; future ATTN modifications must
   respect that QI/KJ/SI/IC are not live while AT_BPR is running,
   and BW/BI/BO/BC are not live outside Steps 1–3.

6. **Softmax LUT availability.** SFTMX depends on EXPTBL (256-byte
   exp LUT). The LUT is already loaded and validated (t_exptbl,
   t_sftmx passing). No new dependency from ATTN.

7. **Stack depth during inner VDOT call.** Step 4 calls VDOT
   inside a doubly-nested loop. VDOT uses `PSHS B` (1 byte) plus
   its own internal state. ATTN's outer stack frame is also
   minimal (counter bytes at most). Combined stack depth well
   under 32 bytes — no concern with STACK_TOP at $4800.

8. **Q-register reuse across composite calls.** VDOT, VTMUL,
   SFTMX all use Q internally. ATTN does not use Q directly
   across primitive calls — each call self-contains its Q usage.
   No concern.

## Section N — Deferred items

These are out of scope for the ATTN commit but should be tracked:

- **FWD_CACHE layout fix.** Resize all FC_* fields to 2 bytes per
  element and update the memory map. Required before FORWRD
  integration. Not required for t_attn testing.

- **Shift-count optimization.** If AT.SHF is ever allowed to
  vary (e.g. d_model other than 16 in a future stage), the inline
  ASR-loop is already general. For Stage 1 always-2, an unrolled
  `ASRD/ASRD` would save the loop overhead — defer until
  benchmarking justifies it.

- **Steps 1–3 fusion.** Three sequential SEQ loops could collapse
  to one loop doing Q/K/V per position. Saves ~2*SEQ iterations
  of loop overhead. ATTN/11 chose not to fuse; we mirror that
  choice for Stage 1 correctness, optimize later if needed.

- **Per-step public entries (Option C fallback).** If Phase 2
  needs granular diagnosability, expose ATTN_Q, ATTN_SCORES,
  ATTN_SOFTMAX, ATTN_OUTPUT as internal but linkable symbols.
  Not needed for initial monolithic attempt.

## Summary

ATTN is **7 steps, 9 params, 25 words of state, ~350 bytes of
code**, built exclusively from existing validated primitives
(VDOT, VTMUL, SFTMX) plus an inline 3-instruction variable-ASR
loop. No new primitives, no new test infrastructure beyond a
slightly smarter harness with per-element divergence reporting.

Three divergences from prototype.shf are inherited from ATTN/11
(Step 4 approximate-shift for 1/sqrt(d), Step 7 no-clamp residual,
pre-existing VTMUL per-product rounding) — all documented, all
blessed by the PDP-11 author, none introduced by this port.

Phase 2 workflow: equates → source → generator → harness → run →
(iterate) → regression → commit. Review-gated at each step. Expect
1–2 iterations on the harness or generator before 5/5 pass.
