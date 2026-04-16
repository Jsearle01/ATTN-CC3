# Integration test preparation notes

Carries forward items surfaced during LAYER development that must
be resolved before the Step 6 integration test (full forward pass
through EMBED → ATTN → PROJ against FWD_CACHE), but are deferred
in this commit to keep the ATTN commit focused.

## Known issue: FWD_CACHE Q8SZ/DSEQ sizing mismatch

### Symptom

`include/equates.inc` defines:
```
Q8SZ            EQU     1
DSEQ            EQU     D*SEQ           ; 128
```

and sizes the forward-cache fields as if each element were one byte:

```
FC_X            EQU     FWD_CACHE               ; SEQ*D, post-embed
FC_Q            EQU     FC_X+DSEQ              ; SEQ*D
FC_K            EQU     FC_Q+DSEQ
...
```

But all forward-path assembly (EMBED, PROJ, ATTN, VTMUL, VDOT,
SFTMX) loads and stores 16-bit words via `LDD`, `STD`, `ADDD`.
Q7.8 values are held as 16-bit quantities throughout; the Q8SZ=1
definition does not reflect word semantics.

**Consequence.** FC_X through FC_PROBS are each allocated half the
bytes they actually need. If FORWRD writes EMBED output into FC_X,
then EMBED's SEQ*DIM = 128 *words* = 256 bytes will overflow the
128 bytes reserved at FC_X, corrupting FC_Q.

### Current impact: none

- EMBED, PROJ, and ATTN unit tests (Stage 1 LAYER) all allocate
  their own buffers in the test-buffer region (`$1760+`). None
  of them touch FC_*.
- LAYER-level code (`src/layer.asm`) references neither FWD_CACHE
  nor FC_X. The cache is declared in equates but not yet consumed.

### Must resolve before Step 6 (integration test)

Two paths:

**A. Fix Q8SZ to reflect word semantics.** Change `Q8SZ EQU 2` (or
introduce a word-aware `DSEQB EQU D*SEQ*Q16SZ`). Re-derive all FC_*
offsets. Risk: any code or data structure that depends on the current
byte arithmetic in FC_* addressing needs re-verification. Low blast
radius today (no consumers), but the principle of "bytes stored in
byte-counted offsets" should be restored.

**B. Deprecate FWD_CACHE; carry the WRK-allocation pattern through
the integration harness.** The EMBED/PROJ/ATTN test harnesses
already demonstrate the pattern: caller allocates a dedicated
buffer region with known sizes. Stage 1 could adopt this pattern
at the integration level and leave FWD_CACHE as a future Stage-2
concern.

**Resolved: Path B.** The integration harness (t_integ.asm) allocates
all intermediate buffers explicitly at SCRATCH_BASE+offsets, matching
the pattern established by t_attn's WRK_BUF. FWD_CACHE definitions
remain in equates.inc unchanged; the integration test does not
reference them. No changes to Q8SZ or FC_* EQUs. FWD_CACHE cleanup
deferred to Stage 2 (when FORWRD/BKWRD routines need cache storage).

## Known divergences from prototype.shf (inherited from ATTN/11)

These are already documented in DEVIATIONS.md but repeated here
because they matter for integration-test reference generation:

- **EMBED Step 2 residual add:** no clamp. Prototype says
  `clamp16`, assembly does raw ADD. EMBED test vector
  `EMBED_WRAP` exercises this.
- **ATTN Step 7 residual add:** no clamp. Same pattern as EMBED.
  ATTN test vector must exercise this (to be added with t_attn).
- **VTMUL per-product rounding:** deliberate precision tradeoff
  documented in MATOP.MAC line 132. Generator must match (lesson
  from PROJ Step 4 generator fix).

Integration-test reference generator must honor all three.

## Workspace sizing cheat-sheet

For Stage 1 (SEQ=8, DIM=16, VOC=10):

| Step | Intermediate | Size (words) | Size (bytes) |
|------|--------------|--------------|--------------|
| EMBED | X (output) | 128 | 256 |
| ATTN | Q | 128 | 256 |
| ATTN | K | 128 | 256 |
| ATTN | V | 128 | 256 |
| ATTN | S / A (shared) | 64 | 128 |
| ATTN | Y (output to caller) | 128 | 256 |
| PROJ | logits | 80 | 160 |

ATTN's scratch (Q+K+V+S) = 896 bytes. This is the AT_WRK
allocation the caller must provide to ATTN. It is *not* the same
as the caller-provided AT_YOT buffer (256 bytes).

Forward pass total intermediate storage: 256 (X) + 896 (ATTN WRK) +
256 (Y) + 160 (logits) = **1568 bytes**. Current FWD_CACHE budget
is 1.5KB (1536 bytes) — 32 bytes short. Under path A (fix Q8SZ),
FWD_CACHE needs to expand or some intermediates share storage
(e.g. X and Y could overlap since X is read-only by Step 7).

Revisit when Step 6 exploration begins.
