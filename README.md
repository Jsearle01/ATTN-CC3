# ATTN-CC3 — Transformer on the TRS-80 Color Computer 3

A complete transformer neural network (single-head attention, forward pass + training)
ported from PDP-11 assembly to Hitachi HD6309 assembly, running on the CoCo3 under MAME.

**188 tests passing | 0 failing (4 unreachable in t_fxmath — see PROJECT_ENV.md)**

## What this is

- A port of ATTN/11 (a minimal transformer implementation in PDP-11 assembly) to the HD6309 processor on the TRS-80 Color Computer 3
- The model learns a digit-reversal task: input 8 random digits [0-9], predict the reversed sequence
- Architecture: single-head attention with d_model=16, seq_len=8, vocab=10
- All arithmetic is fixed-point: Q8 activations, Q15 gradients, Q16 weight accumulators
- No floating point hardware — everything is integer math with explicit shift-and-clamp

## Technical stack

- **Processor:** Hitachi HD6309 (enhanced 6809) in native mode
- **Platform:** TRS-80 Color Computer 3 (CoCo3), emulated via MAME 0.281
- **Assembler:** LWASM (6309 mode)
- **Reference:** ATTN/11 PDP-11 assembly + prototype.shf (Clojure-like Q8 reference)
- **Test framework:** Per-routine test harnesses with Lua trampolines for automated MAME execution
- **Validation:** Byte-exact match against Q8 reference for all operations

## Architecture

```
Layer        What it does                              Tests
─────────────────────────────────────────────────────────────
FXMATH       Q8/Q15 fixed-point multiply, divide       64
VECOP        Vector dot, add, sub, scale, max,          44
             copy, clear, scalar-add
MATOP        Matrix-vector multiply (MVMUL, VTMUL),     21
             matrix-vector add (MVADD), outer product
ACTFN        Softmax (LUT-based), exp table,            25
             fixed-point division
LAYER        EMBED (token+position embedding)            6
             PROJ (output projection via VTMUL)          5
             ATTN (7-step attention composite)           5
Integration  Full forward pass: EMBED→ATTN→PROJ          1
TRAIN        CLOSS (cross-entropy loss)                  4
             BKWRD Step 1 (dLogits, dWout, dY)           4
             BKWRD Steps 2-3 (dA, dV, dSc)              4
             BKWRD Steps 4-6 (dQ/dK, dX, embedding)      4
             UPDAT (WUPDT, CVT16, INITW, RAND)           5
```

## Current status

**Forward pass: COMPLETE.** EMBED→ATTN→PROJ validated end-to-end with byte-exact
match against Q8 reference. Tokens in → logits out, through embedding, single-head
attention (Q/K/V projections, scaled dot-product scores, softmax, weighted aggregation,
residual connection), and output projection.

**Backward pass: COMPLETE (all 6 steps).**

- ✅ Step 1: dLogits → dWout, dY (softmax-minus-one-hot, <<7 shift, OUTER + MVMUL)
- ✅ Step 2: dA, dV from backward through O=A@V (VDOT + VSADD)
- ✅ Step 3: dSc from softmax backward (clamped subtract, inline multiply, variable shift)
- ✅ Step 4: dQ, dK from backward through Q·K^T (VTMUL + column extraction)
- ✅ Step 5: dX + weight gradients dWq/dWk/dWv (MVADD + OUTER triple accumulation)
- ✅ Step 6: Embedding gradients (scatter-add with clamping)

**UPDAT: COMPLETE.** SGD weight update with Q16 split hi/lo accumulators, Q16↔Q8
conversion, weight initialization via 15-bit LCG PRNG, gradient zeroing.

**Training binary: ASSEMBLED.** `src/main.asm` orchestrates GENSM → CVT16_ALL →
FORWRD → BKWRD → WUPDT_ALL → COUNT, with REPORT every 50 steps and FINAL_TEST at
the end. Binary size 5459 bytes, fits at $0600-$1B4F with data starting $1C00.
I/O pipeline smoke-tested on CoCo3 hardware (banner, digit output, PUTDEC, PUTLSS
all rendering correctly). Full 350-step run requires manual observation in MAME
(one training step exceeds the 32-frame autoboot-callback window).

## Key technical achievements

1. **No new primitives needed for backward pass.** Every backward operation maps to an
   existing forward-pass primitive (VTMUL, MVMUL, MVADD, OUTER, VDOT, VSADD). Q8×Q15→Q15
   uses the identical >>8 shift as Q8×Q8→Q8.

2. **Per-product vs full-row accumulation semantics.** VTMUL uses per-product
   shift-accumulate (documented as deliberate in ATTN/11: "Per-product Q8 rounding,
   acceptable for d_model=16"). MVMUL uses full 32-bit row accumulation. Both validated
   against PDP-11 source and prototype.

3. **Automated test execution via MAME Lua trampolines.** PC-stall polling detects
   harness completion within MAME's ~30-frame autoboot callback window. No manual
   intervention needed.

4. **Mixed-precision gradient arithmetic.** Forward activations in Q8, gradients in
   Q15, weight accumulators in Q16 — all using the same shift-based primitives with
   different semantic interpretation.

## Repository structure

```
include/
  equates.inc          Memory map, parameter blocks (MP_*, SF_*, EM_*, PR_*, AT_*, CL_*, BK_*)
  macros.inc           Assembly macros
  fxmath.inc           Fixed-point math constants

src/
  fxmath.asm           Q8/Q15 multiply and divide
  vecop.asm            Vector operations (VDOT, VADD, VSUB, VSCL, VMAX, VCPY, VCLR, VSADD)
  matop.asm            Matrix operations (MVMUL, MVADD, VTMUL, OUTER)
  actfn.asm            Activation functions (SFTMX, EXPTBL, FXDIV)
  layer.asm            Forward composites (EMBED, PROJ, ATTN + AT_BPR helper)
  train.asm            Training routines (CLOSS, BKWRD Steps 1-3, future: Steps 4-6 + UPDAT)

tables/
  gen_*.py             Reference generators (Python, produce test vectors)
  *_vectors.asm        Generated test vectors
  exptbl.asm           256-entry exp lookup table for softmax
  logtbl.asm           257-entry log lookup table for cross-entropy loss

test/
  t_*.asm              Per-routine test harnesses

tools/
  mame_run.sh          MAME test execution wrapper
```

## Key project files

- `PROJECT_ENV.md` — Tool paths, MAME invocation, development environment
- `DEVIATIONS.md` — Deliberate divergences from ATTN/11 (zero-length guards, residual wrapping, etc.)
- `INTEGRATION_NOTES.md` — Known issues for forward-pass integration (FWD_CACHE sizing)
- `ATTN_PLAN.md` — ATTN composite architecture and implementation plan
- `TRAIN_PLAN.md` — Training phase plan with memory budget and implementation order

## Building and testing

```bash
# Assemble a test harness
lwasm --6309 --format=raw --includedir=. --includedir=include -o test/t_proj.bin test/t_proj.asm

# Run under MAME (requires CoCo3 ROMs in /c/mame/)
tools/mame_run.sh t_proj

# Run all active harnesses
for t in t_vcpycl t_vsadd t_embed t_proj t_attn t_integ t_closs t_bkwrd1 t_bkwrd23; do
  echo "=== $t ===" && tools/mame_run.sh $t
done
```

## What remains

**To complete training:**

1. BKWRD Steps 4-6 (~3 sessions)
2. UPDAT — SGD weight update + initialization (~1 session)
3. Training loop integration + digit-reversal task (~1-2 sessions)
4. Training validation — verify loss decreases and model learns (~1 session)

**Deferred polish:**

- t_attn_full standalone harness (SEQ=8 DIM=16 production dimensions)
- Shared test-framework include (harness_common.asm)
- MATOP sign-extension optimization
- SFTMX max-count documentation

## License

ATTN-CC3 is licensed under the GNU General Public License v3.0.
See [LICENSE](LICENSE) for the full text.

## Credits

Ported from [ATTN/11](https://github.com/dbrll/ATTN-11) by Jay Searle with Claude (Anthropic).
ATTN/11 original by Damien Boureille, MIT-licensed; this port is GPL-3.0 as permitted by
the MIT license's relicensing terms.
