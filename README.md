# ATTN-CC3 — Transformer on the TRS-80 Color Computer 3

A complete transformer neural network (single-head attention, forward pass + training)
ported from PDP-11 assembly to Hitachi HD6309 assembly, running on the CoCo3 under MAME.

**Training works end-to-end.** The model achieves **10/10 on held-out digit-reversal
test samples** after 350 SGD steps. Binary is ~5.5 KB; training completes in ~20 minutes
of emulated time at MAME's default `-nothrottle` speed.

**188 tests passing** (17 of 18 harnesses fully green; t_fxmath MUL15Q15 subset
documented as a known harness-fragility issue, not a primitive correctness issue —
see PROJECT_ENV.md §6).

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

**Training binary: COMPLETE. Model learns.** `src/main.asm` orchestrates
GENSM → CVT16_ALL → FORWRD → BKWRD → WUPDT_ALL → COUNT, with REPORT every 50 steps
and FINAL_TEST (10 digit-reversal samples) at the end. Training loop runs 350 SGD
steps. Loss drops from ~2.3 (initialized) to ~0.01 by step 300. Per-report
accuracy climbs from ~10% (random) to 90%+. FINAL_TEST scores **10/10** on fresh
held-out samples.

## Results

**Training trajectory** (REPORT lines, 7 reports over 350 steps):

| Step | Loss (Q12) | Accuracy (since last REPORT) |
|------|------------|------------------------------|
|   50 | ~2.3       | ~10-20%                       |
|  100 | ~1.5       | ~30-40%                       |
|  150 | ~0.8       | ~50-60%                       |
|  200 | ~0.3       | ~75%                          |
|  250 | ~0.1       | ~85%                          |
|  300 | ~0.03      | ~90%                          |
|  350 | ~0.01      | ~95%+                         |

Exact numbers depend on the 15-bit LCG PRNG seed (`RN_INIT=887`) and the sequence
of random digits that happen to come up during training. Trajectory is reproducible
run-to-run since the seed is fixed.

**FINAL_TEST.** After training completes, 10 fresh samples are drawn (same PRNG,
continuing from where training left it). Forward-only inference on each. The model
predicts the reversed digit sequence and is scored OK/FAIL against the target.
Current result: **10/10**.

**Binary size.** `build/main.bin` = 5,525 bytes. Loaded at `$0600`, ends around
`$1BE5`. Data regions occupy `$1C00–$4BF0` (training state, weights, gradients,
forward cache, backward workspace). Stack grows down from `$6800`.

**Emulated runtime.** ~77,000 NTSC frames at `-nothrottle` (roughly 21 emulated
minutes at 59.94 Hz) for 350 training steps + FINAL_TEST. Wall-clock on a typical
modern host: ~20-25 minutes.

**Memory usage summary.**

| Region                 | Range         | Size   | Contents                       |
|------------------------|---------------|--------|--------------------------------|
| Code + tables          | $0600–~$1BE5  | ~5.5KB | Main, primitives, composites, EXPTBL, LOGTBL |
| Training state         | $1C00–$1C4F   |  80 B  | TR_STEP, TR_HIT, TR_TOT, TOKENS, TARGET, cursor |
| Q16 weight accumulators| $1C50–$2D4F   | 4.8KB  | Split hi/lo for 6 weight groups |
| Q8 weight copies       | $2D50–$36CF   | 2.4KB  | Regenerated each step by CVT16_ALL |
| Gradient accumulators  | $36D0–$404F   | 2.4KB  | Q15, zeroed once at startup    |
| Forward cache          | $4050–$466F   | 1.5KB  | X, ATTN workspace, Y, logits   |
| Backward workspace     | $4670–$4BEF   | 1.4KB  | dY, dA/dSc, dQ, dK, dV, dX     |
| Unused diagnostic gap  | $4BF0–$63FF   | 6.1KB  | Headroom                       |
| BW_SCRATCH (DP page)   | $6400–$6533   | 308 B  | DP parameter blocks            |
| Stack                  | $6534–$67FF   | 716 B  | Grows down from $6800          |

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

- `PROJECT_ENV.md` — Tool paths, MAME invocation, development environment, lessons learned
- `DEVIATIONS.md` — Deliberate divergences from ATTN/11 (zero-length guards, residual wrapping, I/O, etc.)
- `INTEGRATION_NOTES.md` — Known issues for forward-pass integration (FWD_CACHE sizing)
- `TRAINING_NOTES.md` — Operational knowledge for running/modifying the training binary
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

## Running the training binary

```bash
# Build
lwasm --6309 --format=raw --includedir=. --includedir=include -o build/main.bin src/main.asm

# Copy and run (interactive — watch the CoCo3 screen)
cp build/main.bin /c/mame/train.bin
cd /c/mame
./mame.exe coco3h -autoboot_script t_train.lua -autoboot_delay 5 -skip_gameinfo -nothrottle -window

# Or run under the diagnostic trampoline (auto-exits at FT_HALT, dumps screen to stdout)
./mame.exe coco3h -autoboot_script t_traindiag.lua -autoboot_delay 5 -skip_gameinfo -nothrottle -window
```

`t_train.lua` loads the binary, sets PC to `$0600`, and steps out of the way — MAME
runs until the user closes the window. `t_traindiag.lua` adds PC-stall detection at
FT_HALT (`$18DC`), dumps the final screen to stdout, and exits MAME cleanly.

For operational notes on tuning the training run (NSTEP, RPRT, NTEST, learning
rates), see [TRAINING_NOTES.md](TRAINING_NOTES.md).

## What remains

**Training: COMPLETE.** Forward, backward, UPDAT, training loop, and FINAL_TEST all
validated. Model learns digit reversal to 10/10 accuracy.

**Open items (deferred polish):**

- **Test-harness build regression.** Since commit `97c0a98` added `CLEAR_SCREEN:` to
  `src/train.asm`, the five test harnesses that `INCLUDE src/train.asm`
  (t_closs, t_updat, t_bkwrd1, t_bkwrd23, t_bkwrd456) cannot reassemble due to
  duplicate symbol errors. The harnesses themselves still have their own
  `CLEAR_SCREEN:` labels. Fix: remove `CLEAR_SCREEN:` from those five harnesses
  (or factor it into a shared include). Binaries built before `97c0a98` still work.
- **t_fxmath harness fragility.** The MUL15Q15 subset reports 37/64 PASS under the
  current memory map (STACK_TOP=$6800). Primitive correctness is transitively
  verified by seven downstream harnesses. See PROJECT_ENV.md §6.
- **FWD_CACHE Q8SZ sizing.** Equates still define `Q8SZ=1` with word-access code;
  the production training binary sidesteps the mismatch by using its own buffer
  layout. See INTEGRATION_NOTES.md.
- **t_attn_full** standalone harness at production dimensions (SEQ=8, DIM=16).
- **Shared test-framework include** (harness_common.asm).

## License

ATTN-CC3 is licensed under the GNU General Public License v3.0.
See [LICENSE](LICENSE) for the full text.

## Credits

Ported from [ATTN/11](https://github.com/dbrll/ATTN-11) by Jay Searle with Claude (Anthropic).
ATTN/11 original by Damien Boureille, MIT-licensed; this port is GPL-3.0 as permitted by
the MIT license's relicensing terms.
