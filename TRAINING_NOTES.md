# TRAINING_NOTES.md — Operational notes for the training binary

Operational knowledge for running and modifying `src/main.asm` (the
training binary). Architectural background lives in `TRAIN_PLAN.md`;
environment/tool information lives in `PROJECT_ENV.md`. This file is
for anyone who wants to actually run training, tune hyperparameters,
or retarget the binary to a different task.

---

## 1. What the binary does

`src/main.asm` orchestrates a single-head transformer training run:

1. `ORCC #$50` to mask IRQ/FIRQ. `LDMD #1` for 6309 native mode.
2. `LDS #STACK_TOP` ($6800).
3. `CLEAR_SCREEN` and print header "ATTN-CC3 TRAINING".
4. Seed the 15-bit LCG PRNG (`RN_SED = RN_INIT = 887`).
5. `INITW_ALL` — randomize all 6 weight groups (Q8 → Q16 hi/lo).
6. `BKWRD_SETUP` + `ZEROG` — zero all gradient accumulators **once**.
7. Enter `TRAIN_LOOP` for NSTEP=350 iterations:
   - `GENSM` — generate one random input + reversed target.
   - `CVT16_ALL` — regenerate Q8 weight copies from Q16 hi/lo.
   - `FORWRD` — forward pass (EMBED → ATTN → PROJ).
   - `BKWRD` — full 6-step backward pass into gradient buffers.
   - `WUPDT_ALL` — SGD update into Q16 hi/lo, **zeros each gradient
     element as it's consumed**.
   - `COUNT` — compare argmax(logits[i]) against TARGET[i] for each
     position, tally into TR_HIT / TR_TOT.
   - If `step % RPRT == 0`: `REPORT` (calls CLOSS, prints status
     line, resets TR_HIT/TR_TOT).
   - Increment TR_STEP, branch back.
8. `FINAL_TEST` — NTEST=10 held-out samples, forward-only inference,
   print `<input>-><prediction> OK/FAIL`, then final `ACC N/NTEST`.
9. `FT_HALT: BRA FT_HALT` — freeze screen indefinitely.

---

## 2. Expected trajectory

With the default PRNG seed (RN_INIT=887) and current
hyperparameters (NSTEP=350, RPRT=50, LR_SHF_EMB=4, LR_SHF_ATN=2,
LR_SHF_OUT=6), the 7 REPORT lines show roughly:

| Step | Loss (Q12) | Hit/Tot |
|------|------------|---------|
|   50 | ~2.3       | ~40/400  |
|  100 | ~1.5       | ~120/400 |
|  150 | ~0.8       | ~220/400 |
|  200 | ~0.3       | ~300/400 |
|  250 | ~0.1       | ~340/400 |
|  300 | ~0.03      | ~360/400 |
|  350 | ~0.01      | ~380/400 |

"Hit/Tot" is per-position accuracy summed across all 50 training
steps since the last REPORT (8 positions × 50 steps = 400 total
positions per report interval).

FINAL_TEST runs 10 fresh samples. Current result: **10/10**. Any
drop below 9/10 under the fixed seed signals a regression.

---

## 3. Tuning training-loop hyperparameters

All constants live in `include/equates.inc`. Reassemble after each
change (no dependency tracking).

### `NSTEP` — number of training steps

Default 350. Raising to 500 tends to push the model deeper into
the minimum (loss ~0.005) but past ~400 the marginal improvement is
in the noise, and wall-clock time scales linearly. Lower values
(e.g. 200) show the model mid-training — useful for diagnostics.

### `RPRT` — report interval

Default 50 (→ 7 reports at NSTEP=350). Must divide NSTEP evenly to
get a final report at the last step (350 / 50 = 7). Smaller RPRT
means more screen output but also more CLOSS invocations (expensive).

### `NTEST` — FINAL_TEST sample count

Default 10. Each sample takes ~2 screen rows in the current display
format (one sample line + trailing OK/FAIL). Going past 12-13
samples with the verbose format triggers the SCROLL path; with the
compacted format (21-23 char lines) up to 16 fit without scroll.

### `LR_SHF_EMB` / `LR_SHF_ATN` / `LR_SHF_OUT`

SGD learning rates, encoded as right-shifts of the gradient before
subtracting from the Q16 weight accumulator:

```
w_q16 -= grad_q15 >> (lr_shift - 1)
```

Shift+1 halves the effective LR. Defaults:

| Symbol        | Default | Effective LR |
|---------------|---------|--------------|
| LR_SHF_EMB    | 4       | 2^-3 ≈ 0.125  |
| LR_SHF_ATN    | 2       | 2^-1 = 0.5    |
| LR_SHF_OUT    | 6       | 2^-5 ≈ 0.031  |

**`LR_SHF_ATN` was raised from 1 to 2 in commit `2c5d096`.** See
§5 "The divergence incident" for why.

Raising a shift value makes training slower (model converges later)
but more stable. Lowering it makes training faster but risks
divergence near convergence (classic LR-too-high symptom). The
three values are decoupled; adjust the one whose weight group is
implicated in the failure mode.

### `RN_INIT` — PRNG seed

Default 887 (inherited from ATTN/11). Changing this changes the
entire training trajectory — different initial weights, different
sample sequence, different final accuracy. FINAL_TEST results at
different seeds are not directly comparable to the 10/10 reference.

---

## 4. Why REPORT calls CLOSS but COUNT does not

`CLOSS` (cross-entropy loss) runs softmax on all SEQ logit rows,
looks up `-ln(p/256)*4096` from LOGTBL, and sums into a 32-bit
Q12 accumulator. It is **expensive** — roughly 8× SFTMX + 8
LOGTBL lookups + 32-bit divide-by-SEQ per call. A full SFTMX
is ~1-2 ms of emulated time on a CoCo3; CLOSS totals ~10-20 ms.

`COUNT` (argmax per position, compare to target) runs VMAX on each
row (8 cheap 10-element scans) and does 8 word compares. It is
~100× cheaper than CLOSS. We want per-position accuracy on every
training step (for REPORT to print a meaningful ratio), so COUNT
runs every step.

Loss is only displayed at REPORT boundaries, so we only need CLOSS
at those points. Matches ATTN/11's TRAIN.MAC placement.

If REPORT frequency changes (smaller RPRT), CLOSS cost per total
run grows proportionally. At RPRT=10 (35 reports), CLOSS accounts
for ~30% of wall-clock time.

---

## 5. Why ZEROG runs once at startup, not per step

`ZEROG` walks all 6 gradient accumulators and writes zero to every
element (1216 Q15 words = 2432 bytes of zeroing). Running it every
step would add ~5 ms per step × 350 steps = ~1.75 s emulated time.

We call it once at startup to establish the initial state. On
every subsequent step, `WUPDT_ONE` zeroes each gradient element
**as it reads it** — the read, the `>>shift`, the Q16 subtract,
and the zero-write are all inlined into the per-element loop body
of WUPDT. Net cost: zero extra cycles per step vs a separate zeroing
pass. Matches ATTN/11's UPDAT.MAC pattern.

This also means the gradient buffer is write-before-read across
BKWRD and WUPDT boundaries — if the order is ever broken (e.g.
running WUPDT without running BKWRD first), the stale gradients
from the previous step get applied as updates.

---

## 6. The divergence incident (step 300→350 at LR_SHF_ATN=1)

Before commit `2c5d096`, `LR_SHF_ATN=1` (grad>>0 = full gradient
into Wq/Wk/Wv). Training was beautiful through step ~300: loss
dropped monotonically to ~0.01, per-report accuracy reached ~90%.

Then between steps 300 and 350, loss jumped from ~0.01 to ~5 and
per-report accuracy regressed to ~52%. FINAL_TEST (evaluating
post-step-350 weights) scored **0/10**.

Root cause: near convergence, most gradients are tiny, but a
single sample with an unusual argmax pattern produces a
disproportionately large dL. At LR=full, that single-sample kick
lands the attention weights far enough outside the trained basin
that the next forward pass produces garbage logits. Subsequent
steps then train toward the garbage, and the model spirals out.

Fix: `LR_SHF_ATN=2` (halved). Same final convergence speed, no
divergence. FINAL_TEST scores 10/10.

Classic symptom, classic fix. If a future architecture change
brings the divergence back (different sample distribution, larger
d_model, different vocabulary), try `LR_SHF_ATN=3` before anything
more elaborate. LR decay schedules would also work but are not
implemented (would require per-step state).

Diagnostic signal: if the last REPORT line shows loss > 10× the
second-to-last, divergence has almost certainly occurred. Bail
out early (drop NSTEP to one report before the divergence) or
raise LR_SHF_ATN.

---

## 7. Retargeting to a different task

The binary learns digit reversal because `GENSM` generates random
digits and writes their reverse to TARGET. To retarget:

1. **Modify GENSM.** The current GENSM fills TOKENS[0..7] with 8
   random 0-9 and TARGET[i] = TOKENS[SEQ-1-i]. Replace with the
   new task's sample/target generator.

2. **Consider dimensions.** Current architecture:
   - `SEQ=8` (sequence length)
   - `D=16` (embedding dim)
   - `V=10` (vocab size, because digits 0-9)

   Changing V forces LOGTBL regeneration (`tables/gen_logtbl.py`)
   if V exceeds the table's 257-entry Q12 `-ln(x/256)` range —
   but 257 was chosen because it covers p ∈ [0, 1] at Q8, so
   V up to ~100 is safe.

   Changing D changes `ARCH_SHF` (the softmax-scaling right-shift
   = `log2(sqrt(D))`). For D=64 use ARCH_SHF=3, for D=4 use ARCH_SHF=1.

   Changing SEQ doesn't change anything else (all loops are
   SEQ-parameterized).

3. **Re-tune learning rates.** The current `LR_SHF_*` values were
   chosen for digit reversal at the reference dimensions. A different
   task may need different shifts. Watch for the divergence pattern
   (§6) and raise shifts until it goes away; then see if training
   still converges in NSTEP steps.

4. **FINAL_TEST.** If the task isn't "predict a sequence of tokens
   argmax-matching TARGET", FINAL_TEST's OK/FAIL logic (compare 8
   TE_PRED words to 8 TARGET words) needs to change. The input/
   prediction print formatting assumes single-digit vocab symbols.

5. **Memory.** The Q16/Q8/gradient/cache/workspace region is
   dimensioned for V=10, D=16, SEQ=8. For bigger vocabs or
   dimensions, recompute the memory map in `include/equates.inc`
   and widen BWD_WORK / STACK_TOP accordingly (see PROJECT_ENV.md
   §14 "Binary growth forces memory-map reorgs").

---

## 8. Known failure modes

- **All-zeros output from FINAL_TEST.** Weights wedged at zero.
  Likely cause: ZEROG running after INITW_ALL instead of before,
  or something else resetting weights between init and training.

- **"Same digit repeated" prediction pattern.** Classic collapsed-
  attention failure. Usually means LR_SHF_ATN is too low (LR too
  high) combined with an unlucky early sample. Raise LR_SHF_ATN.

- **Screen garbled with random characters after some REPORT line.**
  Likely cause: PUTC/NEWLINE writing past screen-end without SCROLL.
  Check that SCROLL is still called on bottom-row-hit. Screen
  corruption into code memory can also produce this.

- **MAME stays open indefinitely after training completes.** The
  `t_train.lua` trampoline doesn't call `mach:exit()` — the binary
  ends at `FT_HALT: BRA FT_HALT` and MAME waits for the user.
  Close the window manually. Use `t_traindiag.lua` for automated
  exit.

- **FINAL_TEST 0/10 with healthy REPORT lines.** Divergence between
  the last REPORT and the end of training. See §6.

---

## 9. Useful symbols for poking with the debugger

From `build/main.sym` (rebuild with `--symbol-dump=build/main.sym`):

| Symbol        | Purpose                                    |
|---------------|--------------------------------------------|
| `MAIN`        | Training entry point                       |
| `TRAIN_LOOP`  | Top of per-step loop                       |
| `FINAL_TEST`  | NTEST-sample held-out eval                 |
| `FT_LP`       | Per-sample loop inside FINAL_TEST          |
| `FT_HALT`     | End-of-training freeze                     |
| `REPORT`      | Called on RPRT-boundary steps              |
| `COUNT`       | Per-step argmax/accuracy accumulator       |
| `GENSM`       | Random-sample generator                    |
| `INITW_ALL`   | Startup weight randomization               |
| `ZEROG`       | Startup gradient zeroing                   |
| `CLOSS`       | Cross-entropy loss (invoked by REPORT)     |

All live in `src/train.asm`. `FORWRD`, `BKWRD`, `CVT16_ALL`,
`WUPDT_ALL` are the main composites, each called once per training
step in that order.

---

## 10. Where to look when things break

1. Look at the **final REPORT** line first. If loss is much bigger
   than the second-to-last, it's the divergence pattern (§6).
2. Look at **FINAL_TEST output** next. 10/10 with good REPORT means
   training works. 0/10 after healthy REPORTs is divergence. 6/10
   with healthy REPORTs means training didn't fully converge (raise
   NSTEP, or check that FINAL_TEST is calling CVT16_ALL — it does,
   at `train.asm:1672`).
3. **Garbled screen** → screen-overflow + code corruption. Check
   PUTC/NEWLINE scroll paths.
4. **MAME hangs** → trampoline polling, or the binary crashed into
   wilderness. Use `t_traindiag.lua` to see PC and SP at crash.
5. **Training loop stack check**: after training completes (at
   FT_HALT), SP should be `$6800 - 1` (one byte on the stack from
   the sample counter in FINAL_TEST). The training loop itself
   is stack-balanced — any drift points at a `PSHS` without `PULS`
   somewhere, which would be a real bug.
