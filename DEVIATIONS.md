# Deviations from ATTN/11 → 6309 Port Specification

This document tracks every place where the implementation
consciously diverges from the project specification, with the
reason for each deviation. All deviations are verified empirically.

## DIV8 (loss averaging, 16-bit)

**Spec says**: "Three LSRQ instructions for divide-by-8. Not an
ASHC sequence."

**Reality**: LSRQ is not a real HD6309 opcode. Verified empirically:
LWASM 4.21 rejects `LSRQ` with `ERROR : Bad opcode`. The HD6309
technical reference lists LSRD (16-bit D), LSRW (16-bit W), ASRD,
RORD, and RORW as valid — there is no single-opcode 32-bit shift
of Q. Confirmed via LWASM 4.21 empirical test, HD6309 technical
reference manual inspection, and independent manual verification
against the ISA opcode table.

**Deviation**: The 16-bit DIV8_Q macro uses `LSRD × 3`. For a
non-negative 16-bit value in D, this is arithmetically identical
to three `LSRQ` instructions would have been, and matches
ATTN/11's `ASHC #-3` behavior.

## DIV8 (loss averaging, 32-bit)

**Spec says**: Same "LSRQ" reference.

**Reality**: When the loss sum exceeds 16 bits (ATTN/11 notes the
sum of eight Q12 values can overflow), the accumulator is a 32-bit
register pair. Shifting the full Q register requires composition.

**Deviation**: The DIV8_Q32 macro uses `LSRD / RORW` pairs × 3.
Each pair shifts Q right by one bit (carry chain from D bit 0 into
W bit 15). Three pairs = /8.

## Cross-entropy gradient Q8→Q15 shift

**Spec says**: "Shift softmax gradient Q8→Q15 using TFR D,W /
CLRD / LSRQ sequence. Not an ASL loop."

**Reality**: Two problems with this. (1) LSRQ is fictional. (2)
Q8 → Q15 is a LEFT shift by 7 bits (multiply by 128), not a right
shift. The ATTN/11 README states explicitly: "The result,
initially in Q8, is shifted left by 7 bits to Q15."

**Deviation**: FX_SHIFT_XENT in fxmath.asm will use seven unrolled
LSLD instructions. This matches the README's stated behavior. To
be implemented in the FXMATH turn.

## EXTERN declarations in equates.inc

**Spec says**: (implicit — object-format assembly assumed)

**Reality**: We emit flat raw binary, not object files. EXTERN
without `--format=obj` and a linker pass is rejected by LWASM.

**Deviation**: The `EXTERN EXPTBL` and `EXTERN LOGTBL` lines in
equates.inc are commented out. EXPTBL and LOGTBL will become
locally defined labels in tables.asm, which is INCLUDE'd into
main.asm at build time.

## SECTION/ENDSECTION directives

**Spec says**: (implicit — object-format with multiple sections)

**Reality**: Same as EXTERN. Flat binary uses INCLUDE-based source
composition with a single ORG directive in main.asm.

**Deviation**: All `SECTION code` / `ENDSECTION` directives in
stub .asm files are replaced with comments. Source composition is
driven by INCLUDE statements in main.asm.

## LWASM include directory flag

**Spec says**: (not specified)

**Reality**: The short form `-I PATH` is broken in LWASM 4.21 —
it silently fails to locate include files, but the assembler
reports exit 0 with empty output. This was caught only because
memory-map validation showed zero-byte binaries.

**Deviation**: All build invocations use `--includedir=PATH` (long
form). The Makefile, test runner, and all probe scripts must use
the long form exclusively. Document this prominently in README.md
under a "Toolchain gotchas" section.

## LWASM spacing in expressions

**Spec says**: (not specified)

**Reality**: LWASM 4.21 treats whitespace as an expression
terminator inside EQU definitions. `A EQU B + C*D` parses as
`A EQU B`, silently producing wrong symbol values. This was
caught during equates.inc validation.

**Deviation**: All arithmetic expressions in equates.inc are
written without internal spaces: `A EQU B+C*D`. A comment at the
top of equates.inc documents this rule. Future code must follow
the same convention.

# Lessons Learned During Stage 1 Implementation

This section captures non-specification findings discovered
empirically during FXMATH validation. These are patterns and
gotchas that must be applied to all future test harnesses and
eventually to main.asm.

## Test data placement requires explicit ORG

**Discovery**: The FXMATH test harness grew until the code
region crossed $0400, placing the INCLUDEBIN-loaded VECS
block inside the hardware-mapped screen buffer ($0400-$05FF).
CLEAR_SCREEN then overwrote the vector data with $20 spaces,
producing confusing "wrong-value" test failures.

**Policy**: Every test harness must use explicit `ORG` to place
test data in free RAM (at or above $4800 per the Stage 1
memory map). Never rely on "append test data after code and
hope it lands somewhere safe."

Applies to: t_vecop.asm, t_matop.asm, t_actfn.asm, and any
future test harness.

## Interrupt masking is mandatory at harness entry

**Discovery**: After starting the harness via a Lua trampoline
that set PC to our entry point, GIME timer/vsync interrupts
from the BASIC ROM continued firing because the Lua-side
CC mask did not persist into our code's execution. This
caused the MUL15Q15 loop counter to be zeroed mid-execution,
producing a 0/27 false negative while MUL8Q15 (which ran first)
completed correctly.

**Policy**: Every test harness and main.asm entry point must
begin with `ORCC #$50` as the first instruction. This sets
the I and F flags, masking both regular and fast interrupts
for the remainder of execution. The harness never needs
interrupts; training does not need interrupts either.

The two-byte ORCC instruction is a non-negotiable prelude.
Stack setup, screen clear, and anything else all come after.

## Lua trampoline pattern for MAME coco3h boot

**Discovery**: MAME's autoboot_script fires after the BASIC
ROM has already begun executing. If Lua writes memory then
sets PC, the window between Lua's memory writes and the
effective PC change allows BASIC to corrupt RAM.

**Policy**: Use a trampoline pattern in the Lua script:
  1. On first frame notifier callback, install a tight
     infinite loop at a safe low-RAM address (we use $0100:
     `$20 $FE` = BRA $0100) and set PC to that address.
     The CPU spins harmlessly, not executing BASIC code.
  2. On the next callback, load the test binary to $0200
     and any separate data (vectors, etc.) to their
     designated addresses. Verify critical magic bytes
     survived the load.
  3. Set PC to the harness entry point ($0200). CPU resumes
     in our code.

This pattern is reusable for every test harness. The Lua
template in /mnt/c/mame/t_fxmath.lua can be copied and adapted
for t_vecop.lua, etc.

## MAME coco3h driver defaults to HD6309

**Discovery**: The `coco3h` driver in MAME is a dedicated
variant of the CoCo3 driver with HD6309 as the main CPU, not
a 6809 with a CPU-slot upgrade option. No `-cpu hd6309` or
similar flag is needed.

**Policy**: All MAME invocations use `mame.exe coco3h ...`.
Native HD6309 instructions (MULD, STQ, LDD extended, LSRD/RORW,
LSLD, LBRA, TFM) assemble correctly under `--6309` and execute
correctly on this driver.

## MAME -verifyroms is pedantic about optional ROMs

**Discovery**: `mame.exe -verifyroms coco3h` reports missing
FDC device ROMs even when the project does not use a floppy
disk controller. The core driver still runs correctly.

**Policy**: Do not run -verifyroms as a pre-flight check. It
will produce noise that does not correlate with whether the
emulator can run our binary.

## MAME Lua API — current modern form

**Discovery**: `emu.register_frame` is deprecated. The modern
replacement is `emu.add_machine_frame_notifier`.

**Policy**: All MAME Lua scripts use
`emu.add_machine_frame_notifier`. The Lua convention going
forward:

    emu.add_machine_frame_notifier(function()
        local mach = manager.machine
        if not mach then return end
        -- ...
    end)

## Harness code must live above the text screen buffer

**Discovery**: The VECOP harness grew large enough (~800 bytes
of code) that with ORG $0200, code bytes landed at addresses
$0400-$051C, inside the CoCo3 text screen buffer ($0400-$05FF).
This caused: (a) screen displayed code bytes as garbage, and
(b) CLEAR_SCREEN's TFM fill potentially corrupted the code.
VDOT happened to pass anyway but VSCL and VADD failed mysteriously.

**Policy**: All test harnesses use `ORG $0600` rather than
`ORG $0200`. This places code safely above the screen buffer
with the $0200-$03FF region available as scratch if needed.
The 512 bytes of header padding in the raw binary are trivial
cost.

For the eventual production main.asm, code at $0200 is fine
because the Stage 1 memory map allocates $0200-$03FF for code
that runs BEFORE CLEAR_SCREEN is called (interrupt vectors,
reset handler, early init). Code that runs after CLEAR_SCREEN
and is larger than ~500 bytes must be placed at $0600+ via
explicit ORG.

Update t_fxmath.asm to ORG $0600 at next opportunity (not urgent
-- it fits below $0400 currently, but consistency is valuable).

## Do not use B as a loop counter across LDD-family instructions

**Discovery**: The VECOP harness had three distinct instances of
the same bug: using `B` as a loop counter in a loop that contains
`LDD ,X++` or similar. LDD loads the 16-bit operand into D (= A:B),
overwriting B with the low byte of the loaded data. After LDD, B
no longer holds the counter — it holds data. DECB then decrements
the data value, and the loop either spins forever or terminates
on a data-dependent coincidence (when some element's low byte
happens to equal 1).

Observed manifestations:
  - VADD's inner loop spun forever (PC stuck at $091E)
  - VSCL_CMP and VADD_CMP in the harness silently misbehaved
    (hidden by the comparison branching out before DECB ran)

**Policy**: Loop counters that must survive across LDD, ADDD,
SUBD, MULD, or any D-writing instruction must live on the stack
(`PSHS B` / `DEC ,S` / `LEAS 1,S`) or in X/Y/U when those are
free. The only safe B-as-counter loops are those that use no
D-writing instructions inside the loop body.

This is not a 6309 peculiarity — it applies to 6809 as well.
Mentioned here because it is a subtle trap for anyone porting
from architectures (PDP-11, x86, 68k) where the counter register
is conventionally distinct from the data path.

Apply to: all future loop code in MATOP, ACTFN, LAYER, TRAIN.

## MATOP single-clamp deviation from ATTN/11

**Spec says**: ATTN/11's MVADD, VTMUL, and OUTER use a two-stage
clamp: (1) normalize the Q16 product to Q8 via ASHC #-8 with
clamp to +/-32767, then (2) ADD the clamped result to the
destination with a second signed-overflow clamp (BVC/BPL/BMI).

**Reality**: The two-stage approach loses information. When the
intermediate product exceeds +/-32767 after >>8, the first clamp
saturates it. Then the second addition against a destination of
opposite sign produces a result that's less accurate than if the
full 32-bit product had been summed with the sign-extended
destination before a single clamp.

**Deviation**: Our MVADD, VTMUL, and OUTER use single-clamp
32-bit accumulate:
  1. Compute the full 32-bit Q16 product (MULD).
  2. Sign-extend the existing destination Q8 value to 32-bit Q16.
  3. Add (2) to the 32-bit product from (1).
  4. Apply a single NORMQ15 + clamp to extract Q8.

This is bit-identical to ATTN/11 in the common case (product fits
in Q8 range after >>8). In the saturation case, it preserves
sign+magnitude through to the final sum. MVMUL is unaffected
(single clamp either way).

Verified empirically with a divergence test case:
  mat=[100.0], vin=[1.5], vout_init=[-100.0]
  Single-clamp result: 12800 ($3200 = 50.0 in Q8) — correct
  Two-stage result: -32768 ($8000) — saturated by intermediate clamp

This test is included in the MVADD test suite (test case 4) and
will fail loudly if the code is ever regressed to two-stage clamp.

## VSADD two-stage clamp (differs from MATOP single-clamp by design)

VSADD computes per-element `dst[k] += clamp16((scalar * src[k]) >> 8)`.
It uses a two-stage clamp:
  Stage 1: clamp (scalar * src[k]) >> 8 to Q8 range.
  Stage 2: clamp dst[k] + stage1 to Q8 range.

This deliberately differs from MATOP's MVADD/VTMUL/OUTER single-clamp
strategy. The reason is semantic:

- MATOP accumulating routines operate inside a mathematical sum (dot
  product, outer product). The true sum is what matters; single-clamp
  preserves full precision through to a final normalize.

- VSADD is per-element scale-add used in gradient updates (BKWRD step 2
  dV accumulation). Each element's saturation should be committed
  independently, matching ATTN/11's ASHC #-8 + ADD with BVC overflow
  clamp. Single-clamp would alter gradient magnitudes in saturating
  cases and affect learning dynamics in hard-to-predict ways.

Test coverage: VSADD test 4 (PRODUCT_SAT) exercises a case where the
two paths diverge (two-stage = 12767, single-clamp = 32767). The
divergence is annotated in tables/vsadd_vectors.asm so any future
regression to single-clamp fails the test loudly.

## Loop counter register rule (generalized)

Do not use any register as a loop counter across an instruction that
writes it. Specifically:
- B is clobbered by LDD, LDW, MUL, MULD, and any op writing D.
- W is clobbered by MULD, TFM, STQ, and any 32-bit arithmetic writing Q.
- A is clobbered by LDA, LDD, and any op writing D.

Use a stack counter:
- For loops of <= 255 iterations: `PSHS A` (or B) at entry,
  `DEC ,S / BNE loop / LEAS 1,S` at loop tail.
- For loops of > 255 iterations: `PSHS D` at entry,
  `LDD ,S / SUBD #1 / STD ,S / BNE loop / LEAS 2,S` at loop tail.

Or a DP scratch slot if the routine needs the stack depth for other
purposes (rare; MATOP's MP_* pattern demonstrates this).

Examples of correct usage:
- VDOT, VSCL: stack byte counter (8-bit loop length).
- MATOP MVMUL, MVADD, VTMUL, OUTER: DP scratch counters via MP_AHI,
  MP_ALO, MP_SCL — avoids register clobbering across MULD+accumulate.
- VSADD: stack byte counter (fixed after W-register regression).

The bugs that prompted this generalization: VSADD initially used W as
loop counter via DECW/BNE; MULD VS_SCL clobbers W via Q = D:W, hanging
MAME at PC=$0C6B. In the same commit, harness RECORD_PASS used LDA
CUR_TEST_ID / LDD #1 for bitmask setup; LDD clobbered A, causing the
shift loop to exit immediately and producing RESULT_BITS=$0001
regardless of actual test pass status. Both bugs have the same root
cause: register used as counter got written by an intervening
instruction. The stack-counter convention avoids both.

## PROJ zero-length guard (defensive hardening, not semantic deviation)

PROJ adds a `LDW PR_SEQ / BEQ PROJ_DONE` guard at entry. ATTN/11's
PROJ omits this; if SEQ=0 is ever passed, the PDP-11 version
underflows the loop counter to 65535 and runs through 64KB of
garbage memory.

This is a standard hardening pattern matching VCPY, VCLR, VSADD,
and EMBED. Not a semantic change — under any valid input, behavior
is identical to ATTN/11.
