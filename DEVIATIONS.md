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
