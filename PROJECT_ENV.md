# PROJECT_ENV.md — Environment and invocation reference

Purpose: capture every path, tool location, invocation pattern, and
environment detail that would otherwise be rediscovered at the start of
each session. A new Claude Code session should read this file first.

All values below were verified (via `ls`/`which`/probe) on the date this
file was last updated. Stale values are bugs — see "Maintenance" at the
bottom.

---

## 1. Platform and shell

- **Host OS**: Windows 11 Home 10.0.26100
- **WSL**: default distro, `bash` reachable via `wsl` from Git-Bash.
  Windows paths are mounted under `/mnt/c/...` inside WSL; from Git-Bash
  (MSYS2) they appear as `/c/...`. Both resolve to the same files.
- **Shell used by this project**: Git-Bash (MSYS2) for path convenience
  (`/c/mame/...`, `/c/Projects/...`). LWASM and Python live inside WSL
  and are invoked through `wsl` or `wsl bash -c '...'`.
- **Working-directory convention**: repo is at
  `C:\Projects\cocoai\attn6309\`. From Git-Bash: `/c/Projects/cocoai/attn6309/`.
  From WSL: `/mnt/c/Projects/cocoai/attn6309/`.

## 2. Repository location

- **Windows**: `C:\Projects\cocoai\attn6309\` (verified)
- **Git-Bash**: `/c/Projects/cocoai/attn6309/` (verified)
- **WSL**: `/mnt/c/Projects/cocoai/attn6309/` (verified)
- **Git remote (origin)**: `git@github.com:Jsearle01/ATTN-CC3.git` (SSH, both fetch/push)
- **.gitignore**: excludes `build/`, `*.bin` (except `test/vectors.bin`),
  `*.log`, `mame_out.log`, `__pycache__/`, `probe_*.asm`, `probe_*.bin`,
  `test/vectors.txt`.

## 3. ATTN/11 reference source (read-only)

- **Location**: `C:\Projects\cocoai\.claude\ATTN-11-main\` (verified)
- **Subdirs**: `nn11/`, `proto/`, `model/`, plus top-level `IO.MAC`, `TRAIN.MAC`, `README.md`
- **Key files**:
  - `nn11/LAYER.MAC` — EMBED, PROJ, ATTN reference (PDP-11 source)
  - `nn11/MATOP.MAC` — MVMUL, VTMUL, MVADD, OUTER
  - `nn11/VECOP.MAC` — VDOT, VSCL, VADD
  - `nn11/FXMATH.MAC`, `nn11/ACTFN.MAC`
  - `proto/prototype.shf` — Q8 reference (authoritative for semantics)
  - `proto/prototype-float.shf` — float reference (NOT used for validation)
  - `proto/gen-lut.shf`, `proto/fortran/`

## 4. Assembler (LWASM)

- **Location**: `/usr/local/bin/lwasm` inside WSL (invoked via `wsl lwasm ...`)
- **Version**: `lwasm from lwtools 4.21`
- **Standard invocation** (from repo root, via WSL):
  ```bash
  wsl lwasm --6309 --format=raw --includedir=. --includedir=include \
       -o build/<out>.bin <src>.asm
  ```
- **Diagnostic flags**:
  - `-l build/<name>.lst` — listing with addresses
  - `--symbol-dump=build/<name>.sym` — global symbol dump
  - `-m build/<name>.map` — map file
  - `-s` — inline symbols in listing (requires `-l`)
- **Known LWASM 4.21 quirks** (see `.claude/.../memory/feedback_lwasm_quirks.md`):
  1. Spaces around operators terminate EQU expressions. Write
     `W_EMB+V*D*Q16SZ` (no spaces), not `W_EMB + V*D*Q16SZ`.
  2. `-I <path>` is broken in 4.21; always use `--includedir=<path>`.
  3. `LSRQ` is documented but not implemented; use three `LSRD` or
     `LSRD`+`RORW`.
  4. `SECTION`/`ENDSECT`/`EXTERN` require `--format=obj`. For raw
     binary, compose via `INCLUDE`.

## 5. MAME

- **Executable**: `/c/mame/mame.exe` (verified, 356 MB, dated 2025-09-24)
- **Version**: `MAME v0.281 (mame0281)`
- **Driver**: `coco3h` (HD6309 variant of CoCo3). Driver defaults to
  HD6309; no extra flags needed to force 6309 mode.
- **ROMs**: `/c/mame/roms/coco3.zip` and `/c/mame/roms/coco3h.zip` (verified)
- **Lua trampoline directory**: `/c/mame/` (same dir as mame.exe)
- **Binary staging directory**: `/c/mame/` (assembled `.bin` is copied
  here before run; Lua `BIN_PATH` is filename-relative-to-CWD)
- **Working invocation (verified this session)**:
  ```bash
  cd /c/mame && ./mame.exe coco3h \
      -autoboot_script <test>.lua \
      -skip_gameinfo -nothrottle -window
  ```
- **What does NOT work**: `-debugscript <test>.lua` — MAME launches,
  but the Lua script's `emu.add_machine_frame_notifier` callback does
  not fire in that context (verified: MAME ran 209 emulated seconds
  with no `LOADED ...` print). Use `-autoboot_script`.
- **Critical — autoboot callback lifetime**: MAME's
  `emu.add_machine_frame_notifier` under `-autoboot_script` fires for
  **~32 frames only** (~0.53 s at NTSC 59.94 Hz), then silently stops.
  All Lua trampolines must detect completion and call `mach:exit()`
  within that window. The PC-stall polling pattern (Section 6) handles
  this; never use a fixed `WAIT_FRAMES` > 28.
- **Exit discipline**: the Lua trampoline calls `mach:exit()` after
  printing results, so MAME self-closes. A safety `timeout 30` wrapper
  around the `mame.exe` invocation is recommended to catch harness
  hangs. Example:
  ```bash
  cd /c/mame && timeout 30 ./mame.exe coco3h -autoboot_script t_proj.lua \
      -skip_gameinfo -nothrottle -window 2>&1
  ```

## 6. Lua trampoline pattern

Each test has a dedicated `.lua` in `/c/mame/` that loads the `.bin`,
jumps to `$0600`, polls for completion, reads scratch RAM, and prints
results.

**PC-stall polling template** (used by t_vcpycl, t_vsadd, t_embed,
t_proj — all four active harnesses). Detects harness completion by
checking if PC is unchanged between two consecutive frame callbacks
(harness sits in `HALT: BRA HALT`). Falls back to a safety timeout
at frame 28 (inside the ~32-frame autoboot callback lifetime).

Key parameters per trampoline (only these differ between files):
- `BIN_PATH` — filename of `.bin` to load
- `SCRATCH_BASE` — currently `0x2000` for all four active harnesses
- `TEST_NAMES` — 1-based Lua table of per-test labels
- `BITS_WIDTH` — `8` for ≤8 tests, `16` for >8 (e.g. t_vsadd)
- `LAST_FAIL_ADDR` — `SCRATCH_BASE+5` when RESULT_BITS is 1 byte,
  `SCRATCH_BASE+6` when 2 bytes

Design properties:
- `pcall()` wraps result-reading so Lua exceptions are printed, not
  silently swallowed
- `io.flush()` after every `print()` — prevents stdout full-buffering
  from eating output on MAME exit
- `MIN_POLL_FRAME = 2` — skips initial frames to avoid false-positive
  PC-stall during CPU startup
- `MAX_FRAME = 28` — safety timeout, always inside the autoboot window

Reference implementation: `/c/mame/t_vcpycl.lua` (simplest, 8 tests).

Older trampolines (t_fxmath, t_vecop, t_matop, t_vmax, t_fxdiv,
t_exptbl, t_sftmx) use a different fixed-frame-count pattern with
32-bit bitmap iteration. They still use `$1700` scratch. Not yet
migrated — they work but lack the PC-stall robustness.

**Creating a new trampoline**: copy `t_vcpycl.lua`, change `BIN_PATH`,
`TEST_NAMES`, `NUM_TESTS`, `BITS_WIDTH`, and `LAST_FAIL_ADDR` if
RESULT_BITS is 2 bytes.

**Scratch RAM layout** (active harnesses, `SCRATCH_BASE = $2000`):

- `$2000` — PASS_CNT (u16)
- `$2002` — FAIL_CNT (u16)
- `$2004` — RESULT_BITS (u8 or u16 depending on test count)
- `$2005` — LAST_FAIL_ID (u8) when RESULT_BITS is 1 byte
- `$2006` — LAST_FAIL_ID (u8) when RESULT_BITS is 2 bytes (t_vsadd)
- `$2006`/`$2007` — CUR_TEST_ID (u8)

Non-migrated harnesses still read `$1700`-based addresses.

## 7. Test execution workflow

```bash
# 1. Assemble (from repo root, via WSL)
cd /c/Projects/cocoai/attn6309 && \
wsl lwasm --6309 --format=raw --includedir=. --includedir=include \
     -o build/t_proj.bin test/t_proj.asm

# 2. Run via mame_run.sh (copies bin, launches MAME, prints results)
tools/mame_run.sh t_proj

# Or manually:
# cp build/t_proj.bin /c/mame/t_proj.bin
# cd /c/mame && timeout 30 ./mame.exe coco3h \
#     -autoboot_script t_proj.lua \
#     -skip_gameinfo -nothrottle -window 2>&1
```

**Stdout buffering caveat**: MAME's stdout is line-buffered to a
terminal but full-buffered to a pipe/file. If you redirect through
`| head` and MAME crashes (rather than exiting cleanly via
`mach:exit()`), the buffered Lua prints are lost. Prefer `2>&1` to a
file or straight to stdout; use `head` only after MAME has exited.

## 8. Python

- **Location (WSL)**: `/usr/bin/python3` (Python 3.12.3) — use `python3`
- **Used for**: reference generators (`tables/gen_*_vectors.py`),
  `tools/checkvals.py` (stub)
- **Invocation pattern**:
  ```bash
  wsl bash -c 'cd /mnt/c/Projects/cocoai/attn6309 && python3 tables/gen_proj_vectors.py'
  ```
- **Probe-file convention** (gitignored): `probe_*.py` / `probe_*.asm`

## 9. Test inventory

Values marked `(check)` are unverified; update on first successful run.

| Harness  | Lua trampoline   | Count | Expected RESULT_BITS | INCLUDE graph                                           |
|----------|------------------|-------|----------------------|---------------------------------------------------------|
| t_fxmath | t_fxmath.lua     | 64    | (check)              | fxmath.asm                                              |
| t_vecop  | t_vecop.lua      | 18    | (check)              | fxmath.asm, vecop.asm                                   |
| t_vmax   | t_vmax.lua       | 8     | (check)              | fxmath.asm, vecop.asm                                   |
| t_vcpycl | t_vcpycl.lua     | 8     | $FF                  | fxmath.asm, vecop.asm                                   |
| t_vsadd  | t_vsadd.lua      | 10    | $03FF                | fxmath.asm, vecop.asm                                   |
| t_fxdiv  | t_fxdiv.lua      | 10    | (check)              | fxmath.asm                                              |
| t_exptbl | t_exptbl.lua     | 5     | (check)              | tables/exptbl.asm                                       |
| t_sftmx  | t_sftmx.lua      | 10    | (check)              | fxmath.asm, vecop.asm, actfn.asm                        |
| t_matop  | t_matop.lua      | 21    | (check)              | fxmath.asm, vecop.asm, matop.asm                        |
| t_embed  | t_embed.lua      | 6     | $3F                  | fxmath.asm, vecop.asm, matop.asm, actfn.asm, layer.asm  |
| t_proj   | t_proj.lua       | 5     | $1F                  | fxmath.asm, vecop.asm, matop.asm, actfn.asm, layer.asm  |
| t_attn   | (not yet)        | (tbd) | (tbd)                | fxmath.asm, vecop.asm, matop.asm, actfn.asm, layer.asm  |

Diagnostic variant: `t_proj_diag.lua` loads `t_proj.bin` and dumps
OUT_BUF / scratch memory rather than parsing pass/fail. Not in the
regression set; used for post-mortem inspection only.

## 10. Memory map quick reference

From `include/equates.inc`. Confirm against that file if anything here
looks wrong — equates are authoritative.

```
$0000-$01FF    VEC_BASE          Interrupt vectors (512 B)
$0200-$03FF    CODE_BASE         Code + tables (5.3 KB allocation)
$0400-$05FF    SCREEN            Hardware-mapped text, 32×16 (512 B)
$0600          (harness ORG)     Test harness code loads here
~$0B00-$1720   (binary tail)     End of harness binary (varies by INCLUDE depth)
$1800-$18C7    STR_BASE          String literals (200 B, production only)
$1900-$191F    TOK_BASE          Tokens + target (32 B, production only)
$1A00-$2BFF    WEIGHT_BASE       Q16 weight accumulators (4.8 KB, production only)
$2000-$20FF    Test scratch      SCRATCH_BASE: PASS_CNT, FAIL_CNT, RESULT_BITS, ...
$2030-$203F    Decimal scratch   DEC_BUF, DEC_DTEMP, DEC_SCRPTR
$2060-$23FF    Test buffers      DST_BUF, SRC_BUF, etc. (varies per harness)
$2C00-$37FF    GRAD_BASE         Gradient accumulators (2.4 KB, production only)
$3800-$3DFF    FWD_CACHE         Forward activations (1.5 KB)
$3E00-$47FF    BWD_WORK          Backward workspace (2 KB, holds MP_*/SF_*/EM_*/PR_*/AT_*)
$4800          STACK_TOP         Stack grows down from here
$4800-$B7FF    FREE_BASE         ~26 KB free
```

**Scratch at $2000**: active test harnesses (t_vcpycl, t_vsadd,
t_embed, t_proj) use `SCRATCH_BASE EQU $2000`. This reclaims the
WEIGHT_BASE address space, which is not active during test execution.
Non-migrated harnesses (t_fxmath, t_vecop, etc.) still use `$1700`.

**Historical note**: scratch was originally at `$1700`. When t_embed's
binary grew to 4379 B (spanning $0600-$171A) from the actfn.asm
INCLUDE addition, static vector data at $1700-$171A was clobbered by
the harness RUN_TESTS init. Migrating to $2000 provides ~741 B of
clearance above the largest binary (t_embed).

## 11. Known LWASM / 6309 instruction status

**Confirmed working** (assemble under `--6309`, execute correctly on
MAME `coco3h`):
- `ASRD` (0x1047)
- `LDE` (page-1 prefix 0x11, various addressing modes; immediate 0x11 86)
- `DECE` (0x11 4A)
- `LSLD`, `LSRD`
- `MULD` (16×16 → 32 in D:W)
- `DIVQ`
- `TFR W,D`, `TFR D,W`
- `LBRA`, `LBNE`, `LBEQ`
- `TFM X+,U+` (block-move)
- `ORD`
- `ADDR W,W` (noted in DEVIATIONS.md)

**Confirmed absent / rejected**:
- `LSLW` (not a real 6309 opcode)
- `LSRQ` (documented elsewhere, not implemented in LWASM 4.21)

**Unprobed but likely-present** (not yet exercised in the test harnesses):
- `LDF`, `DECF` (symmetric to LDE/DECE)
- `ADCD`, `SBCD`
- `AIM`, `OIM`, `EIM`, `TIM` (6309 memory-bit ops)

When introducing a new instruction, probe first — write a 4-line
program to `/tmp/probe_<mnemonic>.asm`, assemble with LWASM, and
inspect bytes with `xxd`. Record the result in this section.

## 12. DEVIATIONS

`DEVIATIONS.md` in the repo root catalogs intentional departures from
the ATTN/11 reference (clamping vs wrapping, guard additions, workspace
layout) and lessons learned during validation. Read it before making a
judgment call about whether a behavior difference is a bug or a
sanctioned divergence.

Persistent lessons also live in auto-memory at
`C:\Users\jayse\.claude\projects\c--Projects-cocoai\memory\`:
- `feedback_lwasm_quirks.md` — LWASM parsing quirks
- `feedback_6309_d_preservation.md` — D-register preservation across small loops

## Maintenance

- When any path, version, or invocation changes, update this file in
  the same commit as the change.
- When a new harness is added, add its row to section 9.
- When a new instruction is probed (per section 11), record the
  result in the same commit.
- When the MAME command line flags are adjusted, re-verify section 5
  with a t_vcpycl probe and update the working invocation.
