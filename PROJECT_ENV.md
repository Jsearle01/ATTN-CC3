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
jumps to `$0600`, waits N frames, reads scratch RAM, and prints results.

**Modern template** (used by t_vcpycl, t_vsadd, t_embed, t_proj):

```lua
-- t_NAME.lua
local BIN_PATH = "t_NAME.bin"
local BIN_ADDR = 0x0600
local STAGE = "wait"
local load_frame = nil
local frame_count = 0

local TEST_NAMES = {
    [0] = "TEST_0_LABEL",
    [1] = "TEST_1_LABEL",
    -- ...
}

emu.add_machine_frame_notifier(function()
    local mach = manager.machine
    if not mach then return end
    local cpu = mach.devices[":maincpu"]
    if not cpu then return end
    local mem = cpu.spaces["program"]
    if not mem then return end
    frame_count = frame_count + 1

    if STAGE == "wait" then
        if frame_count < 3 then return end                -- let BASIC settle
        local f = io.open(BIN_PATH, "rb")
        if not f then print("FAIL: no " .. BIN_PATH); mach:exit(); return end
        local data = f:read("*all"); f:close()
        for i = 1, #data do mem:write_u8(BIN_ADDR + i - 1, string.byte(data, i)) end
        print(string.format("LOADED %d bytes", #data))
        cpu.state["PC"].value = BIN_ADDR
        load_frame = frame_count
        STAGE = "running"
        return
    end

    if STAGE == "running" then
        if (frame_count - load_frame) < WAIT_FRAMES then return end
        local function u16(a) return mem:read_u8(a) * 256 + mem:read_u8(a+1) end
        local pc = cpu.state["PC"].value
        local pass_cnt = u16(0x1700)
        local fail_cnt = u16(0x1702)
        local bits = mem:read_u8(0x1704)
        local last_fail = mem:read_u8(0x1705)
        print(string.format("PC=$%04X", pc))
        print(string.format("PASS_CNT=%d FAIL_CNT=%d", pass_cnt, fail_cnt))
        print(string.format("RESULT_BITS=$%02X LAST_FAIL_ID=$%02X", bits, last_fail))
        print("")
        for i = 0, N-1 do
            local bit_set = (bits >> i) & 1
            local status = bit_set == 1 and "PASS" or "FAIL"
            print(string.format("TEST_%d (%s): %s", i, TEST_NAMES[i], status))
        end
        print(string.format("Total: %d/N pass", pass_cnt))
        STAGE = "done"
        mach:exit()
    end
end)
```

**WAIT_FRAMES tuning** — how many frames after PC-set before reading results:

| Harness  | WAIT_FRAMES (in Lua) | Notes                                  |
|----------|----------------------|----------------------------------------|
| t_vcpycl | 5                    | tiny, primitive tests                  |
| t_proj   | 30                   | heavy VTMUL; may still be insufficient |
| t_embed  | (verify)             | 6 tests, moderate                      |
| t_vsadd  | (verify)             | 10 tests, light                        |

Older trampolines (t_fxmath, t_vecop, t_matop, t_vmax, t_fxdiv,
t_exptbl, t_sftmx) use a different 32-bit bitmap pattern and iterate
`for i = 0, 31 do`. Count lives in the harness, reported as
`TOTAL: PASS=N FAIL=M (of T)` by the Lua.

**Creating a new trampoline**: copy the nearest-matching existing file
(e.g. `t_proj.lua` for a VTMUL-heavy test), change `BIN_PATH`,
`TEST_NAMES`, the `for i = 0, N-1` bound, the `Total: %d/N` string,
and tune `WAIT_FRAMES` for the test's compute weight.

**Scratch RAM read addresses** (modern pattern, from `include/equates.inc`):

- `$1700` — PASS_CNT (u16)
- `$1702` — FAIL_CNT (u16)
- `$1704` — RESULT_BITS (u8 for ≤8 tests; u16 for ≤16 like t_vsadd)
- `$1705` — LAST_FAIL_ID (u8 when bits is u8) / `$1706` (u8 when bits is u16, e.g. t_vsadd)
- `$1707` — CUR_TEST_ID (u8)

## 7. Test execution workflow

```bash
# 1. Assemble (from repo root, via WSL)
cd /c/Projects/cocoai/attn6309 && \
wsl lwasm --6309 --format=raw --includedir=. --includedir=include \
     -o build/t_proj.bin test/t_proj.asm

# 2. Stage binary in MAME directory
cp build/t_proj.bin /c/mame/t_proj.bin

# 3. Run MAME with Lua trampoline (timeout wrapper catches hangs)
cd /c/mame && timeout 30 ./mame.exe coco3h \
    -autoboot_script t_proj.lua \
    -skip_gameinfo -nothrottle -window 2>&1

# 4. Parse stdout for the PASS_CNT / FAIL_CNT / RESULT_BITS lines.
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
$1700-$170F    Test scratch      PASS_CNT, FAIL_CNT, RESULT_BITS, ...
$1730-$173F    Decimal scratch   DEC_BUF, DEC_DTEMP, DEC_SCRPTR
$1760-$17FF    Test buffers      DST_BUF, SRC_BUF (varies per harness)
$1800-$18C7    STR_BASE          String literals (200 B)
$1900-$191F    TOK_BASE          Tokens + target (32 B)
$1A00-$2BFF    WEIGHT_BASE       Q16 weight accumulators (4.8 KB)
$2C00-$37FF    GRAD_BASE         Gradient accumulators (2.4 KB)
$3800-$3DFF    FWD_CACHE         Forward activations (1.5 KB)
$3E00-$47FF    BWD_WORK          Backward workspace (2 KB, holds MP_*/SF_*/EM_*/PR_*/AT_*)
$4800          STACK_TOP         Stack grows down from here
$4800-$B7FF    FREE_BASE         ~26 KB free
```

**Collision warning**: harness `.bin` images ORG'd at `$0600` can grow
past `$1700` and physically overlap the test scratch. The harness
zeros `$1700..$1707` at `RUN_TESTS` entry, so any static INCLUDE'd
data that landed in that range is clobbered. Check binary size before
dismissing mysterious single-test failures.

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
