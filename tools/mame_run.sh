#!/bin/bash
# mame_run.sh — Run a test harness under MAME coco3h with Lua trampoline.
#
# Usage: tools/mame_run.sh <test_name>
#   e.g.: tools/mame_run.sh t_proj
#
# Prerequisites:
#   - /c/mame/mame.exe (MAME 0.281+, coco3h ROMs)
#   - /c/mame/<test_name>.lua (per-test Lua trampoline with PC-stall polling)
#   - build/<test_name>.bin (assembled binary)
#
# Copies .bin to /c/mame/, runs MAME with -autoboot_script.
# MAME's autoboot callback has a ~30-frame lifetime (~0.53s at NTSC
# 59.94Hz). Our Lua template uses PC-stall polling (detect halt loop)
# to complete within that window. See PROJECT_ENV.md for details.

set -euo pipefail

TEST_NAME="${1:?Usage: tools/mame_run.sh <test_name>}"
BIN_SRC="build/${TEST_NAME}.bin"
LUA_SCRIPT="/c/mame/${TEST_NAME}.lua"
MAME_EXE="/c/mame/mame.exe"

[[ -f "$BIN_SRC" ]] || { echo "ERROR: $BIN_SRC not found. Assemble first."; exit 1; }
[[ -f "$LUA_SCRIPT" ]] || { echo "ERROR: $LUA_SCRIPT not found."; exit 1; }
[[ -f "$MAME_EXE" ]] || { echo "ERROR: $MAME_EXE not found."; exit 1; }

cp "$BIN_SRC" "/c/mame/${TEST_NAME}.bin"

cd /c/mame && timeout 30 "$MAME_EXE" coco3h \
    -autoboot_script "${TEST_NAME}.lua" \
    -skip_gameinfo \
    -nothrottle \
    -window \
    2>&1
