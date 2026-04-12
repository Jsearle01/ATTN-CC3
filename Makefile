# ============================================================
# Makefile — ATTN/11 → 6309 build rules (LWASM + test runner)
# ============================================================
# Dependencies: lwasm, python3, mame (for integration tests)
#
# TODO: uncomment rules once test harness exists
# ============================================================

LWASM   = lwasm
PYTHON  = python3
BUILD   = build
SRC     = src
TEST    = test
INCLUDE = include
TABLES  = tables

# Assembler flags
# ASMFLAGS = --6309 --format=raw --includedir=$(INCLUDE)

# ---- Main binary ----
# $(BUILD)/attn6309.bin: $(SRC)/main.asm $(SRC)/*.asm $(INCLUDE)/*.inc
# 	$(LWASM) $(ASMFLAGS) -o $@ $<

# ---- Reference vectors ----
# $(TEST)/vectors.bin: $(TABLES)/ref_attn11.py
# 	cd $(TABLES) && $(PYTHON) ref_attn11.py

# ---- Lookup tables ----
# $(TABLES)/exptbl.bin: $(TABLES)/gen_exptbl.py
# 	$(PYTHON) $<

# $(TABLES)/logtbl.bin: $(TABLES)/gen_logtbl.py
# 	$(PYTHON) $<

# ---- Test binaries ----
# $(BUILD)/t_fxmath.bin: $(TEST)/t_fxmath.asm $(SRC)/fxmath.asm $(TEST)/runner.asm
# 	$(LWASM) $(ASMFLAGS) -o $@ $<

# $(BUILD)/t_vecop.bin: $(TEST)/t_vecop.asm $(SRC)/vecop.asm $(SRC)/fxmath.asm $(TEST)/runner.asm
# 	$(LWASM) $(ASMFLAGS) -o $@ $<

# $(BUILD)/t_matop.bin: $(TEST)/t_matop.asm $(SRC)/matop.asm $(SRC)/vecop.asm $(SRC)/fxmath.asm $(TEST)/runner.asm
# 	$(LWASM) $(ASMFLAGS) -o $@ $<

# $(BUILD)/t_actfn.bin: $(TEST)/t_actfn.asm $(SRC)/actfn.asm $(SRC)/fxmath.asm $(SRC)/tables.asm $(TEST)/runner.asm
# 	$(LWASM) $(ASMFLAGS) -o $@ $<

# ---- Test targets ----
# test-fxmath: $(BUILD)/t_fxmath.bin
# 	tools/mame_run.sh $<

# test-vecop: $(BUILD)/t_vecop.bin
# 	tools/mame_run.sh $<

# test-matop: $(BUILD)/t_matop.bin
# 	tools/mame_run.sh $<

# test-actfn: $(BUILD)/t_actfn.bin
# 	tools/mame_run.sh $<

# test: test-fxmath test-vecop test-matop test-actfn

# ---- Validation against golden reference ----
# validate: $(BUILD)/attn6309.bin
# 	tools/mame_run.sh $< 600 > $(BUILD)/screen_dump.txt
# 	$(PYTHON) tools/checkvals.py $(BUILD)/screen_dump.txt

# ---- Cleanup ----
# clean:
# 	rm -f $(BUILD)/*.bin $(TEST)/vectors.bin $(TEST)/vectors.txt
# 	rm -f $(TABLES)/exptbl.bin $(TABLES)/logtbl.bin

.PHONY: test test-fxmath test-vecop test-matop test-actfn validate clean
