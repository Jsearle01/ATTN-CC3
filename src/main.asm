* ============================================================
* main.asm — ATTN-CC3 training binary
* ============================================================
* Trains a single-head attention transformer on digit reversal.
* Architecture: SEQ=8 positions, DIM=16, VOC=10. NSTEP=350
* SGD steps, REPORT every RPRT=50. FINAL_TEST runs NTEST=10.
* Fixed-point throughout: Q8 activations, Q15 gradients, Q16
* weight accumulators (split hi/lo per ATTN/11 layout).
* ============================================================

                ORG     $0600
                JMP     MAIN

* --- Include all primitives, composites, and training routines ---
                INCLUDE include/equates.inc
                INCLUDE include/macros.inc
                INCLUDE include/fxmath.inc
                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm
                INCLUDE src/matop.asm
                INCLUDE src/actfn.asm
                INCLUDE src/layer.asm
                INCLUDE src/train.asm

* ============================================================
* MAIN — training entry point
* ============================================================
MAIN:
                ORCC    #$50            ; disable IRQ/FIRQ
                LDMD    #1              ; 6309 native mode
                LDS     #STACK_TOP      ; stack at $6800

* Clear screen and print banner FIRST so the user sees text even
* while the ~5-frame INITW_ALL is still running.
                JSR     CLEAR_SCREEN
                LDX     #STR_HEADER
                JSR     PUTS
                JSR     NEWLINE

* Initialise PRNG seed
                LDD     #RN_INIT
                STD     RN_SED

* Initialise weights (random Q8 -> Q16 hi/lo, 6 groups)
                JSR     INITW_ALL

* One-time gradient zeroing (before first BKWRD). ZEROG uses
* the BK_D* fields, so BKWRD_SETUP must run first.
                JSR     BKWRD_SETUP
                JSR     ZEROG

* Initialise training state (1-indexed step counter, matches ATTN/11)
                LDD     #1
                STD     TR_STEP
                CLRA
                CLRB
                STD     TR_HIT
                STD     TR_TOT

* ============================================================
* Training loop — per-step: GENSM, CVT16_ALL, FORWRD, BKWRD,
* WUPDT_ALL, COUNT, (conditional REPORT).
* CLOSS is called only inside REPORT (matches ATTN/11).
* ============================================================
TRAIN_LOOP:
                JSR     GENSM
                JSR     CVT16_ALL
                JSR     FORWRD
                JSR     BKWRD
                JSR     WUPDT_ALL
                JSR     COUNT

                LDD     TR_STEP
                TFR     D,W
                LDD     #0
                DIVQ    MAIN_RPRT
                CMPD    #0
                BNE     TR_NORPT
                JSR     REPORT
TR_NORPT:
                LDD     TR_STEP
                ADDD    #1
                STD     TR_STEP
                CMPD    #NSTEP+1
                LBNE    TRAIN_LOOP

* FINAL_TEST: run NTEST samples forward-only, print predictions.
                JSR     NEWLINE
                LDX     #STR_TESTHDR
                JSR     PUTS
                JSR     NEWLINE
                JSR     FINAL_TEST
HALT:           BRA     HALT

MAIN_RPRT:      FDB     RPRT            ; constant for step mod RPRT DIVQ

                END     MAIN
