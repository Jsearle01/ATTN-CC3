* ============================================================
* fxmath.asm — Scalar fixed-point arithmetic primitives
* ============================================================
* Dependencies: equates.inc (MULSCR), macros.inc (NORMQ15,
*               NORMQ15_HI)
* Exports: FX_MUL_Q8Q15, FX_MUL_Q15Q15, FX_SHIFT_XENT
* ============================================================
* NOTE: This file is INCLUDE'd into the parent (main.asm or
* test harness). Do not INCLUDE equates.inc or macros.inc here
* — the parent already has them and double-include would cause
* duplicate symbol errors.
* ============================================================

* ------------------------------------------------------------
* FX_MUL_Q8Q15 — Q8 x Q15 -> Q15 (forward pass multiply)
* ------------------------------------------------------------
* Entry:  B = Q8 operand (signed 8-bit)
*         Y -> Q15 operand in memory (2 bytes, big-endian)
* Exit:   D = Q15 result (truncating, matches ATTN/11 ASHC #-8)
* Clobbers: CC, W (via MULD/STQ in NORMQ15)
* Cycles: SEX(2) + MULD(28) + NORMQ15(11) + RTS(4) = ~45
* ------------------------------------------------------------
FX_MUL_Q8Q15:
                SEX                     ; sign-extend B -> D (Q8 in 16 bits)
                MULD    ,Y              ; Q = D * (Y), Q23 product
                NORMQ15                 ; D = bits [23:8] of Q = Q15 result
                RTS

* ------------------------------------------------------------
* FX_MUL_Q15Q15 — Q15 x Q15 -> Q15 (backward pass multiply)
* ------------------------------------------------------------
* Entry:  D = Q15 operand
*         Y -> Q15 operand in memory (2 bytes, big-endian)
* Exit:   D = Q15 result (truncating)
* Clobbers: CC, W (via MULD/STQ in NORMQ15_HI)
* Cycles: MULD(28) + NORMQ15_HI(14) + RTS(4) = ~46
* ------------------------------------------------------------
FX_MUL_Q15Q15:
                MULD    ,Y              ; Q = D * (Y), Q30 product
                NORMQ15_HI              ; D = bits [30:15] << 1 = Q15 result
                RTS

* ------------------------------------------------------------
* FX_SHIFT_XENT — Q8 -> Q15 left shift for softmax gradient
* ------------------------------------------------------------
* Per DEVIATIONS.md: spec called for "TFR D,W / CLRD / LSRQ"
* which is wrong (LSRQ fictional, direction wrong). Correct
* operation is left-shift by 7 (multiply by 128) per ATTN/11
* README: "result, initially Q8, is shifted left by 7 to Q15."
* ------------------------------------------------------------
* Entry:  D = Q8 gradient value (sign-extended to 16-bit)
* Exit:   D = Q15 value (= input * 128)
* Clobbers: CC
* Cycles: LSLD(2) x 7 + RTS(4) = 18
* ------------------------------------------------------------
FX_SHIFT_XENT:
                LSLD
                LSLD
                LSLD
                LSLD
                LSLD
                LSLD
                LSLD
                RTS
