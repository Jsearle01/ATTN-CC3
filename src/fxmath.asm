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

* ------------------------------------------------------------
* FXDIV — Q8 division: D = D / (Y)
* ------------------------------------------------------------
* Matches ATTN/11 FXDIV: (a << 8) / b, truncating toward zero.
* Uses 6309 DIVQ instruction.
*
* Entry:  D = dividend (Q8, 16-bit signed)
*         Y -> divisor in memory (Q8, 16-bit signed)
* Exit:   D = quotient (Q8, truncated toward zero)
*         CC: C set on divide-by-zero
* Clobbers: D, W, Q, CC
* Uses: MULSCR (4B) for building 32-bit dividend
*
* Divide-by-zero: returns $7FFF if a > 0, $8000 if a < 0,
* $0000 if a == 0. Sets carry flag.
*
* DIVQ: Q (32-bit D:W) / memory (16-bit) -> W = quotient, D = remainder
* DIVQ truncates toward zero (matching PDP-11 DIV behavior).
* ------------------------------------------------------------
FXDIV:
* Check for divide-by-zero
                LDW     ,Y              ; W = divisor (for later check)
                CMPW    #0
                BNE     FXD_OK
* Divide by zero — saturate based on sign of dividend
                TSTA                    ; test high byte of D (dividend)
                BEQ     FXD_DZTEST_B
                BPL     FXD_DZ_POS
                BRA     FXD_DZ_NEG
FXD_DZTEST_B:
                TSTB                    ; A=0, check B
                BEQ     FXD_DZ_ZERO
FXD_DZ_POS:
                LDD     #$7FFF          ; +saturate
                ORCC    #$01            ; set carry
                RTS
FXD_DZ_NEG:
                LDD     #$8000          ; -saturate
                ORCC    #$01            ; set carry
                RTS
FXD_DZ_ZERO:
                CLRA                    ; 0/0 = 0
                CLRB
                ORCC    #$01            ; set carry
                RTS

FXD_OK:
* Build 32-bit dividend = a << 8.
* Place a's bytes at positions [1] and [2] of 4-byte MULSCR:
*   MULSCR+0 = sign extension ($00 or $FF)
*   MULSCR+1 = D high byte (A)
*   MULSCR+2 = D low byte (B)
*   MULSCR+3 = $00
                STA     MULSCR+1        ; a high byte
                STB     MULSCR+2        ; a low byte
                CLR     MULSCR+3        ; low byte = 0
                TSTA
                BPL     FXD_SPOS
                LDA     #$FF
                STA     MULSCR          ; negative sign extension
                BRA     FXD_DO
FXD_SPOS:
                CLR     MULSCR          ; positive sign extension
FXD_DO:
* Load Q from MULSCR (32-bit dividend = a << 8)
                LDQ     MULSCR
* DIVQ: Q / (Y) -> W = quotient, D = remainder
* DIVQ sets V if quotient overflows 16-bit signed range.
                DIVQ    ,Y
                BVS     FXD_OVERFLOW
* Move quotient from W to D
                TFR     W,D
                ANDCC   #$FE            ; clear carry (success)
                RTS

* Overflow: quotient exceeds 16-bit signed range.
* Determine sign of result from signs of dividend and divisor.
* MULSCR+0 = sign of dividend ($00 pos, $FF neg).
* (Y) = divisor (test sign).
FXD_OVERFLOW:
                LDA     MULSCR          ; sign of dividend
                EORA    ,Y              ; XOR with high byte of divisor
* If result of XOR has bit 7 set, signs differ -> negative result
                BMI     FXD_OV_NEG
                LDD     #$7FFF          ; positive saturate
                ANDCC   #$FE
                RTS
FXD_OV_NEG:
                LDD     #$8000          ; negative saturate
                ANDCC   #$FE
                RTS
