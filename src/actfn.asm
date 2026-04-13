* ============================================================
* actfn.asm — Activation functions for ATTN-CC3
* ============================================================
* Dependencies: equates.inc (SF_*, VDOT_TMP), macros.inc,
*               fxmath.asm (FXDIV), vecop.asm (VMAX)
* Exports: SFTMX
* ============================================================
* NOTE: This file is INCLUDE'd into the parent. Do not INCLUDE
* equates.inc or macros.inc here.
* ============================================================

* ============================================================
* SFTMX — Softmax on Q8 vector
* ============================================================
* Computes softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
* Matches ATTN/11 SFTMX algorithm:
*   1. VMAX to find max for numerical stability
*   2. Per element: (max - x_i) >> 3 -> EXPTBL index, accumulate sum
*   3. Per element: FXDIV(exp_i, sum) -> normalized probability
*
* Caller sets: SF_VEC, SF_LEN, SF_OUT
*   SF_VEC = input vector pointer (Q8, 16-bit elements)
*   SF_LEN = element count (word, low byte used)
*   SF_OUT = output vector pointer (Q8 probabilities)
*
* Exit: SF_OUT[] filled with softmax probabilities (Q8)
*       Values in [0, 256], should sum to ~256 (= 1.0 Q8)
* Clobbers: X, Y, U, D, W, Q, CC
* Uses: SF_MAX, SF_SUM, VDOT_TMP (internal)
* ============================================================
SFTMX:
* ---- Step 1: find max for numerical stability ----
                LDX     SF_VEC          ; X -> input vector
                LDB     SF_LEN+1        ; B = length
                JSR     VMAX            ; D = max value, X = max index
                STD     SF_MAX          ; save max

* ---- Step 2: exp(x_i - max) via EXPTBL, accumulate sum ----
                CLRA
                CLRB
                STD     SF_SUM          ; sum = 0
                LDX     SF_VEC          ; reset to start of input
                LDY     SF_OUT          ; Y -> output
                LDB     SF_LEN+1
                PSHS    B               ; loop counter on stack

SFTMX_EXP:
                LDD     SF_MAX          ; D = max
                SUBD    ,X++            ; D = max - x_i (>= 0), advance X
* Clamp (max - x_i) to avoid EXPTBL overrun.
* After >> 3, index must be <= 255. So (max - x_i) <= 2047.
* Values > 2040 produce exp() = 0; use index 255 directly.
                CMPD    #2040
                BLE     SFTMX_NOCLP
                LDD     #255            ; clamp to max index
                BRA     SFTMX_LOOK
SFTMX_NOCLP:
* Divide by 8 (>> 3) to get table index
                LSRD
                LSRD
                LSRD
SFTMX_LOOK:
* D = index (0..255). Multiply by 2 for word offset.
                LSLD                    ; D = index * 2
                LDU     #EXPTBL
                LDD     D,U             ; D = EXPTBL[index] (Q8 exp value)
                STD     ,Y++            ; store exp value to output
* Accumulate sum (16-bit, safe for count <= 127)
                ADDD    SF_SUM
                STD     SF_SUM
                DEC     ,S
                BNE     SFTMX_EXP
                LEAS    1,S

* ---- Step 3: normalize — divide each exp value by sum ----
* Guard: if sum == 0, output is all zeros (already stored as exp values
* which would be 0 if sum is 0). Skip division.
                LDD     SF_SUM
                BEQ     SFTMX_DONE

                LDX     SF_OUT          ; X -> exp values (in output buf)
                STD     VDOT_TMP        ; store sum for FXDIV (Y -> divisor)
                LDB     SF_LEN+1
                PSHS    B               ; loop counter

SFTMX_DIV:
                LDD     ,X              ; D = exp_i
                LDY     #VDOT_TMP       ; Y -> sum (divisor)
                PSHS    X               ; save output pointer
                JSR     FXDIV           ; D = exp_i / sum (Q8)
                PULS    X
                STD     ,X++            ; store normalized value, advance
                DEC     ,S
                BNE     SFTMX_DIV
                LEAS    1,S

SFTMX_DONE:
                RTS

* ============================================================
* EXPTBL — exp() lookup table (assembled inline)
* ============================================================
                INCLUDE tables/exptbl.asm
