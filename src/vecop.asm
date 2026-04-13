* ============================================================
* vecop.asm — Vector operations: dot product, scale, addition
* ============================================================
* Dependencies: equates.inc (VDOT_ACC, VDOT_TMP, VSCL_ALPHA,
*               MULSCR), macros.inc (NORMQ15)
* Exports: VDOT, VSCL, VADD
* ============================================================
* NOTE: This file is INCLUDE'd into the parent (main.asm or
* test harness). Do not INCLUDE equates.inc or macros.inc here.
* ============================================================
* All values are Q7.8 fixed-point (16-bit signed) unless noted.
* Matches ATTN/11 VECOP.MAC bit-exactly.
* ============================================================

* ============================================================
* VDOT — Dot product with 32-bit accumulation
* ============================================================
* Matches ATTN/11: accumulates all x[i]*y[i] products in a
* 32-bit Q16 accumulator, then extracts bits [23:8] (equivalent
* to ASHC #-8), with overflow clamping to +/-32767.
*
* Entry:  X -> vector x (2 bytes per element, Q8 signed)
*         Y -> vector y (2 bytes per element, Q8 signed)
*         B = element count (1..128)
* Exit:   D = dot product result (Q8, clamped)
* Clobbers: X, Y, D, W, Q, CC
* Uses: VDOT_ACC (4B), MULSCR (4B) in BW_SCRATCH
* ============================================================
VDOT:
                PSHS    B               ; save count on stack
* Clear 32-bit accumulator (high word, low word)
                CLRA
                CLRB
                STD     VDOT_ACC        ; high word = 0
                STD     VDOT_ACC+2      ; low word = 0

VDOT_LOOP:
                LDD     ,X++            ; D = x[i], advance X
                MULD    ,Y++            ; Q = D * (Y), advance Y
*                                       ; D = high 16, W = low 16
* Save product to MULSCR for 32-bit addition
                STQ     MULSCR          ; MULSCR+0..1=D(high), +2..3=W(low)
* Add product to 32-bit accumulator
* Step 1: low words — ADDD preserves carry for step 2
                LDD     VDOT_ACC+2      ; D = acc low (carry NOT affected)
                ADDD    MULSCR+2        ; D += product low, carry set
                STD     VDOT_ACC+2      ; store (carry NOT affected by STD)
* Step 2: high words + carry from step 1
* LDD does NOT affect carry on 6809/6309.
                LDD     VDOT_ACC        ; D = acc high (carry preserved!)
                ADCB    MULSCR+1        ; B += product_high_lo + carry
                ADCA    MULSCR          ; A += product_high_hi + carry
                STD     VDOT_ACC        ; store acc high
* Decrement loop counter on stack
                DEC     ,S
                BNE     VDOT_LOOP

* ---- Extract Q8 result from 32-bit Q16 accumulator ----
* Equivalent to PDP-11 ASHC #-8: take bits [23:8] of accumulator.
* Overflow check matches ATTN/11: check MSB byte (= V >> 24).
*   $00 = positive fits, $FF = negative fits, else clamp.
                LDA     VDOT_ACC        ; A = MSB of 32-bit acc
                BEQ     VDOT_OK         ; $00: positive result fits
                CMPA    #$FF
                BEQ     VDOT_OK         ; $FF: negative result fits
* Overflow — clamp
                TSTA
                BPL     VDOT_CLPOS
                LDD     #$8000          ; clamp to -32768
                BRA     VDOT_EXIT
VDOT_CLPOS:
                LDD     #$7FFF          ; clamp to +32767
                BRA     VDOT_EXIT
VDOT_OK:
                LDD     VDOT_ACC+1      ; D = bits [23:8] = Q8 result
VDOT_EXIT:
                LEAS    1,S             ; drop count byte from stack
                RTS

* ============================================================
* VSCL — Scalar-vector multiply: y[i] = alpha * x[i] (Q8)
* ============================================================
* Per-element multiply with normalize-by-offset (NORMQ15 macro).
* Matches ATTN/11 VSCL: MUL + ASHC #-8 per element.
*
* Entry:  X -> source vector x (2 bytes per element)
*         Y -> destination vector y
*         D = alpha (Q8 scalar, 16-bit signed)
*         U = element count (low byte used, 1..128)
* Exit:   Y populated with scaled values
* Clobbers: X, Y, D, W, Q, U, CC
* Uses: VSCL_ALPHA (2B), MULSCR (4B via NORMQ15)
* ============================================================
VSCL:
                STD     VSCL_ALPHA      ; save scalar alpha
                TFR     U,D             ; move count to D
                PSHS    B               ; push low byte as loop counter

VSCL_LOOP:
                LDD     VSCL_ALPHA      ; reload alpha
                MULD    ,X++            ; Q = alpha * x[i], advance X
                NORMQ15                 ; D = bits [23:8] = Q8 result
                STD     ,Y++            ; y[i] = result, advance Y
                DEC     ,S              ; decrement count
                BNE     VSCL_LOOP

                LEAS    1,S             ; drop count byte
                RTS

* ============================================================
* VADD — Vector addition: z[i] = x[i] + y[i]
* ============================================================
* Plain 16-bit addition, no scaling, wraps on overflow.
* Matches ATTN/11 VADD: MOV + ADD per element.
*
* Entry:  X -> vector x (2 bytes per element)
*         Y -> vector y (2 bytes per element)
*         U -> vector z (destination)
*         B = element count (1..128)
* Exit:   U populated with sums
* Clobbers: X, Y, U, D, CC
* ============================================================
VADD:
                PSHS    B               ; save count on stack
VADD_LOOP:
                LDD     ,X++            ; D = x[i]
                ADDD    ,Y++            ; D = x[i] + y[i]
                STD     ,U++            ; z[i] = sum
                DEC     ,S              ; decrement count (LDD clobbers B)
                BNE     VADD_LOOP
                LEAS    1,S             ; drop count byte
                RTS

* ============================================================
* VMAX — Find maximum element and its index
* ============================================================
* Matches ATTN/11 VMAX: linear scan, returns first occurrence
* of the maximum value.
*
* Entry:  X -> vector (2 bytes per element, Q8 signed)
*         B = element count (1..128)
* Exit:   D = max value (Q8)
*         X = index of max (0-based)
* Clobbers: X, Y, D, CC
* ============================================================
VMAX:
* Save count, init max = vec[0], best_index = 0, cur_index = 1
                PSHS    B               ; save count on stack
                LDD     ,X++            ; D = vec[0], advance X
                STD     VDOT_TMP        ; VDOT_TMP = current max value
                LDY     #0              ; Y = best index = 0
                DEC     ,S              ; one element consumed
                BEQ     VMAX_DONE       ; if count was 1, done
                LDU     #1              ; U = current scanning index
VMAX_LOOP:
                LDD     ,X++            ; D = vec[i]
                CMPD    VDOT_TMP        ; compare with current max
                BLE     VMAX_SKIP       ; not greater, skip
                STD     VDOT_TMP        ; new max
                TFR     U,Y             ; best index = current index
VMAX_SKIP:
                LEAU    1,U             ; current index++
                DEC     ,S              ; decrement remaining count
                BNE     VMAX_LOOP
VMAX_DONE:
                LDD     VDOT_TMP        ; D = max value
                TFR     Y,X             ; X = best index
                LEAS    1,S             ; drop count byte
                RTS
