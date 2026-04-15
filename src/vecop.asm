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

* ============================================================
* VCPY — Vector copy: dst = src
* ============================================================
* Copy V_LEN 16-bit words from V_SRC to V_DST.
*
* Caller sets: V_SRC, V_DST, V_LEN
*   V_LEN is word count (not byte count).
*
* Exit: V_LEN words at V_DST replaced with contents of V_SRC.
* Clobbers: X, U, W, CC
* Uses: 6309 TFM instruction (byte-wise block copy).
*
* Overlap policy: forward-direction TFM (low-to-high byte copy).
*   Safe when V_DST < V_SRC or buffers are disjoint.
*   dst > src with overlap produces corruption (tail of source
*   overwritten before being read).
*
* Edge case: V_LEN = 0 is a no-op. TFM with W=0 would transfer
* 65536 bytes on 6309 (counter wraps). Explicit zero check guards.
*
* Byte count scaling: TFM counts bytes in W, so W = V_LEN * 2.
* Uses ADDR W,W (3 bytes) because LSLW is not available in
* LWASM 4.21 (verified via probe: "Bad opcode" error). ADDR W,W
* is equivalent to LSLW for word counts that fit in the address
* space (< 32768 words = < 64KB).
* ============================================================
VCPY:
                LDW     V_LEN           ; W = word count
                BEQ     VCPY_DONE       ; zero-length no-op
                ADDR    W,W             ; W = byte count (word count * 2)
                LDX     V_SRC
                LDU     V_DST
                TFM     X+,U+           ; block copy W bytes, forward
VCPY_DONE:
                RTS

* ============================================================
* VCLR — Vector clear: dst = 0
* ============================================================
* Write zero to V_LEN 16-bit words starting at V_DST.
*
* Caller sets: V_DST, V_LEN (word count)
* Exit: V_LEN words at V_DST zeroed.
* Clobbers: X, D, W, CC
*
* Uses W as loop counter via DECW. Register counter is safe here
* because there is no LDD in the inner loop (STD does not clobber
* the counter; see the LDD-clobbers-B rule in DEVIATIONS.md).
*
* Edge case: V_LEN = 0 is a no-op.
* ============================================================
VCLR:
                LDW     V_LEN           ; W = word count
                BEQ     VCLR_DONE       ; zero-length no-op
                LDX     V_DST
                CLRA
                CLRB                    ; D = 0
VCLR_LOOP:
                STD     ,X++            ; store zero, advance
                DECW
                BNE     VCLR_LOOP
VCLR_DONE:
                RTS

* ============================================================
* VSADD — Vector scale-add with two-stage clamp
* ============================================================
* dst[k] = clamp16( dst[k] + clamp16( (scalar * src[k]) >> 8 ) )
* for k in 0..V_LEN-1
*
* Caller sets: VS_SRC, VS_DST, VS_SCL, VS_LEN
*   VS_LEN is word count (not byte count).
*   VS_SCL is a Q8 word, can be any signed 16-bit value.
*
* Exit: VS_LEN words at VS_DST updated (scale-add-clamped).
* Clobbers: X, Y, D, W, Q, CC
*
* DEVIATION from MATOP's single-clamp pattern: two-stage is correct
* for per-element scale-add (gradient update semantics). Matches
* ATTN/11's ASHC #-8 followed by ADD with BVC overflow clamp.
* See DEVIATIONS.md for why this differs from MVADD/VTMUL/OUTER:
* VSADD's per-element independence means each element's saturation
* should be committed before summing. MATOP's accumulating routines
* preserve full precision through to a final normalize.
*
* Q16 -> Q8 via bit-[23:8] extraction. Equals arithmetic >>8 for
* signed 32-bit MULD products. Matches ATTN/11's ASHC #-8 semantics.
*
* Two-stage clamp pattern:
*   Stage 1: clamp (scalar * src[k]) >> 8 to Q8 range via MSB byte test
*   Stage 2: clamp dst[k] + stage1 via ADDD/BVS, direction from stage1 sign
*
* MULSCR REUSE: The 4-byte STQ target at MULSCR is reused here for the
* 32-bit product. Safe because VSADD is never called from inside a
* MATOP routine (both exist at the same composition level).
*
* Scaled product preservation: PSHS D across the ADDD. Stack pattern
* chosen over U-holding to keep "BVS must follow ADDD" invariant
* visually obvious (no register manipulation between the two).
*
* V_LEN limit: must be <= 255. Current architecture uses D=16 maximum,
* so this limit is never approached. If a future variant grows vector
* dimension past 255, widen the counter to stack-word via PSHS D and
* use SUBD #1 / STD ,S at loop tail. See DEVIATIONS.md counter rule.
*
* Counter: MULD writes Q=D:W, so W cannot be a loop counter here.
* Use stack byte counter (matches VDOT/VSCL convention).
* ============================================================
VSADD:
                LDW     VS_LEN
                BEQ     VSADD_DONE      ; zero-length no-op
                LDY     VS_SRC
                LDX     VS_DST
                LDA     VS_LEN+1        ; low byte of V_LEN as counter
                PSHS    A               ; counter on stack (MULD clobbers W)
VSADD_LOOP:
* --- Stage 1: compute scaled product, clamp to Q8 ---
                LDD     ,Y++            ; D = src[k], advance src
                MULD    VS_SCL          ; Q = D * VS_SCL (signed 32-bit product)
                STQ     MULSCR          ; save full 32-bit product
* Check if result fits in Q8 (bits [23:8] must be in 16-bit signed range).
* MSB byte = bits [31:24]. $00 = positive fits, $FF = negative fits.
                LDA     MULSCR          ; MSB of 32-bit product
                BEQ     VSADD_S1_OK
                CMPA    #$FF
                BEQ     VSADD_S1_OK
* Overflow in stage 1 — saturate based on sign of full product.
                TSTA
                BPL     VSADD_S1_POS
                LDD     #$8000          ; negative saturate
                BRA     VSADD_S2
VSADD_S1_POS:
                LDD     #$7FFF          ; positive saturate
                BRA     VSADD_S2
VSADD_S1_OK:
                LDD     MULSCR+1        ; bits [23:8] = Q8 scaled product
VSADD_S2:
* D now holds the Q8 scaled product (stage 1 result).
* --- Stage 2: add to dst[k], clamp on overflow ---
* BVS must immediately follow ADDD. PSHS D preserves the scaled
* product for sign-testing on overflow; no V-affecting ops between.
                PSHS    D               ; preserve scaled product
                ADDD    ,X              ; D = scaled + dst[k], V set on overflow
                BVS     VSADD_OVFL
                STD     ,X++            ; no overflow: store sum, advance dst
                LEAS    2,S             ; drop preserved scaled (discard)
                BRA     VSADD_NEXT
VSADD_OVFL:
                PULS    D               ; recover scaled product for sign test
                TSTA                    ; test sign of scaled product
                BMI     VSADD_NEG_SAT
                LDD     #$7FFF          ; positive overflow (both addends positive)
                BRA     VSADD_OVFL_STORE
VSADD_NEG_SAT:
                LDD     #$8000          ; negative overflow
VSADD_OVFL_STORE:
                STD     ,X++            ; store saturated value, advance dst
VSADD_NEXT:
                DEC     ,S              ; decrement stack counter
                BNE     VSADD_LOOP      ; short branch fits (~85 bytes)
                LEAS    1,S             ; drop counter
VSADD_DONE:
                RTS
