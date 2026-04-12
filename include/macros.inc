* ============================================================
* macros.inc — 6309 optimization primitives as LWASM macros
* ============================================================
* Dependencies: equates.inc (for MULSCR)
* Included by: fxmath.asm, vecop.asm, matop.asm, layer.asm, train.asm
* ============================================================

* ------------------------------------------------------------
* NORMQ15 — Normalize Q23 product to Q15 via STQ + LDD offset
* ------------------------------------------------------------
* Preconditions: MULD just executed, Q holds 32-bit Q23 product.
* Postconditions: D = Q15 result (bits 23..8 of Q).
* Clobbers: CC.
* Cost: STQ(6) + LDD(5) = 11 cycles
* Replaces: ASHC #-8 equivalent shift loop (~40+ cycles).
* THIS IS THE SINGLE MOST IMPORTANT OPTIMIZATION IN THE PORT.
* Applied throughout MATOP and VECOP inner loops.
* ------------------------------------------------------------
NORMQ15         MACRO
                STQ     MULSCR
                LDD     MULSCR+1
                ENDM

* ------------------------------------------------------------
* NORMQ15_HI — Take high 16 bits of Q, shift left 1 (Q30 → Q15)
* ------------------------------------------------------------
* Preconditions: MULD just executed on Q15×Q15, Q holds Q30 product.
* Postconditions: D = Q15 result (bits 30..15 of Q).
* Clobbers: CC.
* Used for Q15×Q15 backward-pass multiply (FX_MUL_Q15Q15).
* ------------------------------------------------------------
NORMQ15_HI      MACRO
                STQ     MULSCR
                LDD     MULSCR
                LSLD
                ENDM

* ------------------------------------------------------------
* ZEROFILL — TFM-based zero fill
* ------------------------------------------------------------
* Args: \1 = destination address, \2 = byte count
* Clobbers: X, U, W, CC, one byte at \1 (seeds the fill).
* Used for gradient accumulator zeroing between SGD steps.
* ------------------------------------------------------------
ZEROFILL        MACRO
                CLR     \1              ; seed first byte with zero
                LDX     #\1             ; src = first (now zero) byte
                LDU     #\1+1           ; dst = next byte
                LDW     #\2-1           ; count = remaining bytes
                TFM     X+,U+           ; propagate zero through buffer
                ENDM

* ------------------------------------------------------------
* DIV8_Q — Divide D by 8 via LSRD x 3
* ------------------------------------------------------------
* NOTE: The project specification originally called for three
* "LSRQ" instructions. LSRQ is not a real HD6309 opcode — verified
* empirically with LWASM 4.21 ("Bad opcode" error). The valid
* 6309 register shifts are LSRD (16-bit D), LSRW (16-bit W), and
* composed Q shifts via LSRD/RORW pairs. For a 16-bit non-negative
* value in D, three LSRD instructions produce the correct /8
* result and match ATTN/11's ASHC #-3 behavior exactly.
* ------------------------------------------------------------
* Preconditions: D >= 0 (logical shift — safe only for non-negative)
* Postconditions: D = D / 8
* Clobbers: CC
* Used for: 16-bit loss averaging when sum fits in D.
* ------------------------------------------------------------
DIV8_Q          MACRO
                LSRD
                LSRD
                LSRD
                ENDM

* ------------------------------------------------------------
* DIV8_Q32 — Divide 32-bit Q (D:W) by 8 via LSRD/RORW x 3
* ------------------------------------------------------------
* Q register layout on 6309: Q = D:W (D is high 16, W is low 16).
* A 32-bit logical shift right by one is LSRD (drops D bit 0 into
* carry, fills D bit 15 with 0) then RORW (rotates carry into W
* bit 15, drops W bit 0 — discarded). Three pairs = shift by 3.
* ------------------------------------------------------------
* Preconditions: Q >= 0 (loss is non-negative by construction)
* Postconditions: Q = Q / 8
* Clobbers: CC
* Used for: cross-entropy loss averaging across SEQ=8 positions.
* Matches ATTN/11's ASHC #-3 on the PDP-11 register pair.
* ------------------------------------------------------------
DIV8_Q32        MACRO
                LSRD
                RORW
                LSRD
                RORW
                LSRD
                RORW
                ENDM
