* ============================================================
* fxmath.inc — FXMATH routine interface documentation
* ============================================================
* Dependencies: equates.inc (for MULSCR)
* Usage: this file is for documentation only. The actual code
* lives in src/fxmath.asm, which is INCLUDE'd into main.asm or
* a test harness. All labels below become locally resolved.
* ============================================================
*
* FX_MUL_Q8Q15
*   Q8 x Q15 -> Q15 truncating multiply.
*   Entry:  B = Q8 operand, Y -> Q15 operand in memory
*   Exit:   D = Q15 result
*   Matches ATTN/11 ASHC #-8 semantics (bit-exact).
*
* FX_MUL_Q15Q15
*   Q15 x Q15 -> Q15 truncating multiply.
*   Entry:  D = Q15 operand, Y -> Q15 operand in memory
*   Exit:   D = Q15 result
*
* FX_SHIFT_XENT
*   Q8 -> Q15 left shift by 7 (value x 128).
*   Entry:  D = Q8 gradient (sign-extended to 16-bit)
*   Exit:   D = Q15 gradient
*
* Macros (expand at call site, see macros.inc):
*   DIV8_Q      -- 16-bit /8 for non-negative D
*   DIV8_Q32    -- 32-bit /8 for non-negative Q (D:W)
*   NORMQ15     -- Q23 -> Q15 via STQ + LDD offset
*   NORMQ15_HI  -- Q30 -> Q15 via STQ + LDD + LSLD
