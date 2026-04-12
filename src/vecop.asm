* ============================================================
* vecop.asm — Vector operations (dot product, scale)
* ============================================================
* Dependencies: equates.inc, macros.inc, fxmath.asm
* Exports: VEC_DOT_Q8, VEC_SCALE_Q15, VEC_ADD_Q15
* ============================================================
*
* TODO: implement in subsequent turn
*
* Build order step 3. Requires FXMATH passing tests first.
*
* Will implement:
*   VEC_DOT_Q8     — dot product of two Q8 vectors, length in Y
*                    Accumulates in 32-bit via MULD + add chain
*                    Uses NORMQ15 for final normalization
*   VEC_SCALE_Q15  — scale Q15 vector by Q15 scalar
*   VEC_ADD_Q15    — element-wise add of two Q15 vectors
*

                INCLUDE equates.inc
                INCLUDE macros.inc

* Stub — flat binary, no SECTION needed

* Stub — implementation in subsequent turn


