* ============================================================
* matop.asm — Matrix-vector multiply
* ============================================================
* Dependencies: equates.inc, macros.inc, fxmath.asm, vecop.asm
* Exports: MAT_VEC_Q8, MAT_VEC_Q16Q8, MAT_TVEC_Q15
* ============================================================
*
* TODO: implement in subsequent turn
*
* Build order step 4. Requires VECOP passing tests first.
*
* Will implement:
*   MAT_VEC_Q8     — Q8 matrix × Q8 vector → Q8 result
*                    Inner loop uses MULD + NORMQ15 per element
*   MAT_VEC_Q16Q8  — Q16 weight matrix × Q8 input → Q8 result
*                    Derives Q8 from Q16 high word via W register
*   MAT_TVEC_Q15   — Transpose multiply for backward pass
*

                INCLUDE equates.inc
                INCLUDE macros.inc

* Stub — flat binary, no SECTION needed

* Stub — implementation in subsequent turn


