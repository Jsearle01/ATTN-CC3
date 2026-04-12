* ============================================================
* actfn.asm — Activation functions (softmax, tanh, cross-entropy)
* ============================================================
* Dependencies: equates.inc, macros.inc, fxmath.asm, tables.asm
* Exports: SOFTMAX_Q8, TANH_Q8, CROSS_ENTROPY_Q15
* ============================================================
*
* TODO: implement in subsequent turn
*
* Build order step 5. Requires MATOP passing tests first.
*
* Will implement:
*   SOFTMAX_Q8       — row-wise softmax using EXPTBL lookup
*                      Max subtraction for numerical stability
*   TANH_Q8          — tanh approximation via table or piecewise
*   CROSS_ENTROPY_Q15 — -ln(prob) via LOGTBL lookup, Q12→Q15
*   SOFTMAX_GRAD_Q15 — backward softmax (prob - one_hot)
*                      Uses TFR D,W / CLRD / LSRQ shift trick
*

                INCLUDE equates.inc
                INCLUDE macros.inc

* Stub — flat binary, no SECTION needed

* Stub — implementation in subsequent turn


