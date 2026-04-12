* ============================================================
* layer.asm — Transformer layer (embedding, attention, projection)
* ============================================================
* Dependencies: equates.inc, macros.inc, fxmath.asm, vecop.asm,
*               matop.asm, actfn.asm
* Exports: EMBED, ATTEND, PROJECT, LAYER_FWD, LAYER_BWD
* ============================================================
*
* TODO: implement in subsequent turn
*
* Build order step 6. Requires ACTFN passing tests first.
*
* Will implement:
*   EMBED      — token + position embedding lookup from Q16 weights
*                Derives Q8 on the fly from Q16 high word
*   ATTEND     — Q = Wq*x, K = Wk*x, V = Wv*x
*                ATT = softmax(Q*K^T), CTX = ATT*V
*   PROJECT    — logits = Wout * CTX
*   LAYER_FWD  — full forward pass, populates FWD_CACHE
*   LAYER_BWD  — full backward pass, populates GRAD_BASE
*

                INCLUDE equates.inc
                INCLUDE macros.inc

* Stub — flat binary, no SECTION needed

* Stub — implementation in subsequent turn


