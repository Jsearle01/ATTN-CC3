* ============================================================
* train.asm — SGD training loop
* ============================================================
* Dependencies: equates.inc, macros.inc, layer.asm, screen.asm,
*               data.asm
* Exports: TRAIN
* ============================================================
*
* TODO: implement in subsequent turn
*
* Build order step 7. Requires LAYER passing tests first.
*
* Will implement:
*   TRAIN       — outer loop: MAXSTEPS iterations
*                 Each step:
*                   1. Generate random 8-digit sequence
*                   2. ZEROFILL gradient accumulators (TFM)
*                   3. LAYER_FWD → compute loss + accuracy
*                   4. LAYER_BWD → compute gradients
*                   5. SGD update: W -= lr * grad (per-layer LR)
*                      Three LSRQ for loss/8 averaging
*                   6. Every LOG_EVERY steps: screen output
*   SGD_UPDATE  — apply gradients with per-block learning rates
*                 LR_QKV=5243, LR_EMB=655, LR_WOUT=164
*

                INCLUDE equates.inc
                INCLUDE macros.inc

* Stub — flat binary, no SECTION needed

* Stub — implementation in subsequent turn


