* ============================================================
* train.asm — Training routines
* ============================================================
* Training-phase code. Depends on all forward-pass primitives
* and composites (fxmath, vecop, matop, actfn, layer).
*
* Routines:
*   CLOSS  — Cross-entropy loss (monitoring only)
*   (BKWRD, UPDAT — future)
*
* NOTE: This file is INCLUDE'd into the parent. Do not INCLUDE
* equates.inc or macros.inc here.
* ============================================================

* ============================================================
* CLOSS — Cross-entropy loss computation (monitoring only)
* ============================================================
* For each position i in 0..CL_SEQ-1:
*   1. Copy logits[i] (CL_VOC words) to CL_TMP
*   2. SFTMX(CL_TMP) in-place → probabilities
*   3. p = CL_TMP[target[i]] (Q8, 0..256)
*   4. If p < 256: loss += LOGTBL[p] (Q12 unsigned)
*      If p >= 256: loss += 0 (perfect prediction)
* After loop: loss >>= 3 (average over SEQ=8)
*
* Caller sets: CL_LOG, CL_TGT, CL_SEQ, CL_VOC
* Returns: D = Q12 average cross-entropy loss (16-bit)
* Clobbers: X, Y, U, D, W, CC, SF_*, CL_AHI, CL_ALO, CL_TMP
*
* Counter: stack byte counter for SEQ loop (SEQ <= 255).
* Pointers: X (logits cursor) and U (target cursor) saved/restored
* across SFTMX via stack. Copy loop advances X to next row naturally.
* ============================================================
CLOSS:
                LDW     CL_SEQ
                LBEQ    CL_ZERO         ; zero-length guard (body > 127 B)

* Pre-compute VOC stride and zero accumulator
                LDD     CL_VOC
                LSLD
                STD     CL_VST          ; VOC * 2
                CLRA
                CLRB
                STD     CL_AHI
                STD     CL_ALO

* Init pointers
                LDX     CL_LOG          ; logits cursor
                LDU     CL_TGT          ; target cursor

* Outer counter
                LDA     CL_SEQ+1
                PSHS    A               ; stack: [counter]

CL_LOOP:
* --- Copy logits[i] to CL_TMP (VOC words) ---
* X points to logits[i]. Copy advances X to logits[i+1].
                LDY     #CL_TMP
                LDB     CL_VOC+1
                PSHS    B               ; inner copy counter
CL_COPY:
                LDD     ,X++
                STD     ,Y++
                DEC     ,S
                BNE     CL_COPY
                LEAS    1,S

* X now points to logits[i+1]. Save X and U across SFTMX.
                PSHS    X,U             ; stack: [X(2) U(2) counter(1)]

* --- SFTMX on CL_TMP in-place ---
                LDD     #CL_TMP
                STD     SF_VEC
                STD     SF_OUT
                LDD     CL_VOC
                STD     SF_LEN
                JSR     SFTMX

* --- Look up target[i] probability ---
                PULS    X,U             ; restore; stack: [counter]
                LDD     ,U++            ; D = target token (word), advance U
                LSLD                    ; D = token * 2 (byte offset)
                LDY     #CL_TMP
                LDD     D,Y             ; D = probability p (Q8, 0..256)

* --- Guard: p >= 256 means loss = 0 ---
                CMPD    #256
                BHS     CL_SKIP

* --- LOGTBL lookup ---
                LSLD                    ; D = p * 2 (byte offset into LOGTBL)
                LDY     #LOGTBL
                LDD     D,Y             ; D = LOGTBL[p] (Q12 unsigned)

* --- 32-bit accumulate: CL_AHI:CL_ALO += D ---
                ADDD    CL_ALO
                STD     CL_ALO
                BCC     CL_SKIP
                LDD     CL_AHI
                ADDD    #1
                STD     CL_AHI

CL_SKIP:
                DEC     ,S              ; outer counter
                BNE     CL_LOOP
                LEAS    1,S

* --- Average: 32-bit >>3 (divide by SEQ=8) ---
* Max sum = 8 * 22713 = 181704 = $0002C5A8 (18 bits).
* After >>3 = 22713. Fits 16 bits.
                LDE     #3
CL_AVG:
                LSR     CL_AHI
                ROR     CL_AHI+1
                ROR     CL_ALO
                ROR     CL_ALO+1
                DECE
                BNE     CL_AVG

                LDD     CL_ALO
                RTS

CL_ZERO:
                CLRA
                CLRB
                RTS

* ============================================================
* BKWRD — Backward pass (6 steps)
* ============================================================
* Computes gradients for all weights and embeddings.
* Reverse of forward pass (PROJ → ATTN → EMBED).
*
* Caller sets: BK_LOG, BK_TGT, BK_YY, BK_WOUT, BK_XX,
*   BK_WQ, BK_WK, BK_WV, BK_WRK, BK_DWOUT, BK_DY,
*   BK_SEQ, BK_DIM, BK_VOC, BK_SHF
*
* Caller contract: all gradient output buffers must be zeroed
*   before calling BKWRD. BKWRD accumulates via OUTER — does
*   not zero internally. Training loop calls ZEROG before BKWRD.
*
* Gradient format: Q15. Q8 × Q15 = Q23, >>8 → Q15.
*   Same shift as forward — all primitives (VTMUL, MVMUL, MVADD,
*   OUTER, VDOT, VSADD) use identical >>8.
*
* BK_DL aliases CL_TMP: CLOSS/BKWRD never concurrent.
*
* Clobbers: X, Y, U, D, W, CC, MP_*, SF_*, BK_* internal
*
* Steps: 1 (dLogits→dWout,dY). Steps 2-6 future.
* ============================================================
BKWRD:
                LDD     BK_SEQ
                LBEQ    BKWRD_DONE      ; zero-length guard

* Pre-compute strides
                LDD     BK_DIM
                LSLD
                STD     BK_RSZ          ; DIM*2
                LDD     BK_VOC
                LSLD
                STD     BK_VST          ; VOC*2
                LDD     BK_SEQ
                LSLD
                STD     BK_SST          ; SEQ*2

* Init row cursors
                LDD     BK_YY
                STD     BK_YI           ; Y row cursor
                LDD     BK_DY
                STD     BK_DYI          ; dY row cursor

* ============================================================
* Step 1: dLogits → dWout, dY
* ============================================================
* For each position i:
*   dL = softmax(logits[i]) - one_hot(target[i]), <<7 → Q15
*   dWout += OUTER(Y[i], dL)
*   dY[i] = MVMUL(Wout, dL)
*
* Pointer plan:
*   X = logits row ptr (register; copy loop advances to next row)
*   U = target ptr (register; LDD ,U++ advances by 2)
*   BK_YI = Y row cursor (in scratch; loaded for OUTER, advanced)
*   BK_DYI = dY row cursor (in scratch; loaded for MVMUL, advanced)
*   X and U saved/restored across SFTMX/OUTER/MVMUL via stack.
* ============================================================

                LDX     BK_LOG          ; logits cursor
                LDU     BK_TGT          ; target cursor
                LDA     BK_SEQ+1
                PSHS    A               ; outer counter; stack: [cnt]

BK1_LOOP:
* --- 1a. Copy logits[i] → BK_DL (VOC words) ---
* X points to logits[i]; copy advances X to logits[i+1].
                LDY     #BK_DL
                LDB     BK_VOC+1
                PSHS    B               ; copy counter
BK1_COPY:
                LDD     ,X++
                STD     ,Y++
                DEC     ,S
                BNE     BK1_COPY
                LEAS    1,S
* X now at logits[i+1]. Save X (next row) and U (target) across JSRs.
                PSHS    X,U             ; stack: [X(2) U(2) cnt(1)]

* --- 1b. SFTMX on BK_DL in-place → Q8 probabilities ---
                LDD     #BK_DL
                STD     SF_VEC
                STD     SF_OUT
                LDD     BK_VOC
                STD     SF_LEN
                JSR     SFTMX

* --- 1c. One-hot subtraction: BK_DL[target[i]] -= 256 ---
* Peek at saved U on stack to get target index.
* Stack: [X(2) U(2) cnt(1)]. U is at 2,S.
                LDX     2,S             ; X = saved U (target ptr)
                LDD     ,X              ; D = target token index (0..VOC-1)
                LSLD                    ; D = byte offset into BK_DL
                LDY     #BK_DL
                LEAY    D,Y             ; Y = &BK_DL[target]
                LDD     ,Y              ; D = softmax probability (Q8)
                SUBD    #256            ; subtract one-hot (no clamp needed)
                STD     ,Y              ; store back

* --- 1d. dL <<= 7 (Q8 → Q15, per-element) ---
                LDX     #BK_DL
                LDB     BK_VOC+1
                PSHS    B               ; shift counter
BK1_S7:
                LDD     ,X
                LSLD
                LSLD
                LSLD
                LSLD
                LSLD
                LSLD
                LSLD
                STD     ,X++
                DEC     ,S
                BNE     BK1_S7
                LEAS    1,S

* --- 1e. dWout += OUTER(Y[i], dL) ---
                LDD     BK_DWOUT
                STD     MP_MAT
                LDD     BK_YI
                STD     MP_VIN          ; vx = Y[i]
                LDD     #BK_DL
                STD     MP_OUT          ; vy = dL
                LDD     BK_DIM
                STD     MP_ROW
                LDD     BK_VOC
                STD     MP_COL
                JSR     OUTER

* --- 1f. dY[i] = MVMUL(Wout, dL) ---
                LDD     BK_WOUT
                STD     MP_MAT
                LDD     #BK_DL
                STD     MP_VIN          ; vin = dL
                LDD     BK_DYI
                STD     MP_OUT          ; vout = dY[i]
                LDD     BK_DIM
                STD     MP_ROW
                LDD     BK_VOC
                STD     MP_COL
                JSR     MVMUL

* --- Advance cursors ---
                LDD     BK_YI
                ADDD    BK_RSZ
                STD     BK_YI
                LDD     BK_DYI
                ADDD    BK_RSZ
                STD     BK_DYI

* Restore X (logits next row) and U (target ptr)
                PULS    X,U             ; stack: [cnt]
* U was the saved target ptr BEFORE LDD ,U++.
* We need to advance U past the target we consumed.
                LEAU    2,U             ; advance target ptr by 1 word

                DEC     ,S
                LBNE    BK1_LOOP
                LEAS    1,S

* ============================================================
* Steps 2-6: (future)
* ============================================================

BKWRD_DONE:
                RTS

* ============================================================
* LOGTBL — 257-entry Q12 cross-entropy lookup table
* ============================================================
                INCLUDE tables/logtbl.asm
