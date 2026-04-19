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
* Clobbers: X, Y, U, D, W, CC, MP_*, SF_*, V_*, VS_*, BK_* internal
*
* Steps: 1 (dLogits->dWout,dY), 2 (dA,dV), 3 (dSc in-place). Steps 4-6 future.
* ============================================================
BKWRD:
                LDD     BK_SEQ
                LBEQ    BKWRD_DONE      ; zero-length guard
                JSR     BKWRD_INIT      ; strides + workspace ptrs + row cursors
                JSR     BKWRD_S1        ; Step 1: dLogits -> dWout, dY
                JSR     BKWRD_S2        ; Step 2: dA, dV
                JSR     BKWRD_S3        ; Step 3: dSc in-place over dA
                RTS

* ============================================================
* BKWRD_S1 — Step 1: dLogits → dWout, dY (callable sub)
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
* Entry: BKWRD_INIT has run (strides + row cursors BK_YI/BK_DYI valid).
* ============================================================
BKWRD_S1:
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
                RTS

* ============================================================
* BKWRD_INIT — compute strides and workspace sub-region pointers
* ============================================================
* Factored out so test harnesses can run Steps 2-3 without Step 1.
* Matches ATTN/11 BK.SET layout (BKWRD.MAC lines 453-461).
* Entry: caller has set BK_SEQ, BK_DIM, BK_VOC, BK_WRK, BK_YY, BK_DY.
* Exit:  BK_RSZ/VST/SST + BK_QP/KP/VP/AP + BK_YI/DYI populated.
* Clobbers: X, D, W, Q, CC
* ============================================================
BKWRD_INIT:
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

* Workspace partition: MULD D=SEQ, mem=RSZ -> W = SEQ*DIM*2 = sub-region size.
                LDD     BK_SEQ
                MULD    BK_RSZ          ; Q = SEQ * RSZ
                TFR     W,D             ; D = sub-region size
                PSHS    D
                LDD     BK_WRK
                STD     BK_QP           ; Q base = BK_WRK + 0
                ADDD    ,S
                STD     BK_KP           ; K base
                ADDD    ,S
                STD     BK_VP           ; V base
                ADDD    ,S
                STD     BK_AP           ; A/S base
                LEAS    2,S

* Init row cursors for Step 1
                LDD     BK_YY
                STD     BK_YI
                LDD     BK_DY
                STD     BK_DYI
                RTS

* ============================================================
* BKWRD_S2 — Step 2: Backward O = A @ V  ->  dA, dV
* ============================================================
* dO = dY (identity — dY IS dO, no copy, no separate buffer).
*
* For each (i, j):
*   dA[i][j] = VDOT(V[j], dY[i])           ; Q8 x Q15 with >>8 -> Q15
*   dV[j]   += VSADD(A[i][j], dY[i])       ; scalar * vector accumulate
*
* dV is zeroed at Step 2 entry (VCLR). This differs from dWout
* (caller-zeroed by ZEROG): dV only accumulates within Step 2's
* nested loop, so BKWRD handles it internally. dWout accumulates
* across the entire backward pass (Steps 1 + 5).
*
* Cursor plan (cursor-based, avoids ATTN/11's MUL index recomputation):
*   BK_DYI  — dY[i]  row ptr, advance by BK_RSZ per outer (reset here)
*   BK_VJ   — V[j]   ptr, reset to BK_VP per outer, advance by BK_RSZ per inner
*   BK_DVJ  — dV[j]  ptr, reset to BW_DV per outer, advance by BK_RSZ per inner
*   BK_AI   — A[i][j] advancing cursor, advances by 2 per inner;
*             after SEQ inners it lands at A[i+1][0] naturally
*   BK_DAI  — dA[i][j] advancing write cursor, same pattern as BK_AI
*
* Counters: stack byte counters for both outer (SEQ) and inner (SEQ) loops.
* Clobbers: X, Y, U, D, W, CC, VS_*, V_*, BK_* internal
* Entry: BKWRD_INIT has run (strides + workspace ptrs valid).
* ============================================================
BKWRD_S2:
* --- Zero dV (SEQ*DIM words) ---
                LDD     #BW_DV
                STD     V_DST
                LDD     BK_SEQ
                MULD    BK_DIM          ; Q = SEQ * DIM (W = low 16)
                TFR     W,D
                STD     V_LEN           ; word count
                JSR     VCLR

* --- Initialise per-outer cursors ---
                LDD     BK_DY
                STD     BK_DYI          ; dY[0] (Step 1 advanced past end)
                LDD     BK_AP
                STD     BK_AI           ; A[0][0]
                LDD     #BW_DATT
                STD     BK_DAI          ; dA[0][0]

                LDA     BK_SEQ+1
                PSHS    A               ; outer counter; stack: [oc]

BK2_OUT:
* Reset V[j] / dV[j] cursors for this outer i
                LDD     BK_VP
                STD     BK_VJ
                LDD     #BW_DV
                STD     BK_DVJ

                LDA     BK_SEQ+1
                PSHS    A               ; inner counter; stack: [ic oc]

BK2_IN:
* --- dA[i][j] = VDOT(V[j], dY[i], DIM) ---
                LDX     BK_VJ           ; X = V[j]
                LDY     BK_DYI          ; Y = dY[i]
                LDB     BK_DIM+1        ; B = DIM (<=255)
                JSR     VDOT            ; D = dot (Q8xQ15 >>8 = Q15)
                LDX     BK_DAI
                STD     ,X++            ; dA[i][j] = dot
                STX     BK_DAI          ; advance dA write ptr by 2

* --- dV[j] += VSADD(A[i][j], dY[i], DIM) ---
                LDX     BK_AI           ; X = &A[i][j]
                LDD     ,X++            ; D = A[i][j] (Q8 scalar)
                STD     VS_SCL
                STX     BK_AI           ; advance A read ptr by 2
                LDD     BK_DYI
                STD     VS_SRC          ; src = dY[i]
                LDD     BK_DVJ
                STD     VS_DST          ; dst = dV[j]
                LDD     BK_DIM
                STD     VS_LEN          ; word count = DIM
                JSR     VSADD

* --- Advance per-inner cursors V[j] and dV[j] by RSZ ---
                LDD     BK_VJ
                ADDD    BK_RSZ
                STD     BK_VJ
                LDD     BK_DVJ
                ADDD    BK_RSZ
                STD     BK_DVJ

                DEC     ,S              ; inner counter
                LBNE    BK2_IN
                LEAS    1,S             ; drop inner counter

* --- Advance dY[i] by RSZ; BK_AI / BK_DAI already at next row ---
                LDD     BK_DYI
                ADDD    BK_RSZ
                STD     BK_DYI

                DEC     ,S              ; outer counter
                LBNE    BK2_OUT
                LEAS    1,S
                RTS

* ============================================================
* BKWRD_S3 — Step 3: Backward softmax  ->  dSc (in-place over dA)
* ============================================================
* For each row i:
*   dot_ad = VDOT(A[i], dA[i], SEQ)
*   For each j:
*     diff    = clamp( dA[i][j] - dot_ad )     ; BVC/BPL pattern
*     product = diff * A[i][j]                  ; Q15 * Q8 = Q23 (32-bit)
*     result  = (product >> 8) >> BK_SHF        ; >>8 via STQ/LDD MULSCR+1
*     dSc[i][j] = result                        ; overwrites dA in-place
*
* Clamped subtraction matches ATTN/11 BKWRD.MAC lines 201-208 (BVC.NV /
* BPL.NP / MOV #077777 / BR / MOV #100000). 6309 SUBD sets V identically.
*
* Inline multiply extraction matches MATOP primitives (MVMUL/OUTER/VTMUL):
*   MULD ,Y++ leaves product in Q (W:D, big-endian 4 bytes in STQ target).
*   STQ MULSCR / LDD MULSCR+1 gives bits [23:8] = Q15 result.
*
* BK_SHF loop matches forward ATTN Step 4 (LDE/DECE/ASRD). BK_SHF=0 OK.
*
* In-place safety: dot_ad is computed from the full row via VDOT BEFORE
* any element is overwritten, then diff uses dA[i][j] loaded immediately
* before the multiply. Reading dA and writing dSc at the same cursor is
* safe because the read happens via LDD ,X while X is at the current
* element and the write (STD ,X++) happens after the multiply completes.
*
* Cursors: BK_AI and BK_DAI reset to row 0 at Step 3 entry, then each
* outer i loads them into X/Y for the inner loop. After the inner loop,
* X and Y hold row-(i+1) start and are stored back.
* Entry: BKWRD_INIT and BKWRD_S2 have run (dA at BW_DATT is valid).
* ============================================================
BKWRD_S3:
* --- Reset A and dA row bases to top ---
                LDD     BK_AP
                STD     BK_AI
                LDD     #BW_DATT
                STD     BK_DAI

                LDA     BK_SEQ+1
                PSHS    A               ; outer counter; stack: [oc]

BK3_OUT:
* --- dot_ad = VDOT(A[i], dA[i], SEQ) ---
                LDX     BK_AI           ; X = A[i]
                LDY     BK_DAI          ; Y = dA[i]
                LDB     BK_SEQ+1        ; B = SEQ (<=255)
                JSR     VDOT            ; D = Q15 dot_ad
                STD     BK_DAD

* --- Inner loop: reload cursors (VDOT clobbered X,Y) ---
                LDX     BK_DAI          ; X = dA[i][0] r/w cursor
                LDY     BK_AI           ; Y = A[i][0]  read cursor
                LDA     BK_SEQ+1
                PSHS    A               ; inner counter; stack: [ic oc]

BK3_IN:
* Clamped subtract: D = clamp( dA[i][j] - dot_ad )
                LDD     ,X              ; D = dA[i][j] (Q15)
                SUBD    BK_DAD          ; D -= dot_ad; V set on signed overflow
                BVC     BK3_NV          ; no overflow -> D is correct
                BPL     BK3_NP          ; result positive -> was negative overflow
                LDD     #32767          ; positive overflow -> clamp +max
                BRA     BK3_NV
BK3_NP:
                LDD     #-32768         ; negative overflow -> clamp -max
BK3_NV:
* Inline multiply: Q = diff * A[i][j], extract >>8 via STQ / LDD MULSCR+1
                MULD    ,Y++            ; Q = D * A[i][j]; Y += 2
                STQ     MULSCR          ; save full 32-bit product
                LDD     MULSCR+1        ; bits [23:8] = Q15 (>>8)

* Variable-count ASR by BK_SHF (undo forward sqrt-d scaling)
                LDE     BK_SHF+1        ; E = shift count (preserves D)
                BEQ     BK3_NOSHF
BK3_SHF:
                ASRD
                DECE
                BNE     BK3_SHF
BK3_NOSHF:
                STD     ,X++            ; dSc[i][j] in-place over dA[i][j]

                DEC     ,S              ; inner counter
                LBNE    BK3_IN
                LEAS    1,S             ; drop inner counter

* X and Y now at row-(i+1) start — save back to row-base fields
                STX     BK_DAI
                STY     BK_AI

                DEC     ,S              ; outer counter
                LBNE    BK3_OUT
                LEAS    1,S
                RTS

* ============================================================
* Steps 4-6: (future)
* ============================================================

BKWRD_DONE:
                RTS

* ============================================================
* LOGTBL — 257-entry Q12 cross-entropy lookup table
* ============================================================
                INCLUDE tables/logtbl.asm
