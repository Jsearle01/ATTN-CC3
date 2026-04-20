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
                JSR     BKWRD_S4        ; Step 4: dQ, dK
                JSR     BKWRD_S5        ; Step 5: dX, dWq, dWk, dWv
                JSR     BKWRD_S6        ; Step 6: d_tok_emb, d_pos_emb
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
* BKWRD_S4 — Step 4: Backward Q.K^T -> dQ, dK
* ============================================================
* Part A: for each i, dQ[i] = VTMUL(K, dSc[i], SEQ, DIM)
*   dSc row stride = BK_SST (SEQ*2). dQ row stride = BK_RSZ (DIM*2).
*   VTMUL clears output on entry, so dQ is fully written — no pre-zero.
*
* Part B: for each j, extract column j of dSc into BK_DTMP, then
*   dK[j] = VTMUL(Q, BK_DTMP, SEQ, DIM)
*   Column stride in row-major dSc: BK_SST between consecutive rows
*   (column-j elements are BK_SST bytes apart). Extraction helper
*   BK4_COL is a small local sub called via BSR.
*
* MP_ROW and MP_COL are set ONCE (Part A) and carry through Part B
* unchanged — both VTMULs use SEQ rows × DIM cols.
*
* Entry: BKWRD_INIT and S1-S3 have run. BW_DATT holds dSc (from S3).
* ============================================================
BKWRD_S4:
* --- Part A: dQ[i] = VTMUL(K, dSc[i]) for i=0..SEQ-1 ---
                LDD     BK_KP
                STD     MP_MAT          ; K matrix (constant across Part A)
                LDD     BK_SEQ
                STD     MP_ROW          ; SEQ rows (constant Parts A+B)
                LDD     BK_DIM
                STD     MP_COL          ; DIM cols (constant Parts A+B)

                LDX     #BW_DATT        ; X = dSc row cursor
                LDU     #BW_DQ          ; U = dQ row cursor
                LDA     BK_SEQ+1
                PSHS    A               ; outer counter; stack: [oc]

BK4A_LOOP:
                STX     MP_VIN          ; dSc[i]
                STU     MP_OUT          ; dQ[i]
                PSHS    X,U             ; preserve cursors across VTMUL
                JSR     VTMUL
                PULS    X,U

                LDD     BK_SST
                LEAX    D,X             ; dSc cursor += SEQ*2
                LDD     BK_RSZ
                LEAU    D,U             ; dQ   cursor += DIM*2

                DEC     ,S
                LBNE    BK4A_LOOP
                LEAS    1,S

* --- Part B: dK[j] = VTMUL(Q, col_j(dSc)) for j=0..SEQ-1 ---
                LDD     BK_QP
                STD     MP_MAT          ; Q matrix (constant in Part B)
                LDD     #BK_DTMP
                STD     MP_VIN          ; always BK_DTMP (constant in Part B)
*                                       ; MP_ROW/MP_COL unchanged from Part A

                LDX     #BW_DATT        ; X = column-j start (j=0 -> &dSc[0][0])
                LDU     #BW_DK          ; U = dK row cursor
                LDA     BK_SEQ+1
                PSHS    A               ; outer counter; stack: [oc]

BK4B_LOOP:
                PSHS    X               ; save column-j start; stack: [X(2) oc(1)]
                BSR     BK4_COL         ; extract col j -> BK_DTMP (clobbers X,Y,B)
                STU     MP_OUT          ; dK[j]
                PSHS    U               ; save dK cursor; stack: [U(2) X(2) oc(1)]
                JSR     VTMUL
                PULS    U               ; stack: [X(2) oc(1)]

                LDD     BK_RSZ
                LEAU    D,U             ; dK cursor += DIM*2
                PULS    X               ; restore column-j start; stack: [oc(1)]
                LEAX    2,X             ; next column (j+1) starts 2 bytes later

                DEC     ,S
                LBNE    BK4B_LOOP
                LEAS    1,S
                RTS

* --- BK4_COL: extract column from dSc into BK_DTMP (leaf sub) ---
* Entry: X = &dSc[0][j] (column-j start).
* Exit:  BK_DTMP holds SEQ words = column j. X advanced past last row.
* Clobbers: X, Y, B, D, CC. Stack balanced.
BK4_COL:
                LDY     #BK_DTMP
                LDB     BK_SEQ+1
                PSHS    B               ; inner counter
BK4_COL_LOOP:
                LDD     ,X              ; dSc[row][j]
                STD     ,Y++            ; append to BK_DTMP, advance Y
                LDD     BK_SST
                LEAX    D,X             ; advance to next row (stride = SEQ*2)
                DEC     ,S
                BNE     BK4_COL_LOOP
                LEAS    1,S
                RTS

* ============================================================
* BKWRD_S5 — Step 5: Backward projections -> dX, dWq, dWk, dWv
* ============================================================
* dX = VCPY(dY)                      ; residual init
*
* For each position i (BK_OFF = i * BK_RSZ advances per outer):
*   dX[i] += MVADD(Wq, dQ[i])        ; vout += mat * vin
*   dWq   += OUTER(X[i], dQ[i])      ; mat += vx(outer)vy
*   dX[i] += MVADD(Wk, dK[i])
*   dWk   += OUTER(X[i], dK[i])
*   dX[i] += MVADD(Wv, dV[i])
*   dWv   += OUTER(X[i], dV[i])
*
* MP_ROW=MP_COL=DIM for all 6 calls (square weight matrices) — set
* once outside the loop. Only MP_MAT/MP_VIN/MP_OUT change per call.
*
* All pointer addresses derived as base + BK_OFF where base is either
* a caller-provided pointer (BK_XX, BK_DWQ, BK_DWK, BK_DWV) or a
* fixed workspace address (#BW_DQ, #BW_DK, #BW_DV, #BW_DX).
*
* dWq/dWk/dWv are caller-zeroed (ZEROG contract, same as dWout).
*
* Entry: BW_DQ/DK/DV filled by S2+S4. BK_DY holds dY from S1.
* ============================================================
BKWRD_S5:
* --- dX = VCPY(dY) — residual pathway init ---
                LDD     BK_DY
                STD     V_SRC
                LDD     #BW_DX
                STD     V_DST
                LDD     BK_SEQ
                MULD    BK_DIM          ; W = SEQ*DIM (word count)
                TFR     W,D
                STD     V_LEN
                JSR     VCPY

* --- Constants for all 6 primitive calls ---
                LDD     BK_DIM
                STD     MP_ROW          ; DIM rows
                STD     MP_COL          ; DIM cols

* --- Init BK_OFF = 0 (position offset advances by BK_RSZ per outer) ---
                CLRA
                CLRB
                STD     BK_OFF

                LDA     BK_SEQ+1
                PSHS    A               ; outer counter; stack: [oc]

BK5_LOOP:
* --- Q path: dX[i] += MVADD(Wq, dQ[i]) ---
                LDD     BK_WQ
                STD     MP_MAT
                LDD     BK_OFF
                ADDD    #BW_DQ
                STD     MP_VIN          ; dQ[i] = BW_DQ + BK_OFF
                LDD     BK_OFF
                ADDD    #BW_DX
                STD     MP_OUT          ; dX[i] = BW_DX + BK_OFF
                JSR     MVADD

* --- Q path: dWq += OUTER(X[i], dQ[i]) ---
                LDD     BK_DWQ
                STD     MP_MAT
                LDD     BK_OFF
                ADDD    BK_XX
                STD     MP_VIN          ; X[i] = BK_XX + BK_OFF (vx)
                LDD     BK_OFF
                ADDD    #BW_DQ
                STD     MP_OUT          ; dQ[i] (vy)
                JSR     OUTER

* --- K path: dX[i] += MVADD(Wk, dK[i]) ---
                LDD     BK_WK
                STD     MP_MAT
                LDD     BK_OFF
                ADDD    #BW_DK
                STD     MP_VIN          ; dK[i]
                LDD     BK_OFF
                ADDD    #BW_DX
                STD     MP_OUT          ; dX[i]
                JSR     MVADD

* --- K path: dWk += OUTER(X[i], dK[i]) ---
                LDD     BK_DWK
                STD     MP_MAT
                LDD     BK_OFF
                ADDD    BK_XX
                STD     MP_VIN          ; X[i]
                LDD     BK_OFF
                ADDD    #BW_DK
                STD     MP_OUT          ; dK[i]
                JSR     OUTER

* --- V path: dX[i] += MVADD(Wv, dV[i]) ---
                LDD     BK_WV
                STD     MP_MAT
                LDD     BK_OFF
                ADDD    #BW_DV
                STD     MP_VIN          ; dV[i]
                LDD     BK_OFF
                ADDD    #BW_DX
                STD     MP_OUT          ; dX[i]
                JSR     MVADD

* --- V path: dWv += OUTER(X[i], dV[i]) ---
                LDD     BK_DWV
                STD     MP_MAT
                LDD     BK_OFF
                ADDD    BK_XX
                STD     MP_VIN          ; X[i]
                LDD     BK_OFF
                ADDD    #BW_DV
                STD     MP_OUT          ; dV[i]
                JSR     OUTER

* --- Advance BK_OFF by BK_RSZ for next position ---
                LDD     BK_OFF
                ADDD    BK_RSZ
                STD     BK_OFF

                DEC     ,S
                LBNE    BK5_LOOP
                LEAS    1,S
                RTS

* ============================================================
* BKWRD_S6 — Step 6: Backward embedding -> d_tok_emb, d_pos_emb
* ============================================================
* For each position i:
*   tok = tokens[i]                 (read from BK_TOKS, advance by 2)
*   d_tok[tok][d] += dX[i][d]       (scatter-add, clamp per element)
*   d_pos[i][d]   += dX[i][d]       (direct-add,  clamp per element)
*
* Clamp pattern (matches VSADD stage 2 exactly): PSHS D / ADDD ,Y /
* BVS overflow-path / on-overflow check sign of the saved addend
* (TSTA of its high byte) to pick +32767 or -32768.
*
* Cursors:
*   X   = dX[i] base ptr. Saved on stack, reloaded between the two
*         inner loops (token scatter + pos add both traverse dX[i]).
*         After both inners, X is at dX[i+1] (ready for next outer).
*   U   = BK_TOKS cursor (advances by 2 per outer).
*   BK_OFF = d_pos[i] cursor (advances by BK_RSZ per outer, stored
*         back to BK_OFF since registers are scarce).
*   Y   = target cursor (d_tok row for token inner, d_pos row for
*         pos inner).
*
* d_tok/d_pos are caller-zeroed (same contract as dWout, dWq/dWk/dWv).
*
* Entry: BW_DX holds dX from S5. BK_TOKS points at input tokens.
* ============================================================
BKWRD_S6:
                LDX     #BW_DX          ; dX[i] base cursor
                LDU     BK_TOKS         ; tokens cursor
                LDD     BK_DPSE
                STD     BK_OFF          ; d_pos cursor (reuse BK_OFF)

                LDA     BK_SEQ+1
                PSHS    A               ; outer counter; stack: [oc]

BK6_LOOP:
                PSHS    X               ; save dX[i] base; stack: [dX_base(2) oc(1)]

* --- Token scatter: Y = &d_tok[tokens[i]] ---
                LDD     ,U++            ; D = tokens[i], advance tokens cursor
                MULD    BK_RSZ          ; Q = token * BK_RSZ; W = low 16 = byte offset
                TFR     W,D
                ADDD    BK_DTKE         ; D = BK_DTKE + token*BK_RSZ
                TFR     D,Y             ; Y = &d_tok[token]

                LDB     BK_DIM+1
                PSHS    B               ; inner counter; stack: [ic dX_base(2) oc(1)]
BK6_TOK:
                LDD     ,X++            ; D = dX[i][d], advance X
                PSHS    D               ; save addend; [addend(2) ic dX_base(2) oc(1)]
                ADDD    ,Y              ; D = dX + d_tok[Y]; V set on signed overflow
                BVS     BK6_TOK_OV
                STD     ,Y++            ; no overflow: store, advance Y
                LEAS    2,S             ; drop saved addend
                BRA     BK6_TOK_NX
BK6_TOK_OV:
                PULS    D               ; restore addend for sign test
                TSTA                    ; sign of addend (dX element)
                BMI     BK6_TOK_NG
                LDD     #32767          ; positive overflow
                BRA     BK6_TOK_ST
BK6_TOK_NG:
                LDD     #-32768         ; negative overflow
BK6_TOK_ST:
                STD     ,Y++
BK6_TOK_NX:
                DEC     ,S              ; ic
                BNE     BK6_TOK
                LEAS    1,S             ; drop ic; stack: [dX_base(2) oc(1)]

* --- Pos direct-add: Y = &d_pos[i], reload X to dX[i] base ---
                LDX     ,S              ; peek: X = dX[i] base
                LDY     BK_OFF          ; Y = d_pos[i] cursor
                LDB     BK_DIM+1
                PSHS    B               ; inner counter; stack: [ic dX_base(2) oc(1)]
BK6_POS:
                LDD     ,X++
                PSHS    D
                ADDD    ,Y
                BVS     BK6_POS_OV
                STD     ,Y++
                LEAS    2,S
                BRA     BK6_POS_NX
BK6_POS_OV:
                PULS    D
                TSTA
                BMI     BK6_POS_NG
                LDD     #32767
                BRA     BK6_POS_ST
BK6_POS_NG:
                LDD     #-32768
BK6_POS_ST:
                STD     ,Y++
BK6_POS_NX:
                DEC     ,S
                BNE     BK6_POS
                LEAS    1,S             ; drop ic; stack: [dX_base(2) oc(1)]

* --- End of outer: drop dX_base (X is already at dX[i+1]); advance d_pos ---
                LEAS    2,S             ; drop dX_base; stack: [oc]
                LDD     BK_OFF
                ADDD    BK_RSZ
                STD     BK_OFF          ; d_pos cursor += DIM*2

                DEC     ,S
                LBNE    BK6_LOOP
                LEAS    1,S
                RTS

BKWRD_DONE:
                RTS

* ============================================================
* WUPDT_ONE — SGD update for one weight group (Q16 split hi/lo)
* ============================================================
* Caller sets: UP_WHI, UP_WLO, UP_PTR (gradient), UP_CNT, UP_SHF.
*
* For each element i:
*   delta = grad[i] >> (UP_SHF - 1)    ; arithmetic right shift
*   grad[i] = 0                          ; zero immediately after read
*   w_hi:w_lo -= delta                   ; 32-bit subtract via NEG+ADD
*
* 32-bit subtract sequence (matches ATTN/11 UPDAT.MAC:133-147):
*   D = -delta       (NEGD gives two's complement)
*   W = D            (TFR D,W preserves -delta for sign test later,
*                     since ADDD will destroy both D and the V/N flags)
*   w_lo += D        (ADDD ,Y / STD ,Y++; sets carry if unsigned overflow)
*   if carry: w_hi += 1  (BCC/LDD/ADDD #1/STD)
*   if -delta < 0 (TSTW / BPL): w_hi -= 1  ; sign extension
*
* The sign-extension step handles the high-word propagation for
* negative -delta (i.e. positive delta, where w -= positive reduces w).
*
* PDP-11 version keeps -delta in R0 across the ADD (ADD doesn't clobber
* source). 6309 ADDD clobbers D, so we preserve -delta in W.
*
* Gradient zero: LDD ,U followed by two CLR ,U+ (1-byte increments
* total 2 bytes) mirrors PDP-11 MOV (R4),R0 / CLR (R4)+. Belt-and-
* suspenders with ZEROG's bulk zero.
*
* UP_SHF >= 1 guaranteed (shift count = UP_SHF - 1 can be 0 for LR.ATN).
* If UP_SHF == 0, DECA wraps to 255 and the ASRD loop runs too long;
* caller contract prevents this.
*
* Clobbers: X, Y, U, D, W, E, CC. Counter: stack byte (UP_CNT <= 255).
* ============================================================
WUPDT_ONE:
                LDD     UP_CNT
                LBEQ    WU_DONE         ; zero-length guard

                LDX     UP_WHI
                LDY     UP_WLO
                LDU     UP_PTR          ; gradient array cursor

* Pre-compute shift count (UP_SHF - 1) and push onto stack
                LDA     UP_SHF+1
                DECA                    ; A = shift count (0..5 for our LR constants)
                PSHS    A               ; stack: [shf]

                LDA     UP_CNT+1
                PSHS    A               ; stack: [cnt shf]

WU_LOOP:
* --- Read gradient, zero slot ---
                LDD     ,U              ; D = grad[i] (Q15)
                CLR     ,U+             ; zero hi byte, advance U by 1
                CLR     ,U+             ; zero lo byte, advance U by 1

* --- Apply shift: delta = grad >> shift_count ---
                LDE     1,S             ; E = shift count (at 1,S)
                BEQ     WU_NOSHF
WU_SHF:
                ASRD                    ; signed >>1
                DECE
                BNE     WU_SHF
WU_NOSHF:

* --- 32-bit subtract: w -= delta via NEG + ADD + carry + sign-ext ---
                NEGD                    ; D = -delta
                TFR     D,W             ; W = -delta (preserved for sign test)
                ADDD    ,Y              ; D = (-delta) + w_lo; carry set on unsigned overflow
                STD     ,Y++            ; store new w_lo, advance Y
                BCC     WU_NOCY
                LDD     ,X              ; w_hi += 1 (carry propagation)
                ADDD    #1
                STD     ,X
WU_NOCY:
                TSTW                    ; test sign of preserved -delta
                BPL     WU_NOSEX        ; -delta >= 0 → no sign extension
                LDD     ,X              ; -delta < 0 → w_hi -= 1 (sign extension)
                SUBD    #1
                STD     ,X
WU_NOSEX:
                LEAX    2,X             ; advance w_hi cursor

                DEC     ,S              ; inner counter
                BNE     WU_LOOP
                LEAS    2,S             ; drop counter and shift_cnt
WU_DONE:
                RTS

* ============================================================
* CVT16_ONE — Convert Q16 hi:lo pairs to Q8 (one weight group)
* ============================================================
* Caller sets: UP_WHI, UP_WLO, UP_PTR (Q8 destination), UP_CNT.
*
* Q8 = middle 16 bits of Q16 = bits [23:8] of (hi << 16 | lo).
* On 6309 big-endian memory:
*   hi word bytes: [hi_msb(bits31-24), hi_lsb(bits23-16)]
*   lo word bytes: [lo_msb(bits15-8),  lo_lsb(bits7-0)]
*   Q8 high byte = hi_lsb @ (X+1)
*   Q8 low  byte = lo_msb @ (Y+0)
*
* No clamp (matches ATTN/11 ASHC #-8). Trusts Q16 stays within Q15 range.
*
* Clobbers: X, Y, U, D, CC. Counter: stack byte.
* ============================================================
CVT16_ONE:
                LDD     UP_CNT
                LBEQ    CV_DONE

                LDX     UP_WHI
                LDY     UP_WLO
                LDU     UP_PTR          ; Q8 destination cursor

                LDA     UP_CNT+1
                PSHS    A               ; counter

CV_LOOP:
                LDA     1,X             ; A = hi_lsb (bits 23:16)
                LDB     ,Y              ; B = lo_msb (bits 15:8)
                STD     ,U++            ; Q8 = bits [23:8]
                LEAX    2,X             ; advance hi cursor
                LEAY    2,Y             ; advance lo cursor
                DEC     ,S
                BNE     CV_LOOP
                LEAS    1,S
CV_DONE:
                RTS

* ============================================================
* INITW_ONE — Random Q8 init, converted to Q16 (one weight group)
* ============================================================
* Caller sets: UP_WHI, UP_WLO, UP_CNT. UP_PTR unused.
*
* For each element:
*   JSR RAND        → D = [0, 32767]
*   ANDD #$00FF     → D = [0, 255]
*   SUBD #128       → D = [-128, 127] (Q8 signed 16-bit)
*   TFR D,W         → save Q8 for lo construction + sign test
*   Sign extension → hi = 0 or -1 based on A (sign byte)
*   lo = Q8 * 256  → TFR B,A / CLRB constructs [byte, 0]
*
* Matches ATTN/11 INITW/IN.FIL (UPDAT.MAC:58-68).
*
* Clobbers: X, Y, D, W, CC, RN_SED (via RAND). Counter: stack byte.
* ============================================================
INITW_ONE:
                LDD     UP_CNT
                LBEQ    IW_DONE

                LDX     UP_WHI
                LDY     UP_WLO

                LDA     UP_CNT+1
                PSHS    A

IW_LOOP:
                JSR     RAND            ; D = random 15-bit
                ANDD    #$00FF          ; D = [0, 255]
                SUBD    #128            ; D = Q8 signed [-128, 127]
                TFR     D,W             ; W = Q8 (preserve for lo step)

* hi = sign extension of Q8 (0 for non-negative, $FFFF for negative)
                TSTA                    ; test sign byte
                BMI     IW_NEG
                CLRA
                CLRB                    ; D = 0 (positive hi)
                BRA     IW_HI_ST
IW_NEG:
                LDD     #$FFFF          ; D = -1 (negative hi)
IW_HI_ST:
                STD     ,X++            ; store hi, advance

* lo = Q8 << 8 = (value_byte, 0). Recover Q8 from W.
                TFR     W,D             ; D = original Q8
                TFR     B,A             ; A = value byte (was B)
                CLRB                    ; D = [byte, 0] = Q8 * 256
                STD     ,Y++            ; store lo, advance

                DEC     ,S
                BNE     IW_LOOP
                LEAS    1,S
IW_DONE:
                RTS

* ============================================================
* ZEROG — Zero all long-lived gradient buffers (6 VCLR calls)
* ============================================================
* Zeros: dWout, dWq, dWk, dWv, d_tok_emb, d_pos_emb. Each buffer
* pointer comes from the corresponding BKWRD parameter field.
* Does NOT zero dY/dQ/dK/dV/dX/dA — those are written fresh each
* BKWRD (VTMUL clears, VCPY initializes dX, Step 2 VCLRs dV).
*
* Word counts: DIM*VOC, DIM*DIM (x3), VOC*DIM, SEQ*DIM.
*
* Reuses V_LEN across same-size calls (Wk, Wv share Wq's count).
*
* Clobbers: X, D, W, CC, V_DST, V_LEN.
* ============================================================
ZEROG:
* dWout (DIM * VOC)
                LDD     BK_DWOUT
                STD     V_DST
                LDD     BK_DIM
                MULD    BK_VOC
                TFR     W,D
                STD     V_LEN
                JSR     VCLR

* dWq (DIM * DIM)
                LDD     BK_DWQ
                STD     V_DST
                LDD     BK_DIM
                MULD    BK_DIM
                TFR     W,D
                STD     V_LEN
                JSR     VCLR

* dWk (same size as dWq)
                LDD     BK_DWK
                STD     V_DST
                JSR     VCLR

* dWv (same size)
                LDD     BK_DWV
                STD     V_DST
                JSR     VCLR

* d_tok_emb (VOC * DIM)
                LDD     BK_DTKE
                STD     V_DST
                LDD     BK_VOC
                MULD    BK_DIM
                TFR     W,D
                STD     V_LEN
                JSR     VCLR

* d_pos_emb (SEQ * DIM)
                LDD     BK_DPSE
                STD     V_DST
                LDD     BK_SEQ
                MULD    BK_DIM
                TFR     W,D
                STD     V_LEN
                JSR     VCLR

                RTS

* ============================================================
* RAND — 15-bit LCG pseudo-random number generator
* ============================================================
* seed = ((seed * RN_MUL) + RN_ADD) & RN_MASK    ; RN_MASK = $7FFF
* Returns D = new seed in [0, 32767]. State: RN_SED.
* Matches ATTN/11 UPDAT.MAC:10-18.
*
* MULD reads multiplicand from memory; the FDB constant lives
* adjacent to the RTS but isn't executed.
*
* Clobbers: D, W, CC, RN_SED.
* ============================================================
RAND:
                LDD     RN_SED
                MULD    RN_MUL_K        ; Q = seed * RN_MUL. D = HIGH 16, W = LOW 16.
                TFR     W,D             ; D = low 16 (the part we want mod 2^16)
                ADDD    #RN_ADD         ; D += 13849
                ANDA    #$7F            ; clear bit 15 (mask to $7FFF)
                STD     RN_SED
                RTS
RN_MUL_K:       FDB     RN_MUL          ; constant for MULD (non-executed data)

* ============================================================
* GENSM — Generate one training sample (digit reversal)
* ============================================================
* Fills TOKENS[0..SEQ-1] with 8 random digits (each mod 10).
* Fills TARGET[0..SEQ-1] with reversed TOKENS.
* Mod-10 via DIVQ of zero-extended 32-bit value.
* ============================================================
GENSM:
                LDX     #TOKENS
                LDA     #SEQ
                PSHS    A
GS_LP:
                JSR     RAND            ; D = [0, 32767]
                TFR     D,W             ; W = value (low 16 of Q)
                LDD     #0              ; D = 0 (high 16 of Q); Q = zero-extended value
                DIVQ    GS_TEN          ; W = quot, D = remainder (0-9)
                STD     ,X++            ; store digit as word
                DEC     ,S
                BNE     GS_LP
                LEAS    1,S

* Reverse TOKENS into TARGET
                LDX     #TOKENS+(SEQ-1)*2
                LDY     #TARGET
                LDA     #SEQ
                PSHS    A
GS_RV:
                LDD     ,X
                STD     ,Y++
                LEAX    -2,X
                DEC     ,S
                BNE     GS_RV
                LEAS    1,S
                RTS
GS_TEN:         FDB     10

* ============================================================
* CLEAR_SCREEN — fill $0400-$05FF with CoCo3 VDG "normal space"
* ($60). Rest cursor to $0400.
* ============================================================
* CoCo3 VDG text mode: $60 = normal space (dark background),
* $20 = inverse space (green block). $60 fill keeps unwritten
* cells dark. PUTC writes raw ASCII: letters $41-$5A are normal
* uppercase in VDG; digits $30-$39 display as inverse digits.
CLEAR_SCREEN:
                LDX     #SCREEN
                LDD     #$6060          ; CoCo3 normal-space fill
CLS_LP:
                STD     ,X++
                CMPX    #SCREEN+512
                BNE     CLS_LP
                LDD     #SCREEN
                STD     IO_CURS
                RTS

* ============================================================
* PUTC — write character in B to screen at IO_CURS, advance cursor
* ============================================================
PUTC:
                LDX     IO_CURS
                STB     ,X+
                STX     IO_CURS
                RTS

* ============================================================
* PUTS — print null-terminated string at X
* Clobbers: X, B, CC. Advances X past the terminator.
* ============================================================
PUTS:
                LDB     ,X+
                BEQ     PUTS_DONE
                PSHS    X               ; preserve X across PUTC (PUTC clobbers X)
                JSR     PUTC
                PULS    X
                BRA     PUTS
PUTS_DONE:
                RTS

* ============================================================
* NEWLINE — advance IO_CURS to next row start (32-col aligned)
* Next-row start = (cursor | $1F) + 1
* ============================================================
NEWLINE:
                LDD     IO_CURS
                ORB     #$1F
                ADDD    #1
                STD     IO_CURS
                RTS

* ============================================================
* PUTDEC — print 16-bit unsigned D as decimal (no leading zeros)
* ============================================================
* Push digits via DIVQ by 10, then pop and print. Always prints
* at least one digit (handles D=0).
* ============================================================
PUTDEC:
                LDX     #0              ; sentinel marker
                PSHS    X               ; push 2-byte zero sentinel
PD_DIV:
                TFR     D,W
                LDD     #0
                DIVQ    PD_TEN          ; W = quot, D = remainder (0-9)
                ADDB    #'0
                PSHS    B               ; push ASCII digit
                TFR     W,D             ; D = quotient (TFR does not set flags)
                CMPD    #0              ; explicit flag update
                BNE     PD_DIV

* Pop and print until sentinel
* PULS B does NOT set Z flag on 6309 — need explicit TSTB.
PD_POP:
                PULS    B
                TSTB
                BEQ     PD_FIN
                JSR     PUTC
                BRA     PD_POP
PD_FIN:
                PULS    B               ; drop second byte of sentinel
                RTS
PD_TEN:         FDB     10

* ============================================================
* PUTLSS — print Q12 loss value in D as "N.NNNN"
* ============================================================
* Integer part = D >> 12 (max ~5 for losses).
* Fractional = (D & $0FFF) * 10000 / 4096, printed as 4 digits.
* Result-of-MULD bit extraction: bits [27:12] of 32-bit product.
* ============================================================
PUTLSS:
                PSHS    D               ; save Q12 value

* --- Integer part ---
                LSRD
                LSRD
                LSRD
                LSRD
                LSRD
                LSRD
                LSRD
                LSRD
                LSRD
                LSRD
                LSRD
                LSRD                    ; 12 LSRDs => D >>= 12
                JSR     PUTDEC

                LDB     #'.
                JSR     PUTC

* --- Fractional part ---
                PULS    D
                ANDD    #$0FFF          ; keep low 12 bits
                MULD    PL_10K          ; Q = frac * 10000
                STQ     MULSCR          ; save 4 bytes

* Extract bits [27:12] of the 32-bit product as a 16-bit result.
* Layout: bits 27-24 = b0[3:0], bits 23-16 = b1, bits 15-12 = b2[7:4].
* Algorithm: LDD MULSCR+1 gives D = b1:b2 (bits 23:8 of P).
* LSRD*4 shifts D right by 4 → D holds ((b1>>4):((b1<<4)|(b2>>4))).
* Then OR in b0's low nibble (shifted to A's high nibble) for bits 15-12.
                LDA     MULSCR          ; A = b0 (bits 31-24 of product)
                ANDA    #$0F            ; keep low nibble of b0
                LSLA
                LSLA
                LSLA
                LSLA                    ; A = (b0 & 0x0F) << 4
                PSHS    A               ; save high-nibble contribution

                LDD     MULSCR+1        ; D = b1:b2 (bits 23-8 of product)
                LSRD
                LSRD
                LSRD
                LSRD                    ; D = (b1:b2) >> 4

                ORA     ,S+             ; A |= saved high-nibble contribution
* D now = bits [27:12] of product = fractional 0..9999

* Split into 4 digits via repeated DIVQ by 10, push
                TFR     D,W
                LDD     #0
                DIVQ    PL_TEN          ; W = D/10, D = D%10 (ones)
                PSHS    D
                TFR     W,D
                TFR     D,W
                LDD     #0
                DIVQ    PL_TEN          ; tens
                PSHS    D
                TFR     W,D
                TFR     D,W
                LDD     #0
                DIVQ    PL_TEN          ; hundreds
                PSHS    D
                TFR     W,D
                TFR     D,W
                LDD     #0
                DIVQ    PL_TEN          ; thousands (W=0 after; D = digit)
                PSHS    D

* Pop and print 4 digits (thousands, hundreds, tens, ones)
                PULS    D               ; thousands
                ADDB    #'0
                JSR     PUTC
                PULS    D
                ADDB    #'0
                JSR     PUTC
                PULS    D
                ADDB    #'0
                JSR     PUTC
                PULS    D
                ADDB    #'0
                JSR     PUTC
                RTS
PL_10K:         FDB     10000
PL_TEN:         FDB     10

* ============================================================
* FORWRD — Forward pass: EMBED -> ATTN -> PROJ
* ============================================================
* Sets parameter blocks from production addresses and architecture
* constants, then invokes the three composites in sequence.
* Buffer handoff: TOKENS -> EMBED -> FC_X -> ATTN -> FC_Y -> PROJ -> FC_LOGITS.
* ============================================================
FORWRD:
* --- EMBED ---
                LDD     #TOKENS
                STD     EM_TOK
                LDD     #W_TKEQ8
                STD     EM_TKE
                LDD     #W_PSEQ8
                STD     EM_PSE
                LDD     #FC_X
                STD     EM_OUT
                LDD     #SEQ
                STD     EM_SEQ
                LDD     #D
                STD     EM_DIM
                JSR     EMBED

* --- ATTN ---
                LDD     #FC_X
                STD     AT_XIN
                LDD     #W_WQQ8
                STD     AT_WQ
                LDD     #W_WKQ8
                STD     AT_WK
                LDD     #W_WVQ8
                STD     AT_WV
                LDD     #FC_Y
                STD     AT_YOT
                LDD     #FC_WORK
                STD     AT_WRK
                LDD     #SEQ
                STD     AT_SEQ
                LDD     #D
                STD     AT_DIM
                LDD     #ARCH_SHF
                STD     AT_SHF
                JSR     ATTN

* --- PROJ ---
                LDD     #FC_Y
                STD     PR_Y
                LDD     #W_WOTQ8
                STD     PR_WOUT
                LDD     #FC_LOGITS
                STD     PR_LOG
                LDD     #SEQ
                STD     PR_SEQ
                LDD     #D
                STD     PR_DIM
                LDD     #V
                STD     PR_VOC
                JSR     PROJ
                RTS

* ============================================================
* BKWRD_SETUP — configure BK_* fields from production addresses
* ============================================================
BKWRD_SETUP:
                LDD     #FC_LOGITS
                STD     BK_LOG
                LDD     #TARGET
                STD     BK_TGT
                LDD     #FC_Y
                STD     BK_YY
                LDD     #W_WOTQ8
                STD     BK_WOUT
                LDD     #FC_X
                STD     BK_XX
                LDD     #W_WQQ8
                STD     BK_WQ
                LDD     #W_WKQ8
                STD     BK_WK
                LDD     #W_WVQ8
                STD     BK_WV
                LDD     #FC_WORK
                STD     BK_WRK
                LDD     #G_WOUT
                STD     BK_DWOUT
                LDD     #BW_DY
                STD     BK_DY
                LDD     #SEQ
                STD     BK_SEQ
                LDD     #D
                STD     BK_DIM
                LDD     #V
                STD     BK_VOC
                LDD     #ARCH_SHF
                STD     BK_SHF
                LDD     #TOKENS
                STD     BK_TOKS
                LDD     #G_WQ
                STD     BK_DWQ
                LDD     #G_WK
                STD     BK_DWK
                LDD     #G_WV
                STD     BK_DWV
                LDD     #G_TKE
                STD     BK_DTKE
                LDD     #G_PSE
                STD     BK_DPSE
                RTS

* ============================================================
* CLOSS_SETUP — configure CL_* parameter block
* ============================================================
CLOSS_SETUP:
                LDD     #FC_LOGITS
                STD     CL_LOG
                LDD     #TARGET
                STD     CL_TGT
                LDD     #SEQ
                STD     CL_SEQ
                LDD     #V
                STD     CL_VOC
                RTS

* ============================================================
* Weight group descriptor table
* 6 groups x 12 bytes/entry: hi, lo, q8, grad, count, lr_shift
* ============================================================
WG_TABLE:
                FDB     W_TKEH,W_TKEL,W_TKEQ8,G_TKE,160,LR_SHF_EMB
                FDB     W_PSEH,W_PSEL,W_PSEQ8,G_PSE,128,LR_SHF_EMB
                FDB     W_WQH,W_WQL,W_WQQ8,G_WQ,256,LR_SHF_ATN
                FDB     W_WKH,W_WKL,W_WKQ8,G_WK,256,LR_SHF_ATN
                FDB     W_WVH,W_WVL,W_WVQ8,G_WV,256,LR_SHF_ATN
                FDB     W_WOTH,W_WOTL,W_WOTQ8,G_WOUT,160,LR_SHF_OUT
WG_END:
WG_ESZ          EQU     12              ; bytes per entry

* ============================================================
* INITW_ALL — initialise all 6 weight groups with random Q16
* ============================================================
INITW_ALL:
                LDX     #WG_TABLE
INA_LP:
                LDD     ,X              ; hi ptr
                STD     UP_WHI
                LDD     2,X             ; lo ptr
                STD     UP_WLO
                LDD     8,X             ; count
                STD     UP_CNT
                PSHS    X
                JSR     INITW_ONE
                PULS    X
                LEAX    WG_ESZ,X
                CMPX    #WG_END
                BNE     INA_LP
                RTS

* ============================================================
* CVT16_ALL — convert Q16 -> Q8 for all 6 weight groups
* ============================================================
CVT16_ALL:
                LDX     #WG_TABLE
CVA_LP:
                LDD     ,X
                STD     UP_WHI
                LDD     2,X
                STD     UP_WLO
                LDD     4,X             ; q8 dest
                STD     UP_PTR
                LDD     8,X             ; count
                STD     UP_CNT
                PSHS    X
                JSR     CVT16_ONE
                PULS    X
                LEAX    WG_ESZ,X
                CMPX    #WG_END
                BNE     CVA_LP
                RTS

* ============================================================
* WUPDT_ALL — SGD update all 6 weight groups (per-group lr_shift)
* ============================================================
WUPDT_ALL:
                LDX     #WG_TABLE
WUA_LP:
                LDD     ,X
                STD     UP_WHI
                LDD     2,X
                STD     UP_WLO
                LDD     6,X             ; grad ptr
                STD     UP_PTR
                LDD     8,X
                STD     UP_CNT
                LDD     10,X            ; lr_shift
                STD     UP_SHF
                PSHS    X
                JSR     WUPDT_ONE
                PULS    X
                LEAX    WG_ESZ,X
                CMPX    #WG_END
                BNE     WUA_LP
                RTS

* ============================================================
* COUNT — per-position argmax comparison, update TR_HIT / TR_TOT
* ============================================================
* For each i in 0..SEQ-1:
*   VMAX(logits[i], VOC) -> X = argmax index
*   if X == TARGET[i]: TR_HIT += 1
*   TR_TOT += 1
* Uses VMAX (vecop.asm): entry X=vector, B=count; exit X=index.
* ============================================================
COUNT:
                LDU     #FC_LOGITS      ; logits row cursor (advances by V*2)
                LDY     #TARGET
                LDA     #SEQ
                PSHS    A               ; outer counter
CT_LP:
                TFR     U,X             ; X = logits[i] (VMAX mutates X)
                LDB     #V
                JSR     VMAX            ; X = argmax index, D = max value
                LDD     ,Y++            ; D = target[i]
                PSHS    D
                TFR     X,D             ; D = predicted index
                CMPD    ,S++            ; pop target, compare
                BNE     CT_NM

* Match: TR_HIT++
                LDD     TR_HIT
                ADDD    #1
                STD     TR_HIT
CT_NM:
* Always: TR_TOT++
                LDD     TR_TOT
                ADDD    #1
                STD     TR_TOT

                LEAU    2*V,U           ; next logits row
                DEC     ,S
                BNE     CT_LP
                LEAS    1,S
                RTS

* ============================================================
* REPORT — print "STEP NNN LOSS N.NNNN ACC N/N" and reset counts
* ============================================================
* Called when step mod RPRT == 0. Internally invokes CLOSS.
* After printing, TR_HIT and TR_TOT are reset to 0.
* ============================================================
REPORT:
                JSR     CLOSS_SETUP
                JSR     CLOSS           ; D = Q12 loss
                PSHS    D               ; save

                LDX     #STR_STEP
                JSR     PUTS
                LDD     TR_STEP
                JSR     PUTDEC

                LDX     #STR_LOSS
                JSR     PUTS
                PULS    D
                JSR     PUTLSS

                LDX     #STR_ACC
                JSR     PUTS
                LDD     TR_HIT
                JSR     PUTDEC
                LDB     #'/
                JSR     PUTC
                LDD     TR_TOT
                JSR     PUTDEC
                JSR     NEWLINE

* Reset accumulators
                CLRA
                CLRB
                STD     TR_HIT
                STD     TR_TOT
                RTS

* ============================================================
* FINAL_TEST — run NTEST samples forward-only, print predictions
* ============================================================
* Format per sample: "d d d d d d d d -> p p p p p p p p OK/FAIL"
* Final line: "ACC N/NTEST"
* ============================================================
FINAL_TEST:
                CLRA
                CLRB
                STD     TE_SOK

                LDA     #NTEST
                PSHS    A               ; sample counter
FT_LP:
                JSR     GENSM
                JSR     CVT16_ALL
                JSR     FORWRD

* Print input tokens
                LDX     #TOKENS
                LDB     #SEQ
                PSHS    B
FT_IN:
                LDD     ,X++
                ADDB    #'0
                PSHS    X               ; PUTC clobbers X
                JSR     PUTC
                LDB     #$20
                JSR     PUTC
                PULS    X
                DEC     ,S
                BNE     FT_IN
                LEAS    1,S

* "-> "
                LDX     #STR_ARROW
                JSR     PUTS
                LDB     #$20            ; space
                JSR     PUTC

* Argmax per position: print predicted digit, save to TE_PRED
                LDU     #FC_LOGITS
                LDX     #TE_PRED
                LDB     #SEQ
                PSHS    B
FT_OUT:
                PSHS    X               ; save TE_PRED cursor (VMAX uses X)
                TFR     U,X             ; X = logits[i]
                LDB     #V
                JSR     VMAX            ; X = argmax
                TFR     X,D             ; D = predicted index
                PULS    X               ; restore TE_PRED cursor
                STD     ,X++            ; save predicted digit (word)
                ADDB    #'0
                PSHS    X               ; PUTC clobbers X
                JSR     PUTC
                LDB     #$20
                JSR     PUTC
                PULS    X
                LEAU    2*V,U
                DEC     ,S
                BNE     FT_OUT
                LEAS    1,S

* Check match: all 8 TE_PRED == TARGET?
                LDX     #TE_PRED
                LDY     #TARGET
                LDA     #1              ; all-match flag
                LDB     #SEQ
                PSHS    D               ; [B=counter, A=flag]
FT_CHK:
                LDD     ,X++
                CMPD    ,Y++
                BEQ     FT_EQ
                CLR     1,S             ; clear flag
FT_EQ:
                DEC     ,S
                BNE     FT_CHK
                PULS    D               ; A = flag, B = 0
                TSTA
                BEQ     FT_FAIL
                LDD     TE_SOK
                ADDD    #1
                STD     TE_SOK
                LDX     #STR_OK
                BRA     FT_PRT
FT_FAIL:
                LDX     #STR_FAIL
FT_PRT:
                JSR     PUTS
                JSR     NEWLINE

                DEC     ,S
                LBNE    FT_LP
                LEAS    1,S

* Final accuracy line: "ACC N/NTEST"
                JSR     NEWLINE
                LDX     #STR_ACC
                JSR     PUTS
                LDD     TE_SOK
                JSR     PUTDEC
                LDB     #'/
                JSR     PUTC
                LDD     #NTEST
                JSR     PUTDEC
                JSR     NEWLINE
                RTS

* ============================================================
* String labels (program-level, shared by REPORT/FINAL_TEST)
* ============================================================
STR_HEADER:     FCC     "ATTN-CC3 TRAINING"
                FCB     0
STR_STEP:       FCC     "STEP "
                FCB     0
STR_LOSS:       FCC     " LOSS "
                FCB     0
STR_ACC:        FCC     " ACC "
                FCB     0
STR_ARROW:      FCC     "->"
                FCB     0
STR_OK:         FCC     " OK"
                FCB     0
STR_FAIL:       FCC     " FAIL"
                FCB     0
STR_TESTHDR:    FCC     "FINAL TEST:"
                FCB     0

* ============================================================
* LOGTBL — 257-entry Q12 cross-entropy lookup table
* ============================================================
                INCLUDE tables/logtbl.asm
