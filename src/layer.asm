* ============================================================
* layer.asm — LAYER-level composites
* ============================================================
* Composite routines that call MATOP/VECOP/ACTFN primitives.
*
* Dependencies (must be INCLUDE'd before this file):
*   - include/equates.inc
*   - include/macros.inc
*   - src/fxmath.asm
*   - src/vecop.asm
*   - src/matop.asm  (PROJ and ATTN use VTMUL; ATTN also uses VDOT)
*   - src/actfn.asm  (ATTN uses SFTMX)
*
* Routines:
*   - EMBED  (moved from vecop.asm, was committed in 489aa61)
*   - PROJ   (added in 3fff71b)
*   - ATTN + AT_BPR helper (this file's final composite)
* ============================================================

* ============================================================
* EMBED — Token embedding + additive positional encoding
* ============================================================
* out[i][j] = tkemb[tokens[i]][j] + psemb[i][j]
* for i in 0..EM_SEQ-1, j in 0..EM_DIM-1
*
* Caller sets: EM_TOK, EM_TKE, EM_PSE, EM_OUT, EM_SEQ, EM_DIM
* Computed internally: EM_RBS (row size bytes), EM_TPTR (advancing token ptr)
*
* Exit: EM_SEQ * EM_DIM words at EM_OUT populated.
* Clobbers: X, Y, U, D, W, CC
*
* NO CLAMPING. Uses raw ADDD, matches ATTN/11 assembly behavior.
* The prototype.shf applies clamp16, but the PDP-11 assembly does
* not — this is a pre-existing ATTN/11 inconsistency, not a
* deviation we introduce. The test harness includes a wrap test
* vector that asserts wrapping behavior explicitly.
*
* Counter: stack byte counters for both outer (SEQ) and inner (D)
* loops. SEQ, D both <= 255 by architecture invariant (current:
* SEQ=8, D=16). See DEVIATIONS.md counter rule.
*
* Address math: uses MULD EM_RBS / TFR W,D to compute token row
* offset. Max value: V=10 * D*2=32 = 320 bytes, fits in W.
* If V or D grow past 255 each, reconsider overflow into D:W.
*
* Caller's EM_TOK is preserved: routine copies it to EM_TPTR and
* advances EM_TPTR during the loop. After RTS, EM_TOK is unchanged;
* EM_TPTR holds the end-of-array address (internal, discard).
* ============================================================
EMBED:
                LDW     EM_SEQ
                BEQ     EMBED_DONE      ; zero-length no-op

* Pre-compute row size in bytes
                LDD     EM_DIM
                LSLD                    ; D = D * 2 (row size in bytes)
                STD     EM_RBS

* Initialize advancing token pointer (preserves caller's EM_TOK)
                LDD     EM_TOK
                STD     EM_TPTR

* Initialize outer pointers (advance naturally across inner loop)
                LDU     EM_OUT          ; output write ptr
                LDY     EM_PSE          ; position embedding ptr

* Outer counter on stack (SEQ <= 255 by architecture invariant)
                LDA     EM_SEQ+1
                PSHS    A

EMBED_ROW:
* --- Look up token and compute tkemb row address ---
                LDX     EM_TPTR         ; X = current token-ptr
                LDD     ,X++            ; D = token ID, advance X
                STX     EM_TPTR         ; save advanced token-ptr
                MULD    EM_RBS          ; Q = token_id * row_size_bytes
                TFR     W,D             ; D = low 16 bits = offset
                ADDD    EM_TKE          ; D = tkemb base + offset = row addr
                TFR     D,X             ; X = tkemb row ptr for this token

* --- Inner loop: out[i][j] = tkemb[tok][j] + psemb[i][j] ---
                LDA     EM_DIM+1        ; DIM <= 255 by architecture invariant
                PSHS    A

EMBED_COL:
                LDD     ,X++            ; tkemb row[j]
                ADDD    ,Y++            ; + psemb row[j]  (raw ADDD, wrapping)
                STD     ,U++            ; -> out row[j]
                DEC     ,S
                BNE     EMBED_COL
                LEAS    1,S             ; drop inner counter

* Y and U already advanced to next row naturally
                DEC     ,S
                BNE     EMBED_ROW       ; short-branch attempt (body ~30 bytes)
                LEAS    1,S             ; drop outer counter

EMBED_DONE:
                RTS

* ============================================================
* PROJ — Output projection (logits)
* ============================================================
* For each position i in 0..SEQ-1:
*   logits[i][:] = Wout^T @ Y[i][:]
* where Y is [SEQ][DIM], Wout is [DIM][VOC], logits is [SEQ][VOC].
*
* Caller sets: PR_Y, PR_WOUT, PR_LOG, PR_SEQ, PR_DIM, PR_VOC
* Computed internally: PR_YST (DIM*2), PR_LST (VOC*2)
*
* Exit: SEQ * VOC words at PR_LOG populated with raw logits.
* Clobbers: X, Y, U, D, W, CC, MP_* (used as VTMUL parameters)
*
* PROJ uses VTMUL (not MVMUL) because Wout is laid out [d_model][vocab]
* — dimension-major. VTMUL computes out = Wout^T @ y, which is the
* correct projection. MVMUL would compute Wout @ y, the wrong shape.
*
* No bias, no activation, no softmax. Raw logits. Softmax is applied
* later (loss computation, gradient).
*
* Zero-length guard at entry on PR_SEQ. ATTN/11's PROJ omits this
* guard; SEQ=0 would underflow the loop counter to 65535. Our guard
* matches the standard hardening pattern across VCPY/VCLR/VSADD/EMBED.
* Documented in DEVIATIONS.md as defensive hardening, not a semantic
* deviation — under any valid input, behavior matches ATTN/11.
*
* Stack discipline: VTMUL clobbers X, Y, U, D, W, Q, CC. PROJ
* preserves Y/LOG row pointers (X, U) across the JSR VTMUL via
* PSHS X,U / PULS X,U. Outer counter on stack is at 4,S during
* the call, at ,S after PULS.
*
* Counter: stack byte counter for outer SEQ loop (SEQ <= 255 by
* architecture invariant). See DEVIATIONS.md counter rule.
* ============================================================
PROJ:
                LDW     PR_SEQ
                BEQ     PROJ_DONE       ; zero-length guard

* Pre-compute row strides once
                LDD     PR_DIM
                LSLD                    ; D = DIM * 2
                STD     PR_YST
                LDD     PR_VOC
                LSLD                    ; D = VOC * 2
                STD     PR_LST

* Set VTMUL fixed parameters (constant across all positions)
                LDD     PR_WOUT
                STD     MP_MAT
                LDD     PR_DIM
                STD     MP_ROW
                LDD     PR_VOC
                STD     MP_COL

* Initialize per-iteration row pointers
                LDX     PR_Y            ; X = current Y row ptr
                LDU     PR_LOG          ; U = current LOG row ptr

* Outer counter on stack (SEQ <= 255 by architecture invariant)
                LDA     PR_SEQ+1
                PSHS    A

* Stack at loop entry: [counter at ,S]
PROJ_LOOP:
                STX     MP_VIN          ; VTMUL vin = current Y row
                STU     MP_OUT          ; VTMUL vout = current LOG row
                PSHS    X,U             ; preserve across VTMUL
* Stack now: [X_hi X_lo U_hi U_lo counter] — counter at 4,S
                JSR     VTMUL
                PULS    X,U             ; restore — stack back to [counter]
                LDD     PR_YST
                LEAX    D,X             ; advance Y row ptr by DIM*2
                LDD     PR_LST
                LEAU    D,U             ; advance LOG row ptr by VOC*2
                DEC     ,S              ; decrement outer counter
                BNE     PROJ_LOOP       ; short branch (body ~27 bytes)
                LEAS    1,S             ; drop counter

PROJ_DONE:
                RTS

* ============================================================
* ATTN — Self-attention forward pass
* ============================================================
* Mathematical operation (matches ATTN/11 LAYER.MAC lines 134-261
* and prototype.shf lines 106-121):
*
*   Step 1:  Q[i]   = Wq^T  . X[i]          for i in 0..SEQ-1
*   Step 2:  K[i]   = Wk^T  . X[i]          for i in 0..SEQ-1
*   Step 3:  V[i]   = Wv^T  . X[i]          for i in 0..SEQ-1
*   Step 4:  S[i][j] = (Q[i] . K[j]) >> AT_SHF     for i,j in 0..SEQ-1
*   Step 5:  A[i]   = softmax(S[i])                for i in 0..SEQ-1 (in-place over S)
*   Step 6:  Y[i]   = V^T  . A[i]                  for i in 0..SEQ-1
*   Step 7:  Y[i][j] += X[i][j]                    (raw ADD, no clamp)
*
* Caller contract:
*   AT_XIN  — input X ptr [SEQ][DIM]   (e.g. post-EMBED output)
*   AT_WQ, AT_WK, AT_WV — three [DIM][DIM] weight matrices (Q8)
*   AT_YOT  — output Y ptr [SEQ][DIM]; Step 6 writes here, Step 7 adds X in-place
*   AT_WRK  — 896-byte workspace: Q (256B), K (256B), V (256B), S/A (128B)
*   AT_SEQ  — sequence length (<= 255; <= 8 for Stage 1)
*   AT_DIM  — d_model (<= 255; <= 16 for Stage 1; constrained so SEQ*DIM <= 255)
*   AT_SHF  — sqrt(d) right-shift count (= 2 for d=16)
*
* Workspace layout at AT_WRK (computed into AT_QQ/KK/VV/SS at entry):
*   AT_QQ = AT_WRK + 0            Q   [SEQ][DIM] Q8 words
*   AT_KK = AT_QQ + SEQ*DIM*2     K
*   AT_VV = AT_KK + SEQ*DIM*2     V
*   AT_SS = AT_VV + SEQ*DIM*2     S/A (shared; softmax is in-place)
*
* Exit: AT_SEQ*AT_DIM words at AT_YOT populated with attention output + residual.
* Clobbers: X, Y, U, D, W, Q, CC, all MP_*, all SF_*, all AT_* internal.
*
* AT_SHF. PDP-11 ATTN uses `ASH AT.SHF, R0` with NEG to do variable
* right shift. 6309 has no ASH; Step 4 emits an inline `ASRD/DEC/BNE`
* loop using the low byte of AT_SHF as the count (see Step 4 below).
* We do NOT negate AT_SHF on entry — it stores the raw positive shift.
*
* Counter discipline. SEQ, DIM both <= 255 and SEQ*DIM <= 255 by
* architecture invariant (Stage 1: SEQ=8, DIM=16, SEQ*DIM=128).
* All loops use stack byte counters. MULD clobbers W (= low half of Q)
* so no register-counter-across-MULD; LDD clobbers A and B so no
* register-counter-across-LDD. See DEVIATIONS.md counter rule.
*
* Divergence — Step 7 residual. The PDP-11 assembly at LAYER.MAC
* lines 252-258 uses plain ADD with SOB (no BVS, no clamp), so
* `Y[i][j] + X[i][j]` wraps on overflow. Prototype.shf line 121
* says `clamp16 (+ O X)` — a divergence between PDP-11 and prototype.
* We inherit the PDP-11 assembly behavior, matching EMBED's same
* pattern. See DEVIATIONS.md. Test harness includes a wrap vector.
*
* Zero-length guard. LDW AT_SEQ / BEQ ATTN_DONE at entry. ATTN/11's
* ATTN omits this — SEQ=0 on PDP-11 underflows DEC loops into 65535
* iterations. Defensive hardening matching EMBED/PROJ pattern. See
* DEVIATIONS.md for the general hardening convention.
* ============================================================
ATTN:
                LDW     AT_SEQ
                LBEQ    ATTN_DONE       ; zero-length guard (long branch; body >>127 B)

* --- Pre-compute derived sizes (constants for the whole call) ---
                LDD     AT_DIM
                LSLD                    ; D = DIM * 2
                STD     AT_RSZ
                LDD     AT_SEQ
                LSLD                    ; D = SEQ * 2
                STD     AT_SSZ

* --- Partition workspace into Q / K / V / S sub-regions ---
* Offset between sub-region bases = SEQ * DIM * 2 bytes.
* MULD: D = SEQ, memory = RSZ. Q = D*RSZ. W = low 16 = SEQ*DIM*2.
                LDD     AT_SEQ
                MULD    AT_RSZ          ; Q = SEQ * RSZ (W = low 16 = SEQ*DIM*2)
                TFR     W,D             ; D = sub-region size in bytes
                PSHS    D               ; save on stack
                LDD     AT_WRK
                STD     AT_QQ           ; Q base
                ADDD    ,S
                STD     AT_KK           ; K base
                ADDD    ,S
                STD     AT_VV           ; V base
                ADDD    ,S
                STD     AT_SS           ; S/A base
                LEAS    2,S             ; drop sub-region size

* ============================================================
* Step 1: Q = Wq^T @ X  (per-position VTMUL via AT_BPR helper)
* ============================================================
* AT_BPR sets MP_MAT/ROW/COL once, loops VTMUL over SEQ positions.
                LDD     AT_WQ
                STD     AT_BW
                LDD     AT_XIN
                STD     AT_BI
                LDD     AT_QQ
                STD     AT_BO
                JSR     AT_BPR

* ============================================================
* Step 2: K = Wk^T @ X
* ============================================================
                LDD     AT_WK
                STD     AT_BW
                LDD     AT_XIN
                STD     AT_BI
                LDD     AT_KK
                STD     AT_BO
                JSR     AT_BPR

* ============================================================
* Step 3: V = Wv^T @ X
* ============================================================
* End of Steps 1-3: AT_BW/BI/BO/BC are dead after AT_BPR returns.
* Aliased slots AT_QI/KJ/SI/IC become live starting Step 4.
                LDD     AT_WV
                STD     AT_BW
                LDD     AT_XIN
                STD     AT_BI
                LDD     AT_VV
                STD     AT_BO
                JSR     AT_BPR

* ============================================================
* Step 4: S[i][j] = (Q[i] . K[j]) >> AT_SHF
* ============================================================
* Doubly-nested VDOT loop. AT_QI advances per outer, AT_KJ resets
* per outer + advances per inner, AT_SI advances per inner (SEQ*SEQ
* scores written row-major), AT_IC = inner counter, AT_OC = outer.
* After VDOT returns D = Q8 clamped dot, the inline ASRD loop applies
* AT_SHF right shifts before storing to S[i][j].
                LDD     AT_QQ
                STD     AT_QI           ; Q row cursor (outer)
                LDD     AT_SS
                STD     AT_SI           ; S cell cursor (inner, row-major)
                LDA     AT_SEQ+1
                PSHS    A               ; outer counter
ATTN_S4_OUT:
                LDD     AT_KK
                STD     AT_KJ           ; K row cursor, reset per outer
                LDA     AT_SEQ+1
                PSHS    A               ; inner counter
ATTN_S4_IN:
                LDX     AT_QI           ; X = Q[i] (VDOT advances its own X)
                LDY     AT_KJ           ; Y = K[j]
                LDB     AT_DIM+1        ; B = DIM (low byte, DIM<=255)
                JSR     VDOT            ; D = (Q[i] . K[j]) >> 8 clamped Q15

* Variable-count ASR. Loop ASRD AT_SHF times. AT_SHF=0 is a no-op.
* LDE (not LDA) — D holds the VDOT score and must survive this loop
* until STD ,X++ at ATTN_S4_NOSHF. LDA/LDB would clobber D's halves.
                LDE     AT_SHF+1        ; E = shift count (preserves D)
                BEQ     ATTN_S4_NOSHF
ATTN_S4_SHF:
                ASRD                    ; signed >>1 on D
                DECE
                BNE     ATTN_S4_SHF
ATTN_S4_NOSHF:
* Store scaled score, advance S cell ptr
                LDX     AT_SI
                STD     ,X++
                STX     AT_SI
* Advance K row ptr by RSZ
                LDD     AT_KJ
                ADDD    AT_RSZ
                STD     AT_KJ
                DEC     ,S              ; inner counter
                LBNE    ATTN_S4_IN      ; long branch (body ~55 bytes, safe margin)
                LEAS    1,S
* Advance Q row ptr by RSZ
                LDD     AT_QI
                ADDD    AT_RSZ
                STD     AT_QI
                DEC     ,S              ; outer counter
                LBNE    ATTN_S4_OUT     ; long branch (body well over 127 bytes)
                LEAS    1,S

* ============================================================
* Step 5: A[i] = softmax(S[i])   (in-place, SEQ calls to SFTMX)
* ============================================================
* SF_VEC = SF_OUT = row ptr. SF_LEN = SEQ. After the loop, S/A region
* holds attention weights (same 128B as the score matrix occupied).
                LDD     AT_SEQ
                STD     SF_LEN
                LDD     AT_SS
                STD     AT_SI           ; row cursor into S/A
                LDA     AT_SEQ+1
                PSHS    A               ; outer counter
ATTN_S5:
                LDD     AT_SI
                STD     SF_VEC
                STD     SF_OUT          ; in-place: input ptr = output ptr
                JSR     SFTMX
                LDD     AT_SI
                ADDD    AT_SSZ
                STD     AT_SI
                DEC     ,S
                BNE     ATTN_S5
                LEAS    1,S

* ============================================================
* Step 6: Y[i] = V^T @ A[i]   (per-position VTMUL over SEQ)
* ============================================================
* Note: MP_ROW = SEQ (not DIM as in Steps 1-3). V has shape [SEQ][DIM];
* VTMUL sums over rows = SEQ, producing DIM outputs per call.
                LDD     AT_VV
                STD     MP_MAT
                LDD     AT_SEQ
                STD     MP_ROW          ; rows of V = SEQ  (<-- different from Steps 1-3)
                LDD     AT_DIM
                STD     MP_COL          ; cols of V = DIM
                LDD     AT_SS
                STD     AT_SI           ; A row cursor (input)
                LDD     AT_YOT
                STD     AT_YI           ; Y row cursor (output)
                LDA     AT_SEQ+1
                PSHS    A               ; outer counter
ATTN_S6:
                LDD     AT_SI
                STD     MP_VIN
                LDD     AT_YI
                STD     MP_OUT
                JSR     VTMUL
                LDD     AT_SI
                ADDD    AT_SSZ
                STD     AT_SI
                LDD     AT_YI
                ADDD    AT_RSZ
                STD     AT_YI
                DEC     ,S
                BNE     ATTN_S6
                LEAS    1,S

* ============================================================
* Step 7: Y += X   (residual; raw ADDD, no clamp, wraps on overflow)
* ============================================================
* SEQ * DIM words = 128 for Stage 1. Byte counter assumes SEQ*DIM<=255;
* header contract enforces this. Divergence from prototype.shf's
* clamp16 is inherited from ATTN/11 assembly — see DEVIATIONS.md.
                LDX     AT_YOT          ; dst (O, will become Y)
                LDU     AT_XIN          ; src (X)
                LDD     AT_SEQ
                MULD    AT_DIM          ; Q = SEQ * DIM (low 16 in W)
                TFR     W,D             ; D = SEQ * DIM (<= 255 by contract)
                PSHS    B               ; byte counter
ATTN_S7:
                LDD     ,U++            ; D = X[k]
                ADDD    ,X              ; D += Y[k]  (raw ADDD, wraps)
                STD     ,X++            ; store and advance Y ptr
                DEC     ,S
                BNE     ATTN_S7
                LEAS    1,S

ATTN_DONE:
                RTS

* ============================================================
* AT_BPR — Batch Per-Row helper used by Steps 1-3
* ============================================================
* Per-position VTMUL loop. For i in 0..AT_SEQ-1:
*   out_row[i] = weight_mat^T @ in_row[i]
*
* Caller sets: AT_BW (weight mat), AT_BI (input base), AT_BO (output
*   base). Must also have AT_DIM, AT_SEQ, AT_RSZ set (done by ATTN
*   prologue).
* Exit: AT_SEQ * AT_DIM output words written at AT_BO.
* Clobbers: X, Y, U, D, W, Q, CC, all MP_*, and AT_BI / AT_BO
*   (advanced during the loop).
*
* INVARIANT: AT_BW/BI/BO/BC alias AT_QI/KJ/SI/IC (see equates.inc).
* AT_BPR is called only during Steps 1-3, strictly before Step 4
* initializes the Q/K/S cursors. Extending AT_BPR's scope past
* Step 3 breaks the aliasing. See equates.inc header comment for
* the lifecycle contract.
*
* MP_MAT/MP_ROW/MP_COL are set once before the loop (they are
* constant across the SEQ iterations). MP_VIN and MP_OUT are
* updated per iteration.
* ============================================================
AT_BPR:
                LDD     AT_BW
                STD     MP_MAT
                LDD     AT_DIM
                STD     MP_ROW          ; rows = DIM
                STD     MP_COL          ; cols = DIM (square projection)
                LDA     AT_SEQ+1
                PSHS    A               ; stack byte counter
AT_BPR_LOOP:
                LDD     AT_BI
                STD     MP_VIN
                LDD     AT_BO
                STD     MP_OUT
                JSR     VTMUL
                LDD     AT_BI
                ADDD    AT_RSZ
                STD     AT_BI
                LDD     AT_BO
                ADDD    AT_RSZ
                STD     AT_BO
                DEC     ,S
                BNE     AT_BPR_LOOP
                LEAS    1,S
                RTS
