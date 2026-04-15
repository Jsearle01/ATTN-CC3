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
*   - src/matop.asm  (PROJ uses VTMUL)
*   - src/actfn.asm  (future ATTN will use SFTMX)
*
* Routines:
*   - EMBED  (moved from vecop.asm, was committed in 489aa61)
*   - PROJ   (new, this commit)
*   - ATTN   (future)
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
