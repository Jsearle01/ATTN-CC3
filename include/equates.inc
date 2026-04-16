* ============================================================
* equates.inc — ATTN/11 → 6309 Stage 1 constants and memory map
* ============================================================
* Dependencies: none
* Included by: everything
* ============================================================

* -------- Model dimensions (match ATTN/11 exactly) --------
D               EQU     16              ; embedding dim
SEQ             EQU     8               ; sequence length
V               EQU     10              ; vocabulary size
DSEQ            EQU     D*SEQ           ; 128, activation row size
DD              EQU     D*D             ; 256, square weight size
DV              EQU     D*V             ; 160, embed/out proj size

* -------- Learning rates (Q16, = rate * 65536) --------
LR_QKV          EQU     5243            ; 0.08   * 65536
LR_EMB          EQU     655             ; 0.01   * 65536
LR_WOUT         EQU     164             ; 0.0025 * 65536

* -------- Training --------
MAXSTEPS        EQU     400             ; target ~350
LOG_EVERY       EQU     10              ; screen update cadence

* -------- Q-format sizes in bytes --------
Q8SZ            EQU     1
Q15SZ           EQU     2
Q16SZ           EQU     2

* -------- Memory map (Stage 1 base contract — never reorganize) --------
VEC_BASE        EQU     $0000           ; 512B  interrupt vectors
CODE_BASE       EQU     $0200           ; 5.3KB code + tables
SCREEN          EQU     $0400           ; 512B  hw-mapped text, 32x16
STR_BASE        EQU     $1800           ; 200B  string literals
TOK_BASE        EQU     $1900           ; 32B   tokens + target
WEIGHT_BASE     EQU     $1A00           ; 4.8KB Q16 weight accumulators
GRAD_BASE       EQU     $2C00           ; 2.4KB gradient accumulators
FWD_CACHE       EQU     $3800           ; 1.5KB forward activations
BWD_WORK        EQU     $3E00           ; 2.0KB backward workspace
STACK_TOP       EQU     $4800           ; stack grows down from here
FREE_BASE       EQU     $4800           ; ~26KB free

* -------- Weight block layout inside WEIGHT_BASE (Q16) --------
* NOTE: LWASM treats spaces as expression terminators in EQU operands.
*       All arithmetic must be written without spaces around operators.
W_EMB           EQU     WEIGHT_BASE             ; V*D*2  = 320
W_POS           EQU     W_EMB+V*D*Q16SZ        ; SEQ*D*2 = 256
W_Q             EQU     W_POS+SEQ*D*Q16SZ      ; D*D*2  = 512
W_K             EQU     W_Q+DD*Q16SZ           ; 512
W_V             EQU     W_K+DD*Q16SZ           ; 512
W_OUT           EQU     W_V+DD*Q16SZ           ; D*V*2  = 320
W_END           EQU     W_OUT+DV*Q16SZ         ; = 2432 bytes total

* -------- Gradient block mirrors weight block (same offsets, Q15) --------
G_EMB           EQU     GRAD_BASE
G_POS           EQU     G_EMB+V*D*Q15SZ
G_Q             EQU     G_POS+SEQ*D*Q15SZ
G_K             EQU     G_Q+DD*Q15SZ
G_V             EQU     G_K+DD*Q15SZ
G_OUT           EQU     G_V+DD*Q15SZ
G_END           EQU     G_OUT+DV*Q15SZ

* -------- Forward cache offsets (Q8 unless noted) --------
FC_X            EQU     FWD_CACHE               ; SEQ*D, post-embed
FC_Q            EQU     FC_X+DSEQ              ; SEQ*D
FC_K            EQU     FC_Q+DSEQ
FC_V            EQU     FC_K+DSEQ
FC_ATT          EQU     FC_V+DSEQ              ; SEQ*SEQ softmax, Q8
FC_CTX          EQU     FC_ATT+SEQ*SEQ         ; SEQ*D
FC_LOGITS       EQU     FC_CTX+DSEQ            ; SEQ*V, Q15 (pre-softmax)
FC_PROBS        EQU     FC_LOGITS+SEQ*V*Q15SZ  ; SEQ*V, Q8

* -------- Backward workspace (2KB budget, Q15 throughout) --------
BW_DLOGITS      EQU     BWD_WORK                ; SEQ*V*2 = 160
BW_DCTX         EQU     BW_DLOGITS+SEQ*V*Q15SZ
BW_DATT         EQU     BW_DCTX+DSEQ*Q15SZ    ; SEQ*SEQ*2 = 128
BW_DQ           EQU     BW_DATT+SEQ*SEQ*Q15SZ
BW_DK           EQU     BW_DQ+DSEQ*Q15SZ
BW_DV           EQU     BW_DK+DSEQ*Q15SZ
BW_DX           EQU     BW_DV+DSEQ*Q15SZ      ; SEQ*D*2 = 256
BW_SCRATCH      EQU     BW_DX+DSEQ*Q15SZ      ; 158B total scratch (was 116)
BW_END          EQU     BW_SCRATCH+158        ; was +116 (ATTN added 42 B)
*               Total: 1726 B used of 2048 B budget, 322 B BWD_WORK headroom
*               STACK_TOP gap: 834 bytes (was 876)

* -------- Fixed scratch locations inside BW_SCRATCH --------
MULSCR          EQU     BW_SCRATCH              ; 4B STQ target for normalize-by-offset
SOFTMAX_TMP     EQU     BW_SCRATCH+4            ; 32B softmax row buffer
VDOT_ACC        EQU     BW_SCRATCH+40           ; 4B 32-bit dot product accumulator
VDOT_TMP        EQU     BW_SCRATCH+44           ; 2B temp for carry propagation
VSCL_ALPHA      EQU     BW_SCRATCH+46           ; 2B saved scalar for VSCL loop
* -------- MATOP parameter block (caller fills before JSR) --------
MP_MAT          EQU     BW_SCRATCH+48           ; 2B matrix base pointer
MP_VIN          EQU     BW_SCRATCH+50           ; 2B input vector / vx pointer
MP_OUT          EQU     BW_SCRATCH+52           ; 2B output vector / vy pointer
MP_ROW          EQU     BW_SCRATCH+54           ; 2B row count
MP_COL          EQU     BW_SCRATCH+56           ; 2B column count
MP_AHI          EQU     BW_SCRATCH+58           ; 2B accumulator high (internal)
MP_ALO          EQU     BW_SCRATCH+60           ; 2B accumulator low (internal)
MP_SCL          EQU     BW_SCRATCH+62           ; 2B scalar temp (internal)
* -------- SFTMX parameter block (caller fills SF_VEC, SF_LEN, SF_OUT) --------
SF_VEC          EQU     BW_SCRATCH+64           ; 2B softmax input vector pointer
SF_LEN          EQU     BW_SCRATCH+66           ; 2B softmax length
SF_MAX          EQU     BW_SCRATCH+68           ; 2B softmax max value (internal)
SF_SUM          EQU     BW_SCRATCH+70           ; 2B softmax exp sum (16-bit; max count=10)
SF_OUT          EQU     BW_SCRATCH+72           ; 2B softmax output pointer
* -------- VCPY/VCLR parameter block (V_SRC used by VCPY only) --------
V_SRC           EQU     BW_SCRATCH+74           ; 2B source pointer (VCPY)
V_DST           EQU     BW_SCRATCH+76           ; 2B destination pointer
V_LEN           EQU     BW_SCRATCH+78           ; 2B word count (not byte count)

* -------- VSADD parameter block --------
* INVARIANT: V_* and VS_* share physical storage. VSADD must not be
* called while VCPY/VCLR parameters are in preparation. Currently safe
* because VCPY/VCLR are leaf utilities and VSADD is only called from
* BKWRD step 2 (deferred to TRAIN phase) where V_* are dead. If a
* future composite routine ever calls VSADD between setting V_* and
* calling VCPY/VCLR, this overlay silently breaks. See call-graph
* analysis in the Step 2 commit message.
VS_SRC          EQU     V_SRC                   ; alias: source vector ptr
VS_DST          EQU     V_DST                   ; alias: dest vector ptr
VS_LEN          EQU     V_LEN                   ; alias: word count
VS_SCL          EQU     BW_SCRATCH+80           ; 2B Q8 scalar (dedicated)
*               BW_SCRATCH+82 through +83 reserved for future use

* -------- EMBED parameter block (caller-filled) --------
* EMBED parameter block is dedicated (no aliasing). Called at forward
* pass entry before ATTN/PROJ; no other routine's state is live at the
* call site. If a future refactor wants to reclaim these 16 bytes,
* confirm the call graph still holds — particularly that EMBED isn't
* called from inside ATTN or during backward pass.
EM_TOK          EQU     BW_SCRATCH+84           ; 2B token index array ptr
EM_TKE          EQU     BW_SCRATCH+86           ; 2B token embedding table ptr
EM_PSE          EQU     BW_SCRATCH+88           ; 2B position embedding table ptr
EM_OUT          EQU     BW_SCRATCH+90           ; 2B output buffer ptr
EM_SEQ          EQU     BW_SCRATCH+92           ; 2B sequence length
EM_DIM          EQU     BW_SCRATCH+94           ; 2B embedding dimension

* -------- EMBED internal scratch (computed by routine) --------
EM_RBS          EQU     BW_SCRATCH+96           ; 2B row byte size = EM_DIM*2
EM_TPTR         EQU     BW_SCRATCH+98           ; 2B advancing token ptr

* -------- PROJ parameter block (caller-filled) --------
* PROJ parameter block is dedicated. Called at forward-pass exit
* after ATTN returns; no other routine's state is live at the call
* site.
*
* INVARIANT: PROJ writes to MP_MAT/MP_ROW/MP_COL once before the loop
* and to MP_VIN/MP_OUT once per iteration as VTMUL parameters. PROJ
* never reads MP_* values back. If a future routine calls PROJ from
* inside an active MATOP context (where MP_* is mid-use), PROJ will
* overwrite that context's MP_* values silently. Currently safe
* because PROJ is called only at forward-pass exit, after ATTN's
* MATOP usage has completed.
*
* Caller fields ordered to match ATTN/11's PROJ inline params:
* yin, wout, log, seq, dim, voc — for 1:1 source comparability.
PR_Y            EQU     BW_SCRATCH+100          ; 2B input vector base (Y [SEQ][DIM])
PR_WOUT         EQU     BW_SCRATCH+102          ; 2B output projection weights [DIM][VOC]
PR_LOG          EQU     BW_SCRATCH+104          ; 2B output logits buffer [SEQ][VOC]
PR_SEQ          EQU     BW_SCRATCH+106          ; 2B sequence length
PR_DIM          EQU     BW_SCRATCH+108          ; 2B d_model
PR_VOC          EQU     BW_SCRATCH+110          ; 2B vocab size

* -------- PROJ internal scratch (computed at routine entry) --------
* Pre-computed once in PROJ_PROLOGUE to avoid recomputing per iteration.
PR_YST          EQU     BW_SCRATCH+112          ; 2B Y row stride bytes = DIM*2
PR_LST          EQU     BW_SCRATCH+114          ; 2B LOG row stride bytes = VOC*2

* -------- ATTN parameter block (caller-filled) --------
* ATTN parameter block is dedicated. Called once per forward pass,
* after EMBED returns and before PROJ. No live state from other
* routines at the call site.
*
* Caller fields ordered to match ATTN/11's ATTN inline params for
* 1:1 source comparability (LAYER.MAC lines 154-162 / 283-291).
* Field names also match (AT_XIN, AT_YOT, etc.).
*
* AT_WRK points to an 896-byte workspace the caller must allocate
* (Q, K, V regions 256B each + shared S/A region 128B).
* S and A share storage: SFTMX at Step 5 writes the softmax in-place
* over each score row, so SEQ*SEQ*2 = 128B covers both. Matches
* ATTN/11 work budget (LAYER.MAC line 147: work >= 3*seq*dim + seq*seq).
AT_XIN          EQU     BW_SCRATCH+116          ; 2B input X ptr [SEQ][DIM]
AT_WQ           EQU     BW_SCRATCH+118          ; 2B Wq [DIM][DIM]
AT_WK           EQU     BW_SCRATCH+120          ; 2B Wk [DIM][DIM]
AT_WV           EQU     BW_SCRATCH+122          ; 2B Wv [DIM][DIM]
AT_YOT          EQU     BW_SCRATCH+124          ; 2B output Y ptr [SEQ][DIM]
AT_WRK          EQU     BW_SCRATCH+126          ; 2B workspace base (896B)
AT_SEQ          EQU     BW_SCRATCH+128          ; 2B sequence length
AT_DIM          EQU     BW_SCRATCH+130          ; 2B embedding dimension
AT_SHF          EQU     BW_SCRATCH+132          ; 2B sqrt(d) right-shift count

* -------- ATTN internal scratch (computed by routine) --------
* Pre-computed once at entry to avoid recomputing per step.
AT_RSZ          EQU     BW_SCRATCH+134          ; 2B row stride bytes = DIM*2
AT_SSZ          EQU     BW_SCRATCH+136          ; 2B seq element stride = SEQ*2

* -------- Workspace sub-region pointers (computed from AT_WRK) --------
* Q/K/V occupy SEQ*DIM*2 = 256B each for Stage 1 (SEQ=8, DIM=16).
* S/A share SEQ*SEQ*2 = 128B (see header note on in-place softmax).
AT_QQ           EQU     BW_SCRATCH+138          ; 2B Q region base  (= AT_WRK + 0)
AT_KK           EQU     BW_SCRATCH+140          ; 2B K region base  (= AT_WRK + 256)
AT_VV           EQU     BW_SCRATCH+142          ; 2B V region base  (= AT_WRK + 512)
AT_SS           EQU     BW_SCRATCH+144          ; 2B S/A region base (= AT_WRK + 768)

* -------- Step 4+ cursors and inner counter (aliased with BPR helpers) --------
* INVARIANT: AT_QI/KJ/SI/IC share storage with AT_BW/BI/BO/BC.
*   AT_BW/BI/BO/BC are live only during Steps 1-3 (inside the AT_BPR
*   helper). They are dead by the time AT_BPR returns for the last time.
*   AT_QI/KJ/SI/IC are not initialized until Step 4 begins, which is
*   strictly after all three AT_BPR invocations complete.
*
* The aliasing matches the VS_*/V_* pattern (VSADD aliases VCPY/VCLR
* params). If a future refactor extends AT_BPR's scope into Step 4+,
* or moves BPR's invocation to after Step 4's cursor init, this
* aliasing silently breaks. Document the call graph if touching ATTN.
AT_QI           EQU     BW_SCRATCH+146          ; 2B Q row cursor (Step 4)
AT_KJ           EQU     BW_SCRATCH+148          ; 2B K row cursor (Step 4, resets per outer)
AT_SI           EQU     BW_SCRATCH+150          ; 2B S/A row cursor (Steps 4-6)
AT_IC           EQU     BW_SCRATCH+152          ; 2B inner (SEQ) counter (Step 4)

AT_BW           EQU     AT_QI                   ; alias: BPR weight ptr (Steps 1-3)
AT_BI           EQU     AT_KJ                   ; alias: BPR input ptr  (Steps 1-3)
AT_BO           EQU     AT_SI                   ; alias: BPR output ptr (Steps 1-3)
AT_BC           EQU     AT_IC                   ; alias: BPR counter    (Steps 1-3)

* -------- Dedicated cursors (no aliasing) --------
* AT_YI is Step-6-only (output row ptr during V^T . S[i] loop).
* AT_OC is outer SEQ counter reused across Steps 4/5/6. Its liveness
* overlaps AT_SI's, so AT_OC cannot share storage with any aliased slot.
AT_YI           EQU     BW_SCRATCH+154          ; 2B Y output row cursor (Step 6)
AT_OC           EQU     BW_SCRATCH+156          ; 2B outer SEQ counter (Steps 4-6)

* -------- Lookup tables (labels defined in tables.asm via INCLUDE) --------
*               EXTERN  EXPTBL                  ; 256 B, Q8 exp(-i/32)
*               EXTERN  LOGTBL                  ; 514 B, Q12 -ln(x/256)*4096

* -------- Hardware --------
GIME_INIT0      EQU     $FF90
GIME_VMODE      EQU     $FF98
GIME_VRES       EQU     $FF99
PIA1_DDRA       EQU     $FF00
