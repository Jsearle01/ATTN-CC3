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
BW_SCRATCH      EQU     BW_DX+DSEQ*Q15SZ      ; 100B total scratch (was 84)
BW_END          EQU     BW_SCRATCH+100        ; was +84
*               Total: 1668 B used of 2048 B budget, 380 B headroom
*               STACK_TOP gap: 892 bytes (was 908)

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

* -------- Lookup tables (labels defined in tables.asm via INCLUDE) --------
*               EXTERN  EXPTBL                  ; 256 B, Q8 exp(-i/32)
*               EXTERN  LOGTBL                  ; 514 B, Q12 -ln(x/256)*4096

* -------- Hardware --------
GIME_INIT0      EQU     $FF90
GIME_VMODE      EQU     $FF98
GIME_VRES       EQU     $FF99
PIA1_DDRA       EQU     $FF00
