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
* LOGTBL — 257-entry Q12 cross-entropy lookup table
* ============================================================
                INCLUDE tables/logtbl.asm
