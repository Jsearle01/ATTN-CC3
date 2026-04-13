* ============================================================
* matop.asm — Matrix-vector operations
* ============================================================
* Dependencies: equates.inc (MP_*, MULSCR), macros.inc (NORMQ15)
* Exports: MVMUL, MVADD, VTMUL, OUTER
* ============================================================
* NOTE: This file is INCLUDE'd into the parent (main.asm or
* test harness). Do not INCLUDE equates.inc or macros.inc here.
* ============================================================
* All values Q7.8 fixed-point. Matrices row-major.
*
* DEVIATION from ATTN/11: MVADD, VTMUL, OUTER use single-clamp
* 32-bit accumulate instead of two-stage clamp. See DEVIATIONS.md.
* ============================================================

* ============================================================
* Parameter block — caller fills before JSR.
* Located in BW_SCRATCH region (see equates.inc).
* ============================================================
* MP_MAT  — matrix base pointer        (2 bytes)
* MP_VIN  — input vector / vx pointer   (2 bytes)
* MP_OUT  — output vector / vy pointer  (2 bytes)
* MP_ROW  — row count                   (2 bytes)
* MP_COL  — column count                (2 bytes)
* MP_AHI  — 32-bit accumulator high     (2 bytes, internal)
* MP_ALO  — 32-bit accumulator low      (2 bytes, internal)
* MP_SCL  — scalar temp                 (2 bytes, internal)

* ============================================================
* MVMUL — Matrix-vector multiply: vout = mat * vin
* ============================================================
* vout[i] = sum_j( mat[i][j] * vin[j] )
* 32-bit Q16 accumulation per row, then NORMQ15 + clamp.
* Identical to ATTN/11 (single clamp, no deviation).
*
* Caller sets: MP_MAT, MP_VIN, MP_OUT, MP_ROW, MP_COL
* Exit: vout[] populated
* Clobbers: X, Y, U, D, W, Q, CC
* ============================================================
MVMUL:
                LDY     MP_MAT          ; Y = matrix ptr (advances)
                LDU     MP_OUT          ; U = output ptr (advances)
                LDB     MP_ROW+1        ; B = row count
                PSHS    B               ; outer counter on stack

MVMUL_ROW:
* Clear 32-bit accumulator
                CLRA
                CLRB
                STD     MP_AHI
                STD     MP_ALO
* Reset vin pointer for each row
                LDX     MP_VIN          ; X = vin ptr
                LDB     MP_COL+1        ; B = col count
                PSHS    B               ; inner counter on stack

MVMUL_COL:
                LDD     ,Y++            ; D = mat[i][j], advance mat
                MULD    ,X++            ; Q = mat[i][j] * vin[j], advance vin
*                                       ; D = high, W = low
                STQ     MULSCR          ; save product
                LDD     MP_ALO          ; acc low
                ADDD    MULSCR+2        ; += product low, carry set
                STD     MP_ALO
                LDD     MP_AHI          ; acc high (carry preserved)
                ADCB    MULSCR+1        ; += product high lo + carry
                ADCA    MULSCR          ; += product high hi + carry
                STD     MP_AHI
                DEC     ,S              ; decrement col counter
                BNE     MVMUL_COL
                LEAS    1,S             ; drop col counter

* Extract Q8 result: NORMQ15 + clamp
                LDA     MP_AHI          ; MSB of 32-bit acc
                BEQ     MVMUL_OK
                CMPA    #$FF
                BEQ     MVMUL_OK
                TSTA
                BPL     MVMUL_CLPOS
                LDD     #$8000
                BRA     MVMUL_STORE
MVMUL_CLPOS:
                LDD     #$7FFF
                BRA     MVMUL_STORE
MVMUL_OK:
                LDD     MP_AHI+1        ; bits [23:8] = Q8 result
MVMUL_STORE:
                STD     ,U++            ; vout[i] = result
                DEC     ,S              ; decrement row counter
                LBNE    MVMUL_ROW
                LEAS    1,S             ; drop row counter
                RTS

* ============================================================
* MVADD — Matrix-vector multiply-add: vout += mat * vin
* ============================================================
* Single-clamp: accumulate dot product in 32 bits, sign-extend
* existing vout[i] into the accumulator, NORMQ15 + clamp once.
*
* DEVIATION: ATTN/11 clamps the dot product to Q8 first, then
* ADDs to vout with a second overflow clamp. Our single-clamp
* preserves the full 32-bit sum before normalizing.
*
* Caller sets: MP_MAT, MP_VIN, MP_OUT, MP_ROW, MP_COL
* Exit: vout[] updated (added)
* Clobbers: X, Y, U, D, W, Q, CC
* ============================================================
MVADD:
                LDY     MP_MAT
                LDU     MP_OUT
                LDB     MP_ROW+1
                PSHS    B

MVADD_ROW:
* Clear 32-bit accumulator
                CLRA
                CLRB
                STD     MP_AHI
                STD     MP_ALO
                LDX     MP_VIN
                LDB     MP_COL+1
                PSHS    B

MVADD_COL:
                LDD     ,Y++
                MULD    ,X++
                STQ     MULSCR
                LDD     MP_ALO
                ADDD    MULSCR+2
                STD     MP_ALO
                LDD     MP_AHI
                ADCB    MULSCR+1
                ADCA    MULSCR
                STD     MP_AHI
                DEC     ,S
                BNE     MVADD_COL
                LEAS    1,S

* Single-clamp: sign-extend vout[i] to Q16 (32-bit), add to accumulator.
* vout[i] is Q8. As Q16: vout * 256. In 4 bytes: [sign, hi, lo, $00].
* Write to MULSCR, then 32-bit add to MP_AHI:MP_ALO.
                LDD     ,U              ; D = vout[i]
                TSTA
                BPL     MVADD_VP
                LDA     #$FF
                STA     MULSCR          ; sign byte = $FF (negative)
                BRA     MVADD_VS
MVADD_VP:
                CLR     MULSCR          ; sign byte = $00 (positive)
MVADD_VS:
* TODO post-validation: eliminate this reload by storing
* hi/lo bytes before the sign test (A still holds hi byte
* from the original LDD).
                LDD     ,U              ; reload vout[i]
                STA     MULSCR+1        ; vout hi byte
                STB     MULSCR+2        ; vout lo byte
                CLR     MULSCR+3        ; low byte = 0
* MULSCR[0..3] = sign-extended vout[i] << 8 as 32-bit Q16
* 32-bit add MULSCR to MP_AHI:MP_ALO
                LDD     MP_ALO
                ADDD    MULSCR+2        ; add low words
                STD     MP_ALO
                LDD     MP_AHI
                ADCB    MULSCR+1        ; add high words + carry
                ADCA    MULSCR
                STD     MP_AHI

* Now clamp (same as MVMUL)
                LDA     MP_AHI
                BEQ     MVADD_OK
                CMPA    #$FF
                BEQ     MVADD_OK
                TSTA
                BPL     MVADD_CLPOS
                LDD     #$8000
                BRA     MVADD_STORE
MVADD_CLPOS:
                LDD     #$7FFF
                BRA     MVADD_STORE
MVADD_OK:
                LDD     MP_AHI+1
MVADD_STORE:
                STD     ,U++            ; vout[i] = result
                DEC     ,S              ; row counter
                LBNE    MVADD_ROW
                LEAS    1,S
                RTS

* ============================================================
* VTMUL — Transpose-vector multiply: vout = mat^T * vin
* ============================================================
* vout[j] = sum_i( mat[i][j] * vin[i] )
* Clears vout, then accumulates: for each row i, adds
* vin[i] * row_i to vout element-wise.
*
* Single-clamp per element: sign-extend vout[j] to 32 bits,
* add product, NORMQ15+clamp, store.
*
* Caller sets: MP_MAT, MP_VIN, MP_OUT, MP_ROW, MP_COL
* Exit: vout[] populated
* Clobbers: X, Y, U, D, W, Q, CC
* ============================================================
VTMUL:
* Clear output vector
                LDX     MP_OUT
                LDB     MP_COL+1
                PSHS    B
VTMUL_CLR:
                CLRA
                CLRB
                STD     ,X++
                DEC     ,S
                BNE     VTMUL_CLR
                LEAS    1,S

* Process rows
                LDY     MP_MAT          ; Y = matrix ptr (advances through all)
                LDU     MP_VIN          ; U = vin ptr (advances per row)
                LDB     MP_ROW+1
                PSHS    B               ; outer (row) counter

VTMUL_ROW:
                LDD     ,U++            ; D = vin[i], advance vin
                STD     MP_SCL          ; scalar = vin[i]
                LDX     MP_OUT          ; X = vout ptr (reset each row)
                LDB     MP_COL+1
                PSHS    B               ; inner (col) counter

VTMUL_COL:
* product = mat[i][j] * scalar (32-bit Q16)
                LDD     ,Y++            ; D = mat[i][j], advance mat
                MULD    MP_SCL          ; Q = mat[i][j] * vin[i]
*                                       ; D = high, W = low
* Single-clamp: sign-extend vout[j], add product, clamp
* Build 32-bit Q16 of vout[j] in MULSCR: [sign, hi, lo, 0]
                STQ     MP_AHI          ; save product to MP_AHI:MP_ALO temp
                LDD     ,X              ; D = vout[j]
                TSTA
                BPL     VTMUL_VP
                LDA     #$FF
                STA     MULSCR
                BRA     VTMUL_VS
VTMUL_VP:
                CLR     MULSCR
VTMUL_VS:
* TODO post-validation: eliminate this reload by storing
* hi/lo bytes before the sign test (A still holds hi byte
* from the original LDD).
                LDD     ,X              ; reload vout[j]
                STA     MULSCR+1
                STB     MULSCR+2
                CLR     MULSCR+3
* 32-bit add: acc (product in MP_AHI:MP_ALO) += vout (in MULSCR)
                LDD     MP_ALO
                ADDD    MULSCR+2
                STD     MP_ALO
                LDD     MP_AHI
                ADCB    MULSCR+1
                ADCA    MULSCR
                STD     MP_AHI
* Clamp
                LDA     MP_AHI
                BEQ     VTMUL_OK
                CMPA    #$FF
                BEQ     VTMUL_OK
                TSTA
                BPL     VTMUL_CLPOS
                LDD     #$8000
                BRA     VTMUL_ST
VTMUL_CLPOS:
                LDD     #$7FFF
                BRA     VTMUL_ST
VTMUL_OK:
                LDD     MP_AHI+1
VTMUL_ST:
                STD     ,X++            ; vout[j] = result, advance
                DEC     ,S              ; col counter
                LBNE    VTMUL_COL
                LEAS    1,S
                DEC     ,S              ; row counter
                LBNE    VTMUL_ROW
                LEAS    1,S
                RTS

* ============================================================
* OUTER — Outer product accumulate: mat += vx (x) vy
* ============================================================
* mat[i][j] += vx[i] * vy[j]
* Single-clamp per element.
*
* Caller sets: MP_MAT, MP_VIN (=vx), MP_OUT (=vy), MP_ROW, MP_COL
* Exit: mat[] updated
* Clobbers: X, Y, U, D, W, Q, CC
* ============================================================
OUTER:
                LDY     MP_MAT          ; Y = matrix ptr (advances)
                LDU     MP_VIN          ; U = vx ptr (advances per row)
                LDB     MP_ROW+1
                PSHS    B               ; outer (row) counter

OUTER_ROW:
                LDD     ,U++            ; D = vx[i], advance vx
                STD     MP_SCL          ; scalar = vx[i]
                LDX     MP_OUT          ; X = vy ptr (reset each row)
                LDB     MP_COL+1
                PSHS    B               ; inner (col) counter

OUTER_COL:
* product = vy[j] * scalar (32-bit Q16)
                LDD     ,X++            ; D = vy[j], advance vy
                MULD    MP_SCL          ; Q = vy[j] * vx[i]
*                                       ; D = high, W = low
* Single-clamp: sign-extend mat[i][j], add product, clamp
                STQ     MP_AHI          ; save product
                LDD     ,Y              ; D = mat[i][j]
                TSTA
                BPL     OUTER_VP
                LDA     #$FF
                STA     MULSCR
                BRA     OUTER_VS
OUTER_VP:
                CLR     MULSCR
OUTER_VS:
* TODO post-validation: eliminate this reload by storing
* hi/lo bytes before the sign test (A still holds hi byte
* from the original LDD).
                LDD     ,Y              ; reload mat[i][j]
                STA     MULSCR+1
                STB     MULSCR+2
                CLR     MULSCR+3
* 32-bit add
                LDD     MP_ALO
                ADDD    MULSCR+2
                STD     MP_ALO
                LDD     MP_AHI
                ADCB    MULSCR+1
                ADCA    MULSCR
                STD     MP_AHI
* Clamp
                LDA     MP_AHI
                BEQ     OUTER_OK
                CMPA    #$FF
                BEQ     OUTER_OK
                TSTA
                BPL     OUTER_CLPOS
                LDD     #$8000
                BRA     OUTER_ST
OUTER_CLPOS:
                LDD     #$7FFF
                BRA     OUTER_ST
OUTER_OK:
                LDD     MP_AHI+1
OUTER_ST:
                STD     ,Y++            ; mat[i][j] = result, advance
                DEC     ,S              ; col counter
                LBNE    OUTER_COL
                LEAS    1,S
                DEC     ,S              ; row counter
                LBNE    OUTER_ROW
                LEAS    1,S
                RTS
