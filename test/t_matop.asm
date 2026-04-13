* ============================================================
* t_matop.asm — MATOP test harness
* ============================================================
* Tests MVMUL, MVADD, VTMUL, OUTER against reference vectors.
* Dependencies: equates.inc, macros.inc, fxmath.asm, vecop.asm,
*               matop.asm
* Inputs: test/vectors_matop.bin (from gen_matop_vectors.py)
* ============================================================

                INCLUDE include/equates.inc
                INCLUDE include/macros.inc
                INCLUDE include/fxmath.inc

                ORG     $0600

* -------- Entry point --------
START:
                ORCC    #$50            ; mask IRQ and FIRQ
                LDS     #STACK_TOP
                JSR     CLEAR_SCREEN
                JSR     RUN_TESTS
                JSR     DISPLAY_RESULTS
HALT:           BRA     HALT

* -------- Scratch RAM --------
PASS_MVM        EQU     $1700           ; 2 bytes
FAIL_MVM        EQU     $1702
PASS_MVA        EQU     $1704
FAIL_MVA        EQU     $1706
PASS_VTM        EQU     $1708
FAIL_VTM        EQU     $170A
PASS_OUT        EQU     $170C
FAIL_OUT        EQU     $170E
LOOP_CNT        EQU     $1710           ; 2 bytes
CUR_ROWS        EQU     $1712           ; 1 byte
CUR_COLS        EQU     $1713           ; 1 byte
REC_PTR         EQU     $1714           ; 2 bytes
CMP_TMP         EQU     $1716           ; 2 bytes
DEC_BUF         EQU     $1730           ; 5 bytes
DEC_DTEMP       EQU     $1736           ; 2 bytes
DEC_SCRPTR      EQU     $1738           ; 2 bytes
* Result buffers for MVMUL/MVADD/VTMUL output
* Max dimension: 3x3 -> 3 output elements = 6 bytes
RESULT_BUF      EQU     $1740           ; 32 bytes (up to 16 elements)
* Copy of vout_init for MVADD (routine modifies in place)
VOUT_COPY       EQU     $1760           ; 32 bytes
* Copy of mat_init for OUTER (routine modifies in place)
MAT_COPY        EQU     $1780           ; 64 bytes

* ============================================================
* RUN_TESTS
* ============================================================
RUN_TESTS:
                CLRA
                CLRB
                STD     PASS_MVM
                STD     FAIL_MVM
                STD     PASS_MVA
                STD     FAIL_MVA
                STD     PASS_VTM
                STD     FAIL_VTM
                STD     PASS_OUT
                STD     FAIL_OUT

* Verify magic "MV"
                LDX     #VECS
                LDA     ,X
                CMPA    #$4D            ; 'M'
                LBNE    MAGIC_FAIL
                LDA     1,X
                CMPA    #$56            ; 'V'
                LBNE    MAGIC_FAIL

* Header: MV 01 00 n_mvmul[2] n_mvadd[2] n_vtmul[2] n_outer[2] reserved[2]
                LEAX    14,X            ; skip 14-byte header

* ======== STQ byte-order sanity check ========
* (tested implicitly by MVMUL — if STQ byte order is wrong,
* all 32-bit accumulation produces wrong results and MVMUL fails)

* ======== MVMUL tests ========
                LDY     #VECS
                LDD     4,Y             ; n_mvmul
                STD     LOOP_CNT

MVMUL_TEST:
                LDD     LOOP_CNT
                LBEQ    MVMUL_TDONE
* Record: rows[1] cols[1] mat[r*c*2] vin[c*2] exp[r*2]
                LDA     ,X              ; rows
                STA     CUR_ROWS
                LDB     1,X             ; cols
                STB     CUR_COLS
                LEAX    2,X             ; X -> mat
                STX     REC_PTR         ; save for pointer arithmetic
* Set up MP_ parameter block
                STX     MP_MAT          ; mat pointer
* vin = mat + rows*cols*2
                LDA     CUR_ROWS
                LDB     CUR_COLS
                MUL                     ; D = rows*cols (8x8->16 unsigned)
                LSLD                    ; D = rows*cols*2
                LEAY    D,X             ; Y -> vin
                STY     MP_VIN
* expected = vin + cols*2
                LDB     CUR_COLS
                CLRA
                LSLD                    ; D = cols*2
                STD     CMP_TMP         ; save for later
                LEAY    D,Y             ; Y -> expected (save for compare)
                PSHS    Y               ; push expected ptr
* Set up output
                LDY     #RESULT_BUF
                STY     MP_OUT
                LDB     CUR_ROWS
                CLRA
                STD     MP_ROW
                LDB     CUR_COLS
                STD     MP_COL
                JSR     MVMUL
* Compare RESULT_BUF with expected
                PULS    Y               ; Y -> expected
                LDX     #RESULT_BUF
                LDB     CUR_ROWS
                PSHS    B               ; counter on stack
MVMUL_CMP:
                LDD     ,X++
                CMPD    ,Y++
                BNE     MVMUL_CFAIL
                DEC     ,S
                BNE     MVMUL_CMP
                LEAS    1,S
                LDD     PASS_MVM
                ADDD    #1
                STD     PASS_MVM
                BRA     MVMUL_CNEXT
MVMUL_CFAIL:
                LEAS    1,S
                LDD     FAIL_MVM
                ADDD    #1
                STD     FAIL_MVM
MVMUL_CNEXT:
* Advance X past expected: Y is already past if all matched
* Use REC_PTR + total record data size
* total = 2 + rows*cols*2 + cols*2 + rows*2
* = 2 + (rows*cols + cols + rows)*2
                TFR     Y,X             ; X = past expected (Y advanced)
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    MVMUL_TEST
MVMUL_TDONE:

* ======== MVADD tests ========
                LDY     #VECS
                LDD     6,Y             ; n_mvadd
                STD     LOOP_CNT

MVADD_TEST:
                LDD     LOOP_CNT
                LBEQ    MVADD_TDONE
* Record: rows[1] cols[1] mat[r*c*2] vin[c*2] vout_init[r*2] exp[r*2]
                LDA     ,X
                STA     CUR_ROWS
                LDB     1,X
                STB     CUR_COLS
                LEAX    2,X
                STX     MP_MAT
* vin = mat + rows*cols*2
                LDA     CUR_ROWS
                LDB     CUR_COLS
                MUL
                LSLD
                LEAY    D,X             ; Y -> vin
                STY     MP_VIN
* vout_init = vin + cols*2
                LDB     CUR_COLS
                CLRA
                LSLD
                LEAY    D,Y             ; Y -> vout_init
* Copy vout_init to VOUT_COPY (MVADD modifies in place)
                LDU     #VOUT_COPY
                LDB     CUR_ROWS
                PSHS    B
MVA_COPY:
                LDD     ,Y++
                STD     ,U++
                DEC     ,S
                BNE     MVA_COPY
                LEAS    1,S
* Y now points past vout_init = expected
                PSHS    Y               ; save expected ptr
* Set up MP_ block
                LDU     #VOUT_COPY
                STU     MP_OUT
                LDB     CUR_ROWS
                CLRA
                STD     MP_ROW
                LDB     CUR_COLS
                STD     MP_COL
                JSR     MVADD
* Compare VOUT_COPY with expected
                PULS    Y               ; Y -> expected
                LDX     #VOUT_COPY
                LDB     CUR_ROWS
                PSHS    B
MVADD_CMP:
                LDD     ,X++
                CMPD    ,Y++
                BNE     MVADD_CFAIL
                DEC     ,S
                BNE     MVADD_CMP
                LEAS    1,S
                LDD     PASS_MVA
                ADDD    #1
                STD     PASS_MVA
                BRA     MVADD_CNEXT
MVADD_CFAIL:
                LEAS    1,S
                LDD     FAIL_MVA
                ADDD    #1
                STD     FAIL_MVA
MVADD_CNEXT:
                TFR     Y,X
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    MVADD_TEST
MVADD_TDONE:

* ======== VTMUL tests ========
                LDY     #VECS
                LDD     8,Y             ; n_vtmul
                STD     LOOP_CNT

VTMUL_TEST:
                LDD     LOOP_CNT
                LBEQ    VTMUL_TDONE
* Record: rows[1] cols[1] mat[r*c*2] vin[r*2] exp[c*2]
                LDA     ,X
                STA     CUR_ROWS
                LDB     1,X
                STB     CUR_COLS
                LEAX    2,X
                STX     MP_MAT
* vin = mat + rows*cols*2
                LDA     CUR_ROWS
                LDB     CUR_COLS
                MUL
                LSLD
                LEAY    D,X             ; Y -> vin
                STY     MP_VIN
* expected = vin + rows*2
                LDB     CUR_ROWS
                CLRA
                LSLD
                LEAY    D,Y             ; Y -> expected
                PSHS    Y
* Set up
                LDY     #RESULT_BUF
                STY     MP_OUT
                LDB     CUR_ROWS
                CLRA
                STD     MP_ROW
                LDB     CUR_COLS
                STD     MP_COL
                JSR     VTMUL
* Compare
                PULS    Y
                LDX     #RESULT_BUF
                LDB     CUR_COLS        ; output has cols elements
                PSHS    B
VTMUL_CMP:
                LDD     ,X++
                CMPD    ,Y++
                BNE     VTMUL_CFAIL
                DEC     ,S
                BNE     VTMUL_CMP
                LEAS    1,S
                LDD     PASS_VTM
                ADDD    #1
                STD     PASS_VTM
                BRA     VTMUL_CNEXT
VTMUL_CFAIL:
                LEAS    1,S
                LDD     FAIL_VTM
                ADDD    #1
                STD     FAIL_VTM
VTMUL_CNEXT:
                TFR     Y,X
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    VTMUL_TEST
VTMUL_TDONE:

* ======== OUTER tests ========
                LDY     #VECS
                LDD     10,Y            ; n_outer
                STD     LOOP_CNT

OUTER_TEST:
                LDD     LOOP_CNT
                LBEQ    OUTER_TDONE
* Record: rows[1] cols[1] mat_init[r*c*2] vx[r*2] vy[c*2] exp[r*c*2]
                LDA     ,X
                STA     CUR_ROWS
                LDB     1,X
                STB     CUR_COLS
                LEAX    2,X
* Copy mat_init to MAT_COPY
                LDA     CUR_ROWS
                LDB     CUR_COLS
                MUL                     ; D = rows*cols
                PSHS    D               ; save element count
                LSLD                    ; D = bytes
                STD     CMP_TMP         ; save mat byte count
                LDB     1,S             ; element count low byte
                LDY     #MAT_COPY
                PSHS    B               ; copy counter
OUT_MCOPY:
                LDD     ,X++
                STD     ,Y++
                DEC     ,S
                BNE     OUT_MCOPY
                LEAS    1,S
                PULS    D               ; clean saved element count
* X now past mat_init = vx
                STX     CMP_TMP         ; save vx ptr temporarily
                LDY     #MAT_COPY
                STY     MP_MAT
                LDX     CMP_TMP         ; X -> vx
                STX     MP_VIN
* vy = vx + rows*2
                LDB     CUR_ROWS
                CLRA
                LSLD
                LEAY    D,X             ; Y -> vy
                STY     MP_OUT
* expected = vy + cols*2
                LDB     CUR_COLS
                CLRA
                LSLD
                LEAY    D,Y             ; Y -> expected
                PSHS    Y               ; save expected ptr
* Set up dimensions
                LDB     CUR_ROWS
                CLRA
                STD     MP_ROW
                LDB     CUR_COLS
                STD     MP_COL
                JSR     OUTER
* Compare MAT_COPY with expected
                PULS    Y               ; Y -> expected
                LDX     #MAT_COPY
                LDA     CUR_ROWS
                LDB     CUR_COLS
                MUL                     ; D = total elements
                PSHS    B               ; counter
OUTER_CMP:
                LDD     ,X++
                CMPD    ,Y++
                BNE     OUTER_CFAIL
                DEC     ,S
                BNE     OUTER_CMP
                LEAS    1,S
                LDD     PASS_OUT
                ADDD    #1
                STD     PASS_OUT
                BRA     OUTER_CNEXT
OUTER_CFAIL:
                LEAS    1,S
                LDD     FAIL_OUT
                ADDD    #1
                STD     FAIL_OUT
OUTER_CNEXT:
                TFR     Y,X
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    OUTER_TEST
OUTER_TDONE:
                RTS

* -------- Magic check failure --------
MAGIC_FAIL:
                LDX     #SCREEN
                LDU     #BADHDR_MSG
MF_LOOP:
                LDA     ,U+
                BEQ     MF_DONE
                STA     ,X+
                BRA     MF_LOOP
MF_DONE:
                LBRA    HALT

* ============================================================
* CLEAR_SCREEN
* ============================================================
CLEAR_SCREEN:
                LDA     #$20
                STA     SCREEN
                LDX     #SCREEN
                LDU     #SCREEN+1
                LDW     #511
                TFM     X+,U+
                RTS

* ============================================================
* DISPLAY_RESULTS — 5 rows
* ============================================================
DISPLAY_RESULTS:
                LDX     #SCREEN
                LDU     #LBL_MVM
                JSR     COPY_STRING
                LDD     PASS_MVM
                JSR     WRITE_DECIMAL_4
                LDU     #LBL_MID
                JSR     COPY_STRING
                LDD     FAIL_MVM
                JSR     WRITE_DECIMAL_4

                LDX     #SCREEN+32
                LDU     #LBL_MVA
                JSR     COPY_STRING
                LDD     PASS_MVA
                JSR     WRITE_DECIMAL_4
                LDU     #LBL_MID
                JSR     COPY_STRING
                LDD     FAIL_MVA
                JSR     WRITE_DECIMAL_4

                LDX     #SCREEN+64
                LDU     #LBL_VTM
                JSR     COPY_STRING
                LDD     PASS_VTM
                JSR     WRITE_DECIMAL_4
                LDU     #LBL_MID
                JSR     COPY_STRING
                LDD     FAIL_VTM
                JSR     WRITE_DECIMAL_4

                LDX     #SCREEN+96
                LDU     #LBL_OUT
                JSR     COPY_STRING
                LDD     PASS_OUT
                JSR     WRITE_DECIMAL_4
                LDU     #LBL_MID
                JSR     COPY_STRING
                LDD     FAIL_OUT
                JSR     WRITE_DECIMAL_4

                LDX     #SCREEN+128
                LDU     #LBL_DONE
                JSR     COPY_STRING
                RTS

* ============================================================
* String / decimal helpers (same as t_vecop)
* ============================================================
COPY_STRING:
                LDA     ,U+
                BEQ     CS_DONE
                STA     ,X+
                BRA     COPY_STRING
CS_DONE:        RTS

WRITE_DECIMAL_4:
                STX     DEC_SCRPTR
                LDU     #DEC_BUF
                LDX     #DIV_TABLE
                JSR     WD4_DIGIT
                JSR     WD4_DIGIT
                JSR     WD4_DIGIT
                ADDB    #$30
                STB     ,U
                LDX     DEC_SCRPTR
                LDD     DEC_BUF
                STD     ,X++
                LDD     DEC_BUF+2
                STD     ,X++
                RTS

WD4_DIGIT:
                LDY     ,X++
                STY     DEC_DTEMP
                LDY     #$002F
WD4_LOOP:
                LEAY    1,Y
                SUBD    DEC_DTEMP
                BCC     WD4_LOOP
                ADDD    DEC_DTEMP
                PSHS    D
                TFR     Y,D
                STB     ,U+
                PULS    D
                RTS

DIV_TABLE:      FDB     1000
                FDB     100
                FDB     10

* ============================================================
* String constants
* ============================================================
LBL_MVM:        FCC     "MVM: P="
                FCB     0
LBL_MVA:        FCC     "MVA: P="
                FCB     0
LBL_VTM:        FCC     "VTM: P="
                FCB     0
LBL_OUT:        FCC     "OUT: P="
                FCB     0
LBL_MID:        FCC     " F="
                FCB     0
LBL_DONE:       FCC     "DONE"
                FCB     0
BADHDR_MSG:     FCC     "BAD MATOP HEADER"
                FCB     0

* ============================================================
* Include code modules
* ============================================================
                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm
                INCLUDE src/matop.asm

* ============================================================
* Test vectors — loaded separately by Lua at $4800
* ============================================================
VECS            EQU     $4800

                END     START
