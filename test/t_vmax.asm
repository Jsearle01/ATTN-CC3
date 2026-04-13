* ============================================================
* t_vmax.asm — VMAX test harness
* ============================================================
* Tests find-max-and-index against reference vectors.
* Dependencies: equates.inc, macros.inc, fxmath.asm, vecop.asm
* Inputs: test/vectors_vmax.bin (from gen_vmax_vectors.py)
* ============================================================

                INCLUDE include/equates.inc
                INCLUDE include/macros.inc
                INCLUDE include/fxmath.inc

                ORG     $0600

* -------- Entry point --------
START:
                ORCC    #$50
                LDS     #STACK_TOP
                JSR     CLEAR_SCREEN
                JSR     RUN_TESTS
                JSR     DISPLAY_RESULTS
HALT:           BRA     HALT

* -------- Scratch RAM --------
PASS_CNT        EQU     $1700
FAIL_CNT        EQU     $1702
LOOP_CNT        EQU     $1704
DEC_BUF         EQU     $1730
DEC_DTEMP       EQU     $1736
DEC_SCRPTR      EQU     $1738

* ============================================================
* RUN_TESTS
* ============================================================
RUN_TESTS:
                CLRA
                CLRB
                STD     PASS_CNT
                STD     FAIL_CNT

* Verify magic "VM"
                LDX     #VECS
                LDA     ,X
                CMPA    #$56            ; 'V'
                LBNE    MAGIC_FAIL
                LDA     1,X
                CMPA    #$4D            ; 'M'
                LBNE    MAGIC_FAIL

                LDD     4,X
                STD     LOOP_CNT
                LEAX    8,X             ; skip header

* -------- VMAX test loop --------
* Record: count[1], pad[1], vec[count*2], exp_max[2], exp_idx[2]
VMAX_TEST:
                LDD     LOOP_CNT
                LBEQ    VMAX_TDONE

                LDB     ,X              ; B = count
                LEAX    2,X             ; skip count+pad, X -> vec[]
                PSHS    X               ; save vec[] pointer
                JSR     VMAX            ; D = max value, X = max index
* Save results
                STD     VDOT_ACC+2      ; save max value
                TFR     X,D
                STD     VDOT_ACC        ; save max index (in D, low byte)

                PULS    X               ; restore vec[] pointer
* Advance past vec[count*2] to expected
                LDB     -2,X            ; re-read count from record
* Wait — X was pushed before JSR, points to vec[0]. But VMAX
* advanced X past the data. We restored X to vec[0]. Need to
* advance by count*2.
                CLRA
                LSLD                    ; D = count*2
                LEAX    D,X             ; X -> exp_max

* Compare max value
                LDD     VDOT_ACC+2      ; our max value
                CMPD    ,X              ; expected max value
                BNE     VMAX_FAIL
* Compare index
                LDD     VDOT_ACC        ; our max index
                CMPD    2,X             ; expected index (unsigned 16-bit)
                BNE     VMAX_FAIL

                LDD     PASS_CNT
                ADDD    #1
                STD     PASS_CNT
                BRA     VMAX_NEXT
VMAX_FAIL:
                LDD     FAIL_CNT
                ADDD    #1
                STD     FAIL_CNT
VMAX_NEXT:
                LEAX    4,X             ; skip exp_max[2] + exp_idx[2]
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    VMAX_TEST
VMAX_TDONE:
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

CLEAR_SCREEN:
                LDA     #$20
                STA     SCREEN
                LDX     #SCREEN
                LDU     #SCREEN+1
                LDW     #511
                TFM     X+,U+
                RTS

DISPLAY_RESULTS:
                LDX     #SCREEN
                LDU     #LBL_PASS
                JSR     COPY_STRING
                LDD     PASS_CNT
                JSR     WRITE_DECIMAL_4
                LDU     #LBL_MID
                JSR     COPY_STRING
                LDD     FAIL_CNT
                JSR     WRITE_DECIMAL_4
                LDX     #SCREEN+32
                LDU     #LBL_DONE
                JSR     COPY_STRING
                RTS

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

LBL_PASS:       FCC     "MAX: P="
                FCB     0
LBL_MID:        FCC     " F="
                FCB     0
LBL_DONE:       FCC     "DONE"
                FCB     0
BADHDR_MSG:     FCC     "BAD VMAX HEADER"
                FCB     0

                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm

VECS            EQU     $4800

                END     START
