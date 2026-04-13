* ============================================================
* t_exptbl.asm — EXPTBL load and index test
* ============================================================
* Verifies 5 known EXPTBL entries by index lookup.
* Dependencies: equates.inc, macros.inc, tables/exptbl.asm
* ============================================================

                INCLUDE include/equates.inc
                INCLUDE include/macros.inc
                INCLUDE include/fxmath.inc

                ORG     $0600

START:
                ORCC    #$50
                LDS     #STACK_TOP
                JSR     CLEAR_SCREEN
                JSR     RUN_TESTS
                JSR     DISPLAY_RESULTS
HALT:           BRA     HALT

PASS_CNT        EQU     $1700
FAIL_CNT        EQU     $1702
DEC_BUF         EQU     $1730
DEC_DTEMP       EQU     $1736
DEC_SCRPTR      EQU     $1738

* ============================================================
* RUN_TESTS — check 5 known EXPTBL entries
* ============================================================
RUN_TESTS:
                CLRA
                CLRB
                STD     PASS_CNT
                STD     FAIL_CNT

* Test 0: EXPTBL[0] = 256 (exp(0) = 1.0)
                LDX     #EXPTBL
                LDD     ,X              ; EXPTBL[0]
                CMPD    #256
                BNE     ET_FAIL
                JSR     INC_PASS
                BRA     ET_1
ET_FAIL:
                JSR     INC_FAIL

* Test 1: EXPTBL[16] = 155 (exp(-0.5))
ET_1:
                LDD     #16
                LSLD                    ; D = 32 (word offset)
                LDX     #EXPTBL
                LDD     D,X             ; indexed by D
                CMPD    #155
                BNE     ET_1F
                JSR     INC_PASS
                BRA     ET_2
ET_1F:          JSR     INC_FAIL

* Test 2: EXPTBL[32] = 94 (exp(-1.0))
ET_2:
                LDD     #32
                LSLD
                LDX     #EXPTBL
                LDD     D,X
                CMPD    #94
                BNE     ET_2F
                JSR     INC_PASS
                BRA     ET_3
ET_2F:          JSR     INC_FAIL

* Test 3: EXPTBL[64] = 35 (exp(-2.0))
ET_3:
                LDD     #64
                LSLD
                LDX     #EXPTBL
                LDD     D,X
                CMPD    #35
                BNE     ET_3F
                JSR     INC_PASS
                BRA     ET_4
ET_3F:          JSR     INC_FAIL

* Test 4: EXPTBL[255] = 0 (exp(-7.97))
ET_4:
                LDD     #255
                LSLD
                LDX     #EXPTBL
                LDD     D,X
                CMPD    #0
                BNE     ET_4F
                JSR     INC_PASS
                BRA     ET_DONE
ET_4F:          JSR     INC_FAIL
ET_DONE:
                RTS

INC_PASS:
                LDD     PASS_CNT
                ADDD    #1
                STD     PASS_CNT
                RTS

INC_FAIL:
                LDD     FAIL_CNT
                ADDD    #1
                STD     FAIL_CNT
                RTS

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

LBL_PASS:       FCC     "EXP: P="
                FCB     0
LBL_MID:        FCC     " F="
                FCB     0
LBL_DONE:       FCC     "DONE"
                FCB     0

* ============================================================
* EXPTBL data (assembled inline)
* ============================================================
                INCLUDE tables/exptbl.asm

                END     START
