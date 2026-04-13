* ============================================================
* t_fxdiv.asm — FXDIV test harness
* ============================================================
* Tests Q8 division against reference vectors.
* Dependencies: equates.inc, macros.inc, fxmath.asm
* Inputs: test/vectors_fxdiv.bin (from gen_fxdiv_vectors.py)
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
DIV_B_ADDR      EQU     $1706           ; 2B: address of divisor for FXDIV call
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

* Verify magic "FD"
                LDX     #VECS
                LDA     ,X
                CMPA    #$46            ; 'F'
                LBNE    MAGIC_FAIL
                LDA     1,X
                CMPA    #$44            ; 'D'
                LBNE    MAGIC_FAIL

* Read count, skip 8-byte header
                LDD     4,X
                STD     LOOP_CNT
                LEAX    8,X

* -------- FXDIV test loop --------
* Record: a[2], b[2], expected[2]
FXDIV_TEST:
                LDD     LOOP_CNT
                LBEQ    FXDIV_DONE

* Load a into D, point Y at b in the record
                LDD     ,X              ; D = dividend (a)
                LEAY    2,X             ; Y -> divisor (b) in record
                PSHS    X               ; save record pointer
                JSR     FXDIV           ; D = quotient
                PULS    X               ; restore record pointer
                CMPD    4,X             ; compare with expected at offset 4
                BEQ     FXDIV_PASS
                LDD     FAIL_CNT
                ADDD    #1
                STD     FAIL_CNT
                BRA     FXDIV_NEXT
FXDIV_PASS:
                LDD     PASS_CNT
                ADDD    #1
                STD     PASS_CNT
FXDIV_NEXT:
                LEAX    6,X             ; advance to next record
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    FXDIV_TEST
FXDIV_DONE:
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
* DISPLAY_RESULTS
* ============================================================
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

LBL_PASS:       FCC     "DIV: P="
                FCB     0
LBL_MID:        FCC     " F="
                FCB     0
LBL_DONE:       FCC     "DONE"
                FCB     0
BADHDR_MSG:     FCC     "BAD FXDIV HEADER"
                FCB     0

* ============================================================
* FXMATH code
* ============================================================
                INCLUDE src/fxmath.asm

* ============================================================
* Test vectors — loaded separately by Lua at $4800
* ============================================================
VECS            EQU     $4800

                END     START
