* ============================================================
* t_sftmx.asm — SFTMX test harness
* ============================================================
* Tests softmax against reference vectors.
* Dependencies: equates.inc, macros.inc, fxmath.asm, vecop.asm,
*               actfn.asm (includes EXPTBL)
* Inputs: test/vectors_sftmx.bin (from gen_sftmx_vectors.py)
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
LOOP_CNT        EQU     $1704
CUR_LEN         EQU     $1706           ; 1 byte
REC_PTR         EQU     $1708           ; 2 bytes: saved vec[] start
DEC_BUF         EQU     $1730
DEC_DTEMP       EQU     $1736
DEC_SCRPTR      EQU     $1738
SM_BUF          EQU     $1740           ; 32 bytes output buffer

* ============================================================
* RUN_TESTS
* ============================================================
RUN_TESTS:
                CLRA
                CLRB
                STD     PASS_CNT
                STD     FAIL_CNT

                LDX     #VECS
                LDA     ,X
                CMPA    #$53            ; 'S'
                LBNE    MAGIC_FAIL
                LDA     1,X
                CMPA    #$4D            ; 'M'
                LBNE    MAGIC_FAIL

                LDD     4,X
                STD     LOOP_CNT
                LEAX    8,X

* -------- SFTMX test loop --------
* Record: count[1], pad[1], vec[count*2], expected[count*2]
TSM_TEST:
                LDD     LOOP_CNT
                LBEQ    TSM_DONE

                LDB     ,X              ; count
                STB     CUR_LEN
                LEAX    2,X             ; X -> vec[]
                STX     REC_PTR         ; save vec[] start

* Set up SFTMX call
                STX     SF_VEC
                LDB     CUR_LEN
                CLRA
                STD     SF_LEN
                LDD     #SM_BUF
                STD     SF_OUT
                JSR     SFTMX

* Compare SM_BUF against expected (exact match)
                LDX     REC_PTR         ; X -> vec[] start
                LDB     CUR_LEN
                CLRA
                LSLD                    ; D = count*2
                LEAX    D,X             ; X -> expected[]
                LDY     #SM_BUF         ; Y -> our output
                LDB     CUR_LEN
                PSHS    B               ; compare counter
TSM_CMP:
                LDD     ,Y++
                CMPD    ,X++
                BNE     TSM_CFAIL
                DEC     ,S
                BNE     TSM_CMP
                LEAS    1,S
                LDD     PASS_CNT
                ADDD    #1
                STD     PASS_CNT
                BRA     TSM_NEXT
TSM_CFAIL:
                LEAS    1,S
                LDD     FAIL_CNT
                ADDD    #1
                STD     FAIL_CNT
TSM_NEXT:
* Advance X to next record: vec_start + count*4 (vec + expected)
                LDX     REC_PTR
                LDB     CUR_LEN
                CLRA
                LSLD
                LSLD                    ; D = count*4
                LEAX    D,X             ; X -> next record
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    TSM_TEST
TSM_DONE:
                RTS

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

LBL_PASS:       FCC     "SMX: P="
                FCB     0
LBL_MID:        FCC     " F="
                FCB     0
LBL_DONE:       FCC     "DONE"
                FCB     0
BADHDR_MSG:     FCC     "BAD SFTMX HEADER"
                FCB     0

                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm
                INCLUDE src/actfn.asm

VECS            EQU     $4800

                END     START
