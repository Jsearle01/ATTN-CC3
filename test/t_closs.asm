* ============================================================
* t_closs.asm — CLOSS cross-entropy loss test harness
* ============================================================
* 4 tests: uniform, perfect, wrong, mixed.
* All tests SEQ=8 VOC=10 (matching CLOSS hardcoded >>3 average).
* Dependencies: equates.inc, macros.inc, fxmath.inc,
*   fxmath.asm, vecop.asm, actfn.asm (SFTMX+EXPTBL),
*   train.asm (CLOSS+LOGTBL)
* Vectors: tables/closs_vectors.asm (gen_closs_vectors.py)
* ============================================================

                INCLUDE include/equates.inc
                INCLUDE include/macros.inc
                INCLUDE include/fxmath.inc

                ORG     $0600

* -------- Entry point --------
START:
                ORCC    #$50
                LDMD    #1
                LDS     #STACK_TOP
                JSR     CLEAR_SCREEN
                JSR     RUN_TESTS
                JSR     DISPLAY_RESULTS
HALT:           BRA     HALT

* -------- Scratch RAM at $2000 --------
SCRATCH_BASE    EQU     $2000
PASS_CNT        EQU     SCRATCH_BASE+0
FAIL_CNT        EQU     SCRATCH_BASE+2
RESULT_BITS     EQU     SCRATCH_BASE+4
LAST_FAIL_ID    EQU     SCRATCH_BASE+5
CUR_TEST_ID     EQU     SCRATCH_BASE+6
DEC_BUF         EQU     SCRATCH_BASE+$30
DEC_DTEMP       EQU     SCRATCH_BASE+$36
DEC_SCRPTR      EQU     SCRATCH_BASE+$38

* ============================================================
* RUN_TESTS
* ============================================================
RUN_TESTS:
                CLRA
                CLRB
                STD     PASS_CNT
                STD     FAIL_CNT
                CLR     RESULT_BITS
                LDA     #$FF
                STA     LAST_FAIL_ID

                JSR     TEST_0
                JSR     TEST_1
                JSR     TEST_2
                JSR     TEST_3
                RTS

* ============================================================
* Framework helpers
* ============================================================
RECORD_PASS:
                LDD     PASS_CNT
                ADDD    #1
                STD     PASS_CNT
                LDA     CUR_TEST_ID
                LDB     #1
RP_SHIFT:
                TSTA
                BEQ     RP_DONE
                LSLB
                DECA
                BRA     RP_SHIFT
RP_DONE:
                ORB     RESULT_BITS
                STB     RESULT_BITS
                RTS

RECORD_FAIL:
                LDD     FAIL_CNT
                ADDD    #1
                STD     FAIL_CNT
                LDA     CUR_TEST_ID
                STA     LAST_FAIL_ID
                RTS

* ============================================================
* SETUP_CLOSS — load CL_* params from descriptor, call CLOSS
* Entry: X -> descriptor (SEQ, VOC, LOGITS data, TARGETS data)
* Exit: D = Q12 loss. X -> EXPECT word.
* ============================================================
SETUP_CLOSS:
                LDD     ,X              ; SEQ
                STD     CL_SEQ
                LDD     2,X             ; VOC
                STD     CL_VOC
                LEAX    4,X             ; X -> LOGITS data

* LOGITS are in the binary image. Point CL_LOG directly.
                STX     CL_LOG

* Advance X past LOGITS (SEQ * VOC * 2 bytes)
                LDD     CL_SEQ
                MULD    CL_VOC          ; Q = SEQ*VOC, W = low 16
                TFR     W,D
                LSLD                    ; D = SEQ*VOC*2
                LEAX    D,X             ; X -> TARGETS data

* TARGETS also in binary image.
                STX     CL_TGT

* Advance X past TARGETS (SEQ * 2 bytes)
                LDD     CL_SEQ
                LSLD                    ; D = SEQ*2
                LEAX    D,X             ; X -> EXPECT word
                PSHS    X               ; save EXPECT ptr across CLOSS
                JSR     CLOSS           ; D = Q12 loss (clobbers X)
                PULS    X               ; restore EXPECT ptr
                RTS

* ============================================================
* Test 0: CLOSS_UNIFORM
* ============================================================
TEST_0:
                CLR     CUR_TEST_ID
                LDX     #SEQ_0
                JSR     SETUP_CLOSS
* D = computed loss, X -> EXPECT_0
                CMPD    ,X
                LBNE    T0_FAIL
                JSR     RECORD_PASS
                RTS
T0_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 1: CLOSS_PERFECT
* ============================================================
TEST_1:
                LDA     #1
                STA     CUR_TEST_ID
                LDX     #SEQ_1
                JSR     SETUP_CLOSS
                CMPD    ,X
                LBNE    T1_FAIL
                JSR     RECORD_PASS
                RTS
T1_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 2: CLOSS_WRONG
* ============================================================
TEST_2:
                LDA     #2
                STA     CUR_TEST_ID
                LDX     #SEQ_2
                JSR     SETUP_CLOSS
                CMPD    ,X
                LBNE    T2_FAIL
                JSR     RECORD_PASS
                RTS
T2_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 3: CLOSS_MIXED
* ============================================================
TEST_3:
                LDA     #3
                STA     CUR_TEST_ID
                LDX     #SEQ_3
                JSR     SETUP_CLOSS
                CMPD    ,X
                LBNE    T3_FAIL
                JSR     RECORD_PASS
                RTS
T3_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Display / screen helpers
* ============================================================
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
                LDU     #LBL_BITS
                JSR     COPY_STRING
                CLRA
                LDB     RESULT_BITS
                JSR     WRITE_HEX_4
                LDU     #LBL_LF
                JSR     COPY_STRING
                CLRA
                LDB     LAST_FAIL_ID
                JSR     WRITE_HEX_4
                LDX     #SCREEN+64
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

WRITE_HEX_4:
                PSHS    D
                LDA     ,S
                JSR     WHEX_BYTE
                LDA     1,S
                JSR     WHEX_BYTE
                LEAS    2,S
                RTS

WHEX_BYTE:
                PSHS    A
                LSRA
                LSRA
                LSRA
                LSRA
                JSR     WHEX_NIB
                PULS    A
                ANDA    #$0F
                JSR     WHEX_NIB
                RTS

WHEX_NIB:
                CMPA    #10
                BLT     WHEX_D
                ADDA    #'A-10
                BRA     WHEX_OUT
WHEX_D:
                ADDA    #'0
WHEX_OUT:
                STA     ,X+
                RTS

DIV_TABLE:      FDB     1000
                FDB     100
                FDB     10

LBL_PASS:       FCC     "CLS: P="
                FCB     0
LBL_MID:        FCC     " F="
                FCB     0
LBL_BITS:       FCC     "B="
                FCB     0
LBL_LF:         FCC     " LF="
                FCB     0
LBL_DONE:       FCC     "DONE"
                FCB     0

* ============================================================
* Include code (CLOSS needs SFTMX which needs VMAX and FXDIV)
* ============================================================
                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm
                INCLUDE src/actfn.asm
                INCLUDE src/train.asm

* ============================================================
* Test vectors
* ============================================================
                INCLUDE tables/closs_vectors.asm

                END     START
