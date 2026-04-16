* ============================================================
* t_attn.asm — ATTN self-attention test harness
* ============================================================
* 5 tests for full ATTN forward pass (Steps 1-7).
* Per-test reporting via result bitmap at SCRATCH_BASE+4.
* Dependencies: equates.inc, macros.inc, fxmath.inc, fxmath.asm,
*   vecop.asm (VDOT, VMAX), matop.asm (VTMUL), actfn.asm (SFTMX,
*   EXPTBL), layer.asm (ATTN, AT_BPR)
* Vectors: tables/attn_vectors.asm (gen_attn_vectors.py)
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

* -------- Test data buffers --------
* X_BUF:   SEQ*DIM*2 max = 8*16*2 = 256 B   ($2060-$215F)
* WQ_BUF:  DIM*DIM*2 max = 16*16*2 = 512 B  ($2160-$235F)
* WK_BUF:  same                               ($2360-$255F)
* WV_BUF:  same                               ($2560-$275F)
* WRK_BUF: 3*SEQ*DIM*2 + SEQ*SEQ*2 = 896 B  ($2760-$2ADF)
* OUT_BUF: SEQ*DIM*2 = 256 B                 ($2AE0-$2BDF)
* ---- SETUP_ATTN internal scratch (saved descriptor counts) ----
SA_XCNT         EQU     SCRATCH_BASE+$50
SA_WCNT         EQU     SCRATCH_BASE+$52
SA_YCNT         EQU     SCRATCH_BASE+$54

X_BUF           EQU     SCRATCH_BASE+$60
WQ_BUF          EQU     SCRATCH_BASE+$160
WK_BUF          EQU     SCRATCH_BASE+$360
WV_BUF          EQU     SCRATCH_BASE+$560
WRK_BUF         EQU     SCRATCH_BASE+$760
OUT_BUF         EQU     SCRATCH_BASE+$AE0

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
                JSR     TEST_4
                RTS

* ============================================================
* Framework helpers
* ============================================================
LOAD_BUF:
                CMPW    #0
                BEQ     LB_EXIT
LB_LOOP:
                LDD     ,X++
                STD     ,Y++
                DECW
                BNE     LB_LOOP
LB_EXIT:
                RTS

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
* SETUP_ATTN — common ATTN parameter setup subroutine
* ============================================================
* Entry: X -> test descriptor block (SEQ, DIM, SHF, X_CNT, W_CNT,
*         Y_CNT, then data: X, WQ, WK, WV, EXPECT).
* Loads all buffers, sets AT_* parameters, calls ATTN.
* Exit: OUT_BUF contains Y result. X -> EXPECT data. W = Y_CNT.
* ============================================================
SETUP_ATTN:
* Read descriptor header: SEQ, DIM, SHF, X_CNT, W_CNT, Y_CNT (12 bytes)
                LDD     ,X              ; SEQ
                STD     AT_SEQ
                LDD     2,X             ; DIM
                STD     AT_DIM
                LDD     4,X             ; SHF
                STD     AT_SHF
                LDD     6,X             ; X_CNT
                STD     SA_XCNT
                LDD     8,X             ; W_CNT
                STD     SA_WCNT
                LDD     10,X            ; Y_CNT
                STD     SA_YCNT
                LEAX    12,X            ; X -> start of X data

* Load X_BUF
                LDY     #X_BUF
                LDW     SA_XCNT
                JSR     LOAD_BUF

* Load WQ_BUF
                LDY     #WQ_BUF
                LDW     SA_WCNT
                JSR     LOAD_BUF

* Load WK_BUF
                LDY     #WK_BUF
                LDW     SA_WCNT
                JSR     LOAD_BUF

* Load WV_BUF
                LDY     #WV_BUF
                LDW     SA_WCNT
                JSR     LOAD_BUF

* X now points to EXPECT data. Save for verify step.
                PSHS    X               ; save EXPECT ptr

* Set ATTN caller parameters
                LDD     #X_BUF
                STD     AT_XIN
                LDD     #WQ_BUF
                STD     AT_WQ
                LDD     #WK_BUF
                STD     AT_WK
                LDD     #WV_BUF
                STD     AT_WV
                LDD     #OUT_BUF
                STD     AT_YOT
                LDD     #WRK_BUF
                STD     AT_WRK

* Call ATTN
                JSR     ATTN

* Restore EXPECT ptr and Y_CNT for caller's verify loop
                PULS    X               ; X -> EXPECT
                LDW     SA_YCNT
                RTS

* ============================================================
* Test 0: ATTN_TINY  (SEQ=2 DIM=4 SHF=1) — hand-verifiable
* ============================================================
TEST_0:
                CLR     CUR_TEST_ID
                LDX     #SEQ_0
                JSR     SETUP_ATTN
* X -> EXPECT_0, W = Y_CNT_0 = 8
                LDY     #OUT_BUF
T0_CHK:
                LDD     ,Y++
                CMPD    ,X++
                LBNE    T0_FAIL
                DECW
                BNE     T0_CHK
                JSR     RECORD_PASS
                RTS
T0_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 1: ATTN_NOSHIFT  (SEQ=2 DIM=4 SHF=0) — BEQ skip path
* ============================================================
TEST_1:
                LDA     #1
                STA     CUR_TEST_ID
                LDX     #SEQ_1
                JSR     SETUP_ATTN
                LDY     #OUT_BUF
T1_CHK:
                LDD     ,Y++
                CMPD    ,X++
                LBNE    T1_FAIL
                DECW
                BNE     T1_CHK
                JSR     RECORD_PASS
                RTS
T1_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 2: ATTN_RESIDUAL_WRAP  (SEQ=2 DIM=4 SHF=1) — wrap safety net
* ============================================================
TEST_2:
                LDA     #2
                STA     CUR_TEST_ID
                LDX     #SEQ_2
                JSR     SETUP_ATTN
                LDY     #OUT_BUF
T2_CHK:
                LDD     ,Y++
                CMPD    ,X++
                LBNE    T2_FAIL
                DECW
                BNE     T2_CHK
                JSR     RECORD_PASS
                RTS
T2_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 3: ATTN_SAT_SCORE  (SEQ=2 DIM=4 SHF=0) — VDOT clamp safety net
* ============================================================
TEST_3:
                LDA     #3
                STA     CUR_TEST_ID
                LDX     #SEQ_3
                JSR     SETUP_ATTN
                LDY     #OUT_BUF
T3_CHK:
                LDD     ,Y++
                CMPD    ,X++
                LBNE    T3_FAIL
                DECW
                BNE     T3_CHK
                JSR     RECORD_PASS
                RTS
T3_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 4: ATTN_MEDIUM  (SEQ=4 DIM=8 SHF=2) — non-trivial dimensions
* ============================================================
TEST_4:
                LDA     #4
                STA     CUR_TEST_ID
                LDX     #SEQ_4
                JSR     SETUP_ATTN
                LDY     #OUT_BUF
T4_CHK:
                LDD     ,Y++
                CMPD    ,X++
                LBNE    T4_FAIL
                DECW
                BNE     T4_CHK
                JSR     RECORD_PASS
                RTS
T4_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Display / screen / decimal / hex helpers
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

LBL_PASS:       FCC     "ATN: P="
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
* Include code (provides ATTN and all dependencies)
* ============================================================
                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm
                INCLUDE src/matop.asm
                INCLUDE src/actfn.asm
                INCLUDE src/layer.asm

* ============================================================
* Test vectors (generated by gen_attn_vectors.py)
* ============================================================
                INCLUDE tables/attn_vectors.asm

                END     START
