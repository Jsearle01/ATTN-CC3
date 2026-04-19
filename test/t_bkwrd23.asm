* ============================================================
* t_bkwrd23.asm — BKWRD Steps 2-3 test harness
* ============================================================
* 4 tests for backward attention:
*   Test 0: Step 2 only, verify dA + dV (SEQ=2 DIM=2)
*   Test 1: Steps 2+3 chained, verify dV + dSc (SEQ=2 DIM=2 SHF=1)
*   Test 2: Steps 2+3 chained, verify dV + dSc (SEQ=4 DIM=8 SHF=3)
*   Test 3: Steps 2+3 chained, clamp trigger (SEQ=2 DIM=4 SHF=0)
*
* Inputs live in the binary image (WORK_N / DY_N labels).
* Outputs land at fixed addresses BW_DATT (dA/dSc) and BW_DV
* inside BWD_WORK — sized for SEQ=8 DIM=16, so all 4 tests fit.
*
* Dependencies: equates.inc, macros.inc, fxmath.inc,
*   fxmath.asm, vecop.asm, matop.asm, actfn.asm, train.asm
* Vectors: tables/bkwrd23_vectors.asm (gen_bkwrd23_vectors.py)
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

* CMP_BUF — compare W words at X vs Y. Returns Z=1 if match.
CMP_BUF:
                CMPW    #0
                BEQ     CB_MATCH
CB_LOOP:
                LDD     ,X++
                CMPD    ,Y++
                BNE     CB_FAIL
                DECW
                BNE     CB_LOOP
CB_MATCH:
                ORCC    #$04            ; set Z (match)
                RTS
CB_FAIL:
                ANDCC   #$FB            ; clear Z (mismatch)
                RTS

* ============================================================
* Test 0: BK23_T0_S2ONLY  (SEQ=2 DIM=2) — Step 2 only
* ============================================================
TEST_0:
                CLR     CUR_TEST_ID
* Parameters
                LDD     SEQ_0
                STD     BK_SEQ
                LDD     DIM_0
                STD     BK_DIM
                LDD     SHF_0
                STD     BK_SHF
                LDD     #1
                STD     BK_VOC          ; dummy (Steps 2-3 don't read VST)
                LDD     #0
                STD     BK_YY           ; dummy (Steps 2-3 don't read YI)
                LDD     #WORK_0
                STD     BK_WRK
                LDD     #DY_0
                STD     BK_DY
* Run INIT + Step 2 only
                JSR     BKWRD_INIT
                JSR     BKWRD_S2
* Verify dA at BW_DATT
                LDX     #BW_DATT
                LDY     #EXP_DA_0
                LDW     DA_CNT_0
                JSR     CMP_BUF
                LBNE    T0_FAIL
* Verify dV at BW_DV
                LDX     #BW_DV
                LDY     #EXP_DV_0
                LDW     DV_CNT_0
                JSR     CMP_BUF
                LBNE    T0_FAIL
                JSR     RECORD_PASS
                RTS
T0_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 1: BK23_T1_CHAIN_TINY  (SEQ=2 DIM=2 SHF=1) — chained
* ============================================================
TEST_1:
                LDA     #1
                STA     CUR_TEST_ID
                LDD     SEQ_1
                STD     BK_SEQ
                LDD     DIM_1
                STD     BK_DIM
                LDD     SHF_1
                STD     BK_SHF
                LDD     #1
                STD     BK_VOC
                LDD     #0
                STD     BK_YY
                LDD     #WORK_1
                STD     BK_WRK
                LDD     #DY_1
                STD     BK_DY
                JSR     BKWRD_INIT
                JSR     BKWRD_S2
                JSR     BKWRD_S3
* Verify dV
                LDX     #BW_DV
                LDY     #EXP_DV_1
                LDW     DV_CNT_1
                JSR     CMP_BUF
                LBNE    T1_FAIL
* Verify dSc (overwrites dA in BW_DATT)
                LDX     #BW_DATT
                LDY     #EXP_DSC_1
                LDW     DA_CNT_1
                JSR     CMP_BUF
                LBNE    T1_FAIL
                JSR     RECORD_PASS
                RTS
T1_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 2: BK23_T2_CHAIN_MED  (SEQ=4 DIM=8 SHF=3) — chained
* ============================================================
TEST_2:
                LDA     #2
                STA     CUR_TEST_ID
                LDD     SEQ_2
                STD     BK_SEQ
                LDD     DIM_2
                STD     BK_DIM
                LDD     SHF_2
                STD     BK_SHF
                LDD     #1
                STD     BK_VOC
                LDD     #0
                STD     BK_YY
                LDD     #WORK_2
                STD     BK_WRK
                LDD     #DY_2
                STD     BK_DY
                JSR     BKWRD_INIT
                JSR     BKWRD_S2
                JSR     BKWRD_S3
                LDX     #BW_DV
                LDY     #EXP_DV_2
                LDW     DV_CNT_2
                JSR     CMP_BUF
                LBNE    T2_FAIL
                LDX     #BW_DATT
                LDY     #EXP_DSC_2
                LDW     DA_CNT_2
                JSR     CMP_BUF
                LBNE    T2_FAIL
                JSR     RECORD_PASS
                RTS
T2_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 3: BK23_T3_CLAMP  (SEQ=2 DIM=4 SHF=0) — clamp trigger
* ============================================================
TEST_3:
                LDA     #3
                STA     CUR_TEST_ID
                LDD     SEQ_3
                STD     BK_SEQ
                LDD     DIM_3
                STD     BK_DIM
                LDD     SHF_3
                STD     BK_SHF
                LDD     #1
                STD     BK_VOC
                LDD     #0
                STD     BK_YY
                LDD     #WORK_3
                STD     BK_WRK
                LDD     #DY_3
                STD     BK_DY
                JSR     BKWRD_INIT
                JSR     BKWRD_S2
                JSR     BKWRD_S3
                LDX     #BW_DV
                LDY     #EXP_DV_3
                LDW     DV_CNT_3
                JSR     CMP_BUF
                LBNE    T3_FAIL
                LDX     #BW_DATT
                LDY     #EXP_DSC_3
                LDW     DA_CNT_3
                JSR     CMP_BUF
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

LBL_PASS:       FCC     "BK23: P="
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
* Include code
* ============================================================
                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm
                INCLUDE src/matop.asm
                INCLUDE src/actfn.asm
                INCLUDE src/train.asm

* ============================================================
* Test vectors
* ============================================================
                INCLUDE tables/bkwrd23_vectors.asm

                END     START
