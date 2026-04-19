* ============================================================
* t_bkwrd456.asm — BKWRD Steps 4-6 test harness
* ============================================================
* 4 tests covering the backward pass tail:
*   Test 0: S4 only, verify dQ + dK (SEQ=2 DIM=2)
*   Test 1: S4+S5+S6 chained, verify 8 buffers (SEQ=2 DIM=2 VOC=10)
*   Test 2: S4+S5+S6 chained (SEQ=4 DIM=8 VOC=10, 400 words verified)
*   Test 3: S4+S5+S6 chained, Step 6 d_tok clamp trigger
*
* Dependencies: equates.inc, macros.inc, fxmath.inc,
*   fxmath.asm, vecop.asm, matop.asm, actfn.asm, train.asm
* Vectors: tables/bkwrd456_vectors.asm (gen_bkwrd456_vectors.py)
*
* Setup per chained test:
*   1. Set BK_SEQ/DIM/VOC/SHF/WRK/DY/XX/WQ/WK/WV/TOKS/DWQ/DWK/DWV/DTKE/DPSE
*   2. Copy dSc_N -> BW_DATT (word count = SEQ*SEQ)
*   3. Copy dV_N  -> BW_DV  (word count = SEQ*DIM)   [chained only]
*   4. Zero DWQ_BUF, DWK_BUF, DWV_BUF, DTOK_BUF, DPOS_BUF
*   5. JSR BKWRD_INIT + BKWRD_S4 (+ S5 + S6 if chained)
*   6. Verify buffers via VERIFY_BUF — short-circuit to FAIL on first miss
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

* -------- Scratch RAM at $2400 (past the 7308-byte binary end ~$228C) --------
* Deviation from project convention ($2000): this harness's vector
* table is large enough that EXP_DTOK_2 etc. spill past $2000 and
* would be overwritten by PASS_CNT/FAIL_CNT writes. Moving scratch
* past binary end avoids the overlap. Lua trampoline updated to match.
SCRATCH_BASE    EQU     $2400
PASS_CNT        EQU     SCRATCH_BASE+0
FAIL_CNT        EQU     SCRATCH_BASE+2
RESULT_BITS     EQU     SCRATCH_BASE+4
LAST_FAIL_ID    EQU     SCRATCH_BASE+5
CUR_TEST_ID     EQU     SCRATCH_BASE+6
DEC_BUF         EQU     SCRATCH_BASE+$30
DEC_DTEMP       EQU     SCRATCH_BASE+$36
DEC_SCRPTR      EQU     SCRATCH_BASE+$38

* -------- Output buffers (sized for max test: SEQ=4 DIM=8 VOC=10) --------
* dWq/dWk/dWv = DIM*DIM*2 = 128 B each
* d_tok       = VOC*DIM*2 = 160 B
* d_pos       = SEQ*DIM*2 =  64 B
*
* Buffers placed at SCRATCH_BASE+$400 ($2400) — past the 7268-byte
* binary end at ~$2264 (bkwrd456 vector table is the largest in the
* project). End at $2660, well below FWD_CACHE ($3800).
DWQ_BUF         EQU     SCRATCH_BASE+$400       ; 128 B
DWK_BUF         EQU     SCRATCH_BASE+$480       ; 128 B
DWV_BUF         EQU     SCRATCH_BASE+$500       ; 128 B
DTOK_BUF        EQU     SCRATCH_BASE+$580       ; 160 B
DPOS_BUF        EQU     SCRATCH_BASE+$620       ; 64 B

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
* COPY_BUF — copy W words from X to Y
* Exit: X, Y advanced past their buffers
* ============================================================
COPY_BUF:
                CMPW    #0
                BEQ     CPB_DONE
CPB_LOOP:
                LDD     ,X++
                STD     ,Y++
                DECW
                BNE     CPB_LOOP
CPB_DONE:
                RTS

* ============================================================
* ZERO_BUF — zero W words at X
* ============================================================
ZERO_BUF:
                CMPW    #0
                BEQ     ZB_DONE
                CLRA
                CLRB
ZB_LOOP:
                STD     ,X++
                DECW
                BNE     ZB_LOOP
ZB_DONE:
                RTS

* ============================================================
* VERIFY_BUF — compare W words at X vs Y. Z=1 if match, Z=0 on miss.
* Caller should LBNE <fail_label> after JSR.
* ============================================================
VERIFY_BUF:
                CMPW    #0
                BEQ     VB_MATCH
VB_LOOP:
                LDD     ,X++
                CMPD    ,Y++
                BNE     VB_FAIL
                DECW
                BNE     VB_LOOP
VB_MATCH:
                ORCC    #$04            ; set Z (match)
                RTS
VB_FAIL:
                ANDCC   #$FB            ; clear Z (mismatch)
                RTS

* ============================================================
* Test 0: BK456_T0_S4ONLY  (SEQ=2 DIM=2) — Step 4 only
* ============================================================
TEST_0:
                CLR     CUR_TEST_ID
* Parameters (BK_VOC/SHF dummy, BK_YY dummy)
                LDD     SEQ_0
                STD     BK_SEQ
                LDD     DIM_0
                STD     BK_DIM
                LDD     VOC_0
                STD     BK_VOC
                CLRA
                CLRB
                STD     BK_SHF
                STD     BK_YY
                LDD     #WORK_0
                STD     BK_WRK
* Copy dSc_0 -> BW_DATT (SEQ*SEQ words)
                LDX     #DSC_0
                LDY     #BW_DATT
                LDD     BK_SEQ
                MULD    BK_SEQ
                TFR     W,D
                TFR     D,W             ; W = SEQ*SEQ word count
                JSR     COPY_BUF
* Run INIT + S4
                JSR     BKWRD_INIT
                JSR     BKWRD_S4
* Verify dQ (at BW_DQ)
                LDX     #BW_DQ
                LDY     #EXP_DQ_0
                LDW     DQ_CNT_0
                JSR     VERIFY_BUF
                LBNE    T0_FAIL
* Verify dK (at BW_DK)
                LDX     #BW_DK
                LDY     #EXP_DK_0
                LDW     DK_CNT_0
                JSR     VERIFY_BUF
                LBNE    T0_FAIL
                JSR     RECORD_PASS
                RTS
T0_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 1: BK456_T1_TINY  (SEQ=2 DIM=2 VOC=10) — chained
* ============================================================
TEST_1:
                LDA     #1
                STA     CUR_TEST_ID
                LDD     SEQ_1
                STD     BK_SEQ
                LDD     DIM_1
                STD     BK_DIM
                LDD     VOC_1
                STD     BK_VOC
                CLRA
                CLRB
                STD     BK_SHF
                STD     BK_YY
                LDD     #WORK_1
                STD     BK_WRK
                LDD     #DY_1
                STD     BK_DY
                LDD     #X_1
                STD     BK_XX
                LDD     #WQ_1
                STD     BK_WQ
                LDD     #WK_1
                STD     BK_WK
                LDD     #WV_1
                STD     BK_WV
                LDD     #TOKS_1
                STD     BK_TOKS
                LDD     #DWQ_BUF
                STD     BK_DWQ
                LDD     #DWK_BUF
                STD     BK_DWK
                LDD     #DWV_BUF
                STD     BK_DWV
                LDD     #DTOK_BUF
                STD     BK_DTKE
                LDD     #DPOS_BUF
                STD     BK_DPSE
* Copy dSc_1 -> BW_DATT (SEQ*SEQ)
                LDX     #DSC_1
                LDY     #BW_DATT
                LDD     BK_SEQ
                MULD    BK_SEQ
                TFR     W,D
                TFR     D,W
                JSR     COPY_BUF
* Copy dV_1 -> BW_DV (SEQ*DIM)
                LDX     #DV_1
                LDY     #BW_DV
                LDD     BK_SEQ
                MULD    BK_DIM
                TFR     W,D
                TFR     D,W
                JSR     COPY_BUF
* Zero output buffers
                LDX     #DWQ_BUF
                LDW     DWQ_CNT_1
                JSR     ZERO_BUF
                LDX     #DWK_BUF
                LDW     DWK_CNT_1
                JSR     ZERO_BUF
                LDX     #DWV_BUF
                LDW     DWV_CNT_1
                JSR     ZERO_BUF
                LDX     #DTOK_BUF
                LDW     DTOK_CNT_1
                JSR     ZERO_BUF
                LDX     #DPOS_BUF
                LDW     DPOS_CNT_1
                JSR     ZERO_BUF
* Run chained
                JSR     BKWRD_INIT
                JSR     BKWRD_S4
                JSR     BKWRD_S5
                JSR     BKWRD_S6
* Verify 8 output buffers (short-circuit to FAIL on first miss)
                LDX     #BW_DQ
                LDY     #EXP_DQ_1
                LDW     DQ_CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #BW_DK
                LDY     #EXP_DK_1
                LDW     DK_CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #BW_DX
                LDY     #EXP_DX_1
                LDW     DX_CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #DWQ_BUF
                LDY     #EXP_DWQ_1
                LDW     DWQ_CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #DWK_BUF
                LDY     #EXP_DWK_1
                LDW     DWK_CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #DWV_BUF
                LDY     #EXP_DWV_1
                LDW     DWV_CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #DTOK_BUF
                LDY     #EXP_DTOK_1
                LDW     DTOK_CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #DPOS_BUF
                LDY     #EXP_DPOS_1
                LDW     DPOS_CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                JSR     RECORD_PASS
                RTS
T1_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 2: BK456_T2_MEDIUM  (SEQ=4 DIM=8 VOC=10) — chained
* ============================================================
TEST_2:
                LDA     #2
                STA     CUR_TEST_ID
                LDD     SEQ_2
                STD     BK_SEQ
                LDD     DIM_2
                STD     BK_DIM
                LDD     VOC_2
                STD     BK_VOC
                CLRA
                CLRB
                STD     BK_SHF
                STD     BK_YY
                LDD     #WORK_2
                STD     BK_WRK
                LDD     #DY_2
                STD     BK_DY
                LDD     #X_2
                STD     BK_XX
                LDD     #WQ_2
                STD     BK_WQ
                LDD     #WK_2
                STD     BK_WK
                LDD     #WV_2
                STD     BK_WV
                LDD     #TOKS_2
                STD     BK_TOKS
                LDD     #DWQ_BUF
                STD     BK_DWQ
                LDD     #DWK_BUF
                STD     BK_DWK
                LDD     #DWV_BUF
                STD     BK_DWV
                LDD     #DTOK_BUF
                STD     BK_DTKE
                LDD     #DPOS_BUF
                STD     BK_DPSE
                LDX     #DSC_2
                LDY     #BW_DATT
                LDD     BK_SEQ
                MULD    BK_SEQ
                TFR     W,D
                TFR     D,W
                JSR     COPY_BUF
                LDX     #DV_2
                LDY     #BW_DV
                LDD     BK_SEQ
                MULD    BK_DIM
                TFR     W,D
                TFR     D,W
                JSR     COPY_BUF
                LDX     #DWQ_BUF
                LDW     DWQ_CNT_2
                JSR     ZERO_BUF
                LDX     #DWK_BUF
                LDW     DWK_CNT_2
                JSR     ZERO_BUF
                LDX     #DWV_BUF
                LDW     DWV_CNT_2
                JSR     ZERO_BUF
                LDX     #DTOK_BUF
                LDW     DTOK_CNT_2
                JSR     ZERO_BUF
                LDX     #DPOS_BUF
                LDW     DPOS_CNT_2
                JSR     ZERO_BUF
                JSR     BKWRD_INIT
                JSR     BKWRD_S4
                JSR     BKWRD_S5
                JSR     BKWRD_S6
                LDX     #BW_DQ
                LDY     #EXP_DQ_2
                LDW     DQ_CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                LDX     #BW_DK
                LDY     #EXP_DK_2
                LDW     DK_CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                LDX     #BW_DX
                LDY     #EXP_DX_2
                LDW     DX_CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                LDX     #DWQ_BUF
                LDY     #EXP_DWQ_2
                LDW     DWQ_CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                LDX     #DWK_BUF
                LDY     #EXP_DWK_2
                LDW     DWK_CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                LDX     #DWV_BUF
                LDY     #EXP_DWV_2
                LDW     DWV_CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                LDX     #DTOK_BUF
                LDY     #EXP_DTOK_2
                LDW     DTOK_CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                LDX     #DPOS_BUF
                LDY     #EXP_DPOS_2
                LDW     DPOS_CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                JSR     RECORD_PASS
                RTS
T2_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 3: BK456_T3_CLAMP  (SEQ=2 DIM=4 VOC=10) — clamp trigger
* ============================================================
TEST_3:
                LDA     #3
                STA     CUR_TEST_ID
                LDD     SEQ_3
                STD     BK_SEQ
                LDD     DIM_3
                STD     BK_DIM
                LDD     VOC_3
                STD     BK_VOC
                CLRA
                CLRB
                STD     BK_SHF
                STD     BK_YY
                LDD     #WORK_3
                STD     BK_WRK
                LDD     #DY_3
                STD     BK_DY
                LDD     #X_3
                STD     BK_XX
                LDD     #WQ_3
                STD     BK_WQ
                LDD     #WK_3
                STD     BK_WK
                LDD     #WV_3
                STD     BK_WV
                LDD     #TOKS_3
                STD     BK_TOKS
                LDD     #DWQ_BUF
                STD     BK_DWQ
                LDD     #DWK_BUF
                STD     BK_DWK
                LDD     #DWV_BUF
                STD     BK_DWV
                LDD     #DTOK_BUF
                STD     BK_DTKE
                LDD     #DPOS_BUF
                STD     BK_DPSE
                LDX     #DSC_3
                LDY     #BW_DATT
                LDD     BK_SEQ
                MULD    BK_SEQ
                TFR     W,D
                TFR     D,W
                JSR     COPY_BUF
                LDX     #DV_3
                LDY     #BW_DV
                LDD     BK_SEQ
                MULD    BK_DIM
                TFR     W,D
                TFR     D,W
                JSR     COPY_BUF
                LDX     #DWQ_BUF
                LDW     DWQ_CNT_3
                JSR     ZERO_BUF
                LDX     #DWK_BUF
                LDW     DWK_CNT_3
                JSR     ZERO_BUF
                LDX     #DWV_BUF
                LDW     DWV_CNT_3
                JSR     ZERO_BUF
                LDX     #DTOK_BUF
                LDW     DTOK_CNT_3
                JSR     ZERO_BUF
                LDX     #DPOS_BUF
                LDW     DPOS_CNT_3
                JSR     ZERO_BUF
                JSR     BKWRD_INIT
                JSR     BKWRD_S4
                JSR     BKWRD_S5
                JSR     BKWRD_S6
                LDX     #BW_DQ
                LDY     #EXP_DQ_3
                LDW     DQ_CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                LDX     #BW_DK
                LDY     #EXP_DK_3
                LDW     DK_CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                LDX     #BW_DX
                LDY     #EXP_DX_3
                LDW     DX_CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                LDX     #DWQ_BUF
                LDY     #EXP_DWQ_3
                LDW     DWQ_CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                LDX     #DWK_BUF
                LDY     #EXP_DWK_3
                LDW     DWK_CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                LDX     #DWV_BUF
                LDY     #EXP_DWV_3
                LDW     DWV_CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                LDX     #DTOK_BUF
                LDY     #EXP_DTOK_3
                LDW     DTOK_CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                LDX     #DPOS_BUF
                LDY     #EXP_DPOS_3
                LDW     DPOS_CNT_3
                JSR     VERIFY_BUF
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

LBL_PASS:       FCC     "BK456: P="
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
                INCLUDE tables/bkwrd456_vectors.asm

                END     START
