* ============================================================
* t_bkwrd1.asm — BKWRD Step 1 test harness
* ============================================================
* 4 tests for dLogits -> dWout, dY.
* Each test verifies both dWout and dY output buffers.
* Dependencies: equates.inc, macros.inc, fxmath.inc,
*   fxmath.asm, vecop.asm, matop.asm (OUTER, MVMUL),
*   actfn.asm (SFTMX+EXPTBL), train.asm (BKWRD+LOGTBL)
* Vectors: tables/bkwrd1_vectors.asm (gen_bkwrd1_vectors.py)
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

* -------- Test data buffers (max: SEQ=4 DIM=8 VOC=10) --------
* Inputs point directly into vector table (no copy needed).
* Only gradient outputs need RAM allocation + zeroing.
DWOUT_BUF       EQU     SCRATCH_BASE+$60        ; 160 B max ([8][10])
DY_BUF          EQU     SCRATCH_BASE+$100       ; 64 B max  ([4][8])

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

* ============================================================
* SETUP_BK1 — Parse descriptor, zero dWout, call BKWRD Step 1
* ============================================================
* Entry: X -> descriptor (SEQ, DIM, VOC, LOGITS, TARGETS, Y, WOUT,
*         DWOUT_CNT, DY_CNT, EXP_DWOUT, EXP_DY)
* Exit: X -> EXP_DWOUT, W = DWOUT_CNT (for first verify)
*       EXP_DY ptr and DY_CNT saved internally for second verify
* ============================================================
* Descriptor layout (sequential FDB words):
*   +0  SEQ
*   +2  DIM
*   +4  VOC
*   +6  LOGITS data (SEQ*VOC words)
*   +6+SEQ*VOC*2  TARGETS data (SEQ words)
*   +..  Y data (SEQ*DIM words)
*   +..  WOUT data (DIM*VOC words)
*   +..  DWOUT_CNT
*   +..  DY_CNT
*   +..  EXP_DWOUT data (DIM*VOC words)
*   +..  EXP_DY data (SEQ*DIM words)

SB1_DYCNT       EQU     SCRATCH_BASE+$50        ; 2B saved DY_CNT
SB1_EXPDY       EQU     SCRATCH_BASE+$52        ; 2B saved EXP_DY ptr

SETUP_BK1:
* Read dimensions
                LDD     ,X              ; SEQ
                STD     BK_SEQ
                LDD     2,X             ; DIM
                STD     BK_DIM
                LDD     4,X             ; VOC
                STD     BK_VOC
                LEAX    6,X             ; X -> LOGITS data

* Set BK_LOG (logits in binary image)
                STX     BK_LOG
* Advance past LOGITS (SEQ*VOC*2)
                LDD     BK_SEQ
                MULD    BK_VOC
                TFR     W,D
                LSLD
                LEAX    D,X             ; X -> TARGETS

* Set BK_TGT
                STX     BK_TGT
* Advance past TARGETS (SEQ*2)
                LDD     BK_SEQ
                LSLD
                LEAX    D,X             ; X -> Y data

* Set BK_YY
                STX     BK_YY
* Advance past Y (SEQ*DIM*2)
                LDD     BK_SEQ
                MULD    BK_DIM
                TFR     W,D
                LSLD
                LEAX    D,X             ; X -> WOUT data

* Set BK_WOUT
                STX     BK_WOUT
* Advance past WOUT (DIM*VOC*2)
                LDD     BK_DIM
                MULD    BK_VOC
                TFR     W,D
                LSLD
                LEAX    D,X             ; X -> DWOUT_CNT

* Read counts
                LDD     ,X++            ; DWOUT_CNT
                PSHS    D               ; save DWOUT_CNT
                LDD     ,X++            ; DY_CNT
                STD     SB1_DYCNT
* X -> EXP_DWOUT data

* Save EXP_DWOUT ptr, then advance to find EXP_DY
                PSHS    X               ; save EXP_DWOUT ptr
* Advance past EXP_DWOUT to find EXP_DY
                LDD     2,S             ; DWOUT_CNT
                LSLD                    ; byte count
                LEAX    D,X             ; X -> EXP_DY data
                STX     SB1_EXPDY
                PULS    X               ; restore EXP_DWOUT ptr
* Stack: [DWOUT_CNT(2)]

* Set output buffer pointers
                LDD     #DWOUT_BUF
                STD     BK_DWOUT
                LDD     #DY_BUF
                STD     BK_DY

* Zero DWOUT_BUF (DWOUT_CNT words)
                LDX     #DWOUT_BUF
                LDW     ,S              ; W = DWOUT_CNT
                CLRA
                CLRB
SB1_ZERO:
                STD     ,X++
                DECW
                BNE     SB1_ZERO

* Call BKWRD
                PSHS    X               ; save (X is garbage now, but need to preserve stack)
                JSR     BKWRD_INIT
                JSR     BKWRD_S1
                PULS    X

* Restore: X -> EXP_DWOUT from earlier save
* Wait — X was clobbered. The EXP_DWOUT ptr is at... let me rethink.
* After the zero loop, X is past DWOUT_BUF. Then BKWRD clobbers X.
* The EXP_DWOUT ptr was NOT saved after the PULS X above.
* I need to save it before calling BKWRD.

* Let me restructure: save EXP_DWOUT ptr to a scratch location.
* ACTUALLY: the PULS X above restored EXP_DWOUT ptr, then I did
* the zero loop which clobbered X. Need to save it BEFORE zeroing.

* This is getting tangled. Let me use scratch locations for both pointers.

* REDESIGN: save both EXPECT pointers to scratch, then zero, then BKWRD.
* Already have SB1_EXPDY. Add SB1_EXPDW.

* For now: I'll restructure the code below. The descriptor is parsed,
* pointers saved, then zeroing + BKWRD call.

* (I realize I need to restructure — let me redo this routine cleanly.)
                RTS

* ============================================================
* Tests use a simpler approach: inline descriptor parsing per test.
* Each test sets BK_* params, zeros DWOUT_BUF, calls BKWRD,
* then verifies both output buffers.
* ============================================================

* Helper: ZERO_BUF — zero W words at X
ZERO_BUF:
                CMPW    #0
                BEQ     ZB_EXIT
                CLRA
                CLRB
ZB_LOOP:
                STD     ,X++
                DECW
                BNE     ZB_LOOP
ZB_EXIT:
                RTS

* Helper: CMP_BUF — compare W words at X vs Y. Returns Z=1 if match.
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
* Test 0: BK1_TINY  (SEQ=2 DIM=2 VOC=3)
* ============================================================
TEST_0:
                CLR     CUR_TEST_ID
* Set params from vector table labels
                LDD     SEQ_0
                STD     BK_SEQ
                LDD     DIM_0
                STD     BK_DIM
                LDD     VOC_0
                STD     BK_VOC
                LDD     #LOGITS_0
                STD     BK_LOG
                LDD     #TARGETS_0
                STD     BK_TGT
                LDD     #Y_0
                STD     BK_YY
                LDD     #WOUT_0
                STD     BK_WOUT
                LDD     #DWOUT_BUF
                STD     BK_DWOUT
                LDD     #DY_BUF
                STD     BK_DY
* Zero DWOUT_BUF
                LDX     #DWOUT_BUF
                LDW     DWOUT_CNT_0
                JSR     ZERO_BUF
* Call BKWRD
                JSR     BKWRD_INIT
                JSR     BKWRD_S1
* Verify dWout
                LDX     #DWOUT_BUF
                LDY     #EXP_DWOUT_0
                LDW     DWOUT_CNT_0
                JSR     CMP_BUF
                LBNE    T0_FAIL
* Verify dY
                LDX     #DY_BUF
                LDY     #EXP_DY_0
                LDW     DY_CNT_0
                JSR     CMP_BUF
                LBNE    T0_FAIL
                JSR     RECORD_PASS
                RTS
T0_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 1: BK1_UNIFORM  (SEQ=2 DIM=4 VOC=10)
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
                LDD     #LOGITS_1
                STD     BK_LOG
                LDD     #TARGETS_1
                STD     BK_TGT
                LDD     #Y_1
                STD     BK_YY
                LDD     #WOUT_1
                STD     BK_WOUT
                LDD     #DWOUT_BUF
                STD     BK_DWOUT
                LDD     #DY_BUF
                STD     BK_DY
                LDX     #DWOUT_BUF
                LDW     DWOUT_CNT_1
                JSR     ZERO_BUF
                JSR     BKWRD_INIT
                JSR     BKWRD_S1
                LDX     #DWOUT_BUF
                LDY     #EXP_DWOUT_1
                LDW     DWOUT_CNT_1
                JSR     CMP_BUF
                LBNE    T1_FAIL
                LDX     #DY_BUF
                LDY     #EXP_DY_1
                LDW     DY_CNT_1
                JSR     CMP_BUF
                LBNE    T1_FAIL
                JSR     RECORD_PASS
                RTS
T1_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 2: BK1_MEDIUM  (SEQ=4 DIM=8 VOC=10)
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
                LDD     #LOGITS_2
                STD     BK_LOG
                LDD     #TARGETS_2
                STD     BK_TGT
                LDD     #Y_2
                STD     BK_YY
                LDD     #WOUT_2
                STD     BK_WOUT
                LDD     #DWOUT_BUF
                STD     BK_DWOUT
                LDD     #DY_BUF
                STD     BK_DY
                LDX     #DWOUT_BUF
                LDW     DWOUT_CNT_2
                JSR     ZERO_BUF
                JSR     BKWRD_INIT
                JSR     BKWRD_S1
                LDX     #DWOUT_BUF
                LDY     #EXP_DWOUT_2
                LDW     DWOUT_CNT_2
                JSR     CMP_BUF
                LBNE    T2_FAIL
                LDX     #DY_BUF
                LDY     #EXP_DY_2
                LDW     DY_CNT_2
                JSR     CMP_BUF
                LBNE    T2_FAIL
                JSR     RECORD_PASS
                RTS
T2_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 3: BK1_SAT  (SEQ=2 DIM=4 VOC=10)
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
                LDD     #LOGITS_3
                STD     BK_LOG
                LDD     #TARGETS_3
                STD     BK_TGT
                LDD     #Y_3
                STD     BK_YY
                LDD     #WOUT_3
                STD     BK_WOUT
                LDD     #DWOUT_BUF
                STD     BK_DWOUT
                LDD     #DY_BUF
                STD     BK_DY
                LDX     #DWOUT_BUF
                LDW     DWOUT_CNT_3
                JSR     ZERO_BUF
                JSR     BKWRD_INIT
                JSR     BKWRD_S1
                LDX     #DWOUT_BUF
                LDY     #EXP_DWOUT_3
                LDW     DWOUT_CNT_3
                JSR     CMP_BUF
                LBNE    T3_FAIL
                LDX     #DY_BUF
                LDY     #EXP_DY_3
                LDW     DY_CNT_3
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

LBL_PASS:       FCC     "BK1: P="
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
                INCLUDE tables/bkwrd1_vectors.asm

                END     START
