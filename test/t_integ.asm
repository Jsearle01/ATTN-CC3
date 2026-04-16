* ============================================================
* t_integ.asm — Integration test: EMBED → ATTN → PROJ
* ============================================================
* Full forward pass of tiny transformer (SEQ=2, D=4, V=10, SHF=1):
*   tokens [3, 7] → EMBED → X → ATTN → Y → PROJ → logits [2][10]
* Byte-exact comparison of 20 logit words against reference.
* Dependencies: equates.inc, macros.inc, fxmath.inc, fxmath.asm,
*   vecop.asm, matop.asm, actfn.asm (EXPTBL), layer.asm (EMBED,
*   ATTN, AT_BPR, PROJ)
* Vectors: tables/integration_vectors.asm (gen_integration_vectors.py)
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
                JSR     RUN_TEST
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

* -------- Intermediate buffers --------
* Weights live in the binary image (vector-table labels).
* Only intermediates need RAM allocation.
* X_BUF:   SEQ*DIM*2 = 2*4*2 = 16 B   EMBED out → ATTN in
* WRK_BUF: 3*16 + 8 = 56 B            ATTN workspace (Q+K+V+S/A)
* Y_BUF:   SEQ*DIM*2 = 16 B            ATTN out → PROJ in
* LOG_BUF: SEQ*VOC*2 = 2*10*2 = 40 B   PROJ out (final)
X_BUF           EQU     SCRATCH_BASE+$60
WRK_BUF         EQU     SCRATCH_BASE+$70
Y_BUF           EQU     SCRATCH_BASE+$A8
LOG_BUF         EQU     SCRATCH_BASE+$B8

* ============================================================
* RUN_TEST — single integration test
* ============================================================
RUN_TEST:
                CLRA
                CLRB
                STD     PASS_CNT
                STD     FAIL_CNT
                CLR     RESULT_BITS
                LDA     #$FF
                STA     LAST_FAIL_ID
                CLR     CUR_TEST_ID

* ---- Step 1: EMBED ----
                LDD     #TOKENS_0
                STD     EM_TOK
                LDD     #TKEMB_0
                STD     EM_TKE
                LDD     #PSEMB_0
                STD     EM_PSE
                LDD     #X_BUF
                STD     EM_OUT
                LDD     SEQ_0
                STD     EM_SEQ
                LDD     DIM_0
                STD     EM_DIM
                JSR     EMBED

* ---- Step 2: ATTN ----
                LDD     #X_BUF
                STD     AT_XIN
                LDD     #WQ_0
                STD     AT_WQ
                LDD     #WK_0
                STD     AT_WK
                LDD     #WV_0
                STD     AT_WV
                LDD     #Y_BUF
                STD     AT_YOT
                LDD     #WRK_BUF
                STD     AT_WRK
                LDD     SEQ_0
                STD     AT_SEQ
                LDD     DIM_0
                STD     AT_DIM
                LDD     SHF_0
                STD     AT_SHF
                JSR     ATTN

* ---- Step 3: PROJ ----
                LDD     #Y_BUF
                STD     PR_Y
                LDD     #WOUT_0
                STD     PR_WOUT
                LDD     #LOG_BUF
                STD     PR_LOG
                LDD     SEQ_0
                STD     PR_SEQ
                LDD     DIM_0
                STD     PR_DIM
                LDD     VOC_0
                STD     PR_VOC
                JSR     PROJ

* ---- Verify: compare LOG_BUF against EXPECT_0 (20 words) ----
                LDX     #LOG_BUF
                LDY     #EXPECT_0
                LDW     #20
INTEG_CHK:
                LDD     ,X++
                CMPD    ,Y++
                LBNE    INTEG_FAIL
                DECW
                BNE     INTEG_CHK
                JSR     RECORD_PASS
                RTS

INTEG_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Framework helpers
* ============================================================
RECORD_PASS:
                LDD     PASS_CNT
                ADDD    #1
                STD     PASS_CNT
                LDB     #1
                ORB     RESULT_BITS
                STB     RESULT_BITS
                RTS

RECORD_FAIL:
                LDD     FAIL_CNT
                ADDD    #1
                STD     FAIL_CNT
                CLR     CUR_TEST_ID
                STA     LAST_FAIL_ID
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

LBL_PASS:       FCC     "INT: P="
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
                INCLUDE src/layer.asm

* ============================================================
* Test vectors
* ============================================================
                INCLUDE tables/integration_vectors.asm

                END     START
