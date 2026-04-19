* ============================================================
* t_updat.asm — UPDAT routines test harness
* ============================================================
* 5 tests for training-phase utilities:
*   Test 0: WUPDT_ONE basic (lr_shift=1, no shift)
*   Test 1: WUPDT_ONE shifted (lr_shift=4, grad>>3, sign-ext path)
*   Test 2: CVT16_ONE (Q16 hi:lo -> Q8 middle-byte extract)
*   Test 3: INITW_ONE from seed=887, verify hi/lo/Q8 pipeline
*   Test 4: WUPDT_ONE carry + sign-extension edge cases
*
* Dependencies: equates.inc, macros.inc, fxmath.inc, fxmath.asm,
*   vecop.asm (VCLR for ZEROG), matop.asm, actfn.asm, layer.asm
*   (transitively via train.asm's CLOSS/BKWRD references), train.asm.
*
* Full INCLUDE chain is necessary because train.asm contains CLOSS
* (SFTMX, EXPTBL) and BKWRD (VTMUL, MVMUL, MVADD, OUTER, VDOT, VSADD).
* Accepting the binary bloat to keep train.asm monolithic.
*
* Vectors: tables/updat_vectors.asm (gen_updat_vectors.py).
* SCRATCH_BASE at $2400 to match the expanded harness convention.
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

* -------- Scratch RAM at $2400 --------
SCRATCH_BASE    EQU     $2400
PASS_CNT        EQU     SCRATCH_BASE+0
FAIL_CNT        EQU     SCRATCH_BASE+2
RESULT_BITS     EQU     SCRATCH_BASE+4
LAST_FAIL_ID    EQU     SCRATCH_BASE+5
CUR_TEST_ID     EQU     SCRATCH_BASE+6
DEC_BUF         EQU     SCRATCH_BASE+$30
DEC_DTEMP       EQU     SCRATCH_BASE+$36
DEC_SCRPTR      EQU     SCRATCH_BASE+$38

* -------- Working buffers (max 8 words per test for WUPDT/INITW) --------
* Max allocation: 8 words = 16 bytes each. Generously 64 bytes each.
WHI_BUF         EQU     SCRATCH_BASE+$60        ; working w_hi (64 B)
WLO_BUF         EQU     SCRATCH_BASE+$A0        ; working w_lo (64 B)
GRAD_BUF        EQU     SCRATCH_BASE+$E0        ; working gradients (64 B)
Q8_BUF          EQU     SCRATCH_BASE+$120       ; Q8 output (64 B)

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

* COPY_BUF — copy W words from X to Y
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

* VERIFY_BUF — compare W words at X vs Y. Z=1 match, Z=0 mismatch.
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
                ORCC    #$04
                RTS
VB_FAIL:
                ANDCC   #$FB
                RTS

* ============================================================
* Test 0: UP_T0_WUPDT_BASIC — lr_shift=1, no shift
* ============================================================
TEST_0:
                CLR     CUR_TEST_ID
* Copy w_hi/w_lo/grad into working buffers (WUPDT mutates them)
                LDX     #WHI_0
                LDY     #WHI_BUF
                LDW     CNT_0
                JSR     COPY_BUF
                LDX     #WLO_0
                LDY     #WLO_BUF
                LDW     CNT_0
                JSR     COPY_BUF
                LDX     #GRAD_0
                LDY     #GRAD_BUF
                LDW     CNT_0
                JSR     COPY_BUF
* Set UP_* parameters
                LDD     #WHI_BUF
                STD     UP_WHI
                LDD     #WLO_BUF
                STD     UP_WLO
                LDD     #GRAD_BUF
                STD     UP_PTR
                LDD     CNT_0
                STD     UP_CNT
                LDD     SHF_0
                STD     UP_SHF
                JSR     WUPDT_ONE
* Verify w_hi, w_lo, grad (zeroed)
                LDX     #WHI_BUF
                LDY     #EXP_WHI_0
                LDW     CNT_0
                JSR     VERIFY_BUF
                LBNE    T0_FAIL
                LDX     #WLO_BUF
                LDY     #EXP_WLO_0
                LDW     CNT_0
                JSR     VERIFY_BUF
                LBNE    T0_FAIL
                LDX     #GRAD_BUF
                LDY     #EXP_GRAD_0
                LDW     CNT_0
                JSR     VERIFY_BUF
                LBNE    T0_FAIL
                JSR     RECORD_PASS
                RTS
T0_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 1: UP_T1_WUPDT_SHIFTED — lr_shift=4
* ============================================================
TEST_1:
                LDA     #1
                STA     CUR_TEST_ID
                LDX     #WHI_1
                LDY     #WHI_BUF
                LDW     CNT_1
                JSR     COPY_BUF
                LDX     #WLO_1
                LDY     #WLO_BUF
                LDW     CNT_1
                JSR     COPY_BUF
                LDX     #GRAD_1
                LDY     #GRAD_BUF
                LDW     CNT_1
                JSR     COPY_BUF
                LDD     #WHI_BUF
                STD     UP_WHI
                LDD     #WLO_BUF
                STD     UP_WLO
                LDD     #GRAD_BUF
                STD     UP_PTR
                LDD     CNT_1
                STD     UP_CNT
                LDD     SHF_1
                STD     UP_SHF
                JSR     WUPDT_ONE
                LDX     #WHI_BUF
                LDY     #EXP_WHI_1
                LDW     CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #WLO_BUF
                LDY     #EXP_WLO_1
                LDW     CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                LDX     #GRAD_BUF
                LDY     #EXP_GRAD_1
                LDW     CNT_1
                JSR     VERIFY_BUF
                LBNE    T1_FAIL
                JSR     RECORD_PASS
                RTS
T1_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 2: UP_T2_CVT16_BASIC — Q16 -> Q8 middle-byte extract
* ============================================================
TEST_2:
                LDA     #2
                STA     CUR_TEST_ID
* CVT16 doesn't mutate hi/lo so we can point at the vector data directly
                LDD     #WHI_2
                STD     UP_WHI
                LDD     #WLO_2
                STD     UP_WLO
                LDD     #Q8_BUF
                STD     UP_PTR          ; Q8 destination
                LDD     CNT_2
                STD     UP_CNT
                JSR     CVT16_ONE
                LDX     #Q8_BUF
                LDY     #EXP_Q8_2
                LDW     CNT_2
                JSR     VERIFY_BUF
                LBNE    T2_FAIL
                JSR     RECORD_PASS
                RTS
T2_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 3: UP_T3_INITW_RAND — seed=887, generate 8 weights
* ============================================================
TEST_3:
                LDA     #3
                STA     CUR_TEST_ID
* Set RN_SED to the test seed
                LDD     SEED_3
                STD     RN_SED
* Set UP_* params for INITW_ONE (writes to WHI_BUF and WLO_BUF)
                LDD     #WHI_BUF
                STD     UP_WHI
                LDD     #WLO_BUF
                STD     UP_WLO
                LDD     CNT_3
                STD     UP_CNT
                JSR     INITW_ONE
* Verify w_hi and w_lo
                LDX     #WHI_BUF
                LDY     #EXP_WHI_3
                LDW     CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                LDX     #WLO_BUF
                LDY     #EXP_WLO_3
                LDW     CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
* Also run CVT16_ONE on the initialized weights and verify Q8
                LDD     #WHI_BUF
                STD     UP_WHI
                LDD     #WLO_BUF
                STD     UP_WLO
                LDD     #Q8_BUF
                STD     UP_PTR
                LDD     CNT_3
                STD     UP_CNT
                JSR     CVT16_ONE
                LDX     #Q8_BUF
                LDY     #EXP_Q8_3
                LDW     CNT_3
                JSR     VERIFY_BUF
                LBNE    T3_FAIL
                JSR     RECORD_PASS
                RTS
T3_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 4: UP_T4_WUPDT_CARRY — carry + sign-extension edges
* ============================================================
TEST_4:
                LDA     #4
                STA     CUR_TEST_ID
                LDX     #WHI_4
                LDY     #WHI_BUF
                LDW     CNT_4
                JSR     COPY_BUF
                LDX     #WLO_4
                LDY     #WLO_BUF
                LDW     CNT_4
                JSR     COPY_BUF
                LDX     #GRAD_4
                LDY     #GRAD_BUF
                LDW     CNT_4
                JSR     COPY_BUF
                LDD     #WHI_BUF
                STD     UP_WHI
                LDD     #WLO_BUF
                STD     UP_WLO
                LDD     #GRAD_BUF
                STD     UP_PTR
                LDD     CNT_4
                STD     UP_CNT
                LDD     SHF_4
                STD     UP_SHF
                JSR     WUPDT_ONE
                LDX     #WHI_BUF
                LDY     #EXP_WHI_4
                LDW     CNT_4
                JSR     VERIFY_BUF
                LBNE    T4_FAIL
                LDX     #WLO_BUF
                LDY     #EXP_WLO_4
                LDW     CNT_4
                JSR     VERIFY_BUF
                LBNE    T4_FAIL
                LDX     #GRAD_BUF
                LDY     #EXP_GRAD_4
                LDW     CNT_4
                JSR     VERIFY_BUF
                LBNE    T4_FAIL
                JSR     RECORD_PASS
                RTS
T4_FAIL:
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

LBL_PASS:       FCC     "UPDAT: P="
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
* Include code (full chain for train.asm)
* ============================================================
                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm
                INCLUDE src/matop.asm
                INCLUDE src/actfn.asm
                INCLUDE src/layer.asm
                INCLUDE src/train.asm

* ============================================================
* Test vectors
* ============================================================
                INCLUDE tables/updat_vectors.asm

                END     START
