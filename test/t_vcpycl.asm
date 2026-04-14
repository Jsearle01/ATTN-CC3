* ============================================================
* t_vcpycl.asm — VCPY / VCLR test harness
* ============================================================
* Unit tests for VCPY and VCLR. 8 tests total, per-test
* reporting via a result bitmap at $1704 (bit N = test N pass).
* Dependencies: equates.inc, macros.inc, fxmath.asm, vecop.asm
* No external vector file; all test data inline.
* ============================================================

                INCLUDE include/equates.inc
                INCLUDE include/macros.inc
                INCLUDE include/fxmath.inc

                ORG     $0600

* -------- Entry point --------
START:
                ORCC    #$50            ; mask IRQ and FIRQ
                LDMD    #1              ; native mode (idempotent, harness pattern)
                LDS     #STACK_TOP
                JSR     CLEAR_SCREEN
                JSR     RUN_TESTS
                JSR     DISPLAY_RESULTS
HALT:           BRA     HALT

* -------- Scratch RAM --------
PASS_CNT        EQU     $1700           ; 2B total pass count
FAIL_CNT        EQU     $1702           ; 2B total fail count
RESULT_BITS     EQU     $1704           ; 1B: bit N set if test N passed
LAST_FAIL_ID    EQU     $1705           ; 1B: ID of last-failed test ($FF if none)
CUR_TEST_ID     EQU     $1706           ; 1B: currently-executing test ID
DEC_BUF         EQU     $1730           ; 5B: decimal conversion scratch
DEC_DTEMP       EQU     $1736           ; 2B
DEC_SCRPTR      EQU     $1738           ; 2B

* -------- Test data buffers --------
SRC_BUF         EQU     $1760           ; 40B source buffer
DST_BUF         EQU     $1790           ; 40B destination buffer

* -------- Sentinel constant --------
SENTINEL        EQU     $DEAD

* ============================================================
* RUN_TESTS — run all 8 tests, record per-test pass/fail
* ============================================================
RUN_TESTS:
                CLRA
                CLRB
                STD     PASS_CNT
                STD     FAIL_CNT
                CLR     RESULT_BITS
                LDA     #$FF
                STA     LAST_FAIL_ID

                JSR     TEST_0          ; VCPY_LEN0
                JSR     TEST_1          ; VCPY_LEN1
                JSR     TEST_2          ; VCPY_LEN16
                JSR     TEST_3          ; VCPY_DISJOINT
                JSR     TEST_4          ; VCLR_LEN0
                JSR     TEST_5          ; VCLR_LEN1
                JSR     TEST_6          ; VCLR_LEN16
                JSR     TEST_7          ; VCLR_PATTERN
                RTS

* ============================================================
* Test framework helpers
* ============================================================

* FILL_BUF — fill memory with a 16-bit value
* Entry: X -> buffer, D = value, W = word count
* Exit: buffer filled with W copies of D; W = 0
* Clobbers: X, W, CC
* Zero-guard: W=0 is a no-op (prevents DECW wrap to 65536).
FILL_BUF:
                CMPW    #0
                BEQ     FB_EXIT
FB_LOOP:
                STD     ,X++
                DECW
                BNE     FB_LOOP
FB_EXIT:
                RTS

* RECORD_PASS — increment PASS_CNT, set bit for CUR_TEST_ID
RECORD_PASS:
                LDD     PASS_CNT
                ADDD    #1
                STD     PASS_CNT
* Set bit corresponding to CUR_TEST_ID
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

* RECORD_FAIL — increment FAIL_CNT, store CUR_TEST_ID to LAST_FAIL_ID
RECORD_FAIL:
                LDD     FAIL_CNT
                ADDD    #1
                STD     FAIL_CNT
                LDA     CUR_TEST_ID
                STA     LAST_FAIL_ID
                RTS

* ============================================================
* Test 0: VCPY_LEN0 — zero-length no-op
* Pre-fill dest with $DEAD, call VCPY(len=0), verify unchanged.
* ============================================================
TEST_0:
                CLR     CUR_TEST_ID     ; ID = 0
* Fill DST_BUF[0..3] with sentinel
                LDX     #DST_BUF
                LDD     #SENTINEL
                LDW     #4
                JSR     FILL_BUF
* Call VCPY with V_LEN=0
                LDD     #SRC_BUF        ; V_SRC arbitrary (not read)
                STD     V_SRC
                LDD     #DST_BUF
                STD     V_DST
                CLRA
                CLRB
                STD     V_LEN           ; V_LEN = 0
                JSR     VCPY
* Verify DST_BUF[0..3] all still $DEAD
                LDX     #DST_BUF
                LDW     #4
T0_CHK:
                LDD     ,X++
                CMPD    #SENTINEL
                LBNE    T0_FAIL
                DECW
                BNE     T0_CHK
                JSR     RECORD_PASS
                RTS
T0_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 1: VCPY_LEN1 — single-word copy with bounds
* Source: [$1234, $5678]. Dest: [$DEAD, $DEAD, $DEAD, $DEAD].
* Call VCPY(len=1). Verify dest[0]=$1234, dest[1]=$DEAD.
* ============================================================
TEST_1:
                LDA     #1
                STA     CUR_TEST_ID
* Fill SRC_BUF[0] = $1234, SRC_BUF[1] = $5678
                LDX     #SRC_BUF
                LDD     #$1234
                STD     ,X++
                LDD     #$5678
                STD     ,X
* Fill DST_BUF[0..3] with sentinel
                LDX     #DST_BUF
                LDD     #SENTINEL
                LDW     #4
                JSR     FILL_BUF
* Call VCPY(len=1)
                LDD     #SRC_BUF
                STD     V_SRC
                LDD     #DST_BUF
                STD     V_DST
                LDD     #1
                STD     V_LEN
                JSR     VCPY
* Verify dest[0] = $1234
                LDX     #DST_BUF
                LDD     ,X
                CMPD    #$1234
                LBNE    T1_FAIL
* Verify dest[1] = $DEAD (bounds check)
                LDD     2,X
                CMPD    #SENTINEL
                LBNE    T1_FAIL
                JSR     RECORD_PASS
                RTS
T1_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 2: VCPY_LEN16 — full-length copy + bounds check
* Source: 16 unique words (pattern $0100, $0200, ..., $1000).
* Dest: 17 words of $DEAD.
* Call VCPY(len=16). Verify dest[0..15] match pattern and
* dest[16] still = $DEAD.
* ============================================================
TEST_2:
                LDA     #2
                STA     CUR_TEST_ID
* Fill SRC_BUF with pattern: SRC_BUF[i] = (i+1)*$0100
                LDX     #SRC_BUF
                LDD     #$0100          ; first value
                LDW     #16
T2_FILL:
                STD     ,X++
                ADDD    #$0100
                DECW
                BNE     T2_FILL
* Fill DST_BUF[0..16] with sentinel (17 words)
                LDX     #DST_BUF
                LDD     #SENTINEL
                LDW     #17
                JSR     FILL_BUF
* Call VCPY(len=16)
                LDD     #SRC_BUF
                STD     V_SRC
                LDD     #DST_BUF
                STD     V_DST
                LDD     #16
                STD     V_LEN
                JSR     VCPY
* Verify dest[0..15] match pattern
                LDX     #DST_BUF
                LDD     #$0100
                LDW     #16
T2_CHK:
                CMPD    ,X++
                LBNE    T2_FAIL
                ADDD    #$0100
                DECW
                BNE     T2_CHK
* Verify dest[16] = $DEAD (bounds)
                LDD     ,X
                CMPD    #SENTINEL
                LBNE    T2_FAIL
                JSR     RECORD_PASS
                RTS
T2_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 3: VCPY_DISJOINT — verify source unchanged
* Source: 8 known values at SRC_BUF.
* Dest: 8 words at DST_BUF (disjoint, far apart).
* Call VCPY(len=8). Verify dest matches AND source unchanged.
* ============================================================
TEST_3:
                LDA     #3
                STA     CUR_TEST_ID
* Fill SRC_BUF with pattern: $A000, $A001, ..., $A007
                LDX     #SRC_BUF
                LDD     #$A000
                LDW     #8
T3_FILL:
                STD     ,X++
                ADDD    #1
                DECW
                BNE     T3_FILL
* Fill DST_BUF with sentinel
                LDX     #DST_BUF
                LDD     #SENTINEL
                LDW     #8
                JSR     FILL_BUF
* Call VCPY(len=8)
                LDD     #SRC_BUF
                STD     V_SRC
                LDD     #DST_BUF
                STD     V_DST
                LDD     #8
                STD     V_LEN
                JSR     VCPY
* Verify dest[0..7] = $A000..$A007
                LDX     #DST_BUF
                LDD     #$A000
                LDW     #8
T3_CHK_DST:
                CMPD    ,X++
                LBNE    T3_FAIL
                ADDD    #1
                DECW
                BNE     T3_CHK_DST
* Verify source unchanged: SRC_BUF[0..7] = $A000..$A007
                LDX     #SRC_BUF
                LDD     #$A000
                LDW     #8
T3_CHK_SRC:
                CMPD    ,X++
                LBNE    T3_FAIL
                ADDD    #1
                DECW
                BNE     T3_CHK_SRC
                JSR     RECORD_PASS
                RTS
T3_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 4: VCLR_LEN0 — zero-length no-op
* Pre-fill 4 words with $DEAD, call VCLR(len=0), verify unchanged.
* ============================================================
TEST_4:
                LDA     #4
                STA     CUR_TEST_ID
                LDX     #DST_BUF
                LDD     #SENTINEL
                LDW     #4
                JSR     FILL_BUF
                LDD     #DST_BUF
                STD     V_DST
                CLRA
                CLRB
                STD     V_LEN
                JSR     VCLR
* Verify all 4 words still $DEAD
                LDX     #DST_BUF
                LDW     #4
T4_CHK:
                LDD     ,X++
                CMPD    #SENTINEL
                LBNE    T4_FAIL
                DECW
                BNE     T4_CHK
                JSR     RECORD_PASS
                RTS
T4_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 5: VCLR_LEN1 — single-word clear with bounds
* Pre-fill 2 words with $DEAD. Call VCLR(len=1).
* Verify dest[0]=$0000, dest[1]=$DEAD.
* ============================================================
TEST_5:
                LDA     #5
                STA     CUR_TEST_ID
                LDX     #DST_BUF
                LDD     #SENTINEL
                LDW     #2
                JSR     FILL_BUF
                LDD     #DST_BUF
                STD     V_DST
                LDD     #1
                STD     V_LEN
                JSR     VCLR
* Verify dest[0] = $0000
                LDX     #DST_BUF
                LDD     ,X
                CMPD    #0
                LBNE    T5_FAIL
* Verify dest[1] = $DEAD (bounds)
                LDD     2,X
                CMPD    #SENTINEL
                LBNE    T5_FAIL
                JSR     RECORD_PASS
                RTS
T5_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 6: VCLR_LEN16 — full-length clear with bounds
* Pre-fill 17 words with $DEAD. Call VCLR(len=16).
* Verify dest[0..15]=$0000, dest[16]=$DEAD.
* ============================================================
TEST_6:
                LDA     #6
                STA     CUR_TEST_ID
                LDX     #DST_BUF
                LDD     #SENTINEL
                LDW     #17
                JSR     FILL_BUF
                LDD     #DST_BUF
                STD     V_DST
                LDD     #16
                STD     V_LEN
                JSR     VCLR
* Verify dest[0..15] all zero
                LDX     #DST_BUF
                LDW     #16
T6_CHK:
                LDD     ,X++
                CMPD    #0
                LBNE    T6_FAIL
                DECW
                BNE     T6_CHK
* Verify dest[16] = $DEAD
                LDD     ,X
                CMPD    #SENTINEL
                LBNE    T6_FAIL
                JSR     RECORD_PASS
                RTS
T6_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Test 7: VCLR_PATTERN — clear buffer with varied initial values
* Fill 8 words with varied pattern (AAAA/5555/FFFF/8000/...).
* Call VCLR(len=8). Verify all 8 words are $0000.
* ============================================================
TEST_7:
                LDA     #7
                STA     CUR_TEST_ID
* Fill DST_BUF[0..7] with varied pattern
                LDX     #DST_BUF
                LDD     #$AAAA
                STD     ,X++
                LDD     #$5555
                STD     ,X++
                LDD     #$FFFF
                STD     ,X++
                LDD     #$8000
                STD     ,X++
                LDD     #$7FFF
                STD     ,X++
                LDD     #$0001
                STD     ,X++
                LDD     #$C3C3
                STD     ,X++
                LDD     #$3C3C
                STD     ,X
* Call VCLR(len=8)
                LDD     #DST_BUF
                STD     V_DST
                LDD     #8
                STD     V_LEN
                JSR     VCLR
* Verify all 8 words are zero
                LDX     #DST_BUF
                LDW     #8
T7_CHK:
                LDD     ,X++
                CMPD    #0
                LBNE    T7_FAIL
                DECW
                BNE     T7_CHK
                JSR     RECORD_PASS
                RTS
T7_FAIL:
                JSR     RECORD_FAIL
                RTS

* ============================================================
* Display / screen / decimal helpers
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

LBL_PASS:       FCC     "CPY: P="
                FCB     0
LBL_MID:        FCC     " F="
                FCB     0
LBL_DONE:       FCC     "DONE"
                FCB     0

* ============================================================
* Include VECOP code (provides VCPY and VCLR)
* ============================================================
                INCLUDE src/fxmath.asm
                INCLUDE src/vecop.asm

                END     START
