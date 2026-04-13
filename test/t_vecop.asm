* ============================================================
* t_vecop.asm — VECOP test harness
* ============================================================
* Loads vectors_vecop.bin, runs VDOT/VSCL/VADD tests, counts
* PASS/FAIL per routine, displays on screen.
* Dependencies: equates.inc, macros.inc, fxmath.asm, vecop.asm
* Inputs: test/vectors_vecop.bin (from gen_vecop_vectors.py)
* ============================================================

                INCLUDE include/equates.inc
                INCLUDE include/macros.inc
                INCLUDE include/fxmath.inc

                ORG     $0600

* -------- Entry point --------
START:
                ORCC    #$50            ; mask IRQ and FIRQ
                LDS     #STACK_TOP
                JSR     CLEAR_SCREEN
                JSR     RUN_TESTS
                JSR     DISPLAY_RESULTS
HALT:           BRA     HALT

* -------- Scratch RAM --------
PASS_VDOT       EQU     $1700           ; 2 bytes
FAIL_VDOT       EQU     $1702           ; 2 bytes
PASS_VSCL       EQU     $1704           ; 2 bytes
FAIL_VSCL       EQU     $1706           ; 2 bytes
PASS_VADD       EQU     $1708           ; 2 bytes
FAIL_VADD       EQU     $170A           ; 2 bytes
LOOP_CNT        EQU     $170C           ; 2 bytes
ELEM_CNT        EQU     $170E           ; 2 bytes: element count per record
REC_PTR         EQU     $1714           ; 2 bytes: saved record pointer
CMP_CNT         EQU     $1716           ; 1 byte: compare loop counter
DEC_BUF         EQU     $1730           ; 5 bytes
DEC_DTEMP       EQU     $1736           ; 2 bytes
DEC_SCRPTR      EQU     $1738           ; 2 bytes
DIAG_VSCL_ALPHA EQU     $171A           ; 2 bytes
DIAG_VSCL_X0    EQU     $171C           ; 2 bytes: first input element
DIAG_VSCL_EXP0  EQU     $171E           ; 2 bytes: first expected element
DIAG_VSCL_ACT0  EQU     $1720           ; 2 bytes: first actual output
DIAG_VSCL_CAP   EQU     $1722           ; 1 byte: captured flag
VADD_DST        EQU     $1740           ; 64 bytes: VADD output buffer
VSCL_DST        EQU     $1780           ; 64 bytes: VSCL output buffer

* ============================================================
* RUN_TESTS
* ============================================================
RUN_TESTS:
                CLRA
                CLRB
                STD     PASS_VDOT
                STD     FAIL_VDOT
                STD     PASS_VSCL
                STD     FAIL_VSCL
                STD     PASS_VADD
                STD     FAIL_VADD

* Verify magic "VC"
                LDX     #VECS
                LDA     ,X
                CMPA    #$56            ; 'V'
                LBNE    MAGIC_FAIL
                LDA     1,X
                CMPA    #$43            ; 'C'
                LBNE    MAGIC_FAIL

* Read counts from header
* Header: VC 01 00 n_vdot[2] n_vscl[2] n_vadd[2] reserved[2]
                LEAX    12,X            ; skip 12-byte header, X -> first VDOT record

* ======== VDOT test loop ========
                LDY     #VECS
                LDD     4,Y             ; D = n_vdot
                STD     LOOP_CNT

VDOT_TEST:
                LDD     LOOP_CNT
                LBEQ    VDOT_DONE
* Record: count[2], x[count*2], y[count*2], expected[2]
                LDD     ,X              ; D = count
                STD     ELEM_CNT
                STX     REC_PTR         ; save record start
                LEAX    2,X             ; X -> x[] data
* X now points to x[0]. Y needs to point to y[0] = X + count*2
* Compute Y = X + count*2
                PSHS    X               ; save X (-> x[])
                LDD     ELEM_CNT
                LSLD                    ; D = count * 2
                LEAY    D,X             ; Y = X + count*2 -> y[]
* B = count for VDOT call (low byte of ELEM_CNT)
                LDB     ELEM_CNT+1      ; low byte of count
                JSR     VDOT            ; D = dot product result
* D has result. Compute expected location:
* expected is at X_saved + count*2 + count*2 = X_saved + count*4
* But VDOT advanced X and Y past data. Use REC_PTR.
                PULS    X               ; restore X (-> x[0])
                PSHS    D               ; save result
                LDD     ELEM_CNT
                LSLD
                LSLD                    ; D = count * 4
                LEAX    D,X             ; X -> expected
                PULS    D               ; restore result
                CMPD    ,X              ; compare with expected
                BEQ     VDOT_PASS
                LDD     FAIL_VDOT
                ADDD    #1
                STD     FAIL_VDOT
                BRA     VDOT_NEXT
VDOT_PASS:
                LDD     PASS_VDOT
                ADDD    #1
                STD     PASS_VDOT
VDOT_NEXT:
* Advance X past expected (2 bytes) to next record
                LEAX    2,X
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    VDOT_TEST
VDOT_DONE:

* ======== VSCL test loop ========
* X now points to first VSCL record
                LDY     #VECS
                LDD     6,Y             ; D = n_vscl
                STD     LOOP_CNT
                CLR     DIAG_VSCL_CAP   ; zero diagnostic flag

VSCL_TEST:
                LDD     LOOP_CNT
                LBEQ    VSCL_DONE
* Record: count[2], alpha[2], x[count*2], exp[count*2]
                LDD     ,X              ; D = count
                STD     ELEM_CNT
                LDD     2,X             ; D = alpha
                PSHS    D               ; save alpha on stack
                LEAX    4,X             ; X -> x[] data
                STX     REC_PTR         ; save start of x[]

* ---- DIAGNOSTIC: capture first record inputs ----
                LDA     DIAG_VSCL_CAP
                BNE     VSCL_SKIP_DCAP
                LDD     ,S              ; alpha (still on stack)
                STD     DIAG_VSCL_ALPHA
                LDD     ,X              ; x[0]
                STD     DIAG_VSCL_X0
VSCL_SKIP_DCAP:
* ---- end diagnostic input capture ----

* Set up VSCL call: X -> src, Y -> dst, D = alpha, U = count
                LDY     #VSCL_DST       ; destination buffer
                LDU     ELEM_CNT        ; U = count
                PULS    D               ; D = alpha
                PSHS    X               ; save x[] pointer
                JSR     VSCL
                PULS    X               ; restore x[] pointer

* ---- DIAGNOSTIC: capture first record outputs ----
                LDA     DIAG_VSCL_CAP
                BNE     VSCL_SKIP_DCAP2
                LDD     VSCL_DST        ; act[0] = first output element
                STD     DIAG_VSCL_ACT0
VSCL_SKIP_DCAP2:
* ---- end diagnostic output capture ----

* Compare VSCL_DST with expected data
* Expected is at: x[] + count*2
                LDD     ELEM_CNT
                LSLD                    ; D = count * 2
                LEAY    D,X             ; Y -> expected[]

* ---- DIAGNOSTIC: capture first expected ----
                LDA     DIAG_VSCL_CAP
                BNE     VSCL_SKIP_DCAP3
                LDD     ,Y              ; exp[0]
                STD     DIAG_VSCL_EXP0
                LDA     #1
                STA     DIAG_VSCL_CAP   ; mark captured
VSCL_SKIP_DCAP3:
* ---- end diagnostic expected capture ----

                LDX     #VSCL_DST       ; X -> our output
                LDB     ELEM_CNT+1      ; B = count
                PSHS    B               ; counter on stack (LDD clobbers B)
VSCL_CMP:
                LDD     ,X++
                CMPD    ,Y++
                BNE     VSCL_CMP_FAIL
                DEC     ,S
                BNE     VSCL_CMP
* All elements matched
                LEAS    1,S             ; drop counter
                LDD     PASS_VSCL
                ADDD    #1
                STD     PASS_VSCL
                BRA     VSCL_NEXT
VSCL_CMP_FAIL:
                LEAS    1,S             ; drop counter
                LDD     FAIL_VSCL
                ADDD    #1
                STD     FAIL_VSCL
VSCL_NEXT:
* Advance to next record using REC_PTR
                LDX     REC_PTR         ; X = start of x[]
                LDD     ELEM_CNT
                LSLD
                LSLD                    ; D = count * 4 (x[] + exp[])
                LEAX    D,X             ; X -> next record
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    VSCL_TEST
VSCL_DONE:

* ======== VADD test loop ========
                LDY     #VECS
                LDD     8,Y             ; D = n_vadd
                STD     LOOP_CNT

VADD_TEST:
                LDD     LOOP_CNT
                LBEQ    VADD_DONE
* Record: count[2], x[count*2], y[count*2], exp[count*2]
                LDD     ,X              ; D = count
                STD     ELEM_CNT
                LEAX    2,X             ; X -> x[]
                STX     REC_PTR         ; save x[] start
* Set up VADD: X -> x, Y -> y (x + count*2), U -> dst
                LDD     ELEM_CNT
                LSLD                    ; D = count*2
                LEAY    D,X             ; Y -> y[]
                LDU     #VADD_DST
                LDB     ELEM_CNT+1      ; B = count
                PSHS    X               ; save x[] pointer
                JSR     VADD
                PULS    X               ; restore x[]
* Compare VADD_DST with expected
* Expected at: x[] + count*2 + count*2 = x[] + count*4
                LDD     ELEM_CNT
                LSLD
                LSLD                    ; D = count*4
                LEAY    D,X             ; Y -> expected[]
                LDX     #VADD_DST
                LDB     ELEM_CNT+1      ; B = count
                PSHS    B               ; counter on stack (LDD clobbers B)
VADD_CMP:
                LDD     ,X++
                CMPD    ,Y++
                BNE     VADD_CMP_FAIL
                DEC     ,S
                BNE     VADD_CMP
* All matched
                LEAS    1,S             ; drop counter
                LDD     PASS_VADD
                ADDD    #1
                STD     PASS_VADD
                BRA     VADD_NEXT
VADD_CMP_FAIL:
                LEAS    1,S             ; drop counter
                LDD     FAIL_VADD
                ADDD    #1
                STD     FAIL_VADD
VADD_NEXT:
* Advance to next record
* VADD record: count[2] + x[count*2] + y[count*2] + exp[count*2] = 2 + count*6
                LDX     REC_PTR         ; X = start of x[]
                LDD     ELEM_CNT
                LSLD
                PSHS    D               ; count*2
                LSLD                    ; count*4
                ADDD    ,S++            ; D = count*4 + count*2 = count*6
                LEAX    D,X             ; X -> next record
                LDD     LOOP_CNT
                SUBD    #1
                STD     LOOP_CNT
                LBRA    VADD_TEST
VADD_DONE:
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
* DISPLAY_RESULTS — 4 rows
* Row 0: "DOT: P=nnnn F=nnnn"
* Row 1: "SCL: P=nnnn F=nnnn"
* Row 2: "ADD: P=nnnn F=nnnn"
* Row 3: "DONE"
* ============================================================
DISPLAY_RESULTS:
                LDX     #SCREEN
                LDU     #DOT_LABEL
                JSR     COPY_STRING
                LDD     PASS_VDOT
                JSR     WRITE_DECIMAL_4
                LDU     #MID_LABEL
                JSR     COPY_STRING
                LDD     FAIL_VDOT
                JSR     WRITE_DECIMAL_4

                LDX     #SCREEN+32
                LDU     #SCL_LABEL
                JSR     COPY_STRING
                LDD     PASS_VSCL
                JSR     WRITE_DECIMAL_4
                LDU     #MID_LABEL
                JSR     COPY_STRING
                LDD     FAIL_VSCL
                JSR     WRITE_DECIMAL_4

                LDX     #SCREEN+64
                LDU     #ADD_LABEL
                JSR     COPY_STRING
                LDD     PASS_VADD
                JSR     WRITE_DECIMAL_4
                LDU     #MID_LABEL
                JSR     COPY_STRING
                LDD     FAIL_VADD
                JSR     WRITE_DECIMAL_4

                LDX     #SCREEN+96
                LDU     #DONE_LABEL
                JSR     COPY_STRING
                RTS

* ============================================================
* COPY_STRING
* ============================================================
COPY_STRING:
                LDA     ,U+
                BEQ     CS_DONE
                STA     ,X+
                BRA     COPY_STRING
CS_DONE:
                RTS

* ============================================================
* WRITE_DECIMAL_4
* ============================================================
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

DIV_TABLE:
                FDB     1000
                FDB     100
                FDB     10

* ============================================================
* String constants
* ============================================================
DOT_LABEL:      FCC     "DOT: P="
                FCB     0
SCL_LABEL:      FCC     "SCL: P="
                FCB     0
ADD_LABEL:      FCC     "ADD: P="
                FCB     0
MID_LABEL:      FCC     " F="
                FCB     0
DONE_LABEL:     FCC     "DONE"
                FCB     0
BADHDR_MSG:     FCC     "BAD VECOP HEADER"
                FCB     0

* ============================================================
* FXMATH code (needed by VECOP for NORMQ15 macro expansion)
* ============================================================
                INCLUDE src/fxmath.asm

* ============================================================
* VECOP code
* ============================================================
                INCLUDE src/vecop.asm

* ============================================================
* Test vectors — loaded separately by Lua at $4800
* ============================================================
VECS            EQU     $4800

                END     START
