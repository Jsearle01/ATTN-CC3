* ============================================================
* screen.asm — 32x16 text output at $0400
* ============================================================
* Dependencies: equates.inc
* Exports: SCR_INIT, SCR_CLEAR, SCR_PUTC, SCR_PUTS, SCR_HEX16,
*          SCR_DEC16, SCR_NEWLINE
* ============================================================
*
* TODO: implement in subsequent turn
*
* Will implement:
*   SCR_INIT    — configure GIME for 32x16 text at SCREEN ($0400)
*   SCR_CLEAR   — zero-fill 512B screen buffer
*   SCR_PUTC    — write char in A to current cursor, advance
*   SCR_PUTS    — write NUL-terminated string at X
*   SCR_HEX16   — write D as 4-digit hex
*   SCR_DEC16   — write D as unsigned decimal (for loss/acc display)
*   SCR_NEWLINE — advance cursor to next row
*
* All writes go directly to hardware-mapped memory at $0400.
* No ROM calls, no BASIC, no OS.
*

                INCLUDE equates.inc

* Stub — flat binary, no SECTION needed

* Stub — implementation in subsequent turn


