* ============================================================
* main.asm — Entry point, reset handler, system init
* ============================================================
* Dependencies: equates.inc, macros.inc
* Includes: all src/*.asm modules
* ============================================================
*
* TODO: implement in subsequent turn
*
* Will contain:
*   - ORG $0200
*   - HD6309 native mode switch (LDMD #$01)
*   - Stack init (LDS #STACK_TOP)
*   - GIME init for 32x16 text mode at $0400
*   - Weight initialization (small random Q16)
*   - Call TRAIN main loop
*   - Halt on completion
*   - Reset vector at $FFFE pointing to entry
*

                INCLUDE equates.inc
                INCLUDE macros.inc

* Stub — flat binary, no SECTION needed

* Entry point — stub
ENTRY
                RTS


