* ============================================================
* data.asm — Token stream, target buffer, RNG state
* ============================================================
* Dependencies: equates.inc
* Exports: TOK_INPUT, TOK_TARGET, RNG_STATE, RNG_NEXT
* ============================================================
*
* TODO: implement in subsequent turn
*
* Will contain:
*   TOK_INPUT   — 8-byte input token buffer at TOK_BASE ($1900)
*   TOK_TARGET  — 8-byte target (reversed) at TOK_BASE+8
*   RNG_STATE   — 16-bit LFSR state for sequence generation
*   RNG_NEXT    — generate next pseudo-random digit (0-9)
*   GEN_SEQ     — fill TOK_INPUT with random digits,
*                 TOK_TARGET with reversed copy
*

                INCLUDE equates.inc

* Stub — flat binary, no SECTION needed

* Stub — implementation in subsequent turn


