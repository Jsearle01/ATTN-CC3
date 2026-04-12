#!/usr/bin/env python3
"""
gen_logtbl.py — Generate LOGTBL lookup table for cross-entropy loss.

TODO: implement in subsequent turn

Will emit: tables/logtbl.bin (514 bytes = 257 entries × 2 bytes)
Format: Q12 -ln(x/256) * 4096, big-endian 16-bit signed
  - Index 0 → -ln(0/256) = +inf, clamped to max Q12
  - Index 256 → -ln(1.0) = 0

Used by CROSS_ENTROPY_Q15 in actfn.asm.
Probability input is Q8 unsigned [0..256], table returns Q12 -log.
"""
