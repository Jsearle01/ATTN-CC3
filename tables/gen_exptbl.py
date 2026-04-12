#!/usr/bin/env python3
"""
gen_exptbl.py — Generate EXPTBL lookup table for softmax.

TODO: implement in subsequent turn

Will emit: tables/exptbl.bin (256 bytes)
Format: Q8 exp(-i/32) for i = 0..255
  - Index 0 → exp(0) = 1.0 → 0xFF (Q8 unsigned) or 0x7F (Q8 signed)
  - Index 255 → exp(-7.97) ≈ 0.00035 → 0x00

Used by SOFTMAX_Q8 in actfn.asm for numerical-stability softmax.
The max value is subtracted first, so all indices are non-negative.
"""
