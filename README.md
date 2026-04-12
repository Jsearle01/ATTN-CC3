# ATTN/11 → HD6309 Port (Stage 1)

Port of ATTN/11 (github.com/dbrll/ATTN-11), a single-layer transformer that
trains to reverse sequences of digits, from PDP-11 MACRO-11 to HD6309
assembly targeting the TRS-80 Color Computer 3.

## Stage 1 Goals
- Single-node 6309 implementation on bare-metal CoCo3
- Match ATTN/11 numerics exactly (Q8/Q15/Q16 fixed-point)
- Converge to accuracy 1.0 in ~350 steps
- Runtime 6-8 minutes at 1.78MHz
- Total memory ~17KB

## Build Order
1. FXMATH  — scalar primitives
2. Test harness validating against reference vectors
3. VECOP   — dot product, scaling
4. MATOP   — matrix-vector multiply
5. ACTFN   — softmax, tanh, lookup tables
6. LAYER   — embedding, attention, projection
7. TRAIN   — SGD loop

Each stage must pass tests before the next is implemented.

## Golden Reference (from ATTN/11 PDP-11 run)
    STEP  50 LOSS=1.6113 ACC=0.217
    STEP 100 LOSS=2.1865 ACC=0.255
    STEP 150 LOSS=2.1511 ACC=0.267
    STEP 200 LOSS=1.3874 ACC=0.395
    STEP 250 LOSS=0.0500 ACC=0.662
    STEP 300 LOSS=0.0019 ACC=0.982
    STEP 350 LOSS=0.0009 ACC=1.000

These are the values our port must reproduce.

## Toolchain
- Assembler: LWASM
- Emulator: MAME coco3h (headless for CI)
- Reference generator: Python 3 (tables/ref_attn11.py)

## License

ATTN-CC3 is licensed under the GNU General Public License v3.0.
See [LICENSE](LICENSE) for the full text.

## Acknowledgments

This project is a port of [ATTN/11](https://github.com/dbrll/ATTN-11)
by Damien Boureille, originally written in PDP-11 assembly.
ATTN/11 is MIT-licensed; this port is GPL-3.0-licensed as
permitted by the MIT license's relicensing terms.
